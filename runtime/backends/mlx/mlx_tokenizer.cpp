#include "runtime/backends/mlx/mlx_tokenizer.h"
#include "runtime/string_utils.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace inferflux {

// ---------------------------------------------------------------------------
// Byte-to-unicode table (GPT-2 / LLaMA-3 ByteLevel encoding).
//
// Bytes in {33..126, 161..172, 174..255} map to their own Unicode code point.
// The remaining 68 bytes map to U+0100..U+0143 in order of byte value.
// ---------------------------------------------------------------------------

namespace {

// Encode a Unicode code point to UTF-8 string.
std::string CpToUtf8(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

// Build the byte→unicode and unicode→byte tables once.
struct ByteUnicodeTable {
  std::array<std::string, 256> byte_to_str;         // byte → UTF-8 string
  std::unordered_map<std::string, int> str_to_byte; // UTF-8 string → byte

  ByteUnicodeTable() {
    // Direct-mapping bytes.
    for (int b = 33; b <= 126; ++b)
      byte_to_str[b] = CpToUtf8(b);
    for (int b = 161; b <= 172; ++b)
      byte_to_str[b] = CpToUtf8(b);
    for (int b = 174; b <= 255; ++b)
      byte_to_str[b] = CpToUtf8(b);
    // Remaining bytes → U+0100 onwards.
    uint32_t extra = 0x100;
    for (int b = 0; b < 256; ++b) {
      if (byte_to_str[b].empty())
        byte_to_str[b] = CpToUtf8(extra++);
    }
    for (int b = 0; b < 256; ++b)
      str_to_byte[byte_to_str[b]] = b;
  }
};

const ByteUnicodeTable &GetBUT() {
  static const ByteUnicodeTable t;
  return t;
}

// U+2581 ▁  (LOWER ONE EIGHTH BLOCK — used as space marker in Metaspace).
constexpr const char *kMetaMark = "\xe2\x96\x81";

// Upper bound on a vocab/added_tokens id read from tokenizer.json. Real
// vocabularies top out in the low hundreds of thousands (the largest known
// multilingual tokenizers are under 1M); this is a generous ceiling that
// still rejects a corrupted or adversarial id before it is used to size or
// index id_to_token_. Without this, a negative id becomes a huge size_t
// via implicit conversion in id_to_token_[id] (out-of-bounds write), and a
// merely large positive id causes id_to_token_.assign()/resize() to
// attempt allocating and zero-constructing hundreds of millions of
// std::string objects (memory-exhaustion denial of service).
constexpr int32_t kMaxReasonableTokenId = 10'000'000;

// Takes int64_t, not int32_t: nlohmann::json::get<int32_t>() performs an
// unchecked static_cast from its internal 64-bit storage with no range
// check, so an id like 2^32 + 1 silently truncates to 1 and would pass a
// same-width bounds check while colliding with (and overwriting) the
// legitimate token at id 1. Validating the untruncated 64-bit value before
// narrowing closes that gap.
bool IsValidTokenId(int64_t id) {
  return id >= 0 && id <= kMaxReasonableTokenId;
}

} // namespace

// ---------------------------------------------------------------------------
// Public static helpers
// ---------------------------------------------------------------------------

const std::string &MlxTokenizer::ByteToUnicode(uint8_t b) {
  return GetBUT().byte_to_str[b];
}

int MlxTokenizer::UnicodeToByte(const std::string &utf8_char) {
  auto it = GetBUT().str_to_byte.find(utf8_char);
  return it != GetBUT().str_to_byte.end() ? it->second : -1;
}

std::vector<std::string> MlxTokenizer::SplitUtf8(const std::string &s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((c & 0x80) == 0x00)
      len = 1;
    else if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pre-tokenization
// ---------------------------------------------------------------------------

std::vector<std::string>
MlxTokenizer::PreTokenize(const std::string &text) const {
  if (pre_tok_ == PreTokenizerType::Metaspace) {
    // Replace spaces with ▁ and optionally prepend ▁.
    std::string modified;
    if (add_prefix_space_ && !text.empty() && text[0] != ' ')
      modified = kMetaMark;
    for (char c : text) {
      if (c == ' ')
        modified += kMetaMark;
      else
        modified += c;
    }
    // Split at ▁ boundaries, keeping ▁ at the start of each chunk.
    std::vector<std::string> result;
    std::string current;
    size_t i = 0;
    while (i < modified.size()) {
      auto uc = static_cast<unsigned char>(modified[i]);
      if (uc == 0xE2 && i + 2 < modified.size() &&
          static_cast<unsigned char>(modified[i + 1]) == 0x96 &&
          static_cast<unsigned char>(modified[i + 2]) == 0x81) {
        if (!current.empty())
          result.push_back(current);
        current = kMetaMark;
        i += 3;
      } else {
        current += modified[i++];
      }
    }
    if (!current.empty())
      result.push_back(current);
    return result;
  }

  if (pre_tok_ == PreTokenizerType::ByteLevel) {
    // Encode each byte through the byte-to-unicode table.
    // Spaces are encoded as Ġ (U+0120) and prepended to the NEXT word.
    std::vector<std::string> result;
    std::string current;
    for (size_t i = 0; i < text.size();) {
      unsigned char c = static_cast<unsigned char>(text[i]);
      if (c == ' ') {
        if (!current.empty()) {
          result.push_back(current);
          current.clear();
        }
        // The encoded space becomes part of the next pre-token.
        current = ByteToUnicode(c);
        ++i;
      } else {
        // Determine UTF-8 char length.
        size_t len = 1;
        if ((c & 0x80) == 0x00)
          len = 1;
        else if ((c & 0xE0) == 0xC0)
          len = 2;
        else if ((c & 0xF0) == 0xE0)
          len = 3;
        else if ((c & 0xF8) == 0xF0)
          len = 4;
        // Encode each byte of this UTF-8 char.
        for (size_t b = 0; b < len && i + b < text.size(); ++b)
          current += ByteToUnicode(static_cast<unsigned char>(text[i + b]));
        i += len;
      }
    }
    if (!current.empty())
      result.push_back(current);
    return result;
  }

  // Unknown pre-tokenizer: split on whitespace.
  std::vector<std::string> result;
  std::istringstream iss(text);
  std::string w;
  while (iss >> w)
    result.push_back(w);
  return result;
}

// ---------------------------------------------------------------------------
// BPE encode
// ---------------------------------------------------------------------------

std::vector<std::string>
MlxTokenizer::BpeEncode(const std::string &word) const {
  auto chars = SplitUtf8(word);
  while (chars.size() > 1) {
    int32_t best_rank = INT_MAX;
    int best_idx = -1;
    for (int i = 0; i + 1 < static_cast<int>(chars.size()); ++i) {
      const std::string key = chars[i] + " " + chars[i + 1];
      auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }
    if (best_idx == -1)
      break;
    chars[best_idx] += chars[best_idx + 1];
    chars.erase(chars.begin() + best_idx + 1);
  }
  return chars;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

void MlxTokenizer::Reset() {
  loaded_ = false;
  pre_tok_ = PreTokenizerType::Unknown;
  add_prefix_space_ = true;
  add_bos_token_ = true;
  vocab_.clear();
  id_to_token_.clear();
  merge_rank_.clear();
  special_ids_.clear();
  special_token_strings_.clear();
  bos_id_ = 1;
  eos_id_ = 2;
  vocab_size_ = 0;
  chat_template_.clear();
}

MlxTokenizer::PreTokenizerType
MlxTokenizer::DetectPreTokenizerType(const std::string &pre_tokenizer_hint,
                                     bool *add_prefix_space) {
  if (add_prefix_space) {
    *add_prefix_space = true;
  }
  const std::string hint = ToLower(pre_tokenizer_hint);
  if (hint.empty()) {
    return PreTokenizerType::Unknown;
  }

  if (hint == "bytelevel" || hint == "gpt2" || hint == "qwen" ||
      hint == "qwen2" || hint == "qwen3" || hint == "phi3" ||
      hint == "starcoder" || hint == "starcoder2" || hint == "deepseek" ||
      hint == "deepseek-coder" || hint == "command-r" || hint == "byte_level") {
    if (add_prefix_space) {
      *add_prefix_space = false;
    }
    return PreTokenizerType::ByteLevel;
  }

  if (hint == "metaspace" || hint == "sentencepiece" || hint == "spm" ||
      hint == "llama" || hint == "mistral" || hint == "gemma" ||
      hint == "gemma2" || hint == "gemma3") {
    if (add_prefix_space) {
      *add_prefix_space = true;
    }
    return PreTokenizerType::Metaspace;
  }

  return PreTokenizerType::Unknown;
}

bool MlxTokenizer::InitializeFromBpeData(
    const std::vector<std::string> &id_to_token,
    const std::vector<std::string> &merges,
    const std::string &pre_tokenizer_hint, int32_t bos_id, int32_t eos_id,
    const std::string &chat_template, bool add_bos_token,
    const std::unordered_set<int32_t> &special_ids) {
  Reset();
  if (id_to_token.empty()) {
    log::Error("mlx_tokenizer",
               "InitializeFromBpeData requires a non-empty vocabulary");
    return false;
  }

  id_to_token_ = id_to_token;
  vocab_size_ = static_cast<int32_t>(id_to_token_.size());
  for (int32_t id = 0; id < vocab_size_; ++id) {
    vocab_.emplace(id_to_token_[static_cast<size_t>(id)], id);
  }

  int32_t rank = 0;
  for (const auto &merge : merges) {
    merge_rank_[merge] = rank++;
  }

  bool add_prefix_space = true;
  pre_tok_ = DetectPreTokenizerType(pre_tokenizer_hint, &add_prefix_space);
  add_prefix_space_ = add_prefix_space;
  bos_id_ = bos_id;
  eos_id_ = eos_id;
  add_bos_token_ = add_bos_token;
  chat_template_ = chat_template;
  special_ids_ = special_ids;
  if (bos_id_ >= 0) {
    special_ids_.insert(bos_id_);
  }
  if (eos_id_ >= 0) {
    special_ids_.insert(eos_id_);
  }

  // Build sorted list of special token strings for greedy matching during
  // encoding. Only include tokens with multi-char surface forms (single chars
  // are handled naturally by BPE).
  special_token_strings_.clear();
  for (int32_t id : special_ids_) {
    if (id >= 0 && id < vocab_size_) {
      const auto &tok = id_to_token_[static_cast<size_t>(id)];
      if (tok.size() > 1) {
        special_token_strings_.push_back({tok, id});
      }
    }
  }
  // Sort longest-first for greedy matching.
  std::sort(special_token_strings_.begin(), special_token_strings_.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });

  loaded_ = true;
  return true;
}

bool MlxTokenizer::Load(const std::filesystem::path &model_dir) {
  Reset();
  const auto tok_path = model_dir / "tokenizer.json";
  std::ifstream f(tok_path);
  if (!f.is_open()) {
    log::Error("mlx_tokenizer", "Cannot open " + tok_path.string());
    return false;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception &e) {
    log::Error("mlx_tokenizer",
               std::string("tokenizer.json parse error: ") + e.what());
    return false;
  }

  // Parse model section.
  if (!j.contains("model") || !j["model"].is_object()) {
    log::Error("mlx_tokenizer", "Missing 'model' section");
    return false;
  }
  const auto &model = j["model"];
  if (!model.contains("type") || model["type"] != "BPE") {
    log::Error("mlx_tokenizer", "Only BPE model type is supported");
    return false;
  }

  // Vocabulary.
  if (model.contains("vocab") && model["vocab"].is_object()) {
    int32_t max_id = -1;
    for (const auto &[tok, id_val] : model["vocab"].items()) {
      if (!id_val.is_number()) {
        log::Warn("mlx_tokenizer",
                  "Skipping vocab entry '" + tok + "' with non-numeric id");
        continue;
      }
      // Read as int64_t and range-check before narrowing -- get<int32_t>()
      // would silently truncate an out-of-range value instead of rejecting
      // it (see IsValidTokenId's comment).
      const int64_t id64 = id_val.get<int64_t>();
      if (!IsValidTokenId(id64)) {
        log::Warn("mlx_tokenizer", "Skipping vocab entry '" + tok +
                                       "' with out-of-range id " +
                                       std::to_string(id64));
        continue;
      }
      const int32_t id = static_cast<int32_t>(id64);
      vocab_[tok] = id;
      max_id = std::max(max_id, id);
    }
    id_to_token_.assign(max_id + 1, "");
    for (const auto &[tok, id] : vocab_)
      id_to_token_[id] = tok;
    vocab_size_ = max_id + 1;
  }

  // Merges. Two shapes are valid HuggingFace tokenizer.json output: a
  // space-joined string ("a b", the older/common form) or a 2-element array
  // of the two pieces (["a", "b"], used by some newer exporters). Both map
  // to the same internal merge_rank_ key so BpeEncode() doesn't need to
  // know which shape the file used.
  if (model.contains("merges") && model["merges"].is_array()) {
    int32_t rank = 0;
    for (const auto &m : model["merges"]) {
      if (m.is_string()) {
        merge_rank_[m.get<std::string>()] = rank++;
      } else if (m.is_array() && m.size() == 2 && m[0].is_string() &&
                 m[1].is_string()) {
        merge_rank_[m[0].get<std::string>() + " " + m[1].get<std::string>()] =
            rank++;
      } else {
        log::Warn("mlx_tokenizer",
                  "Skipping merges entry with unexpected shape (expected a "
                  "string or a 2-element array of strings)");
      }
    }
  }

  // Small type-safe field accessors. tokenizer.json comes from many
  // different exporters and hand-edited/malformed files can carry a field
  // under the expected key but with the wrong JSON type (e.g. a number
  // where a string is expected). nlohmann::json::value<T>() throws a
  // type_error in that case; Load() must return false on bad input, never
  // throw, so every field read below goes through one of these instead.
  auto get_string_field = [](const json &obj, const char *key,
                             const std::string &def) -> std::string {
    if (!obj.contains(key))
      return def;
    if (!obj[key].is_string()) {
      log::Warn("mlx_tokenizer",
                std::string("Ignoring non-string '") + key + "' field");
      return def;
    }
    return obj[key].get<std::string>();
  };
  auto get_bool_field = [](const json &obj, const char *key, bool def) -> bool {
    if (!obj.contains(key))
      return def;
    if (!obj[key].is_boolean()) {
      log::Warn("mlx_tokenizer",
                std::string("Ignoring non-boolean '") + key + "' field");
      return def;
    }
    return obj[key].get<bool>();
  };

  // Pre-tokenizer type.
  if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object()) {
    auto apply_pre_tokenizer_type = [this, &get_string_field,
                                     &get_bool_field](const json &pt) {
      const std::string type = get_string_field(pt, "type", "");
      if (type == "Metaspace") {
        pre_tok_ = PreTokenizerType::Metaspace;
        add_prefix_space_ = get_bool_field(pt, "add_prefix_space", true);
        return true;
      }
      if (type == "ByteLevel") {
        pre_tok_ = PreTokenizerType::ByteLevel;
        add_prefix_space_ = get_bool_field(pt, "add_prefix_space", false);
        return true;
      }
      return false;
    };

    const auto &pt = j["pre_tokenizer"];
    const std::string type = get_string_field(pt, "type", "");
    if (type == "Sequence" && pt.contains("pretokenizers") &&
        pt["pretokenizers"].is_array()) {
      // HuggingFace wraps most GPT-2-family tokenizers (Qwen, Llama-3,
      // Phi-3, StarCoder, ...) as Sequence[Split, ByteLevel]; the nested
      // entry, not the wrapper, determines byte encoding/decoding.
      for (const auto &nested : pt["pretokenizers"]) {
        if (nested.is_object() && apply_pre_tokenizer_type(nested))
          break;
      }
    } else {
      apply_pre_tokenizer_type(pt);
    }
  }

  // Added / special tokens.
  if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
    for (const auto &at : j["added_tokens"]) {
      if (!at.contains("id") || !at.contains("content"))
        continue;
      if (!at["id"].is_number() || !at["content"].is_string()) {
        log::Warn("mlx_tokenizer",
                  "Skipping added_tokens entry with non-numeric id or "
                  "non-string content");
        continue;
      }
      // Read as int64_t and range-check before narrowing -- see the
      // matching comment in the vocab loop above.
      const int64_t id64 = at["id"].get<int64_t>();
      if (!IsValidTokenId(id64)) {
        log::Warn("mlx_tokenizer", "Skipping added_tokens entry with "
                                   "out-of-range id " +
                                       std::to_string(id64));
        continue;
      }
      const int32_t id = static_cast<int32_t>(id64);
      const std::string content = at["content"].get<std::string>();
      // Insert into vocab if not already present.
      vocab_.emplace(content, id);
      if (id >= static_cast<int32_t>(id_to_token_.size()))
        id_to_token_.resize(id + 1);
      id_to_token_[id] = content;
      vocab_size_ = std::max(vocab_size_, id + 1);
      if (get_bool_field(at, "special", false))
        special_ids_.insert(id);
    }
  }

  // Resolve BOS/EOS from tokenizer_config.json.
  bool bos_resolved = false;
  bool eos_resolved = false;
  const auto cfg_path = model_dir / "tokenizer_config.json";
  std::ifstream cfg_f(cfg_path);
  if (cfg_f.is_open()) {
    json cfg;
    try {
      cfg_f >> cfg;
    } catch (const std::exception &) {
    }

    auto resolve_tok = [&](const char *key) -> std::string {
      if (!cfg.contains(key))
        return "";
      const auto &v = cfg[key];
      if (v.is_string())
        return v.get<std::string>();
      if (v.is_object() && v.contains("content") && v["content"].is_string())
        return v["content"].get<std::string>();
      return "";
    };

    const std::string bos_str = resolve_tok("bos_token");
    const std::string eos_str = resolve_tok("eos_token");
    if (!bos_str.empty() && vocab_.count(bos_str)) {
      bos_id_ = vocab_.at(bos_str);
      bos_resolved = true;
    }
    if (!eos_str.empty() && vocab_.count(eos_str)) {
      eos_id_ = vocab_.at(eos_str);
      eos_resolved = true;
    }

    // Chat template (Jinja2 string used by FormatChatMessages override).
    if (cfg.contains("chat_template") && cfg["chat_template"].is_string())
      chat_template_ = cfg["chat_template"].get<std::string>();
  }

  // Fall back to a standalone chat_template.jinja file — the convention
  // transformers v4.44+ / vLLM / SGLang use when tokenizer_config.json has
  // no embedded chat_template (or doesn't exist at all).
  if (chat_template_.empty()) {
    std::ifstream jinja_f(model_dir / "chat_template.jinja");
    if (jinja_f.is_open()) {
      std::ostringstream jinja_ss;
      jinja_ss << jinja_f.rdbuf();
      chat_template_ = jinja_ss.str();
    }
  }

  // Fall back to config.json's bos_token_id/eos_token_id (plain vocab IDs,
  // no string lookup needed) when tokenizer_config.json is absent or didn't
  // specify them. Without this, bos_id_/eos_id_ silently keep their
  // Reset() defaults (1/2) for model directories that ship only
  // tokenizer.json + config.json — wrong for most vocabularies, and fatal
  // to generation quality since the model never emits a stop token the
  // executor recognizes (see runtime/backends/cuda/inferflux_cuda_executor.cpp,
  // which builds its stop-token set directly from EosTokenId()).
  if (!bos_resolved || !eos_resolved) {
    std::ifstream model_cfg_f(model_dir / "config.json");
    if (model_cfg_f.is_open()) {
      json model_cfg;
      try {
        model_cfg_f >> model_cfg;
      } catch (const std::exception &) {
      }
      auto resolve_id = [&](const char *key) -> int32_t {
        if (!model_cfg.contains(key))
          return -1;
        const auto &v = model_cfg[key];
        if (v.is_number_integer())
          return v.get<int32_t>();
        if (v.is_array() && !v.empty() && v[0].is_number_integer())
          return v[0].get<int32_t>(); // some configs list multiple eos ids
        return -1;
      };
      if (!bos_resolved) {
        const int32_t id = resolve_id("bos_token_id");
        if (id >= 0)
          bos_id_ = id;
      }
      if (!eos_resolved) {
        const int32_t id = resolve_id("eos_token_id");
        if (id >= 0)
          eos_id_ = id;
      }
    }
  }

  // Build special token string list for greedy matching.
  special_token_strings_.clear();
  for (int32_t id : special_ids_) {
    if (id >= 0 && id < vocab_size_) {
      const auto &tok = id_to_token_[static_cast<size_t>(id)];
      if (tok.size() > 1) {
        special_token_strings_.push_back({tok, id});
      }
    }
  }
  std::sort(special_token_strings_.begin(), special_token_strings_.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });

  loaded_ = true;
  log::Info("mlx_tokenizer",
            "Loaded: vocab=" + std::to_string(vocab_size_) +
                " merges=" + std::to_string(merge_rank_.size()) +
                " bos=" + std::to_string(bos_id_) +
                " eos=" + std::to_string(eos_id_) + " special_tokens=" +
                std::to_string(special_token_strings_.size()));
  return true;
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

MlxTokenizerResult MlxTokenizer::Encode(const std::string &text,
                                        bool add_bos) const {
  MlxTokenizerResult result;
  if (!loaded_)
    return result;

  if (add_bos && add_bos_token_ && bos_id_ >= 0)
    result.ids.push_back(bos_id_);

  // Split text around special tokens first. Special tokens (e.g.
  // <|im_start|>, <|im_end|>) must be emitted as single token IDs, not
  // broken into bytes by the BPE encoder.
  struct Segment {
    std::string text;
    int32_t special_id{-1}; // >= 0 → emit this ID directly
  };
  std::vector<Segment> segments;
  if (special_token_strings_.empty()) {
    segments.push_back({text, -1});
  } else {
    size_t pos = 0;
    while (pos < text.size()) {
      bool matched = false;
      for (const auto &[tok_str, tok_id] : special_token_strings_) {
        if (pos + tok_str.size() <= text.size() &&
            text.compare(pos, tok_str.size(), tok_str) == 0) {
          segments.push_back({"", tok_id});
          pos += tok_str.size();
          matched = true;
          break;
        }
      }
      if (!matched) {
        if (segments.empty() || segments.back().special_id >= 0) {
          segments.push_back({"", -1});
        }
        segments.back().text += text[pos];
        ++pos;
      }
    }
  }

  for (const auto &seg : segments) {
    if (seg.special_id >= 0) {
      result.ids.push_back(seg.special_id);
      continue;
    }
    const auto pre_tokens = PreTokenize(seg.text);
    for (const auto &pre_tok : pre_tokens) {
      for (const auto &sub : BpeEncode(pre_tok)) {
        auto it = vocab_.find(sub);
        if (it != vocab_.end()) {
          result.ids.push_back(it->second);
        } else {
          // Unknown sub-token: try byte-fallback (emit byte-level token IDs).
          bool found_any = false;
          for (unsigned char b : sub) {
            const std::string byte_str = ByteToUnicode(b);
            auto bit = vocab_.find(byte_str);
            if (bit != vocab_.end()) {
              result.ids.push_back(bit->second);
              found_any = true;
            }
          }
          if (!found_any && vocab_.count("")) {
            result.ids.push_back(vocab_.at("")); // unk
          }
        }
      }
    }
  }

  result.ok = true;
  return result;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::string MlxTokenizer::Decode(const std::vector<int32_t> &ids,
                                 bool skip_special) const {
  if (!loaded_)
    return "";

  std::string combined;
  for (int32_t id : ids) {
    if (skip_special && special_ids_.count(id))
      continue;
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size()))
      continue;
    combined += id_to_token_[id];
  }

  if (pre_tok_ == PreTokenizerType::Metaspace) {
    // Replace ▁ with space, strip leading space.
    std::string result;
    const auto chars = SplitUtf8(combined);
    for (const auto &ch : chars) {
      if (ch == kMetaMark)
        result += ' ';
      else
        result += ch;
    }
    // Strip the leading space that add_prefix_space inserted.
    if (!result.empty() && result[0] == ' ')
      result.erase(0, 1);
    return result;
  }

  if (pre_tok_ == PreTokenizerType::ByteLevel) {
    // Decode each unicode char back to its original byte.
    std::string result;
    const auto chars = SplitUtf8(combined);
    for (const auto &ch : chars) {
      int b = UnicodeToByte(ch);
      if (b >= 0)
        result += static_cast<char>(b);
      else
        result += ch; // pass through (e.g. emojis not in table)
    }
    return result;
  }

  return combined;
}

} // namespace inferflux
