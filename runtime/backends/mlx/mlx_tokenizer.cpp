#include "runtime/backends/mlx/mlx_tokenizer.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <cctype>
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

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
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
      int32_t id = id_val.get<int32_t>();
      vocab_[tok] = id;
      max_id = std::max(max_id, id);
    }
    id_to_token_.assign(max_id + 1, "");
    for (const auto &[tok, id] : vocab_)
      id_to_token_[id] = tok;
    vocab_size_ = max_id + 1;
  }

  // Merges.
  if (model.contains("merges") && model["merges"].is_array()) {
    int32_t rank = 0;
    for (const auto &m : model["merges"]) {
      const std::string s = m.get<std::string>();
      merge_rank_[s] = rank++;
    }
  }

  // Pre-tokenizer type.
  if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object()) {
    const auto &pt = j["pre_tokenizer"];
    const std::string type = pt.value("type", "");
    if (type == "Metaspace") {
      pre_tok_ = PreTokenizerType::Metaspace;
      add_prefix_space_ = pt.value("add_prefix_space", true);
    } else if (type == "ByteLevel") {
      pre_tok_ = PreTokenizerType::ByteLevel;
      add_prefix_space_ = pt.value("add_prefix_space", false);
    }
  }

  // Added / special tokens.
  if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
    for (const auto &at : j["added_tokens"]) {
      if (!at.contains("id") || !at.contains("content"))
        continue;
      const int32_t id = at["id"].get<int32_t>();
      const std::string content = at["content"].get<std::string>();
      // Insert into vocab if not already present.
      vocab_.emplace(content, id);
      if (id >= static_cast<int32_t>(id_to_token_.size()))
        id_to_token_.resize(id + 1);
      id_to_token_[id] = content;
      vocab_size_ = std::max(vocab_size_, id + 1);
      if (at.value("special", false))
        special_ids_.insert(id);
    }
  }

  // Resolve BOS/EOS from tokenizer_config.json.
  const auto cfg_path = model_dir / "tokenizer_config.json";
  std::ifstream cfg_f(cfg_path);
  if (cfg_f.is_open()) {
    json cfg;
    try {
      cfg_f >> cfg;
    } catch (...) {
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
    if (!bos_str.empty() && vocab_.count(bos_str))
      bos_id_ = vocab_.at(bos_str);
    if (!eos_str.empty() && vocab_.count(eos_str))
      eos_id_ = vocab_.at(eos_str);

    // Chat template (Jinja2 string used by FormatChatMessages override).
    if (cfg.contains("chat_template") && cfg["chat_template"].is_string())
      chat_template_ = cfg["chat_template"].get<std::string>();
  }

  loaded_ = true;
  log::Info("mlx_tokenizer",
            "Loaded: vocab=" + std::to_string(vocab_size_) +
                " merges=" + std::to_string(merge_rank_.size()) + " bos=" +
                std::to_string(bos_id_) + " eos=" + std::to_string(eos_id_));
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

  const auto pre_tokens = PreTokenize(text);
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
