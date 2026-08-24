#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inferflux {

// Result of a tokenization call.
struct MlxTokenizerResult {
  std::vector<int32_t> ids;
  bool ok{false};
};

// Lightweight BPE tokenizer that reads HuggingFace tokenizer.json.
//
// Supports:
//   - Metaspace / SentencePiece pre-tokenizer  (LLaMA 2, Mistral)
//   - ByteLevel pre-tokenizer                  (LLaMA 3, GPT-2, Phi-3)
//
// Not supported yet (Stage 4): regex-based split for contractions,
// tiktoken format, SentencePiece binary (.model) format.
class MlxTokenizer {
public:
  MlxTokenizer() = default;

  // Load tokenizer.json (and optionally tokenizer_config.json) from model_dir.
  // Returns true on success.
  bool Load(const std::filesystem::path &model_dir);
  bool
  InitializeFromBpeData(const std::vector<std::string> &id_to_token,
                        const std::vector<std::string> &merges,
                        const std::string &pre_tokenizer_hint, int32_t bos_id,
                        int32_t eos_id, const std::string &chat_template = "",
                        bool add_bos_token = true,
                        const std::unordered_set<int32_t> &special_ids = {});
  bool Loaded() const { return loaded_; }

  // Encode text to a vector of token IDs. Prepends BOS when add_bos=true.
  // Returns ok=false if the tokenizer is not loaded.
  MlxTokenizerResult Encode(const std::string &text, bool add_bos = true) const;

  // Decode a vector of token IDs back to UTF-8 text.
  // Special tokens are omitted when skip_special=true (default).
  std::string Decode(const std::vector<int32_t> &ids,
                     bool skip_special = true) const;

  int32_t BosId() const { return bos_id_; }
  int32_t EosId() const { return eos_id_; }
  int32_t VocabSize() const { return vocab_size_; }
  bool IsSpecial(int32_t id) const { return special_ids_.count(id) > 0; }

  // Chat template (Jinja2 string from tokenizer_config.json, if present).
  const std::string &ChatTemplate() const { return chat_template_; }
  bool HasChatTemplate() const { return !chat_template_.empty(); }

private:
  enum class PreTokenizerType { Metaspace, ByteLevel, Unknown };

  void Reset();
  static PreTokenizerType
  DetectPreTokenizerType(const std::string &pre_tokenizer_hint,
                         bool *add_prefix_space);

  bool loaded_{false};
  PreTokenizerType pre_tok_{PreTokenizerType::Unknown};
  bool add_prefix_space_{true}; // Metaspace: prepend ▁ to first token

  // Vocabulary: both directions.
  std::unordered_map<std::string, int32_t> vocab_; // token string → id
  std::vector<std::string> id_to_token_;           // id → token string

  // BPE merges: "part_a part_b" → merge rank (lower = applied first).
  std::unordered_map<std::string, int32_t> merge_rank_;

  std::unordered_set<int32_t> special_ids_;
  int32_t bos_id_{1};
  int32_t eos_id_{2};
  int32_t vocab_size_{0};
  bool add_bos_token_{true};

  std::string chat_template_; // Jinja2 template string, empty if absent

  // Split UTF-8 string into individual codepoint strings (1–4 bytes each).
  static std::vector<std::string> SplitUtf8(const std::string &s);

  // Pre-tokenize input text into BPE-ready chunks.
  std::vector<std::string> PreTokenize(const std::string &text) const;

  // Apply BPE merge rules to a single pre-token, return list of sub-tokens.
  std::vector<std::string> BpeEncode(const std::string &word) const;

  // GPT-2/LLaMA-3 byte-to-unicode table (256 entries).
  static const std::string &ByteToUnicode(uint8_t b);

  // Reverse of ByteToUnicode: unicode codepoint string → byte value.
  // Returns -1 when the character is not in the table.
  static int UnicodeToByte(const std::string &utf8_char);
};

} // namespace inferflux
