#pragma once

#include <string>
#include <utility>
#include <vector>

namespace inferflux {

/**
 * ITokenizer: pluggable tokenizer strategy interface.
 *
 * All backends (CUDA native, ROCm, MPS, CPU, OpenCL, etc.) use this
 * interface for tokenization, detokenization, and chat template rendering.
 * Default implementation: LlamaTokenizer (llama.cpp, zero GPU layers).
 * Future implementations: SentencePiece, tiktoken, HuggingFace tokenizers.
 */
class ITokenizer {
public:
  virtual ~ITokenizer() = default;

  /** Load tokenizer from a model file or directory. */
  virtual bool Load(const std::string &model_path) = 0;

  /** Tokenize text to token IDs. */
  virtual std::vector<int> Tokenize(const std::string &text,
                                    bool add_bos = true) const = 0;

  /** Convert token IDs back to text. */
  virtual std::string Detokenize(const std::vector<int> &tokens) const = 0;

  /** Convert a single token ID to its text piece. */
  virtual std::string TokenToString(int token_id) const = 0;

  /** True when token_id is a non-user-visible control/special token. */
  virtual bool IsSpecialToken(int token_id) const {
    return token_id == BosTokenId() || token_id == EosTokenId();
  }

  /**
   * True when generation should stop on this token.
   *
   * Default policy is conservative for server-side generation: any
   * tokenizer-defined special/control token ends generation.
   */
  virtual bool IsTerminalGeneratedToken(int token_id) const {
    return token_id < 0 || IsSpecialToken(token_id);
  }

  /** Count tokens in text (convenience — defaults to Tokenize().size()). */
  virtual int TokenCount(const std::string &text) const {
    return static_cast<int>(Tokenize(text, false).size());
  }

  /** Chat template result. */
  struct ChatResult {
    std::string prompt;
    bool valid{false};
  };

  /**
   * Apply the model's chat template to a list of (role, content) messages.
   * Returns {prompt, valid}. If no template is available, valid=false.
   */
  virtual ChatResult ApplyChatTemplate(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const = 0;

  virtual int BosTokenId() const = 0;
  virtual int EosTokenId() const = 0;
  virtual int VocabSize() const = 0;
  virtual bool IsLoaded() const = 0;
};

} // namespace inferflux
