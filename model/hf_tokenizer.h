#pragma once

#include "model/tokenizer.h"
#include "runtime/backends/mlx/mlx_tokenizer.h"

#include <string>
#include <utility>
#include <vector>

namespace inferflux {

/**
 * HFTokenizer: ITokenizer adapter for HuggingFace tokenizer.json format.
 *
 * Wraps MlxTokenizer (which is a standalone BPE tokenizer, despite the name)
 * to provide ITokenizer interface for safetensors models that ship with
 * tokenizer.json and tokenizer_config.json.
 *
 * Supports:
 *   - Metaspace / SentencePiece pre-tokenizer (LLaMA 2, Mistral)
 *   - ByteLevel pre-tokenizer (LLaMA 3, GPT-2, Phi-3)
 *   - Chat templates from tokenizer_config.json (Jinja2-based)
 */
class HFTokenizer final : public ITokenizer {
public:
  HFTokenizer() = default;
  ~HFTokenizer() override = default;

  bool Load(const std::string &model_path) override;

  std::vector<int> Tokenize(const std::string &text,
                            bool add_bos = true) const override;

  std::string Detokenize(const std::vector<int> &tokens) const override;
  std::string TokenToString(int token_id) const override;
  bool IsSpecialToken(int token_id) const override;
  int TokenCount(const std::string &text) const override;

  ChatResult ApplyChatTemplate(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const override;

  int BosTokenId() const override;
  int EosTokenId() const override;
  int VocabSize() const override;
  bool IsLoaded() const override { return inner_.Loaded(); }

private:
  MlxTokenizer inner_;
};

} // namespace inferflux
