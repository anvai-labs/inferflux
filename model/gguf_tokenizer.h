#pragma once

#include "model/tokenizer.h"
#include "runtime/backends/mlx/mlx_tokenizer.h"

#include <string>
#include <vector>

namespace inferflux {

class GGUFTokenizer final : public ITokenizer {
public:
  GGUFTokenizer() = default;
  ~GGUFTokenizer() override = default;

  bool Load(const std::string &model_path) override;
  bool InitializeFromMetadata(const std::vector<std::string> &pieces,
                              const std::vector<std::string> &merges,
                              const std::string &pre_tokenizer_hint,
                              int bos_token_id, int eos_token_id,
                              bool add_bos_token = true,
                              const std::string &chat_template = "");

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
  std::string chat_template_;
};

} // namespace inferflux
