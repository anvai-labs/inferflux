#include "model/hf_tokenizer.h"
#include "server/logging/logger.h"

#include <filesystem>

namespace inferflux {

bool HFTokenizer::Load(const std::string &model_path) {
  std::filesystem::path p(model_path);

  // If model_path is a file (e.g. a .gguf), look in its parent directory
  if (std::filesystem::is_regular_file(p)) {
    p = p.parent_path();
  }

  if (!inner_.Load(p)) {
    return false;
  }
  log::Info("hf_tokenizer", "Loaded HuggingFace tokenizer from: " + p.string() +
                                " vocab=" + std::to_string(inner_.VocabSize()) +
                                " bos=" + std::to_string(inner_.BosId()) +
                                " eos=" + std::to_string(inner_.EosId()));
  return true;
}

std::vector<int> HFTokenizer::Tokenize(const std::string &text,
                                       bool add_bos) const {
  auto result = inner_.Encode(text, add_bos);
  if (!result.ok) {
    return {};
  }
  return std::vector<int>(result.ids.begin(), result.ids.end());
}

std::string HFTokenizer::Detokenize(const std::vector<int> &tokens) const {
  std::vector<int32_t> ids32(tokens.begin(), tokens.end());
  return inner_.Decode(ids32, /*skip_special=*/true);
}

std::string HFTokenizer::TokenToString(int token_id) const {
  return inner_.Decode({static_cast<int32_t>(token_id)},
                       /*skip_special=*/false);
}

bool HFTokenizer::IsSpecialToken(int token_id) const {
  return inner_.IsSpecial(static_cast<int32_t>(token_id));
}

int HFTokenizer::TokenCount(const std::string &text) const {
  return static_cast<int>(Tokenize(text, false).size());
}

ITokenizer::ChatResult HFTokenizer::ApplyChatTemplate(
    const std::vector<std::pair<std::string, std::string>> & /*messages*/,
    bool /*add_assistant_prefix*/) const {
  // MlxTokenizer stores the Jinja2 template string but doesn't render it.
  // Chat template rendering requires a Jinja2 engine.
  // For now, return invalid so callers use their fallback path.
  // TODO: Add minimal Jinja2 renderer or use llama_chat_apply_template.
  return {};
}

int HFTokenizer::BosTokenId() const { return inner_.BosId(); }

int HFTokenizer::EosTokenId() const { return inner_.EosId(); }

int HFTokenizer::VocabSize() const { return inner_.VocabSize(); }

} // namespace inferflux
