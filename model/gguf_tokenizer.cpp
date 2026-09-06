#include "model/gguf_tokenizer.h"

#include "model/chat_template_renderer.h"
#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "server/logging/logger.h"

namespace inferflux {

bool GGUFTokenizer::Load(const std::string &model_path) {
  runtime::cuda::native::GGUFModelLoader loader;
  if (!loader.Load(model_path)) {
    log::Error("gguf_tokenizer",
               "Failed to load GGUF metadata from: " + model_path);
    return false;
  }
  return InitializeFromMetadata(
      loader.TokenizerPieces(), loader.TokenizerMerges(),
      loader.TokenizerPreTokenizer(), loader.TokenizerBosTokenId(),
      loader.TokenizerEosTokenId(), loader.TokenizerAddBosToken(),
      loader.TokenizerChatTemplate(), loader.TokenizerTokenTypes());
}

bool GGUFTokenizer::InitializeFromMetadata(
    const std::vector<std::string> &pieces,
    const std::vector<std::string> &merges,
    const std::string &pre_tokenizer_hint, int bos_token_id, int eos_token_id,
    bool add_bos_token, const std::string &chat_template,
    const std::vector<int32_t> &token_types) {
  // Build special_ids from GGUF token_type array.
  // GGUF token types: 1=normal, 2=unknown, 3=control, 4=user_defined,
  // 5=unused, 6=byte. Type 3 (control) marks special tokens like
  // <|im_start|>, <|im_end|>, <|endoftext|>.
  std::unordered_set<int32_t> special_ids;
  for (size_t i = 0; i < token_types.size(); ++i) {
    if (token_types[i] == 3) { // control token
      special_ids.insert(static_cast<int32_t>(i));
    }
  }
  log::Debug("gguf_tokenizer",
             "Token types: " + std::to_string(token_types.size()) +
                 " entries, " + std::to_string(special_ids.size()) +
                 " control tokens");
  if (!inner_.InitializeFromBpeData(pieces, merges, pre_tokenizer_hint,
                                    bos_token_id, eos_token_id, chat_template,
                                    add_bos_token, special_ids)) {
    return false;
  }
  chat_template_ = chat_template;
  return true;
}

std::vector<int> GGUFTokenizer::Tokenize(const std::string &text,
                                         bool add_bos) const {
  auto encoded = inner_.Encode(text, add_bos);
  if (!encoded.ok) {
    return {};
  }
  return std::vector<int>(encoded.ids.begin(), encoded.ids.end());
}

std::string GGUFTokenizer::Detokenize(const std::vector<int> &tokens) const {
  std::vector<int32_t> ids(tokens.begin(), tokens.end());
  return inner_.Decode(ids, /*skip_special=*/true);
}

std::string GGUFTokenizer::TokenToString(int token_id) const {
  return inner_.Decode({static_cast<int32_t>(token_id)},
                       /*skip_special=*/false);
}

bool GGUFTokenizer::IsSpecialToken(int token_id) const {
  return inner_.IsSpecial(static_cast<int32_t>(token_id));
}

int GGUFTokenizer::TokenCount(const std::string &text) const {
  return static_cast<int>(Tokenize(text, /*add_bos=*/false).size());
}

// Chat template rendering (ChatML/Llama/Mistral/Gemma) is shared with
// HFTokenizer — see model/chat_template_renderer.h. The correct family is
// selected from the Jinja2 template string stored in GGUF metadata
// (tokenizer.chat_template), provided by the GGUF loader at init time.

ITokenizer::ChatResult GGUFTokenizer::ApplyChatTemplate(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) const {
  if (messages.empty()) {
    return {};
  }

  std::string prompt =
      RenderChatTemplate(chat_template_, messages, add_assistant_prefix);

  ChatResult result;
  result.prompt = prompt;
  result.valid = !prompt.empty();
  return result;
}

int GGUFTokenizer::BosTokenId() const { return inner_.BosId(); }

int GGUFTokenizer::EosTokenId() const { return inner_.EosId(); }

int GGUFTokenizer::VocabSize() const { return inner_.VocabSize(); }

} // namespace inferflux
