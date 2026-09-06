#pragma once

#include <string>
#include <utility>
#include <vector>

namespace inferflux {

// Renders (role, content) chat messages into a model's native prompt
// format, selecting the rendering strategy (ChatML/Llama/Mistral/Gemma)
// from the Jinja2 chat_template string found in the model's tokenizer
// metadata (GGUF `tokenizer.chat_template` or HF `tokenizer_config.json` /
// `chat_template.jinja`). Shared by GGUFTokenizer and HFTokenizer so both
// backends produce identical prompt formatting for the same template
// family instead of drifting apart. Detection is family-level, not a full
// Jinja2 renderer — sufficient for standard single-turn/no-tools chat
// messages.
std::string RenderChatTemplate(
    const std::string &jinja_template,
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix);

} // namespace inferflux
