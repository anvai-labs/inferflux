#include "model/chat_template_renderer.h"

namespace inferflux {

namespace {

// Strategy: ChatML (Qwen, TinyLlama, Yi, Phi, OpenChat, etc.)
// Format: <|im_start|>role\ncontent<|im_end|>\n
std::string
RenderChatML(const std::vector<std::pair<std::string, std::string>> &messages,
             bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
  }
  if (add_assistant_prefix) {
    out += "<|im_start|>assistant\n";
  }
  return out;
}

// Strategy: Llama-2 / Llama-3 instruct
// Format: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant </s>
std::string
RenderLlama(const std::vector<std::pair<std::string, std::string>> &messages,
            bool add_assistant_prefix) {
  std::string out;
  bool first_user = true;
  for (const auto &[role, content] : messages) {
    if (role == "system") {
      out += "[INST] <<SYS>>\n" + content + "\n<</SYS>>\n\n";
    } else if (role == "user") {
      if (first_user && out.empty()) {
        out += "[INST] ";
      } else if (!first_user) {
        out += "[INST] ";
      }
      out += content + " [/INST]";
      first_user = false;
    } else if (role == "assistant") {
      out += " " + content + " </s>";
    }
  }
  (void)add_assistant_prefix; // Llama format implicit
  return out;
}

// Strategy: Mistral / Mixtral
// Format: [INST] content [/INST] (no system wrapping)
std::string
RenderMistral(const std::vector<std::pair<std::string, std::string>> &messages,
              bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    if (role == "user") {
      out += "[INST] " + content + " [/INST]";
    } else if (role == "assistant") {
      out += content + "</s> ";
    } else if (role == "system") {
      out += "[INST] " + content + "\n\n";
    }
  }
  (void)add_assistant_prefix;
  return out;
}

// Strategy: Gemma (Google)
// Format: <start_of_turn>role\ncontent<end_of_turn>\n
std::string
RenderGemma(const std::vector<std::pair<std::string, std::string>> &messages,
            bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    out += "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
  }
  if (add_assistant_prefix) {
    out += "<start_of_turn>model\n";
  }
  return out;
}

// Detect template family from a Jinja2 template string. Returns a function
// pointer to the matching renderer.
using TemplateRenderer = std::string (*)(
    const std::vector<std::pair<std::string, std::string>> &, bool);

TemplateRenderer DetectTemplateFamily(const std::string &jinja_template) {
  if (jinja_template.find("im_start") != std::string::npos)
    return &RenderChatML;
  if (jinja_template.find("start_of_turn") != std::string::npos)
    return &RenderGemma;
  if (jinja_template.find("[INST]") != std::string::npos) {
    // Distinguish Llama (has <<SYS>>) from Mistral (no <<SYS>>)
    if (jinja_template.find("<<SYS>>") != std::string::npos ||
        jinja_template.find("bos_token") != std::string::npos)
      return &RenderLlama;
    return &RenderMistral;
  }
  // Default: ChatML is the safest fallback for instruct models
  return &RenderChatML;
}

} // namespace

std::string RenderChatTemplate(
    const std::string &jinja_template,
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  auto renderer = DetectTemplateFamily(jinja_template);
  return renderer(messages, add_assistant_prefix);
}

} // namespace inferflux
