#include "model/chat_template_renderer.h"

#include <ctime>

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

// Server-clock date in the exact "Current date: YYYY-MM-DD" form gpt-oss's
// own system preamble expects (harmony's minja `strftime_now("%Y-%m-%d")`).
std::string CurrentDateYyyyMmDd() {
  std::time_t now = std::time(nullptr);
  std::tm tm_buf{};
#ifdef _WIN32
  if (gmtime_s(&tm_buf, &now) != 0) {
    return {};
  }
#else
  if (gmtime_r(&now, &tm_buf) == nullptr) {
    return {};
  }
#endif
  char buf[16]{};
  if (std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf) == 0) {
    return {};
  }
  return std::string(buf);
}

// Strategy: Harmony (OpenAI gpt-oss channel-based format)
// Format: <|start|>role<|message|>content<|end|>, with a fixed system
// preamble and channel markers (analysis/final) on assistant turns.
//
// PROVENANCE: this is a hand transcription of gpt-oss-20b's embedded
// `tokenizer.chat_template` (extracted from the GGUF and read directly,
// 2026-09-15) — not a Jinja2 executor. If a future gpt-oss checkpoint
// changes the system preamble wording or channel contract, this drifts out
// of sync silently (degraded quality, not a crash) — re-diff against the
// live GGUF template if output quality regresses on a newer checkpoint.
//
// v1 scope: no tool/builtin_tools rendering (the template's TypeScript
// tool-signature macro is skipped entirely — tool calling with gpt-oss is
// not implemented), reasoning_effort is hardcoded to "medium" (not exposed
// as a request-level knob yet), and prior assistant turns are re-rendered
// with only their final-channel content — reasoning_content is never
// round-tripped back into message history (matches how the real template
// treats history too: CoT is dropped for every turn except a training-only
// edge case that never applies to inference).
std::string
RenderHarmony(const std::vector<std::pair<std::string, std::string>> &messages,
              bool add_assistant_prefix) {
  std::string out;
  out += "<|start|>system<|message|>";
  out += "You are ChatGPT, a large language model trained by OpenAI.\n";
  out += "Knowledge cutoff: 2024-06\n";
  out += "Current date: " + CurrentDateYyyyMmDd() + "\n\n";
  out += "Reasoning: medium\n\n";
  out += "# Valid channels: analysis, commentary, final. Channel must be "
         "included for every message.";
  out += "<|end|>";

  // Leading system/developer-role messages map to harmony's "developer"
  // block. The real template only ever reads messages[0] — a stricter
  // transcription would silently drop a second leading system message,
  // which InferFlux's HTTP layer produces routinely (BuildToolSystemPrompt
  // prepends its own synthesized system message ahead of the caller's real
  // one when tools[] is present). Fold every leading system/developer
  // message into one combined block instead of dropping the rest — an
  // intentional divergence from strict transcription to avoid silent
  // instruction loss, not a transcription error. "developer" is also
  // recognized directly: it's the role OpenAI-compatible clients targeting
  // reasoning models increasingly send instead of "system".
  std::size_t start_idx = 0;
  std::string developer_content;
  while (start_idx < messages.size() &&
         (messages[start_idx].first == "system" ||
          messages[start_idx].first == "developer")) {
    if (!developer_content.empty()) {
      developer_content += "\n\n";
    }
    developer_content += messages[start_idx].second;
    ++start_idx;
  }
  if (!developer_content.empty()) {
    out += "<|start|>developer<|message|># Instructions\n\n";
    out += developer_content;
    out += "\n\n<|end|>";
  }

  for (std::size_t i = start_idx; i < messages.size(); ++i) {
    const auto &role = messages[i].first;
    const auto &content = messages[i].second;
    if (role == "user") {
      out += "<|start|>user<|message|>" + content + "<|end|>";
    } else if (role == "assistant") {
      out +=
          "<|start|>assistant<|channel|>final<|message|>" + content + "<|end|>";
    }
    // Other roles (a stray mid-conversation "system"/"developer", or "tool"
    // without tool-calling support) are dropped, matching gpt-oss's own
    // template — its per-turn loop only handles assistant/tool/user, and a
    // mid-list system message matches none of those branches there either.
  }

  if (add_assistant_prefix) {
    out += "<|start|>assistant";
  }
  return out;
}

// Detect template family from a Jinja2 template string.
ChatTemplateFamily DetectTemplateFamilyImpl(const std::string &jinja_template) {
  if (jinja_template.find("<|channel|>") != std::string::npos)
    return ChatTemplateFamily::kHarmony;
  if (jinja_template.find("im_start") != std::string::npos)
    return ChatTemplateFamily::kChatML;
  if (jinja_template.find("start_of_turn") != std::string::npos)
    return ChatTemplateFamily::kGemma;
  if (jinja_template.find("[INST]") != std::string::npos) {
    // Distinguish Llama (has <<SYS>>) from Mistral (no <<SYS>>)
    if (jinja_template.find("<<SYS>>") != std::string::npos ||
        jinja_template.find("bos_token") != std::string::npos)
      return ChatTemplateFamily::kLlama;
    return ChatTemplateFamily::kMistral;
  }
  // Default: ChatML is the safest fallback for instruct models
  return ChatTemplateFamily::kChatML;
}

} // namespace

ChatTemplateFamily DetectChatTemplateFamily(const std::string &jinja_template) {
  return DetectTemplateFamilyImpl(jinja_template);
}

std::string RenderChatTemplate(
    const std::string &jinja_template,
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  switch (DetectTemplateFamilyImpl(jinja_template)) {
  case ChatTemplateFamily::kHarmony:
    return RenderHarmony(messages, add_assistant_prefix);
  case ChatTemplateFamily::kGemma:
    return RenderGemma(messages, add_assistant_prefix);
  case ChatTemplateFamily::kLlama:
    return RenderLlama(messages, add_assistant_prefix);
  case ChatTemplateFamily::kMistral:
    return RenderMistral(messages, add_assistant_prefix);
  case ChatTemplateFamily::kChatML:
  default:
    return RenderChatML(messages, add_assistant_prefix);
  }
}

} // namespace inferflux
