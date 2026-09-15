#include <catch2/catch_amalgamated.hpp>

#include <regex>

#include "model/chat_template_renderer.h"

using namespace inferflux;
using Messages = std::vector<std::pair<std::string, std::string>>;

// ---------------------------------------------------------------------------
// Family detection and rendering — shared by GGUFTokenizer and HFTokenizer.
// ---------------------------------------------------------------------------

TEST_CASE("RenderChatTemplate detects ChatML from im_start marker",
          "[chat_template_renderer]") {
  const std::string jinja = "{{ '<|im_start|>' + role }}";
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate(jinja, messages, true);
  REQUIRE(out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("RenderChatTemplate detects Gemma from start_of_turn marker",
          "[chat_template_renderer]") {
  const std::string jinja = "{{ '<start_of_turn>' + role }}";
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate(jinja, messages, true);
  REQUIRE(out ==
          "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n");
}

TEST_CASE("RenderChatTemplate detects Llama from [INST] + <<SYS>> markers",
          "[chat_template_renderer]") {
  const std::string jinja = "[INST] <<SYS>>{{ bos_token }}";
  const Messages messages = {{"system", "sys"}, {"user", "hi"}};
  const std::string out = RenderChatTemplate(jinja, messages, true);
  REQUIRE(out == "[INST] <<SYS>>\nsys\n<</SYS>>\n\nhi [/INST]");
}

TEST_CASE("RenderChatTemplate detects Mistral from [INST] without <<SYS>>",
          "[chat_template_renderer]") {
  const std::string jinja = "[INST] {{ message }} [/INST]";
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate(jinja, messages, true);
  REQUIRE(out == "[INST] hi [/INST]");
}

TEST_CASE("RenderChatTemplate defaults to ChatML for an empty template",
          "[chat_template_renderer]") {
  // Undiscoverable template (no tokenizer_config.json / chat_template.jinja)
  // must still produce a usable, well-formed prompt rather than an empty
  // or raw-concatenated one — ChatML is the safest default for instruct
  // models, matching GGUFTokenizer's long-standing fallback behavior.
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate("", messages, true);
  REQUIRE(out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("RenderChatTemplate omits assistant prefix when not requested",
          "[chat_template_renderer]") {
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate("im_start", messages, false);
  REQUIRE(out == "<|im_start|>user\nhi<|im_end|>\n");
}

// ---------------------------------------------------------------------------
// Harmony (gpt-oss) — transcribed from the model's own embedded jinja
// template (extracted from the GGUF and read directly); see the PROVENANCE
// comment on RenderHarmony in chat_template_renderer.cpp.
// ---------------------------------------------------------------------------

TEST_CASE("DetectChatTemplateFamily detects Harmony from <|channel|> marker",
          "[chat_template_renderer]") {
  const std::string jinja = "{{ '<|channel|>' + name }}";
  REQUIRE(DetectChatTemplateFamily(jinja) == ChatTemplateFamily::kHarmony);
}

TEST_CASE("RenderChatTemplate: harmony system preamble is fixed and "
          "well-formed",
          "[chat_template_renderer]") {
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);

  REQUIRE(out.rfind("<|start|>system<|message|>You are ChatGPT, a large "
                    "language model trained by OpenAI.\n",
                    0) == 0);
  REQUIRE(out.find("Knowledge cutoff: 2024-06\n") != std::string::npos);
  REQUIRE(out.find("Reasoning: medium\n\n") != std::string::npos);
  REQUIRE(out.find("# Valid channels: analysis, commentary, final. Channel "
                   "must be included for every message.<|end|>") !=
          std::string::npos);

  static const std::regex kDatePattern(
      R"(Current date: \d{4}-\d{2}-\d{2}\n\n)");
  REQUIRE(std::regex_search(out, kDatePattern));
}

TEST_CASE("RenderChatTemplate: harmony renders user/assistant turns and "
          "generation prompt",
          "[chat_template_renderer]") {
  const Messages messages = {
      {"user", "What is 2+2?"}, {"assistant", "4"}, {"user", "Thanks"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, true);

  REQUIRE(out.find("<|start|>user<|message|>What is 2+2?<|end|>") !=
          std::string::npos);
  REQUIRE(out.find("<|start|>assistant<|channel|>final<|message|>4<|end|>") !=
          std::string::npos);
  REQUIRE(out.find("<|start|>user<|message|>Thanks<|end|>") !=
          std::string::npos);
  // Generation prompt is the bare role marker — the model picks its own
  // channel (per "Channel must be included for every message" above).
  REQUIRE(out.rfind("<|start|>assistant") ==
          out.size() - std::string("<|start|>assistant").size());
}

TEST_CASE("RenderChatTemplate: harmony omits generation prompt when not "
          "requested",
          "[chat_template_renderer]") {
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);
  REQUIRE(out.find("<|start|>user<|message|>hi<|end|>") != std::string::npos);
  // No trailing bare "<|start|>assistant" generation-prompt marker.
  REQUIRE(out.rfind("<|start|>assistant") == std::string::npos);
}

TEST_CASE("RenderChatTemplate: harmony maps a leading system role to the "
          "developer block",
          "[chat_template_renderer]") {
  const Messages messages = {{"system", "Be concise."}, {"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);

  REQUIRE(out.find("<|start|>developer<|message|># Instructions\n\n"
                   "Be concise.\n\n<|end|>") != std::string::npos);
  // The developer block precedes the user turn.
  REQUIRE(out.find("<|start|>developer<|message|>") <
          out.find("<|start|>user<|message|>"));
}

TEST_CASE("RenderChatTemplate: harmony has no developer block when no "
          "system role is present",
          "[chat_template_renderer]") {
  const Messages messages = {{"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);
  REQUIRE(out.find("<|start|>developer") == std::string::npos);
}

TEST_CASE("RenderChatTemplate: harmony maps a leading developer role to the "
          "developer block",
          "[chat_template_renderer]") {
  // OpenAI-compatible clients targeting reasoning models increasingly send
  // "developer" directly instead of "system".
  const Messages messages = {{"developer", "Be concise."}, {"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);
  REQUIRE(out.find("<|start|>developer<|message|># Instructions\n\n"
                   "Be concise.\n\n<|end|>") != std::string::npos);
}

TEST_CASE("RenderChatTemplate: harmony folds multiple leading system/"
          "developer messages into one developer block instead of dropping "
          "them",
          "[chat_template_renderer]") {
  // InferFlux's HTTP layer prepends a synthesized system message ahead of
  // the caller's own when tools[] is present (BuildToolSystemPrompt); the
  // real gpt-oss template only ever reads messages[0], which would
  // silently drop the caller's real instructions in that combination.
  const Messages messages = {{"system", "TOOL SCHEMA: {...}"},
                             {"system", "Always answer in French."},
                             {"user", "hi"}};
  const std::string out = RenderChatTemplate("<|channel|>", messages, false);
  REQUIRE(out.find("<|start|>developer<|message|># Instructions\n\n"
                   "TOOL SCHEMA: {...}\n\nAlways answer in French.\n\n"
                   "<|end|>") != std::string::npos);
  // Only one developer block, not two.
  REQUIRE(out.find("<|start|>developer", out.find("<|start|>developer") + 1) ==
          std::string::npos);
  REQUIRE(out.find("<|start|>user<|message|>hi<|end|>") != std::string::npos);
}
