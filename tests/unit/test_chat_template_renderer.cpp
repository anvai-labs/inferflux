#include <catch2/catch_amalgamated.hpp>

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
