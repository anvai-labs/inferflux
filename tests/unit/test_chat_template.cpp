// Unit tests for §2.3 model-native chat template support.
//
// Covers:
//   1. LlamaCppBackend::FormatChatMessages — returns invalid when model absent.
//   2. LlamaCppBackend::ChatTemplateResult struct defaults.
//   3. FormatChatMessages with empty messages list.
//   4. FormatChatMessages with test_ready_ set but no real model_ pointer.
//
// Note: multi-format tool call extraction (DetectToolCalls) is the REAL
// implementation from server/http/completion_payload.{h,cpp} - inferflux_tests
// links inferflux_core, so these cases exercise the shipped detector directly
// (the file previously tested a drift-prone ported copy).

#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"

#include "server/http/completion_payload.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace inferflux;
using json = nlohmann::json;

// Adapter: first extracted call (or a default-constructed "no detection"
// result) from the real detector - preserves the historical per-case shape.
static ToolCallResult first_detected(const std::string &text) {
  auto ex = DetectToolCalls(text);
  return ex.calls.empty() ? ToolCallResult{} : ex.calls.front();
}

TEST_CASE("DetectToolCall: plain text has no tool call", "[chat_template]") {
  auto m = first_detected("The weather in London is sunny.");
  REQUIRE_FALSE(m.detected);
}

TEST_CASE("DetectToolCall: InferFlux preamble format detected",
          "[chat_template]") {
  std::string text =
      R"({"tool_call":{"name":"get_weather","arguments":{"location":"London"}}})";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "get_weather");
  REQUIRE_FALSE(m.arguments_json.empty());
}

TEST_CASE(
    "DetectToolCall: InferFlux format preceded by prose (no trailing garbage)",
    "[chat_template]") {
  // json::parse requires the substring from the '{' to end-of-string to be
  // valid JSON; trailing non-JSON chars after '}' prevent detection.
  std::string text =
      R"(Sure! {"tool_call":{"name":"search","arguments":{"q":"cats"}}})";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "search");
}

TEST_CASE("DetectToolCall: Hermes/Llama-3.1 XML tool_call format",
          "[chat_template]") {
  std::string text =
      R"(<tool_call>{"name":"get_time","arguments":{"tz":"UTC"}}</tool_call>)";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "get_time");
}

TEST_CASE("DetectToolCall: XML format with surrounding prose",
          "[chat_template]") {
  std::string text = "I will call a tool now.\n"
                     "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"id\":"
                     "42}}\n</tool_call>\n"
                     "Done.";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "lookup");
}

TEST_CASE("DetectToolCall: Mistral [TOOL_CALLS] format", "[chat_template]") {
  std::string text =
      R"([TOOL_CALLS] [{"name":"calculator","arguments":{"expr":"2+2"}}])";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "calculator");
}

TEST_CASE("DetectToolCall: parameters key accepted as alias for arguments",
          "[chat_template]") {
  std::string text =
      R"(<tool_call>{"name":"fn","parameters":{"x":1}}</tool_call>)";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "fn");
  REQUIRE(m.arguments_json == "{\"x\":1}");
}

TEST_CASE("DetectToolCall: bare generic JSON format", "[chat_template]") {
  std::string text = R"({"name":"ping","arguments":{}})";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "ping");
  REQUIRE(m.arguments_json == "{}");
}

TEST_CASE("DetectToolCall: malformed JSON does not crash", "[chat_template]") {
  REQUIRE_NOTHROW(first_detected("{\"tool_call\": broken}"));
  REQUIRE_NOTHROW(first_detected("<tool_call>bad</tool_call>"));
  REQUIRE_NOTHROW(first_detected("[TOOL_CALLS] not-json"));
}

TEST_CASE("DetectToolCall: Mistral format with [/TOOL_CALLS] sentinel",
          "[chat_template]") {
  // Mistral-family models append [/TOOL_CALLS] after the JSON array.
  // The parser must strip it before calling json::parse or detection fails.
  std::string text =
      R"([TOOL_CALLS] [{"name":"weather","arguments":{"city":"Paris"}}][/TOOL_CALLS])";
  auto m = first_detected(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "weather");
}

TEST_CASE("DetectToolCall: empty string returns no detection",
          "[chat_template]") {
  auto m = first_detected("");
  REQUIRE_FALSE(m.detected);
}
