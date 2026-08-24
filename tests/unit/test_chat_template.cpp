// Unit tests for §2.3 model-native chat template support.
//
// Covers:
//   1. LlamaCppBackend::FormatChatMessages — returns invalid when model absent.
//   2. LlamaCppBackend::ChatTemplateResult struct defaults.
//   3. FormatChatMessages with empty messages list.
//   4. FormatChatMessages with test_ready_ set but no real model_ pointer.
//
// Note: parsing of multi-format tool call outputs (DetectToolCall) lives inside
// http_server.cpp's anonymous namespace and is exercised by StubIntegration +
// SSECancel tests.  The logic is regression-tested here via a portable re-
// implementation of the format matchers that can be included without pulling in
// the full HTTP server translation unit.

#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace inferflux;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Portable re-implementation of the multi-format detector used in
// http_server.cpp.  Kept here to unit-test format variations independently of
// the HTTP layer.
// ---------------------------------------------------------------------------

struct ToolMatch {
  bool detected{false};
  std::string function_name;
  std::string arguments_json;
};

static bool fill_from_obj(const json &tc, ToolMatch &out) {
  if (!tc.contains("name") || !tc["name"].is_string())
    return false;
  out.function_name = tc["name"].get<std::string>();
  const char *args_key =
      tc.contains("arguments")
          ? "arguments"
          : (tc.contains("parameters") ? "parameters" : nullptr);
  if (args_key && tc.contains(args_key)) {
    out.arguments_json = tc[args_key].is_object()
                             ? tc[args_key].dump()
                             : tc[args_key].get<std::string>();
  } else {
    out.arguments_json = "{}";
  }
  out.detected = true;
  return true;
}

static ToolMatch detect_multi_format(const std::string &text) {
  ToolMatch m;

  // Format 1: {"tool_call":{"name":...}}
  auto try_inferflux = [&](const std::string &s) -> bool {
    try {
      auto j = json::parse(s);
      if (j.contains("tool_call") && j["tool_call"].is_object())
        return fill_from_obj(j["tool_call"], m);
    } catch (...) {
    }
    return false;
  };
  if (try_inferflux(text))
    return m;
  {
    auto pos = text.find("{\"tool_call\"");
    if (pos != std::string::npos && try_inferflux(text.substr(pos)))
      return m;
  }

  // Format 3: <tool_call>…</tool_call>
  {
    const std::string open = "<tool_call>";
    const std::string close = "</tool_call>";
    auto a = text.find(open);
    auto b = text.find(close);
    if (a != std::string::npos && b != std::string::npos && b > a) {
      auto inner = text.substr(a + open.size(), b - a - open.size());
      auto ws0 = inner.find_first_not_of(" \t\r\n");
      auto ws1 = inner.find_last_not_of(" \t\r\n");
      if (ws0 != std::string::npos)
        inner = inner.substr(ws0, ws1 - ws0 + 1);
      try {
        auto j = json::parse(inner);
        if (j.is_object() && fill_from_obj(j, m))
          return m;
      } catch (...) {
      }
    }
  }

  // Format 4: [TOOL_CALLS] [{"name":...}]  (Mistral)
  // Walk bracket depth to isolate the JSON array before any trailing
  // [/TOOL_CALLS] sentinel, which nlohmann::json rejects as trailing garbage.
  {
    const std::string tag = "[TOOL_CALLS]";
    auto pos = text.find(tag);
    if (pos != std::string::npos) {
      auto bracket = text.find('[', pos + tag.size());
      if (bracket != std::string::npos) {
        std::size_t depth = 0;
        std::size_t close = std::string::npos;
        for (std::size_t i = bracket; i < text.size(); ++i) {
          if (text[i] == '[')
            ++depth;
          else if (text[i] == ']') {
            if (--depth == 0) {
              close = i;
              break;
            }
          }
        }
        if (close != std::string::npos) {
          try {
            auto arr = json::parse(text.substr(bracket, close - bracket + 1));
            if (arr.is_array() && !arr.empty() && arr[0].is_object() &&
                fill_from_obj(arr[0], m))
              return m;
          } catch (...) {
          }
        }
      }
    }
  }

  // Format 2: {"name":"...","arguments":{...}}  (bare generic)
  {
    auto pos = text.find("{\"name\"");
    if (pos != std::string::npos) {
      try {
        auto j = json::parse(text.substr(pos));
        if (j.is_object() && fill_from_obj(j, m))
          return m;
      } catch (...) {
      }
    }
  }

  return m;
}

// ---------------------------------------------------------------------------
// LlamaCppBackend::FormatChatMessages tests
// ---------------------------------------------------------------------------

TEST_CASE("FormatChatMessages: returns invalid when no model loaded",
          "[chat_template]") {
  LlamaCppBackend backend;
  // model_ is null — must return valid=false without crashing.
  auto r = backend.FormatChatMessages({{"user", "Hello"}});
  REQUIRE_FALSE(r.valid);
  REQUIRE(r.prompt.empty());
}

TEST_CASE("FormatChatMessages: returns invalid for empty message list",
          "[chat_template]") {
  LlamaCppBackend backend;
  auto r = backend.FormatChatMessages({});
  REQUIRE_FALSE(r.valid);
}

TEST_CASE("FormatChatMessages: test_ready_ alone does not enable template "
          "(model_ is null)",
          "[chat_template]") {
  LlamaCppBackend backend;
  backend.ForceReadyForTests(); // sets test_ready_ = true but model_ stays null
  auto r = backend.FormatChatMessages({{"user", "Hi"}, {"assistant", "Hello"}});
  // FormatChatMessages checks model_ first; test_ready_ does not bypass it.
  REQUIRE_FALSE(r.valid);
}

TEST_CASE("ChatTemplateResult: default-constructed is invalid",
          "[chat_template]") {
  LlamaCppBackend::ChatTemplateResult r;
  REQUIRE_FALSE(r.valid);
  REQUIRE(r.prompt.empty());
}

// ---------------------------------------------------------------------------
// Multi-format tool call detection tests
// ---------------------------------------------------------------------------

TEST_CASE("DetectToolCall: plain text has no tool call", "[chat_template]") {
  auto m = detect_multi_format("The weather in London is sunny.");
  REQUIRE_FALSE(m.detected);
}

TEST_CASE("DetectToolCall: InferFlux preamble format detected",
          "[chat_template]") {
  std::string text =
      R"({"tool_call":{"name":"get_weather","arguments":{"location":"London"}}})";
  auto m = detect_multi_format(text);
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
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "search");
}

TEST_CASE("DetectToolCall: Hermes/Llama-3.1 XML tool_call format",
          "[chat_template]") {
  std::string text =
      R"(<tool_call>{"name":"get_time","arguments":{"tz":"UTC"}}</tool_call>)";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "get_time");
}

TEST_CASE("DetectToolCall: XML format with surrounding prose",
          "[chat_template]") {
  std::string text = "I will call a tool now.\n"
                     "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"id\":"
                     "42}}\n</tool_call>\n"
                     "Done.";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "lookup");
}

TEST_CASE("DetectToolCall: Mistral [TOOL_CALLS] format", "[chat_template]") {
  std::string text =
      R"([TOOL_CALLS] [{"name":"calculator","arguments":{"expr":"2+2"}}])";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "calculator");
}

TEST_CASE("DetectToolCall: parameters key accepted as alias for arguments",
          "[chat_template]") {
  std::string text =
      R"(<tool_call>{"name":"fn","parameters":{"x":1}}</tool_call>)";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "fn");
  REQUIRE(m.arguments_json == "{\"x\":1}");
}

TEST_CASE("DetectToolCall: bare generic JSON format", "[chat_template]") {
  std::string text = R"({"name":"ping","arguments":{}})";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "ping");
  REQUIRE(m.arguments_json == "{}");
}

TEST_CASE("DetectToolCall: malformed JSON does not crash", "[chat_template]") {
  REQUIRE_NOTHROW(detect_multi_format("{\"tool_call\": broken}"));
  REQUIRE_NOTHROW(detect_multi_format("<tool_call>bad</tool_call>"));
  REQUIRE_NOTHROW(detect_multi_format("[TOOL_CALLS] not-json"));
}

TEST_CASE("DetectToolCall: Mistral format with [/TOOL_CALLS] sentinel",
          "[chat_template]") {
  // Mistral-family models append [/TOOL_CALLS] after the JSON array.
  // The parser must strip it before calling json::parse or detection fails.
  std::string text =
      R"([TOOL_CALLS] [{"name":"weather","arguments":{"city":"Paris"}}][/TOOL_CALLS])";
  auto m = detect_multi_format(text);
  REQUIRE(m.detected);
  REQUIRE(m.function_name == "weather");
}

TEST_CASE("DetectToolCall: empty string returns no detection",
          "[chat_template]") {
  auto m = detect_multi_format("");
  REQUIRE_FALSE(m.detected);
}
