// Unit tests for the multi-format tool-call extraction and emission surface
// (server/http/completion_payload.h — real implementations linked from
// inferflux_core, not a ported copy).
//
// Covers:
//   1. DetectToolCalls: plain text → empty; all four accepted formats;
//      multi-call ordering; parameters alias; residual prose preservation.
//   2. BuildToolCallEntry: base shape; index only when engaged.
//   3. BuildToolCallStreamChunks: role frame, per-call index frames, exactly
//      one finish_reason="tool_calls" terminal frame.

#include <catch2/catch_amalgamated.hpp>

#include "server/http/completion_payload.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace inferflux;
using json = nlohmann::json;

namespace {

ToolCallResult make_call(const std::string &name,
                         const std::string &arguments_json) {
  ToolCallResult tc;
  tc.detected = true;
  tc.call_id = "call_" + name + "_0";
  tc.function_name = name;
  tc.arguments_json = arguments_json;
  return tc;
}

} // namespace

// ---------------------------------------------------------------------------
// DetectToolCalls
// ---------------------------------------------------------------------------

TEST_CASE("DetectToolCalls returns empty for plain text with residual",
          "[tool_calls]") {
  const std::string text = "Just an answer, no tools here.";
  auto ex = DetectToolCalls(text);
  REQUIRE(ex.calls.empty());
  REQUIRE(ex.remaining_text == text);
}

TEST_CASE("DetectToolCalls extracts InferFlux preamble convention",
          "[tool_calls]") {
  auto ex = DetectToolCalls(
      R"({"tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}})");
  REQUIRE(ex.calls.size() == 1);
  REQUIRE(ex.calls[0].function_name == "calculator");
  REQUIRE(json::parse(ex.calls[0].arguments_json) ==
          json{{"expression", "2+2"}});
  REQUIRE(ex.remaining_text.empty());
}

TEST_CASE("DetectToolCalls extracts bare OpenAI-style objects",
          "[tool_calls]") {
  auto ex = DetectToolCalls(R"({"name":"read","parameters":{"path":"a.py"}})");
  REQUIRE(ex.calls.size() == 1);
  REQUIRE(ex.calls[0].function_name == "read");
  REQUIRE(json::parse(ex.calls[0].arguments_json) == json{{"path", "a.py"}});
}

TEST_CASE("DetectToolCalls extracts Hermes/Qwen XML spans", "[tool_calls]") {
  auto ex =
      DetectToolCalls("Sure.\n<tool_call>{\"name\":\"write\",\"arguments\":{"
                      "\"path\":\"a.py\"}}</tool_call>\nDone.");
  REQUIRE(ex.calls.size() == 1);
  REQUIRE(ex.calls[0].function_name == "write");
  // Prose around the span survives as residual.
  REQUIRE(ex.remaining_text.find("Sure.") != std::string::npos);
  REQUIRE(ex.remaining_text.find("Done.") != std::string::npos);
}

TEST_CASE("DetectToolCalls extracts Mistral arrays", "[tool_calls]") {
  auto ex = DetectToolCalls("[TOOL_CALLS] "
                            "[{\"name\":\"a\",\"arguments\":{}},{\"name\":"
                            "\"b\",\"arguments\":{}}] tail");
  REQUIRE(ex.calls.size() == 2);
  REQUIRE(ex.calls[0].function_name == "a");
  REQUIRE(ex.calls[1].function_name == "b");
  REQUIRE(ex.remaining_text == "tail");
}

TEST_CASE("DetectToolCalls preserves multi-call ordering and per-call ids",
          "[tool_calls]") {
  const std::string text =
      "{\"tool_call\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"x\"}}}"
      "\n"
      "{\"tool_call\":{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}}";
  auto ex = DetectToolCalls(text);
  REQUIRE(ex.calls.size() == 2);
  REQUIRE(ex.calls[0].function_name == "write_file");
  REQUIRE(ex.calls[0].call_id == "call_write_file_0");
  REQUIRE(ex.calls[1].function_name == "shell");
  REQUIRE(ex.calls[1].call_id == "call_shell_1");
  REQUIRE(ex.remaining_text.empty());
}

TEST_CASE("DetectToolCalls keeps malformed JSON in residual without crashing",
          "[tool_calls]") {
  const std::string text = "{\"tool_call\":{\"name\":";
  auto ex = DetectToolCalls(text);
  REQUIRE(ex.calls.empty());
  REQUIRE(ex.remaining_text == text);
}

// ---------------------------------------------------------------------------
// BuildToolCallEntry
// ---------------------------------------------------------------------------

TEST_CASE("BuildToolCallEntry shape without index", "[tool_calls]") {
  auto entry = BuildToolCallEntry(make_call("calc", "{\"x\":1}"), std::nullopt);
  REQUIRE(entry["id"] == "call_calc_0");
  REQUIRE(entry["type"] == "function");
  REQUIRE(entry["function"]["name"] == "calc");
  REQUIRE(entry["function"]["arguments"] == "{\"x\":1}");
  REQUIRE(entry.find("index") == entry.end()); // not engaged
}

TEST_CASE("BuildToolCallEntry carries index when engaged", "[tool_calls]") {
  auto entry = BuildToolCallEntry(make_call("calc", "{}"), 2);
  REQUIRE(entry["index"] == 2);
}

// ---------------------------------------------------------------------------
// BuildToolCallStreamChunks
// ---------------------------------------------------------------------------

TEST_CASE("BuildToolCallStreamChunks emits per-call indices and one finish",
          "[tool_calls]") {
  std::vector<ToolCallResult> calls;
  calls.push_back(make_call("write_file", "{\"path\":\"a.py\"}"));
  calls.push_back(make_call("shell", "{\"cmd\":\"pytest\"}"));

  const std::string payload =
      BuildToolCallStreamChunks("chatcmpl-test", "m", 1234567890, calls);

  std::vector<json> frames;
  size_t pos = 0;
  while (pos < payload.size()) {
    const auto eol = payload.find('\n', pos);
    std::string line = payload.substr(pos, eol - pos);
    if (line.rfind("data: ", 0) == 0 && line != "data: [DONE]") {
      frames.push_back(json::parse(line.substr(6)));
    }
    pos = eol + 1;
  }

  REQUIRE(frames.size() == 1 + 2 * calls.size() + 1);

  // First frame: role delta, content null.
  REQUIRE(frames[0]["choices"][0]["delta"]["role"] == "assistant");
  REQUIRE(frames[0]["choices"][0]["delta"]["content"] == nullptr);

  // Per-call frames carry the call index; name frame first, args frame second.
  size_t frame = 1;
  for (size_t k = 0; k < calls.size(); ++k) {
    const auto &name_frame = frames[frame];
    REQUIRE(name_frame["choices"][0]["delta"]["tool_calls"][0]["index"] ==
            static_cast<int>(k));
    REQUIRE(name_frame["choices"][0]["delta"]["tool_calls"][0]["function"]
                      ["name"] == calls[k].function_name);
    REQUIRE(name_frame["choices"][0]["delta"]["tool_calls"][0]["function"]
                      ["arguments"] == "");
    ++frame;
    const auto &args_frame = frames[frame];
    REQUIRE(args_frame["choices"][0]["delta"]["tool_calls"][0]["index"] ==
            static_cast<int>(k));
    REQUIRE(args_frame["choices"][0]["delta"]["tool_calls"][0]["function"]
                      ["arguments"] == calls[k].arguments_json);
    ++frame;
  }

  // Terminal frame: finish_reason tool_calls, exactly once, last.
  const auto &finish = frames.back();
  REQUIRE(finish["choices"][0]["finish_reason"] == "tool_calls");
  size_t finish_frames = 0;
  for (const auto &f : frames) {
    if (f["choices"][0]["finish_reason"] == "tool_calls") {
      ++finish_frames;
    }
  }
  REQUIRE(finish_frames == 1);
}
