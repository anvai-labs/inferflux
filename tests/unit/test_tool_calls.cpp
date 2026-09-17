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

#include "runtime/text/response_splitter.h"
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

TEST_CASE("DetectToolCalls preserves order across mixed call formats",
          "[tool_calls]") {
  const std::string text =
      R"({"tool_call":{"name":"write_file","arguments":{}}} <tool_call>{"name":"shell","arguments":{}}</tool_call> [TOOL_CALLS] [{"name":"read","arguments":{}}])";
  const auto result = DetectToolCalls(text);
  REQUIRE(result.calls.size() == 3);
  REQUIRE(result.calls[0].function_name == "write_file");
  REQUIRE(result.calls[1].function_name == "shell");
  REQUIRE(result.calls[2].function_name == "read");
  REQUIRE(result.calls[0].call_id == "call_write_file_0");
  REQUIRE(result.calls[1].call_id == "call_shell_1");
  REQUIRE(result.remaining_text.empty());
}

TEST_CASE("DetectToolCalls preserves ordinary nested JSON atomically",
          "[tool_calls]") {
  for (const std::string text :
       {R"(Report: {"profile":{"name":"Alice"}})",
        R"(Report: {"profile":{"name":"shell","arguments":{}}})",
        R"(Report: [{"name":"shell","arguments":{}}])",
        R"({"name":"Alice"})"}) {
    const auto result = DetectToolCalls(text);
    REQUIRE(result.calls.empty());
    REQUIRE(result.remaining_text == text);
  }
}

TEST_CASE(
    "DetectToolCalls rejects partial invalid arrays without duplicate calls",
    "[tool_calls]") {
  for (
      const std::string text :
      {R"(<tool_call>[{"name":"write_file","arguments":{}},{"name":"shell","arguments":7}]</tool_call>)",
       R"([TOOL_CALLS] [{"name":"write_file","arguments":{}},{"not_a_call":true}])"}) {
    const auto result = DetectToolCalls(text);
    REQUIRE(result.calls.empty());
    REQUIRE(result.remaining_text == text);
  }
}

TEST_CASE(
    "Buffered tool stream preserves separated reasoning and residual prose",
    "[tool_calls][reasoning]") {
  const auto split = ResponseSplitter::Split(
      ChatTemplateFamily::kChatML,
      R"(<think>plan</think>Before <tool_call>{"name":"read","arguments":{}}</tool_call> After)");
  const auto extraction = DetectToolCalls(split.content);
  REQUIRE(extraction.calls.size() == 1);
  const auto stream =
      BuildToolCallStreamChunks("id", "m", 1, extraction.calls, split.reasoning,
                                extraction.remaining_text);
  std::string reasoning, content;
  int finish = 0, tool_calls = 0;
  for (std::size_t pos = 0; pos < stream.size();) {
    const auto end = stream.find("\n\n", pos);
    REQUIRE(end != std::string::npos);
    const auto frame = json::parse(stream.substr(pos + 6, end - pos - 6));
    const auto &choice = frame["choices"][0];
    const auto &delta = choice["delta"];
    reasoning += delta.value("reasoning_content", std::string{});
    if (delta.contains("content") && delta["content"].is_string())
      content += delta["content"].get<std::string>();
    if (delta.contains("tool_calls"))
      ++tool_calls;
    if (!choice["finish_reason"].is_null()) {
      REQUIRE(choice["finish_reason"] == "tool_calls");
      ++finish;
    }
    pos = end + 2;
  }
  REQUIRE(reasoning == "plan");
  REQUIRE(content == "Before  After");
  REQUIRE(tool_calls == 2);
  REQUIRE(finish == 1);
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
