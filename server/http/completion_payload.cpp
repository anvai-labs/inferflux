#include "server/http/completion_payload.h"

#include "server/logging/logger.h"

#include <ctime>
#include <optional>
#include <string_view>

namespace inferflux {

using json = nlohmann::json;

std::string SerializeJsonUtf8Safe(const nlohmann::json &payload) {
  return payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// ---------------------------------------------------------------------------
// Tool-call extraction + emission (moved from http_server.cpp so the test
// target links the real implementations; see header declarations).
// ---------------------------------------------------------------------------

void LogJsonParseFailure(const char *context, const std::exception &ex) {
  log::Debug("http_server", std::string(context) + ": " + ex.what());
}

// Leading/trailing-whitespace trim for the visible text left over after
// detected tool calls are removed from a completion.
static std::string TrimWs(const std::string &sv) {
  const auto a = sv.find_first_not_of(" \t\r\n");
  if (a == std::string::npos)
    return {};
  const auto b = sv.find_last_not_of(" \t\r\n");
  return sv.substr(a, b - a + 1);
}

// Index one past the '}' matching the '{' at `start`, tracking JSON string
// literals (escapes included) so braces inside argument payloads don't
// misnest. std::string::npos when unbalanced.
static std::size_t ScanBalancedObject(const std::string &sv,
                                      std::size_t start) {
  int depth = 0;
  bool in_str = false;
  bool esc = false;
  const char open = sv[start];
  const char close = open == '[' ? ']' : '}';
  for (std::size_t k = start; k < sv.size(); ++k) {
    const char c = sv[k];
    if (in_str) {
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == '"') {
        in_str = false;
      }
      continue;
    }
    if (c == '"') {
      in_str = true;
    } else if (c == open) {
      ++depth;
    } else if (c == close) {
      if (--depth == 0)
        return k + 1;
      if (depth < 0)
        return std::string::npos;
    }
  }
  return std::string::npos;
}

// Extract EVERY tool call in `text`. Accepted shapes per call:
//   {"tool_call":{"name":...,"arguments":...}}   (InferFlux convention)
//   {"name":"...","arguments"|"parameters":...}  (OpenAI-style bare object)
//   <tool_call>{"name":...}</tool_call>          (Hermes / Qwen XML), and
//   [TOOL_CALLS] [{...},{...}]                   (Mistral)
// Models routinely emit several calls in one completion (one per line when
// chaining, e.g. write-then-run), which a single-object parse rejects as
// "Extra data". Non-tool prose survives as extraction.remaining_text.
ToolCallExtraction DetectToolCalls(const std::string &text) {
  ToolCallExtraction extraction;
  auto &calls = extraction.calls;
  auto fill = [&calls](const json &obj, bool explicit_call) -> bool {
    json tc = obj;
    if (tc.contains("tool_call") && tc["tool_call"].is_object()) {
      tc = tc["tool_call"];
      explicit_call = true;
    }
    if (!tc.is_object() || !tc.contains("name") || !tc["name"].is_string() ||
        tc["name"].get<std::string>().empty())
      return false;
    const char *args_key =
        tc.contains("arguments")
            ? "arguments"
            : (tc.contains("parameters") ? "parameters" : nullptr);
    // A generic {"name":"Alice"} object is not an implicit tool call.
    if (!explicit_call && !args_key)
      return false;
    ToolCallResult r;
    r.function_name = tc["name"].get<std::string>();
    r.call_id = "call_" + r.function_name + "_" + std::to_string(calls.size());
    if (args_key) {
      if (!tc[args_key].is_object() && !tc[args_key].is_string())
        return false;
      r.arguments_json = tc[args_key].is_object()
                             ? tc[args_key].dump()
                             : tc[args_key].get<std::string>();
    } else {
      r.arguments_json = "{}";
    }
    r.detected = true;
    calls.push_back(std::move(r));
    return true;
  };

  // Parse each envelope atomically: invalid array entries must not leave calls
  // behind while retaining the original envelope in the visible content.
  auto parse = [&](std::size_t begin, std::size_t end, bool explicit_call) {
    const auto before = calls.size();
    bool matched = false;
    try {
      const auto value = json::parse(text.substr(begin, end - begin));
      if (value.is_object()) {
        matched = fill(value, explicit_call);
      } else if (explicit_call && value.is_array() && !value.empty()) {
        matched = true;
        for (const auto &item : value) {
          if (!fill(item, true)) {
            matched = false;
            break;
          }
        }
      }
    } catch (const json::exception &ex) {
      matched = false;
      LogJsonParseFailure("DetectToolCalls", ex);
    }
    if (!matched)
      calls.resize(before);
    return matched;
  };

  // One left-to-right pass preserves ordering across mixed formats. A balanced
  // ordinary JSON object/array is consumed intact, never rescanned for nested
  // "name" fields that happen to look like tool calls.
  const std::string open = "<tool_call>", close = "</tool_call>";
  const std::string mistral = "[TOOL_CALLS]";
  std::string residual;
  for (std::size_t pos = 0; pos < text.size();) {
    std::size_t body = pos, end = std::string::npos, consumed = pos;
    bool explicit_call = false;
    if (text.compare(pos, open.size(), open) == 0) {
      body = pos + open.size();
      end = text.find(close, body);
      if (end == std::string::npos) {
        residual.append(text, pos, std::string::npos);
        break;
      }
      consumed = end + close.size();
      explicit_call = true;
    } else if (text.compare(pos, mistral.size(), mistral) == 0) {
      body = text.find_first_not_of(" \t\r\n", pos + mistral.size());
      if (body != std::string::npos && text[body] == '[') {
        end = ScanBalancedObject(text, body);
        consumed = end;
        explicit_call = true;
        if (end != std::string::npos) {
          const auto suffix = text.find_first_not_of(" \t\r\n", end);
          if (suffix != std::string::npos &&
              text.compare(suffix, 13, "[/TOOL_CALLS]") == 0)
            consumed = suffix + 13;
        }
      }
    } else if (text[pos] == '{' || text[pos] == '[') {
      end = ScanBalancedObject(text, pos);
      consumed = end;
      if (end == std::string::npos) {
        residual.append(text, pos, std::string::npos);
        break;
      }
    }
    if (end != std::string::npos) {
      if (!parse(body, end, explicit_call))
        residual.append(text, pos, consumed - pos);
      pos = consumed;
    } else {
      residual += text[pos++];
    }
  }
  extraction.remaining_text = TrimWs(residual);
  return extraction;
}
json BuildToolCallEntry(const ToolCallResult &tc, std::optional<int> index) {
  json entry = {
      {"id", tc.call_id},
      {"type", "function"},
      {"function",
       {{"name", tc.function_name}, {"arguments", tc.arguments_json}}}};
  if (index.has_value()) {
    entry["index"] = *index;
  }
  return entry;
}

std::string BuildToolCallStreamChunks(
    const std::string &id, std::string_view model, std::time_t ts,
    const std::vector<ToolCallResult> &tool_calls, std::string_view reasoning,
    std::string_view content) {
  std::string out;
  out.reserve(4096 * (tool_calls.size() + 1));
  auto base = [&]() -> json {
    json j;
    j["id"] = id;
    j["object"] = "chat.completion.chunk";
    j["created"] = ts;
    j["model"] = model;
    return j;
  };

  // Chunk 1: role=assistant, content=null (once for the message).
  {
    json j = base();
    j["choices"] =
        json::array({{{"index", 0},
                      {"delta", {{"role", "assistant"}, {"content", nullptr}}},
                      {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Buffered tool calls have not emitted any tokens yet. Preserve separated
  // reasoning and residual prose before the structured calls and finish frame.
  for (const auto &[key, value] :
       {std::pair<std::string_view, std::string_view>{"reasoning_content",
                                                      reasoning},
        {"content", content}}) {
    if (!value.empty()) {
      json j = base();
      j["choices"] =
          json::array({{{"index", 0},
                        {"delta", {{std::string(key), std::string(value)}}},
                        {"finish_reason", nullptr}}});
      out += "data: " + SerializeJsonUtf8Safe(j) + "\n\n";
    }
  }

  // Per call: id/type/name frame with empty arguments, then the arguments
  // frame - tool_call `index` distinguishes parallel calls per the OpenAI
  // streaming shape.
  for (std::size_t k = 0; k < tool_calls.size(); ++k) {
    const ToolCallResult &tc = tool_calls[k];
    const int call_index = static_cast<int>(k);
    {
      json j = base();
      json tc_delta = json::array(
          {{{"index", call_index},
            {"id", tc.call_id},
            {"type", "function"},
            {"function", {{"name", tc.function_name}, {"arguments", ""}}}}});
      j["choices"] = json::array({{{"index", 0},
                                   {"delta", {{"tool_calls", tc_delta}}},
                                   {"finish_reason", nullptr}}});
      out += "data: " + j.dump() + "\n\n";
    }

    if (!tc.arguments_json.empty()) {
      json j = base();
      json arg_delta =
          json::array({{{"index", call_index},
                        {"function", {{"arguments", tc.arguments_json}}}}});
      j["choices"] = json::array({{{"index", 0},
                                   {"delta", {{"tool_calls", arg_delta}}},
                                   {"finish_reason", nullptr}}});
      out += "data: " + j.dump() + "\n\n";
    }
  }

  // Final frame: finish_reason=tool_calls (once for the message).
  {
    json j = base();
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", json::object()},
                                 {"finish_reason", "tool_calls"}}});
    out += "data: " + j.dump() + "\n\n";
  }

  return out;
}

} // namespace inferflux
