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
    } else if (c == '{') {
      ++depth;
    } else if (c == '}') {
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
  auto fill = [&calls](const json &obj) -> bool {
    json tc = obj;
    if (tc.contains("tool_call") && tc["tool_call"].is_object())
      tc = tc["tool_call"];
    if (!tc.is_object() || !tc.contains("name") || !tc["name"].is_string())
      return false;
    ToolCallResult r;
    r.function_name = tc["name"].get<std::string>();
    r.call_id = "call_" + r.function_name + "_" + std::to_string(calls.size());
    const char *args_key =
        tc.contains("arguments")
            ? "arguments"
            : (tc.contains("parameters") ? "parameters" : nullptr);
    if (args_key && tc.contains(args_key)) {
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

  // 1) <tool_call>...</tool_call> spans - all of them, arrays supported.
  static const std::string kOpenTag = "<tool_call>";
  static const std::string kCloseTag = "</tool_call>";
  std::string residual;
  residual.reserve(text.size());
  {
    std::size_t pos = 0;
    while (true) {
      const auto a = text.find(kOpenTag, pos);
      if (a == std::string::npos)
        break;
      const auto b = text.find(kCloseTag, a + kOpenTag.size());
      if (b == std::string::npos)
        break;
      residual.append(text, pos, a - pos);
      const std::string inner =
          TrimWs(text.substr(a + kOpenTag.size(), b - a - kOpenTag.size()));
      bool matched = false;
      try {
        const auto j = json::parse(inner);
        if (j.is_object()) {
          matched = fill(j);
        } else if (j.is_array()) {
          for (const auto &el : j)
            if (el.is_object())
              matched = fill(el) || matched;
        }
      } catch (const json::exception &ex) {
        LogJsonParseFailure("DetectToolCalls.tool_call_tag", ex);
      }
      if (!matched)
        residual.append(text, a, b + kCloseTag.size() - a);
      pos = b + kCloseTag.size();
    }
    residual.append(text, pos, text.size() - pos);
  }

  // 2) [TOOL_CALLS] [{...},...] (Mistral) - walk bracket depth for the
  //    matching ']' so trailing prose doesn't break the parse.
  {
    static const std::string kMistral = "[TOOL_CALLS]";
    for (;;) {
      const auto tag = residual.find(kMistral);
      if (tag == std::string::npos)
        break;
      const auto bracket = residual.find('[', tag + kMistral.size());
      if (bracket == std::string::npos)
        break;
      int depth = 0;
      bool in_str = false;
      bool esc = false;
      std::size_t close = std::string::npos;
      for (std::size_t k = bracket; k < residual.size(); ++k) {
        const char c = residual[k];
        if (in_str) {
          if (esc)
            esc = false;
          else if (c == '\\')
            esc = true;
          else if (c == '"')
            in_str = false;
          continue;
        }
        if (c == '"')
          in_str = true;
        else if (c == '[')
          ++depth;
        else if (c == ']') {
          if (--depth == 0) {
            close = k;
            break;
          }
        }
      }
      if (close == std::string::npos)
        break;
      bool matched = false;
      try {
        const auto arr =
            json::parse(residual.substr(bracket, close - bracket + 1));
        if (arr.is_array()) {
          for (const auto &el : arr)
            if (el.is_object())
              matched = fill(el) || matched;
        }
      } catch (const json::exception &ex) {
        LogJsonParseFailure("DetectToolCalls.mistral", ex);
      }
      if (matched) {
        residual.erase(tag, close + 1 - tag);
      } else {
        // Not a usable array; keep scanning after this occurrence.
        const std::string after = residual.substr(close + 1);
        residual = residual.substr(0, tag + kMistral.size()) + after;
        break;
      }
    }
  }

  // 3) Bare JSON objects - one or many, brace-balanced so adjacent calls
  //    or surrounding prose never yield "Extra data".
  {
    std::string out;
    out.reserve(residual.size());
    std::size_t i = 0;
    while (i < residual.size()) {
      if (residual[i] == '{') {
        const auto end = ScanBalancedObject(residual, i);
        if (end != std::string::npos) {
          const std::string cand = residual.substr(i, end - i);
          if (cand.find("\"tool_call\"") != std::string::npos ||
              cand.find("\"name\"") != std::string::npos) {
            bool filled = false;
            try {
              filled = fill(json::parse(cand));
            } catch (const json::exception &ex) {
              LogJsonParseFailure("DetectToolCalls.bare_object", ex);
            }
            if (filled) {
              i = end;
              continue;
            }
          }
        }
      }
      out += residual[i++];
    }
    residual = std::move(out);
  }

  // Strip tool-call scaffolding the phases did not consume (e.g. a
  // [/TOOL_CALLS] sentinel or unmatched tags around a rescued call) so it
  // never leaks into visible content.
  if (!calls.empty()) {
    const std::string scaffolding[] = {kOpenTag, kCloseTag, "[TOOL_CALLS]",
                                       "[/TOOL_CALLS]"};
    for (const auto &tag : scaffolding) {
      std::size_t at;
      while ((at = residual.find(tag)) != std::string::npos) {
        residual.erase(at, tag.size());
      }
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

std::string
BuildToolCallStreamChunks(const std::string &id, std::string_view model,
                          std::time_t ts,
                          const std::vector<ToolCallResult> &tool_calls) {
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
