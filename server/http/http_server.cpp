#include "server/http/http_server.h"

#include "server/http/completion_payload.h"
#include "server/http/model_json.h"

#include "model/model_format.h"
#include "runtime/backends/backend_capabilities.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/multimodal/image_preprocessor.h"
#include "runtime/string_utils.h"
#include "scheduler/model_selection.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"
#include "server/tracing/span.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define SHUT_RDWR SD_BOTH
inline int inferflux_close_socket(int fd) { return ::closesocket(fd); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
inline int inferflux_close_socket(int fd) { return ::close(fd); }
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openssl/err.h>
#include <openssl/ssl.h>

using json = nlohmann::json;

namespace inferflux {

namespace {

int ParseNonNegativeEnvInt(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (!raw || *raw == '\0') {
    return default_value;
  }
  try {
    return std::max(0, std::stoi(raw));
  } catch (const std::exception &) {
    inferflux::log::Warn("http", std::string("Invalid value for ") + name +
                                     ": " + raw + "; using default " +
                                     std::to_string(default_value));
    return default_value;
  }
}

// ParseBoolEnv is provided by runtime/string_utils.h (inferflux::ParseBoolEnv).
// GetHeaderValue is provided by server/http/http_utils.h.

std::string PoolRoleToString(HttpServer::PoolRole role) {
  switch (role) {
  case HttpServer::PoolRole::kPrefill:
    return "prefill";
  case HttpServer::PoolRole::kDecode:
    return "decode";
  case HttpServer::PoolRole::kUnified:
  default:
    return "unified";
  }
}

json OptionalIntToJson(const std::optional<int64_t> &value) {
  return value.has_value() ? json(*value) : json(nullptr);
}

} // namespace

std::string BuildResponse(const std::string &body, int status,
                          std::string_view status_text,
                          const std::string &extra_headers) {
  std::string headers = "HTTP/1.1 " + std::to_string(status) + " " +
                        std::string(status_text) + "\r\n";
  headers += "Content-Type: application/json\r\n";
  headers += "Access-Control-Allow-Origin: *\r\n";
  headers += "Connection: keep-alive\r\n";
  if (!extra_headers.empty()) {
    headers += extra_headers;
  }
  headers += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
  return headers + body;
}

// ChatMessage, Tool, ToolCallResult, CompletionRequestPayload are defined
// in server/http/completion_payload.h.

namespace {

// kKiB/kMiB/kGiB are provided by runtime/string_utils.h (inferflux namespace).

void LogJsonParseFailure(const char *context, const std::exception &ex) {
  log::Debug("http_server", std::string(context) + ": " + ex.what());
}

constexpr std::size_t kMaxResponseFormatBytes =
    16ULL * kKiB; // 16 KB cap for schemas/grammars.

} // namespace

// CompletionRequestPayload is defined in server/http/completion_payload.h.

// Model JSON builders (BuildCapabilitiesJson, BuildModelIdentityJson, etc.)
// are defined in server/http/model_json.h/.cpp.

std::string BuildModelNotFoundResponse() {
  return BuildResponse(json({{"error", "model_not_found"}}).dump(), 404,
                       "Not Found");
}

bool HasPrefix(const std::string &value, const std::string &prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

std::string StripPrefix(const std::string &value, const std::string &prefix) {
  if (!HasPrefix(value, prefix)) {
    return value;
  }
  std::string out = value.substr(prefix.size());
  out.erase(out.begin(),
            std::find_if(out.begin(), out.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
  return out;
}

std::string ParseJsonStringField(const std::string &body,
                                 std::string_view field) {
  try {
    const std::string field_name(field);
    auto j = json::parse(body);
    if (j.contains(field_name) && j[field_name].is_string()) {
      return j[field_name].get<std::string>();
    }
  } catch (const json::exception &ex) {
    LogJsonParseFailure("ParseJsonStringField", ex);
  }
  return "";
}

const ModelInfo *FindModelById(const std::vector<ModelInfo> &models,
                               const std::string &model_id) {
  auto it = std::find_if(
      models.begin(), models.end(),
      [&](const ModelInfo &candidate) { return candidate.id == model_id; });
  if (it == models.end()) {
    return nullptr;
  }
  return &(*it);
}

BackendFeatureRequirements
BuildGenerationRequirements(const CompletionRequestPayload &payload,
                            bool speculative_enabled) {
  return BuildGenerationFeatureRequirements(
      payload.stream, payload.logprobs, payload.has_response_format,
      payload.has_images, speculative_enabled && !payload.has_response_format);
}

bool IsDefaultModelAlias(const std::string &model) {
  std::string normalized = model;
  std::transform(
      normalized.begin(), normalized.end(), normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized == "default";
}

CompletionRequestPayload ParseJsonPayload(const std::string &body) {
  CompletionRequestPayload payload;
  if (body.empty()) {
    return payload;
  }
  try {
    auto j = json::parse(body);
    if (j.contains("prompt") && j["prompt"].is_string()) {
      payload.prompt = j["prompt"].get<std::string>();
    }
    if (j.contains("model") && j["model"].is_string()) {
      payload.model = j["model"].get<std::string>();
    }
    if (j.contains("session_id") && j["session_id"].is_string()) {
      payload.session_id = j["session_id"].get<std::string>();
    }
    if (j.contains("client_request_id") && j["client_request_id"].is_string()) {
      payload.client_request_id = j["client_request_id"].get<std::string>();
    }
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
      payload.max_tokens = j["max_tokens"].get<int>();
    }
    // OpenAI newer API alias for max_tokens.
    if (j.contains("max_completion_tokens") &&
        j["max_completion_tokens"].is_number_integer()) {
      payload.max_tokens = j["max_completion_tokens"].get<int>();
    }
    if (j.contains("stream") && j["stream"].is_boolean()) {
      payload.stream = j["stream"].get<bool>();
    }
    // OpenAI logprobs:
    //  Chat: logprobs (bool) + top_logprobs (int, 1-20)
    //  Completions: logprobs (int, 0-5) — treat as top_logprobs count
    if (j.contains("logprobs")) {
      auto &lp = j["logprobs"];
      if (lp.is_boolean()) {
        payload.logprobs = lp.get<bool>();
      } else if (lp.is_number_integer()) {
        int n = lp.get<int>();
        if (n > 0) {
          payload.logprobs = true;
          payload.top_logprobs = std::min(n, 20);
        }
      }
    }
    if (j.contains("top_logprobs") && j["top_logprobs"].is_number_integer()) {
      int n = j["top_logprobs"].get<int>();
      payload.top_logprobs = std::min(std::max(n, 0), 20);
      if (payload.top_logprobs > 0) {
        payload.logprobs = true;
      }
    }
    if (j.contains("temperature") && j["temperature"].is_number()) {
      payload.temperature = j["temperature"].get<float>();
    }
    if (j.contains("top_p") && j["top_p"].is_number()) {
      payload.top_p = j["top_p"].get<float>();
    }
    if (j.contains("top_k") && j["top_k"].is_number_integer()) {
      payload.top_k = j["top_k"].get<int>();
    }
    if (j.contains("min_p") && j["min_p"].is_number()) {
      payload.min_p = j["min_p"].get<float>();
    }
    if (j.contains("frequency_penalty") && j["frequency_penalty"].is_number()) {
      payload.frequency_penalty = j["frequency_penalty"].get<float>();
    }
    if (j.contains("presence_penalty") && j["presence_penalty"].is_number()) {
      payload.presence_penalty = j["presence_penalty"].get<float>();
    }
    if (j.contains("repetition_penalty") &&
        j["repetition_penalty"].is_number()) {
      payload.repetition_penalty = j["repetition_penalty"].get<float>();
    }
    if (j.contains("seed") && j["seed"].is_number_integer()) {
      payload.seed = j["seed"].get<uint32_t>();
    }
    // OpenAI `logit_bias` parameter: map token IDs to bias values.
    // Bias values should be between -100 and 100.
    if (j.contains("logit_bias") && j["logit_bias"].is_object()) {
      for (auto &[key, value] : j["logit_bias"].items()) {
        if (value.is_number()) {
          int token_id = std::stoi(key);
          float bias = value.get<float>();
          // Clamp bias to OpenAI's allowed range [-100, 100]
          bias = std::max(-100.0f, std::min(100.0f, bias));
          payload.logit_bias[token_id] = bias;
        }
      }
    }
    // OpenAI-compatible: response_format controls structured output.
    if (j.contains("response_format") && j["response_format"].is_object()) {
      auto &rf = j["response_format"];
      if (rf.contains("type") && rf["type"].is_string()) {
        payload.has_response_format = true;
        payload.response_format_type = rf["type"].get<std::string>();
        auto type = payload.response_format_type;

        if (type == "json_object" || type == "json_schema") {
          payload.json_mode = true;
          if (rf.contains("schema")) {
            payload.response_format_schema = rf["schema"].dump();
          }
          if (rf.contains("json_schema")) {
            payload.response_format_schema = rf["json_schema"].dump();
          }
          if (payload.response_format_schema.size() > kMaxResponseFormatBytes) {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format schema exceeds 16KB limit";
          }
          if (type == "json_schema" && payload.response_format_schema.empty()) {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format json_schema requires a schema definition";
          }
        } else if (type == "grammar") {
          if (rf.contains("grammar") && rf["grammar"].is_string()) {
            payload.response_format_grammar = rf["grammar"].get<std::string>();
            if (payload.response_format_grammar.size() >
                kMaxResponseFormatBytes) {
              payload.response_format_ok = false;
              payload.response_format_error =
                  "response_format grammar exceeds 16KB limit";
            }
          } else {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format grammar missing 'grammar' string";
          }
          if (rf.contains("root") && rf["root"].is_string()) {
            payload.response_format_root = rf["root"].get<std::string>();
          }
        } else if (type == "text" || type.empty()) {
          payload.has_response_format = false; // treat as default behavior.
        } else {
          payload.response_format_ok = false;
          payload.response_format_error =
              "Unsupported response_format.type '" + type + "'";
        }
      } else {
        payload.response_format_ok = false;
        payload.response_format_error = "response_format.type must be a string";
      }
    }
    if (j.contains("messages") && j["messages"].is_array()) {
      for (const auto &msg : j["messages"]) {
        ChatMessage m;
        if (msg.contains("role") && msg["role"].is_string()) {
          m.role = msg["role"].get<std::string>();
        }
        if (msg.contains("content")) {
          if (msg["content"].is_string()) {
            m.content = msg["content"].get<std::string>();
          } else if (msg["content"].is_array()) {
            // §2.2: OpenAI multipart content — extract text and image_url
            // parts.
            auto mm =
                ImagePreprocessor::ProcessContentArray(msg["content"].dump());
            m.content = mm.text;
            if (!mm.images.empty()) {
              payload.has_images = true;
              for (auto &img : mm.images) {
                payload.images.push_back(std::move(img));
              }
            }
          }
        }
        if (!m.role.empty() || !m.content.empty()) {
          payload.messages.push_back(std::move(m));
        }
      }
    }
    // §2.3: parse tool definitions.
    if (j.contains("tools") && j["tools"].is_array()) {
      payload.has_tool_schema = true;
      for (const auto &t : j["tools"]) {
        if (!t.is_object())
          continue;
        Tool tool;
        if (t.contains("type") && t["type"].is_string()) {
          tool.type = t["type"].get<std::string>();
        }
        if (t.contains("function") && t["function"].is_object()) {
          const auto &f = t["function"];
          if (f.contains("name") && f["name"].is_string()) {
            tool.function.name = f["name"].get<std::string>();
          }
          if (f.contains("description") && f["description"].is_string()) {
            tool.function.description = f["description"].get<std::string>();
          }
          if (f.contains("parameters")) {
            tool.function.parameters = f["parameters"];
          }
        }
        if (!tool.function.name.empty()) {
          if (payload.first_tool_name.empty()) {
            payload.first_tool_name = tool.function.name;
          }
          payload.tools.push_back(std::move(tool));
        }
      }
    }
    if (j.contains("tool_choice")) {
      if (j["tool_choice"].is_string()) {
        payload.tool_choice = j["tool_choice"].get<std::string>();
      } else if (j["tool_choice"].is_object()) {
        // {"type":"function","function":{"name":"..."}} — required + specific
        // fn.
        payload.tool_choice = "required";
        const auto &tc = j["tool_choice"];
        if (tc.contains("function") && tc["function"].is_object() &&
            tc["function"].contains("name") &&
            tc["function"]["name"].is_string()) {
          payload.tool_choice_function =
              tc["function"]["name"].get<std::string>();
        }
      }
    }
    // `stop`: string or array of strings (up to 4).
    if (j.contains("stop")) {
      if (j["stop"].is_string()) {
        payload.stop.push_back(j["stop"].get<std::string>());
      } else if (j["stop"].is_array()) {
        for (const auto &s : j["stop"]) {
          if (s.is_string() && !s.get<std::string>().empty()) {
            payload.stop.push_back(s.get<std::string>());
            if (payload.stop.size() >= 4)
              break;
          }
        }
      }
    }
    // stream_options.include_usage: emit token counts in a final SSE chunk.
    if (j.contains("stream_options") && j["stream_options"].is_object()) {
      const auto &so = j["stream_options"];
      if (so.contains("include_usage") && so["include_usage"].is_boolean()) {
        payload.stream_include_usage = so["include_usage"].get<bool>();
      }
    }
    // `n`: number of completions to return (1-10).
    if (j.contains("n") && j["n"].is_number_integer()) {
      payload.n = std::min(std::max(j["n"].get<int>(), 1), 10);
    }
    // `best_of`: generate this many completions, return top n (best_of >= n).
    if (j.contains("best_of") && j["best_of"].is_number_integer()) {
      payload.best_of = std::min(std::max(j["best_of"].get<int>(), 1), 20);
    }
    // Enforce best_of >= n (clamp up rather than error at parse time).
    if (payload.best_of < payload.n) {
      payload.best_of = payload.n;
    }
  } catch (const json::exception &ex) {
    LogJsonParseFailure("ParseJsonPayload", ex);
    // Return defaults on parse failure.
  }
  return payload;
}

std::string FlattenMessages(const std::vector<ChatMessage> &messages) {
  std::string prompt;
  for (const auto &message : messages) {
    if (!prompt.empty()) {
      prompt.append("\n");
    }
    std::string role = message.role.empty() ? "user" : message.role;
    prompt.append(role);
    prompt.append(": ");
    prompt.append(message.content);
  }
  return prompt;
}

// §2.3: builds a system-level preamble that teaches the model the tool-call
// protocol.
std::string BuildToolSystemPrompt(const std::vector<Tool> &tools,
                                  const std::string &required_function = {}) {
  std::string s = "You have access to the following tools:\n\n";
  for (const auto &t : tools) {
    s += "- " + t.function.name;
    if (!t.function.description.empty()) {
      s += ": " + t.function.description;
    }
    s += "\n";
    if (!t.function.parameters.is_null() && !t.function.parameters.empty()) {
      s += "  Parameters: " + t.function.parameters.dump() + "\n";
    }
  }
  s += "\nWhen you want to call a tool, respond with ONLY this JSON (no other "
       "text):\n";
  s +=
      "{\"tool_call\":{\"name\":\"<function_name>\",\"arguments\":{<args>}}}\n";
  if (!required_function.empty()) {
    s +=
        "\nYou MUST call the function named '" + required_function +
        "'. Do not call any other function and do not respond with plain text.";
  }
  return s;
}

// §2.3: multi-format tool call detection.
//
// Recognised output formats (model-native):
//   1. InferFlux preamble format  :
//   {"tool_call":{"name":"...","arguments":{...}}}
//   2. OpenAI-style / Generic JSON: {"name":"...","arguments":{...}}
//   3. Hermes / Llama-3.1 XML     :
//   <tool_call>{"name":"...","arguments":{...}}</tool_call>
//   4. Mistral                    : [TOOL_CALLS]
//   [{"name":"...","arguments":{...}}]
//
// Returns the first detected tool call or an empty result.
ToolCallResult DetectToolCall(const std::string &text) {
  ToolCallResult result;

  // Helper: populate result from a parsed JSON object that has "name" key.
  auto fill_from_obj = [&](const json &tc) -> bool {
    if (!tc.contains("name") || !tc["name"].is_string())
      return false;
    result.function_name = tc["name"].get<std::string>();
    result.call_id = "call_" + result.function_name + "_0";
    const char *args_key =
        tc.contains("arguments")
            ? "arguments"
            : (tc.contains("parameters") ? "parameters" : nullptr);
    if (args_key && tc.contains(args_key)) {
      result.arguments_json = tc[args_key].is_object()
                                  ? tc[args_key].dump()
                                  : tc[args_key].get<std::string>();
    } else {
      result.arguments_json = "{}";
    }
    result.detected = true;
    return true;
  };

  // Format 1: {"tool_call":{...}}  (InferFlux preamble convention)
  auto try_inferflux = [&](const std::string &s) -> bool {
    try {
      auto j = json::parse(s);
      if (j.contains("tool_call") && j["tool_call"].is_object())
        return fill_from_obj(j["tool_call"]);
    } catch (const json::exception &ex) {
      LogJsonParseFailure("ExtractToolCall.try_inferflux", ex);
    }
    return false;
  };
  if (try_inferflux(text))
    return result;
  {
    auto pos = text.find("{\"tool_call\"");
    if (pos != std::string::npos && try_inferflux(text.substr(pos)))
      return result;
  }

  // Format 3: <tool_call>…</tool_call>  (Hermes / Llama-3.1)
  {
    static const std::string open_tag = "<tool_call>";
    static const std::string close_tag = "</tool_call>";
    auto a = text.find(open_tag);
    auto b = text.find(close_tag);
    if (a != std::string::npos && b != std::string::npos && b > a) {
      std::string inner =
          text.substr(a + open_tag.size(), b - a - open_tag.size());
      // Trim whitespace
      auto ws_start = inner.find_first_not_of(" \t\r\n");
      auto ws_end = inner.find_last_not_of(" \t\r\n");
      if (ws_start != std::string::npos)
        inner = inner.substr(ws_start, ws_end - ws_start + 1);
      try {
        auto j = json::parse(inner);
        if (j.is_object() && fill_from_obj(j))
          return result;
      } catch (const json::exception &ex) {
        LogJsonParseFailure("ExtractToolCall.tool_call_tag", ex);
      }
    }
  }

  // Format 4: [TOOL_CALLS] [{"name":"...","arguments":{...}}]  (Mistral)
  // Mistral models may append [/TOOL_CALLS] after the JSON array.  Parsing
  // text.substr(bracket) to end-of-string fails because nlohmann::json rejects
  // trailing non-JSON content.  Walk the bracket depth to find the matching ']'
  // and parse only that bounded substring.
  {
    static const std::string mistral_tag = "[TOOL_CALLS]";
    auto pos = text.find(mistral_tag);
    if (pos != std::string::npos) {
      auto bracket = text.find('[', pos + mistral_tag.size());
      if (bracket != std::string::npos) {
        // Find matching closing bracket, tracking nesting depth.
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
                fill_from_obj(arr[0]))
              return result;
          } catch (const json::exception &ex) {
            LogJsonParseFailure("ExtractToolCall.mistral_tool_calls", ex);
          }
        }
      }
    }
  }

  // Format 2: bare {"name":"...","arguments":{...}}  (generic OpenAI-style
  // output)
  {
    auto pos = text.find("{\"name\"");
    if (pos != std::string::npos) {
      try {
        auto j = json::parse(text.substr(pos));
        if (j.is_object() && fill_from_obj(j))
          return result;
      } catch (const json::exception &ex) {
        LogJsonParseFailure("ExtractToolCall.bare_name_object", ex);
      }
    }
  }

  return result;
}

// Build logprobs JSON for one result (shared helper).
// Chat format: {"content":[...]}; completions format: {tokens, token_logprobs,
// top_logprobs}.
static json BuildLogprobsJson(const InferenceResult &result, bool chat_mode) {
  if (result.logprobs.empty())
    return nullptr;
  if (chat_mode) {
    json content = json::array();
    for (const auto &tlp : result.logprobs) {
      json top_arr = json::array();
      for (const auto &[alt_tok, alt_lp] : tlp.top_logprobs) {
        std::vector<int> alt_bytes;
        alt_bytes.reserve(alt_tok.size());
        for (unsigned char c : alt_tok)
          alt_bytes.push_back(static_cast<int>(c));
        top_arr.push_back(
            {{"token", alt_tok}, {"logprob", alt_lp}, {"bytes", alt_bytes}});
      }
      content.push_back({{"token", tlp.token},
                         {"logprob", tlp.logprob},
                         {"bytes", tlp.bytes},
                         {"top_logprobs", top_arr}});
    }
    return {{"content", content}};
  } else {
    json tokens_arr = json::array();
    json token_logprobs_arr = json::array();
    json top_logprobs_arr = json::array();
    for (const auto &tlp : result.logprobs) {
      tokens_arr.push_back(tlp.token);
      token_logprobs_arr.push_back(tlp.logprob);
      json alt_map = json::object();
      for (const auto &[alt_tok, alt_lp] : tlp.top_logprobs)
        alt_map[alt_tok] = alt_lp;
      top_logprobs_arr.push_back(alt_map);
    }
    return {{"tokens", tokens_arr},
            {"token_logprobs", token_logprobs_arr},
            {"top_logprobs", top_logprobs_arr}};
  }
}

// Build one choice object for a single result.
static json BuildChoice(int idx, const InferenceResult &result,
                        const ToolCallResult &tool_call, bool chat_mode) {
  json logprobs_json = BuildLogprobsJson(result, chat_mode);
  if (chat_mode) {
    if (tool_call.detected) {
      json tc_arr =
          json::array({{{"id", tool_call.call_id},
                        {"type", "function"},
                        {"function",
                         {{"name", tool_call.function_name},
                          {"arguments", tool_call.arguments_json}}}}});
      return {{"index", idx},
              {"message",
               {{"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", tc_arr}}},
              {"logprobs", logprobs_json},
              {"finish_reason", "tool_calls"}};
    } else {
      const std::string fr = result.finish_reason_length ? "length" : "stop";
      return {
          {"index", idx},
          {"message", {{"role", "assistant"}, {"content", result.completion}}},
          {"logprobs", logprobs_json},
          {"finish_reason", fr}};
    }
  } else {
    const std::string fr = result.finish_reason_length ? "length" : "stop";
    return {{"index", idx},
            {"text", result.completion},
            {"logprobs", logprobs_json},
            {"finish_reason", fr}};
  }
}

// Multi-result overload: used when n>1 or best_of>1.
// total_completion_tokens covers all generated completions (including
// best_of candidates not returned), matching OpenAI's usage counting.
std::string BuildCompletionBody(const std::vector<InferenceResult> &results,
                                int total_completion_tokens,
                                const CompletionRequestPayload &request,
                                bool chat_mode,
                                const std::vector<ToolCallResult> &tool_calls) {
  auto now = std::chrono::system_clock::now();
  auto ts =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();
  std::string id_prefix = chat_mode ? "chatcmpl-" : "cmpl-";
  int prompt_toks = results.empty() ? 0 : results[0].prompt_tokens;

  json j;
  j["id"] = id_prefix + std::to_string(ts);
  j["object"] = chat_mode ? "chat.completion" : "text_completion";
  j["created"] = ts;
  j["model"] = request.model;
  j["usage"] = {{"prompt_tokens", prompt_toks},
                {"completion_tokens", total_completion_tokens},
                {"total_tokens", prompt_toks + total_completion_tokens}};

  json choices = json::array();
  for (int i = 0; i < static_cast<int>(results.size()); ++i) {
    const ToolCallResult &tc = (i < static_cast<int>(tool_calls.size()))
                                   ? tool_calls[i]
                                   : ToolCallResult{};
    choices.push_back(BuildChoice(i, results[i], tc, chat_mode));
  }
  j["choices"] = choices;
  return j.dump();
}

// Single-result overload: preserves the original call sites unchanged.
std::string
BuildCompletionBody(const InferenceResult &result,
                    const CompletionRequestPayload &request, bool chat_mode,
                    const ToolCallResult &tool_call = ToolCallResult{}) {
  return BuildCompletionBody(std::vector<InferenceResult>{result},
                             result.completion_tokens, request, chat_mode,
                             std::vector<ToolCallResult>{tool_call});
}

std::string BuildErrorBody(const std::string &error) {
  return json({{"error", error}}).dump();
}

std::vector<std::string> SplitForStreaming(const std::string &text) {
  std::vector<std::string> chunks;
  std::string current;
  for (char c : text) {
    current.push_back(c);
    if (std::isspace(static_cast<unsigned char>(c))) {
      chunks.push_back(current);
      current.clear();
    }
  }
  if (!current.empty()) {
    chunks.push_back(current);
  }
  if (chunks.empty()) {
    chunks.push_back(text);
  }
  return chunks;
}

// `logprob` is non-null when the caller collected per-token logprobs (i.e.
// the request had logprobs=true with streaming).  Ignored on finish chunks.
std::string BuildStreamChunk(const std::string &id, std::string_view model,
                             std::time_t ts, const std::string &content,
                             bool finish,
                             std::string_view finish_reason = "stop",
                             const TokenLogprob *logprob = nullptr) {
  json j;
  j["id"] = id;
  j["object"] = "chat.completion.chunk";
  j["created"] = ts;
  j["model"] = model;

  if (finish) {
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", json::object()},
                                 {"finish_reason", finish_reason}}});
  } else {
    json choice = {{"index", 0},
                   {"delta", {{"content", content}}},
                   {"finish_reason", nullptr}};
    if (logprob != nullptr) {
      // Per-token logprob in OpenAI streaming format.
      std::vector<int> tok_bytes;
      tok_bytes.reserve(content.size());
      for (unsigned char c : content)
        tok_bytes.push_back(static_cast<int>(c));
      json top_arr = json::array();
      for (const auto &[alt_tok, alt_lp] : logprob->top_logprobs) {
        std::vector<int> alt_bytes;
        for (unsigned char c : alt_tok)
          alt_bytes.push_back(static_cast<int>(c));
        top_arr.push_back(
            {{"token", alt_tok}, {"logprob", alt_lp}, {"bytes", alt_bytes}});
      }
      choice["logprobs"] = {
          {"content", json::array({{{"token", logprob->token},
                                    {"logprob", logprob->logprob},
                                    {"bytes", tok_bytes},
                                    {"top_logprobs", top_arr}}})}};
    }
    j["choices"] = json::array({choice});
  }
  return "data: " + j.dump() + "\n\n";
}

// §2.3: emit SSE delta sequence for a streaming tool call response.
// Sequence per OpenAI spec:
//   1. role=assistant, content=null
//   2. tool_calls[0] with id + type + function.name + empty arguments
//   3. tool_calls[0] function.arguments (full JSON string)
//   4. finish_reason=tool_calls, empty delta
std::string BuildToolCallStreamChunks(const std::string &id,
                                      std::string_view model, std::time_t ts,
                                      const ToolCallResult &tc) {
  std::string out;
  out.reserve(4096);
  auto base = [&]() -> json {
    json j;
    j["id"] = id;
    j["object"] = "chat.completion.chunk";
    j["created"] = ts;
    j["model"] = model;
    return j;
  };

  // Chunk 1: role=assistant, content=null
  {
    json j = base();
    j["choices"] =
        json::array({{{"index", 0},
                      {"delta", {{"role", "assistant"}, {"content", nullptr}}},
                      {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 2: tool_calls[0] — id, type, function name, empty arguments
  {
    json j = base();
    json tc_delta = json::array(
        {{{"index", 0},
          {"id", tc.call_id},
          {"type", "function"},
          {"function", {{"name", tc.function_name}, {"arguments", ""}}}}});
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", {{"tool_calls", tc_delta}}},
                                 {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 3: function arguments
  if (!tc.arguments_json.empty()) {
    json j = base();
    json arg_delta = json::array(
        {{{"index", 0}, {"function", {{"arguments", tc.arguments_json}}}}});
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", {{"tool_calls", arg_delta}}},
                                 {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 4: finish_reason=tool_calls
  {
    json j = base();
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", json::object()},
                                 {"finish_reason", "tool_calls"}}});
    out += "data: " + j.dump() + "\n\n";
  }

  return out;
}

std::string BuildApiKeysPayload(const std::vector<PolicyKeyEntry> &keys) {
  json arr = json::array();
  for (const auto &k : keys) {
    arr.push_back({{"key", k.key}, {"scopes", k.scopes}});
  }
  return json({{"api_keys", arr}}).dump();
}

HttpServer::HttpServer(std::string host, int port, Scheduler *scheduler,
                       std::shared_ptr<ApiKeyAuth> auth,
                       MetricsRegistry *metrics, OIDCValidator *oidc,
                       RateLimiter *rate_limiter, Guardrail *guardrail,
                       AuditLogger *audit_logger, PolicyBackend *policy_store,
                       std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                       const TlsConfig &tls_config, int num_workers,
                       const ModelSelectionOptions &model_selection_options)
    : host_(std::move(host)), port_(port), scheduler_(scheduler),
      auth_(std::move(auth)), metrics_(metrics), oidc_(oidc),
      rate_limiter_(rate_limiter), guardrail_(guardrail),
      audit_logger_(audit_logger), policy_store_(policy_store),
      speculative_decoder_(std::move(speculative_decoder)),
      model_selection_options_(model_selection_options),
      num_workers_(num_workers > 0 ? num_workers : 4) {
  admission_fail_closed_on_disagg_degraded_ =
      ParseBoolEnv("INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED", false);
  readyz_disagg_timeout_streak_threshold_ = ParseNonNegativeEnvInt(
      "INFERFLUX_READYZ_DISAGG_TIMEOUT_STREAK_THRESHOLD", 3);
  readyz_disagg_timeout_debt_threshold_ =
      ParseNonNegativeEnvInt("INFERFLUX_READYZ_DISAGG_TIMEOUT_DEBT_THRESHOLD",
                             readyz_disagg_timeout_streak_threshold_ > 0
                                 ? readyz_disagg_timeout_streak_threshold_ * 2
                                 : 0);
#if INFERFLUX_ENABLE_WEBUI
  webui_renderer_ = std::make_unique<WebUiRenderer>();
#endif
  if (tls_config.enabled) {
    if (tls_config.cert_path.empty() || tls_config.key_path.empty()) {
      inferflux::log::Warn(
          "http", "TLS enabled without cert/key; falling back to HTTP");
    } else {
      SSL_load_error_strings();
      OpenSSL_add_ssl_algorithms();
      ssl_ctx_ = SSL_CTX_new(TLS_server_method());
      if (!ssl_ctx_) {
        inferflux::log::Error("http", "Failed to initialize TLS context");
      } else {
        SSL_CTX_set_ecdh_auto(ssl_ctx_, 1);
        if (SSL_CTX_use_certificate_file(ssl_ctx_, tls_config.cert_path.c_str(),
                                         SSL_FILETYPE_PEM) <= 0) {
          inferflux::log::Error("http", "Failed to load TLS certificate",
                                tls_config.cert_path);
          SSL_CTX_free(ssl_ctx_);
          ssl_ctx_ = nullptr;
        } else if (SSL_CTX_use_PrivateKey_file(ssl_ctx_,
                                               tls_config.key_path.c_str(),
                                               SSL_FILETYPE_PEM) <= 0) {
          inferflux::log::Error("http", "Failed to load TLS key",
                                tls_config.key_path);
          SSL_CTX_free(ssl_ctx_);
          ssl_ctx_ = nullptr;
        } else {
          tls_enabled_ = true;
          std::cout << "[http] TLS enabled using cert=" << tls_config.cert_path
                    << std::endl;
        }
      }
    }
  }
}

HttpServer::~HttpServer() {
  Stop();
  if (ssl_ctx_) {
    SSL_CTX_free(ssl_ctx_);
    ssl_ctx_ = nullptr;
  }
}

HttpServer::ReadyStatus HttpServer::EvaluateReadyStatus() const {
  ReadyStatus status;
  const PoolRole role = role_.load(std::memory_order_relaxed);
  status.role = PoolRoleToString(role);
  status.disagg_timeout_debt_threshold =
      static_cast<uint64_t>(readyz_disagg_timeout_debt_threshold_);
  status.disagg_timeout_streak_threshold =
      static_cast<uint64_t>(readyz_disagg_timeout_streak_threshold_);

  if (role == PoolRole::kDecode) {
    if (scheduler_ && scheduler_->Router()) {
      status.model_loaded = !scheduler_->Router()->DefaultModelId().empty();
    } else {
      status.model_loaded = model_ready_.load(std::memory_order_relaxed);
    }
    status.decode_pool_warm = scheduler_ &&
                              scheduler_->ConfiguredDecodeWorkers() > 0 &&
                              scheduler_->LiveDecodeWorkers() ==
                                  scheduler_->ConfiguredDecodeWorkers();
    status.ready = status.model_loaded && status.decode_pool_warm;
    if (!status.ready) {
      status.reason = !status.model_loaded ? "no model backend loaded"
                                           : "decode pool not ready";
    }
  } else {
    if (scheduler_ && scheduler_->Router()) {
      status.model_loaded = !scheduler_->Router()->DefaultModelId().empty();
    } else {
      status.model_loaded = model_ready_.load(std::memory_order_relaxed);
    }
    status.decode_pool_warm = !scheduler_ ||
                              scheduler_->ConfiguredDecodeWorkers() == 0 ||
                              scheduler_->LiveDecodeWorkers() ==
                                  scheduler_->ConfiguredDecodeWorkers();
    status.ready = status.model_loaded;
    if (!status.ready) {
      status.reason = "no model backend loaded";
    }
  }

  const bool should_check_disagg =
      metrics_ && scheduler_ && scheduler_->HasKVTransport() &&
      role != PoolRole::kPrefill &&
      (readyz_disagg_timeout_streak_threshold_ > 0 ||
       readyz_disagg_timeout_debt_threshold_ > 0);
  if (should_check_disagg) {
    status.disagg_timeout_debt = metrics_->GetDisaggKVTimeoutDebt();
    status.disagg_timeout_streak = metrics_->GetDisaggKVTimeoutStreak();
    const bool streak_degraded =
        status.disagg_timeout_streak_threshold > 0 &&
        status.disagg_timeout_streak >= status.disagg_timeout_streak_threshold;
    const bool debt_degraded =
        status.disagg_timeout_debt_threshold > 0 &&
        status.disagg_timeout_debt >= status.disagg_timeout_debt_threshold;
    status.disagg_transport_degraded = streak_degraded || debt_degraded;
    if (status.disagg_transport_degraded && status.ready) {
      status.ready = false;
      status.reason = "distributed kv transport degraded";
    }
  }

  return status;
}

HttpServer::AdminPoolsStatus HttpServer::EvaluateAdminPoolsStatus() const {
  AdminPoolsStatus status;
  status.pool_health = EvaluateReadyStatus();
  if (!metrics_) {
    return status;
  }

  status.queue_depth = static_cast<int64_t>(metrics_->GetQueueDepth());
  status.prefill_queue_depth =
      static_cast<int64_t>(metrics_->GetPrefillQueueDepth());
  status.decode_queue_depth =
      static_cast<int64_t>(metrics_->GetDecodeQueueDepth());
  status.batch_limit_size =
      static_cast<int64_t>(metrics_->GetSchedulerBatchLimitSize());
  status.batch_limit_tokens =
      static_cast<int64_t>(metrics_->GetSchedulerBatchLimitTokens());
  status.distributed_kv.enqueue_rejections_total =
      static_cast<int64_t>(metrics_->GetDisaggKVEnqueueRejections());
  status.distributed_kv.enqueue_exhausted_total =
      static_cast<int64_t>(metrics_->GetDisaggKVEnqueueExhausted());
  status.distributed_kv.tickets_enqueued_total =
      static_cast<int64_t>(metrics_->GetDisaggKVTicketsEnqueued());
  status.distributed_kv.tickets_acknowledged_total =
      static_cast<int64_t>(metrics_->GetDisaggKVTicketsAcknowledged());
  status.distributed_kv.tickets_committed_total =
      static_cast<int64_t>(metrics_->GetDisaggKVTicketsCommitted());
  status.distributed_kv.tickets_timed_out_total =
      static_cast<int64_t>(metrics_->GetDisaggKVTicketsTimedOut());
  return status;
}

HttpServer::AdmissionDecision
HttpServer::EvaluateGenerationAdmissionDecision() const {
  AdmissionDecision decision;
  if (!admission_fail_closed_on_disagg_degraded_) {
    return decision;
  }

  const ReadyStatus ready_status = EvaluateReadyStatus();
  if (!ready_status.disagg_transport_degraded) {
    return decision;
  }

  decision.allowed = false;
  decision.http_status = 503;
  decision.error = "distributed_kv_transport_degraded";
  decision.reason = ready_status.reason.empty()
                        ? "distributed kv transport degraded"
                        : ready_status.reason;
  return decision;
}

void HttpServer::Start() {
  if (running_) {
    return;
  }
  running_ = true;
  // Start worker threads.
  for (int i = 0; i < num_workers_; ++i) {
    workers_.emplace_back(&HttpServer::WorkerLoop, this);
  }
  accept_thread_ = std::thread(&HttpServer::Run, this);
}

void HttpServer::Stop() {
  if (!running_) {
    return;
  }
  running_ = false;
  // Close the listening socket to unblock the accept() call in Run().
  int fd = server_fd_.exchange(-1);
  if (fd >= 0) {
    ::shutdown(fd, SHUT_RDWR);
    inferflux_close_socket(fd);
  }
  // Wake all worker threads.
  queue_cv_.notify_all();
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  for (auto &w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
  workers_.clear();
  // Drain any remaining clients in the queue.
  std::lock_guard<std::mutex> lock(queue_mutex_);
  while (!client_queue_.empty()) {
    auto session = std::move(client_queue_.front());
    client_queue_.pop();
    CloseSession(session);
  }
}

void HttpServer::WorkerLoop() {
  while (true) {
    ClientSession session;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock,
                     [this] { return !client_queue_.empty() || !running_; });
      if (!running_ && client_queue_.empty()) {
        return;
      }
      session = std::move(client_queue_.front());
      client_queue_.pop();
    }
    if (session.fd >= 0) {
      // HTTP/1.1 keep-alive: process multiple requests on the same
      // connection until the client closes it or an error occurs.
      constexpr int kMaxKeepAliveRequests = 100;
      for (int ka = 0; ka < kMaxKeepAliveRequests && running_; ++ka) {
        HandleClient(session);
        if (!session.keep_alive)
          break;
      }
      CloseSession(session);
    }
  }
}

void HttpServer::Run() {
#ifdef _WIN32
  WSADATA wsa_data;
  int wsa_err = WSAStartup(MAKEWORD(2, 2), &wsa_data);
  if (wsa_err != 0) {
    inferflux::log::Error("http", "WSAStartup failed with error: " +
                                      std::to_string(wsa_err));
    return;
  }
#endif

  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::perror("socket");
#ifdef _WIN32
    inferflux::log::Error("http", "socket() failed, WSAGetLastError=" +
                                      std::to_string(WSAGetLastError()));
#endif
    return;
  }

  int opt = 1;
#ifdef _WIN32
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  addr.sin_addr.s_addr = inet_addr(host_.c_str());

  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    inferflux_close_socket(fd);
    return;
  }

  if (::listen(fd, 128) < 0) {
    std::perror("listen");
    inferflux_close_socket(fd);
    return;
  }

  server_fd_.store(fd);

  while (running_) {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd =
        ::accept(fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
    if (client_fd < 0) {
      break; // Socket closed by Stop() or error — exit loop.
    }
    if (!running_) {
      inferflux_close_socket(client_fd);
      break;
    }
    // Disable Nagle's algorithm for low-latency responses (especially SSE).
    {
      int nodelay = 1;
#ifdef _WIN32
      ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char *>(&nodelay), sizeof(nodelay));
#else
      ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay,
                   sizeof(nodelay));
#endif
    }
    ClientSession session;
    session.fd = client_fd;
    if (tls_enabled_) {
      SSL *ssl = SSL_new(ssl_ctx_);
      if (!ssl) {
        inferflux_close_socket(client_fd);
        continue;
      }
      SSL_set_fd(ssl, client_fd);
      if (SSL_accept(ssl) != 1) {
        SSL_free(ssl);
        inferflux_close_socket(client_fd);
        continue;
      }
      session.ssl = ssl;
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      client_queue_.push(std::move(session));
    }
    queue_cv_.notify_one();
  }

  // If Stop() hasn't already closed the socket, close it now.
  int expected = fd;
  if (server_fd_.compare_exchange_strong(expected, -1)) {
    inferflux_close_socket(fd);
  }
#ifdef _WIN32
  WSACleanup();
#endif
}

bool HttpServer::ResolveSubject(const std::string &headers,
                                AuthContext *ctx) const {
  ctx->subject = "anonymous";
  ctx->scopes.clear();
  bool require_auth =
      (auth_ && auth_->HasKeys()) || (oidc_ && oidc_->Enabled());
  if (!require_auth) {
    ctx->scopes.insert("read");
    ctx->scopes.insert("generate");
    return true;
  }
  auto pos = headers.find("Authorization:");
  if (pos == std::string::npos) {
    return false;
  }
  auto end = headers.find("\r\n", pos);
  std::string line = headers.substr(pos, end - pos);
  auto token_pos = line.find("Bearer");
  if (token_pos == std::string::npos) {
    return false;
  }
  std::string token = line.substr(token_pos + 6);
  token.erase(0, token.find_first_not_of(' '));
  token.erase(token.find_last_not_of(' ') + 1);
  if (auth_ && auth_->HasKeys()) {
    auto hash = ApiKeyAuth::HashKey(token);
    if (auth_->IsAllowedByHash(hash)) {
      ctx->subject = token;
      auto scopes = auth_->ScopesByHash(hash);
      ctx->scopes.insert(scopes.begin(), scopes.end());
      return true;
    }
  }
  if (oidc_ && oidc_->Enabled()) {
    std::string sub;
    if (oidc_->Validate(token, &sub)) {
      ctx->subject = sub.empty() ? "oidc-user" : sub;
      ctx->scopes.insert("generate");
      ctx->scopes.insert("read");
      return true;
    }
  }
  return false;
}

bool HttpServer::RequireScope(const AuthContext &ctx, const std::string &scope,
                              ClientSession &session,
                              const std::string &error_message) {
  if (ctx.scopes.find(scope) != ctx.scopes.end()) {
    return true;
  }
  auto payload =
      BuildResponse(BuildErrorBody(error_message.empty() ? "insufficient_scope"
                                                         : error_message),
                    403, "Forbidden");
  SendAll(session, payload);
  if (audit_logger_) {
    audit_logger_->Log(ctx.subject, "", "insufficient_scope", error_message);
  }
  return false;
}

void HttpServer::HandleClient(ClientSession &session) {
  // Reset keep-alive flag for this request.
  session.keep_alive = false;
  // RAII guard: decrement connections and record latency on all exit paths.
  auto req_start = std::chrono::steady_clock::now();
  struct ConnectionGuard {
    MetricsRegistry *metrics;
    std::chrono::steady_clock::time_point start;
    ~ConnectionGuard() {
      if (metrics) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        metrics->RecordLatency(ms);
        metrics->DecrementConnections();
      }
    }
  } guard{metrics_, req_start};
  if (metrics_) {
    metrics_->IncrementConnections();
  }

  // Dynamically read the full HTTP request: headers + body based on
  // Content-Length.
  constexpr std::size_t kInitialBuf = 4096;
  constexpr std::size_t kMaxRequest = 16ULL * kMiB; // 16 MB hard limit
  std::string request;
  request.resize(kInitialBuf);
  std::size_t total = 0;
  std::size_t header_end_pos = std::string::npos;

  // Phase 1: read until we find the end-of-headers marker.
  while (header_end_pos == std::string::npos) {
    if (total >= request.size()) {
      if (request.size() >= kMaxRequest) {
        auto response = BuildResponse(BuildErrorBody("request_too_large"), 413,
                                      "Payload Too Large");
        SendAll(session, response);
        return;
      }
      request.resize(std::min(request.size() * 2, kMaxRequest));
    }
    ssize_t bytes = Receive(session, &request[total], request.size() - total);
    if (bytes <= 0) {
      return;
    }
    total += static_cast<std::size_t>(bytes);
    request.resize(total); // Shrink to actual for find()
    header_end_pos = request.find("\r\n\r\n");
    if (header_end_pos == std::string::npos) {
      // Grow for next iteration.
      request.resize(std::max(total + kInitialBuf, total * 2));
    }
  }

  // Phase 2: parse Content-Length and read remaining body bytes.
  std::size_t body_start = header_end_pos + 4;
  std::size_t content_length = 0;
  {
    auto cl_pos = request.find("Content-Length:");
    if (cl_pos == std::string::npos) {
      cl_pos = request.find("content-length:");
    }
    if (cl_pos != std::string::npos && cl_pos < header_end_pos) {
      auto val_start = cl_pos + 15; // strlen("Content-Length:")
      while (val_start < header_end_pos && request[val_start] == ' ') {
        ++val_start;
      }
      auto val_end = request.find("\r\n", val_start);
      if (val_end != std::string::npos) {
        try {
          content_length =
              std::stoull(request.substr(val_start, val_end - val_start));
        } catch (const std::exception &ex) {
          log::Debug("http_server", "Invalid Content-Length header: " +
                                        std::string(ex.what()));
        }
      }
    }
  }

  if (content_length > kMaxRequest) {
    auto response = BuildResponse(BuildErrorBody("request_too_large"), 413,
                                  "Payload Too Large");
    SendAll(session, response);
    return;
  }

  std::size_t needed = body_start + content_length;
  if (needed > total) {
    request.resize(needed);
    while (total < needed) {
      ssize_t bytes = Receive(session, &request[total], needed - total);
      if (bytes <= 0) {
        return;
      }
      total += static_cast<std::size_t>(bytes);
    }
  }
  request.resize(total);

  auto header_end = header_end_pos;
  std::string headers =
      header_end == std::string::npos ? request : request.substr(0, header_end);
  std::string body = header_end == std::string::npos
                         ? std::string()
                         : request.substr(header_end + 4);
  auto first_line_end = headers.find("\r\n");
  std::string first_line = headers.substr(0, first_line_end);
  auto method_end = first_line.find(' ');
  auto path_end = first_line.find(' ', method_end + 1);
  std::string method = first_line.substr(0, method_end);
  std::string path =
      first_line.substr(method_end + 1, path_end - method_end - 1);

  // HTTP/1.1 defaults to keep-alive unless the client sends Connection: close.
  {
    auto conn_hdr = GetHeaderValue(headers, "Connection");
    bool client_wants_close = false;
    for (auto &c : conn_hdr)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (conn_hdr.find("close") != std::string::npos) {
      client_wants_close = true;
    }
    // Enable keep-alive for non-streaming requests (streaming uses the
    // connection for the event stream and closes when done).
    session.keep_alive = !client_wants_close;
  }

#if INFERFLUX_ENABLE_WEBUI
  if (method == "GET" && path == "/ui") {
    // Serve a static HTML shell. The browser's JavaScript fetches the model
    // list via the authenticated /v1/models endpoint rather than embedding
    // server-side state here — this keeps /ui unauthenticated (safe for
    // browser navigation) while avoiding model ID disclosure to unauthenticated
    // callers.
    std::string body = webui_renderer_
                           ? webui_renderer_->RenderIndex("")
                           : "<html><body><h1>InferFlux UI</h1></body></html>";
    std::string response = "HTTP/1.1 200 OK\r\n"
                           "Content-Type: text/html; charset=utf-8\r\n"
                           "Cache-Control: no-cache\r\n"
                           "Content-Length: " +
                           std::to_string(body.size()) + "\r\n\r\n" + body;
    SendAll(session, response);
    return;
  }
#endif

  // Unauthenticated health/readiness probes.
  if (method == "GET" && path == "/healthz") {
    // Liveness probe: always 200 while the process is running.
    // Use /readyz for readiness (model loaded + pool warm).
    bool ready = model_ready_.load();
    json j = {{"status", "ok"}, {"model_ready", ready}};
    SendAll(session, BuildResponse(j.dump()));
    return;
  }
  if (method == "GET" && path == "/livez") {
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    return;
  }
  if (method == "GET" && path == "/readyz") {
    const ReadyStatus ready_status = EvaluateReadyStatus();
    json body = {{"status", ready_status.ready ? "ready" : "not_ready"},
                 {"role", ready_status.role},
                 {"model_loaded", ready_status.model_loaded},
                 {"decode_pool_warm", ready_status.decode_pool_warm}};
    if ((ready_status.disagg_timeout_streak_threshold > 0 ||
         ready_status.disagg_timeout_debt_threshold > 0) &&
        scheduler_ && scheduler_->HasKVTransport()) {
      body["disagg_timeout_debt"] = ready_status.disagg_timeout_debt;
      body["disagg_timeout_debt_threshold"] =
          ready_status.disagg_timeout_debt_threshold;
      body["disagg_timeout_streak"] = ready_status.disagg_timeout_streak;
      body["disagg_timeout_streak_threshold"] =
          ready_status.disagg_timeout_streak_threshold;
      body["disagg_transport_degraded"] =
          ready_status.disagg_transport_degraded;
    }
    if (!ready_status.ready)
      body["reason"] = ready_status.reason;
    int status_code = ready_status.ready ? 200 : 503;
    std::string status_text = ready_status.ready ? "OK" : "Service Unavailable";
    SendAll(session, BuildResponse(body.dump(), status_code, status_text));
    return;
  }

  // CORS preflight.
  if (method == "OPTIONS") {
    std::string cors_headers =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n\r\n";
    SendAll(session, cors_headers);
    return;
  }

  AuthContext auth_ctx;
  if (!ResolveSubject(headers, &auth_ctx)) {
    auto response =
        BuildResponse(BuildErrorBody("unauthorized"), 401, "Unauthorized");
    SendAll(session, response);
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "unauthorized",
                         "missing or invalid credentials");
    }
    return;
  }
  if (rate_limiter_ && rate_limiter_->Enabled() &&
      !rate_limiter_->Allow(auth_ctx.subject)) {
    auto response =
        BuildResponse(BuildErrorBody("rate_limited"), 429, "Too Many Requests");
    SendAll(session, response);
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "rate_limited",
                         "token bucket exceeded");
    }
    return;
  }

  if (method == "GET" && path == "/metrics") {
    if (!RequireScope(auth_ctx, "read", session, "metrics scope required")) {
      return;
    }
    std::string metrics_body = metrics_ ? metrics_->RenderPrometheus() : "";
    std::string metrics_headers =
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; "
        "version=0.0.4\r\nContent-Length: " +
        std::to_string(metrics_body.size()) + "\r\n\r\n";
    SendAll(session, metrics_headers + metrics_body);
    return;
  }

  if (method == "GET" && path == "/v1/admin/guardrails") {
    if (!guardrail_) {
      SendAll(session, BuildResponse(BuildErrorBody("guardrail_disabled"), 503,
                                     "Unavailable"));
      return;
    }
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto list = guardrail_->Blocklist();
    SendAll(session, BuildResponse(json({{"blocklist", list}}).dump()));
    return;
  }

  if (method == "PUT" && path == "/v1/admin/guardrails") {
    if (!guardrail_) {
      SendAll(session, BuildResponse(BuildErrorBody("guardrail_disabled"), 503,
                                     "Unavailable"));
      return;
    }
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    std::vector<std::string> list;
    try {
      auto j = json::parse(body);
      if (j.contains("blocklist") && j["blocklist"].is_array()) {
        list = j["blocklist"].get<std::vector<std::string>>();
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.guardrails.put", ex);
    }
    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> policy_lock(policy_update_mutex_);
      auto previous_blocklist = guardrail_->Blocklist();
      auto previous_store_blocklist = policy_store_
                                          ? policy_store_->GuardrailBlocklist()
                                          : std::vector<std::string>{};

      guardrail_->UpdateBlocklist(list);
      if (policy_store_) {
        policy_store_->SetGuardrailBlocklist(list);
        if (!policy_store_->Save()) {
          policy_store_->SetGuardrailBlocklist(previous_store_blocklist);
          guardrail_->UpdateBlocklist(previous_blocklist);
          SendAll(session,
                  BuildResponse(BuildErrorBody("policy_persist_failed"), 500,
                                "Internal Server Error"));
          if (audit_logger_) {
            audit_logger_->Log(auth_ctx.subject, "", "policy_persist_failed",
                               "guardrail_update");
          }
          return;
        }
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] guardrail update applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "guardrail_update",
                         "updated blocklist");
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/pools") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    const AdminPoolsStatus pools = EvaluateAdminPoolsStatus();
    json payload{
        {"status", "ok"},
        {"pool_health",
         {{"ready", pools.pool_health.ready},
          {"http_status", pools.pool_health.ready ? 200 : 503},
          {"role", pools.pool_health.role},
          {"reason", pools.pool_health.reason},
          {"model_loaded", pools.pool_health.model_loaded},
          {"decode_pool_warm", pools.pool_health.decode_pool_warm},
          {"disagg_transport_degraded",
           pools.pool_health.disagg_transport_degraded},
          {"disagg_timeout_debt", pools.pool_health.disagg_timeout_debt},
          {"disagg_timeout_debt_threshold",
           pools.pool_health.disagg_timeout_debt_threshold},
          {"disagg_timeout_streak", pools.pool_health.disagg_timeout_streak},
          {"disagg_timeout_streak_threshold",
           pools.pool_health.disagg_timeout_streak_threshold}}},
        {"scheduler",
         {{"queue_depth", OptionalIntToJson(pools.queue_depth)},
          {"prefill_queue_depth", OptionalIntToJson(pools.prefill_queue_depth)},
          {"decode_queue_depth", OptionalIntToJson(pools.decode_queue_depth)},
          {"batch_limit_size", OptionalIntToJson(pools.batch_limit_size)},
          {"batch_limit_tokens", OptionalIntToJson(pools.batch_limit_tokens)}}},
        {"distributed_kv",
         {{"enqueue_rejections_total",
           OptionalIntToJson(pools.distributed_kv.enqueue_rejections_total)},
          {"enqueue_exhausted_total",
           OptionalIntToJson(pools.distributed_kv.enqueue_exhausted_total)},
          {"tickets_enqueued_total",
           OptionalIntToJson(pools.distributed_kv.tickets_enqueued_total)},
          {"tickets_acknowledged_total",
           OptionalIntToJson(pools.distributed_kv.tickets_acknowledged_total)},
          {"tickets_committed_total",
           OptionalIntToJson(pools.distributed_kv.tickets_committed_total)},
          {"tickets_timed_out_total",
           OptionalIntToJson(pools.distributed_kv.tickets_timed_out_total)}}},
    };
    SendAll(session, BuildResponse(payload.dump()));
    return;
  }

  if (method == "GET" && path == "/v1/admin/rate_limit") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    int limit = rate_limiter_ ? rate_limiter_->CurrentLimit() : 0;
    SendAll(session,
            BuildResponse(json({{"tokens_per_minute", limit}}).dump()));
    return;
  }

  if (method == "PUT" && path == "/v1/admin/rate_limit") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    int value = 0;
    bool valid = false;
    try {
      auto j = json::parse(body);
      if (j.contains("tokens_per_minute") &&
          j["tokens_per_minute"].is_number_integer()) {
        value = j["tokens_per_minute"].get<int>();
        valid = true;
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.rate_limit.put", ex);
    }
    if (!valid) {
      SendAll(session,
              BuildResponse(BuildErrorBody("tokens_per_minute is required"),
                            400, "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> policy_lock(policy_update_mutex_);
      int previous_limit = rate_limiter_ ? rate_limiter_->CurrentLimit() : 0;
      int previous_store_limit =
          policy_store_ ? policy_store_->RateLimitPerMinute() : 0;

      if (rate_limiter_) {
        rate_limiter_->UpdateLimit(value);
      }
      if (policy_store_) {
        policy_store_->SetRateLimitPerMinute(value);
        if (!policy_store_->Save()) {
          policy_store_->SetRateLimitPerMinute(previous_store_limit);
          if (rate_limiter_) {
            rate_limiter_->UpdateLimit(previous_limit);
          }
          SendAll(session,
                  BuildResponse(BuildErrorBody("policy_persist_failed"), 500,
                                "Internal Server Error"));
          if (audit_logger_) {
            audit_logger_->Log(auth_ctx.subject, "", "policy_persist_failed",
                               "rate_limit_update");
          }
          return;
        }
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] rate-limit update applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "rate_limit_update",
                         std::to_string(value));
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    auto keys = policy_store_->ApiKeys();
    SendAll(session, BuildResponse(BuildApiKeysPayload(keys)));
    return;
  }

  if (method == "POST" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    std::string key;
    std::vector<std::string> scopes;
    try {
      auto j = json::parse(body);
      if (j.contains("key") && j["key"].is_string()) {
        key = j["key"].get<std::string>();
      }
      if (j.contains("scopes") && j["scopes"].is_array()) {
        scopes = j["scopes"].get<std::vector<std::string>>();
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.api_keys.post", ex);
    }
    if (key.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("key is required"), 400,
                                     "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> policy_lock(policy_update_mutex_);
      std::optional<std::vector<std::string>> previous_scopes;
      const std::string key_hash = ApiKeyAuth::HashKey(key);
      for (const auto &entry : policy_store_->ApiKeys()) {
        if (entry.key == key_hash) {
          previous_scopes = entry.scopes;
          break;
        }
      }

      auth_->AddKey(key, scopes);
      policy_store_->SetApiKey(key, scopes);
      if (!policy_store_->Save()) {
        if (previous_scopes.has_value()) {
          auth_->AddKeyHashed(key_hash, *previous_scopes);
          policy_store_->SetApiKey(key, *previous_scopes);
        } else {
          auth_->RemoveKey(key);
          policy_store_->RemoveApiKey(key);
        }
        SendAll(session, BuildResponse(BuildErrorBody("policy_persist_failed"),
                                       500, "Internal Server Error"));
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, "", "policy_persist_failed",
                             "api_key_upsert");
        }
        return;
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] api-key upsert applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "api_key_upsert", key);
    }
    return;
  }

  if (method == "DELETE" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    std::string key;
    try {
      auto j = json::parse(body);
      if (j.contains("key") && j["key"].is_string()) {
        key = j["key"].get<std::string>();
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.api_keys.delete", ex);
    }
    if (key.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("key is required"), 400,
                                     "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> policy_lock(policy_update_mutex_);
      std::optional<std::vector<std::string>> previous_scopes;
      const std::string key_hash = ApiKeyAuth::HashKey(key);
      for (const auto &entry : policy_store_->ApiKeys()) {
        if (entry.key == key_hash) {
          previous_scopes = entry.scopes;
          break;
        }
      }

      auth_->RemoveKey(key);
      policy_store_->RemoveApiKey(key);
      if (!policy_store_->Save()) {
        if (previous_scopes.has_value()) {
          auth_->AddKeyHashed(key_hash, *previous_scopes);
          policy_store_->SetApiKey(key, *previous_scopes);
        }
        SendAll(session, BuildResponse(BuildErrorBody("policy_persist_failed"),
                                       500, "Internal Server Error"));
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, "", "policy_persist_failed",
                             "api_key_remove");
        }
        return;
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] api-key removal applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "api_key_remove", key);
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    auto models = router->ListModels();
    std::string default_id = router->DefaultModelId();
    json payload;
    payload["default_model"] = default_id;
    payload["models"] = json::array();
    for (const auto &info : models) {
      payload["models"].push_back(BuildAdminModelJson(info, default_id));
    }
    SendAll(session, BuildResponse(payload.dump()));
    return;
  }

  if (method == "POST" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string path_value;
    std::string backend_hint;
    std::string requested_id;
    std::string requested_format = "auto";
    bool set_default = false;
    try {
      auto j = json::parse(body);
      if (j.contains("path") && j["path"].is_string()) {
        path_value = j["path"].get<std::string>();
      }
      if (j.contains("backend") && j["backend"].is_string()) {
        backend_hint = j["backend"].get<std::string>();
      }
      if (j.contains("id") && j["id"].is_string()) {
        requested_id = j["id"].get<std::string>();
      }
      if (j.contains("format") && j["format"].is_string()) {
        requested_format = j["format"].get<std::string>();
      }
      if (j.contains("default")) {
        if (j["default"].is_boolean()) {
          set_default = j["default"].get<bool>();
        } else if (j["default"].is_string()) {
          std::string val = j["default"].get<std::string>();
          for (auto &ch : val) {
            ch =
                static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
          }
          set_default = (val == "true" || val == "1" || val == "yes");
        }
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.models.post", ex);
    }
    if (path_value.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("path is required"), 400,
                                     "Bad Request"));
      return;
    }
    if (!requested_format.empty() && !IsModelFormatValue(requested_format) &&
        NormalizeModelFormat(requested_format) != "auto") {
      SendAll(session, BuildResponse(BuildErrorBody("invalid model format"),
                                     400, "Bad Request"));
      return;
    }
    auto id = router->LoadModel(path_value, backend_hint, requested_id,
                                requested_format);
    if (id.empty()) {
      const std::string load_error = router->LastLoadError();
      if (HasPrefix(load_error, "backend_policy_violation:")) {
        const std::string reason =
            StripPrefix(load_error, "backend_policy_violation:");
        json payload{
            {"error", "backend_policy_violation"},
            {"reason",
             reason.empty() ? "backend policy rejected model load" : reason},
        };
        SendAll(session,
                BuildResponse(payload.dump(), 422, "Unprocessable Entity"));
        return;
      }
      if (!load_error.empty()) {
        SendAll(
            session,
            BuildResponse(
                json({{"error", "load_failed"}, {"reason", load_error}}).dump(),
                500, "Internal Server Error"));
        return;
      }
      SendAll(session, BuildResponse(BuildErrorBody("load_failed"), 500,
                                     "Internal Server Error"));
      return;
    }
    if (set_default) {
      router->SetDefaultModel(id);
    }
    SendAll(session,
            BuildResponse(json({{"status", "ok"}, {"id", id}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, id, "model_load", path_value);
    }
    return;
  }

  if (method == "DELETE" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string id = ParseJsonStringField(body, "id");
    if (id.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("id is required"), 400,
                                     "Bad Request"));
      return;
    }
    if (!router->UnloadModel(id)) {
      SendAll(session, BuildModelNotFoundResponse());
      return;
    }
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, id, "model_unload",
                         "Removed via admin API");
    }
    return;
  }

  if (method == "PUT" && path == "/v1/admin/models/default") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string id = ParseJsonStringField(body, "id");
    if (id.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("id is required"), 400,
                                     "Bad Request"));
      return;
    }
    if (!router->SetDefaultModel(id)) {
      SendAll(session, BuildModelNotFoundResponse());
      return;
    }
    SendAll(
        session,
        BuildResponse(json({{"status", "ok"}, {"default_model", id}}).dump()));
    return;
  }

  if (method == "GET" && path == "/v1/admin/routing") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    ModelSelectionOptions current;
    {
      std::lock_guard<std::mutex> lock(model_selection_mutex_);
      current = model_selection_options_;
    }
    json payload{
        {"allow_default_fallback",
         current.allow_capability_fallback_for_default},
        {"require_ready_backend", current.require_ready_backend},
        {"fallback_scope",
         CapabilityFallbackScopeToString(current.capability_fallback_scope)},
    };
    SendAll(session, BuildResponse(payload.dump()));
    return;
  }

  if (method == "PUT" && path == "/v1/admin/routing") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    ModelSelectionOptions updated;
    {
      std::lock_guard<std::mutex> lock(model_selection_mutex_);
      updated = model_selection_options_;
    }
    bool touched = false;
    try {
      auto j = json::parse(body);
      if (j.contains("allow_default_fallback")) {
        if (!j["allow_default_fallback"].is_boolean()) {
          SendAll(session,
                  BuildResponse(BuildErrorBody(
                                    "allow_default_fallback must be a boolean"),
                                400, "Bad Request"));
          return;
        }
        updated.allow_capability_fallback_for_default =
            j["allow_default_fallback"].get<bool>();
        touched = true;
      }
      if (j.contains("require_ready_backend")) {
        if (!j["require_ready_backend"].is_boolean()) {
          SendAll(session,
                  BuildResponse(
                      BuildErrorBody("require_ready_backend must be a boolean"),
                      400, "Bad Request"));
          return;
        }
        updated.require_ready_backend = j["require_ready_backend"].get<bool>();
        touched = true;
      }
      if (j.contains("fallback_scope")) {
        if (!j["fallback_scope"].is_string()) {
          SendAll(session, BuildResponse(BuildErrorBody(
                                             "fallback_scope must be a string"),
                                         400, "Bad Request"));
          return;
        }
        const std::string scope_value = j["fallback_scope"].get<std::string>();
        if (!IsCapabilityFallbackScopeValue(scope_value)) {
          SendAll(session,
                  BuildResponse(
                      BuildErrorBody("fallback_scope must be any_compatible or "
                                     "same_path_only"),
                      400, "Bad Request"));
          return;
        }
        updated.capability_fallback_scope =
            ParseCapabilityFallbackScope(scope_value);
        touched = true;
      }
    } catch (const json::exception &) {
      SendAll(session, BuildResponse(BuildErrorBody("invalid JSON body"), 400,
                                     "Bad Request"));
      return;
    }
    if (!touched) {
      SendAll(session, BuildResponse(BuildErrorBody(
                                         "at least one routing policy field is "
                                         "required"),
                                     400, "Bad Request"));
      return;
    }

    {
      std::lock_guard<std::mutex> policy_lock(policy_update_mutex_);
      ModelSelectionOptions previous;
      {
        std::lock_guard<std::mutex> lock(model_selection_mutex_);
        previous = model_selection_options_;
        model_selection_options_ = updated;
      }
      if (scheduler_) {
        scheduler_->UpdateModelSelectionOptions(updated);
      }
      if (policy_store_) {
        auto previous_store_policy = policy_store_->RoutingPolicy();
        RoutingPolicyEntry routing_policy;
        routing_policy.allow_default_fallback =
            updated.allow_capability_fallback_for_default;
        routing_policy.require_ready_backend = updated.require_ready_backend;
        routing_policy.fallback_scope =
            CapabilityFallbackScopeToString(updated.capability_fallback_scope);
        policy_store_->SetRoutingPolicy(routing_policy);
        if (!policy_store_->Save()) {
          if (previous_store_policy.has_value()) {
            policy_store_->SetRoutingPolicy(*previous_store_policy);
          } else {
            policy_store_->ClearRoutingPolicy();
          }
          {
            std::lock_guard<std::mutex> lock(model_selection_mutex_);
            model_selection_options_ = previous;
          }
          if (scheduler_) {
            scheduler_->UpdateModelSelectionOptions(previous);
          }
          SendAll(session,
                  BuildResponse(BuildErrorBody("policy_persist_failed"), 500,
                                "Internal Server Error"));
          if (audit_logger_) {
            audit_logger_->Log(auth_ctx.subject, "", "policy_persist_failed",
                               "routing_policy_update");
          }
          return;
        }
      }
    }

    json payload{
        {"status", "ok"},
        {"allow_default_fallback",
         updated.allow_capability_fallback_for_default},
        {"require_ready_backend", updated.require_ready_backend},
        {"fallback_scope",
         CapabilityFallbackScopeToString(updated.capability_fallback_scope)},
    };
    SendAll(session, BuildResponse(payload.dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "routing_policy_update",
                         payload.dump());
    }
    return;
  }

  // ── Prefix cache admin endpoints (Workstream C) ──────────────────────────
  // GET  /v1/admin/cache       — live cache size/capacity + hit/miss metrics
  // POST /v1/admin/cache/warm  — pre-seed a cache entry with raw BPE token IDs

  if (method == "GET" && path == "/v1/admin/cache") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *cache = scheduler_ ? scheduler_->PrefixCache() : nullptr;
    auto *paged_kv = scheduler_ ? scheduler_->Cache() : nullptr;
    auto cm = GlobalMetrics().GetCacheMetrics();
    const auto inferflux_cuda_model_memory =
        metrics_ ? metrics_->GetInferfluxCudaModelMemorySnapshot()
                 : MetricsRegistry::InferfluxCudaModelMemorySnapshot{};
    const auto inferflux_cuda_kv_memory =
        metrics_ ? metrics_->GetInferfluxCudaKvMemorySnapshot()
                 : MetricsRegistry::InferfluxCudaKvMemorySnapshot{};
    const auto paged_kv_usage =
        paged_kv ? paged_kv->GetUsageSnapshot() : PagedKVCache::UsageSnapshot{};
    const auto prefix_memory =
        cache ? cache->MemorySnapshot() : RadixPrefixMemorySnapshot{};
    double hit_rate = (cm.hits + cm.misses) > 0
                          ? static_cast<double>(cm.hits) /
                                static_cast<double>(cm.hits + cm.misses)
                          : 0.0;
    json memory_domains = json::object();
    for (const auto &entry : inferflux_cuda_model_memory.domains) {
      memory_domains[entry.first] = {
          {"reserved_bytes", static_cast<int64_t>(entry.second.reserved_bytes)},
          {"in_use_bytes", static_cast<int64_t>(entry.second.in_use_bytes)},
          {"high_water_bytes",
           static_cast<int64_t>(entry.second.high_water_bytes)},
          {"evictable_bytes",
           static_cast<int64_t>(entry.second.evictable_bytes)},
      };
    }
    json payload{
        {"size", cache ? static_cast<int64_t>(cache->Size()) : 0},
        {"capacity", cache ? static_cast<int64_t>(cache->Capacity()) : 0},
        {"hits", static_cast<int64_t>(cm.hits)},
        {"misses", static_cast<int64_t>(cm.misses)},
        {"hit_rate", hit_rate},
        {"partial_hits", static_cast<int64_t>(cm.partial_hits)},
        {"matched_tokens", static_cast<int64_t>(cm.matched_tokens)},
        {"kv_reuse_count", static_cast<int64_t>(cm.kv_reuse_count)},
        {"kv_reuse_tokens", static_cast<int64_t>(cm.kv_reuse_tokens)},
        {"memory",
         {{"inferflux_cuda_model",
           {{"model", inferflux_cuda_model_memory.model_label},
            {"reserved_bytes", static_cast<int64_t>(inferflux_cuda_model_memory
                                                        .total.reserved_bytes)},
            {"in_use_bytes", static_cast<int64_t>(inferflux_cuda_model_memory
                                                      .total.in_use_bytes)},
            {"high_water_bytes",
             static_cast<int64_t>(
                 inferflux_cuda_model_memory.total.high_water_bytes)},
            {"evictable_bytes",
             static_cast<int64_t>(
                 inferflux_cuda_model_memory.total.evictable_bytes)},
            {"domains", std::move(memory_domains)}}},
          {"inferflux_cuda_kv",
           {{"total_bytes",
             static_cast<int64_t>(inferflux_cuda_kv_memory.total_bytes)},
            {"active_bytes",
             static_cast<int64_t>(inferflux_cuda_kv_memory.active_bytes)},
            {"prefix_retained_bytes",
             static_cast<int64_t>(
                 inferflux_cuda_kv_memory.prefix_retained_bytes)},
            {"free_bytes",
             static_cast<int64_t>(inferflux_cuda_kv_memory.free_bytes)},
            {"active_sequences", inferflux_cuda_kv_memory.active_sequences},
            {"prefix_retained_sequences",
             inferflux_cuda_kv_memory.prefix_retained_sequences},
            {"free_sequences", inferflux_cuda_kv_memory.free_sequences},
            {"max_sequences", inferflux_cuda_kv_memory.max_sequences}}},
          {"paged_kv",
           {{"total_blocks", static_cast<int64_t>(paged_kv_usage.total_blocks)},
            {"used_blocks", static_cast<int64_t>(paged_kv_usage.used_blocks)},
            {"free_blocks", static_cast<int64_t>(paged_kv_usage.free_blocks)},
            {"page_size_bytes",
             static_cast<int64_t>(paged_kv_usage.page_size_bytes)},
            {"total_bytes", static_cast<int64_t>(paged_kv_usage.total_bytes())},
            {"used_bytes", static_cast<int64_t>(paged_kv_usage.used_bytes())},
            {"free_bytes", static_cast<int64_t>(paged_kv_usage.free_bytes())},
            {"prefix_retained_blocks",
             static_cast<int64_t>(prefix_memory.unique_retained_blocks)},
            {"prefix_retained_bytes",
             static_cast<int64_t>(prefix_memory.retained_bytes)},
            {"prefix_live_sequences",
             static_cast<int64_t>(prefix_memory.live_sequences)}}}}}};
    SendAll(session, BuildResponse(payload.dump()));
    return;
  }

  if (method == "POST" && path == "/v1/admin/cache/warm") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *cache = scheduler_ ? scheduler_->PrefixCache() : nullptr;
    if (!cache) {
      SendAll(session, BuildResponse(BuildErrorBody("cache_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::vector<int> tokens;
    std::vector<int> block_table;
    try {
      auto j = json::parse(body);
      if (j.contains("tokens") && j["tokens"].is_array()) {
        tokens = j["tokens"].get<std::vector<int>>();
      }
      if (j.contains("block_table") && j["block_table"].is_array()) {
        block_table = j["block_table"].get<std::vector<int>>();
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("admin.cache.warm.post", ex);
    }
    if (tokens.empty() || block_table.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody(
                                         "tokens and block_table are required"),
                                     400, "Bad Request"));
      return;
    }
    // We use a dummy sequence_id and null backend for administrative warming.
    cache->Insert(tokens, block_table, -1, nullptr);
    SendAll(session,
            BuildResponse(json({{"status", "ok"},
                                {"size", static_cast<int64_t>(cache->Size())}})
                              .dump()));
    return;
  }

  // ── OpenAI-standard /v1/models
  // ────────────────────────────────────────────── GET /v1/models          —
  // list all registered models (read scope) GET /v1/models/{id}     — describe
  // a single model (read scope)
  //
  // Distinct from the admin-only /v1/admin/models (load/unload/default).
  // Every OpenAI-compatible SDK (LangChain, LlamaIndex, openai-python) calls
  // this endpoint to discover available models.
  static constexpr std::size_t kV1ModelsPrefixLen = 11; // "/v1/models/"
  if (method == "GET" &&
      (path == "/v1/models" ||
       (path.size() > kV1ModelsPrefixLen &&
        path.substr(0, kV1ModelsPrefixLen) == "/v1/models/"))) {
    if (!RequireScope(auth_ctx, "read", session, "read scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    auto models = router ? router->ListModels() : std::vector<ModelInfo>{};
    // Epoch for created timestamp — use a stable server-start approximation.
    const auto created_ts = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    if (path == "/v1/models") {
      json data = json::array();
      for (const auto &m : models) {
        data.push_back(BuildOpenAIModelJson(m, created_ts));
      }
      SendAll(session,
              BuildResponse(json({{"object", "list"}, {"data", data}}).dump()));
      return;
    }

    // /v1/models/{id}
    std::string model_id = path.substr(kV1ModelsPrefixLen); // strip prefix
    if (const ModelInfo *model = FindModelById(models, model_id)) {
      SendAll(session,
              BuildResponse(BuildOpenAIModelJson(*model, created_ts).dump()));
      return;
    }
    SendAll(session, BuildModelNotFoundResponse());
    return;
  }

  // ── POST /v1/embeddings ──────────────────────────────────────────────────
  // OpenAI-compatible embeddings endpoint.  Two llama_context instances share
  // one llama_model* so weights are in RAM only once (structurally impossible
  // across process boundaries).  Requires "read" scope.
  if (method == "POST" && path == "/v1/embeddings") {
    if (!RequireScope(auth_ctx, "read", session, "read scope required")) {
      return;
    }
    std::string embed_model;
    std::vector<std::string> inputs;
    try {
      auto j = json::parse(body);
      if (j.contains("model") && j["model"].is_string()) {
        embed_model = j["model"].get<std::string>();
      }
      if (j.contains("input")) {
        if (j["input"].is_string()) {
          inputs.push_back(j["input"].get<std::string>());
        } else if (j["input"].is_array()) {
          for (const auto &item : j["input"]) {
            if (item.is_string()) {
              inputs.push_back(item.get<std::string>());
            }
          }
        }
      }
    } catch (const json::exception &ex) {
      LogJsonParseFailure("embeddings.post", ex);
    }
    if (inputs.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("input is required"), 400,
                                     "Bad Request"));
      return;
    }

    // Resolve backend with capability-aware fallback semantics.
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    std::shared_ptr<BackendInterface> embed_backend;
    std::string resolved_model = embed_model.empty() ? "default" : embed_model;
    if (router) {
      BackendFeatureRequirements requirements =
          BuildEmbeddingFeatureRequirements();
      ModelSelectionOptions embedding_options;
      {
        std::lock_guard<std::mutex> lock(model_selection_mutex_);
        embedding_options = model_selection_options_;
      }
      embedding_options.require_ready_backend = true;
      auto selection = SelectModelForRequest(router, embed_model, requirements,
                                             embedding_options);
      if (selection.status == ModelSelectionStatus::kNotFound &&
          !embed_model.empty() && !IsDefaultModelAlias(embed_model)) {
        SendAll(session, BuildResponse(BuildErrorBody("model_not_found"), 404,
                                       "Not Found"));
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, embed_model, "model_not_found",
                             "Unknown embeddings model");
        }
        return;
      }
      if (selection.status == ModelSelectionStatus::kUnsupported) {
        if (metrics_) {
          metrics_->RecordCapabilityRejection(selection.info.backend,
                                              selection.missing_feature);
        }
        SendAll(session,
                BuildResponse(
                    BuildErrorBody(selection.reason.empty()
                                       ? "Selected model does not support "
                                         "embeddings"
                                       : selection.reason),
                    422, "Unprocessable Entity"));
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject,
                             selection.info.id.empty() ? resolved_model
                                                       : selection.info.id,
                             "capability_reject",
                             selection.reason.empty() ? "embeddings unsupported"
                                                      : selection.reason);
        }
        return;
      }
      if (selection.status == ModelSelectionStatus::kSelected) {
        embed_backend = selection.backend;
        if (!selection.info.id.empty()) {
          resolved_model = selection.info.id;
        }
        if (selection.used_fallback && metrics_) {
          metrics_->RecordCapabilityRouteFallback(
              selection.fallback_from_backend, selection.info.backend,
              selection.fallback_feature.empty() ? "unsupported_feature"
                                                 : selection.fallback_feature);
        }
      }
    }
    if (!embed_backend || !embed_backend->IsReady()) {
      SendAll(session, BuildResponse(BuildErrorBody("no_backend"), 503,
                                     "Service Unavailable"));
      return;
    }

    // Generate embeddings for each input.
    json data = json::array();
    int total_tokens = 0;
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
      std::vector<float> emb = embed_backend->Embed(inputs[idx]);
      if (emb.empty()) {
        SendAll(session, BuildResponse(BuildErrorBody(
                                           "model_does_not_support_embeddings"),
                                       422, "Unprocessable Entity"));
        return;
      }
      total_tokens += embed_backend->TokenCount(inputs[idx]);
      json entry = {{"object", "embedding"},
                    {"embedding", emb},
                    {"index", static_cast<int>(idx)}};
      data.push_back(std::move(entry));
    }

    json resp = {
        {"object", "list"},
        {"data", data},
        {"model", resolved_model},
        {"usage",
         {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}}}};
    SendAll(session, BuildResponse(resp.dump()));
    return;
  }

  if (method == "POST" &&
      (path == "/v1/completions" || path == "/v1/chat/completions")) {
    if (!RequireScope(auth_ctx, "generate", session,
                      "generate scope required")) {
      return;
    }
    auto parsed = ParseJsonPayload(body);
    auto LogToolEvent = [](const std::string &line) {
      if (const char *path = std::getenv("INFERFLUX_LOG_TOOL_CALLS")) {
        std::ofstream out(path, std::ios::app);
        if (out) {
          out << line << std::endl;
        }
      }
    };
    if (!parsed.response_format_ok) {
      auto payload = BuildResponse(BuildErrorBody(parsed.response_format_error),
                                   400, "Bad Request");
      SendAll(session, payload);
      return;
    }

    InferenceRequest req;
    // Note: req.prompt is set below after the tool/template block which
    // handles both the messages chat path and the direct prompt path.
    if (parsed.max_tokens > 0) {
      req.max_tokens = parsed.max_tokens;
    }
    req.model = parsed.model;
    req.session_id = parsed.session_id;
    if (req.session_id.empty()) {
      req.session_id = GetHeaderValue(headers, "x-inferflux-session-id");
    }
    req.client_request_id = parsed.client_request_id;
    if (req.client_request_id.empty()) {
      req.client_request_id =
          GetHeaderValue(headers, "x-inferflux-client-request-id");
    }
    req.json_mode = parsed.json_mode;
    if (parsed.has_response_format) {
      req.response_format.has_format = true;
      req.response_format.type = parsed.response_format_type;
      req.response_format.schema = parsed.response_format_schema;
      req.response_format.grammar = parsed.response_format_grammar;
      req.response_format.root = parsed.response_format_root;
    }
    req.stream = parsed.stream;
    if (parsed.logprobs) {
      req.collect_logprobs = true;
      // top_logprobs=0 means selected-token logprob only (no alternatives).
      // CollectLogprob skips the O(V log V) partial-sort when top_n==0.
      req.logprob_top_n = parsed.top_logprobs; // 0-20, already clamped above
    }

    // Sampling parameters (temperature, top_p, top_k, etc.).
    req.sampling = {parsed.temperature,
                    parsed.top_p,
                    parsed.top_k,
                    parsed.min_p,
                    parsed.frequency_penalty,
                    parsed.presence_penalty,
                    parsed.repetition_penalty,
                    /*penalty_last_n=*/64,
                    parsed.seed,
                    parsed.logit_bias};

    // Stop sequences.
    req.stop = parsed.stop;

    // §2.2: attach decoded images (populated when messages contain image_url
    // parts).
    if (parsed.has_images) {
      req.has_images = true;
      req.images = std::move(parsed.images);
      GlobalMetrics().RecordImagePreprocess(static_cast<int>(req.images.size()),
                                            0.0);
    }

    // W3C Trace Context (OBS-2): propagate trace-id from incoming traceparent
    // header.
    {
      std::string tp_header = GetHeaderValue(headers, "traceparent");
      if (!tp_header.empty()) {
        auto parent_ctx = tracing::ParseTraceparent(tp_header);
        req.trace_id = parent_ctx.trace_id;
      }
    }

    // §2.3: tool schema injection + model-native chat template formatting.
    bool use_tools = (parsed.has_tool_schema || !parsed.tools.empty()) &&
                     parsed.tool_choice != "none";
    if (use_tools && std::getenv("INFERFLUX_LOG_TOOL_CALLS")) {
      std::cout << "[tools] request provided " << parsed.tools.size()
                << " tool definition(s); choice=" << parsed.tool_choice
                << std::endl;
    }
    if (use_tools) {
      LogToolEvent("request tools=" + std::to_string(parsed.tools.size()) +
                   " choice=" + parsed.tool_choice);
    }

    // §2.3 model-native chat template path.
    // When a real model is loaded, format the conversation using the model's
    // built-in template (llama_chat_apply_template).  Tool definitions are
    // injected as the first system message so the model-specific role tokens
    // wrap them correctly.
    //
    // /v1/completions with a raw prompt is auto-wrapped as a user message so
    // instruct-tuned models receive proper chat formatting.  This eliminates
    // the semantic divergence between native and llama.cpp backends that occurs
    // when raw text is fed to instruction-tuned models without special tokens.
    //
    // When no model is available (stub mode) or the template is unsupported,
    // we fall back to FlattenMessages + BuildToolSystemPrompt.
    bool use_native_template = false;

    // Auto-wrap raw /v1/completions prompt as a chat user message.
    if (!parsed.prompt.empty() && parsed.messages.empty()) {
      parsed.messages.push_back({/*role=*/"user", /*content=*/parsed.prompt});
      parsed.prompt.clear();
    }

    if (!parsed.messages.empty()) {
      auto *router = scheduler_->Router();
      if (router) {
        auto *info = router->Resolve(parsed.model);
        if (info) {
          auto backend = router->GetBackend(info->id);
          if (backend && backend->IsReady()) {
            // Build message list: prepend tool schema as system message if
            // needed.
            std::vector<std::pair<std::string, std::string>> msgs;
            if (use_tools && !parsed.tools.empty()) {
              msgs.push_back(
                  {"system", BuildToolSystemPrompt(
                                 parsed.tools, parsed.tool_choice_function)});
            }
            for (const auto &m : parsed.messages) {
              if (!m.role.empty() || !m.content.empty()) {
                msgs.push_back({m.role, m.content});
              }
            }
            auto tmpl = backend->FormatChatMessages(
                msgs, /*add_assistant_prefix=*/true);
            if (tmpl.valid) {
              req.prompt = tmpl.prompt;
              use_native_template = true;
              LogToolEvent("native_template=true msgs=" +
                           std::to_string(msgs.size()));
            }
          }
        }
      }
    }

    if (!use_native_template) {
      // Fallback: flat concatenation + preamble injection (stub / unsupported
      // template models).
      if (!parsed.prompt.empty()) {
        req.prompt = parsed.prompt;
      } else if (!parsed.messages.empty()) {
        req.prompt = FlattenMessages(parsed.messages);
      }
      if (use_tools && !parsed.tools.empty()) {
        std::string tool_prefix =
            BuildToolSystemPrompt(parsed.tools, parsed.tool_choice_function);
        req.prompt =
            req.prompt.empty() ? tool_prefix : tool_prefix + "\n" + req.prompt;
      }
    }

    req.priority = static_cast<int>(auth_ctx.scopes.count("admin") ? 10 : 0);
    if (req.prompt.empty()) {
      auto payload =
          BuildResponse(BuildErrorBody("prompt or messages are required"), 400,
                        "Bad Request");
      SendAll(session, payload);
      return;
    }

    const auto admission = EvaluateGenerationAdmissionDecision();
    if (!admission.allowed) {
      auto payload =
          BuildResponse(BuildErrorBody(admission.error), admission.http_status,
                        "Service Unavailable");
      SendAll(session, payload);
      if (audit_logger_) {
        audit_logger_->Log(auth_ctx.subject, parsed.model, admission.error,
                           admission.reason);
      }
      return;
    }

    if (auto *router = scheduler_ ? scheduler_->Router() : nullptr) {
      ModelSelectionOptions generation_options;
      {
        std::lock_guard<std::mutex> lock(model_selection_mutex_);
        generation_options = model_selection_options_;
      }
      generation_options.require_ready_backend = false;
      auto selection = SelectModelForRequest(
          router, parsed.model,
          BuildGenerationRequirements(
              parsed, speculative_decoder_ && speculative_decoder_->Enabled()),
          generation_options);
      if (selection.status == ModelSelectionStatus::kNotFound &&
          !parsed.model.empty() && !IsDefaultModelAlias(parsed.model)) {
        auto payload =
            BuildResponse(BuildErrorBody("model_not_found"), 404, "Not Found");
        SendAll(session, payload);
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, parsed.model, "model_not_found",
                             "Unknown requested model");
        }
        return;
      }
      if (selection.status == ModelSelectionStatus::kUnsupported) {
        if (metrics_) {
          metrics_->RecordCapabilityRejection(selection.info.backend,
                                              selection.missing_feature);
        }
        auto payload = BuildResponse(
            BuildErrorBody(selection.reason.empty()
                               ? "Selected model does not support "
                                 "requested features"
                               : selection.reason),
            422, "Unprocessable Entity");
        SendAll(session, payload);
        if (audit_logger_) {
          audit_logger_->Log(
              auth_ctx.subject,
              selection.info.id.empty() ? parsed.model : selection.info.id,
              "capability_reject",
              selection.reason.empty() ? "requested feature unsupported"
                                       : selection.reason);
        }
        return;
      }
    }

    bool chat_mode =
        (path == "/v1/chat/completions") || !parsed.messages.empty();

    // /v1/completions is deprecated per OpenAI API (Jan 2024).
    // Log once and add Deprecation header to response.
    const bool is_legacy_completions =
        (path == "/v1/completions" && !chat_mode);
    if (is_legacy_completions) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        inferflux::log::Warn(
            "server",
            "/v1/completions is deprecated; use /v1/chat/completions instead");
      }
    }
    std::string guard_reason;
    if (guardrail_ && guardrail_->Enabled() &&
        !guardrail_->Check(req.prompt, &guard_reason)) {
      auto payload =
          BuildResponse(BuildErrorBody(guard_reason), 400, "Bad Request");
      SendAll(session, payload);
      if (audit_logger_) {
        audit_logger_->Log(auth_ctx.subject, parsed.model, "blocked",
                           guard_reason);
      }
      return;
    }
    // ── Multi-completion path (n>1 or best_of>1) ──────────────────────────
    // Must be handled before the streaming setup because n>1 is incompatible
    // with SSE streaming (no way to interleave independent completion tokens).
    if (parsed.n > 1 || parsed.best_of > 1) {
      if (parsed.stream) {
        SendAll(session,
                BuildResponse(
                    BuildErrorBody("n>1 and best_of not supported with stream"),
                    400, "Bad Request"));
        return;
      }
      const int n_return = parsed.n;
      const int n_generate = parsed.best_of; // best_of >= n enforced at parse

      // For best_of ranking we need per-token logprobs; force them internally
      // even when the caller did not request them.
      bool need_internal_logprobs =
          (n_generate > n_return) && !req.collect_logprobs;
      if (need_internal_logprobs) {
        req.collect_logprobs = true;
        // logprob_top_n stays 0: only selected-token logprob needed for sum.
      }

      std::vector<std::future<InferenceResult>> futures;
      futures.reserve(n_generate);
      try {
        for (int i = 0; i < n_generate; ++i) {
          InferenceRequest cur = req;
          // Give each generation a different seed so results vary.
          if (i > 0) {
            cur.sampling.seed = (parsed.seed == UINT32_MAX)
                                    ? UINT32_MAX
                                    : parsed.seed + static_cast<uint32_t>(i);
          }
          futures.push_back(scheduler_->Generate(std::move(cur)));
        }
      } catch (const std::exception &ex) {
        if (metrics_)
          metrics_->RecordError();
        SendAll(session,
                BuildResponse(BuildErrorBody(ex.what()), 500, "Error"));
        return;
      }

      std::vector<InferenceResult> all_results;
      all_results.reserve(n_generate);
      int total_completion_tokens = 0;
      for (auto &f : futures) {
        auto r = f.get();
        total_completion_tokens += r.completion_tokens;
        all_results.push_back(std::move(r));
      }

      // If best_of > n, rank by cumulative logprob and keep top n.
      if (n_generate > n_return) {
        auto cumlogprob = [](const InferenceResult &r) -> double {
          double s = 0.0;
          for (const auto &tlp : r.logprobs)
            s += static_cast<double>(tlp.logprob);
          return s;
        };
        std::sort(all_results.begin(), all_results.end(),
                  [&](const InferenceResult &a, const InferenceResult &b) {
                    return cumlogprob(a) > cumlogprob(b);
                  });
        all_results.resize(n_return);
        // Strip logprobs if they were only needed for ranking.
        if (need_internal_logprobs) {
          for (auto &r : all_results)
            r.logprobs.clear();
        }
      }

      // Detect tool calls per choice.
      std::vector<ToolCallResult> tool_calls;
      tool_calls.reserve(all_results.size());
      for (auto &r : all_results) {
        if (use_tools) {
          bool is_stub =
              r.no_backend || r.completion.find("No model backend is loaded") !=
                                  std::string::npos;
          if (is_stub && !parsed.tools.empty()) {
            ToolCallResult tc;
            tc.detected = true;
            tc.function_name = parsed.tools.front().function.name;
            if (tc.function_name.empty())
              tc.function_name = "stub_tool";
            tc.call_id = "call_stub_" + tc.function_name;
            tc.arguments_json = json{{"reason", "no_model_available"}}.dump();
            tool_calls.push_back(std::move(tc));
          } else {
            tool_calls.push_back(DetectToolCall(r.completion));
          }
        } else {
          tool_calls.push_back(ToolCallResult{});
        }
      }

      // Record aggregate metrics using the first result for token counts.
      if (metrics_ && !all_results.empty() && !all_results[0].no_backend &&
          !IsBackendEmptyResponse(all_results[0])) {
        metrics_->RecordSuccess(all_results[0].prompt_tokens,
                                total_completion_tokens);
      }
      if (audit_logger_ && !all_results.empty()) {
        audit_logger_->LogRequest(auth_ctx.subject, parsed.model, req.prompt,
                                  all_results[0].completion,
                                  all_results[0].prompt_tokens,
                                  total_completion_tokens);
      }

      SpanContext mc_parent;
      mc_parent.trace_id = req.trace_id;
      SpanContext mc_ctx = tracing::ChildContext(mc_parent);
      std::string mc_trace_hdr;
      if (mc_ctx.valid())
        mc_trace_hdr = "traceparent: " + mc_ctx.ToTraceparent() + "\r\n";

      SendAll(session, BuildResponse(BuildCompletionBody(
                                         all_results, total_completion_tokens,
                                         parsed, chat_mode, tool_calls),
                                     200, "OK", mc_trace_hdr));
      return;
    }
    // ── End multi-completion path ──────────────────────────────────────────

    // Build a child SpanContext for this request so downstream services can
    // correlate spans. The traceparent is emitted in the response header.
    SpanContext parent_ctx;
    parent_ctx.trace_id = req.trace_id;
    SpanContext request_ctx = tracing::ChildContext(parent_ctx);
    std::string trace_response_header;
    if (request_ctx.valid()) {
      trace_response_header =
          "traceparent: " + request_ctx.ToTraceparent() + "\r\n";
    }
    if (is_legacy_completions) {
      trace_response_header +=
          "Deprecation: true\r\n"
          "Sunset: 2025-06-01\r\n"
          "Link: </v1/chat/completions>; rel=\"successor-version\"\r\n";
    }

    std::string stream_id;
    std::time_t stream_ts = 0;
    auto stream_mutex = std::make_shared<std::mutex>();
    auto stream_active = std::make_shared<std::atomic<bool>>(false);
    auto stream_had_chunk = std::make_shared<std::atomic<bool>>(false);
    auto stream_cancel_flag = std::make_shared<std::atomic<bool>>(false);
    // Declared here (outer scope) so they're visible in both the streaming
    // setup block and the post-Generate streaming completion block.
    auto token_buffer = std::make_shared<std::vector<std::string>>();
    bool buffer_tokens = use_tools;
    if (parsed.stream) {
      stream_ts = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
      stream_active->store(true);
      // SSE consumes the connection — disable keep-alive for this session.
      session.keep_alive = false;
      stream_id = std::string("chatcmpl-") + std::to_string(stream_ts);
      std::string stream_headers = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/event-stream\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Connection: keep-alive\r\n" +
                                   trace_response_header + "\r\n";
      if (!SendAll(session, stream_headers)) {
        return;
      }
      req.cancellation_flag = stream_cancel_flag;
      ClientSession *stream_session = &session;
      std::string stream_model = parsed.model;
      // §2.3: when tools are active we cannot stream tokens as content deltas
      // because we don't know until the full completion arrives whether the
      // model produced a tool_call JSON envelope or plain text.  Buffer all
      // tokens during generation; after Generate() returns either:
      //   a) tool_call detected  → discard buffer, emit tool_calls delta
      //   sequence b) no tool_call        → replay buffer as content deltas,
      //   then stop chunk
      // When use_tools=false the buffer is never populated and the normal
      // per-token streaming path runs unchanged.
      bool stream_collect_logprobs = req.collect_logprobs;
      req.on_token = [this, stream_session, stream_mutex, stream_active,
                      stream_had_chunk, stream_cancel_flag, stream_id,
                      stream_model, stream_ts, token_buffer, buffer_tokens,
                      stream_collect_logprobs](const std::string &chunk,
                                               const TokenLogprob *lp) {
        if (chunk.empty() || !stream_active->load()) {
          return;
        }
        if (buffer_tokens) {
          // Accumulate without sending; will be replayed or discarded below.
          std::lock_guard<std::mutex> lock(*stream_mutex);
          token_buffer->push_back(chunk);
          return;
        }
        // When logprobs are requested emit the full token as a single SSE
        // delta (no splitting) so the logprob is paired 1:1 with its token.
        if (stream_collect_logprobs && lp != nullptr) {
          std::string payload = BuildStreamChunk(
              stream_id, stream_model, stream_ts, chunk, false, "stop", lp);
          std::lock_guard<std::mutex> lock(*stream_mutex);
          if (!stream_active->load()) {
            return;
          }
          if (!SendAll(*stream_session, payload)) {
            stream_active->store(false);
            stream_cancel_flag->store(true);
            return;
          }
          stream_had_chunk->store(true);
          return;
        }
        auto pieces = SplitForStreaming(chunk);
        for (const auto &piece : pieces) {
          std::string payload = BuildStreamChunk(stream_id, stream_model,
                                                 stream_ts, piece, false);
          std::lock_guard<std::mutex> lock(*stream_mutex);
          if (!stream_active->load()) {
            return;
          }
          if (!SendAll(*stream_session, payload)) {
            stream_active->store(false);
            stream_cancel_flag->store(true);
            return;
          }
          stream_had_chunk->store(true);
        }
      };
    }

    const std::string audit_prompt = req.prompt;
    try {
      auto future = scheduler_->Generate(std::move(req));
      auto result = future.get();
      if (result.no_backend) {
        if (metrics_) {
          metrics_->RecordError();
        }
        if (parsed.stream) {
          // SSE headers were already sent. Complete the stream body rather than
          // sending a second HTTP response (which would corrupt the framing).
          ToolCallResult nb_tc;
          if (use_tools &&
              (!parsed.tools.empty() || !parsed.first_tool_name.empty())) {
            std::string fname = !parsed.tools.empty()
                                    ? parsed.tools.front().function.name
                                    : parsed.first_tool_name;
            if (fname.empty())
              fname = "stub_tool";
            nb_tc.detected = true;
            nb_tc.function_name = fname;
            nb_tc.call_id = "call_stub_" + fname;
            nb_tc.arguments_json =
                json{{"reason", "no_model_available"}}.dump();
          }
          {
            std::lock_guard<std::mutex> lock(*stream_mutex);
            if (nb_tc.detected) {
              SendAll(session, BuildToolCallStreamChunks(
                                   stream_id, parsed.model, stream_ts, nb_tc));
            } else {
              SendAll(session,
                      BuildStreamChunk(stream_id, parsed.model, stream_ts,
                                       result.completion, false));
              SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                stream_ts, "", true));
            }
            SendAll(session, "data: [DONE]\n\n");
          }
          stream_active->store(false);
        } else {
          auto payload =
              BuildResponse(BuildCompletionBody(result, parsed, chat_mode), 200,
                            "OK", trace_response_header);
          SendAll(session, payload);
        }
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, parsed.model, "no_backend",
                             result.completion);
        }
        return;
      }
      if (metrics_ && !IsBackendEmptyResponse(result)) {
        metrics_->RecordSuccess(result.prompt_tokens, result.completion_tokens);
        metrics_->RecordModelTokens(result.model_id, "", result.prompt_tokens,
                                    result.completion_tokens);
        metrics_->RecordSpeculative(result.speculative.total_chunks,
                                    result.speculative.accepted_chunks,
                                    result.speculative.reused_tokens);
      }
      if (audit_logger_) {
        audit_logger_->LogRequest(auth_ctx.subject, parsed.model, audit_prompt,
                                  result.completion, result.prompt_tokens,
                                  result.completion_tokens);
      }
      // §2.3: detect tool call in model output (used by non-streaming
      // responses).
      ToolCallResult tool_call;
      if (use_tools) {
        bool stub_completion =
            result.no_backend ||
            result.completion.find("No model backend is loaded") !=
                std::string::npos;
        LogToolEvent("tool_state no_backend=" +
                     std::to_string(result.no_backend) + " contains_stub=" +
                     std::to_string(
                         result.completion.find("No model backend is loaded") !=
                         std::string::npos));
        if (stub_completion &&
            (!parsed.tools.empty() || !parsed.first_tool_name.empty())) {
          std::string fallback_name = !parsed.tools.empty()
                                          ? parsed.tools.front().function.name
                                          : parsed.first_tool_name;
          if (fallback_name.empty()) {
            fallback_name = "stub_tool";
          }
          json arguments = {
              {"reason", "no_model_available"},
              {"hint", "set INFERFLUX_MODEL_PATH or configure models[]"}};
          tool_call.detected = true;
          tool_call.function_name = fallback_name;
          tool_call.call_id = "call_stub_" + fallback_name;
          tool_call.arguments_json = arguments.dump();
          std::string log_line = "[tools] stub tool call for " + fallback_name;
          LogToolEvent(log_line);
          std::cout << log_line << std::endl;
          if (audit_logger_) {
            audit_logger_->Log(auth_ctx.subject, parsed.model, "tool_call_stub",
                               arguments.dump());
          }
        } else {
          tool_call = DetectToolCall(result.completion);
        }
      }
      if (parsed.stream) {
        {
          std::lock_guard<std::mutex> lock(*stream_mutex);
          if (stream_active->load()) {
            const std::string stream_finish_reason =
                result.finish_reason_length ? "length" : "stop";
            if (tool_call.detected) {
              // §2.3: emit structured tool_calls delta sequence (role → name →
              // args → finish).
              SendAll(session,
                      BuildToolCallStreamChunks(stream_id, parsed.model,
                                                stream_ts, tool_call));
            } else if (buffer_tokens && !token_buffer->empty()) {
              // Model produced plain text despite tools[] being present (no
              // tool call detected).  Replay the buffered tokens as content
              // deltas.
              for (const auto &tok : *token_buffer) {
                for (const auto &piece : SplitForStreaming(tok)) {
                  SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                    stream_ts, piece, false));
                }
              }
              SendAll(session,
                      BuildStreamChunk(stream_id, parsed.model, stream_ts, "",
                                       true, stream_finish_reason));
            } else {
              SendAll(session,
                      BuildStreamChunk(stream_id, parsed.model, stream_ts, "",
                                       true, stream_finish_reason));
            }
            if (parsed.stream_include_usage) {
              json uc;
              uc["id"] = stream_id;
              uc["object"] = "chat.completion.chunk";
              uc["created"] = stream_ts;
              uc["model"] = parsed.model;
              uc["choices"] = json::array();
              uc["usage"] = {{"prompt_tokens", result.prompt_tokens},
                             {"completion_tokens", result.completion_tokens},
                             {"total_tokens",
                              result.prompt_tokens + result.completion_tokens}};
              SendAll(session, "data: " + uc.dump() + "\n\n");
            }
            SendAll(session, "data: [DONE]\n\n");
          }
        }
        stream_active->store(false);
        return;
      } else {
        auto payload = BuildResponse(
            BuildCompletionBody(result, parsed, chat_mode, tool_call), 200,
            "OK", trace_response_header);
        SendAll(session, payload);
      }
    } catch (const std::exception &ex) {
      if (metrics_) {
        metrics_->RecordError();
      }
      auto payload = BuildResponse(BuildErrorBody(ex.what()), 500, "Error");
      SendAll(session, payload);
      if (audit_logger_) {
        audit_logger_->Log(auth_ctx.subject, parsed.model, "error", ex.what());
      }
    }
    return;
  }

  auto response = BuildResponse(BuildErrorBody("not_found"), 404, "Not Found");
  SendAll(session, response);
  if (audit_logger_) {
    audit_logger_->Log(auth_ctx.subject, "", "not_found", path);
  }
}

bool HttpServer::SendAll(ClientSession &session, const std::string &payload) {
  std::string connection_scoped_payload;
  const std::string *wire_payload = &payload;
  if (!session.keep_alive) {
    constexpr std::string_view kKeepAlive = "Connection: keep-alive\r\n";
    auto pos = payload.find(kKeepAlive);
    if (pos != std::string::npos) {
      connection_scoped_payload = payload;
      connection_scoped_payload.replace(pos, kKeepAlive.size(),
                                        "Connection: close\r\n");
      wire_payload = &connection_scoped_payload;
    }
  }
  const char *data = wire_payload->c_str();
  std::size_t remaining = wire_payload->size();
  while (remaining > 0) {
    int sent = 0;
    if (session.ssl) {
      sent = SSL_write(session.ssl, data, static_cast<int>(remaining));
      if (sent <= 0) {
        int err = SSL_get_error(session.ssl, sent);
        if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
          continue;
        }
        return false;
      }
    } else {
      sent = static_cast<int>(::send(session.fd, data, remaining, 0));
      if (sent <= 0) {
        if (errno == EINTR) {
          continue;
        }
        return false;
      }
    }
    data += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return true;
}

ssize_t HttpServer::Receive(ClientSession &session, char *buffer,
                            std::size_t length) {
  if (session.ssl) {
    while (true) {
      int received = SSL_read(session.ssl, buffer, static_cast<int>(length));
      if (received > 0) {
        return received;
      }
      int err = SSL_get_error(session.ssl, received);
      if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
        continue;
      }
      return -1;
    }
  }
  while (true) {
    ssize_t received = ::recv(session.fd, buffer, length, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    return received;
  }
}

void HttpServer::CloseSession(ClientSession &session) {
  if (session.ssl) {
    SSL_shutdown(session.ssl);
    SSL_free(session.ssl);
    session.ssl = nullptr;
  }
  if (session.fd >= 0) {
    inferflux_close_socket(session.fd);
    session.fd = -1;
  }
}

} // namespace inferflux
