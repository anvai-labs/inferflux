#pragma once

/// @file completion_payload.h
/// @brief Shared types for HTTP request parsing and response building.
///
/// Extracted from http_server.cpp (Phase C2) to enable reuse across
/// completion_payload.cpp and server/http/http_server.cpp.

#include "runtime/multimodal/image_preprocessor.h"
#include "scheduler/request_batch.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

struct ChatMessage {
  std::string role;
  std::string content;
};

// §2.3 — tool/function calling types.
struct ToolFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters; // JSON Schema object (may be null/empty).
};

struct Tool {
  std::string type{"function"}; // Only "function" is supported.
  ToolFunction function;
};

struct ToolCallResult {
  bool detected{false};
  std::string call_id;
  std::string function_name;
  std::string arguments_json; // JSON-encoded arguments object.
};

// Aggregate returned by DetectToolCalls: every extracted call plus the
// visible prose left over after the calls were removed. remaining_text lives
// here (once) rather than duplicated on each call.
struct ToolCallExtraction {
  std::vector<ToolCallResult> calls;
  std::string remaining_text;
};

struct CompletionRequestPayload {
  std::string prompt;
  std::string model{"unknown"};
  std::string session_id;
  std::string client_request_id;
  int max_tokens{256};
  std::vector<ChatMessage> messages;
  bool stream{false};
  bool json_mode{false};   // true when response_format.type == "json_object"
  std::vector<Tool> tools; // §2.3: function definitions available to the model
  std::string first_tool_name;
  bool has_tool_schema{false};
  std::string tool_choice{"auto"}; // "auto" | "none" | "required"
  std::string tool_choice_function;
  bool has_images{false};
  std::vector<DecodedImage> images;
  bool has_response_format{false};
  std::string response_format_type;
  std::string response_format_schema;
  std::string response_format_grammar;
  std::string response_format_root{"root"};
  bool response_format_ok{true};
  std::string response_format_error;
  bool logprobs{false};
  int top_logprobs{0};

  // Sampling parameters.
  float temperature{1.0f};
  float top_p{1.0f};
  int top_k{0};
  float min_p{0.0f};
  float frequency_penalty{0.0f};
  float presence_penalty{0.0f};
  float repetition_penalty{1.0f};
  uint32_t seed{UINT32_MAX};
  std::unordered_map<int, float> logit_bias;
  std::vector<std::string> stop;
  bool stream_include_usage{false};
  int n{1};
  int best_of{1};
};

/// Serialize a response payload while replacing malformed UTF-8 emitted by a
/// model with U+FFFD so one bad token cannot turn a completion into HTTP 500.
std::string SerializeJsonUtf8Safe(const nlohmann::json &payload);

/// Build a JSON error response body.
std::string BuildErrorBody(const std::string &error);

/// Build a complete HTTP response with headers.
std::string BuildResponse(const std::string &body, int status = 200,
                          std::string_view status_text = "OK",
                          const std::string &extra_headers = "");

/// Tool-call extraction (shared by the single, multi, and streaming
/// completion paths; implementation in completion_payload.cpp). Exposed so
/// inferflux_tests (which link inferflux_core) exercise the real
/// implementation instead of a ported copy.
ToolCallExtraction DetectToolCalls(const std::string &text);

/// One OpenAI tool_call entry: {"id","type","function"{name,arguments}}.
/// `index` is included only when engaged (streaming frames carry it).
nlohmann::json BuildToolCallEntry(const ToolCallResult &tc,
                                  std::optional<int> index);

/// SSE frames for a detected tool call batch: role frame, per-call
/// name/arguments frames with tool_call index, single
/// finish_reason="tool_calls" frame.
std::string BuildToolCallStreamChunks(
    const std::string &id, std::string_view model, std::time_t ts,
    const std::vector<ToolCallResult> &tool_calls,
    std::string_view reasoning = {}, std::string_view content = {});

/// Debug log for JSON parse failures (level: debug).
void LogJsonParseFailure(const char *context, const std::exception &ex);

} // namespace inferflux
