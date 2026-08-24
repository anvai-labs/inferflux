#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "runtime/logprob.h"
#include "runtime/multimodal/image_preprocessor.h"
#include "runtime/structured_output/structured_constraint.h"

namespace inferflux {

// Per-request sampling parameters (OpenAI-compatible).
// UINT32_MAX is used for seed because llama.cpp defines
// LLAMA_DEFAULT_SEED == UINT32_MAX, avoiding a transitive llama.h dependency
// in this scheduler-level header.
struct SamplingParams {
  float temperature{1.0f};        // 0 = greedy, >0 = stochastic
  float top_p{1.0f};              // nucleus; 1.0 = disabled
  int top_k{0};                   // 0 = disabled
  float min_p{0.0f};              // 0.0 = disabled
  float frequency_penalty{0.0f};  // OpenAI: penalise frequent tokens
  float presence_penalty{0.0f};   // OpenAI: penalise any prior token
  float repetition_penalty{1.0f}; // multiplicative; 1.0 = disabled
  int penalty_last_n{64};         // lookback window for penalties
  uint32_t seed{UINT32_MAX};      // UINT32_MAX = random per call
  // OpenAI logit_bias: map token IDs to bias values (-100 to 100).
  std::unordered_map<int, float> logit_bias;
};

// InferenceResult surfaced by the scheduler and BatchExecutor to HTTP handlers.
inline constexpr std::string_view kBackendEmptyResponseText =
    "[backend returned empty response]";

struct InferenceResult {
  std::string model_id;
  std::string completion;
  int completion_tokens{0};
  int prompt_tokens{0};
  bool no_backend{false};
  bool finish_reason_length{false};
  std::vector<TokenLogprob> logprobs;

  struct {
    int total_chunks{0};
    int accepted_chunks{0};
    int reused_tokens{0};
  } speculative;
};

inline bool IsBackendEmptyResponse(std::string_view text) {
  return text == kBackendEmptyResponseText;
}

inline bool IsBackendEmptyResponse(const InferenceResult &result) {
  return !result.no_backend && result.completion_tokens == 0 &&
         IsBackendEmptyResponse(result.completion);
}

// Phase of an inference request in the scheduler-driven batching pipeline.
enum class RequestPhase {
  kPending,  // Queued, not yet scheduled.
  kPrefill,  // In prefill (prompt processing).
  kDecode,   // In decode (token generation).
  kFinished, // Generation complete.
  kAborted,  // Cancelled or timed out.
};

// InferenceRequest holds per-request state for the scheduler.
// It replaces the legacy GenerateRequest DTO with a richer structure that
// supports scheduler-driven batching, priority scheduling, and per-request
// accounting.
struct InferenceRequest {
  uint64_t id{0};
  std::string model;          // Requested model ID (empty = default).
  std::string resolved_model; // Assigned model ID after router resolution.
  // Optional upper-layer session handle (feature-flagged server behavior).
  // Empty keeps default stateless OpenAI-compatible semantics.
  std::string session_id;
  std::string client_request_id; // Stable caller-provided trace/debug tag.
  bool session_lease_acquired{false};
  std::string prompt;
  int max_tokens{256};
  int priority{0};         // Higher = more urgent. 0 = default.
  int priority_level{0};   // Cached priority level (for fairness controller).
  int service_tokens{0};   // Tokens consumed so far (fairness accounting).
  int timeslice_tokens{0}; // Timeslice cap when fairness preemption is enabled.
  int remaining_decode_tokens{-1}; // Remaining tokens after fairness slice.
  int reported_prompt_tokens{
      -1}; // Prompt tokens counted once for metrics/logs.
  int total_completion_tokens{
      0}; // Completion tokens accumulated across slices.
  int last_timeslice_tokens{
      0}; // Last applied fairness slice limit (observability).
  bool fairness_yielded{
      false}; // True when the last slice yielded for fairness.
  bool json_mode{false};
  // Phased prefill/decode state (§2.5 Option A).
  // Set by Scheduler::ProcessBatch after calling LlamaCppBackend::Prefill().
  // BatchExecutor::ExecuteRequest() calls Decode() when n_past >= 0 instead of
  // Generate().
  int n_past{-1}; // KV position after prefill; -1 = use legacy Generate() path.
  int sequence_id{-1}; // KV cache sequence slot; -1 = unassigned.
  uint64_t sequence_generation{
      0}; // Generation-stamped lease identity for sequence_id.
  // BPE token count from Prefill() (= pr.n_past at prefill time).  Stored
  // separately because n_past is updated by each Decode() step and by fairness
  // slice rewrites, so by donation time it no longer reflects the prompt
  // length. Used by DonateKVPrefix to record n_kv_tokens for
  // CopySequencePrefix.
  int prompt_bpe_tokens{-1};
  int prefill_offset{0}; // Progress through chunked prefill (§P1d).
  // BPE token IDs produced by LlamaCppBackend::TokenizeForCache() during the
  // prefill block.  Used by LookupKVPrefix / DonateKVPrefix instead of
  // prompt_tokens (SimpleTokenizer) to ensure prefix matching is done in the
  // same BPE-token space as the KV cache, avoiding the boundary mismatch that
  // arises when SimpleTokenizer and llama.cpp disagree on word splits (INF-7).
  std::vector<int> bpe_prompt_tokens;
  // First output token pre-sampled by Prefill() while the logit buffer is
  // fresh.  Carrying it here avoids the logit-buffer race when multiple
  // sequences are prefilled sequentially (each Prefill() overwrites the
  // buffer). -1 = not set (EOS was first token, or fallback Generate() path is
  // used).
  int first_token{-1};
  std::string first_piece; // text of first_token; empty when first_token == -1
  bool has_images{false};  // §2.2: true when request contains image_url parts.
  bool stream{false};
  bool cancelled{false};
  RequestPhase phase{RequestPhase::kPending};
  bool has_response_format{false};
  std::string response_format_type;    // "json_object", "grammar", etc.
  std::string response_format_schema;  // Raw JSON schema string (if provided).
  std::string response_format_grammar; // Resolved GBNF string.
  std::string response_format_root{"root"};
  bool response_format_ready{
      false}; // True once grammar string compiled/resolved.
  // Feature gate for backend/model capability checks. Historically used for
  // response_format, now reused for all request-level feature compatibility.
  bool response_format_supported{true};
  std::string response_format_error;
  StructuredConstraint response_constraint;

  // Persistent execution state for step-wise batch execution.
  // These fields allow BatchExecutor to pause and resume a request at any token
  // step without losing progress.
  bool exec_initialized{false};
  bool exec_active{true};
  int exec_tokens_generated{0};
  int exec_decode_limit{0};
  int exec_current_token{-1};
  bool exec_slice_active{false};
  bool exec_in_prefill{false};
  InferenceResult exec_result{};

  std::vector<DecodedImage>
      images; // §2.2: decoded images (parallel to <__media__> markers).

  // Token state (populated during scheduling).
  std::vector<int> prompt_tokens;
  std::vector<int> output_tokens;

  // Timing for SLO tracking.
  std::chrono::steady_clock::time_point enqueue_time;
  std::chrono::steady_clock::time_point first_token_time;

  // W3C trace-id propagated from the incoming HTTP traceparent header.
  // Empty string if no traceparent was present in the request.
  std::string trace_id;

  // Callback for streaming: invoked with each generated token string and an
  // optional per-token logprob pointer (non-null when collect_logprobs=true).
  // Null for non-streaming requests.
  std::function<void(const std::string &, const TokenLogprob *)> on_token;
  std::string
      accumulated_output; // Aggregated completion text across fairness slices.

  // Shared cancellation flag toggled when the HTTP connection closes.
  std::shared_ptr<std::atomic<bool>> cancellation_flag;
  std::vector<int> block_table; // Logical-to-physical KV block mappings.
  // Distributed prefill/decode handoff retry counter. Incremented when
  // kv_transport->Enqueue rejects a packet; reset on successful enqueue.
  int disagg_enqueue_retries{0};

  // Swapping state (§P1c).
  bool is_swapped{false};
  std::vector<int> swapped_host_handles; // Handles to blocks in Host RAM.

  // Logprob collection (OpenAI logprobs API).
  // collect_logprobs enables per-token logprob recording.
  // logprob_top_n controls how many top alternatives to include (0 = none,
  // just the selected token's logprob; max 20).
  // The two are kept separate so logprobs=true,top_logprobs=0 doesn't trigger
  // the O(V log V) partial-sort for alternatives.
  bool collect_logprobs{false};
  int logprob_top_n{0};

  // OpenAI-compatible sampling parameters. Applied by the backend on every
  // Generate()/Decode() call via SamplerScope in batch_executor.cpp.
  SamplingParams sampling{};

  // OpenAI `stop` parameter: up to 4 strings at which generation halts.
  // The matched stop sequence is NOT included in the completion text.
  std::vector<std::string> stop;
};

// RequestBatch groups requests that will execute together in a single
// forward pass. The scheduler fills batches from the pending queue,
// respecting memory limits and priority order.
//
// This abstraction is the foundation for:
//   - Continuous batching (mix prefill + decode in one batch)
//   - Preemption (evict low-priority requests to make room)
//   - Token accounting (track total tokens in flight)
//   - Disaggregated prefill/decode (tag batches by phase)
struct RequestBatch {
  uint64_t batch_id{0};
  std::vector<InferenceRequest *> requests; // Non-owning pointers.

  // Capacity tracking.
  int total_tokens() const {
    int sum = 0;
    for (const auto *req : requests) {
      sum += static_cast<int>(req->prompt_tokens.size());
      sum += static_cast<int>(req->output_tokens.size());
    }
    return sum;
  }

  bool empty() const { return requests.empty(); }
  std::size_t size() const { return requests.size(); }
};

} // namespace inferflux
