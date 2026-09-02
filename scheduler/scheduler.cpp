#include "scheduler/scheduler.h"

#include "runtime/backends/backend_utils.h"
#include "runtime/execution/batch_executor.h"
#include "runtime/execution/parallel_context.h"
#include "runtime/string_utils.h"
#include "runtime/structured_output/structured_output_adapter.h"
#include "scheduler/model_selection.h"
#include "scheduler/request_requeue.h"
#include "scheduler/single_model_router.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"
#include "server/tracing/span.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_set>

namespace inferflux {

namespace {

bool FairnessTraceEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("INFERFLUX_DEBUG_FAIRNESS");
    return value && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "FALSE";
  }();
  return enabled;
}

bool SequenceSlotDebugEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("INFERFLUX_DEBUG_SEQUENCE_SLOTS");
    return value && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "FALSE";
  }();
  return enabled;
}

bool StickyDecodeAccumulationWaitEnabled() {
  // Cached: this is consulted every DecodeWorkerLoop iteration — a getenv +
  // string compares per token is pure hot-path waste.
  static const bool enabled = []() {
    const char *value =
        std::getenv("INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT");
    return value && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "FALSE";
  }();
  return enabled;
}

std::string RequestPhaseDebugString(RequestPhase phase) {
  switch (phase) {
  case RequestPhase::kPending:
    return "pending";
  case RequestPhase::kPrefill:
    return "prefill";
  case RequestPhase::kDecode:
    return "decode";
  case RequestPhase::kFinished:
    return "finished";
  case RequestPhase::kAborted:
    return "aborted";
  }
  return "unknown";
}

void LogSequenceSlotEvent(std::string_view stage, int64_t request_id,
                          int sequence_id, uint64_t sequence_generation,
                          RequestPhase phase, int n_past,
                          int remaining_decode_tokens,
                          std::string_view detail = {}) {
  if (!SequenceSlotDebugEnabled()) {
    return;
  }
  std::string message =
      "sequence_slot[" + std::string(stage) +
      "]: request_id=" + std::to_string(request_id) +
      ", sequence_id=" + std::to_string(sequence_id) +
      ", sequence_generation=" + std::to_string(sequence_generation) +
      ", phase=" + RequestPhaseDebugString(phase) +
      ", n_past=" + std::to_string(n_past) +
      ", remaining_decode_tokens=" + std::to_string(remaining_decode_tokens);
  if (!detail.empty()) {
    message += ", detail=" + std::string(detail);
  }
  log::Info("scheduler", message);
}

void LogSequenceSlotEvent(std::string_view stage,
                          const InferenceRequest &inference,
                          std::string_view detail = {}) {
  LogSequenceSlotEvent(stage, static_cast<int64_t>(inference.id),
                       inference.sequence_id, inference.sequence_generation,
                       inference.phase, inference.n_past,
                       inference.fairness.remaining_decode_tokens, detail);
}

} // namespace

std::string SchedulerBatchPolicyToString(SchedulerBatchPolicy policy) {
  switch (policy) {
  case SchedulerBatchPolicy::kLpmPriority:
    return "lpm_priority";
  case SchedulerBatchPolicy::kThroughputBalanced:
    return "throughput_balanced";
  case SchedulerBatchPolicy::kPriorityAge:
  default:
    return "priority_age";
  }
}

bool IsSchedulerBatchPolicyValue(const std::string &value) {
  const std::string normalized = ToLower(value);
  return normalized == "priority_age" || normalized == "lpm_priority" ||
         normalized == "throughput_balanced";
}

SchedulerBatchPolicy
ParseSchedulerBatchPolicy(const std::string &value,
                          SchedulerBatchPolicy default_policy) {
  const std::string normalized = ToLower(value);
  if (normalized == "priority_age") {
    return SchedulerBatchPolicy::kPriorityAge;
  }
  if (normalized == "lpm_priority") {
    return SchedulerBatchPolicy::kLpmPriority;
  }
  if (normalized == "throughput_balanced") {
    return SchedulerBatchPolicy::kThroughputBalanced;
  }
  return default_policy;
}

namespace {
// Number of distinct KV sequence slots available for phased prefill/decode.
// Must exceed typical in-flight batch width.  Previous value of 16 caused
// "No free slots available" crashes at concurrency >= 2 with batch_size > 8.
constexpr int kMaxSequenceSlots = 128;
// Warm KV prefix store capacity (dedicated sequence slots for cached prefixes).
constexpr int kPrefixStoreCap = 4;
// Minimum prompt token count to be eligible for prefix store donation.
constexpr int kMinPrefixTokens = 32;
// Tokens per physical KV block (PagedAttention style).
constexpr int kTokensPerBlock = 16;
std::atomic<uint64_t> g_disagg_ticket_seq{0};

bool TokensHavePrefix(const std::vector<int> &tokens,
                      const std::vector<int> &prefix) {
  if (prefix.size() > tokens.size()) {
    return false;
  }
  return std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

std::size_t EstimateQueueTokenCost(const InferenceRequest &inference,
                                   bool from_decode,
                                   const FairnessConfig &fairness_config) {
  // Prefill items are still charged by prompt width; decode items should be
  // charged by per-iteration decode demand to avoid re-charging full prompts on
  // every fairness requeue.
  if (!from_decode) {
    return std::max<std::size_t>(1, inference.prompt_tokens.size());
  }

  int decode_limit = inference.max_tokens;
  if (decode_limit <= 0) {
    decode_limit = 1;
  }
  if (inference.fairness.remaining_decode_tokens >= 0) {
    decode_limit =
        std::min(decode_limit, inference.fairness.remaining_decode_tokens);
  }

  int predicted_slice_limit = inference.fairness.timeslice_tokens;
  // BuildBatchLocked runs before ApplyFairness; predict the same slice cap the
  // fairness pass will apply for lower-priority requests.
  if (predicted_slice_limit <= 0 && fairness_config.max_timeslice_tokens > 0 &&
      inference.priority < fairness_config.high_priority_threshold) {
    predicted_slice_limit = fairness_config.max_timeslice_tokens;
    if (inference.fairness.remaining_decode_tokens > 0) {
      predicted_slice_limit = std::min(
          predicted_slice_limit, inference.fairness.remaining_decode_tokens);
    }
  }
  if (predicted_slice_limit > 0) {
    decode_limit = std::min(decode_limit, predicted_slice_limit);
  }

  return static_cast<std::size_t>(std::max(1, decode_limit));
}

std::vector<uint8_t> SerializeTokens(const std::vector<int> &tokens) {
  std::vector<uint8_t> payload(tokens.size() * sizeof(int));
  if (!tokens.empty()) {
    std::memcpy(payload.data(), tokens.data(), payload.size());
  }
  return payload;
}

void ResetSequenceLease(InferenceRequest *inference) {
  if (!inference) {
    return;
  }
  inference->sequence_id = -1;
  inference->sequence_generation = 0;
}

int ResidentSequenceTokenCount(const InferenceRequest &inference) {
  int token_count = std::max(0, inference.n_past);
  token_count = std::max(token_count, inference.prompt_bpe_tokens);
  token_count = std::max(token_count,
                         static_cast<int>(inference.bpe_prompt_tokens.size()));
  return token_count;
}

bool WaitForUnifiedBatchAsync(
    LlamaCppBackend *backend, UnifiedBatchHandle handle,
    std::vector<UnifiedBatchOutput> *outputs,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  if (!backend || !outputs || handle == 0) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (backend->TryCollectUnifiedBatchAsync(handle, outputs)) {
      return true;
    }
    std::this_thread::yield();
  }
  return backend->TryCollectUnifiedBatchAsync(handle, outputs);
}

struct PrefillStepState {
  int sequence_id{-1};
  int n_past_start{0};
  uint64_t sequence_generation{0};
};

bool ExecutePhasedPrefillStep(LlamaCppBackend *backend,
                              const InferenceRequest &inference,
                              const PrefillStepState &state,
                              PrefillResult *result) {
  const int sequence_id = state.sequence_id;
  const int n_past_start = state.n_past_start;
  if (!backend || !result || sequence_id < 0) {
    return false;
  }
  const auto &prompt_tokens = inference.bpe_prompt_tokens;
  if (prompt_tokens.empty()) {
    return false;
  }
  int bounded_start =
      std::clamp(n_past_start, 0, static_cast<int>(prompt_tokens.size()));
  if (bounded_start >= static_cast<int>(prompt_tokens.size())) {
    // Full-prefix hits must replay at least one token so logits are refreshed
    // for this sequence and first_token sampling remains valid.
    bounded_start = std::max(0, static_cast<int>(prompt_tokens.size()) - 1);
  }
  if (bounded_start == 0) {
    // Ensure reused sequence slots start from a clean KV state on non-prefix
    // phased prefill paths.
    backend->FreeSequence(sequence_id);
  }

  const int token_cap = std::max(1, backend->UnifiedBatchTokenCapacity());
  int chunk_start = bounded_start;
  UnifiedBatchOutput final_output{};
  while (chunk_start < static_cast<int>(prompt_tokens.size())) {
    int chunk_end = std::min(chunk_start + token_cap,
                             static_cast<int>(prompt_tokens.size()));
    std::vector<int> chunk(prompt_tokens.begin() + chunk_start,
                           prompt_tokens.begin() + chunk_end);
    const bool request_logits =
        chunk_end == static_cast<int>(prompt_tokens.size());

    std::vector<UnifiedBatchInput> inputs;
    UnifiedBatchInput input;
    input.sequence_id = sequence_id;
    input.n_past = chunk_start;
    input.tokens = std::move(chunk);
    input.request_logits = request_logits;
    input.sampling = inference.sampling;
    input.request_id = static_cast<int64_t>(inference.id);
    input.sequence_generation = state.sequence_generation;
    inputs.push_back(std::move(input));

    std::vector<UnifiedBatchOutput> outputs;
    if (backend->SupportsAsyncUnifiedBatch()) {
      const auto handle =
          backend->SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kPrefill);
      if (handle == 0 || !WaitForUnifiedBatchAsync(backend, handle, &outputs) ||
          outputs.size() != 1) {
        return false;
      }
    } else {
      outputs = backend->ExecuteUnifiedBatch(inputs);
      if (outputs.size() != 1) {
        return false;
      }
    }

    if (!outputs[0].ok) {
      return false;
    }
    final_output = std::move(outputs[0]);
    chunk_start = chunk_end;
  }

  result->ok = true;
  result->n_past = static_cast<int>(prompt_tokens.size());
  result->first_token = final_output.token;
  result->first_piece = (final_output.token >= 0) ? final_output.piece : "";
  return true;
}

bool HasValidUnifiedStepState(const InferenceRequest &req) {
  if (req.n_past < 0 || req.sequence_id < 0) {
    return false;
  }
  if (req.first_token >= 0) {
    return true;
  }
  if (req.execution.initialized) {
    return true;
  }
  if (req.n_past == 0 && !req.bpe_prompt_tokens.empty()) {
    return true;
  }
  return req.n_past > 0 && !req.bpe_prompt_tokens.empty() &&
         req.prefill_offset > 0 &&
         req.prefill_offset < static_cast<int>(req.bpe_prompt_tokens.size());
}

bool HasBoundDecodeBackend(
    const InferenceRequest &req,
    const std::shared_ptr<LlamaCppBackend> &resolved_backend) {
  return resolved_backend && req.phase == RequestPhase::kDecode &&
         HasValidUnifiedStepState(req);
}

std::string ResolveResultModelId(const InferenceRequest &req) {
  return req.resolved_model.empty() ? req.model : req.resolved_model;
}

bool AppendGeneratedPiece(InferenceRequest *req, std::string_view raw_piece) {
  auto assembled = req->output_utf8.Append(raw_piece);
  GlobalMetrics().RecordOutputUtf8Replacements(assembled.replacements);
  if (assembled.text.empty()) {
    return false;
  }
  req->execution.result.completion.append(assembled.text);
  std::string emit_piece;
  const bool stop_hit = ApplyStop(
      assembled.text, req->execution.result.completion, req->stop, &emit_piece);
  if (req->on_token && !emit_piece.empty()) {
    GlobalMetrics().RecordStreamTokens(1);
    req->on_token(emit_piece, nullptr);
  }
  return stop_hit;
}

void FinishGeneratedText(InferenceRequest *req) {
  auto assembled = req->output_utf8.Finish();
  GlobalMetrics().RecordOutputUtf8Replacements(assembled.replacements);
  if (assembled.text.empty()) {
    return;
  }
  req->execution.result.completion.append(assembled.text);
  if (req->on_token) {
    GlobalMetrics().RecordStreamTokens(1);
    req->on_token(assembled.text, nullptr);
  }
}

void ResetUnifiedStepState(InferenceRequest *req) {
  if (!req) {
    return;
  }
  req->execution.initialized = false;
  req->execution.active = true;
  req->execution.tokens_generated = 0;
  req->execution.non_emitting_steps = 0;
  req->execution.max_non_emitting_steps = 32;
  req->execution.decode_limit = 0;
  req->execution.current_token = -1;
  req->execution.slice_active = false;
  req->execution.in_prefill = false;
  req->execution.result = InferenceResult{};
}

void PrimeUnifiedDecodeStepState(InferenceRequest *req) {
  if (!req || req->execution.initialized) {
    return;
  }

  ResetUnifiedStepState(req);
  req->execution.initialized = true;
  req->execution.result.model_id = ResolveResultModelId(*req);
  req->execution.result.prompt_tokens =
      static_cast<int>(req->prompt_tokens.size());
  req->execution.result.completion = req->accumulated_output;

  const int prior_completion_tokens =
      std::max(0, req->fairness.total_completion_tokens);
  int remaining_decode_tokens = req->fairness.remaining_decode_tokens;
  if (remaining_decode_tokens < 0) {
    remaining_decode_tokens =
        std::max(0, req->max_tokens - prior_completion_tokens);
  }
  req->execution.tokens_generated = prior_completion_tokens;
  req->execution.decode_limit =
      prior_completion_tokens + remaining_decode_tokens;
  req->execution.max_non_emitting_steps = std::max(req->max_tokens * 8, 32);
  req->execution.slice_active = true;

  if (req->n_past >= 0 && req->first_token >= 0) {
    req->execution.current_token = req->first_token;
    const std::string piece = req->first_piece;
    req->first_piece.clear();

    bool stop_hit = false;
    if (!piece.empty()) {
      req->execution.tokens_generated++;
      stop_hit = AppendGeneratedPiece(req, piece);
    } else {
      req->execution.non_emitting_steps++;
    }

    if (stop_hit ||
        (req->cancellation_flag && req->cancellation_flag->load()) ||
        req->execution.non_emitting_steps >=
            req->execution.max_non_emitting_steps ||
        req->execution.tokens_generated >= req->execution.decode_limit) {
      req->execution.active = false;
    }
  } else if (req->n_past == 0 && !req->bpe_prompt_tokens.empty()) {
    req->execution.in_prefill = true;
  } else if (req->n_past > 0 && !req->bpe_prompt_tokens.empty() &&
             req->prefill_offset > 0 &&
             req->prefill_offset <
                 static_cast<int>(req->bpe_prompt_tokens.size())) {
    req->n_past = req->prefill_offset;
    req->execution.in_prefill = true;
  } else {
    req->execution.active = false;
    req->execution.result.completion = "[batch state error]";
  }

  req->accumulated_output = req->execution.result.completion;
  req->fairness.total_completion_tokens = req->execution.tokens_generated;
  req->fairness.remaining_decode_tokens = std::max(
      0, req->execution.decode_limit - req->execution.tokens_generated);
}

bool ShouldContinueUnifiedDecodeStep(const InferenceRequest &req) {
  if (!req.execution.initialized || !req.execution.active) {
    return false;
  }
  if (req.execution.in_prefill) {
    return true;
  }
  return req.execution.tokens_generated < req.execution.decode_limit;
}

void SyncUnifiedDecodeStepProgress(InferenceRequest *req) {
  if (!req || !req->execution.initialized) {
    return;
  }
  req->accumulated_output = req->execution.result.completion;
  req->fairness.total_completion_tokens = req->execution.tokens_generated;
  req->fairness.remaining_decode_tokens = std::max(
      0, req->execution.decode_limit - req->execution.tokens_generated);
  if (ShouldContinueUnifiedDecodeStep(*req)) {
    req->first_token = req->execution.current_token;
    req->first_piece.clear();
  }
}

// Threads per-request usage telemetry (prefix-cache split, accept→last-token
// duration, TTFT for streams) onto a finalized result. Cached tokens are
// clamped to the reported prompt length so a stale radix match can never
// exceed the prompt it is reported against.
void FillResultUsageTelemetry(const InferenceRequest &req,
                              InferenceResult *result) {
  result->cached_prompt_tokens =
      std::clamp(req.cache_matched_tokens, 0, result->prompt_tokens);
  const auto epoch = std::chrono::steady_clock::time_point{};
  if (req.accept_time == epoch) {
    return; // Accept time unavailable; timings stay "not measured".
  }
  const auto now = std::chrono::steady_clock::now();
  result->duration_ms =
      std::chrono::duration<double, std::milli>(now - req.accept_time).count();
  if (req.stream && req.first_token_time != epoch) {
    result->time_to_first_token_ms = std::chrono::duration<double, std::milli>(
                                         req.first_token_time - req.accept_time)
                                         .count();
  }
}

void FinalizeUnifiedDecodeStepResult(InferenceRequest *req,
                                     InferenceResult *result) {
  if (!req || !result) {
    return;
  }

  FinishGeneratedText(req);
  SyncUnifiedDecodeStepProgress(req);
  result->model_id = ResolveResultModelId(*req);
  result->completion = req->accumulated_output;
  result->completion_tokens = req->fairness.total_completion_tokens;
  result->prompt_tokens = req->fairness.reported_prompt_tokens >= 0
                              ? req->fairness.reported_prompt_tokens
                              : static_cast<int>(req->prompt_tokens.size());
  if (result->completion.empty() && !result->no_backend) {
    result->completion = std::string(kBackendEmptyResponseText);
    GlobalMetrics().RecordEmptyGeneration();
  }

  req->fairness.service_tokens = req->fairness.total_completion_tokens;
  req->phase = RequestPhase::kFinished;
  req->fairness.yielded = false;
  req->fairness.timeslice_tokens = 0;
  req->fairness.last_timeslice_tokens = 0;
  ResetUnifiedStepState(req);
}

Scheduler::Config NormalizeSchedulerConfig(const Scheduler::Config &raw) {
  Scheduler::Config normalized = raw;
  if (normalized.max_batch_size <= 0) {
    normalized.max_batch_size = 1;
  }
  if (normalized.max_batch_size > 64) {
    normalized.max_batch_size = 64;
  }
  if (normalized.max_batch_tokens <= 0) {
    normalized.max_batch_tokens = 1;
  }
  if (normalized.max_batch_tokens > 131072) {
    normalized.max_batch_tokens = 131072;
  }
  normalized.decode_burst_tokens = std::max(0, normalized.decode_burst_tokens);
  normalized.chunked_prefill_tokens =
      std::max(1, normalized.chunked_prefill_tokens);
  if (!std::isfinite(normalized.mixed_prefill_budget_ratio)) {
    normalized.mixed_prefill_budget_ratio = 1.0;
  }
  normalized.mixed_prefill_budget_ratio =
      std::clamp(normalized.mixed_prefill_budget_ratio, 0.0, 1.0);
  normalized.session_handles.ttl_ms =
      std::max(1, normalized.session_handles.ttl_ms);
  normalized.session_handles.max_sessions =
      std::max(1, normalized.session_handles.max_sessions);
  return normalized;
}
} // namespace

Scheduler::Scheduler(SimpleTokenizer &tokenizer,
                     std::shared_ptr<CPUDeviceContext> device,
                     std::shared_ptr<PagedKVCache> cache,
                     std::shared_ptr<ModelRouter> router,
                     std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                     std::shared_ptr<RadixPrefixCache> prefix_cache,
                     const FairnessConfig &fairness_config,
                     const DisaggregatedConfig &disagg_config,
                     const ModelSelectionOptions &model_selection_options,
                     Scheduler::Config config)
    : tokenizer_(tokenizer), device_(std::move(device)),
      cache_(std::move(cache)), router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      prefix_cache_(std::move(prefix_cache)),
      fairness_controller_(fairness_config), fairness_config_(fairness_config),
      disagg_config_(disagg_config), config_(NormalizeSchedulerConfig(config)),
      metrics_(config_.metrics ? config_.metrics : &GlobalMetrics()),
      model_selection_options_(model_selection_options) {
  batch_policy_ = CreateBatchSelectionPolicy(config_.batch_policy);

  BatchExecutor::UnifiedBatchTuning tuning;
  tuning.decode_burst_tokens = config_.decode_burst_tokens;
  tuning.chunked_prefill_tokens = config_.chunked_prefill_tokens;
  tuning.mixed_prefill_budget_ratio = config_.mixed_prefill_budget_ratio;
  executor_ =
      std::make_unique<BatchExecutor>(&tokenizer_, device_, cache_, router_,
                                      speculative_decoder_, tuning, metrics_);

  // Initialize sequence slot manager for universal KV cache tracking.
  slot_manager_ =
      std::make_unique<scheduler::SequenceSlotManager>(kMaxSequenceSlots);

  // Enable decode worker pool when a positive pool size is configured.
  // With use_decode_workers_=true, ProcessBatch only runs Prefill and
  // hands off to pending_decode_; decode workers drain that queue.
  use_decode_workers_ = disagg_config_.decode_pool_size > 0;
  if (config_.session_handles.enabled) {
    if (use_decode_workers_) {
      log::Warn("scheduler",
                "Session handles currently support unified scheduler mode "
                "only; disabling session handle feature");
    } else {
      scheduler::SessionHandleManager::Config session_cfg;
      session_cfg.max_sessions =
          static_cast<std::size_t>(config_.session_handles.max_sessions);
      session_cfg.ttl =
          std::chrono::milliseconds(config_.session_handles.ttl_ms);
      session_handle_manager_ =
          std::make_unique<scheduler::SessionHandleManager>(session_cfg);
      log::Info("scheduler",
                "Session handles enabled (ttl_ms=" +
                    std::to_string(config_.session_handles.ttl_ms) +
                    ", max_sessions=" +
                    std::to_string(config_.session_handles.max_sessions) + ")");
    }
  }
  worker_ = std::thread(&Scheduler::WorkerLoop, this);
  if (use_decode_workers_) {
    StartDecodeWorkers();
  }

  // Start eviction worker thread for timeout-based slot cleanup.
  eviction_running_ = true;
  eviction_thread_ = std::thread(&Scheduler::EvictionWorkerLoop, this);

  metrics_->SetSchedulerBatchLimits(config_.max_batch_size,
                                    config_.max_batch_tokens);
  RefreshNativeKvMemoryMetrics();
}

Scheduler::~Scheduler() {
  // Stop eviction worker thread first.
  eviction_running_ = false;
  eviction_cv_.notify_all();
  if (eviction_thread_.joinable()) {
    eviction_thread_.join();
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  prefill_cv_.notify_all();
  decode_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (use_decode_workers_) {
    StopDecodeWorkers();
  }
  if (session_handle_manager_) {
    auto remaining_sessions = session_handle_manager_->DrainAll();
    for (const auto &state : remaining_sessions) {
      ReleaseSessionState(state, nullptr);
    }
  }
  RefreshNativeKvMemoryMetrics();
}

void Scheduler::StartDecodeWorkers() {
  int n = disagg_config_.decode_pool_size;
  if (n <= 0)
    return;
  decode_workers_.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    decode_workers_.emplace_back(&Scheduler::DecodeWorkerLoop, this);
  }
}

void Scheduler::StopDecodeWorkers() {
  for (auto &t : decode_workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
  decode_workers_.clear();
}

int Scheduler::QueueDepth() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return static_cast<int>(pending_prefill_.size() + pending_decode_.size());
}

int Scheduler::LiveDecodeWorkers() const {
  return live_decode_workers_.load(std::memory_order_relaxed);
}

void Scheduler::SyncSequenceSlotProgress(
    const InferenceRequest &request) const {
  if (!slot_manager_ || request.sequence_id < 0) {
    return;
  }
  slot_manager_->UpdateTokenCount(request.sequence_id,
                                  ResidentSequenceTokenCount(request));
}

void Scheduler::RefreshNativeKvMemoryMetrics() const {
  if (!slot_manager_) {
    metrics_->SetInferfluxCudaKvMemoryBytes(/*total_bytes=*/0,
                                            /*active_bytes=*/0,
                                            /*prefix_retained_bytes=*/0,
                                            /*free_bytes=*/0,
                                            /*active_sequences=*/0,
                                            /*prefix_retained_sequences=*/0,
                                            /*free_sequences=*/0,
                                            /*max_sequences=*/0);
    return;
  }

  const uint64_t total_bytes = metrics_->GetInferfluxCudaKvPlannedBytes();
  const int max_sequences = metrics_->GetInferfluxCudaKvMaxSequences();
  if (total_bytes == 0 || max_sequences <= 0) {
    metrics_->SetInferfluxCudaKvMemoryBytes(/*total_bytes=*/0,
                                            /*active_bytes=*/0,
                                            /*prefix_retained_bytes=*/0,
                                            /*free_bytes=*/0,
                                            /*active_sequences=*/0,
                                            /*prefix_retained_sequences=*/0,
                                            /*free_sequences=*/0,
                                            /*max_sequences=*/0);
    return;
  }

  int active_sequences = 0;
  int prefix_retained_sequences = 0;
  for (const auto &slot : slot_manager_->GetSlotStatus()) {
    if (slot.state == scheduler::SequenceState::kPrefilling ||
        slot.state == scheduler::SequenceState::kDecoding) {
      ++active_sequences;
    } else if (slot.state == scheduler::SequenceState::kCompleted) {
      ++prefix_retained_sequences;
    }
  }
  active_sequences = std::min(active_sequences, max_sequences);
  prefix_retained_sequences =
      std::min(prefix_retained_sequences, max_sequences - active_sequences);
  const int free_sequences =
      std::max(0, max_sequences - active_sequences - prefix_retained_sequences);

  const uint64_t bytes_per_sequence =
      total_bytes / static_cast<uint64_t>(std::max(1, max_sequences));
  const uint64_t active_bytes =
      bytes_per_sequence * static_cast<uint64_t>(active_sequences);
  const uint64_t prefix_retained_bytes =
      bytes_per_sequence * static_cast<uint64_t>(prefix_retained_sequences);
  const uint64_t free_bytes =
      total_bytes >= active_bytes + prefix_retained_bytes
          ? total_bytes - active_bytes - prefix_retained_bytes
          : 0;

  metrics_->SetInferfluxCudaKvMemoryBytes(
      total_bytes, active_bytes, prefix_retained_bytes, free_bytes,
      active_sequences, prefix_retained_sequences, free_sequences,
      max_sequences);
}

void Scheduler::UpdateModelSelectionOptions(
    const ModelSelectionOptions &options) {
  std::lock_guard<std::mutex> lock(model_selection_options_mutex_);
  model_selection_options_ = options;
}

ModelSelectionOptions Scheduler::ModelSelectionOptionsSnapshot() const {
  std::lock_guard<std::mutex> lock(model_selection_options_mutex_);
  return model_selection_options_;
}

bool Scheduler::RequestUsesSessionHandle(
    const InferenceRequest &request) const {
  return session_handle_manager_ != nullptr && !request.session_id.empty();
}

bool Scheduler::BackendUsesSplitDecodeWorkers(
    const std::shared_ptr<LlamaCppBackend> &backend) const {
  return use_decode_workers_ && backend != nullptr &&
         backend->SupportsSplitPrefillDecodeHandoff();
}

bool Scheduler::CanAppendToStickyStepBatchLocked(
    const std::shared_ptr<PendingRequest> &pending,
    const std::shared_ptr<LlamaCppBackend> &sticky_step_backend) const {
  if (!pending || !sticky_step_backend || !pending->resolved_backend) {
    return false;
  }
  const auto &inference = pending->inference;
  return pending->resolved_backend.get() == sticky_step_backend.get() &&
         HasValidUnifiedStepState(inference) &&
         !inference.response_format.constraint.has_grammar &&
         !inference.collect_logprobs && !inference.response_format.has_format &&
         inference.response_format.supported;
}

std::size_t Scheduler::AppendCompatiblePendingDecodeLocked(
    std::vector<std::shared_ptr<PendingRequest>> *batch,
    const std::shared_ptr<LlamaCppBackend> &sticky_step_backend,
    std::size_t max_batch_size) {
  if (!batch || !sticky_step_backend || batch->size() >= max_batch_size ||
      pending_decode_.empty()) {
    return 0;
  }

  // Scan the whole decode queue so a request for another backend at the front
  // does not prevent later compatible requests from joining the active sticky
  // cohort. Preserve the relative order of compatible requests that do join.
  std::size_t merged = 0;
  for (auto it = pending_decode_.begin();
       it != pending_decode_.end() && batch->size() < max_batch_size;) {
    if (!CanAppendToStickyStepBatchLocked(*it, sticky_step_backend)) {
      ++it;
      continue;
    }
    batch->push_back(*it);
    ++merged;
    it = pending_decode_.erase(it);
  }
  if (merged > 0) {
    metrics_->RecordDecodeWorkerStickyMerge(merged);
  }
  return merged;
}

std::size_t Scheduler::CountCompatiblePendingDecodeLocked(
    const std::shared_ptr<LlamaCppBackend> &sticky_step_backend) const {
  if (!sticky_step_backend || pending_decode_.empty()) {
    return 0;
  }
  std::size_t compatible = 0;
  for (const auto &pending : pending_decode_) {
    if (CanAppendToStickyStepBatchLocked(pending, sticky_step_backend)) {
      ++compatible;
    }
  }
  return compatible;
}

void Scheduler::DecodeWorkerLoop() {
  // Register as a live worker; deregister on any exit path (stop signal or
  // uncaught exception), so HttpServer::LiveDecodeWorkers() accurately reflects
  // pool health rather than the static startup flag.
  live_decode_workers_.fetch_add(1, std::memory_order_relaxed);
  struct LiveGuard {
    std::atomic<int> &counter;
    ~LiveGuard() { counter.fetch_sub(1, std::memory_order_relaxed); }
  } live_guard{live_decode_workers_};

  std::vector<std::shared_ptr<PendingRequest>> batch;
  std::shared_ptr<LlamaCppBackend> sticky_step_backend;

  while (true) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      constexpr auto kDisaggTransportPollInterval =
          std::chrono::milliseconds(10);
      const std::size_t accumulation_target =
          config_.min_batch_size <= 1
              ? 1
              : std::max<std::size_t>(
                    2, static_cast<std::size_t>(config_.min_batch_size));
      const bool sticky_accumulation_wait_enabled =
          StickyDecodeAccumulationWaitEnabled();

      // Batch accumulation for decode workers
      auto has_transport_work = [&] {
        return disagg_config_.kv_transport &&
               disagg_config_.kv_transport->Size() > 0;
      };
      auto has_work = [&] {
        return stop_ || !batch.empty() || !pending_decode_.empty() ||
               has_transport_work();
      };
      auto has_min_decode_batch = [&] {
        if (stop_) {
          return true;
        }
        if (!batch.empty()) {
          return true;
        }
        return pending_decode_.size() >= accumulation_target;
      };

      if (config_.batch_accumulation_ms > 0) {
        if (disagg_config_.kv_transport) {
          while (!stop_ && batch.empty() && pending_decode_.empty() &&
                 !has_transport_work()) {
            decode_cv_.wait_for(lock, kDisaggTransportPollInterval);
          }
          if (!stop_ && batch.empty() && !pending_decode_.empty()) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.batch_accumulation_ms);
            while (!stop_) {
              if (has_transport_work()) {
                break;
              }
              if (has_min_decode_batch()) {
                break;
              }
              if (std::chrono::steady_clock::now() >= deadline) {
                break;
              }
              decode_cv_.wait_for(lock, kDisaggTransportPollInterval);
            }
          }
        } else {
          decode_cv_.wait(lock, has_work);
          if (!stop_ && batch.empty() && !pending_decode_.empty()) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.batch_accumulation_ms);
            while (!stop_ && !has_min_decode_batch()) {
              if (decode_cv_.wait_until(lock, deadline) ==
                  std::cv_status::timeout) {
                break;
              }
            }
          }
        }
      } else if (disagg_config_.kv_transport) {
        while (!stop_ && batch.empty() && pending_decode_.empty() &&
               !has_transport_work()) {
          decode_cv_.wait_for(lock, kDisaggTransportPollInterval);
        }
      } else {
        decode_cv_.wait(lock, has_work);
      }

      if (stop_ && batch.empty() && pending_decode_.empty() &&
          !has_transport_work()) {
        break;
      }

      const std::size_t max_batch_size =
          static_cast<std::size_t>(config_.max_batch_size);
      if (batch.empty()) {
        const std::size_t ready_before = pending_decode_.size();
        std::size_t n = std::min(ready_before, max_batch_size);
        batch.assign(pending_decode_.begin(),
                     pending_decode_.begin() + static_cast<std::ptrdiff_t>(n));
        pending_decode_.erase(pending_decode_.begin(),
                              pending_decode_.begin() +
                                  static_cast<std::ptrdiff_t>(n));
        metrics_->RecordDecodeAssemblySnapshot("worker_initial", ready_before,
                                               batch.size(), 0, 0);
      } else if (batch.size() < max_batch_size && sticky_step_backend) {
        const std::size_t ready_before = pending_decode_.size();
        const std::size_t compatible_before =
            CountCompatiblePendingDecodeLocked(sticky_step_backend);
        std::size_t merged = AppendCompatiblePendingDecodeLocked(
            &batch, sticky_step_backend, max_batch_size);
        if (sticky_accumulation_wait_enabled &&
            config_.batch_accumulation_ms > 0 && merged == 0 &&
            batch.size() < max_batch_size &&
            batch.size() < accumulation_target) {
          auto compatible_pending_decode = [&]() {
            return std::any_of(
                pending_decode_.begin(), pending_decode_.end(),
                [&](const std::shared_ptr<PendingRequest> &pending) {
                  return CanAppendToStickyStepBatchLocked(pending,
                                                          sticky_step_backend);
                });
          };
          const auto deadline =
              std::chrono::steady_clock::now() +
              std::chrono::milliseconds(config_.batch_accumulation_ms);
          while (!stop_ && !has_transport_work() &&
                 !compatible_pending_decode() &&
                 std::chrono::steady_clock::now() < deadline) {
            if (decode_cv_.wait_until(lock, deadline) ==
                std::cv_status::timeout) {
              break;
            }
          }
          if (!stop_ && batch.size() < max_batch_size) {
            AppendCompatiblePendingDecodeLocked(&batch, sticky_step_backend,
                                                max_batch_size);
          }
        }
        const std::size_t incompatible_before =
            ready_before >= compatible_before ? ready_before - compatible_before
                                              : 0;
        metrics_->RecordDecodeAssemblySnapshot("worker_sticky", ready_before,
                                               batch.size(), compatible_before,
                                               incompatible_before);
      }
      UpdateQueueDepthLocked();
    }

    // If a KV transport is configured, drain cross-process packets that may
    // have arrived from remote prefill workers.  For the in-process case the KV
    // state is already resident in the llama context and no hydration is
    // needed.
    if (disagg_config_.kv_transport) {
      while (auto pkt = disagg_config_.kv_transport->TryDequeue()) {
        if (pkt->ticket_id > 0) {
          disagg_config_.kv_transport->UpdateTicketStage(
              pkt->ticket_id, disaggregated::KVTicketStage::kAcknowledged);
          metrics_->RecordDisaggKVTicketStage(
              disaggregated::KVTicketStageToString(
                  disaggregated::KVTicketStage::kAcknowledged));
        }
        // Record KV transfer latency (§2.5 item 12): time from Enqueue to
        // dequeue.
        auto dequeue_time = std::chrono::steady_clock::now();
        double transfer_ms = std::chrono::duration<double, std::milli>(
                                 dequeue_time - pkt->enqueue_time)
                                 .count();
        metrics_->RecordKVTransfer(transfer_ms);

        // Match the packet to a decode request by request_id and hydrate its
        // KV state. Decode workers remove requests from pending_decode_ into
        // the local `batch` before draining the transport, so we must inspect
        // both collections or packets for the active batch will time out
        // spuriously.
        auto try_apply_packet = [&](std::shared_ptr<PendingRequest> &pending) {
          if (!pending || pending->inference.id != pkt->request_id ||
              pkt->n_past < 0) {
            return false;
          }

          // Packet carries a serialised KV blob; hydrate it now.
          // (blob is currently empty for in-process path — no-op when empty.)
          // kv_blob holds raw bytes from LlamaCppBackend::SerializeSequence.
          // Pass it directly to HydrateSequence — no cast or size adjustment.
          if (!pkt->kv_blob.empty() && pending->resolved_backend) {
            uint64_t seq_generation = 0;
            int seq_id = AllocSeqSlot(static_cast<int64_t>(pending->sequence),
                                      &seq_generation);
            if (seq_id >= 0) {
              if (pending->resolved_backend->HydrateSequence(seq_id,
                                                             pkt->kv_blob)) {
                pending->inference.sequence_id = seq_id;
                pending->inference.sequence_generation = seq_generation;
                pending->inference.n_past = pkt->n_past;
                return true;
              }
              FreeSeqSlot(seq_id, seq_generation);
            }
            return false;
          }
          return true;
        };

        // All AllocSeqSlot/FreeSeqSlot calls must be protected by
        // queue_mutex_ to prevent concurrent sequence slot operations
        // that corrupt llama.cpp internal state (double-free at c≥16).
        bool ticket_committed = false;
        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          for (auto &pending : batch) {
            if (try_apply_packet(pending)) {
              ticket_committed = true;
              break;
            }
          }
          if (!ticket_committed) {
            for (auto &pending : pending_decode_) {
              if (try_apply_packet(pending)) {
                ticket_committed = true;
                break;
              }
            }
          }
          if (!ticket_committed) {
            for (auto &pending : publishing_decode_) {
              if (try_apply_packet(pending)) {
                ticket_committed = true;
                break;
              }
            }
          }
        }
        if (pkt->ticket_id > 0) {
          const auto ticket_stage =
              ticket_committed ? disaggregated::KVTicketStage::kCommitted
                               : disaggregated::KVTicketStage::kTimedOut;
          disagg_config_.kv_transport->UpdateTicketStage(pkt->ticket_id,
                                                         ticket_stage);
          metrics_->RecordDisaggKVTicketStage(
              disaggregated::KVTicketStageToString(ticket_stage));
        }
      }
    }

    if (batch.empty())
      continue;

    metrics_->RecordDecodeWorkerBatchSize(batch.size());

    std::size_t decode_tokens = 0;
    for (const auto &pending : batch) {
      int slice_tokens = pending->inference.fairness.timeslice_tokens;
      if (slice_tokens <= 0) {
        if (pending->inference.fairness.remaining_decode_tokens > 0) {
          slice_tokens = pending->inference.fairness.remaining_decode_tokens;
        } else {
          slice_tokens = pending->inference.max_tokens;
        }
      }
      decode_tokens += static_cast<std::size_t>(std::max(1, slice_tokens));
    }
    metrics_->RecordSchedulerIteration(
        /*prefill_requests=*/0,
        /*decode_requests=*/batch.size(), decode_tokens);
    metrics_->RecordSchedulerPolicyIteration(
        SchedulerBatchPolicyToString(config_.batch_policy),
        /*prefill_requests=*/0, /*decode_requests=*/batch.size());

    auto resolve_direct_step_backend =
        [&]() -> std::shared_ptr<LlamaCppBackend> {
      if (config_.decode_burst_tokens != 0) {
        return nullptr;
      }
      std::shared_ptr<LlamaCppBackend> direct_backend;
      for (const auto &pending : batch) {
        if (!pending) {
          return nullptr;
        }
        const auto &inference = pending->inference;
        const auto &backend = pending->resolved_backend;
        if (!HasBoundDecodeBackend(inference, backend) || !backend ||
            !backend->IsReady() || !inference.response_format.supported ||
            inference.response_format.constraint.has_grammar ||
            inference.collect_logprobs ||
            inference.response_format.has_format) {
          return nullptr;
        }
        if (sticky_step_backend && backend.get() != sticky_step_backend.get()) {
          return nullptr;
        }
        if (!direct_backend) {
          direct_backend = backend;
        } else if (direct_backend.get() != backend.get()) {
          return nullptr;
        }
      }
      return direct_backend;
    };

    if (auto direct_step_backend = resolve_direct_step_backend()) {
      metrics_->RecordDecodeWorkerExecutionPath("direct_stepwise");
      const bool native_direct_step_backend =
          direct_step_backend->Name() == "inferflux_cuda";
      if (native_direct_step_backend) {
        metrics_->RecordDecodeWorkerExecutionPath("direct_stepwise_native");
      }

      RequestBatch step_batch;
      step_batch.batch_id =
          next_batch_id_.fetch_add(1, std::memory_order_relaxed);
      step_batch.requests.reserve(batch.size());

      for (auto &pending : batch) {
        auto *inference = &pending->inference;
        if (slot_manager_ && inference->sequence_id >= 0) {
          slot_manager_->MarkProcessing(inference->sequence_id);
        }
        PrimeUnifiedDecodeStepState(inference);
        if (ShouldContinueUnifiedDecodeStep(*inference)) {
          step_batch.requests.push_back(inference);
        } else {
          SyncUnifiedDecodeStepProgress(inference);
          SyncSequenceSlotProgress(*inference);
        }
      }

      if (!step_batch.empty()) {
        executor_->ExecuteUnifiedBatchStep(step_batch, direct_step_backend);
      }

      std::vector<std::shared_ptr<PendingRequest>> active_requeue;
      active_requeue.reserve(batch.size());
      for (auto &pending : batch) {
        auto *inference = &pending->inference;
        InferenceResult result;
        SyncUnifiedDecodeStepProgress(inference);
        SyncSequenceSlotProgress(*inference);
        if (ShouldContinueUnifiedDecodeStep(*inference)) {
          LogSequenceSlotEvent("decode_step_requeue", *inference,
                               "decode_worker");
          active_requeue.push_back(pending);
          continue;
        }

        FinalizeUnifiedDecodeStepResult(inference, &result);
        FillResultUsageTelemetry(*inference, &result);
        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          if (inference->session_lease_acquired &&
              RequestUsesSessionHandle(*inference)) {
            FinalizeSessionLease(pending.get(), !result.no_backend);
          } else if (inference->sequence_id >= 0) {
            if (cache_) {
              cache_->ReleaseBlocksRef(inference->block_table);
            }
            inference->block_table.clear();
            FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                        pending->resolved_backend);
            ResetSequenceLease(inference);
          }
        }
        pending->promise.set_value(std::move(result));
      }

      batch = std::move(active_requeue);
      sticky_step_backend =
          batch.empty() ? nullptr : std::move(direct_step_backend);
      continue;
    }

    ResolveBackends(batch);

    RequestBatch exec_batch;
    exec_batch.batch_id =
        next_batch_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::shared_ptr<LlamaCppBackend>> overrides;
    std::vector<std::shared_ptr<PendingRequest>> exec_pending;
    overrides.reserve(batch.size());
    exec_pending.reserve(batch.size());

    for (auto &pending : batch) {
      auto *inference = &pending->inference;
      if (!inference->response_format.supported) {
        InferenceResult error;
        error.no_backend = true;
        error.completion =
            inference->response_format.error.empty()
                ? "Selected model does not support requested features"
                : inference->response_format.error;
        pending->promise.set_value(std::move(error));
        // Return the sequence slot and KV blocks so they are not leaked.
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (inference->session_lease_acquired &&
            RequestUsesSessionHandle(*inference)) {
          FinalizeSessionLease(pending.get(), false);
        } else if (inference->sequence_id >= 0) {
          if (cache_) {
            cache_->ReleaseBlocksRef(inference->block_table);
          }
          inference->block_table.clear();
          FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                      pending->resolved_backend);
          ResetSequenceLease(inference);
        }
        continue;
      }
      if (inference->response_format.has_format &&
          !inference->response_format.ready) {
        StructuredConstraint constraint;
        std::string adapter_error;
        if (!StructuredOutputAdapter::BuildConstraint(
                inference->response_format.type,
                inference->response_format.schema,
                inference->response_format.grammar,
                inference->response_format.root, &constraint, &adapter_error)) {
          InferenceResult error;
          error.no_backend = true;
          error.completion = adapter_error.empty()
                                 ? "response_format could not be converted"
                                 : adapter_error;
          pending->promise.set_value(std::move(error));
          std::lock_guard<std::mutex> lock(queue_mutex_);
          if (inference->session_lease_acquired &&
              RequestUsesSessionHandle(*inference)) {
            FinalizeSessionLease(pending.get(), false);
          } else if (inference->sequence_id >= 0) {
            if (cache_) {
              cache_->ReleaseBlocksRef(inference->block_table);
            }
            inference->block_table.clear();
            FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                        pending->resolved_backend);
            ResetSequenceLease(inference);
          }
          continue;
        }
        inference->response_format.constraint = constraint;
        inference->response_format.ready = true;
      }
      if (slot_manager_ && inference->sequence_id >= 0) {
        slot_manager_->MarkProcessing(inference->sequence_id);
      }
      exec_pending.push_back(pending);
      overrides.push_back(pending->resolved_backend);
      exec_batch.requests.push_back(inference);
    }

    if (exec_pending.empty())
      continue;

    metrics_->RecordDecodeWorkerExecutionPath("general");

    bool use_stepwise_decode = config_.decode_burst_tokens == 0;
    std::shared_ptr<LlamaCppBackend> step_backend;
    if (use_stepwise_decode) {
      for (std::size_t i = 0; i < exec_pending.size(); ++i) {
        auto *inference = &exec_pending[i]->inference;
        const auto &backend = overrides[i];
        if (!backend || !backend->IsReady() ||
            !HasValidUnifiedStepState(*inference) ||
            inference->response_format.constraint.has_grammar ||
            inference->collect_logprobs ||
            inference->response_format.has_format) {
          use_stepwise_decode = false;
          break;
        }
        if (!step_backend) {
          step_backend = backend;
        } else if (step_backend.get() != backend.get()) {
          use_stepwise_decode = false;
          break;
        }
      }
    }

    std::vector<InferenceResult> responses;
    if (use_stepwise_decode) {
      const bool native_step_backend =
          step_backend && step_backend->Name() == "inferflux_cuda";
      if (native_step_backend) {
        metrics_->RecordDecodeWorkerExecutionPath("direct_stepwise_native");
      }
      RequestBatch step_batch;
      step_batch.batch_id = exec_batch.batch_id;
      step_batch.requests.reserve(exec_pending.size());
      for (auto &pending : exec_pending) {
        auto *inference = &pending->inference;
        PrimeUnifiedDecodeStepState(inference);
        if (ShouldContinueUnifiedDecodeStep(*inference)) {
          step_batch.requests.push_back(inference);
        } else {
          SyncUnifiedDecodeStepProgress(inference);
          SyncSequenceSlotProgress(*inference);
        }
      }
      if (!step_batch.empty()) {
        executor_->ExecuteUnifiedBatchStep(step_batch, step_backend);
      }
      for (auto &pending : exec_pending) {
        SyncUnifiedDecodeStepProgress(&pending->inference);
        SyncSequenceSlotProgress(pending->inference);
      }
    } else {
      bool native_backend_decode = !overrides.empty();
      for (const auto &backend : overrides) {
        if (!backend || backend->Name() != "inferflux_cuda") {
          native_backend_decode = false;
          break;
        }
      }
      if (native_backend_decode) {
        metrics_->RecordDecodeWorkerExecutionPath(
            "general_native_backend_decode");
      }
      responses = executor_->ExecuteBatch(exec_batch, overrides);
    }

    std::vector<std::shared_ptr<PendingRequest>> active_requeue;
    active_requeue.reserve(exec_pending.size());
    std::vector<std::shared_ptr<PendingRequest>> queue_requeue;
    queue_requeue.reserve(exec_pending.size());

    for (std::size_t i = 0; i < exec_pending.size(); ++i) {
      auto &pending = exec_pending[i];
      auto *inference = &pending->inference;
      InferenceResult result;
      if (!use_stepwise_decode && i < responses.size()) {
        result = std::move(responses[i]);
        SyncSequenceSlotProgress(*inference);
      }

      if (use_stepwise_decode) {
        if (ShouldContinueUnifiedDecodeStep(*inference)) {
          LogSequenceSlotEvent("decode_step_requeue", *inference,
                               "decode_worker");
          active_requeue.push_back(pending);
          continue;
        }
        FinalizeUnifiedDecodeStepResult(inference, &result);
      } else if (inference->fairness.yielded) {
        auto now = std::chrono::steady_clock::now();
        pending->enqueue_time = now;
        PrepareFairnessDecodeRequeue(inference, now);
        LogSequenceSlotEvent("fairness_requeue", *inference, "decode_worker");
        queue_requeue.push_back(pending);
        continue;
      }

      if (!use_stepwise_decode && !inference->accumulated_output.empty()) {
        result.completion = inference->accumulated_output;
        if (inference->fairness.total_completion_tokens > 0) {
          result.completion_tokens =
              inference->fairness.total_completion_tokens;
        }
      }
      if (!use_stepwise_decode &&
          inference->fairness.reported_prompt_tokens >= 0) {
        result.prompt_tokens = inference->fairness.reported_prompt_tokens;
      }

      // Return KV memory and sequence slot.  Both must happen together.
      // ExecuteRequest intentionally does NOT call FreeSequence — it leaves
      // sequence_id set so the scheduler (here or in ProcessBatch) handles
      // both FreeSequence and FreeSeqSlot together in one place.
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (inference->session_lease_acquired &&
            RequestUsesSessionHandle(*inference)) {
          FinalizeSessionLease(pending.get(), !result.no_backend);
        } else if (inference->sequence_id >= 0) {
          if (cache_) {
            cache_->ReleaseBlocksRef(inference->block_table);
          }
          inference->block_table.clear();
          FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                      pending->resolved_backend);
          ResetSequenceLease(inference);
        }
      }
      if (!use_stepwise_decode) {
        inference->phase = RequestPhase::kFinished;
      }
      FillResultUsageTelemetry(*inference, &result);
      pending->promise.set_value(std::move(result));
    }

    batch = std::move(active_requeue);
    if (batch.empty()) {
      sticky_step_backend.reset();
    }

    if (!queue_requeue.empty()) {
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto &pending : queue_requeue) {
          pending_decode_.push_back(pending);
        }
        UpdateQueueDepthLocked();
      }
      decode_cv_.notify_one();
    }

    if (!use_stepwise_decode) {
      batch.clear();
      sticky_step_backend.reset();
    } else {
      sticky_step_backend = batch.empty() ? nullptr : step_backend;
    }
  }
}

std::future<InferenceResult> Scheduler::Generate(InferenceRequest request) {
  auto pending = std::make_shared<PendingRequest>();
  pending->inference = std::move(request);
  pending->priority = pending->inference.priority;
  pending->sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed);
  pending->enqueue_time = std::chrono::steady_clock::now();
  pending->inference.id = pending->sequence;
  pending->inference.phase = RequestPhase::kPending;
  pending->inference.session_lease_acquired = false;
  pending->inference.enqueue_time = pending->enqueue_time;
  pending->inference.accept_time = pending->enqueue_time;
  if (pending->inference.max_tokens <= 0) {
    pending->inference.max_tokens = 1;
  }
  pending->inference.fairness.remaining_decode_tokens =
      pending->inference.max_tokens;
  pending->inference.accumulated_output.clear();
  ResetUnifiedStepState(&pending->inference);
  if (pending->inference.prompt_tokens.empty()) {
    pending->inference.prompt_tokens =
        tokenizer_.Encode(pending->inference.prompt);
  }
  pending->inference.fairness.reported_prompt_tokens =
      static_cast<int>(pending->inference.prompt_tokens.size());
  pending->inference.output_tokens.clear();
  pending->inference.first_token_time = {};
  pending->inference.phase = RequestPhase::kPending;

  auto future = pending->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    pending_prefill_.push_back(pending);
    UpdateQueueDepthLocked();
  }
  prefill_cv_.notify_one();
  return future;
}

void Scheduler::UpdateFairnessConfig(const FairnessConfig &config) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    fairness_config_ = config;
    fairness_controller_.UpdateConfig(config);
  }
}

void Scheduler::WorkerLoop() {
  auto &pc = ParallelContext::Get();
  bool is_distributed = pc.IsInitialized() && pc.WorldSize() > 1;

  while (true) {
    BatchSelection selection;

    if (is_distributed && !pc.IsMaster()) {
      // §P1g: Worker ranks wait for the master's batch decision.
      std::vector<int> request_ids;
      std::vector<int> phases;
      if (!pc.ReceiveBatch(request_ids, phases)) {
        if (stop_)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // In a real implementation we'd reconstruct 'selection' from active
      // queues. For this foundation, we just Barrier to keep the worker rank
      // aligned with the master's processing cadence.
      pc.Comm()->Barrier();
    } else {
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        const std::size_t accumulation_target =
            config_.min_batch_size <= 1
                ? 1
                : std::max<std::size_t>(
                      2, static_cast<std::size_t>(config_.min_batch_size));

        // Batch accumulation: wait for minimum batch size or timeout
        // to improve GPU utilization. Trade-off: slightly higher latency
        // for significantly better throughput.
        auto has_work = [&] {
          return stop_ || !pending_prefill_.empty() ||
                 (!use_decode_workers_ && !pending_decode_.empty());
        };

        auto has_min_batch = [&] {
          if (stop_) {
            return true;
          }
          std::size_t total_pending = pending_prefill_.size();
          if (!use_decode_workers_) {
            total_pending += pending_decode_.size();
          }
          return total_pending >= accumulation_target;
        };

        if (config_.batch_accumulation_ms > 0) {
          prefill_cv_.wait(lock, has_work);
          if (!stop_) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.batch_accumulation_ms);
            while (!stop_ && !has_min_batch()) {
              if (prefill_cv_.wait_until(lock, deadline) ==
                  std::cv_status::timeout) {
                break;
              }
            }
          }
        } else {
          // No batch accumulation: wait for any work immediately
          prefill_cv_.wait(lock, has_work);
        }

        if (stop_ && pending_prefill_.empty() && pending_decode_.empty()) {
          break;
        }
        selection = BuildBatchLocked();
      }

      if (is_distributed) {
        // Broadcast the decision to all workers.
        std::vector<int> ids;
        std::vector<int> phases;
        for (auto &p : selection.pending) {
          ids.push_back(static_cast<int>(p->inference.id));
          phases.push_back(static_cast<int>(p->inference.phase));
        }
        pc.BroadcastBatch(ids, phases);
        pc.Comm()->Barrier(); // Wait for workers to catch up
      }
    }

    if (!selection.pending.empty()) {
      ProcessBatch(std::move(selection));
    }
  }
}

Scheduler::BatchSelection Scheduler::BuildBatchLocked() {
  BatchSelection selection;
  const std::size_t decode_available_before = pending_decode_.size();
  if (pending_prefill_.empty() && pending_decode_.empty()) {
    return selection;
  }

  struct QueueItem {
    std::shared_ptr<PendingRequest> pending;
    bool from_decode{false};
    std::size_t index{0};
    double prefix_affinity_tokens{0.0};
  };
  std::vector<QueueItem> queue_items;
  std::vector<std::size_t> duplicate_remove_prefill;
  std::vector<std::size_t> duplicate_remove_decode;
  std::unordered_set<PendingRequest *> seen_pending;
  // When decode workers are active they own pending_decode_; WorkerLoop must
  // only build prefill batches to avoid both threads competing for the same
  // requests and for the seq slot free-list (which is not separately locked).
  const bool decode_workers_own_decode = use_decode_workers_;
  queue_items.reserve(pending_prefill_.size() +
                      (decode_workers_own_decode ? 0 : pending_decode_.size()));
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    auto *pending_ptr = pending_prefill_[i].get();
    if (!seen_pending.insert(pending_ptr).second) {
      duplicate_remove_prefill.push_back(i);
      log::Warn(
          "scheduler",
          "Deduplicating pending request from prefill queue: request_id=" +
              std::to_string(pending_prefill_[i]->inference.id));
      continue;
    }
    queue_items.push_back(QueueItem{pending_prefill_[i], false, i});
  }
  if (!decode_workers_own_decode) {
    for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
      auto *pending_ptr = pending_decode_[i].get();
      if (!seen_pending.insert(pending_ptr).second) {
        duplicate_remove_decode.push_back(i);
        log::Warn(
            "scheduler",
            "Deduplicating pending request from decode queue: request_id=" +
                std::to_string(pending_decode_[i]->inference.id));
        continue;
      }
      queue_items.push_back(QueueItem{pending_decode_[i], true, i});
    }
  }

  const bool prefix_affinity_enabled =
      batch_policy_->UsesPrefixAffinity() && prefix_cache_ != nullptr;
  if (prefix_affinity_enabled) {
    for (auto &item : queue_items) {
      if (item.from_decode || !item.pending) {
        continue;
      }
      auto &inf = item.pending->inference;
      const std::vector<int> *tokens = !inf.bpe_prompt_tokens.empty()
                                           ? &inf.bpe_prompt_tokens
                                           : &inf.prompt_tokens;
      if (!tokens || tokens->empty()) {
        continue;
      }

      std::shared_ptr<LlamaCppBackend> backend_hint =
          item.pending->resolved_backend;
      if (!backend_hint && router_) {
        ModelInfo *resolved = nullptr;
        if (!inf.resolved_model.empty()) {
          resolved = router_->ResolveExact(inf.resolved_model);
        }
        if (!resolved && !inf.model.empty()) {
          resolved = router_->ResolveExact(inf.model);
        }
        if (!resolved) {
          resolved = router_->Resolve(inf.model);
        }
        if (resolved && !resolved->id.empty()) {
          backend_hint = std::static_pointer_cast<LlamaCppBackend>(
              router_->GetBackend(resolved->id));
        }
      }

      RadixLookupResult lookup;
      const bool hit =
          prefix_cache_->Lookup(*tokens, backend_hint.get(), &lookup);
      const bool affinity_hit = hit && lookup.matched_tokens > 0;
      metrics_->RecordPrefixAffinityProbe(
          affinity_hit, affinity_hit ? lookup.matched_tokens : 0);
      if (affinity_hit) {
        item.prefix_affinity_tokens =
            static_cast<double>(lookup.matched_tokens);
      }
    }
  }

  auto now = std::chrono::steady_clock::now();
  const auto &policy = *batch_policy_;
  std::stable_sort(queue_items.begin(), queue_items.end(),
                   [&](const QueueItem &a, const QueueItem &b) {
                     ScoringItem sa{std::chrono::duration<double, std::milli>(
                                        now - a.pending->enqueue_time)
                                        .count(),
                                    a.pending->priority, a.pending->sequence,
                                    a.prefix_affinity_tokens};
                     ScoringItem sb{std::chrono::duration<double, std::milli>(
                                        now - b.pending->enqueue_time)
                                        .count(),
                                    b.pending->priority, b.pending->sequence,
                                    b.prefix_affinity_tokens};
                     double eff_a = policy.Score(sa);
                     double eff_b = policy.Score(sb);
                     if (eff_a != eff_b) {
                       return eff_a > eff_b;
                     }
                     return a.pending->sequence < b.pending->sequence;
                   });

  std::vector<std::size_t> to_remove_prefill;
  std::vector<std::size_t> to_remove_decode;
  std::size_t token_budget = 0;
  std::size_t decode_selected = 0;
  for (std::size_t i = 0; i < queue_items.size(); ++i) {
    auto &item = queue_items[i].pending;
    std::size_t tokens = EstimateQueueTokenCost(
        item->inference, queue_items[i].from_decode, fairness_config_);
    if (!selection.pending.empty() &&
        token_budget + tokens >
            static_cast<std::size_t>(config_.max_batch_tokens)) {
      metrics_->RecordBatchTokenBudgetSkip();
      continue;
    }

    // Handle swapped-out requests (§P1c).
    if (item->inference.is_swapped) {
      if (!TrySwapIn(item->inference)) {
        // Skip for this batch if we can't swap back in yet.
        continue;
      }
    }

    selection.pending.push_back(item);
    selection.batch.requests.push_back(&item->inference);
    item->inference.phase = queue_items[i].from_decode ? RequestPhase::kDecode
                                                       : RequestPhase::kPrefill;
    token_budget += tokens;
    selection.total_tokens += tokens;
    if (queue_items[i].from_decode) {
      to_remove_decode.push_back(queue_items[i].index);
      ++decode_selected;
    } else {
      to_remove_prefill.push_back(queue_items[i].index);
    }
    if (selection.pending.size() >=
        static_cast<std::size_t>(config_.max_batch_size)) {
      break;
    }
  }

  auto erase_indices = [](auto *queue, std::vector<std::size_t> indices) {
    if (!queue || indices.empty()) {
      return;
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (std::size_t idx = indices.size(); idx-- > 0;) {
      queue->erase(queue->begin() + static_cast<std::ptrdiff_t>(indices[idx]));
    }
  };

  to_remove_prefill.insert(to_remove_prefill.end(),
                           duplicate_remove_prefill.begin(),
                           duplicate_remove_prefill.end());
  to_remove_decode.insert(to_remove_decode.end(),
                          duplicate_remove_decode.begin(),
                          duplicate_remove_decode.end());
  erase_indices(&pending_prefill_, std::move(to_remove_prefill));
  erase_indices(&pending_decode_, std::move(to_remove_decode));
  if (!decode_workers_own_decode && decode_available_before > 0) {
    metrics_->RecordDecodeAssemblySnapshot("unified", decode_available_before,
                                           decode_selected, 0, 0);
  }
  UpdateQueueDepthLocked();
  return selection;
}

void Scheduler::ProcessBatch(BatchSelection selection) {
  if (selection.pending.empty()) {
    return;
  }

  std::size_t prefill_requests = 0;
  std::size_t decode_requests = 0;
  for (const auto &pending : selection.pending) {
    if (pending->inference.phase == RequestPhase::kPrefill) {
      ++prefill_requests;
    } else {
      ++decode_requests;
    }
  }

  metrics_->SetDecodeQueueDepth(static_cast<int>(selection.pending.size()));
  {
    // ApplyFairness iterates/mutates the pending queues under its calling
    // contract; ProcessBatch runs outside the BuildBatchLocked lock scope,
    // so take queue_mutex_ here to pair with Generate()/DecodeWorkerLoop.
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    ApplyFairness(&selection);
  }
  ResolveBackends(selection.pending);
  auto batch_start = std::chrono::steady_clock::now();
  selection.batch.batch_id =
      next_batch_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::shared_ptr<PendingRequest>> staged_decode_local;
  staged_decode_local.reserve(selection.pending.size());
  std::vector<std::shared_ptr<PendingRequest>> staged_decode_worker;
  staged_decode_worker.reserve(selection.pending.size());
  std::vector<std::shared_ptr<PendingRequest>> decode_ready;
  decode_ready.reserve(selection.pending.size());
  for (auto &pending : selection.pending) {
    if (pending->inference.phase == RequestPhase::kPrefill) {
      // Option A phased prefill: run prompt evaluation on the local backend now
      // so that the decode slice (ExecuteRequest) can call Decode() from n_past
      // instead of Generate().
      auto &inf = pending->inference;
      if (inf.sequence_id >= 0 && inf.n_past >= 0 && inf.first_token >= 0) {
        log::Warn("scheduler",
                  "Prefill queue contained decode-ready request; routing "
                  "directly to decode request_id=" +
                      std::to_string(inf.id) +
                      ", sequence_id=" + std::to_string(inf.sequence_id));
        if (BackendUsesSplitDecodeWorkers(pending->resolved_backend)) {
          staged_decode_worker.push_back(pending);
        } else {
          staged_decode_local.push_back(pending);
        }
        continue;
      }
      if (pending->resolved_backend && pending->resolved_backend->IsReady()) {
        // Compute BPE tokens for prefix matching (INF-7).  We do this once per
        // request: if bpe_prompt_tokens is already populated (e.g., a retry
        // after channel-full rejection), reuse the cached result.
        if (inf.bpe_prompt_tokens.empty()) {
          inf.bpe_prompt_tokens =
              pending->resolved_backend->TokenizeForCache(inf.prompt);
        }
        // Keep logprobs/structured-output requests on the full Generate()
        // path so the backend can apply a single consistent sampler/grammar
        // state. Phased prefill + Decode() would split sampler state across
        // phases and force sequence-state handoff between heterogeneous paths.
        if (inf.collect_logprobs || inf.response_format.has_format) {
          inf.n_past = -1;
          inf.prompt_bpe_tokens = 0;
          ResetSequenceLease(&inf);
          inf.first_token = -1;
          inf.first_piece.clear();
          inf.prefill_offset = 0;
          staged_decode_local.push_back(pending);
          continue;
        }
        // KV prefix reuse (§ Item 5): if we have a warm sequence whose BPE
        // tokens are a strict prefix of this request's BPE prompt tokens, copy
        // that KV state and only evaluate the suffix — skipping repeated prompt
        // computation.  Matching on BPE tokens ensures the prefix boundary
        // aligns with the KV cache (fixes INF-7 SimpleTokenizer mismatch).
        // KV prefix reuse (§P1b): lookup in the global Radix Trie to find
        // physical KV blocks matching this request's prefix.
        std::vector<int> cached_blocks;
        int matched_tokens = 0;
        int cached_seq_id = -1;
        uint64_t cached_seq_generation = 0;
        bool prefix_hit = false;
        bool reused_session_state = false;
        if (RequestUsesSessionHandle(inf) && !inf.session_lease_acquired) {
          auto lease = session_handle_manager_->AcquireLease(inf.session_id);
          for (const auto &cleanup_state : lease.cleanup_states) {
            ReleaseSessionState(cleanup_state, nullptr);
          }
          if (lease.status ==
              scheduler::SessionHandleManager::LeaseResult::Status::kAcquired) {
            inf.session_lease_acquired = true;
            if (lease.has_state) {
              const std::string resolved_model =
                  inf.resolved_model.empty() ? inf.model : inf.resolved_model;
              const bool model_compatible =
                  lease.state.model_id.empty() || resolved_model.empty() ||
                  lease.state.model_id == resolved_model;
              const bool prompt_compatible = TokensHavePrefix(
                  inf.bpe_prompt_tokens, lease.state.prompt_tokens);
              if (model_compatible && prompt_compatible &&
                  lease.state.sequence_id >= 0) {
                prefix_hit = true;
                reused_session_state = true;
                cached_seq_id = lease.state.sequence_id;
                cached_seq_generation = lease.state.sequence_generation;
                matched_tokens =
                    static_cast<int>(lease.state.prompt_tokens.size());
                cached_blocks = lease.state.block_table;
              } else {
                ReleaseSessionState(lease.state, pending->resolved_backend);
                session_handle_manager_->DiscardLeasedState(inf.session_id);
              }
            }
          }
        } else if (!RequestUsesSessionHandle(inf)) {
          inf.session_lease_acquired = false;
        }
        if (!reused_session_state && prefix_cache_) {
          RadixLookupResult lookup;
          prefix_hit = prefix_cache_->Lookup(
              inf.bpe_prompt_tokens, pending->resolved_backend.get(), &lookup);
          cached_blocks = std::move(lookup.block_table);
          cached_seq_id = lookup.sequence_id;
          matched_tokens = lookup.matched_tokens;
        }
        // Per-request usage telemetry (usage.cached_tokens): only tokens
        // whose KV blocks are actually reused count as cached — a partial
        // trie match that cannot donate blocks reports zero.
        inf.cache_matched_tokens = prefix_hit ? matched_tokens : 0;

        // PagedAttention Block Allocation: calculate additional blocks needed.
        std::size_t prompt_len = inf.bpe_prompt_tokens.size();

        // Logical blocks already covered by the cached prefix.
        std::size_t warm_blocks = cached_blocks.size();
        std::size_t total_blocks_needed =
            (prompt_len + kTokensPerBlock - 1) / kTokensPerBlock;
        total_blocks_needed += 1;

        std::size_t new_blocks_needed =
            (total_blocks_needed > warm_blocks)
                ? (total_blocks_needed - warm_blocks)
                : 0;

        std::vector<int> new_blocks;
        if (cache_) {
          if (cache_->NumFreeBlocks() < new_blocks_needed) {
            // Memory tight (§P1c): attempt to swap out the lowest priority
            // request from the currently selected batch to make room.
            for (int j = static_cast<int>(selection.pending.size()) - 1; j >= 0;
                 --j) {
              auto &victim = selection.pending[j];
              if (victim != pending && !victim->inference.block_table.empty()) {
                if (TrySwapOut(victim->inference)) {
                  break; // Room potentially made.
                }
              }
            }
          }

          if (cache_->NumFreeBlocks() >= new_blocks_needed) {
            try {
              if (new_blocks_needed > 0) {
                new_blocks = cache_->ReserveBlocks(new_blocks_needed);
              }
            } catch (const std::exception &) {
              new_blocks.clear();
            }
          }
        }

        uint64_t seq_generation =
            reused_session_state ? cached_seq_generation : 0;
        int seq_id = reused_session_state
                         ? cached_seq_id
                         : AllocSeqSlot(static_cast<int64_t>(pending->sequence),
                                        &seq_generation);
        // Admission logic (§ Item 4): can admit if we have a seq slot AND
        // (no paged cache configured OR new blocks were successfully reserved).
        bool can_admit = (seq_id >= 0) && (!cache_ || new_blocks_needed == 0 ||
                                           !new_blocks.empty());
        bool queued_via_unified_prefill = false;

        if (can_admit) {
          if (reused_session_state && slot_manager_ && seq_generation != 0) {
            const bool restored = slot_manager_->RestoreLease(
                {seq_id, seq_generation,
                 static_cast<int64_t>(pending->sequence)},
                static_cast<int64_t>(pending->sequence), seq_id,
                std::max(matched_tokens,
                         static_cast<int>(inf.bpe_prompt_tokens.size())));
            if (!restored) {
              log::Warn("scheduler",
                        "Failed to restore retained sequence lease for slot " +
                            std::to_string(seq_id) +
                            " gen=" + std::to_string(seq_generation));
              can_admit = false;
            } else {
              RefreshNativeKvMemoryMetrics();
            }
          }
        }

        if (can_admit) {
          if (cache_) {
            // Prefix cache warm blocks need a scheduler ref. Session-owned
            // warm blocks already have a dedicated lease-owned ref.
            if (!reused_session_state) {
              cache_->AcquireBlocks(cached_blocks);
            }
            inf.block_table = std::move(cached_blocks);
            inf.block_table.insert(inf.block_table.end(), new_blocks.begin(),
                                   new_blocks.end());
          }

          const bool can_defer_to_unified_prefill =
              !use_decode_workers_ && pending->resolved_backend &&
              pending->resolved_backend->SupportsAsyncUnifiedBatch() &&
              !inf.collect_logprobs && !inf.response_format.has_format &&
              !inf.has_images && !inf.bpe_prompt_tokens.empty();
          if (can_defer_to_unified_prefill) {
            int prefill_start = 0;
            bool copied_prefix = false;
            if (prefix_hit && matched_tokens > 0 && cached_seq_id >= 0) {
              // Correctness Fix (§ Item 3): must copy KV state into new slot!
              // matched_tokens is clamped to the current prompt length before
              // use.
              prefill_start =
                  std::clamp(matched_tokens, 0,
                             static_cast<int>(inf.bpe_prompt_tokens.size()));
              if (!inf.bpe_prompt_tokens.empty() &&
                  prefill_start >=
                      static_cast<int>(inf.bpe_prompt_tokens.size())) {
                // Replay one suffix token on exact hits to refresh logits and
                // avoid zero-token decode continuations.
                prefill_start =
                    static_cast<int>(inf.bpe_prompt_tokens.size()) - 1;
              }
              if (!reused_session_state) {
                pending->resolved_backend->CopySequencePrefix(
                    cached_seq_id, seq_id, prefill_start);
              }
              copied_prefix = true;
            }

            // Defer prefill compute into ExecuteUnifiedBatchPhased so decode
            // steps can overlap with in-flight prefill chunks in the same
            // batch.
            inf.prefill_offset = prefill_start;
            inf.n_past = prefill_start;
            inf.prompt_bpe_tokens =
                static_cast<int>(inf.bpe_prompt_tokens.size());
            inf.sequence_id = seq_id;
            inf.sequence_generation = seq_generation;
            inf.first_token = -1;
            inf.first_piece.clear();
            SyncSequenceSlotProgress(inf);
            if (copied_prefix && prefill_start > 0) {
              metrics_->RecordKVPrefixReuse(prefill_start);
            }
            staged_decode_local.push_back(pending);
            queued_via_unified_prefill = true;
          } else {
            PrefillResult pr;
            bool copied_prefix = false;
            int prefill_start = 0;
            if (prefix_hit && matched_tokens > 0 && cached_seq_id >= 0) {
              // Correctness Fix (§ Item 3): must copy KV state into new slot!
              // matched_tokens is clamped to the current prompt length before
              // use.
              prefill_start =
                  std::clamp(matched_tokens, 0,
                             static_cast<int>(inf.bpe_prompt_tokens.size()));
              if (!inf.bpe_prompt_tokens.empty() &&
                  prefill_start >=
                      static_cast<int>(inf.bpe_prompt_tokens.size())) {
                // Replay one suffix token on exact hits to refresh logits and
                // avoid zero-token decode continuations.
                prefill_start =
                    static_cast<int>(inf.bpe_prompt_tokens.size()) - 1;
              }
              if (!reused_session_state) {
                pending->resolved_backend->CopySequencePrefix(
                    cached_seq_id, seq_id, prefill_start);
              }
              copied_prefix = true;
            }

            bool prefill_ok = ExecutePhasedPrefillStep(
                pending->resolved_backend.get(), inf,
                {seq_id, prefill_start, seq_generation}, &pr);
            if (!prefill_ok) {
              if (copied_prefix) {
                pr = pending->resolved_backend->PrefillPartial(
                    inf.prompt, seq_id, prefill_start);
              } else {
                // Misses and stale-prefix hits both fall back to full prefill.
                pr = pending->resolved_backend->Prefill(inf.prompt, seq_id);
              }
            }
            if (pr.ok && copied_prefix) {
              metrics_->RecordKVPrefixReuse(prefill_start);
            }

            if (pr.ok) {
              inf.n_past = pr.n_past;
              inf.prompt_bpe_tokens = pr.n_past;
              inf.sequence_id = seq_id;
              inf.sequence_generation = seq_generation;
              inf.first_token = pr.first_token;
              inf.first_piece = pr.first_piece;
              SyncSequenceSlotProgress(inf);
            } else {
              // Prefill failed: release only the NEW blocks (warm blocks belong
              // to the cache).
              if (reused_session_state) {
                if (cache_) {
                  cache_->ReleaseBlocksRef(inf.block_table);
                }
                if (inf.session_lease_acquired) {
                  session_handle_manager_->DiscardLeasedState(inf.session_id);
                }
                FreeSeqSlot(seq_id, seq_generation, pending->resolved_backend);
              } else if (cache_) {
                cache_->ReleaseBlocks(new_blocks);
              }
              inf.block_table.clear();
              ResetSequenceLease(&inf);
              inf.n_past = -1;
            }
          }
        } else if (seq_id >= 0 && !reused_session_state) {
          FreeSeqSlot(seq_id, seq_generation);
        }
        if (queued_via_unified_prefill) {
          continue;
        }
      }
      const bool use_split_decode_workers =
          BackendUsesSplitDecodeWorkers(pending->resolved_backend);
      bool enqueued = false;
      bool published_to_decode_worker = false;
      if (use_split_decode_workers && disagg_config_.kv_transport) {
        // Only enqueue when decode workers are live and draining the channel.
        // Without active consumers the channel fills to capacity (default 64)
        // and Enqueue() returns false, causing requests to bounce back into
        // pending_prefill_ indefinitely — a scheduler deadlock.
        const bool process_local_handoff =
            pending->resolved_backend &&
            pending->resolved_backend->SupportsProcessLocalSequenceTransfer() &&
            disagg_config_.kv_transport->IsProcessLocal();
        disaggregated::KVPacket packet;
        packet.request_id = inf.id;
        packet.prompt_tokens = SerializeTokens(inf.prompt_tokens);
        // Process-local native split handoff keeps ownership of the existing
        // sequence slot. Cross-process transports still hydrate from a blob.
        if (inf.sequence_id >= 0 && pending->resolved_backend &&
            !process_local_handoff) {
          packet.kv_blob =
              pending->resolved_backend->SerializeSequence(inf.sequence_id);
        }
        packet.ticket_id =
            g_disagg_ticket_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        packet.ticket_stage = disaggregated::KVTicketStage::kEnqueued;
        packet.n_past = inf.n_past;
        packet.sequence_id = inf.sequence_id;
        packet.metadata = inf.model;
        // Register the request before Enqueue() can make its packet visible.
        // Keep transport I/O outside queue_mutex_: ShmKVTransport may copy a
        // large KV blob, and unrelated scheduler work must remain unblocked.
        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          publishing_decode_.push_back(pending);
        }
        enqueued = disagg_config_.kv_transport->Enqueue(std::move(packet));
        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          auto publication = std::find(publishing_decode_.begin(),
                                       publishing_decode_.end(), pending);
          if (publication != publishing_decode_.end()) {
            publishing_decode_.erase(publication);
          }
          if (enqueued) {
            inf.phase = RequestPhase::kDecode;
            pending_decode_.push_back(pending);
            UpdateQueueDepthLocked();
            published_to_decode_worker = true;
          }
        }
        if (published_to_decode_worker) {
          decode_cv_.notify_one();
        }
      } else {
        enqueued = true;
      }
      if (enqueued) {
        inf.disagg_enqueue_retries = 0;
        if (use_split_decode_workers) {
          metrics_->RecordDisaggKVTicketStage(
              disaggregated::KVTicketStageToString(
                  disaggregated::KVTicketStage::kEnqueued));
          if (!published_to_decode_worker) {
            staged_decode_worker.push_back(pending);
          }
        } else {
          staged_decode_local.push_back(pending);
        }
        continue;
      } else {
        inf.disagg_enqueue_retries += 1;
        const bool retries_exhausted =
            inf.disagg_enqueue_retries > disagg_config_.kv_enqueue_max_retries;
        metrics_->RecordDisaggKVEnqueueRejected(retries_exhausted);

        // Channel rejected the packet (full).  Undo any phased state so the
        // request retries cleanly and does not hold a stale slot or stale
        // n_past.
        if (retries_exhausted) {
          if (inf.session_lease_acquired && session_handle_manager_) {
            FinalizeSessionLease(pending.get(), false);
          } else {
            if (inf.sequence_id >= 0) {
              FreeSeqSlot(inf.sequence_id, inf.sequence_generation,
                          pending->resolved_backend);
              ResetSequenceLease(&inf);
              inf.n_past = -1;
            }
            if (cache_ && !inf.block_table.empty()) {
              cache_->ReleaseBlocksRef(inf.block_table);
              inf.block_table.clear();
            }
          }
          InferenceResult error;
          error.no_backend = true;
          error.completion = "distributed_overloaded: kv transport enqueue "
                             "retries exhausted";
          pending->promise.set_value(std::move(error));
          continue;
        }
        if (inf.sequence_id >= 0) {
          FreeSeqSlot(inf.sequence_id, inf.sequence_generation,
                      pending->resolved_backend);
          ResetSequenceLease(&inf);
          inf.n_past = -1;
        }
        if (cache_ && !inf.block_table.empty()) {
          cache_->ReleaseBlocksRef(inf.block_table);
          inf.block_table.clear();
        }
        inf.phase = RequestPhase::kPending;
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_prefill_.push_back(pending);
      }
    } else {
      decode_ready.push_back(pending);
    }
  }

  if (!staged_decode_worker.empty()) {
    if (use_decode_workers_) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto &pending : staged_decode_worker) {
        pending->inference.phase = RequestPhase::kDecode;
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
      decode_cv_.notify_one();
    } else {
      for (auto &pending : staged_decode_worker) {
        pending->inference.phase = RequestPhase::kDecode;
        decode_ready.push_back(pending);
      }
    }
  }
  for (auto &pending : staged_decode_local) {
    pending->inference.phase = RequestPhase::kDecode;
    decode_ready.push_back(pending);
  }

  std::size_t metrics_decode_requests = decode_requests;
  if (!use_decode_workers_) {
    // In unified (single-worker) mode staged decode requests are executed in
    // the same ProcessBatch iteration, so count them toward mixed-phase
    // accounting.
    metrics_decode_requests = decode_ready.size();
  }
  metrics_->RecordSchedulerIteration(prefill_requests, metrics_decode_requests,
                                     selection.total_tokens);
  metrics_->RecordSchedulerPolicyIteration(
      SchedulerBatchPolicyToString(config_.batch_policy), prefill_requests,
      metrics_decode_requests);

  if (use_decode_workers_) {
    // Split mode: decode workers mutate pending_decode_ under queue_mutex_;
    // read the depth under the same lock (formal UB otherwise, and a torn
    // count is a misleading metric).
    int decode_depth = 0;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      decode_depth = static_cast<int>(pending_decode_.size());
    }
    metrics_->SetDecodeQueueDepth(decode_depth);
  }
  if (use_decode_workers_ && decode_ready.empty()) {
    return;
  }

  if (decode_ready.empty()) {
    metrics_->SetDecodeQueueDepth(0);
    return;
  }

  selection.pending = decode_ready;
  for (auto &req : selection.pending) {
    double wait_ms = std::chrono::duration<double, std::milli>(
                         batch_start - req->enqueue_time)
                         .count();
    metrics_->RecordQueueLatency(wait_ms);
  }

  if (selection.total_tokens == 0) {
    for (const auto *req : selection.batch.requests) {
      selection.total_tokens += req->prompt_tokens.size();
      selection.total_tokens += req->output_tokens.size();
    }
  }
  metrics_->RecordBatch(selection.batch.requests.size(),
                        selection.total_tokens);

  RequestBatch exec_batch;
  exec_batch.batch_id = selection.batch.batch_id;
  std::vector<std::shared_ptr<LlamaCppBackend>> overrides;
  std::vector<std::shared_ptr<PendingRequest>> exec_pending;
  overrides.reserve(selection.pending.size());
  exec_pending.reserve(selection.pending.size());
  for (auto &pending : selection.pending) {
    auto *inference = &pending->inference;
    if (!inference->response_format.supported) {
      InferenceResult error;
      error.no_backend = true;
      error.completion =
          inference->response_format.error.empty()
              ? "Selected model does not support requested features"
              : inference->response_format.error;
      if (inference->session_lease_acquired &&
          RequestUsesSessionHandle(*inference)) {
        FinalizeSessionLease(pending.get(), false);
      } else if (inference->sequence_id >= 0) {
        if (cache_) {
          cache_->ReleaseBlocksRef(inference->block_table);
        }
        inference->block_table.clear();
        FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                    pending->resolved_backend);
        ResetSequenceLease(inference);
      }
      pending->promise.set_value(std::move(error));
      continue;
    }
    if (inference->response_format.has_format) {
      StructuredConstraint constraint;
      std::string adapter_error;
      if (!StructuredOutputAdapter::BuildConstraint(
              inference->response_format.type,
              inference->response_format.schema,
              inference->response_format.grammar,
              inference->response_format.root, &constraint, &adapter_error)) {
        InferenceResult error;
        error.no_backend = true;
        error.completion = adapter_error.empty()
                               ? "response_format could not be converted"
                               : adapter_error;
        if (inference->session_lease_acquired &&
            RequestUsesSessionHandle(*inference)) {
          FinalizeSessionLease(pending.get(), false);
        } else if (inference->sequence_id >= 0) {
          if (cache_) {
            cache_->ReleaseBlocksRef(inference->block_table);
          }
          inference->block_table.clear();
          FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                      pending->resolved_backend);
          ResetSequenceLease(inference);
        }
        pending->promise.set_value(std::move(error));
        continue;
      }
      inference->response_format.constraint = constraint;
      inference->response_format.ready = true;
    } else {
      inference->response_format.constraint = StructuredConstraint{};
      inference->response_format.ready = false;
    }
    if (slot_manager_ && inference->sequence_id >= 0) {
      slot_manager_->MarkProcessing(inference->sequence_id);
    }
    exec_pending.push_back(pending);
    overrides.push_back(pending->resolved_backend);
    exec_batch.requests.push_back(&pending->inference);
  }

  if (exec_pending.empty()) {
    metrics_->SetDecodeQueueDepth(0);
    return;
  }

  auto responses = executor_->ExecuteBatch(exec_batch, overrides);
  std::vector<std::shared_ptr<PendingRequest>> requeue;
  requeue.reserve(exec_pending.size());
  for (std::size_t i = 0; i < exec_pending.size(); ++i) {
    auto &pending = exec_pending[i];
    auto *inference = &pending->inference;
    InferenceResult result;
    if (i < responses.size()) {
      result = std::move(responses[i]);
    }
    if (inference->fairness.yielded) {
      const int slice_tokens = result.completion_tokens;
      auto now = std::chrono::steady_clock::now();
      pending->enqueue_time = now;
      PrepareFairnessDecodeRequeue(inference, now);
      LogSequenceSlotEvent("fairness_requeue", *inference, "worker_loop");
      SpanContext parent_ctx;
      parent_ctx.trace_id = inference->trace_id;
      Span fairness_span(
          "scheduler.fairness.yield", tracing::ChildContext(parent_ctx),
          [req_id = inference->id,
           remaining = inference->fairness.remaining_decode_tokens,
           slice = inference->fairness.last_timeslice_tokens, slice_tokens](
              const std::string &name, const SpanContext &ctx, double ms) {
            if (!FairnessTraceEnabled()) {
              return;
            }
            std::cout << "[fairness] " << name << " request=" << req_id
                      << " trace=" << ctx.trace_id << " limit=" << slice
                      << " generated=" << slice_tokens
                      << " remaining=" << remaining << " duration_ms=" << ms
                      << std::endl;
          });
      fairness_span.Finish();
      requeue.push_back(pending);
      continue;
    }
    if (!inference->accumulated_output.empty()) {
      result.completion = inference->accumulated_output;
      if (inference->fairness.total_completion_tokens > 0) {
        result.completion_tokens = inference->fairness.total_completion_tokens;
      }
    }
    if (inference->fairness.reported_prompt_tokens >= 0) {
      result.prompt_tokens = inference->fairness.reported_prompt_tokens;
    }
    // Return the llama.cpp KV memory and the sequence slot to the pool.
    // ExecuteRequest intentionally leaves sequence_id set so both operations
    // happen here, co-located.
    // KV prefix donation: when the request was a non-yielded phased decode on
    // the local worker (no decode workers), attempt to retain the sequence slot
    // as a warm prefix entry so future requests sharing this prompt prefix can
    // skip re-evaluating those tokens.
    // KV prefix cache (§P1b): insert the completed request's blocks into
    // the global Radix Trie to enable zero-copy reuse for future requests.
    bool donated = false;
    const bool session_owned = inference->session_lease_acquired &&
                               RequestUsesSessionHandle(*inference);
    if (prefix_cache_ && inference->sequence_id >= 0 &&
        !inference->fairness.yielded && !session_owned) {
      // Concatenate prompt BPE tokens and any generated output BPE tokens.
      // (Simplified: we use prompt_bpe_tokens for the architectural
      // foundation).
      if (static_cast<int>(inference->bpe_prompt_tokens.size()) >=
          kMinPrefixTokens) {
        LogSequenceSlotEvent("prefix_donation_begin", *inference);
        if (cache_)
          cache_->AcquireBlocks(
              inference->block_table); // Node ownership (§P1b)
        LogSequenceSlotEvent("prefix_donation_blocks_acquired", *inference);
        prefix_cache_->Insert(inference->bpe_prompt_tokens,
                              inference->block_table, inference->sequence_id,
                              pending->resolved_backend);
        LogSequenceSlotEvent("prefix_donation_end", *inference);
        donated = true;
      }
    }

    if (session_owned) {
      FinalizeSessionLease(pending.get(), !result.no_backend);
    } else if (inference->sequence_id >= 0) {
      // Correctness Fix (§ Item 2): Do NOT FreeSequence if we donated!
      // The trie now owns this sequence's KV state until it is evicted.
      if (!donated) {
        FreeSeqSlot(inference->sequence_id, inference->sequence_generation,
                    pending->resolved_backend);
      }
      // Correctness Fix (§ Item 2): always release scheduler's reference!
      // If donated, prefix_cache already called AcquireBlocks, so ref_count
      // >= 1.
      if (cache_)
        cache_->ReleaseBlocksRef(inference->block_table);
      LogSequenceSlotEvent("scheduler_blocks_released", *inference);
      inference->block_table.clear();
      ResetSequenceLease(inference);
    }
    inference->phase = RequestPhase::kFinished;
    FillResultUsageTelemetry(*inference, &result);
    pending->promise.set_value(std::move(result));
    LogSequenceSlotEvent("promise_completed", *inference);
  }

  if (!requeue.empty()) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto &pending : requeue) {
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
    }
    prefill_cv_.notify_one();
    // Split mode: decode workers wait on decode_cv_ (unbounded when
    // batch_accumulation_ms > 0) — notifying only prefill_cv_ left a
    // fairness-yielded request stranded until an unrelated prefill fired.
    if (use_decode_workers_) {
      decode_cv_.notify_one();
    }
  }
  auto batch_exec_end = std::chrono::steady_clock::now();
  double exec_ms =
      std::chrono::duration<double, std::milli>(batch_exec_end - batch_start)
          .count();
  metrics_->RecordBatchExecution(exec_ms);
  metrics_->SetDecodeQueueDepth(0);
}

void Scheduler::UpdateQueueDepthLocked() const {
  int prefill_depth = static_cast<int>(pending_prefill_.size());
  int decode_depth = static_cast<int>(pending_decode_.size());
  metrics_->SetQueueDepth(prefill_depth + decode_depth);
  metrics_->SetPrefillQueueDepth(prefill_depth);
  metrics_->SetDecodeQueueDepth(decode_depth);
}

// CALLING CONTRACT: callers must hold queue_mutex_ (or otherwise exclude
// Generate()/DecodeWorkerLoop queue mutations). The WorkerLoop's ProcessBatch
// call site takes the lock; tests drive it under the same lock via
// SchedulerTestAccess. This function iterates and mutates
// pending_prefill_/pending_decode_ and pairs fairness_controller_ reads with
// UpdateFairnessConfig's writes (same mutex).
void Scheduler::ApplyFairness(BatchSelection *selection) {
  if (!selection || selection->pending.empty()) {
    return;
  }

  std::vector<FairnessEntry> batch_entries;
  batch_entries.reserve(selection->pending.size());
  for (std::size_t i = 0; i < selection->pending.size(); ++i) {
    auto &pending = selection->pending[i];
    pending->inference.fairness.priority_level = pending->inference.priority;
    if (pending->inference.fairness.total_completion_tokens > 0 &&
        pending->inference.fairness.remaining_decode_tokens > 0) {
      metrics_->RecordFairnessResume(
          pending->inference.fairness.priority_level);
      SpanContext parent_ctx;
      parent_ctx.trace_id = pending->inference.trace_id;
      Span resume_span(
          "scheduler.fairness.resume", tracing::ChildContext(parent_ctx),
          [req_id = pending->inference.id,
           remaining = pending->inference.fairness.remaining_decode_tokens](
              const std::string &name, const SpanContext &ctx, double ms) {
            if (!FairnessTraceEnabled()) {
              return;
            }
            std::cout << "[fairness] " << name << " request=" << req_id
                      << " trace=" << ctx.trace_id << " remaining=" << remaining
                      << " duration_ms=" << ms << std::endl;
          });
      resume_span.Finish();
    }
    batch_entries.push_back(FairnessEntry{
        &pending->inference, pending->inference.fairness.priority_level, i});
  }
  struct QueueItem {
    std::shared_ptr<PendingRequest> pending;
    bool from_decode{false};
    std::size_t index{0};
  };
  // Snapshot the queues under queue_mutex_: Scheduler::Generate() pushes
  // into pending_prefill_ (HTTP workers) and DecodeWorkerLoop erases from
  // pending_decode_ under the same mutex — touching the vectors unlocked
  // here was a data race (vector realloc during iteration = UB). Only the
  // snapshot and the swap mutation below hold the lock; the fairness
  // evaluation runs on the local copies.
  std::vector<QueueItem> queue_refs;
  queue_refs.reserve(pending_prefill_.size() + pending_decode_.size());
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    auto &pending = pending_prefill_[i];
    pending->inference.fairness.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, false, i});
  }
  for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
    auto &pending = pending_decode_[i];
    pending->inference.fairness.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, true, i});
  }
  std::vector<FairnessEntry> queue_entries;
  queue_entries.reserve(queue_refs.size());
  for (std::size_t i = 0; i < queue_refs.size(); ++i) {
    queue_entries.push_back(FairnessEntry{
        &queue_refs[i].pending->inference,
        queue_refs[i].pending->inference.fairness.priority_level, i});
  }

  auto decision = fairness_controller_.Evaluate(batch_entries, queue_entries);
  if (decision.swap && decision.batch_index < selection->pending.size() &&
      decision.queue_index < queue_refs.size()) {
    auto queued_ref = queue_refs[decision.queue_index];
    auto queued = queued_ref.pending;
    auto displaced = selection->pending[decision.batch_index];
    const RequestPhase displaced_phase = displaced->inference.phase;
    selection->pending[decision.batch_index] = queued;
    if (queued_ref.from_decode) {
      pending_decode_.erase(pending_decode_.begin() +
                            static_cast<std::ptrdiff_t>(queued_ref.index));
    } else {
      pending_prefill_.erase(pending_prefill_.begin() +
                             static_cast<std::ptrdiff_t>(queued_ref.index));
    }
    if (displaced_phase == RequestPhase::kDecode) {
      displaced->inference.phase = RequestPhase::kDecode;
      pending_decode_.push_back(displaced);
    } else {
      displaced->inference.phase = RequestPhase::kPrefill;
      pending_prefill_.push_back(displaced);
    }
    batch_entries[decision.batch_index].request = &queued->inference;
    metrics_->RecordFairnessPreemption(
        queued->inference.fairness.priority_level);
    UpdateQueueDepthLocked();
  }
  fairness_controller_.ApplyTimeslice(&batch_entries);
}

int Scheduler::AllocSeqSlot(int64_t request_id, uint64_t *generation_out) {
  PollDeferredSequenceRetirements();
  if (!slot_manager_) {
    return -1;
  }
  auto lease = slot_manager_->AcquireLease(request_id);
  if (!lease) {
    return -1;
  }
  if (generation_out) {
    *generation_out = lease->generation;
  }
  LogSequenceSlotEvent("acquire", request_id, lease->slot_id, lease->generation,
                       RequestPhase::kPrefill, /*n_past=*/0,
                       /*remaining_decode_tokens=*/-1);
  RefreshNativeKvMemoryMetrics();
  return lease->slot_id;
}

void Scheduler::PollDeferredSequenceRetirements() {
  std::lock_guard<std::mutex> lock(deferred_sequence_retirements_mutex_);
  bool changed = false;
  auto it = deferred_sequence_retirements_.begin();
  while (it != deferred_sequence_retirements_.end()) {
    const bool ready = !it->backend || it->backend->PollFreeSequence(it->fence);
    if (!ready) {
      ++it;
      continue;
    }
    if (slot_manager_ && !slot_manager_->CompleteRetiredLease(it->lease)) {
      log::Warn("scheduler",
                "Failed to complete retired sequence lease for slot " +
                    std::to_string(it->lease.slot_id) +
                    " gen=" + std::to_string(it->lease.generation));
    }
    LogSequenceSlotEvent("retire_complete", it->lease.request_id,
                         it->lease.slot_id, it->lease.generation,
                         RequestPhase::kFinished, /*n_past=*/-1,
                         /*remaining_decode_tokens=*/-1);
    const auto lag_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - it->retired_at)
                            .count();
    metrics_->RecordSchedulerDeferredSequenceRetirement(lag_ms);
    it = deferred_sequence_retirements_.erase(it);
    changed = true;
  }
  metrics_->SetSchedulerDeferredSequenceRetirements(
      static_cast<int>(deferred_sequence_retirements_.size()));
  if (changed) {
    RefreshNativeKvMemoryMetrics();
  }
}

void Scheduler::FreeSeqSlot(int slot, uint64_t generation,
                            std::shared_ptr<LlamaCppBackend> backend) {
  if (slot < 0 || slot >= kMaxSequenceSlots) {
    return;
  }
  if (!slot_manager_) {
    if (backend) {
      backend->BeginFreeSequence(slot);
    }
    return;
  }

  if (backend && generation == 0) {
    backend->BeginFreeSequence(slot);
    slot_manager_->ReleaseSlot(slot);
    LogSequenceSlotEvent("release_legacy", /*request_id=*/-1, slot, generation,
                         RequestPhase::kFinished, /*n_past=*/-1,
                         /*remaining_decode_tokens=*/-1);
    RefreshNativeKvMemoryMetrics();
    return;
  }

  if (generation != 0 && backend) {
    LogSequenceSlotEvent("backend_release_begin", /*request_id=*/-1, slot,
                         generation, RequestPhase::kFinished,
                         /*n_past=*/-1, /*remaining_decode_tokens=*/-1,
                         backend->Name());
    auto fence = backend->BeginFreeSequence(slot);
    LogSequenceSlotEvent("backend_release_end", /*request_id=*/-1, slot,
                         generation, RequestPhase::kFinished,
                         /*n_past=*/-1, /*remaining_decode_tokens=*/-1,
                         backend->Name());
    if (fence.pending) {
      const scheduler::SequenceLease lease{slot, generation, -1};
      const bool retired = slot_manager_->RetireLease(
          lease, scheduler::SequenceRetireFence::Pending());
      if (!retired) {
        log::Warn("scheduler",
                  "Rejected stale sequence lease retirement for slot " +
                      std::to_string(slot) +
                      " gen=" + std::to_string(generation));
        return;
      }
      LogSequenceSlotEvent("retire_pending", /*request_id=*/-1, slot,
                           generation, RequestPhase::kFinished, /*n_past=*/-1,
                           /*remaining_decode_tokens=*/-1);
      {
        std::lock_guard<std::mutex> lock(deferred_sequence_retirements_mutex_);
        deferred_sequence_retirements_.push_back(
            DeferredSequenceRetirement{std::move(backend), lease, fence,
                                       std::chrono::steady_clock::now()});
        metrics_->SetSchedulerDeferredSequenceRetirements(
            static_cast<int>(deferred_sequence_retirements_.size()));
      }
      RefreshNativeKvMemoryMetrics();
      return;
    }
  } else if (!backend && generation != 0) {
    const bool released = slot_manager_->ReleaseLease({slot, generation, -1});
    if (!released) {
      log::Warn("scheduler", "Rejected stale sequence lease release for slot " +
                                 std::to_string(slot) +
                                 " gen=" + std::to_string(generation));
    } else {
      LogSequenceSlotEvent("release", /*request_id=*/-1, slot, generation,
                           RequestPhase::kFinished, /*n_past=*/-1,
                           /*remaining_decode_tokens=*/-1);
    }
    RefreshNativeKvMemoryMetrics();
    return;
  }

  if (generation != 0) {
    const bool released = slot_manager_->ReleaseLease({slot, generation, -1});
    if (!released) {
      log::Warn("scheduler", "Rejected stale sequence lease release for slot " +
                                 std::to_string(slot) +
                                 " gen=" + std::to_string(generation));
    } else {
      LogSequenceSlotEvent("release", /*request_id=*/-1, slot, generation,
                           RequestPhase::kFinished, /*n_past=*/-1,
                           /*remaining_decode_tokens=*/-1);
    }
    RefreshNativeKvMemoryMetrics();
    return;
  }
  slot_manager_->ReleaseSlot(slot);
  LogSequenceSlotEvent("release_legacy", /*request_id=*/-1, slot, generation,
                       RequestPhase::kFinished, /*n_past=*/-1,
                       /*remaining_decode_tokens=*/-1);
  RefreshNativeKvMemoryMetrics();
}

void Scheduler::ReleaseSessionState(
    const scheduler::SessionHandleState &state,
    std::shared_ptr<LlamaCppBackend> backend_hint) {
  if (state.sequence_id < 0 && state.block_table.empty()) {
    return;
  }
  if (!backend_hint && router_ && !state.model_id.empty()) {
    backend_hint = std::static_pointer_cast<LlamaCppBackend>(
        router_->GetBackend(state.model_id));
  }
  if (cache_ && !state.block_table.empty()) {
    cache_->ReleaseBlocksRef(state.block_table);
  }
  if (state.sequence_id >= 0) {
    FreeSeqSlot(state.sequence_id, state.sequence_generation, backend_hint);
  }
}

void Scheduler::FinalizeSessionLease(PendingRequest *pending,
                                     bool commit_state) {
  if (!pending || !session_handle_manager_) {
    return;
  }
  auto &inference = pending->inference;
  if (!inference.session_lease_acquired || inference.session_id.empty()) {
    return;
  }

  if (commit_state && inference.sequence_id >= 0) {
    LogSequenceSlotEvent("session_commit", inference, inference.session_id);
    if (slot_manager_ && inference.sequence_generation != 0) {
      const bool marked_completed = slot_manager_->MarkCompleted(
          {inference.sequence_id, inference.sequence_generation,
           static_cast<int64_t>(inference.id)},
          ResidentSequenceTokenCount(inference));
      if (!marked_completed) {
        log::Warn("scheduler",
                  "Failed to mark retained native sequence slot completed for "
                  "session " +
                      inference.session_id +
                      " slot=" + std::to_string(inference.sequence_id) +
                      " gen=" + std::to_string(inference.sequence_generation));
      }
    }
    scheduler::SessionHandleState state;
    state.model_id = inference.resolved_model.empty()
                         ? inference.model
                         : inference.resolved_model;
    state.sequence_id = inference.sequence_id;
    state.sequence_generation = inference.sequence_generation;
    state.prompt_tokens = inference.bpe_prompt_tokens;
    state.block_table = inference.block_table;
    session_handle_manager_->CommitLease(inference.session_id, state);
    inference.block_table.clear();
    ResetSequenceLease(&inference);
    inference.session_lease_acquired = false;
    RefreshNativeKvMemoryMetrics();
    return;
  }

  if (inference.sequence_id >= 0) {
    LogSequenceSlotEvent("session_release", inference, inference.session_id);
    if (cache_ && !inference.block_table.empty()) {
      cache_->ReleaseBlocksRef(inference.block_table);
    }
    inference.block_table.clear();
    FreeSeqSlot(inference.sequence_id, inference.sequence_generation,
                pending->resolved_backend);
    ResetSequenceLease(&inference);
  }
  session_handle_manager_->ReleaseLease(inference.session_id);
  inference.session_lease_acquired = false;
}

void Scheduler::EvictionWorkerLoop() {
  // Periodically evict idle sequences to prevent slot exhaustion.
  // Runs in a separate thread with wakeable 30-second check intervals so
  // teardown doesn't block waiting on sleep_for.
  std::unique_lock<std::mutex> wait_lock(eviction_mutex_);
  while (eviction_running_) {
    const bool stop_requested =
        eviction_cv_.wait_for(wait_lock, std::chrono::seconds(30),
                              [this] { return !eviction_running_.load(); });
    if (stop_requested || !eviction_running_) {
      break;
    }
    wait_lock.unlock();

    PollDeferredSequenceRetirements();

    // Idle slot eviction stays disabled until backend-sequence ownership is
    // tracked end-to-end. Reusing an evicted slot without deterministic
    // backend cleanup can corrupt warm-prefix/session sequence semantics.
    static bool logged_eviction_guard = false;
    if (!logged_eviction_guard) {
      log::Info(
          "scheduler",
          "Idle sequence eviction is guarded off pending backend ownership "
          "cleanup contract");
      logged_eviction_guard = true;
    }

    if (session_handle_manager_) {
      auto expired_sessions = session_handle_manager_->CollectExpired();
      for (const auto &state : expired_sessions) {
        ReleaseSessionState(state, nullptr);
      }
      if (!expired_sessions.empty()) {
        log::Info("scheduler", "Released " +
                                   std::to_string(expired_sessions.size()) +
                                   " expired session handle(s)");
      }
    }
    wait_lock.lock();
  }
}

void Scheduler::ResolveBackends(
    const std::vector<std::shared_ptr<PendingRequest>> &batch) {
  if (!router_) {
    for (auto &pending : batch) {
      if (HasBoundDecodeBackend(pending->inference,
                                pending->resolved_backend)) {
        continue;
      }
      pending->resolved_backend.reset();
      pending->inference.resolved_model.clear();
      pending->inference.response_format.supported = true;
      pending->inference.response_format.error.clear();
    }
    return;
  }
  ModelSelectionOptions selection_options = ModelSelectionOptionsSnapshot();
  for (auto &pending : batch) {
    if (HasBoundDecodeBackend(pending->inference, pending->resolved_backend)) {
      continue;
    }
    pending->resolved_backend.reset();
    pending->inference.resolved_model.clear();
    pending->inference.response_format.supported = true;
    pending->inference.response_format.error.clear();

    BackendFeatureRequirements requirements =
        BuildGenerationFeatureRequirements(
            pending->inference.stream, pending->inference.collect_logprobs,
            pending->inference.response_format.has_format,
            pending->inference.has_images,
            speculative_decoder_ && speculative_decoder_->Enabled() &&
                !pending->inference.response_format.has_format);

    auto selection =
        SelectModelForRequest(router_.get(), pending->inference.model,
                              requirements, selection_options);

    if (selection.status == ModelSelectionStatus::kNotFound) {
      metrics_->RecordModelRoute(pending->inference.model, "", false);
      continue;
    }
    if (selection.status == ModelSelectionStatus::kUnsupported) {
      pending->inference.response_format.supported = false;
      pending->inference.response_format.error =
          selection.reason.empty()
              ? "Selected model does not support requested features"
              : selection.reason;
      metrics_->RecordModelRoute(selection.info.id, selection.info.backend,
                                 false);
      continue;
    }
    if (selection.status == ModelSelectionStatus::kBackendUnavailable) {
      pending->inference.resolved_model = selection.info.id;
      metrics_->RecordModelRoute(selection.info.id, selection.info.backend,
                                 false);
      continue;
    }

    if (selection.used_fallback) {
      metrics_->RecordCapabilityRouteFallback(
          selection.fallback_from_backend, selection.info.backend,
          selection.fallback_feature.empty() ? "unsupported_feature"
                                             : selection.fallback_feature);
    }

    if (selection.backend && selection.backend->IsReady()) {
      pending->resolved_backend =
          std::static_pointer_cast<LlamaCppBackend>(selection.backend);
      pending->inference.resolved_model = selection.info.id;
      metrics_->RecordModelRoute(selection.info.id, selection.info.backend,
                                 true);
      if (selection.info.is_moe) {
        metrics_->RecordMoERequest();
      }
    } else {
      pending->inference.resolved_model = selection.info.id;
      metrics_->RecordModelRoute(selection.info.id, selection.info.backend,
                                 false);
    }
  }
}

bool Scheduler::TrySwapIn(InferenceRequest &inf) {
  if (!inf.is_swapped || !cache_)
    return false;

  std::size_t blocks_needed = inf.swapped_host_handles.size();
  if (cache_->NumFreeBlocks() < blocks_needed)
    return false;

  try {
    std::vector<int> target_blocks = cache_->ReserveBlocks(blocks_needed);
    cache_->SwapIn(inf.swapped_host_handles, target_blocks);
    inf.block_table = std::move(target_blocks);
    inf.swapped_host_handles.clear();
    inf.is_swapped = false;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool Scheduler::TrySwapOut(InferenceRequest &inf) {
  if (inf.is_swapped || inf.block_table.empty() || !cache_)
    return false;

  try {
    inf.swapped_host_handles = cache_->SwapOut(inf.block_table);
    inf.block_table.clear();
    inf.is_swapped = true;
    metrics_->RecordFairnessPreemption(
        inf.fairness.priority_level); // Reuse preemption metric for swap
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

} // namespace inferflux
