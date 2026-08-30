#include "server/metrics/metrics.h"

#include "runtime/backends/cuda/native/dispatch_catalog.h"
#include "server/diagnostics/crash_handler.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace inferflux {

void MetricsRegistry::Reset() {
  // Reconstruct in-place to reset all counters, gauges, maps, and histograms
  // to their default-constructed values.  This is safe because no external
  // code holds references to individual metric fields — all access goes
  // through MetricsRegistry methods.
  this->~MetricsRegistry();
  new (this) MetricsRegistry();
}

namespace {
MetricsRegistry g_metrics;

std::string NormalizeNativePhase(std::string_view phase) {
  return phase == "decode"    ? "decode"
         : phase == "prefill" ? "prefill"
                              : "unknown";
}

std::string NormalizeBurstPhase(std::string_view phase) {
  return phase == "generate" ? "generate" : "decode";
}

std::string NormalizeBurstIneligibleReason(std::string_view reason) {
  if (reason == "disabled") {
    return "disabled";
  }
  if (reason == "missing_runtime") {
    return "missing_runtime";
  }
  if (reason == "missing_tokenizer") {
    return "missing_tokenizer";
  }
  if (reason == "logprobs") {
    return "logprobs";
  }
  if (reason == "streaming_callback") {
    return "streaming_callback";
  }
  if (reason == "temperature") {
    return "temperature";
  }
  if (reason == "stop_sequences") {
    return "stop_sequences";
  }
  if (reason == "scheduler_stepwise") {
    return "scheduler_stepwise";
  }
  return "other";
}

std::string NativeForwardBatchSizeBucket(int batch_size) {
  if (batch_size <= 1) {
    return "1";
  }
  if (batch_size == 2) {
    return "2";
  }
  if (batch_size <= 4) {
    return "3_4";
  }
  if (batch_size <= 8) {
    return "5_8";
  }
  if (batch_size <= 16) {
    return "9_16";
  }
  return "17_plus";
}

std::string InferfluxCudaPrefillBatchSizeBucket(int batch_size) {
  if (batch_size <= 1) {
    return "1";
  }
  if (batch_size == 2) {
    return "2";
  }
  if (batch_size <= 4) {
    return "3_4";
  }
  if (batch_size <= 8) {
    return "5_8";
  }
  if (batch_size <= 16) {
    return "9_16";
  }
  if (batch_size <= 32) {
    return "17_32";
  }
  if (batch_size <= 64) {
    return "33_64";
  }
  if (batch_size <= 128) {
    return "65_128";
  }
  return "129_plus";
}

std::string InferfluxCudaBatchSizeBucket(std::string_view phase,
                                         int batch_size) {
  const std::string phase_label = NormalizeNativePhase(phase);
  if (phase_label == "prefill") {
    return InferfluxCudaPrefillBatchSizeBucket(batch_size);
  }
  return NativeForwardBatchSizeBucket(batch_size);
}

std::string SchedulerDecodeCountBucket(std::size_t count) {
  if (count == 0) {
    return "0";
  }
  return NativeForwardBatchSizeBucket(static_cast<int>(count));
}

std::string NativeLinearDimBucket(int dim) {
  if (dim <= 1024) {
    return "1_1024";
  }
  if (dim <= 2048) {
    return "1025_2048";
  }
  if (dim <= 4096) {
    return "2049_4096";
  }
  if (dim <= 8192) {
    return "4097_8192";
  }
  if (dim <= 16384) {
    return "8193_16384";
  }
  return "16385_plus";
}

std::string NativeGroupedOutputsBucket(int grouped_outputs) {
  if (grouped_outputs <= 1) {
    return "1";
  }
  if (grouped_outputs == 2) {
    return "2";
  }
  if (grouped_outputs == 3) {
    return "3";
  }
  return "4_plus";
}

std::string NormalizeNativeQuant(std::string_view quant) {
  if (quant.empty()) {
    return "unknown";
  }
  return std::string(quant);
}
} // namespace

void LatencyHistogram::Record(double ms) {
  total.fetch_add(1, std::memory_order_relaxed);
  sum_ms.fetch_add(static_cast<uint64_t>(std::max(0.0, ms)),
                   std::memory_order_relaxed);
  // All buckets are cumulative: increment every bucket >= ms.
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    if (ms <= kBuckets[i]) {
      counts[i].fetch_add(1, std::memory_order_relaxed);
    }
  }
  // +Inf bucket always increments.
  counts[kBuckets.size()].fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetBackend(const std::string &backend) {
  std::lock_guard<std::mutex> lock(backend_mutex_);
  backend_ = backend;
}

void MetricsRegistry::RecordSuccess(int prompt_tokens, int completion_tokens) {
  total_requests_.fetch_add(1, std::memory_order_relaxed);
  total_prompt_tokens_.fetch_add(prompt_tokens, std::memory_order_relaxed);
  total_completion_tokens_.fetch_add(completion_tokens,
                                     std::memory_order_relaxed);
}

void MetricsRegistry::RecordError() {
  total_errors_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordEmptyGeneration() {
  total_empty_generations_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordOutputUtf8Replacements(std::size_t replacements) {
  output_utf8_replacements_.fetch_add(replacements, std::memory_order_relaxed);
}

void MetricsRegistry::RecordSpeculative(std::size_t total_chunks,
                                        std::size_t accepted_chunks,
                                        std::size_t reused_tokens) {
  if (total_chunks == 0 && accepted_chunks == 0 && reused_tokens == 0) {
    return;
  }
  speculative_chunks_total_.fetch_add(total_chunks, std::memory_order_relaxed);
  speculative_chunks_accepted_.fetch_add(accepted_chunks,
                                         std::memory_order_relaxed);
  speculative_tokens_reused_.fetch_add(reused_tokens,
                                       std::memory_order_relaxed);
}

void MetricsRegistry::RecordBatch(std::size_t request_count,
                                  std::size_t token_count) {
  if (request_count == 0) {
    return;
  }
  total_batches_.fetch_add(1, std::memory_order_relaxed);
  total_batch_tokens_.fetch_add(token_count, std::memory_order_relaxed);
  uint64_t current_max = max_batch_size_.load(std::memory_order_relaxed);
  while (request_count > current_max &&
         !max_batch_size_.compare_exchange_weak(current_max, request_count,
                                                std::memory_order_relaxed)) {
  }
}

void MetricsRegistry::RecordSchedulerIteration(std::size_t prefill_requests,
                                               std::size_t decode_requests,
                                               std::size_t token_count) {
  if (prefill_requests == 0 && decode_requests == 0) {
    return;
  }
  scheduler_iteration_requests_total_.fetch_add(
      prefill_requests + decode_requests, std::memory_order_relaxed);
  if (token_count > 0) {
    scheduler_iteration_tokens_total_.fetch_add(token_count,
                                                std::memory_order_relaxed);
  }

  if (prefill_requests > 0 && decode_requests > 0) {
    scheduler_iterations_mixed_.fetch_add(1, std::memory_order_relaxed);
  } else if (prefill_requests > 0) {
    scheduler_iterations_prefill_.fetch_add(1, std::memory_order_relaxed);
  } else {
    scheduler_iterations_decode_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordSchedulerPolicyIteration(
    const std::string &policy, std::size_t prefill_requests,
    std::size_t decode_requests) {
  if (prefill_requests == 0 && decode_requests == 0) {
    return;
  }
  const std::string policy_label = policy.empty() ? "unknown" : policy;
  std::string phase = "decode";
  if (prefill_requests > 0 && decode_requests > 0) {
    phase = "mixed";
  } else if (prefill_requests > 0) {
    phase = "prefill";
  }
  const std::string key = policy_label + "|" + phase;
  std::lock_guard<std::mutex> lock(scheduler_policy_metrics_mutex_);
  scheduler_policy_iterations_[key] += 1;
}

void MetricsRegistry::RecordBatchTokenBudgetSkip() {
  scheduler_batch_token_budget_skips_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordPrefixAffinityProbe(bool hit, int matched_tokens) {
  scheduler_prefix_affinity_probes_.fetch_add(1, std::memory_order_relaxed);
  if (!hit) {
    return;
  }
  scheduler_prefix_affinity_hits_.fetch_add(1, std::memory_order_relaxed);
  if (matched_tokens > 0) {
    scheduler_prefix_affinity_matched_tokens_.fetch_add(
        static_cast<uint64_t>(matched_tokens), std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordPrefixLookup(bool hit) {
  if (hit) {
    prefix_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    prefix_misses_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordPrefixMatchedTokens(int tokens) {
  if (tokens <= 0)
    return;
  prefix_matched_tokens_.fetch_add(static_cast<uint64_t>(tokens),
                                   std::memory_order_relaxed);
}

void MetricsRegistry::RecordPartialPrefixHit() {
  prefix_partial_hits_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordKVPrefixReuse(int tokens_saved) {
  kv_prefix_reuse_count_.fetch_add(1, std::memory_order_relaxed);
  if (tokens_saved > 0) {
    kv_prefix_reuse_tokens_.fetch_add(static_cast<uint64_t>(tokens_saved),
                                      std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordStreamTokens(std::size_t tokens) {
  if (tokens == 0) {
    return;
  }
  stream_tokens_.fetch_add(tokens, std::memory_order_relaxed);
}

void MetricsRegistry::RecordStreamCacheHit() {
  stream_cache_hits_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFairnessTokens(int priority_level,
                                           std::size_t tokens) {
  if (tokens == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
  fairness_tokens_[priority_level] += tokens;
}

void MetricsRegistry::RecordFairnessPreemption(int priority_level) {
  fairness_preemptions_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
  (void)fairness_tokens_[priority_level];
}

void MetricsRegistry::RecordFairnessYield(int priority_level,
                                          std::size_t emitted_tokens,
                                          std::size_t remaining_tokens) {
  fairness_yields_.fetch_add(1, std::memory_order_relaxed);
  (void)priority_level;
  (void)emitted_tokens;
  (void)remaining_tokens;
}

void MetricsRegistry::RecordFairnessResume(int priority_level) {
  fairness_resumes_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordQueueLatency(double wait_ms) {
  queue_latency_.Record(wait_ms);
}

void MetricsRegistry::RecordBatchExecution(double exec_ms) {
  batch_exec_latency_.Record(exec_ms);
}

void MetricsRegistry::RecordModelLoad(const std::string &model_id,
                                      const std::string &backend,
                                      double load_seconds) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.load_seconds += std::max(0.0, load_seconds);
  stats.load_events += 1;
}

void MetricsRegistry::RecordModelReady(const std::string &model_id,
                                       const std::string &backend, bool ready) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.ready = ready;
}

void MetricsRegistry::RecordModelRoute(const std::string &model_id,
                                       const std::string &backend, bool hit) {
  if (!hit || model_id.empty()) {
    model_route_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.routes += 1;
}

void MetricsRegistry::RecordModelTokens(const std::string &model_id,
                                        const std::string &backend,
                                        int prompt_tokens,
                                        int completion_tokens) {
  if (model_id.empty()) {
    return;
  }
  if (prompt_tokens <= 0 && completion_tokens <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  if (prompt_tokens > 0) {
    stats.prompt_tokens += static_cast<uint64_t>(prompt_tokens);
  }
  if (completion_tokens > 0) {
    stats.completion_tokens += static_cast<uint64_t>(completion_tokens);
  }
}

void MetricsRegistry::RecordCapabilityRejection(const std::string &backend,
                                                const std::string &feature) {
  if (feature.empty()) {
    return;
  }
  const std::string backend_label = backend.empty() ? "unknown" : backend;
  const std::string key = backend_label + "|" + feature;
  std::lock_guard<std::mutex> lock(capability_metrics_mutex_);
  capability_rejections_[key] += 1;
}

void MetricsRegistry::RecordBackendExposure(
    const std::string &requested_backend, const std::string &exposed_backend,
    const std::string &provider, bool used_fallback) {
  if (exposed_backend.empty()) {
    return;
  }
  const std::string requested_label =
      requested_backend.empty() ? "auto" : requested_backend;
  const std::string exposed_label =
      exposed_backend.empty() ? "unknown" : exposed_backend;
  const std::string provider_label = provider.empty() ? "unknown" : provider;
  const std::string fallback_label = used_fallback ? "true" : "false";
  const std::string key = requested_label + "|" + exposed_label + "|" +
                          provider_label + "|" + fallback_label;
  std::lock_guard<std::mutex> lock(backend_exposure_mutex_);
  backend_exposure_counts_[key] += 1;
}

void MetricsRegistry::RecordCapabilityRouteFallback(
    const std::string &from_backend, const std::string &to_backend,
    const std::string &feature) {
  if (feature.empty()) {
    return;
  }
  const std::string from_label =
      from_backend.empty() ? "unknown" : from_backend;
  const std::string to_label = to_backend.empty() ? "unknown" : to_backend;
  const std::string key = from_label + "|" + to_label + "|" + feature;
  std::lock_guard<std::mutex> lock(capability_route_fallback_mutex_);
  capability_route_fallbacks_[key] += 1;
}

void MetricsRegistry::RecordImagePreprocess(int images, double /*decode_ms*/) {
  if (images > 0) {
    multimodal_images_.fetch_add(static_cast<uint64_t>(images),
                                 std::memory_order_relaxed);
    multimodal_requests_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordMoERequest() {
  moe_requests_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetFlashAttentionEnabled(bool enabled) {
  flash_attention_enabled_.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFlashAttentionExecution(const std::string &kernel,
                                                    double duration_ms,
                                                    int prompt_tokens) {
  // Record execution time histogram
  flash_attention_exec_latency_.Record(duration_ms);

  // Record request counter for the specific kernel
  if (kernel == "fa2") {
    flash_attention_requests_fa2_.fetch_add(1, std::memory_order_relaxed);
  } else if (kernel == "fa3") {
    flash_attention_requests_fa3_.fetch_add(1, std::memory_order_relaxed);
  } else {
    flash_attention_requests_standard_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::SetFlashAttentionMemoryMB(double memory_mb) {
  flash_attention_memory_mb_.store(memory_mb, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFlashAttentionRequest(const std::string &kernel) {
  // This is called when a request starts processing with FlashAttention
  // Separate from RecordFlashAttentionExecution which is called on completion
  if (kernel == "fa2") {
    flash_attention_requests_fa2_.fetch_add(1, std::memory_order_relaxed);
  } else if (kernel == "fa3") {
    flash_attention_requests_fa3_.fetch_add(1, std::memory_order_relaxed);
  } else {
    flash_attention_requests_standard_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordRocmKernelSelection(const std::string &kernel) {
  // Track which kernel is selected for ROCm backend
  // This is called when the ROCm backend selects an attention kernel
  if (kernel == "fa2") {
    rocm_flash_attention_requests_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordRocmFlashAttentionExecution(double duration_ms,
                                                        int tokens) {
  // Record ROCm FlashAttention execution time
  // Reuse the FlashAttention histogram for ROCm
  flash_attention_exec_latency_.Record(duration_ms);
}

void MetricsRegistry::SetRocmMemoryUsageMB(double memory_mb) {
  rocm_memory_mb_.store(memory_mb, std::memory_order_relaxed);
}

void MetricsRegistry::RecordRocmDeviceProperties(int device_id,
                                                 const std::string &arch) {
  // Record ROCm device properties
  rocm_device_arch_ = arch;
}

void MetricsRegistry::RecordKVTransfer(double transfer_ms) {
  kv_transfer_latency_.Record(transfer_ms);
}

void MetricsRegistry::RecordDisaggKVEnqueueRejected(bool retries_exhausted) {
  disagg_kv_enqueue_rejections_.fetch_add(1, std::memory_order_relaxed);
  if (retries_exhausted) {
    disagg_kv_enqueue_exhausted_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordDisaggKVTicketStage(std::string_view stage) {
  if (stage == "enqueued") {
    disagg_kv_tickets_enqueued_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (stage == "acknowledged") {
    disagg_kv_tickets_acknowledged_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (stage == "committed") {
    disagg_kv_tickets_committed_.fetch_add(1, std::memory_order_relaxed);
    disagg_kv_timeout_streak_.store(0, std::memory_order_relaxed);
    uint64_t debt = disagg_kv_timeout_debt_.load(std::memory_order_relaxed);
    while (debt > 0 && !disagg_kv_timeout_debt_.compare_exchange_weak(
                           debt, debt - 1, std::memory_order_relaxed)) {
    }
    return;
  }
  if (stage == "timed_out") {
    disagg_kv_tickets_timed_out_.fetch_add(1, std::memory_order_relaxed);
    disagg_kv_timeout_debt_.fetch_add(1, std::memory_order_relaxed);
    disagg_kv_timeout_streak_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordLlamaPerf(double prefill_ms, double decode_ms,
                                      int32_t prompt_tokens,
                                      int32_t generated_tokens) {
  if (prefill_ms > 0)
    llama_prefill_latency_.Record(prefill_ms);
  if (decode_ms > 0)
    llama_decode_latency_.Record(decode_ms);
  if (prompt_tokens > 0)
    llama_prompt_tokens_.fetch_add(static_cast<uint64_t>(prompt_tokens),
                                   std::memory_order_relaxed);
  if (generated_tokens > 0)
    llama_gen_tokens_.fetch_add(static_cast<uint64_t>(generated_tokens),
                                std::memory_order_relaxed);
}

void MetricsRegistry::RecordLatency(double request_ms) {
  request_latency_.Record(request_ms);
}

void MetricsRegistry::RecordPrefillDuration(double prefill_ms) {
  prefill_latency_.Record(prefill_ms);
}

void MetricsRegistry::RecordDecodeDuration(double decode_ms) {
  decode_latency_.Record(decode_ms);
}

void MetricsRegistry::IncrementConnections() {
  int n = active_connections_.fetch_add(1, std::memory_order_relaxed) + 1;
  diagnostics::SetActiveRequests(n);
}

void MetricsRegistry::DecrementConnections() {
  int n = active_connections_.fetch_sub(1, std::memory_order_relaxed) - 1;
  diagnostics::SetActiveRequests(n);
}

void MetricsRegistry::SetQueueDepth(int depth) {
  queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetPrefillQueueDepth(int depth) {
  prefill_queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetDecodeQueueDepth(int depth) {
  decode_queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetSchedulerBatchLimits(int max_batch_size,
                                              int max_batch_tokens) {
  scheduler_batch_limit_size_.store(max_batch_size, std::memory_order_relaxed);
  scheduler_batch_limit_tokens_.store(max_batch_tokens,
                                      std::memory_order_relaxed);
}

void MetricsRegistry::RecordCudaLaneSubmission(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_submissions_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_submissions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneCompletion(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_completions_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_completions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneExecutionStart(bool decode_lane) {
  bool maybe_start_overlap = false;
  if (decode_lane) {
    cuda_decode_lane_inflight_.fetch_add(1, std::memory_order_relaxed);
    if (cuda_prefill_lane_inflight_.load(std::memory_order_relaxed) > 0) {
      cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
      maybe_start_overlap = true;
    }
  } else {
    cuda_prefill_lane_inflight_.fetch_add(1, std::memory_order_relaxed);
    if (cuda_decode_lane_inflight_.load(std::memory_order_relaxed) > 0) {
      cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
      maybe_start_overlap = true;
    }
  }

  if (maybe_start_overlap) {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    if (!cuda_overlap_active_) {
      cuda_overlap_active_ = true;
      cuda_overlap_started_at_ = std::chrono::steady_clock::now();
    }
  }
}

void MetricsRegistry::RecordCudaLaneExecutionStop(bool decode_lane) {
  if (decode_lane) {
    const int previous =
        cuda_decode_lane_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
      cuda_decode_lane_inflight_.store(0, std::memory_order_relaxed);
    }
  } else {
    const int previous =
        cuda_prefill_lane_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
      cuda_prefill_lane_inflight_.store(0, std::memory_order_relaxed);
    }
  }

  if (cuda_decode_lane_inflight_.load(std::memory_order_relaxed) == 0 &&
      cuda_prefill_lane_inflight_.load(std::memory_order_relaxed) == 0) {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    if (cuda_overlap_active_) {
      const auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - cuda_overlap_started_at_)
              .count();
      if (elapsed_us > 0) {
        cuda_lane_overlap_duration_us_.fetch_add(
            static_cast<uint64_t>(elapsed_us), std::memory_order_relaxed);
      }
      cuda_overlap_active_ = false;
    }
  }
}

void MetricsRegistry::RecordCudaLaneOverlap(double duration_ms) {
  // Record an overlap event and its duration
  cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
  const auto duration_us = static_cast<uint64_t>(duration_ms * 1000.0);
  if (duration_us > 0) {
    cuda_lane_overlap_duration_us_.fetch_add(duration_us,
                                             std::memory_order_relaxed);
  }
}

void MetricsRegistry::SetCudaLaneQueueDepth(bool decode_lane, int depth) {
  if (decode_lane) {
    cuda_decode_lane_queue_depth_.store(depth, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_queue_depth_.store(depth, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneEnqueueReject(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_enqueue_rejects_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_enqueue_rejects_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneCollectTimeout(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_collect_timeouts_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_collect_timeouts_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneWorkerRestart() {
  cuda_lane_worker_restarts_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordDecodeStepLoops(const std::string &mode,
                                            std::size_t loops) {
  if (loops == 0) {
    return;
  }
  const std::string mode_label = mode.empty() ? "unknown" : mode;
  std::lock_guard<std::mutex> lock(scheduler_step_metrics_mutex_);
  scheduler_decode_step_loops_[mode_label] += loops;
}

void MetricsRegistry::RecordPrefillChunkTruncation(
    const std::string &mode, std::size_t truncated_tokens) {
  if (truncated_tokens == 0) {
    return;
  }
  const std::string mode_label = mode.empty() ? "unknown" : mode;
  std::lock_guard<std::mutex> lock(scheduler_step_metrics_mutex_);
  scheduler_prefill_chunk_truncations_[mode_label] += 1;
  scheduler_prefill_chunk_truncated_tokens_[mode_label] += truncated_tokens;
}

void MetricsRegistry::RecordDecodeWorkerBatchSize(std::size_t batch_size) {
  const std::string key = std::to_string(batch_size);
  std::lock_guard<std::mutex> lock(scheduler_decode_worker_batch_size_mutex_);
  scheduler_decode_worker_batch_size_counts_[key] += 1;
}

void MetricsRegistry::RecordDecodeWorkerExecutionPath(std::string_view path) {
  const std::string key = path.empty() ? "unknown" : std::string(path);
  std::lock_guard<std::mutex> lock(scheduler_decode_worker_batch_size_mutex_);
  scheduler_decode_worker_execution_path_counts_[key] += 1;
}

void MetricsRegistry::RecordDecodeWorkerStickyMerge(
    std::size_t merged_requests) {
  if (merged_requests == 0) {
    return;
  }
  const std::string key = std::to_string(merged_requests);
  std::lock_guard<std::mutex> lock(scheduler_decode_worker_batch_size_mutex_);
  scheduler_decode_worker_sticky_merge_counts_[key] += 1;
  scheduler_decode_worker_sticky_merged_requests_total_.fetch_add(
      static_cast<uint64_t>(merged_requests), std::memory_order_relaxed);
}

void MetricsRegistry::RecordDecodeAssemblySnapshot(std::string_view mode,
                                                   std::size_t ready,
                                                   std::size_t selected,
                                                   std::size_t compatible,
                                                   std::size_t incompatible) {
  const std::string mode_key = mode.empty() ? "unknown" : std::string(mode);
  std::lock_guard<std::mutex> lock(scheduler_decode_worker_batch_size_mutex_);
  scheduler_decode_assembly_ready_[mode_key + "|" +
                                   SchedulerDecodeCountBucket(ready)] += 1;
  scheduler_decode_assembly_selected_[mode_key + "|" +
                                      SchedulerDecodeCountBucket(selected)] +=
      1;
  if (compatible > 0) {
    scheduler_decode_assembly_compatible_[mode_key + "|" +
                                          SchedulerDecodeCountBucket(
                                              compatible)] += 1;
  }
  if (incompatible > 0 || (ready > 0 && compatible == 0)) {
    scheduler_decode_assembly_incompatible_[mode_key + "|" +
                                            SchedulerDecodeCountBucket(
                                                incompatible)] += 1;
  }
}

void MetricsRegistry::RecordInferfluxCudaForwardShape(bool is_decode,
                                                      int batch_size) {
  if (is_decode) {
    inferflux_cuda_forward_decode_total_.fetch_add(1,
                                                   std::memory_order_relaxed);
  } else {
    inferflux_cuda_forward_prefill_total_.fetch_add(1,
                                                    std::memory_order_relaxed);
  }
  inferflux_cuda_forward_batch_tokens_total_.fetch_add(
      batch_size, std::memory_order_relaxed);
  RecordInferfluxCudaForwardBatchSize(is_decode ? "decode" : "prefill",
                                      batch_size);
}

void MetricsRegistry::RecordInferfluxCudaForwardLatency(double forward_ms) {
  inferflux_cuda_forward_latency_.Record(forward_ms);
}

void MetricsRegistry::RecordInferfluxCudaForwardPass(bool is_decode,
                                                     int batch_size,
                                                     double forward_ms) {
  RecordInferfluxCudaForwardShape(is_decode, batch_size);
  RecordInferfluxCudaForwardLatency(forward_ms);
}

void MetricsRegistry::RecordInferfluxCudaSampling(int batch_size,
                                                  double sampling_ms) {
  (void)batch_size;
  inferflux_cuda_sampling_latency_.Record(sampling_ms);
}

void MetricsRegistry::RecordInferfluxCudaBatchDecode(int batch_size,
                                                     double total_ms) {
  (void)batch_size;
  (void)total_ms;
  // Tracked via forward + sampling histograms; this is for future use.
}

void MetricsRegistry::RecordInferfluxCudaBurstDecodeChunk(
    std::string_view phase, int requested_tokens, int produced_tokens) {
  const std::string phase_label = NormalizeBurstPhase(phase);
  if (phase_label == "generate") {
    inferflux_cuda_burst_decode_chunks_generate_total_.fetch_add(
        1, std::memory_order_relaxed);
    inferflux_cuda_burst_decode_requested_tokens_generate_total_.fetch_add(
        std::max(0, requested_tokens), std::memory_order_relaxed);
    inferflux_cuda_burst_decode_produced_tokens_generate_total_.fetch_add(
        std::max(0, produced_tokens), std::memory_order_relaxed);
    return;
  }

  inferflux_cuda_burst_decode_chunks_decode_total_.fetch_add(
      1, std::memory_order_relaxed);
  inferflux_cuda_burst_decode_requested_tokens_decode_total_.fetch_add(
      std::max(0, requested_tokens), std::memory_order_relaxed);
  inferflux_cuda_burst_decode_produced_tokens_decode_total_.fetch_add(
      std::max(0, produced_tokens), std::memory_order_relaxed);
}

void MetricsRegistry::RecordInferfluxCudaBurstDecodeFallback(
    std::string_view phase) {
  const std::string phase_label = NormalizeBurstPhase(phase);
  if (phase_label == "generate") {
    inferflux_cuda_burst_decode_fallbacks_generate_total_.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  inferflux_cuda_burst_decode_fallbacks_decode_total_.fetch_add(
      1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordInferfluxCudaBurstDecodeIneligible(
    std::string_view phase, std::string_view reason) {
  const std::string key =
      NormalizeBurstPhase(phase) + "|" + NormalizeBurstIneligibleReason(reason);
  std::lock_guard<std::mutex> lock(
      inferflux_cuda_burst_decode_ineligible_mutex_);
  inferflux_cuda_burst_decode_ineligible_counts_[key] += 1;
}
void MetricsRegistry::RecordInferfluxCudaForwardBatchSize(
    std::string_view phase, int batch_size) {
  const std::string key = NormalizeNativePhase(phase) + "|" +
                          InferfluxCudaBatchSizeBucket(phase, batch_size);
  std::lock_guard<std::mutex> lock(inferflux_cuda_forward_batch_size_mutex_);
  inferflux_cuda_forward_batch_size_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaFfnProjOperator(std::string_view phase,
                                                         std::string_view op) {
  const std::string phase_label = NormalizeNativePhase(phase);
  // Labels come from dispatch_catalog.h (FfnMetricLabel) — recorded
  // verbatim. The old hand-written allowlist mapped q8_1_group_mmq3 and
  // the M>8 remap to "unknown", which the fixed-grid render below never
  // exported: the exact invisibility that hid the kQ81GroupMmq3 executor
  // bug for months.
  const std::string op_label =
      (op == "q8_1_group") ? "q8_1_group_generic" : std::string(op);
  const std::string key = phase_label + "|" + op_label;
  std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
  inferflux_cuda_ffn_proj_operator_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaFfnProjGeometry(
    std::string_view phase, std::string_view op, std::string_view quant,
    int batch_size, int n, int k, int grouped_outputs) {
  const std::string key =
      NormalizeNativePhase(phase) + "|" + std::string(op) + "|" +
      NormalizeNativeQuant(quant) + "|" +
      InferfluxCudaBatchSizeBucket(phase, batch_size) + "|" +
      std::to_string(std::max(0, n)) + "|" + NativeLinearDimBucket(n) + "|" +
      std::to_string(std::max(0, k)) + "|" + NativeLinearDimBucket(k) + "|" +
      NativeGroupedOutputsBucket(grouped_outputs);
  std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
  inferflux_cuda_ffn_proj_geometry_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaDownProjOperator(
    std::string_view phase, std::string_view op) {
  const std::string phase_label = NormalizeNativePhase(phase);
  const std::string op_label =
      op == "mmq"                            ? "mmq"
      : op == "q8_1_gemv_v2"                 ? "q8_1_gemv_v2"
      : op == "q8_1_gemv"                    ? "q8_1_gemv"
      : op == "q8_1_gemv_hot_fixed"          ? "q8_1_gemv_hot_fixed"
      : op == "q8_1_gemv_row_pair_hot_fixed" ? "q8_1_gemv_row_pair_hot_fixed"
      : op == "q8_1_gemv_row_pair_v2"        ? "q8_1_gemv_row_pair_v2"
      : op == "q8_1_gemv_row_pair"           ? "q8_1_gemv_row_pair"
      : op == "q8_1_gemv_row_quad"           ? "q8_1_gemv_row_quad"
      : op == "packed_gemv"                  ? "packed_gemv"
      : op == "fallback"                     ? "fallback"
                                             : "unknown";
  const std::string key = phase_label + "|" + op_label;
  std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
  inferflux_cuda_down_proj_operator_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaDownProjGeometry(
    std::string_view phase, std::string_view op, std::string_view quant,
    int batch_size, int n, int k) {
  const std::string key =
      NormalizeNativePhase(phase) + "|" + std::string(op) + "|" +
      NormalizeNativeQuant(quant) + "|" +
      InferfluxCudaBatchSizeBucket(phase, batch_size) + "|" +
      std::to_string(std::max(0, n)) + "|" + NativeLinearDimBucket(n) + "|" +
      std::to_string(std::max(0, k)) + "|" + NativeLinearDimBucket(k);
  std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
  inferflux_cuda_down_proj_geometry_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaRowPairSelection(
    std::string_view phase, std::string_view op, int batch_rows) {
  if (batch_rows < 2) {
    return;
  }
  const std::string key = NormalizeNativePhase(phase) + "|" + std::string(op) +
                          "|" + InferfluxCudaBatchSizeBucket(phase, batch_rows);
  std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
  inferflux_cuda_rowpair_selection_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaKvAutoTunePlan(
    int requested_max_seq, int planned_max_seq, std::size_t requested_bytes,
    std::size_t planned_bytes, std::size_t budget_bytes) {
  inferflux_cuda_kv_requested_max_seq_.store(std::max(0, requested_max_seq),
                                             std::memory_order_relaxed);
  inferflux_cuda_kv_planned_max_seq_.store(std::max(0, planned_max_seq),
                                           std::memory_order_relaxed);
  inferflux_cuda_kv_requested_bytes_.store(
      static_cast<uint64_t>(requested_bytes), std::memory_order_relaxed);
  inferflux_cuda_kv_planned_bytes_.store(static_cast<uint64_t>(planned_bytes),
                                         std::memory_order_relaxed);
  inferflux_cuda_kv_budget_bytes_.store(static_cast<uint64_t>(budget_bytes),
                                        std::memory_order_relaxed);
  if (planned_max_seq > 0 && requested_max_seq > planned_max_seq) {
    inferflux_cuda_kv_autotune_events_total_.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::SetInferfluxCudaKvCacheOccupancy(int active_sequences,
                                                       int max_sequences) {
  inferflux_cuda_kv_active_sequences_.store(active_sequences,
                                            std::memory_order_relaxed);
  inferflux_cuda_kv_max_sequences_.store(max_sequences,
                                         std::memory_order_relaxed);
}

void MetricsRegistry::SetSchedulerDeferredSequenceRetirements(int depth) {
  scheduler_deferred_sequence_retirements_.store(std::max(0, depth),
                                                 std::memory_order_relaxed);
}

void MetricsRegistry::RecordSchedulerDeferredSequenceRetirement(double lag_ms) {
  scheduler_sequence_retirement_latency_.Record(lag_ms);
  scheduler_deferred_sequence_retirements_completed_.fetch_add(
      1, std::memory_order_relaxed);
}

void MetricsRegistry::SetCudaAttentionKernel(const std::string &kernel) {
  std::string normalized = kernel;
  if (normalized != "fa3" && normalized != "fa2" && normalized != "standard") {
    normalized = "standard";
  }
  std::lock_guard<std::mutex> lock(cuda_attention_kernel_mutex_);
  cuda_attention_kernel_ = normalized;
}

void MetricsRegistry::RecordCudaAttentionKernelFallback(
    const std::string &requested_kernel, const std::string &selected_kernel,
    const std::string &reason) {
  const std::string requested =
      requested_kernel.empty() ? "auto" : requested_kernel;
  const std::string selected =
      selected_kernel.empty() ? "standard" : selected_kernel;
  const std::string fallback_reason = reason.empty() ? "unspecified" : reason;
  const std::string key = requested + "|" + selected + "|" + fallback_reason;
  std::lock_guard<std::mutex> lock(cuda_attention_fallback_mutex_);
  cuda_attention_fallback_counts_[key] += 1;
}

void MetricsRegistry::RecordCudaAttentionKernelSwitch(
    const std::string &from_kernel, const std::string &to_kernel) {
  const std::string from = from_kernel.empty() ? "unknown" : from_kernel;
  const std::string to = to_kernel.empty() ? "unknown" : to_kernel;
  if (from == to) {
    return;
  }
  const std::string key = from + "|" + to;
  std::lock_guard<std::mutex> lock(cuda_attention_switch_mutex_);
  cuda_attention_switch_counts_[key] += 1;
}

void MetricsRegistry::RecordInferfluxCudaDispatchDivergence(
    std::string_view layer, std::string_view selected, std::string_view actual,
    std::string_view reason) {
  const std::string key = std::string(layer) + "|" + std::string(selected) +
                          "|" + std::string(actual) + "|" +
                          std::string(reason.empty() ? "unspecified" : reason);
  std::lock_guard<std::mutex> lock(cuda_dispatch_divergence_mutex_);
  cuda_dispatch_divergence_counts_[key] += 1;
  cuda_dispatch_divergence_total_ += 1;
}

uint64_t MetricsRegistry::GetInferfluxCudaDispatchDivergences() const {
  std::lock_guard<std::mutex> lock(cuda_dispatch_divergence_mutex_);
  return cuda_dispatch_divergence_total_;
}

MetricsRegistry::CacheMetrics MetricsRegistry::GetCacheMetrics() const {
  CacheMetrics cm;
  cm.hits = prefix_hits_.load(std::memory_order_relaxed);
  cm.misses = prefix_misses_.load(std::memory_order_relaxed);
  cm.partial_hits = prefix_partial_hits_.load(std::memory_order_relaxed);
  cm.matched_tokens = prefix_matched_tokens_.load(std::memory_order_relaxed);
  cm.kv_reuse_count = kv_prefix_reuse_count_.load(std::memory_order_relaxed);
  cm.kv_reuse_tokens = kv_prefix_reuse_tokens_.load(std::memory_order_relaxed);
  return cm;
}

void MetricsRegistry::SetInferfluxCudaModelMemorySnapshot(
    std::string model_label, MemoryUsageMetrics total,
    std::unordered_map<std::string, MemoryUsageMetrics> domains) {
  std::lock_guard<std::mutex> lock(inferflux_cuda_model_memory_mutex_);
  inferflux_cuda_model_memory_snapshot_.model_label = std::move(model_label);
  inferflux_cuda_model_memory_snapshot_.total = total;
  inferflux_cuda_model_memory_snapshot_.domains = std::move(domains);
}

MetricsRegistry::InferfluxCudaModelMemorySnapshot
MetricsRegistry::GetInferfluxCudaModelMemorySnapshot() const {
  std::lock_guard<std::mutex> lock(inferflux_cuda_model_memory_mutex_);
  return inferflux_cuda_model_memory_snapshot_;
}

void MetricsRegistry::SetInferfluxCudaKvMemoryBytes(
    uint64_t total_bytes, uint64_t active_bytes, uint64_t prefix_retained_bytes,
    uint64_t free_bytes, int active_sequences, int prefix_retained_sequences,
    int free_sequences, int max_sequences) {
  inferflux_cuda_kv_memory_total_bytes_.store(total_bytes,
                                              std::memory_order_relaxed);
  inferflux_cuda_kv_memory_active_bytes_.store(active_bytes,
                                               std::memory_order_relaxed);
  inferflux_cuda_kv_memory_prefix_retained_bytes_.store(
      prefix_retained_bytes, std::memory_order_relaxed);
  inferflux_cuda_kv_memory_free_bytes_.store(free_bytes,
                                             std::memory_order_relaxed);
  inferflux_cuda_kv_memory_active_sequences_.store(
      std::max(0, active_sequences), std::memory_order_relaxed);
  inferflux_cuda_kv_memory_prefix_retained_sequences_.store(
      std::max(0, prefix_retained_sequences), std::memory_order_relaxed);
  inferflux_cuda_kv_memory_free_sequences_.store(std::max(0, free_sequences),
                                                 std::memory_order_relaxed);
  inferflux_cuda_kv_max_sequences_.store(std::max(0, max_sequences),
                                         std::memory_order_relaxed);
}

MetricsRegistry::InferfluxCudaKvMemorySnapshot
MetricsRegistry::GetInferfluxCudaKvMemorySnapshot() const {
  InferfluxCudaKvMemorySnapshot snapshot;
  snapshot.total_bytes =
      inferflux_cuda_kv_memory_total_bytes_.load(std::memory_order_relaxed);
  snapshot.active_bytes =
      inferflux_cuda_kv_memory_active_bytes_.load(std::memory_order_relaxed);
  snapshot.prefix_retained_bytes =
      inferflux_cuda_kv_memory_prefix_retained_bytes_.load(
          std::memory_order_relaxed);
  snapshot.free_bytes =
      inferflux_cuda_kv_memory_free_bytes_.load(std::memory_order_relaxed);
  snapshot.active_sequences = inferflux_cuda_kv_memory_active_sequences_.load(
      std::memory_order_relaxed);
  snapshot.prefix_retained_sequences =
      inferflux_cuda_kv_memory_prefix_retained_sequences_.load(
          std::memory_order_relaxed);
  snapshot.free_sequences =
      inferflux_cuda_kv_memory_free_sequences_.load(std::memory_order_relaxed);
  snapshot.max_sequences =
      inferflux_cuda_kv_max_sequences_.load(std::memory_order_relaxed);
  return snapshot;
}

std::string MetricsRegistry::RenderPrometheus() const {
  std::string backend;
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    backend = backend_;
  }
  std::ostringstream out;

  // --- Counters ---
  out << "# HELP inferflux_requests_total Total successful generation "
         "requests\n";
  out << "# TYPE inferflux_requests_total counter\n";
  out << "inferflux_requests_total{backend=\"" << backend << "\"} "
      << total_requests_.load() << "\n";

  out << "# HELP inferflux_errors_total Total generation errors\n";
  out << "# TYPE inferflux_errors_total counter\n";
  out << "inferflux_errors_total{backend=\"" << backend << "\"} "
      << total_errors_.load() << "\n";

  auto cuda_errors = diagnostics::GetCudaErrorCount();
  if (cuda_errors > 0) {
    out << "# HELP inferflux_cuda_errors_total Total CUDA API errors "
           "encountered\n";
    out << "# TYPE inferflux_cuda_errors_total counter\n";
    out << "inferflux_cuda_errors_total " << cuda_errors << "\n";
  }

  out << "# HELP inferflux_empty_generations_total Total backend generations "
         "that produced no completion text\n";
  out << "# TYPE inferflux_empty_generations_total counter\n";
  out << "inferflux_empty_generations_total{backend=\"" << backend << "\"} "
      << total_empty_generations_.load() << "\n";

  out << "# HELP inferflux_output_utf8_replacements_total Malformed or "
         "incomplete model-output byte sequences replaced with U+FFFD\n";
  out << "# TYPE inferflux_output_utf8_replacements_total counter\n";
  out << "inferflux_output_utf8_replacements_total{backend=\"" << backend
      << "\"} " << output_utf8_replacements_.load() << "\n";

  out << "# HELP inferflux_prompt_tokens_total Total prompt tokens processed\n";
  out << "# TYPE inferflux_prompt_tokens_total counter\n";
  out << "inferflux_prompt_tokens_total{backend=\"" << backend << "\"} "
      << total_prompt_tokens_.load() << "\n";

  out << "# HELP inferflux_completion_tokens_total Total completion tokens "
         "produced\n";
  out << "# TYPE inferflux_completion_tokens_total counter\n";
  out << "inferflux_completion_tokens_total{backend=\"" << backend << "\"} "
      << total_completion_tokens_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_total Total speculative chunks "
         "proposed\n";
  out << "# TYPE inferflux_spec_chunks_total counter\n";
  out << "inferflux_spec_chunks_total{backend=\"" << backend << "\"} "
      << speculative_chunks_total_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_accepted_total Speculative chunks "
         "accepted by validation\n";
  out << "# TYPE inferflux_spec_chunks_accepted_total counter\n";
  out << "inferflux_spec_chunks_accepted_total{backend=\"" << backend << "\"} "
      << speculative_chunks_accepted_.load() << "\n";

  out << "# HELP inferflux_spec_tokens_reused_total Draft tokens reused after "
         "validation\n";
  out << "# TYPE inferflux_spec_tokens_reused_total counter\n";
  out << "inferflux_spec_tokens_reused_total{backend=\"" << backend << "\"} "
      << speculative_tokens_reused_.load() << "\n";

  out << "# HELP inferflux_batches_total Scheduler batches processed\n";
  out << "# TYPE inferflux_batches_total counter\n";
  out << "inferflux_batches_total{backend=\"" << backend << "\"} "
      << total_batches_.load() << "\n";

  out << "# HELP inferflux_batch_tokens_total Total prompt tokens observed in "
         "batches\n";
  out << "# TYPE inferflux_batch_tokens_total counter\n";
  out << "inferflux_batch_tokens_total{backend=\"" << backend << "\"} "
      << total_batch_tokens_.load() << "\n";

  out << "# HELP inferflux_batch_size_max Largest batch size observed\n";
  out << "# TYPE inferflux_batch_size_max gauge\n";
  out << "inferflux_batch_size_max{backend=\"" << backend << "\"} "
      << max_batch_size_.load() << "\n";

  out << "# HELP inferflux_scheduler_iterations_total Scheduler iterations by "
         "phase composition\n";
  out << "# TYPE inferflux_scheduler_iterations_total counter\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"prefill\"} " << scheduler_iterations_prefill_.load()
      << "\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"decode\"} " << scheduler_iterations_decode_.load() << "\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"mixed\"} " << scheduler_iterations_mixed_.load() << "\n";

  out << "# HELP inferflux_scheduler_iteration_requests_total Requests touched "
         "by scheduler iterations\n";
  out << "# TYPE inferflux_scheduler_iteration_requests_total counter\n";
  out << "inferflux_scheduler_iteration_requests_total{backend=\"" << backend
      << "\"} " << scheduler_iteration_requests_total_.load() << "\n";

  out << "# HELP inferflux_scheduler_iteration_tokens_total Estimated token "
         "budget touched by scheduler iterations\n";
  out << "# TYPE inferflux_scheduler_iteration_tokens_total counter\n";
  out << "inferflux_scheduler_iteration_tokens_total{backend=\"" << backend
      << "\"} " << scheduler_iteration_tokens_total_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_token_budget_skips_total Requests "
         "deferred because adding them would exceed the current batch token "
         "budget\n";
  out << "# TYPE inferflux_scheduler_batch_token_budget_skips_total counter\n";
  out << "inferflux_scheduler_batch_token_budget_skips_total{backend=\""
      << backend << "\"} " << scheduler_batch_token_budget_skips_.load()
      << "\n";

  out << "# HELP inferflux_scheduler_policy_iterations_total Scheduler "
         "iterations by queue policy and phase composition\n";
  out << "# TYPE inferflux_scheduler_policy_iterations_total counter\n";
  {
    std::lock_guard<std::mutex> lock(scheduler_policy_metrics_mutex_);
    for (const auto &[key, count] : scheduler_policy_iterations_) {
      auto split = key.find('|');
      if (split == std::string::npos) {
        continue;
      }
      const std::string policy = key.substr(0, split);
      const std::string phase = key.substr(split + 1);
      out << "inferflux_scheduler_policy_iterations_total{backend=\"" << backend
          << "\",policy=\"" << policy << "\",phase=\"" << phase << "\"} "
          << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_prefix_affinity_probes_total Prefix-"
         "affinity probes performed during queue ranking\n";
  out << "# TYPE inferflux_scheduler_prefix_affinity_probes_total counter\n";
  out << "inferflux_scheduler_prefix_affinity_probes_total{backend=\""
      << backend << "\"} " << scheduler_prefix_affinity_probes_.load() << "\n";

  out << "# HELP inferflux_scheduler_prefix_affinity_hits_total Prefix-"
         "affinity probes that produced reusable hits\n";
  out << "# TYPE inferflux_scheduler_prefix_affinity_hits_total counter\n";
  out << "inferflux_scheduler_prefix_affinity_hits_total{backend=\"" << backend
      << "\"} " << scheduler_prefix_affinity_hits_.load() << "\n";

  out << "# HELP inferflux_scheduler_prefix_affinity_matched_tokens_total "
         "Matched tokens observed in prefix-affinity hits\n";
  out << "# TYPE inferflux_scheduler_prefix_affinity_matched_tokens_total "
         "counter\n";
  out << "inferflux_scheduler_prefix_affinity_matched_tokens_total{backend=\""
      << backend << "\"} " << scheduler_prefix_affinity_matched_tokens_.load()
      << "\n";

  out << "# HELP inferflux_scheduler_decode_step_loops_total Decode-step loop "
         "iterations executed by mode\n";
  out << "# TYPE inferflux_scheduler_decode_step_loops_total counter\n";
  {
    std::lock_guard<std::mutex> lock(scheduler_step_metrics_mutex_);
    for (const auto &[mode, count] : scheduler_decode_step_loops_) {
      out << "inferflux_scheduler_decode_step_loops_total{mode=\"" << mode
          << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_prefill_chunk_truncations_total Number "
         "of prefill chunks truncated by mixed-step token budgets\n";
  out << "# TYPE inferflux_scheduler_prefill_chunk_truncations_total counter\n";
  {
    std::lock_guard<std::mutex> lock(scheduler_step_metrics_mutex_);
    for (const auto &[mode, count] : scheduler_prefill_chunk_truncations_) {
      out << "inferflux_scheduler_prefill_chunk_truncations_total{mode=\""
          << mode << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_prefill_chunk_truncated_tokens_total "
         "Total tokens deferred due to prefill chunk truncation\n";
  out << "# TYPE inferflux_scheduler_prefill_chunk_truncated_tokens_total "
         "counter\n";
  {
    std::lock_guard<std::mutex> lock(scheduler_step_metrics_mutex_);
    for (const auto &[mode, count] :
         scheduler_prefill_chunk_truncated_tokens_) {
      out << "inferflux_scheduler_prefill_chunk_truncated_tokens_total{mode=\""
          << mode << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_deferred_sequence_retirements Current "
         "number of sequence slots waiting on backend release fences\n";
  out << "# TYPE inferflux_scheduler_deferred_sequence_retirements gauge\n";
  out << "inferflux_scheduler_deferred_sequence_retirements{backend=\""
      << backend << "\"} " << scheduler_deferred_sequence_retirements_.load()
      << "\n";

  out << "# HELP "
         "inferflux_scheduler_deferred_sequence_retirements_completed_total "
         "Sequence slot retirements completed after backend fence polling\n";
  out << "# TYPE "
         "inferflux_scheduler_deferred_sequence_retirements_completed_total "
         "counter\n";
  out << "inferflux_scheduler_deferred_sequence_retirements_completed_total{"
         "backend=\""
      << backend << "\"} "
      << scheduler_deferred_sequence_retirements_completed_.load() << "\n";

  out << "# HELP inferflux_scheduler_sequence_retirement_duration_ms Time "
         "between scheduling a deferred sequence retirement and backend "
         "release completion\n";
  out << "# TYPE inferflux_scheduler_sequence_retirement_duration_ms "
         "histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_scheduler_sequence_retirement_duration_ms_bucket{"
           "backend=\""
        << backend << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << scheduler_sequence_retirement_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_scheduler_sequence_retirement_duration_ms_bucket{backend=\""
      << backend << "\",le=\"+Inf\"} "
      << scheduler_sequence_retirement_latency_
             .counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_scheduler_sequence_retirement_duration_ms_sum{backend=\""
      << backend << "\"} "
      << scheduler_sequence_retirement_latency_.sum_ms.load() << "\n";
  out << "inferflux_scheduler_sequence_retirement_duration_ms_count{backend=\""
      << backend << "\"} "
      << scheduler_sequence_retirement_latency_.total.load() << "\n";

  out << "# HELP inferflux_disagg_kv_enqueue_rejections_total "
         "Prefill-to-decode "
         "KV transport enqueue rejections\n";
  out << "# TYPE inferflux_disagg_kv_enqueue_rejections_total counter\n";
  out << "inferflux_disagg_kv_enqueue_rejections_total{backend=\"" << backend
      << "\"} " << disagg_kv_enqueue_rejections_.load() << "\n";

  out << "# HELP inferflux_disagg_kv_enqueue_exhausted_total Requests that "
         "failed after distributed KV enqueue retries were exhausted\n";
  out << "# TYPE inferflux_disagg_kv_enqueue_exhausted_total counter\n";
  out << "inferflux_disagg_kv_enqueue_exhausted_total{backend=\"" << backend
      << "\"} " << disagg_kv_enqueue_exhausted_.load() << "\n";

  out << "# HELP inferflux_disagg_kv_tickets_total Distributed KV ticket "
         "lifecycle transitions by stage\n";
  out << "# TYPE inferflux_disagg_kv_tickets_total counter\n";
  out << "inferflux_disagg_kv_tickets_total{backend=\"" << backend
      << "\",stage=\"enqueued\"} " << disagg_kv_tickets_enqueued_.load()
      << "\n";
  out << "inferflux_disagg_kv_tickets_total{backend=\"" << backend
      << "\",stage=\"acknowledged\"} " << disagg_kv_tickets_acknowledged_.load()
      << "\n";
  out << "inferflux_disagg_kv_tickets_total{backend=\"" << backend
      << "\",stage=\"committed\"} " << disagg_kv_tickets_committed_.load()
      << "\n";
  out << "inferflux_disagg_kv_tickets_total{backend=\"" << backend
      << "\",stage=\"timed_out\"} " << disagg_kv_tickets_timed_out_.load()
      << "\n";
  out << "# HELP inferflux_disagg_kv_timeout_streak Consecutive distributed KV "
         "ticket timeouts since the last committed handoff\n";
  out << "# TYPE inferflux_disagg_kv_timeout_streak gauge\n";
  out << "inferflux_disagg_kv_timeout_streak{backend=\"" << backend << "\"} "
      << disagg_kv_timeout_streak_.load() << "\n";
  out << "# HELP inferflux_disagg_kv_timeout_debt Distributed KV timeout debt "
         "(timeouts minus debt-repaying committed handoffs)\n";
  out << "# TYPE inferflux_disagg_kv_timeout_debt gauge\n";
  out << "inferflux_disagg_kv_timeout_debt{backend=\"" << backend << "\"} "
      << disagg_kv_timeout_debt_.load() << "\n";

  out << "# HELP inferflux_fairness_preemptions_total Scheduler swaps "
         "triggered by fairness\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_yields_total Requests yielded "
         "mid-generation due to fairness timeslices\n";
  out << "# TYPE inferflux_fairness_yields_total counter\n";
  out << "inferflux_fairness_yields_total{backend=\"" << backend << "\"} "
      << fairness_yields_.load() << "\n";

  out << "# HELP inferflux_fairness_resumes_total Yielded requests rescheduled "
         "for completion\n";
  out << "# TYPE inferflux_fairness_resumes_total counter\n";
  out << "inferflux_fairness_resumes_total{backend=\"" << backend << "\"} "
      << fairness_resumes_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority "
         "level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto &[priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{priority=\"" << priority << "\"} "
          << tokens << "\n";
    }
  }

  // Queue wait histogram.
  out << "# HELP inferflux_queue_wait_duration_ms Time requests spend waiting "
         "before execution\n";
  out << "# TYPE inferflux_queue_wait_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << queue_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << queue_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_queue_wait_duration_ms_sum{backend=\"" << backend << "\"} "
      << queue_latency_.sum_ms.load() << "\n";
  out << "inferflux_queue_wait_duration_ms_count{backend=\"" << backend
      << "\"} " << queue_latency_.total.load() << "\n";

  out << "# HELP inferflux_batch_exec_duration_ms Time to execute a batch\n";
  out << "# TYPE inferflux_batch_exec_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << batch_exec_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << batch_exec_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_batch_exec_duration_ms_sum{backend=\"" << backend << "\"} "
      << batch_exec_latency_.sum_ms.load() << "\n";
  out << "inferflux_batch_exec_duration_ms_count{backend=\"" << backend
      << "\"} " << batch_exec_latency_.total.load() << "\n";

  out << "# HELP inferflux_prefix_hits_total Prefix cache hits\n";
  out << "# TYPE inferflux_prefix_hits_total counter\n";
  out << "inferflux_prefix_hits_total{backend=\"" << backend << "\"} "
      << prefix_hits_.load() << "\n";

  out << "# HELP inferflux_prefix_misses_total Prefix cache misses\n";
  out << "# TYPE inferflux_prefix_misses_total counter\n";
  out << "inferflux_prefix_misses_total{backend=\"" << backend << "\"} "
      << prefix_misses_.load() << "\n";

  out << "# HELP inferflux_prefix_matched_tokens_total Tokens matched in "
         "prefix (partial or exact)\n";
  out << "# TYPE inferflux_prefix_matched_tokens_total counter\n";
  out << "inferflux_prefix_matched_tokens_total{backend=\"" << backend << "\"} "
      << prefix_matched_tokens_.load() << "\n";

  out << "# HELP inferflux_prefix_partial_hits_total Lookups with a partial "
         "prefix match but no exact hit\n";
  out << "# TYPE inferflux_prefix_partial_hits_total counter\n";
  out << "inferflux_prefix_partial_hits_total{backend=\"" << backend << "\"} "
      << prefix_partial_hits_.load() << "\n";

  out << "# HELP inferflux_kv_prefix_reuse_total Requests that reused a warm "
         "KV prefix (skipped full Prefill)\n";
  out << "# TYPE inferflux_kv_prefix_reuse_total counter\n";
  out << "inferflux_kv_prefix_reuse_total " << kv_prefix_reuse_count_.load()
      << "\n";
  out << "# HELP inferflux_kv_prefix_reuse_tokens_total Prompt tokens skipped "
         "via KV prefix reuse\n";
  out << "# TYPE inferflux_kv_prefix_reuse_tokens_total counter\n";
  out << "inferflux_kv_prefix_reuse_tokens_total "
      << kv_prefix_reuse_tokens_.load() << "\n";

  out << "# HELP inferflux_stream_tokens_total Tokens streamed via SSE\n";
  out << "# TYPE inferflux_stream_tokens_total counter\n";
  out << "inferflux_stream_tokens_total{backend=\"" << backend << "\"} "
      << stream_tokens_.load() << "\n";

  out << "# HELP inferflux_stream_cache_hits_total SSE completions served "
         "entirely from cache\n";
  out << "# TYPE inferflux_stream_cache_hits_total counter\n";
  out << "inferflux_stream_cache_hits_total{backend=\"" << backend << "\"} "
      << stream_cache_hits_.load() << "\n";

  out << "# HELP inferflux_capability_rejections_total Requests rejected due "
         "to unsupported backend/model features\n";
  out << "# TYPE inferflux_capability_rejections_total counter\n";
  {
    std::lock_guard<std::mutex> lock(capability_metrics_mutex_);
    for (const auto &[key, count] : capability_rejections_) {
      auto split = key.find('|');
      std::string label_backend =
          split == std::string::npos ? backend : key.substr(0, split);
      std::string label_feature =
          split == std::string::npos ? key : key.substr(split + 1);
      out << "inferflux_capability_rejections_total{backend=\"" << label_backend
          << "\",feature=\"" << label_feature << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_backend_exposures_total Model backend exposure "
         "decisions by requested/selected provider path\n";
  out << "# TYPE inferflux_backend_exposures_total counter\n";
  {
    std::lock_guard<std::mutex> lock(backend_exposure_mutex_);
    for (const auto &[key, count] : backend_exposure_counts_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      auto p3 =
          (p2 == std::string::npos) ? std::string::npos : key.find('|', p2 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos ||
          p3 == std::string::npos) {
        continue;
      }
      const std::string requested = key.substr(0, p1);
      const std::string exposed = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string provider = key.substr(p2 + 1, p3 - p2 - 1);
      const std::string fallback = key.substr(p3 + 1);
      out << "inferflux_backend_exposures_total{requested_backend=\""
          << requested << "\",exposed_backend=\"" << exposed << "\",provider=\""
          << provider << "\",fallback=\"" << fallback << "\"} " << count
          << "\n";
    }
  }

  out << "# HELP inferflux_capability_route_fallbacks_total Requests rerouted "
         "to another backend due to unsupported features or backend "
         "unavailability\n";
  out << "# TYPE inferflux_capability_route_fallbacks_total counter\n";
  {
    std::lock_guard<std::mutex> lock(capability_route_fallback_mutex_);
    for (const auto &[key, count] : capability_route_fallbacks_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) {
        continue;
      }
      const std::string from = key.substr(0, p1);
      const std::string to = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string feature = key.substr(p2 + 1);
      out << "inferflux_capability_route_fallbacks_total{from_backend=\""
          << from << "\",to_backend=\"" << to << "\",feature=\"" << feature
          << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_fairness_preemptions_total Number of fairness "
         "preemptions performed\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority "
         "level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto &[priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{backend=\"" << backend
          << "\",priority=\"" << priority << "\"} " << tokens << "\n";
    }
  }

  out << "# HELP inferflux_model_routes_total Total routing decisions per "
         "model\n";
  out << "# TYPE inferflux_model_routes_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_routes_total{model=\"" << model << "\",backend=\""
          << label << "\"} " << stats.routes << "\n";
    }
  }

  out << "# HELP inferflux_model_route_misses_total Requests that could not be "
         "routed to a model\n";
  out << "# TYPE inferflux_model_route_misses_total counter\n";
  out << "inferflux_model_route_misses_total " << model_route_misses_.load()
      << "\n";

  out << "# HELP inferflux_model_load_seconds_total Cumulative time spent "
         "loading models\n";
  out << "# TYPE inferflux_model_load_seconds_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_seconds_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.load_seconds << "\n";
    }
  }

  out << "# HELP inferflux_model_load_events_total Number of times a model was "
         "loaded\n";
  out << "# TYPE inferflux_model_load_events_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_events_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.load_events << "\n";
    }
  }

  out << "# HELP inferflux_model_ready Model readiness (1=ready)\n";
  out << "# TYPE inferflux_model_ready gauge\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_ready{model=\"" << model << "\",backend=\""
          << label << "\"} " << (stats.ready ? 1 : 0) << "\n";
    }
  }

  out << "# HELP inferflux_model_prompt_tokens_total Prompt tokens processed "
         "per model (use rate(...) for prompt throughput)\n";
  out << "# TYPE inferflux_model_prompt_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_prompt_tokens_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.prompt_tokens << "\n";
    }
  }

  out << "# HELP inferflux_model_completion_tokens_total Completion tokens "
         "produced per model (use rate(...) for generation throughput)\n";
  out << "# TYPE inferflux_model_completion_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_completion_tokens_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.completion_tokens
          << "\n";
    }
  }

  // --- Request latency histogram ---
  out << "# HELP inferflux_request_duration_ms Request end-to-end latency in "
         "milliseconds\n";
  out << "# TYPE inferflux_request_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << request_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << request_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_request_duration_ms_sum{backend=\"" << backend << "\"} "
      << request_latency_.sum_ms.load() << "\n";
  out << "inferflux_request_duration_ms_count{backend=\"" << backend << "\"} "
      << request_latency_.total.load() << "\n";

  // --- OBS-2: per-phase latency histograms ---
  out << "# HELP inferflux_prefill_duration_ms Prefill phase latency "
         "(tokenization + prompt eval)\n";
  out << "# TYPE inferflux_prefill_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << prefill_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << prefill_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_prefill_duration_ms_sum{backend=\"" << backend << "\"} "
      << prefill_latency_.sum_ms.load() << "\n";
  out << "inferflux_prefill_duration_ms_count{backend=\"" << backend << "\"} "
      << prefill_latency_.total.load() << "\n";

  out << "# HELP inferflux_decode_duration_ms Decode phase latency (token "
         "generation)\n";
  out << "# TYPE inferflux_decode_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << decode_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << decode_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_decode_duration_ms_sum{backend=\"" << backend << "\"} "
      << decode_latency_.sum_ms.load() << "\n";
  out << "inferflux_decode_duration_ms_count{backend=\"" << backend << "\"} "
      << decode_latency_.total.load() << "\n";

  // --- KV transfer latency (§2.5 item 12) ---
  out << "# HELP inferflux_kv_transfer_duration_ms Prefill-to-decode KV "
         "hand-off latency via KVChannel/ShmKVTransport\n";
  out << "# TYPE inferflux_kv_transfer_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_kv_transfer_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << kv_transfer_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_kv_transfer_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << kv_transfer_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_kv_transfer_duration_ms_sum{backend=\"" << backend << "\"} "
      << kv_transfer_latency_.sum_ms.load() << "\n";
  out << "inferflux_kv_transfer_duration_ms_count{backend=\"" << backend
      << "\"} " << kv_transfer_latency_.total.load() << "\n";

  // --- GGML-native perf (llama_perf_context) ---
  out << "# HELP inferflux_llama_prefill_ms GGML-native prompt eval latency "
         "(from llama_perf_context)\n";
  out << "# TYPE inferflux_llama_prefill_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_llama_prefill_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << llama_prefill_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_llama_prefill_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << llama_prefill_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_llama_prefill_ms_sum{backend=\"" << backend << "\"} "
      << llama_prefill_latency_.sum_ms.load() << "\n";
  out << "inferflux_llama_prefill_ms_count{backend=\"" << backend << "\"} "
      << llama_prefill_latency_.total.load() << "\n";

  out << "# HELP inferflux_llama_decode_ms GGML-native token generation "
         "latency "
         "(from llama_perf_context)\n";
  out << "# TYPE inferflux_llama_decode_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_llama_decode_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << llama_decode_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_llama_decode_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << llama_decode_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_llama_decode_ms_sum{backend=\"" << backend << "\"} "
      << llama_decode_latency_.sum_ms.load() << "\n";
  out << "inferflux_llama_decode_ms_count{backend=\"" << backend << "\"} "
      << llama_decode_latency_.total.load() << "\n";

  out << "# HELP inferflux_llama_prompt_tokens_total Prompt tokens processed "
         "per llama_perf_context\n";
  out << "# TYPE inferflux_llama_prompt_tokens_total counter\n";
  out << "inferflux_llama_prompt_tokens_total{backend=\"" << backend << "\"} "
      << llama_prompt_tokens_.load() << "\n";

  out << "# HELP inferflux_llama_generated_tokens_total Tokens generated per "
         "llama_perf_context\n";
  out << "# TYPE inferflux_llama_generated_tokens_total counter\n";
  out << "inferflux_llama_generated_tokens_total{backend=\"" << backend
      << "\"} " << llama_gen_tokens_.load() << "\n";

  // --- Multimodal (§2.2) ---
  out << "# HELP inferflux_multimodal_images_total Total images preprocessed "
         "from image_url parts\n";
  out << "# TYPE inferflux_multimodal_images_total counter\n";
  out << "inferflux_multimodal_images_total " << multimodal_images_.load()
      << "\n";

  out << "# HELP inferflux_multimodal_requests_total Total requests containing "
         "image_url parts\n";
  out << "# TYPE inferflux_multimodal_requests_total counter\n";
  out << "inferflux_multimodal_requests_total " << multimodal_requests_.load()
      << "\n";

  // --- MoE (§2.6) ---
  out << "# HELP inferflux_moe_requests_total Requests dispatched to MoE "
         "models\n";
  out << "# TYPE inferflux_moe_requests_total counter\n";
  out << "inferflux_moe_requests_total " << moe_requests_.load() << "\n";

  // --- Flash Attention (§2.7) ---
  out << "# HELP inferflux_flash_attention_enabled Flash Attention active for "
         "this instance (0=disabled, 1=enabled)\n";
  out << "# TYPE inferflux_flash_attention_enabled gauge\n";
  out << "inferflux_flash_attention_enabled " << flash_attention_enabled_.load()
      << "\n";

  out << "# HELP inferflux_flash_attention_requests_total Total requests "
         "processed "
         "by Flash Attention kernel type\n";
  out << "# TYPE inferflux_flash_attention_requests_total counter\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"fa2\"} "
      << flash_attention_requests_fa2_.load() << "\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"fa3\"} "
      << flash_attention_requests_fa3_.load() << "\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"standard\"} "
      << flash_attention_requests_standard_.load() << "\n";

  out << "# HELP inferflux_flash_attention_memory_mb Flash Attention KV cache "
         "memory usage in megabytes\n";
  out << "# TYPE inferflux_flash_attention_memory_mb gauge\n";
  out << "inferflux_flash_attention_memory_mb "
      << flash_attention_memory_mb_.load() << "\n";

  out << "# HELP inferflux_flash_attention_execution_ms Flash Attention kernel "
         "execution time in milliseconds\n";
  out << "# TYPE inferflux_flash_attention_execution_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_flash_attention_execution_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << flash_attention_exec_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_flash_attention_execution_ms_bucket{le=\"+Inf\"} "
      << flash_attention_exec_latency_.counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_flash_attention_execution_ms_sum "
      << flash_attention_exec_latency_.sum_ms.load() << "\n";
  out << "inferflux_flash_attention_execution_ms_count "
      << flash_attention_exec_latency_.total.load() << "\n";

  // --- Gauges ---
  out << "# HELP inferflux_active_connections Current number of active HTTP "
         "connections\n";
  out << "# TYPE inferflux_active_connections gauge\n";
  out << "inferflux_active_connections " << active_connections_.load() << "\n";

  out << "# HELP inferflux_scheduler_queue_depth Current number of requests in "
         "the scheduler queue\n";
  out << "# TYPE inferflux_scheduler_queue_depth gauge\n";
  out << "inferflux_scheduler_queue_depth " << queue_depth_.load() << "\n";

  out << "# HELP inferflux_prefill_queue_depth Current number of tickets "
         "waiting in the prefill pool\n";
  out << "# TYPE inferflux_prefill_queue_depth gauge\n";
  out << "inferflux_prefill_queue_depth " << prefill_queue_depth_.load()
      << "\n";

  out << "# HELP inferflux_decode_queue_depth Current number of tickets "
         "waiting in the decode pool\n";
  out << "# TYPE inferflux_decode_queue_depth gauge\n";
  out << "inferflux_decode_queue_depth " << decode_queue_depth_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_limit_size Configured scheduler "
         "max requests per batch\n";
  out << "# TYPE inferflux_scheduler_batch_limit_size gauge\n";
  out << "inferflux_scheduler_batch_limit_size "
      << scheduler_batch_limit_size_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_limit_tokens Configured scheduler "
         "max prompt tokens per batch\n";
  out << "# TYPE inferflux_scheduler_batch_limit_tokens gauge\n";
  out << "inferflux_scheduler_batch_limit_tokens "
      << scheduler_batch_limit_tokens_.load() << "\n";

  out << "# HELP inferflux_scheduler_decode_worker_batch_size_total Decode "
         "worker "
         "batch-size distribution\n";
  out << "# TYPE inferflux_scheduler_decode_worker_batch_size_total counter\n";
  {
    std::vector<std::string> buckets;
    {
      std::lock_guard<std::mutex> lock(
          scheduler_decode_worker_batch_size_mutex_);
      buckets.reserve(scheduler_decode_worker_batch_size_counts_.size());
      for (const auto &entry : scheduler_decode_worker_batch_size_counts_) {
        buckets.push_back(entry.first);
      }
    }
    std::sort(buckets.begin(), buckets.end(),
              [](const std::string &a, const std::string &b) {
                return std::stoll(a) < std::stoll(b);
              });
    for (const auto &bucket : buckets) {
      uint64_t count = 0;
      {
        std::lock_guard<std::mutex> lock(
            scheduler_decode_worker_batch_size_mutex_);
        const auto it = scheduler_decode_worker_batch_size_counts_.find(bucket);
        if (it != scheduler_decode_worker_batch_size_counts_.end()) {
          count = it->second;
        }
      }
      out << "inferflux_scheduler_decode_worker_batch_size_total{bucket=\""
          << bucket << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_decode_worker_execution_path_total Decode "
         "worker execution-path selections\n";
  out << "# TYPE inferflux_scheduler_decode_worker_execution_path_total "
         "counter\n";
  {
    std::vector<std::string> paths;
    {
      std::lock_guard<std::mutex> lock(
          scheduler_decode_worker_batch_size_mutex_);
      paths.reserve(scheduler_decode_worker_execution_path_counts_.size());
      for (const auto &entry : scheduler_decode_worker_execution_path_counts_) {
        paths.push_back(entry.first);
      }
    }
    std::sort(paths.begin(), paths.end());
    for (const auto &path : paths) {
      uint64_t count = 0;
      {
        std::lock_guard<std::mutex> lock(
            scheduler_decode_worker_batch_size_mutex_);
        const auto it =
            scheduler_decode_worker_execution_path_counts_.find(path);
        if (it != scheduler_decode_worker_execution_path_counts_.end()) {
          count = it->second;
        }
      }
      out << "inferflux_scheduler_decode_worker_execution_path_total{path=\""
          << path << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_decode_worker_sticky_merge_total "
         "Compatible decode requests merged into an active sticky cohort by "
         "merge size\n";
  out << "# TYPE inferflux_scheduler_decode_worker_sticky_merge_total "
         "counter\n";
  {
    std::vector<std::string> merge_sizes;
    {
      std::lock_guard<std::mutex> lock(
          scheduler_decode_worker_batch_size_mutex_);
      merge_sizes.reserve(scheduler_decode_worker_sticky_merge_counts_.size());
      for (const auto &entry : scheduler_decode_worker_sticky_merge_counts_) {
        merge_sizes.push_back(entry.first);
      }
    }
    std::sort(merge_sizes.begin(), merge_sizes.end(),
              [](const std::string &a, const std::string &b) {
                return std::stoll(a) < std::stoll(b);
              });
    for (const auto &merge_size : merge_sizes) {
      uint64_t count = 0;
      {
        std::lock_guard<std::mutex> lock(
            scheduler_decode_worker_batch_size_mutex_);
        const auto it =
            scheduler_decode_worker_sticky_merge_counts_.find(merge_size);
        if (it != scheduler_decode_worker_sticky_merge_counts_.end()) {
          count = it->second;
        }
      }
      out << "inferflux_scheduler_decode_worker_sticky_merge_total{merged=\""
          << merge_size << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_scheduler_decode_worker_sticky_merged_requests_"
         "total Total compatible decode requests merged into active sticky "
         "cohorts\n";
  out << "# TYPE inferflux_scheduler_decode_worker_sticky_merged_requests_"
         "total counter\n";
  out << "inferflux_scheduler_decode_worker_sticky_merged_requests_total "
      << scheduler_decode_worker_sticky_merged_requests_total_.load(
             std::memory_order_relaxed)
      << "\n";

  auto render_decode_assembly_counter =
      [&](const char *metric_name, const char *help,
          const std::unordered_map<std::string, uint64_t> &counts) {
        out << "# HELP " << metric_name << " " << help << "\n";
        out << "# TYPE " << metric_name << " counter\n";
        std::vector<std::pair<std::string, uint64_t>> entries;
        {
          std::lock_guard<std::mutex> lock(
              scheduler_decode_worker_batch_size_mutex_);
          entries.reserve(counts.size());
          for (const auto &entry : counts) {
            entries.push_back(entry);
          }
        }
        std::sort(
            entries.begin(), entries.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &[key, count] : entries) {
          const auto sep = key.find('|');
          const std::string mode =
              sep == std::string::npos ? "unknown" : key.substr(0, sep);
          const std::string bucket =
              sep == std::string::npos ? key : key.substr(sep + 1);
          out << metric_name << "{mode=\"" << mode << "\",bucket=\"" << bucket
              << "\"} " << count << "\n";
        }
      };

  render_decode_assembly_counter(
      "inferflux_scheduler_decode_assembly_ready_total",
      "Decode-ready request depth observed while assembling decode cohorts",
      scheduler_decode_assembly_ready_);
  render_decode_assembly_counter(
      "inferflux_scheduler_decode_assembly_selected_total",
      "Decode requests selected into the active cohort while assembling decode "
      "work",
      scheduler_decode_assembly_selected_);
  render_decode_assembly_counter(
      "inferflux_scheduler_decode_assembly_compatible_total",
      "Decode-ready requests compatible with the active sticky cohort",
      scheduler_decode_assembly_compatible_);
  render_decode_assembly_counter(
      "inferflux_scheduler_decode_assembly_incompatible_total",
      "Decode-ready requests incompatible with the active sticky cohort",
      scheduler_decode_assembly_incompatible_);

  // --- InferFlux CUDA backend metrics ---
  out << "# HELP inferflux_cuda_forward_passes_total InferFlux CUDA forward "
         "passes by phase\n";
  out << "# TYPE inferflux_cuda_forward_passes_total counter\n";
  out << "inferflux_cuda_forward_passes_total{phase=\"prefill\"} "
      << inferflux_cuda_forward_prefill_total_.load() << "\n";
  out << "inferflux_cuda_forward_passes_total{phase=\"decode\"} "
      << inferflux_cuda_forward_decode_total_.load() << "\n";

  out << "# HELP inferflux_cuda_forward_batch_tokens_total Total tokens "
         "processed by InferFlux CUDA forward passes\n";
  out << "# TYPE inferflux_cuda_forward_batch_tokens_total counter\n";
  out << "inferflux_cuda_forward_batch_tokens_total "
      << inferflux_cuda_forward_batch_tokens_total_.load() << "\n";

  out << "# HELP inferflux_cuda_forward_batch_size_total InferFlux CUDA "
         "forward "
         "batch-size distribution by phase\n";
  out << "# TYPE inferflux_cuda_forward_batch_size_total counter\n";
  {
    static constexpr const char *kPrefillBuckets[] = {
        "1", "2", "3_4", "5_8", "9_16", "17_32", "33_64", "65_128", "129_plus"};
    static constexpr const char *kDecodeBuckets[] = {"1",   "2",    "3_4",
                                                     "5_8", "9_16", "17_plus"};
    std::lock_guard<std::mutex> lock(inferflux_cuda_forward_batch_size_mutex_);
    for (const char *bucket : kPrefillBuckets) {
      const std::string key = std::string("prefill|") + bucket;
      const auto it = inferflux_cuda_forward_batch_size_counts_.find(key);
      const uint64_t count =
          it == inferflux_cuda_forward_batch_size_counts_.end() ? 0
                                                                : it->second;
      out << "inferflux_cuda_forward_batch_size_total{phase=\"prefill\",bucket="
             "\""
          << bucket << "\"} " << count << "\n";
    }
    for (const char *bucket : kDecodeBuckets) {
      const std::string key = std::string("decode|") + bucket;
      const auto it = inferflux_cuda_forward_batch_size_counts_.find(key);
      const uint64_t count =
          it == inferflux_cuda_forward_batch_size_counts_.end() ? 0
                                                                : it->second;
      out << "inferflux_cuda_forward_batch_size_total{phase=\"decode\",bucket="
             "\""
          << bucket << "\"} " << count << "\n";
    }
    for (const char *bucket : kDecodeBuckets) {
      const std::string key = std::string("unknown|") + bucket;
      const auto it = inferflux_cuda_forward_batch_size_counts_.find(key);
      const uint64_t count =
          it == inferflux_cuda_forward_batch_size_counts_.end() ? 0
                                                                : it->second;
      out << "inferflux_cuda_forward_batch_size_total{phase=\"unknown\",bucket="
             "\""
          << bucket << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_cuda_burst_decode_chunks_total InferFlux CUDA "
         "device-side burst decode chunks completed by phase\n";
  out << "# TYPE inferflux_cuda_burst_decode_chunks_total counter\n";
  out << "inferflux_cuda_burst_decode_chunks_total{phase=\"decode\"} "
      << inferflux_cuda_burst_decode_chunks_decode_total_.load() << "\n";
  out << "inferflux_cuda_burst_decode_chunks_total{phase=\"generate\"} "
      << inferflux_cuda_burst_decode_chunks_generate_total_.load() << "\n";

  out << "# HELP inferflux_cuda_burst_decode_requested_tokens_total Requested "
         "tokens passed into InferFlux CUDA burst decode chunks by phase\n";
  out << "# TYPE inferflux_cuda_burst_decode_requested_tokens_total counter\n";
  out << "inferflux_cuda_burst_decode_requested_tokens_total{phase=\"decode\"} "
      << inferflux_cuda_burst_decode_requested_tokens_decode_total_.load()
      << "\n";
  out << "inferflux_cuda_burst_decode_requested_tokens_total{phase="
         "\"generate\"} "
      << inferflux_cuda_burst_decode_requested_tokens_generate_total_.load()
      << "\n";

  out << "# HELP inferflux_cuda_burst_decode_produced_tokens_total Produced "
         "tokens returned by InferFlux CUDA burst decode chunks by phase\n";
  out << "# TYPE inferflux_cuda_burst_decode_produced_tokens_total counter\n";
  out << "inferflux_cuda_burst_decode_produced_tokens_total{phase=\"decode\"} "
      << inferflux_cuda_burst_decode_produced_tokens_decode_total_.load()
      << "\n";
  out << "inferflux_cuda_burst_decode_produced_tokens_total{phase=\"generate\"}"
         " "
      << inferflux_cuda_burst_decode_produced_tokens_generate_total_.load()
      << "\n";

  out << "# HELP inferflux_cuda_burst_decode_fallbacks_total Burst decode "
         "attempts that fell back to the standard token path by phase\n";
  out << "# TYPE inferflux_cuda_burst_decode_fallbacks_total counter\n";
  out << "inferflux_cuda_burst_decode_fallbacks_total{phase=\"decode\"} "
      << inferflux_cuda_burst_decode_fallbacks_decode_total_.load() << "\n";
  out << "inferflux_cuda_burst_decode_fallbacks_total{phase=\"generate\"} "
      << inferflux_cuda_burst_decode_fallbacks_generate_total_.load() << "\n";

  out << "# HELP inferflux_cuda_burst_decode_ineligible_total Burst decode "
         "requests rejected before launch by phase and reason\n";
  out << "# TYPE inferflux_cuda_burst_decode_ineligible_total counter\n";
  {
    std::vector<std::string> keys;
    {
      std::lock_guard<std::mutex> lock(
          inferflux_cuda_burst_decode_ineligible_mutex_);
      keys.reserve(inferflux_cuda_burst_decode_ineligible_counts_.size());
      for (const auto &entry : inferflux_cuda_burst_decode_ineligible_counts_) {
        keys.push_back(entry.first);
      }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const auto &key : keys) {
      const auto delimiter = key.find('|');
      if (delimiter == std::string::npos) {
        continue;
      }
      uint64_t count = 0;
      {
        std::lock_guard<std::mutex> lock(
            inferflux_cuda_burst_decode_ineligible_mutex_);
        const auto it =
            inferflux_cuda_burst_decode_ineligible_counts_.find(key);
        if (it != inferflux_cuda_burst_decode_ineligible_counts_.end()) {
          count = it->second;
        }
      }
      out << "inferflux_cuda_burst_decode_ineligible_total{phase=\""
          << key.substr(0, delimiter) << "\",reason=\""
          << key.substr(delimiter + 1) << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_cuda_ffn_proj_operator_total InferFlux CUDA FFN "
         "gate+up projection operator selections by phase and operator\n";
  out << "# TYPE inferflux_cuda_ffn_proj_operator_total counter\n";
  {
    static constexpr const char *kPhases[] = {"prefill", "decode"};
    // Operator universe derived from dispatch_catalog.h — plus any key
    // observed at runtime, so a new label can never be silently dropped
    // from exposition again.
    std::set<std::string> ops;
    for (const char *label : kFfnMetricLabels) {
      ops.insert(label);
    }
    {
      std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
      for (const auto &kv : inferflux_cuda_ffn_proj_operator_counts_) {
        const std::string::size_type bar = kv.first.find('|');
        if (bar != std::string::npos) {
          ops.insert(kv.first.substr(bar + 1));
        }
      }
    }
    std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
    for (const char *phase : kPhases) {
      for (const auto &op : ops) {
        const std::string key = std::string(phase) + "|" + op;
        const auto it = inferflux_cuda_ffn_proj_operator_counts_.find(key);
        const uint64_t count =
            it == inferflux_cuda_ffn_proj_operator_counts_.end() ? 0
                                                                 : it->second;
        out << "inferflux_cuda_ffn_proj_operator_total{phase=\"" << phase
            << "\",operator=\"" << op << "\"} " << count << "\n";
      }
    }
  }

  out << "# HELP inferflux_cuda_down_proj_operator_total InferFlux CUDA "
         "down-proj "
         "operator selections by phase and operator\n";
  out << "# TYPE inferflux_cuda_down_proj_operator_total counter\n";
  {
    static constexpr const char *kPhases[] = {"prefill", "decode"};
    // Operator universe derived from dispatch_catalog.h plus any runtime
    // observed key — a new label can never be silently unexported.
    std::set<std::string> ops;
    for (const char *label : kDownMetricLabels) {
      ops.insert(label);
    }
    {
      std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
      for (const auto &kv : inferflux_cuda_down_proj_operator_counts_) {
        const std::string::size_type bar = kv.first.find('|');
        if (bar != std::string::npos) {
          ops.insert(kv.first.substr(bar + 1));
        }
      }
    }
    std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
    for (const char *phase : kPhases) {
      for (const auto &op : ops) {
        const std::string key = std::string(phase) + "|" + op;
        const auto it = inferflux_cuda_down_proj_operator_counts_.find(key);
        const uint64_t count =
            it == inferflux_cuda_down_proj_operator_counts_.end() ? 0
                                                                  : it->second;
        out << "inferflux_cuda_down_proj_operator_total{phase=\"" << phase
            << "\",operator=\"" << op << "\"} " << count << "\n";
      }
    }
  }

  out << "# HELP inferflux_cuda_ffn_proj_geometry_total InferFlux CUDA FFN "
         "gate+up "
         "projection geometry selections by phase, operator, quant, and M/N/K "
         "shape buckets\n";
  out << "# TYPE inferflux_cuda_ffn_proj_geometry_total counter\n";
  {
    std::vector<std::pair<std::string, uint64_t>> entries;
    {
      std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
      entries.assign(inferflux_cuda_ffn_proj_geometry_counts_.begin(),
                     inferflux_cuda_ffn_proj_geometry_counts_.end());
    }
    std::sort(
        entries.begin(), entries.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &entry : entries) {
      std::istringstream key(entry.first);
      std::string phase, op, quant, m_bucket, n_exact, n_bucket, k_exact,
          k_bucket, grouped_bucket;
      if (!std::getline(key, phase, '|') || !std::getline(key, op, '|') ||
          !std::getline(key, quant, '|') || !std::getline(key, m_bucket, '|') ||
          !std::getline(key, n_exact, '|') ||
          !std::getline(key, n_bucket, '|') ||
          !std::getline(key, k_exact, '|') ||
          !std::getline(key, k_bucket, '|') ||
          !std::getline(key, grouped_bucket, '|')) {
        continue;
      }
      out << "inferflux_cuda_ffn_proj_geometry_total{phase=\"" << phase
          << "\",operator=\"" << op << "\",quant=\"" << quant
          << "\",m_bucket=\"" << m_bucket << "\",n=\"" << n_exact
          << "\",n_bucket=\"" << n_bucket << "\",k=\"" << k_exact
          << "\",k_bucket=\"" << k_bucket << "\",grouped_outputs=\""
          << grouped_bucket << "\"} " << entry.second << "\n";
    }
  }

  out << "# HELP inferflux_cuda_down_proj_geometry_total InferFlux CUDA "
         "down-proj "
         "geometry selections by phase, operator, quant, and M/N/K shape "
         "buckets\n";
  out << "# TYPE inferflux_cuda_down_proj_geometry_total counter\n";
  {
    std::vector<std::pair<std::string, uint64_t>> entries;
    {
      std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
      entries.assign(inferflux_cuda_down_proj_geometry_counts_.begin(),
                     inferflux_cuda_down_proj_geometry_counts_.end());
    }
    std::sort(
        entries.begin(), entries.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &entry : entries) {
      std::istringstream key(entry.first);
      std::string phase, op, quant, m_bucket, n_exact, n_bucket, k_exact,
          k_bucket;
      if (!std::getline(key, phase, '|') || !std::getline(key, op, '|') ||
          !std::getline(key, quant, '|') || !std::getline(key, m_bucket, '|') ||
          !std::getline(key, n_exact, '|') ||
          !std::getline(key, n_bucket, '|') ||
          !std::getline(key, k_exact, '|') ||
          !std::getline(key, k_bucket, '|')) {
        continue;
      }
      out << "inferflux_cuda_down_proj_geometry_total{phase=\"" << phase
          << "\",operator=\"" << op << "\",quant=\"" << quant
          << "\",m_bucket=\"" << m_bucket << "\",n=\"" << n_exact
          << "\",n_bucket=\"" << n_bucket << "\",k=\"" << k_exact
          << "\",k_bucket=\"" << k_bucket << "\"} " << entry.second << "\n";
    }
  }

  out << "# HELP inferflux_cuda_rowpair_selection_total Row-pair kernel "
         "dispatch counts for multi-row batches\n";
  out << "# TYPE inferflux_cuda_rowpair_selection_total counter\n";
  static constexpr const char *kRowPairPhases[] = {"prefill", "decode"};
  static constexpr const char *kRowPairOps[] = {
      "q8_1_group_row_pair_w4",       "q8_1_group_row_pair",
      "q8_1_gemv_row_pair",           "q8_1_gemv_row_pair_v2",
      "q8_1_gemv_row_pair_hot_fixed",
  };
  {
    std::lock_guard<std::mutex> lock(inferflux_cuda_operator_metrics_mutex_);
    for (const char *phase : kRowPairPhases) {
      const std::vector<const char *> buckets =
          std::string_view(phase) == "prefill"
              ? std::vector<const char *>{"1",     "2",      "3_4",
                                          "5_8",   "9_16",   "17_32",
                                          "33_64", "65_128", "129_plus"}
              : std::vector<const char *>{"1",   "2",    "3_4",
                                          "5_8", "9_16", "17_plus"};
      for (const char *op : kRowPairOps) {
        for (const char *bucket : buckets) {
          const std::string key = std::string(phase) + "|" + op + "|" + bucket;
          const auto it = inferflux_cuda_rowpair_selection_counts_.find(key);
          const uint64_t count =
              it == inferflux_cuda_rowpair_selection_counts_.end() ? 0
                                                                   : it->second;
          out << "inferflux_cuda_rowpair_selection_total{phase=\"" << phase
              << "\",operator=\"" << op << "\",bucket=\"" << bucket << "\"} "
              << count << "\n";
        }
      }
    }
  }

  out << "# HELP inferflux_cuda_forward_duration_ms InferFlux CUDA forward "
         "pass "
         "latency in milliseconds\n";
  out << "# TYPE inferflux_cuda_forward_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_cuda_forward_duration_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << inferflux_cuda_forward_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_cuda_forward_duration_ms_bucket{le=\"+Inf\"} "
      << inferflux_cuda_forward_latency_
             .counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_cuda_forward_duration_ms_sum "
      << inferflux_cuda_forward_latency_.sum_ms.load() << "\n";
  out << "inferflux_cuda_forward_duration_ms_count "
      << inferflux_cuda_forward_latency_.total.load() << "\n";

  out << "# HELP inferflux_cuda_sampling_duration_ms InferFlux CUDA sampling "
         "latency "
         "in milliseconds\n";
  out << "# TYPE inferflux_cuda_sampling_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_cuda_sampling_duration_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << inferflux_cuda_sampling_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_cuda_sampling_duration_ms_bucket{le=\"+Inf\"} "
      << inferflux_cuda_sampling_latency_
             .counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_cuda_sampling_duration_ms_sum "
      << inferflux_cuda_sampling_latency_.sum_ms.load() << "\n";
  out << "inferflux_cuda_sampling_duration_ms_count "
      << inferflux_cuda_sampling_latency_.total.load() << "\n";

  out << "# HELP inferflux_cuda_kv_active_sequences Active KV cache "
         "sequences in the InferFlux CUDA backend\n";
  out << "# TYPE inferflux_cuda_kv_active_sequences gauge\n";
  out << "inferflux_cuda_kv_active_sequences "
      << inferflux_cuda_kv_active_sequences_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_max_sequences Maximum KV cache sequences "
         "in the InferFlux CUDA backend\n";
  out << "# TYPE inferflux_cuda_kv_max_sequences gauge\n";
  out << "inferflux_cuda_kv_max_sequences "
      << inferflux_cuda_kv_max_sequences_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_autotune_events_total Number of times "
         "InferFlux CUDA KV auto-tuning reduced max_seq during pipeline "
         "planning\n";
  out << "# TYPE inferflux_cuda_kv_autotune_events_total counter\n";
  out << "inferflux_cuda_kv_autotune_events_total "
      << inferflux_cuda_kv_autotune_events_total_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_requested_max_seq Requested InferFlux CUDA "
         "KV "
         "max_seq before auto-tune\n";
  out << "# TYPE inferflux_cuda_kv_requested_max_seq gauge\n";
  out << "inferflux_cuda_kv_requested_max_seq "
      << inferflux_cuda_kv_requested_max_seq_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_planned_max_seq Planned InferFlux CUDA KV "
         "max_seq after auto-tune\n";
  out << "# TYPE inferflux_cuda_kv_planned_max_seq gauge\n";
  out << "inferflux_cuda_kv_planned_max_seq "
      << inferflux_cuda_kv_planned_max_seq_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_requested_bytes Requested InferFlux CUDA KV "
         "reservation bytes before auto-tune\n";
  out << "# TYPE inferflux_cuda_kv_requested_bytes gauge\n";
  out << "inferflux_cuda_kv_requested_bytes "
      << inferflux_cuda_kv_requested_bytes_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_planned_bytes Planned InferFlux CUDA KV "
         "reservation bytes after auto-tune\n";
  out << "# TYPE inferflux_cuda_kv_planned_bytes gauge\n";
  out << "inferflux_cuda_kv_planned_bytes "
      << inferflux_cuda_kv_planned_bytes_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_budget_bytes InferFlux CUDA KV budget bytes "
         "used "
         "for auto-tune planning\n";
  out << "# TYPE inferflux_cuda_kv_budget_bytes gauge\n";
  out << "inferflux_cuda_kv_budget_bytes "
      << inferflux_cuda_kv_budget_bytes_.load() << "\n";

  out << "# HELP inferflux_cuda_kv_memory_total_bytes InferFlux CUDA KV bytes "
         "partitioned by slot state accounting\n";
  out << "# TYPE inferflux_cuda_kv_memory_total_bytes gauge\n";
  out << "inferflux_cuda_kv_memory_total_bytes "
      << inferflux_cuda_kv_memory_total_bytes_.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP inferflux_cuda_kv_memory_active_bytes InferFlux CUDA KV bytes "
         "held "
         "by active slots\n";
  out << "# TYPE inferflux_cuda_kv_memory_active_bytes gauge\n";
  out << "inferflux_cuda_kv_memory_active_bytes "
      << inferflux_cuda_kv_memory_active_bytes_.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP inferflux_cuda_kv_memory_prefix_retained_bytes InferFlux CUDA "
         "KV "
         "bytes retained for session/prefix reuse\n";
  out << "# TYPE inferflux_cuda_kv_memory_prefix_retained_bytes gauge\n";
  out << "inferflux_cuda_kv_memory_prefix_retained_bytes "
      << inferflux_cuda_kv_memory_prefix_retained_bytes_.load(
             std::memory_order_relaxed)
      << "\n";

  out << "# HELP inferflux_cuda_kv_memory_free_bytes InferFlux CUDA KV bytes "
         "available for new slots\n";
  out << "# TYPE inferflux_cuda_kv_memory_free_bytes gauge\n";
  out << "inferflux_cuda_kv_memory_free_bytes "
      << inferflux_cuda_kv_memory_free_bytes_.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP inferflux_cuda_kv_memory_sequences Native KV slot counts by "
         "state partition\n";
  out << "# TYPE inferflux_cuda_kv_memory_sequences gauge\n";
  out << "inferflux_cuda_kv_memory_sequences{state=\"active\"} "
      << inferflux_cuda_kv_memory_active_sequences_.load(
             std::memory_order_relaxed)
      << "\n";
  out << "inferflux_cuda_kv_memory_sequences{state=\"prefix_retained\"} "
      << inferflux_cuda_kv_memory_prefix_retained_sequences_.load(
             std::memory_order_relaxed)
      << "\n";
  out << "inferflux_cuda_kv_memory_sequences{state=\"free\"} "
      << inferflux_cuda_kv_memory_free_sequences_.load(
             std::memory_order_relaxed)
      << "\n";

  {
    const auto snapshot = GetInferfluxCudaModelMemorySnapshot();

    out << "# HELP inferflux_cuda_model_memory_reserved_bytes Native model "
           "memory reserved bytes by scope\n";
    out << "# TYPE inferflux_cuda_model_memory_reserved_bytes gauge\n";
    out << "inferflux_cuda_model_memory_reserved_bytes{scope=\"total\",model=\""
        << snapshot.model_label << "\"} " << snapshot.total.reserved_bytes
        << "\n";
    for (const auto &entry : snapshot.domains) {
      out << "inferflux_cuda_model_memory_reserved_bytes{scope=\"domain\","
             "model=\""
          << snapshot.model_label << "\",domain=\"" << entry.first << "\"} "
          << entry.second.reserved_bytes << "\n";
    }

    out << "# HELP inferflux_cuda_model_memory_in_use_bytes Native model "
           "memory in-use bytes by scope\n";
    out << "# TYPE inferflux_cuda_model_memory_in_use_bytes gauge\n";
    out << "inferflux_cuda_model_memory_in_use_bytes{scope=\"total\",model=\""
        << snapshot.model_label << "\"} " << snapshot.total.in_use_bytes
        << "\n";
    for (const auto &entry : snapshot.domains) {
      out << "inferflux_cuda_model_memory_in_use_bytes{scope=\"domain\",model="
             "\""
          << snapshot.model_label << "\",domain=\"" << entry.first << "\"} "
          << entry.second.in_use_bytes << "\n";
    }

    out << "# HELP inferflux_cuda_model_memory_high_water_bytes Native model "
           "memory high-water bytes by scope\n";
    out << "# TYPE inferflux_cuda_model_memory_high_water_bytes gauge\n";
    out << "inferflux_cuda_model_memory_high_water_bytes{scope=\"total\",model="
           "\""
        << snapshot.model_label << "\"} " << snapshot.total.high_water_bytes
        << "\n";
    for (const auto &entry : snapshot.domains) {
      out << "inferflux_cuda_model_memory_high_water_bytes{scope=\"domain\","
             "model=\""
          << snapshot.model_label << "\",domain=\"" << entry.first << "\"} "
          << entry.second.high_water_bytes << "\n";
    }

    out << "# HELP inferflux_cuda_model_memory_evictable_bytes Native model "
           "memory evictable bytes by scope\n";
    out << "# TYPE inferflux_cuda_model_memory_evictable_bytes gauge\n";
    out << "inferflux_cuda_model_memory_evictable_bytes{scope=\"total\",model="
           "\""
        << snapshot.model_label << "\"} " << snapshot.total.evictable_bytes
        << "\n";
    for (const auto &entry : snapshot.domains) {
      out << "inferflux_cuda_model_memory_evictable_bytes{scope=\"domain\","
             "model=\""
          << snapshot.model_label << "\",domain=\"" << entry.first << "\"} "
          << entry.second.evictable_bytes << "\n";
    }
  }

  out << "# HELP inferflux_cuda_lane_submissions_total Unified-batch lane "
         "submissions in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_submissions_total counter\n";
  out << "inferflux_cuda_lane_submissions_total{lane=\"decode\"} "
      << cuda_decode_lane_submissions_.load() << "\n";
  out << "inferflux_cuda_lane_submissions_total{lane=\"prefill\"} "
      << cuda_prefill_lane_submissions_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_completions_total Unified-batch lane "
         "completions in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_completions_total counter\n";
  out << "inferflux_cuda_lane_completions_total{lane=\"decode\"} "
      << cuda_decode_lane_completions_.load() << "\n";
  out << "inferflux_cuda_lane_completions_total{lane=\"prefill\"} "
      << cuda_prefill_lane_completions_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_queue_depth Pending lane queue depth in "
         "CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_queue_depth gauge\n";
  out << "inferflux_cuda_lane_queue_depth{lane=\"decode\"} "
      << cuda_decode_lane_queue_depth_.load() << "\n";
  out << "inferflux_cuda_lane_queue_depth{lane=\"prefill\"} "
      << cuda_prefill_lane_queue_depth_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_enqueue_rejects_total Unified-batch lane "
         "enqueue rejects in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_enqueue_rejects_total counter\n";
  out << "inferflux_cuda_lane_enqueue_rejects_total{lane=\"decode\"} "
      << cuda_decode_lane_enqueue_rejects_.load() << "\n";
  out << "inferflux_cuda_lane_enqueue_rejects_total{lane=\"prefill\"} "
      << cuda_prefill_lane_enqueue_rejects_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_collect_timeouts_total Unified-batch lane "
         "collection timeouts in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_collect_timeouts_total counter\n";
  out << "inferflux_cuda_lane_collect_timeouts_total{lane=\"decode\"} "
      << cuda_decode_lane_collect_timeouts_.load() << "\n";
  out << "inferflux_cuda_lane_collect_timeouts_total{lane=\"prefill\"} "
      << cuda_prefill_lane_collect_timeouts_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_worker_restarts_total CUDA lane worker "
         "dispatcher restarts\n";
  out << "# TYPE inferflux_cuda_lane_worker_restarts_total counter\n";
  out << "inferflux_cuda_lane_worker_restarts_total "
      << cuda_lane_worker_restarts_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_overlap_events_total CUDA lane execution "
         "windows where decode/prefill overlapped\n";
  out << "# TYPE inferflux_cuda_lane_overlap_events_total counter\n";
  out << "inferflux_cuda_lane_overlap_events_total "
      << cuda_lane_overlap_events_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_overlap_duration_ms_total Total "
         "wall-clock "
         "duration where decode/prefill lanes overlapped\n";
  out << "# TYPE inferflux_cuda_lane_overlap_duration_ms_total counter\n";
  const double overlap_duration_ms =
      static_cast<double>(
          cuda_lane_overlap_duration_us_.load(std::memory_order_relaxed)) /
      1000.0;
  out << "inferflux_cuda_lane_overlap_duration_ms_total " << overlap_duration_ms
      << "\n";

  int overlap_active = 0;
  {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    overlap_active = cuda_overlap_active_ ? 1 : 0;
  }
  out << "# HELP inferflux_cuda_lane_overlap_active Whether decode/prefill "
         "lanes are currently overlapping (1=true)\n";
  out << "# TYPE inferflux_cuda_lane_overlap_active gauge\n";
  out << "inferflux_cuda_lane_overlap_active " << overlap_active << "\n";

  out << "# HELP inferflux_cuda_lane_inflight Active CUDA lane workers "
         "currently executing a batch\n";
  out << "# TYPE inferflux_cuda_lane_inflight gauge\n";
  out << "inferflux_cuda_lane_inflight{lane=\"decode\"} "
      << cuda_decode_lane_inflight_.load() << "\n";
  out << "inferflux_cuda_lane_inflight{lane=\"prefill\"} "
      << cuda_prefill_lane_inflight_.load() << "\n";

  out << "# HELP inferflux_cuda_attention_kernel_selected CUDA attention "
         "kernel currently selected (one-hot gauge)\n";
  out << "# TYPE inferflux_cuda_attention_kernel_selected gauge\n";
  std::string selected_kernel = "standard";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_kernel_mutex_);
    selected_kernel = cuda_attention_kernel_;
  }
  for (const char *kernel : {"fa3", "fa2", "standard"}) {
    out << "inferflux_cuda_attention_kernel_selected{kernel=\"" << kernel
        << "\"} " << (selected_kernel == kernel ? 1 : 0) << "\n";
  }

  out << "# HELP inferflux_cuda_attention_kernel_fallbacks_total CUDA "
         "attention kernel fallback decisions by requested/selected kernel\n";
  out << "# TYPE inferflux_cuda_attention_kernel_fallbacks_total counter\n";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_fallback_mutex_);
    for (const auto &[key, count] : cuda_attention_fallback_counts_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) {
        continue;
      }
      const std::string requested = key.substr(0, p1);
      const std::string selected = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string reason = key.substr(p2 + 1);
      out << "inferflux_cuda_attention_kernel_fallbacks_total{requested=\""
          << requested << "\",selected=\"" << selected << "\",reason=\""
          << reason << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_cuda_dispatch_divergence_total Selected vs "
         "executed native dispatch operator divergences (layer|selected|"
         "actual|reason)\n";
  out << "# TYPE inferflux_cuda_dispatch_divergence_total counter\n";
  {
    std::lock_guard<std::mutex> lock(cuda_dispatch_divergence_mutex_);
    std::map<std::string, uint64_t> sorted(
        cuda_dispatch_divergence_counts_.begin(),
        cuda_dispatch_divergence_counts_.end());
    for (const auto &kv : sorted) {
      const std::string::size_type b1 = kv.first.find('|');
      const std::string::size_type b2 = kv.first.find('|', b1 + 1);
      const std::string::size_type b3 = kv.first.find('|', b2 + 1);
      if (b1 == std::string::npos || b2 == std::string::npos ||
          b3 == std::string::npos) {
        continue;
      }
      out << "inferflux_cuda_dispatch_divergence_total{layer=\""
          << kv.first.substr(0, b1) << "\",selected=\""
          << kv.first.substr(b1 + 1, b2 - b1 - 1) << "\",actual=\""
          << kv.first.substr(b2 + 1, b3 - b2 - 1) << "\",reason=\""
          << kv.first.substr(b3 + 1) << "\"} " << kv.second << "\n";
    }
  }

  out << "# HELP inferflux_cuda_attention_kernel_switches_total CUDA "
         "attention kernel selection switches across model reloads\n";
  out << "# TYPE inferflux_cuda_attention_kernel_switches_total counter\n";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_switch_mutex_);
    for (const auto &[key, count] : cuda_attention_switch_counts_) {
      auto split = key.find('|');
      if (split == std::string::npos) {
        continue;
      }
      const std::string from = key.substr(0, split);
      const std::string to = key.substr(split + 1);
      out << "inferflux_cuda_attention_kernel_switches_total{from_kernel=\""
          << from << "\",to_kernel=\"" << to << "\"} " << count << "\n";
    }
  }

  return out.str();
}

MetricsRegistry &GlobalMetrics() { return g_metrics; }

} // namespace inferflux
