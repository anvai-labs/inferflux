#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace inferflux {

// Latency histogram with fixed buckets (in milliseconds).
// Prometheus-compatible: cumulative counts per bucket + _sum + _count.
struct LatencyHistogram {
  // Upper bounds in milliseconds: 10, 50, 100, 250, 500, 1000, 2500, 5000, +Inf
  static constexpr std::array<double, 8> kBuckets{
      10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0};
  // Cumulative bucket counts (bucket[i] = requests finishing within kBuckets[i]
  // ms).
  std::array<std::atomic<uint64_t>, 9> counts{}; // 8 finite + 1 +Inf
  std::atomic<uint64_t> sum_ms{0};
  std::atomic<uint64_t> total{0};

  void Record(double ms);
};

class MetricsRegistry {
public:
  void SetBackend(const std::string &backend);

  // Counters (existing).
  void RecordSuccess(int prompt_tokens, int completion_tokens);
  void RecordError();
  void RecordEmptyGeneration();
  void RecordSpeculative(std::size_t total_chunks, std::size_t accepted_chunks,
                         std::size_t reused_tokens);
  void RecordBatch(std::size_t request_count, std::size_t token_count);
  // Scheduler iteration accounting (prefill-only / decode-only / mixed).
  void RecordSchedulerIteration(std::size_t prefill_requests,
                                std::size_t decode_requests,
                                std::size_t token_count);
  void RecordSchedulerPolicyIteration(const std::string &policy,
                                      std::size_t prefill_requests,
                                      std::size_t decode_requests);
  void RecordBatchTokenBudgetSkip();
  void RecordPrefixAffinityProbe(bool hit, int matched_tokens);
  void RecordPrefixLookup(bool hit);
  void RecordPrefixMatchedTokens(int tokens);
  void RecordPartialPrefixHit();
  // KV prefix reuse (§ Item 5): called when CopySequencePrefix+PrefillPartial
  // replaces a full Prefill.  tokens_saved = number of prefix tokens skipped.
  void RecordKVPrefixReuse(int tokens_saved);
  void RecordStreamTokens(std::size_t tokens);
  void RecordStreamCacheHit();
  void RecordFairnessTokens(int priority_level, std::size_t tokens);
  void RecordFairnessPreemption(int priority_level);
  void RecordFairnessYield(int priority_level, std::size_t emitted_tokens,
                           std::size_t remaining_tokens);
  void RecordFairnessResume(int priority_level);
  void RecordQueueLatency(double wait_ms);
  void RecordBatchExecution(double exec_ms);
  void RecordModelLoad(const std::string &model_id, const std::string &backend,
                       double load_seconds);
  void RecordModelReady(const std::string &model_id, const std::string &backend,
                        bool ready);
  void RecordModelRoute(const std::string &model_id, const std::string &backend,
                        bool hit);
  // Per-model token counters for throughput dashboards.
  // Exported as counters so Prometheus can derive tokens/sec via rate().
  void RecordModelTokens(const std::string &model_id,
                         const std::string &backend, int prompt_tokens,
                         int completion_tokens);
  void RecordCapabilityRejection(const std::string &backend,
                                 const std::string &feature);
  void RecordBackendExposure(const std::string &requested_backend,
                             const std::string &exposed_backend,
                             const std::string &provider, bool used_fallback);
  void RecordCapabilityRouteFallback(const std::string &from_backend,
                                     const std::string &to_backend,
                                     const std::string &feature);

  // Multimodal (§2.2): record image preprocessing events.
  // images: number of images decoded; decode_ms: preprocessing wall-clock time.
  void RecordImagePreprocess(int images, double decode_ms);

  // MoE (§2.6): record a request dispatched to a MoE model.
  void RecordMoERequest();

  // Flash Attention (§2.7): set the FA enabled gauge (0=disabled, 1=enabled).
  // Called once at server startup after LlamaBackendConfig is applied.
  void SetFlashAttentionEnabled(bool enabled);

  // Flash Attention execution metrics - record per-request FA2 usage.
  // kernel: one of "fa2", "fa3", "standard"
  // duration_ms: execution time in milliseconds
  // prompt_tokens: number of tokens processed
  void RecordFlashAttentionExecution(const std::string &kernel,
                                     double duration_ms, int prompt_tokens);

  // Flash Attention memory usage - update VRAM consumption gauge.
  // memory_mb: memory used in megabytes
  void SetFlashAttentionMemoryMB(double memory_mb);

  // Flash Attention request counter - track total requests using FA.
  // kernel: one of "fa2", "fa3", "standard"
  void RecordFlashAttentionRequest(const std::string &kernel);

  // ROCm backend metrics - track AMD GPU usage
  void RecordRocmKernelSelection(const std::string &kernel);
  void RecordRocmFlashAttentionExecution(double duration_ms, int tokens);
  void SetRocmMemoryUsageMB(double memory_mb);
  void RecordRocmDeviceProperties(int device_id, const std::string &arch);

  // KV transfer latency (§2.5 item 12): elapsed time from
  // KVPacket::enqueue_time to when a decode worker dequeues it from KVChannel /
  // ShmKVTransport.
  void RecordKVTransfer(double transfer_ms);
  void RecordDisaggKVEnqueueRejected(bool retries_exhausted);
  void RecordDisaggKVTicketStage(std::string_view stage);
  uint64_t GetDisaggKVEnqueueRejections() const {
    return disagg_kv_enqueue_rejections_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVEnqueueExhausted() const {
    return disagg_kv_enqueue_exhausted_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTicketsEnqueued() const {
    return disagg_kv_tickets_enqueued_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTicketsAcknowledged() const {
    return disagg_kv_tickets_acknowledged_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTicketsCommitted() const {
    return disagg_kv_tickets_committed_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTicketsTimedOut() const {
    return disagg_kv_tickets_timed_out_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTimeoutDebt() const {
    return disagg_kv_timeout_debt_.load(std::memory_order_relaxed);
  }
  uint64_t GetDisaggKVTimeoutStreak() const {
    return disagg_kv_timeout_streak_.load(std::memory_order_relaxed);
  }

  // GGML-native perf counters (from llama_perf_context). Exposes ground-truth
  // kernel timings that subprocess wrappers cannot surface.
  void RecordLlamaPerf(double prefill_ms, double decode_ms,
                       int32_t prompt_tokens, int32_t generated_tokens);

  // Latency recording — call with full request duration in milliseconds.
  void RecordLatency(double request_ms);

  // Per-phase latency recording (OBS-2).
  // prefill_ms: time from tokenization start to first token produced.
  // decode_ms:  time spent generating completion tokens.
  void RecordPrefillDuration(double prefill_ms);
  void RecordDecodeDuration(double decode_ms);

  // Gauge helpers — active connections and scheduler queue depth.
  void IncrementConnections();
  void DecrementConnections();
  void SetQueueDepth(int depth);
  void SetPrefillQueueDepth(int depth);
  void SetDecodeQueueDepth(int depth);
  void SetSchedulerBatchLimits(int max_batch_size, int max_batch_tokens);
  void SetSchedulerDeferredSequenceRetirements(int depth);
  void RecordSchedulerDeferredSequenceRetirement(double lag_ms);
  void RecordCudaLaneSubmission(bool decode_lane);
  void RecordCudaLaneCompletion(bool decode_lane);
  void RecordCudaLaneExecutionStart(bool decode_lane);
  void RecordCudaLaneExecutionStop(bool decode_lane);
  void RecordCudaLaneOverlap(double duration_ms);
  void SetCudaLaneQueueDepth(bool decode_lane, int depth);
  void RecordCudaLaneEnqueueReject(bool decode_lane);
  void RecordCudaLaneCollectTimeout(bool decode_lane);
  void RecordCudaLaneWorkerRestart();
  void RecordDecodeStepLoops(const std::string &mode, std::size_t loops);
  void RecordPrefillChunkTruncation(const std::string &mode,
                                    std::size_t truncated_tokens);
  void SetCudaAttentionKernel(const std::string &kernel);
  void RecordCudaAttentionKernelFallback(const std::string &requested_kernel,
                                         const std::string &selected_kernel,
                                         const std::string &reason);

  // Dispatch divergence: the executor ran a different tier than the
  // selector chose (layer=ffn|down_proj|graph|probe). The canonical signal
  // for routing regressions like the kQ81GroupMmq3 allowlist bug.
  void RecordInferfluxCudaDispatchDivergence(std::string_view layer,
                                             std::string_view selected,
                                             std::string_view actual,
                                             std::string_view reason);
  uint64_t GetInferfluxCudaDispatchDivergences() const;
  void RecordCudaAttentionKernelSwitch(const std::string &from_kernel,
                                       const std::string &to_kernel);
  void RecordDecodeWorkerBatchSize(std::size_t batch_size);
  void RecordDecodeWorkerExecutionPath(std::string_view path);
  void RecordDecodeWorkerStickyMerge(std::size_t merged_requests);
  void RecordDecodeAssemblySnapshot(std::string_view mode, std::size_t ready,
                                    std::size_t selected,
                                    std::size_t compatible,
                                    std::size_t incompatible);

  // InferFlux CUDA backend metrics
  void RecordInferfluxCudaForwardShape(bool is_decode, int batch_size);
  void RecordInferfluxCudaForwardLatency(double forward_ms);
  void RecordInferfluxCudaForwardPass(bool is_decode, int batch_size,
                                      double forward_ms);
  void RecordInferfluxCudaSampling(int batch_size, double sampling_ms);
  void RecordInferfluxCudaBatchDecode(int batch_size, double total_ms);
  void RecordInferfluxCudaForwardBatchSize(std::string_view phase,
                                           int batch_size);
  void RecordInferfluxCudaFfnProjOperator(std::string_view phase,
                                          std::string_view op);
  void RecordInferfluxCudaFfnProjGeometry(std::string_view phase,
                                          std::string_view op,
                                          std::string_view quant,
                                          int batch_size, int n, int k,
                                          int grouped_outputs);
  void RecordInferfluxCudaDownProjOperator(std::string_view phase,
                                           std::string_view op);
  void RecordInferfluxCudaDownProjGeometry(std::string_view phase,
                                           std::string_view op,
                                           std::string_view quant,
                                           int batch_size, int n, int k);
  void RecordInferfluxCudaRowPairSelection(std::string_view phase,
                                           std::string_view op, int batch_rows);
  void RecordInferfluxCudaKvAutoTunePlan(int requested_max_seq,
                                         int planned_max_seq,
                                         std::size_t requested_bytes,
                                         std::size_t planned_bytes,
                                         std::size_t budget_bytes);
  void SetInferfluxCudaKvCacheOccupancy(int active_sequences,
                                        int max_sequences);
  int GetQueueDepth() const {
    return queue_depth_.load(std::memory_order_relaxed);
  }
  int GetPrefillQueueDepth() const {
    return prefill_queue_depth_.load(std::memory_order_relaxed);
  }
  int GetDecodeQueueDepth() const {
    return decode_queue_depth_.load(std::memory_order_relaxed);
  }
  int GetSchedulerBatchLimitSize() const {
    return scheduler_batch_limit_size_.load(std::memory_order_relaxed);
  }
  int GetSchedulerBatchLimitTokens() const {
    return scheduler_batch_limit_tokens_.load(std::memory_order_relaxed);
  }
  int GetSchedulerDeferredSequenceRetirements() const {
    return scheduler_deferred_sequence_retirements_.load(
        std::memory_order_relaxed);
  }

  // Snapshot of prefix-cache metrics for the /v1/admin/cache endpoint.
  struct CacheMetrics {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t partial_hits{0};
    uint64_t matched_tokens{0};
    uint64_t kv_reuse_count{0};
    uint64_t kv_reuse_tokens{0};
  };
  CacheMetrics GetCacheMetrics() const;

  struct MemoryUsageMetrics {
    uint64_t reserved_bytes{0};
    uint64_t in_use_bytes{0};
    uint64_t high_water_bytes{0};
    uint64_t evictable_bytes{0};
  };

  struct InferfluxCudaModelMemorySnapshot {
    std::string model_label;
    MemoryUsageMetrics total;
    std::unordered_map<std::string, MemoryUsageMetrics> domains;
  };

  struct InferfluxCudaKvMemorySnapshot {
    uint64_t total_bytes{0};
    uint64_t active_bytes{0};
    uint64_t prefix_retained_bytes{0};
    uint64_t free_bytes{0};
    int active_sequences{0};
    int prefix_retained_sequences{0};
    int free_sequences{0};
    int max_sequences{0};
  };

  void SetInferfluxCudaModelMemorySnapshot(
      std::string model_label, MemoryUsageMetrics total,
      std::unordered_map<std::string, MemoryUsageMetrics> domains);
  InferfluxCudaModelMemorySnapshot GetInferfluxCudaModelMemorySnapshot() const;

  void SetInferfluxCudaKvMemoryBytes(uint64_t total_bytes,
                                     uint64_t active_bytes,
                                     uint64_t prefix_retained_bytes,
                                     uint64_t free_bytes, int active_sequences,
                                     int prefix_retained_sequences,
                                     int free_sequences, int max_sequences);
  InferfluxCudaKvMemorySnapshot GetInferfluxCudaKvMemorySnapshot() const;
  int GetInferfluxCudaKvMaxSequences() const {
    return inferflux_cuda_kv_max_sequences_.load(std::memory_order_relaxed);
  }
  uint64_t GetInferfluxCudaKvPlannedBytes() const {
    return inferflux_cuda_kv_planned_bytes_.load(std::memory_order_relaxed);
  }

  std::string RenderPrometheus() const;

private:
  mutable std::mutex backend_mutex_;
  std::string backend_{"cpu"};

  // Counters.
  std::atomic<uint64_t> total_requests_{0};
  std::atomic<uint64_t> total_errors_{0};
  std::atomic<uint64_t> total_empty_generations_{0};
  std::atomic<uint64_t> total_prompt_tokens_{0};
  std::atomic<uint64_t> total_completion_tokens_{0};
  std::atomic<uint64_t> speculative_chunks_total_{0};
  std::atomic<uint64_t> speculative_chunks_accepted_{0};
  std::atomic<uint64_t> speculative_tokens_reused_{0};
  std::atomic<uint64_t> total_batches_{0};
  std::atomic<uint64_t> total_batch_tokens_{0};
  std::atomic<uint64_t> max_batch_size_{0};
  std::atomic<uint64_t> scheduler_iterations_prefill_{0};
  std::atomic<uint64_t> scheduler_iterations_decode_{0};
  std::atomic<uint64_t> scheduler_iterations_mixed_{0};
  std::atomic<uint64_t> scheduler_iteration_requests_total_{0};
  std::atomic<uint64_t> scheduler_iteration_tokens_total_{0};
  std::atomic<uint64_t> scheduler_batch_token_budget_skips_{0};
  std::atomic<uint64_t> scheduler_prefix_affinity_probes_{0};
  std::atomic<uint64_t> scheduler_prefix_affinity_hits_{0};
  std::atomic<uint64_t> scheduler_prefix_affinity_matched_tokens_{0};
  std::atomic<uint64_t> prefix_hits_{0};
  std::atomic<uint64_t> prefix_misses_{0};
  std::atomic<uint64_t> prefix_matched_tokens_{0};
  std::atomic<uint64_t> prefix_partial_hits_{0};
  std::atomic<uint64_t> kv_prefix_reuse_count_{0};
  std::atomic<uint64_t> kv_prefix_reuse_tokens_{0};
  std::atomic<uint64_t> stream_tokens_{0};
  std::atomic<uint64_t> stream_cache_hits_{0};
  std::atomic<uint64_t> fairness_preemptions_{0};
  std::atomic<uint64_t> fairness_yields_{0};
  std::atomic<uint64_t> fairness_resumes_{0};
  mutable std::mutex fairness_metrics_mutex_;
  std::unordered_map<int, uint64_t> fairness_tokens_;
  std::atomic<uint64_t> model_route_misses_{0};
  std::atomic<uint64_t> multimodal_images_{
      0}; // §2.2: total images preprocessed.
  std::atomic<uint64_t> multimodal_requests_{
      0}; // §2.2: total requests with images.
  std::atomic<uint64_t> moe_requests_{
      0}; // §2.6: requests routed to MoE models.
  std::atomic<uint64_t> flash_attention_enabled_{
      0}; // §2.7: gauge — 0=disabled, 1=enabled.

  // Flash Attention execution metrics (§2.7).
  std::atomic<uint64_t> flash_attention_requests_fa2_{0};
  std::atomic<uint64_t> flash_attention_requests_fa3_{0};
  std::atomic<uint64_t> flash_attention_requests_standard_{0};
  mutable std::mutex flash_attention_exec_mutex_;
  LatencyHistogram flash_attention_exec_latency_;
  std::atomic<double> flash_attention_memory_mb_{0.0};

  // ROCm backend metrics (AMD GPU support)
  std::atomic<uint64_t> rocm_requests_total_{0};
  std::atomic<uint64_t> rocm_flash_attention_requests_{0};
  std::atomic<double> rocm_memory_mb_{0.0};
  std::string rocm_device_arch_{"unknown"};

  // Latency histograms.
  LatencyHistogram request_latency_;
  LatencyHistogram queue_latency_;
  LatencyHistogram batch_exec_latency_;
  LatencyHistogram scheduler_sequence_retirement_latency_;
  LatencyHistogram prefill_latency_; // OBS-2: prefill phase
  LatencyHistogram decode_latency_;  // OBS-2: decode phase
  LatencyHistogram
      kv_transfer_latency_; // §2.5 item 12: prefill→decode KV hand-off
  std::atomic<uint64_t> disagg_kv_enqueue_rejections_{0};
  std::atomic<uint64_t> disagg_kv_enqueue_exhausted_{0};
  std::atomic<uint64_t> disagg_kv_tickets_enqueued_{0};
  std::atomic<uint64_t> disagg_kv_tickets_acknowledged_{0};
  std::atomic<uint64_t> disagg_kv_tickets_committed_{0};
  std::atomic<uint64_t> disagg_kv_tickets_timed_out_{0};
  std::atomic<uint64_t> disagg_kv_timeout_debt_{0};
  std::atomic<uint64_t> disagg_kv_timeout_streak_{0};
  LatencyHistogram llama_prefill_latency_; // GGML-native prompt eval latency
  LatencyHistogram llama_decode_latency_;  // GGML-native token gen latency
  std::atomic<uint64_t> llama_prompt_tokens_{0};
  std::atomic<uint64_t> llama_gen_tokens_{0};

  // Gauges.
  std::atomic<int> active_connections_{0};
  std::atomic<int> queue_depth_{0};
  std::atomic<int> prefill_queue_depth_{0};
  std::atomic<int> decode_queue_depth_{0};
  std::atomic<int> scheduler_batch_limit_size_{0};
  std::atomic<int> scheduler_batch_limit_tokens_{0};
  std::atomic<int> scheduler_deferred_sequence_retirements_{0};
  std::atomic<uint64_t> scheduler_deferred_sequence_retirements_completed_{0};
  std::atomic<uint64_t> cuda_decode_lane_submissions_{0};
  std::atomic<uint64_t> cuda_prefill_lane_submissions_{0};
  std::atomic<uint64_t> cuda_decode_lane_completions_{0};
  std::atomic<uint64_t> cuda_prefill_lane_completions_{0};
  mutable std::mutex scheduler_policy_metrics_mutex_;
  std::unordered_map<std::string, uint64_t> scheduler_policy_iterations_;
  mutable std::mutex scheduler_step_metrics_mutex_;
  std::unordered_map<std::string, uint64_t> scheduler_decode_step_loops_;
  std::unordered_map<std::string, uint64_t>
      scheduler_prefill_chunk_truncations_;
  std::unordered_map<std::string, uint64_t>
      scheduler_prefill_chunk_truncated_tokens_;
  mutable std::mutex scheduler_decode_worker_batch_size_mutex_;
  std::unordered_map<std::string, uint64_t>
      scheduler_decode_worker_batch_size_counts_;
  std::unordered_map<std::string, uint64_t>
      scheduler_decode_worker_execution_path_counts_;
  std::unordered_map<std::string, uint64_t>
      scheduler_decode_worker_sticky_merge_counts_;
  std::unordered_map<std::string, uint64_t> scheduler_decode_assembly_ready_;
  std::unordered_map<std::string, uint64_t> scheduler_decode_assembly_selected_;
  std::unordered_map<std::string, uint64_t>
      scheduler_decode_assembly_compatible_;
  std::unordered_map<std::string, uint64_t>
      scheduler_decode_assembly_incompatible_;
  std::atomic<uint64_t> scheduler_decode_worker_sticky_merged_requests_total_{
      0};
  std::atomic<uint64_t> cuda_lane_overlap_events_{0};
  std::atomic<uint64_t> cuda_lane_overlap_duration_us_{0};
  std::atomic<int> cuda_decode_lane_inflight_{0};
  std::atomic<int> cuda_prefill_lane_inflight_{0};
  std::atomic<int> cuda_decode_lane_queue_depth_{0};
  std::atomic<int> cuda_prefill_lane_queue_depth_{0};
  std::atomic<uint64_t> cuda_decode_lane_enqueue_rejects_{0};
  std::atomic<uint64_t> cuda_prefill_lane_enqueue_rejects_{0};
  std::atomic<uint64_t> cuda_decode_lane_collect_timeouts_{0};
  std::atomic<uint64_t> cuda_prefill_lane_collect_timeouts_{0};
  std::atomic<uint64_t> cuda_lane_worker_restarts_{0};
  mutable std::mutex cuda_attention_kernel_mutex_;
  std::string cuda_attention_kernel_{"standard"};
  mutable std::mutex cuda_overlap_timing_mutex_;
  bool cuda_overlap_active_{false};
  std::chrono::steady_clock::time_point cuda_overlap_started_at_{};
  mutable std::mutex cuda_attention_fallback_mutex_;
  std::unordered_map<std::string, uint64_t> cuda_attention_fallback_counts_;
  mutable std::mutex cuda_dispatch_divergence_mutex_;
  std::unordered_map<std::string, uint64_t> cuda_dispatch_divergence_counts_;
  uint64_t cuda_dispatch_divergence_total_{0};
  mutable std::mutex cuda_attention_switch_mutex_;
  std::unordered_map<std::string, uint64_t> cuda_attention_switch_counts_;

  // InferFlux CUDA backend metrics
  std::atomic<uint64_t> inferflux_cuda_forward_prefill_total_{0};
  std::atomic<uint64_t> inferflux_cuda_forward_decode_total_{0};
  std::atomic<uint64_t> inferflux_cuda_forward_batch_tokens_total_{0};
  LatencyHistogram inferflux_cuda_forward_latency_;
  LatencyHistogram inferflux_cuda_sampling_latency_;
  mutable std::mutex inferflux_cuda_forward_batch_size_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_forward_batch_size_counts_;
  mutable std::mutex inferflux_cuda_ffn_proj_operator_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_ffn_proj_operator_counts_;
  mutable std::mutex inferflux_cuda_ffn_proj_geometry_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_ffn_proj_geometry_counts_;
  mutable std::mutex inferflux_cuda_down_proj_operator_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_down_proj_operator_counts_;
  mutable std::mutex inferflux_cuda_down_proj_geometry_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_down_proj_geometry_counts_;
  mutable std::mutex inferflux_cuda_rowpair_selection_mutex_;
  std::unordered_map<std::string, uint64_t>
      inferflux_cuda_rowpair_selection_counts_;
  std::atomic<int> inferflux_cuda_kv_active_sequences_{0};
  std::atomic<int> inferflux_cuda_kv_max_sequences_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_autotune_events_total_{0};
  std::atomic<int> inferflux_cuda_kv_requested_max_seq_{0};
  std::atomic<int> inferflux_cuda_kv_planned_max_seq_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_requested_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_planned_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_budget_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_memory_total_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_memory_active_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_memory_prefix_retained_bytes_{0};
  std::atomic<uint64_t> inferflux_cuda_kv_memory_free_bytes_{0};
  std::atomic<int> inferflux_cuda_kv_memory_active_sequences_{0};
  std::atomic<int> inferflux_cuda_kv_memory_prefix_retained_sequences_{0};
  std::atomic<int> inferflux_cuda_kv_memory_free_sequences_{0};
  mutable std::mutex inferflux_cuda_model_memory_mutex_;
  InferfluxCudaModelMemorySnapshot inferflux_cuda_model_memory_snapshot_{};

  struct ModelStats {
    std::string backend;
    double load_seconds{0.0};
    uint64_t load_events{0};
    uint64_t routes{0};
    uint64_t prompt_tokens{0};
    uint64_t completion_tokens{0};
    bool ready{false};
  };

  mutable std::mutex model_metrics_mutex_;
  std::unordered_map<std::string, ModelStats> model_stats_;
  mutable std::mutex capability_metrics_mutex_;
  std::unordered_map<std::string, uint64_t> capability_rejections_;
  mutable std::mutex backend_exposure_mutex_;
  std::unordered_map<std::string, uint64_t> backend_exposure_counts_;
  mutable std::mutex capability_route_fallback_mutex_;
  std::unordered_map<std::string, uint64_t> capability_route_fallbacks_;
};

MetricsRegistry &GlobalMetrics();

} // namespace inferflux
