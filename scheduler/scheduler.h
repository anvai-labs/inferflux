#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/disaggregated/kv_channel.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/logprob.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/scheduler/sequence_slot_manager.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/model_router.h"
#include "scheduler/model_selection.h"
#include "scheduler/request_batch.h"
#include "scheduler/session_handle_manager.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace inferflux {

class BatchExecutor;

enum class SchedulerBatchPolicy {
  kPriorityAge,
  kLpmPriority,
  kThroughputBalanced,
};

std::string SchedulerBatchPolicyToString(SchedulerBatchPolicy policy);
bool IsSchedulerBatchPolicyValue(const std::string &value);
SchedulerBatchPolicy ParseSchedulerBatchPolicy(
    const std::string &value,
    SchedulerBatchPolicy default_policy = SchedulerBatchPolicy::kPriorityAge);

// Disaggregated prefill/decode configuration (§2.5).
struct DisaggregatedConfig {
  int prefill_pool_size{0}; // 0 = unified
  int decode_pool_size{0};  // 0 = unified
  std::shared_ptr<disaggregated::IKVTransport> kv_transport;
  // Maximum number of kv_transport enqueue rejections tolerated before failing
  // the request with a deterministic distributed-overload error.
  int kv_enqueue_max_retries{3};
};

// Scheduler: handles request queuing, batch selection, and fairness.
class Scheduler {
public:
  struct Config {
    struct SessionHandleConfig {
      bool enabled{false};
      int ttl_ms{300000};
      int max_sessions{1024};
    };

    // Default batch limits optimized for concurrent throughput.
    // max_batch_size=32 allows GPU to process more concurrent requests without
    // queueing. See docs/concurrent_throughput_investigation.md for rationale.
    Config()
        : max_batch_size(32), max_batch_tokens(16384), min_batch_size(1),
          batch_accumulation_ms(2) {}
    int max_batch_size;
    int max_batch_tokens;
    int min_batch_size; // Preferred batch target during accumulation windows
    int batch_accumulation_ms; // Max wait time after first work item arrives
    SchedulerBatchPolicy batch_policy{SchedulerBatchPolicy::kPriorityAge};
    int decode_burst_tokens{0};
    int chunked_prefill_tokens{512};
    double mixed_prefill_budget_ratio{1.0};
    SessionHandleConfig session_handles{};
  };

  explicit Scheduler(
      SimpleTokenizer &tokenizer, std::shared_ptr<CPUDeviceContext> device,
      std::shared_ptr<PagedKVCache> cache, std::shared_ptr<ModelRouter> router,
      std::shared_ptr<SpeculativeDecoder> speculative_decoder = nullptr,
      std::shared_ptr<RadixPrefixCache> prefix_cache = nullptr,
      const FairnessConfig &fairness_config = {},
      const DisaggregatedConfig &disagg_config = {},
      const ModelSelectionOptions &model_selection_options =
          ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                                /*require_ready_backend=*/true},
      Config config = Config());
  ~Scheduler();

  // Non-blocking admission: adds request to queue and returns a future result.
  std::future<InferenceResult> Generate(InferenceRequest request);

  // Update fairness policy at runtime.
  void UpdateFairnessConfig(const FairnessConfig &config);
  void UpdateModelSelectionOptions(const ModelSelectionOptions &options);
  ModelSelectionOptions ModelSelectionOptionsSnapshot() const;

  // Status accessors.
  int QueueDepth() const;
  int LiveDecodeWorkers() const;
  int ConfiguredDecodeWorkers() const {
    return disagg_config_.decode_pool_size;
  }
  bool HasKVTransport() const { return disagg_config_.kv_transport != nullptr; }
  SchedulerBatchPolicy BatchPolicy() const { return config_.batch_policy; }
  int DecodeBurstTokens() const { return config_.decode_burst_tokens; }
  int ChunkedPrefillTokens() const { return config_.chunked_prefill_tokens; }
  double MixedPrefillBudgetRatio() const {
    return config_.mixed_prefill_budget_ratio;
  }
  ModelRouter *Router() const { return router_.get(); }
  RadixPrefixCache *PrefixCache() const { return prefix_cache_.get(); }
  PagedKVCache *Cache() const { return cache_.get(); }

  // Sequence slot allocator for §2.5 phased prefill/decode.
  // Slots are borrowed during Prefill() and returned after full request
  // completion. AllocSeqSlot() is called from worker/decode loops.
  // FreeSeqSlot() may also be called from DecodeWorkerLoop or RadixPrefixCache
  // eviction — callers must hold queue_mutex_ when calling from a
  // non-worker-loop context.
  int AllocSeqSlot(int64_t request_id = -1, uint64_t *generation_out = nullptr);
  void FreeSeqSlot(int slot, uint64_t generation = 0,
                   std::shared_ptr<LlamaCppBackend> backend = nullptr);

  // Sequence slot manager for universal KV cache slot tracking.
  scheduler::SequenceSlotManager *SlotManager() { return slot_manager_.get(); }
  const scheduler::SequenceSlotManager *SlotManager() const {
    return slot_manager_.get();
  }

private:
  struct PendingRequest {
    InferenceRequest inference;
    std::promise<InferenceResult> promise;
    std::chrono::steady_clock::time_point enqueue_time;
    int priority{0};
    int priority_level{0};
    uint64_t sequence{0};
    std::shared_ptr<LlamaCppBackend> resolved_backend;
  };

  struct BatchSelection {
    RequestBatch batch;
    std::vector<std::shared_ptr<PendingRequest>> pending;
    std::size_t total_tokens{0};
  };

  void WorkerLoop();
  void ProcessBatch(BatchSelection selection);
  void DecodeWorkerLoop();
  void StartDecodeWorkers();
  void StopDecodeWorkers();
  void ApplyFairness(BatchSelection *selection);
  void UpdateQueueDepthLocked() const;
  void
  ResolveBackends(const std::vector<std::shared_ptr<PendingRequest>> &batch);
  bool RequestUsesSessionHandle(const InferenceRequest &request) const;
  void ReleaseSessionState(const scheduler::SessionHandleState &state,
                           std::shared_ptr<LlamaCppBackend> backend_hint);
  void FinalizeSessionLease(PendingRequest *pending, bool commit_state);
  bool BackendUsesSplitDecodeWorkers(
      const std::shared_ptr<LlamaCppBackend> &backend) const;
  bool CanAppendToStickyStepBatchLocked(
      const std::shared_ptr<PendingRequest> &pending,
      const std::shared_ptr<LlamaCppBackend> &sticky_step_backend) const;
  std::size_t AppendCompatiblePendingDecodeLocked(
      std::vector<std::shared_ptr<PendingRequest>> *batch,
      const std::shared_ptr<LlamaCppBackend> &sticky_step_backend,
      std::size_t max_batch_size);
  std::size_t CountCompatiblePendingDecodeLocked(
      const std::shared_ptr<LlamaCppBackend> &sticky_step_backend) const;
  void PollDeferredSequenceRetirements();
  void RefreshNativeKvMemoryMetrics() const;
  void SyncSequenceSlotProgress(const InferenceRequest &request) const;

  struct DeferredSequenceRetirement {
    std::shared_ptr<LlamaCppBackend> backend;
    scheduler::SequenceLease lease;
    SequenceReleaseFence fence;
    std::chrono::steady_clock::time_point retired_at;
  };

  BatchSelection BuildBatchLocked();

  // Swapping logic (§P1c).
  bool TrySwapIn(InferenceRequest &inf);
  bool TrySwapOut(InferenceRequest &inf);

  SimpleTokenizer &tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<ModelRouter> router_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  std::shared_ptr<RadixPrefixCache> prefix_cache_;
  std::vector<std::shared_ptr<PendingRequest>> pending_prefill_;
  std::vector<std::shared_ptr<PendingRequest>> pending_decode_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::thread worker_;
  std::vector<std::thread> decode_workers_;
  bool use_decode_workers_{false};
  std::atomic<bool> stop_{false};
  std::unique_ptr<BatchExecutor> executor_;
  std::atomic<uint64_t> next_batch_id_{0};
  std::atomic<uint64_t> next_sequence_{0};
  FairnessController fairness_controller_;
  FairnessConfig fairness_config_;
  DisaggregatedConfig disagg_config_;
  Config config_;
  mutable std::mutex model_selection_options_mutex_;
  ModelSelectionOptions model_selection_options_;

  std::atomic<int> live_decode_workers_{
      0}; // live thread count; 0 = no pool or all exited

  // Universal sequence slot manager for KV cache tracking across both backends.
  std::unique_ptr<scheduler::SequenceSlotManager> slot_manager_;
  std::mutex deferred_sequence_retirements_mutex_;
  std::vector<DeferredSequenceRetirement> deferred_sequence_retirements_;
  std::unique_ptr<scheduler::SessionHandleManager> session_handle_manager_;
  std::thread eviction_thread_;
  std::atomic<bool> eviction_running_{false};
  std::mutex eviction_mutex_;
  std::condition_variable eviction_cv_;
  void EvictionWorkerLoop();
};

} // namespace inferflux
