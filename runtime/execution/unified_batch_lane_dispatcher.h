#pragma once

#include "runtime/backends/llama/llama_cpp_backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace inferflux {

// Persistent two-lane dispatcher used by native CUDA overlap paths.
// It provides bounded submit/collect semantics keyed by UnifiedBatchHandle.
class UnifiedBatchLaneDispatcher {
public:
  using UnifiedBatchInput = LlamaCppBackend::UnifiedBatchInput;
  using UnifiedBatchOutput = LlamaCppBackend::UnifiedBatchOutput;
  using UnifiedBatchHandle = LlamaCppBackend::UnifiedBatchHandle;

  struct Config {
    std::size_t max_pending_per_lane{64};
  };

  struct ExecutionResult {
    bool success{false};
    std::vector<UnifiedBatchOutput> outputs;
    std::string error;
    double elapsed_ms{0.0};
  };

  using ExecuteFn = std::function<ExecutionResult(
      const std::vector<UnifiedBatchInput> &inputs, bool decode_lane)>;

  enum class CollectStatus {
    kPending,
    kSuccess,
    kFailed,
    kMissing,
  };

  UnifiedBatchLaneDispatcher();
  explicit UnifiedBatchLaneDispatcher(Config config);
  ~UnifiedBatchLaneDispatcher();

  UnifiedBatchLaneDispatcher(const UnifiedBatchLaneDispatcher &) = delete;
  UnifiedBatchLaneDispatcher &
  operator=(const UnifiedBatchLaneDispatcher &) = delete;

  bool Start(ExecuteFn execute_fn);
  void Stop();
  bool IsRunning() const;

  UnifiedBatchHandle Submit(const std::vector<UnifiedBatchInput> &inputs,
                            bool decode_lane);

  CollectStatus TryCollect(UnifiedBatchHandle handle,
                           std::vector<UnifiedBatchOutput> *outputs,
                           bool *decode_lane = nullptr,
                           std::string *error = nullptr,
                           double *elapsed_ms = nullptr);

  std::size_t PendingCount(bool decode_lane) const;

private:
  struct WorkItem {
    UnifiedBatchHandle handle{0};
    bool decode_lane{false};
    std::vector<UnifiedBatchInput> inputs;
  };

  struct PendingState {
    bool decode_lane{false};
    bool ready{false};
    bool success{false};
    std::vector<UnifiedBatchOutput> outputs;
    std::string error;
    double elapsed_ms{0.0};
  };

  void WorkerLoop(bool decode_lane);
  static std::size_t NormalizePendingLimit(std::size_t value);

  Config config_;
  ExecuteFn execute_fn_;

  std::atomic<UnifiedBatchHandle> next_handle_{1};
  bool running_{false};
  bool stopping_{false};

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread decode_worker_;
  std::thread prefill_worker_;
  std::deque<WorkItem> decode_queue_;
  std::deque<WorkItem> prefill_queue_;
  std::unordered_map<UnifiedBatchHandle, PendingState> pending_;
  std::size_t pending_decode_count_{0};
  std::size_t pending_prefill_count_{0};
};

} // namespace inferflux
