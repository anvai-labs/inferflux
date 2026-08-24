#include "runtime/execution/unified_batch_lane_dispatcher.h"

#include <algorithm>
#include <exception>
#include <utility>

namespace inferflux {

namespace {

constexpr std::size_t kDefaultPendingLimit = 64;

} // namespace

UnifiedBatchLaneDispatcher::UnifiedBatchLaneDispatcher(Config config)
    : config_(config) {
  config_.max_pending_per_lane =
      NormalizePendingLimit(config_.max_pending_per_lane);
}

UnifiedBatchLaneDispatcher::UnifiedBatchLaneDispatcher()
    : UnifiedBatchLaneDispatcher(Config{}) {}

UnifiedBatchLaneDispatcher::~UnifiedBatchLaneDispatcher() { Stop(); }

bool UnifiedBatchLaneDispatcher::Start(ExecuteFn execute_fn) {
  if (!execute_fn) {
    return false;
  }

  Stop();

  std::lock_guard<std::mutex> lock(mutex_);
  execute_fn_ = std::move(execute_fn);
  stopping_ = false;
  running_ = true;
  decode_worker_ = std::thread(&UnifiedBatchLaneDispatcher::WorkerLoop, this,
                               /*decode_lane=*/true);
  prefill_worker_ = std::thread(&UnifiedBatchLaneDispatcher::WorkerLoop, this,
                                /*decode_lane=*/false);
  return true;
}

void UnifiedBatchLaneDispatcher::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stopping_ = true;
    decode_queue_.clear();
    prefill_queue_.clear();
  }
  cv_.notify_all();

  if (decode_worker_.joinable()) {
    decode_worker_.join();
  }
  if (prefill_worker_.joinable()) {
    prefill_worker_.join();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  pending_.clear();
  pending_decode_count_ = 0;
  pending_prefill_count_ = 0;
  execute_fn_ = {};
  stopping_ = false;
  running_ = false;
}

bool UnifiedBatchLaneDispatcher::IsRunning() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_ && !stopping_;
}

UnifiedBatchHandle
UnifiedBatchLaneDispatcher::Submit(const std::vector<UnifiedBatchInput> &inputs,
                                   bool decode_lane) {
  if (inputs.empty()) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_ || stopping_ || !execute_fn_) {
    return 0;
  }

  std::size_t &lane_pending =
      decode_lane ? pending_decode_count_ : pending_prefill_count_;
  if (lane_pending >= config_.max_pending_per_lane) {
    return 0;
  }

  const UnifiedBatchHandle handle = next_handle_.fetch_add(1);
  PendingState state;
  state.decode_lane = decode_lane;
  pending_.emplace(handle, std::move(state));
  lane_pending++;

  WorkItem item;
  item.handle = handle;
  item.decode_lane = decode_lane;
  item.inputs = inputs;
  auto &queue = decode_lane ? decode_queue_ : prefill_queue_;
  queue.push_back(std::move(item));
  cv_.notify_all();
  return handle;
}

UnifiedBatchLaneDispatcher::CollectStatus
UnifiedBatchLaneDispatcher::TryCollect(UnifiedBatchHandle handle,
                                       std::vector<UnifiedBatchOutput> *outputs,
                                       bool *decode_lane, std::string *error,
                                       double *elapsed_ms) {
  if (handle == 0) {
    return CollectStatus::kMissing;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_.find(handle);
  if (it == pending_.end()) {
    return CollectStatus::kMissing;
  }

  PendingState &state = it->second;
  if (decode_lane) {
    *decode_lane = state.decode_lane;
  }
  if (!state.ready) {
    return CollectStatus::kPending;
  }

  if (error) {
    *error = state.error;
  }
  if (elapsed_ms) {
    *elapsed_ms = state.elapsed_ms;
  }

  std::size_t &lane_pending =
      state.decode_lane ? pending_decode_count_ : pending_prefill_count_;
  if (lane_pending > 0) {
    lane_pending--;
  }

  CollectStatus status =
      state.success ? CollectStatus::kSuccess : CollectStatus::kFailed;
  if (status == CollectStatus::kSuccess && outputs) {
    *outputs = std::move(state.outputs);
  }
  pending_.erase(it);
  return status;
}

std::size_t UnifiedBatchLaneDispatcher::PendingCount(bool decode_lane) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return decode_lane ? pending_decode_count_ : pending_prefill_count_;
}

void UnifiedBatchLaneDispatcher::WorkerLoop(bool decode_lane) {
  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto &queue = decode_lane ? decode_queue_ : prefill_queue_;
      cv_.wait(lock, [&]() { return stopping_ || !queue.empty(); });
      if (stopping_ && queue.empty()) {
        return;
      }
      item = std::move(queue.front());
      queue.pop_front();
    }

    ExecutionResult result;
    try {
      result = execute_fn_(item.inputs, decode_lane);
    } catch (const std::exception &e) {
      result.success = false;
      result.error = e.what();
      result.outputs.clear();
    } catch (...) {
      result.success = false;
      result.error = "unknown lane worker error";
      result.outputs.clear();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto pending_it = pending_.find(item.handle);
    if (pending_it == pending_.end()) {
      continue;
    }
    PendingState &state = pending_it->second;
    state.ready = true;
    state.success = result.success;
    state.outputs = std::move(result.outputs);
    state.error = std::move(result.error);
    state.elapsed_ms = result.elapsed_ms;
  }
}

std::size_t
UnifiedBatchLaneDispatcher::NormalizePendingLimit(std::size_t value) {
  return std::max<std::size_t>(1, value == 0 ? kDefaultPendingLimit : value);
}

} // namespace inferflux
