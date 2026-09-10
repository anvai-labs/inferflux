#pragma once

#include "scheduler/scheduler.h"

namespace inferflux {

/// Friend-based accessor for Scheduler private members in unit tests.
/// Replaces the `#define private public` hack with a clean, MSVC-compatible
/// pattern.  Only available when compiled with -DINFERFLUX_TESTING.
class SchedulerTestAccess {
public:
  explicit SchedulerTestAccess(Scheduler &s) : s_(s) {}

  auto &pending_prefill() { return s_.pending_prefill_; }
  auto &pending_decode() { return s_.pending_decode_; }
  auto &queue_mutex() { return s_.queue_mutex_; }

  auto &slot_manager() { return s_.slot_manager_; }
  int max_batch_size() const { return s_.config_.max_batch_size; }

  auto BuildBatchLocked() { return s_.BuildBatchLocked(); }

  void ProcessBatch(Scheduler::BatchSelection selection) {
    s_.ProcessBatch(std::move(selection));
  }

  void ResolveBackends(
      const std::vector<std::shared_ptr<Scheduler::PendingRequest>> &batch) {
    s_.ResolveBackends(batch);
  }

  void ApplyFairness(Scheduler::BatchSelection *selection) {
    s_.ApplyFairness(selection);
  }

  std::size_t AppendCompatiblePendingDecodeLocked(
      std::vector<std::shared_ptr<Scheduler::PendingRequest>> *batch,
      const std::shared_ptr<LlamaCppBackend> &sticky_step_backend,
      std::size_t max_batch_size) {
    return s_.AppendCompatiblePendingDecodeLocked(batch, sticky_step_backend,
                                                  max_batch_size);
  }

  /// Helper to create a PendingRequest (private nested type).
  static auto MakePendingRequest() {
    return std::make_shared<Scheduler::PendingRequest>();
  }

private:
  Scheduler &s_;
};

} // namespace inferflux
