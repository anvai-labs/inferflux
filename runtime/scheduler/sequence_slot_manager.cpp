#include "runtime/scheduler/sequence_slot_manager.h"
#include "server/logging/logger.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string_view>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace inferflux {
namespace scheduler {

namespace {

bool SequenceSlotDebugEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("INFERFLUX_DEBUG_SEQUENCE_SLOTS");
    return value && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "FALSE";
  }();
  return enabled;
}

std::string SequenceStateToString(SequenceState state) {
  switch (state) {
  case SequenceState::kIdle:
    return "idle";
  case SequenceState::kPrefilling:
    return "prefilling";
  case SequenceState::kDecoding:
    return "decoding";
  case SequenceState::kRetiring:
    return "retiring";
  case SequenceState::kCompleted:
    return "completed";
  case SequenceState::kEvicted:
    return "evicted";
  }
  return "unknown";
}

void LogSlotEvent(std::string_view stage, const SequenceSlot &slot,
                  std::string_view detail = {}) {
  if (!SequenceSlotDebugEnabled()) {
    return;
  }
  std::string message = "slot[" + std::string(stage) +
                        "]: slot_id=" + std::to_string(slot.slot_id) +
                        ", request_id=" + std::to_string(slot.request_id) +
                        ", sequence_id=" + std::to_string(slot.sequence_id) +
                        ", generation=" + std::to_string(slot.generation) +
                        ", state=" + SequenceStateToString(slot.state) +
                        ", tokens=" + std::to_string(slot.token_count);
  if (!detail.empty()) {
    message += ", detail=" + std::string(detail);
  }
  log::Info("slot_manager", message);
}

} // namespace

SequenceSlotManager::SequenceSlotManager(size_t max_slots)
    : max_slots_(max_slots) {
  InitializeSlots();
  log::Info(
      "slot_manager",
      "Initialized SequenceSlotManager with " + std::to_string(max_slots_) +
          " slots (timeout: " + std::to_string(idle_timeout_.count()) + "ms)");
}

void SequenceSlotManager::InitializeSlots() {
  slots_.resize(max_slots_);

  for (size_t i = 0; i < max_slots_; ++i) {
    slots_[i].slot_id = static_cast<int>(i);
    slots_[i].state = SequenceState::kIdle;
    slots_[i].request_id = -1;
    slots_[i].sequence_id = -1;
    slots_[i].generation = 0;
    slots_[i].token_count = 0;
    slots_[i].last_access = std::chrono::steady_clock::now();
    slots_[i].acquired_at = std::chrono::steady_clock::now();
    slots_[i].retire_ready_at = std::chrono::steady_clock::time_point{};
  }
}

std::optional<SequenceLease>
SequenceSlotManager::AcquireLease(int64_t request_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  ReapRetiredSlotsLocked(std::chrono::steady_clock::now());
  EvictIdleSlotsLocked(idle_timeout_);

  auto slot = FindFreeSlot();
  if (!slot) {
    const auto retiring = std::count_if(
        slots_.begin(), slots_.end(), [](const SequenceSlot &entry) {
          return entry.state == SequenceState::kRetiring;
        });
    log::Warn("slot_manager",
              "No free slots available (request " + std::to_string(request_id) +
                  ", used: " + std::to_string(GetUsedSlotCountLocked()) + "/" +
                  std::to_string(max_slots_) +
                  ", retiring: " + std::to_string(retiring) + ")");
    return std::nullopt;
  }

  auto &entry = slots_[*slot];
  const auto now = std::chrono::steady_clock::now();
  ++entry.generation;
  entry.state = SequenceState::kPrefilling;
  entry.request_id = request_id;
  entry.sequence_id = -1;
  entry.acquired_at = now;
  entry.last_access = now;
  entry.retire_ready_at = std::chrono::steady_clock::time_point{};
  entry.token_count = 0;

  log::Debug("slot_manager",
             "Acquired slot " + std::to_string(*slot) +
                 " gen=" + std::to_string(entry.generation) + " for request " +
                 std::to_string(request_id) + " (free slots: " +
                 std::to_string(GetFreeSlotCountLocked()) + ")");
  LogSlotEvent("acquire", entry);

  return SequenceLease{*slot, entry.generation, request_id};
}

std::optional<int> SequenceSlotManager::AcquireSlot(int64_t request_id) {
  auto lease = AcquireLease(request_id);
  return lease ? std::optional<int>(lease->slot_id) : std::nullopt;
}

void SequenceSlotManager::ReleaseSlot(int slot_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    log::Error("slot_manager", "Invalid slot_id: " + std::to_string(slot_id));
    return;
  }

  auto &slot = slots_[slot_id];

  log::Debug("slot_manager",
             "Releasing slot " + std::to_string(slot_id) + " (request " +
                 std::to_string(slot.request_id) +
                 ", tokens: " + std::to_string(slot.token_count) + ")");
  LogSlotEvent("release", slot);

  ResetSlotForReuse(slot);
}

bool SequenceSlotManager::RetireLease(const SequenceLease &lease,
                                      SequenceRetireFence fence) {
  if (!lease.IsValid()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (lease.slot_id < 0 || lease.slot_id >= static_cast<int>(max_slots_)) {
    return false;
  }

  auto &slot = slots_[lease.slot_id];
  if (slot.generation != lease.generation ||
      (slot.state != SequenceState::kPrefilling &&
       slot.state != SequenceState::kDecoding &&
       slot.state != SequenceState::kCompleted)) {
    return false;
  }

  slot.state = SequenceState::kRetiring;
  slot.retire_ready_at = fence.ready_at;
  slot.last_access = std::chrono::steady_clock::now();

  log::Debug("slot_manager",
             "Retiring lease slot=" + std::to_string(lease.slot_id) + " gen=" +
                 std::to_string(lease.generation) + " ready_at_pending");
  LogSlotEvent("retire", slot);
  return true;
}

bool SequenceSlotManager::ReleaseLease(const SequenceLease &lease) {
  if (!RetireLease(lease, SequenceRetireFence::Immediate())) {
    return false;
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return ReapRetiredSlotsLocked(std::chrono::steady_clock::now()) > 0;
}

bool SequenceSlotManager::CompleteRetiredLease(const SequenceLease &lease) {
  if (!lease.IsValid()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (lease.slot_id < 0 || lease.slot_id >= static_cast<int>(max_slots_)) {
    return false;
  }

  auto &slot = slots_[lease.slot_id];
  if (slot.generation != lease.generation ||
      slot.state != SequenceState::kRetiring) {
    return false;
  }

  LogSlotEvent("retire_complete", slot);
  ResetSlotForReuse(slot);
  return true;
}

size_t SequenceSlotManager::ReapRetiredSlots() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return ReapRetiredSlotsLocked(std::chrono::steady_clock::now());
}

// Internal version - assumes lock is already held
std::vector<std::pair<int, int>>
SequenceSlotManager::EvictIdleSlotsLocked(std::chrono::milliseconds timeout) {
  // Note: mutex_ should already be locked by caller
  std::vector<std::pair<int, int>> evicted_slots;

  for (auto &slot : slots_) {
    if (slot.state == SequenceState::kDecoding && slot.IsIdle(timeout)) {
      log::Info("slot_manager",
                "Evicting idle slot " + std::to_string(slot.slot_id) +
                    " (request " + std::to_string(slot.request_id) +
                    ", seq_id " + std::to_string(slot.sequence_id) +
                    ", idle for >" + std::to_string(timeout.count()) + "ms, " +
                    std::to_string(slot.token_count) + " tokens)");

      // Record evicted slot with its sequence ID for backend cleanup
      evicted_slots.push_back({slot.slot_id, slot.sequence_id});

      // Mark as evicted - will be reused after backend cleanup
      slot.state = SequenceState::kEvicted;
      LogSlotEvent("evict", slot);
    }
  }

  if (!evicted_slots.empty()) {
    log::Info("slot_manager",
              "Evicted " + std::to_string(evicted_slots.size()) +
                  " idle slots (timeout: " + std::to_string(timeout.count()) +
                  "ms)");
  }

  return evicted_slots;
}

// Public version - acquires lock
std::vector<std::pair<int, int>>
SequenceSlotManager::EvictIdleSlots(std::chrono::milliseconds timeout) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return EvictIdleSlotsLocked(timeout);
}

void SequenceSlotManager::MarkProcessing(int slot_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    return;
  }

  slots_[slot_id].last_access = std::chrono::steady_clock::now();

  // Transition from prefill to decode if needed
  if (slots_[slot_id].state == SequenceState::kPrefilling) {
    slots_[slot_id].state = SequenceState::kDecoding;
    LogSlotEvent("mark_processing", slots_[slot_id], "transition=decode");
  }
}

void SequenceSlotManager::UpdateTokenCount(int slot_id, int token_count) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    return;
  }

  slots_[slot_id].token_count = token_count;
  LogSlotEvent("update_tokens", slots_[slot_id]);
}

std::optional<uint64_t>
SequenceSlotManager::CurrentGeneration(int slot_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    return std::nullopt;
  }
  return slots_[slot_id].generation;
}

bool SequenceSlotManager::IsLiveLease(const SequenceLease &lease) const {
  if (!lease.IsValid()) {
    return false;
  }

  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (lease.slot_id < 0 || lease.slot_id >= static_cast<int>(max_slots_)) {
    return false;
  }

  const auto &slot = slots_[lease.slot_id];
  return slot.generation == lease.generation &&
         slot.request_id == lease.request_id &&
         (slot.state == SequenceState::kPrefilling ||
          slot.state == SequenceState::kDecoding);
}

size_t SequenceSlotManager::GetRetiringSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return std::count_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
    return s.state == SequenceState::kRetiring;
  });
}

size_t SequenceSlotManager::GetUsedSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
    return s.state == SequenceState::kPrefilling ||
           s.state == SequenceState::kDecoding;
  });
}

size_t SequenceSlotManager::GetFreeSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
    return s.state == SequenceState::kIdle ||
           s.state == SequenceState::kEvicted;
  });
}

std::vector<SequenceSlot> SequenceSlotManager::GetSlotStatus() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  // Return copies of slot status
  return slots_;
}

bool SequenceSlotManager::MarkCompleted(const SequenceLease &lease,
                                        int token_count) {
  if (!lease.IsValid()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (lease.slot_id < 0 || lease.slot_id >= static_cast<int>(max_slots_)) {
    return false;
  }

  auto &slot = slots_[lease.slot_id];
  if (slot.generation != lease.generation ||
      (slot.state != SequenceState::kPrefilling &&
       slot.state != SequenceState::kDecoding &&
       slot.state != SequenceState::kCompleted)) {
    return false;
  }

  slot.state = SequenceState::kCompleted;
  slot.token_count = std::max(0, token_count);
  slot.last_access = std::chrono::steady_clock::now();
  LogSlotEvent("mark_completed", slot);
  return true;
}

bool SequenceSlotManager::RestoreLease(const SequenceLease &lease,
                                       int64_t request_id, int sequence_id,
                                       int token_count) {
  if (!lease.IsValid()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (lease.slot_id < 0 || lease.slot_id >= static_cast<int>(max_slots_)) {
    return false;
  }

  auto &slot = slots_[lease.slot_id];
  if (slot.generation != lease.generation ||
      slot.state != SequenceState::kCompleted) {
    return false;
  }

  const auto now = std::chrono::steady_clock::now();
  slot.request_id = request_id;
  slot.sequence_id = sequence_id;
  slot.state = SequenceState::kPrefilling;
  slot.token_count = std::max(0, token_count);
  slot.last_access = now;
  slot.acquired_at = now;
  slot.retire_ready_at = std::chrono::steady_clock::time_point{};
  LogSlotEvent("restore", slot);
  return true;
}

std::optional<int> SequenceSlotManager::FindFreeSlot() {
  // Note: mutex_ should already be locked by caller
  auto it =
      std::find_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
        return s.state == SequenceState::kIdle ||
               s.state == SequenceState::kEvicted;
      });

  if (it != slots_.end()) {
    return static_cast<int>(it - slots_.begin());
  }

  return std::nullopt;
}

std::optional<int>
SequenceSlotManager::FindSlotByRequestId(int64_t request_id) {
  // Note: mutex_ should already be locked by caller
  auto it = std::find_if(slots_.begin(), slots_.end(),
                         [request_id](const SequenceSlot &s) {
                           return s.request_id == request_id &&
                                  (s.state == SequenceState::kPrefilling ||
                                   s.state == SequenceState::kDecoding);
                         });

  if (it != slots_.end()) {
    return static_cast<int>(it - slots_.begin());
  }

  return std::nullopt;
}

void SequenceSlotManager::SetMaxSlots(size_t max_slots) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (max_slots == max_slots_) {
    return; // No change
  }

  log::Info("slot_manager", "Resizing slots: " + std::to_string(max_slots_) +
                                " -> " + std::to_string(max_slots));

  max_slots_ = max_slots;
  InitializeSlots();
}

void SequenceSlotManager::SetIdleTimeout(std::chrono::milliseconds timeout) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  idle_timeout_ = timeout;

  log::Info("slot_manager",
            "Idle timeout set to " + std::to_string(timeout.count()) + "ms");
}

size_t SequenceSlotManager::GetFreeSlotCountLocked() const {
  // Note: mutex_ should already be locked by caller
  return std::count_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
    return s.state == SequenceState::kIdle ||
           s.state == SequenceState::kEvicted;
  });
}

size_t SequenceSlotManager::GetUsedSlotCountLocked() const {
  // Note: mutex_ should already be locked by caller
  return std::count_if(slots_.begin(), slots_.end(), [](const SequenceSlot &s) {
    return s.state == SequenceState::kPrefilling ||
           s.state == SequenceState::kDecoding;
  });
}

size_t SequenceSlotManager::ReapRetiredSlotsLocked(
    std::chrono::steady_clock::time_point now) {
  size_t reaped = 0;
  for (auto &slot : slots_) {
    if (slot.state == SequenceState::kRetiring && slot.retire_ready_at <= now) {
      ResetSlotForReuse(slot);
      ++reaped;
    }
  }
  return reaped;
}

void SequenceSlotManager::ResetSlotForReuse(SequenceSlot &slot) {
  slot.state = SequenceState::kIdle;
  slot.request_id = -1;
  slot.sequence_id = -1;
  slot.token_count = 0;
  slot.retire_ready_at = std::chrono::steady_clock::time_point{};
}

bool SequenceSlotManager::CanAcceptRequest() const {
#ifdef ENABLE_CUDA
  // Get available GPU memory
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);

  if (err != cudaSuccess) {
    log::Warn("slot_manager", "Failed to query GPU memory: " +
                                  std::string(cudaGetErrorString(err)) +
                                  ", allowing request (fallback behavior)");
    return true; // Allow request if we can't check memory
  }

  // Calculate memory pressure
  size_t used_bytes = total_bytes - free_bytes;
  double pressure_pct = (static_cast<double>(used_bytes) / total_bytes) * 100.0;

  // Log memory status periodically
  log::Debug("slot_manager",
             "GPU memory: " + std::to_string(free_bytes / 1024 / 1024) +
                 " MB free, " + std::to_string(total_bytes / 1024 / 1024) +
                 " MB total (" +
                 std::to_string(static_cast<int>(pressure_pct)) + "% used)");

  // Reject if memory pressure > 90%
  if (pressure_pct > 90.0) {
    log::Warn("slot_manager",
              "Rejecting request: memory pressure too high (" +
                  std::to_string(static_cast<int>(pressure_pct)) + "% > 90%)");
    return false;
  }

  // Allow request if memory is available
  return true;
#else
  // Non-CUDA builds: always allow (no GPU memory to check)
  return true;
#endif
}

int SequenceSlotManager::GetMemoryPressure() const {
#ifdef ENABLE_CUDA
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);

  if (err != cudaSuccess) {
    return -1; // Error: unavailable
  }

  size_t used_bytes = total_bytes - free_bytes;
  return static_cast<int>((static_cast<double>(used_bytes) / total_bytes) *
                          100.0);
#else
  return -1; // Not available on non-CUDA builds
#endif
}

bool SequenceSlotManager::PerformGracefulDegradation() {
#ifdef ENABLE_CUDA
  std::unique_lock<std::shared_mutex> lock(mutex_);

  int pressure = GetMemoryPressure();
  if (pressure < 0) {
    return false; // Can't check memory pressure
  }

  // Degradation thresholds
  constexpr int kWarningThreshold = 80;  // 80% - log warning
  constexpr int kDegradeThreshold = 85;  // 85% - reduce slots
  constexpr int kCriticalThreshold = 90; // 90% - aggressive reduction

  if (pressure >= kCriticalThreshold) {
    // Aggressive degradation: reduce to half or minimum
    size_t old_max = max_slots_;
    size_t new_max = std::max(8UL, old_max / 2);

    if (new_max < old_max) {
      max_slots_ = new_max;
      log::Warn("slot_manager",
                "CRITICAL memory pressure (" + std::to_string(pressure) +
                    "%): reducing max_slots from " + std::to_string(old_max) +
                    " to " + std::to_string(max_slots_));
      return true;
    }
  } else if (pressure >= kDegradeThreshold) {
    // Moderate degradation: reduce by 25%
    size_t old_max = max_slots_;
    size_t new_max = std::max(16UL, (old_max * 3) / 4);

    if (new_max < old_max) {
      max_slots_ = new_max;
      log::Warn("slot_manager",
                "High memory pressure (" + std::to_string(pressure) +
                    "%): reducing max_slots from " + std::to_string(old_max) +
                    " to " + std::to_string(max_slots_));
      return true;
    }
  } else if (pressure >= kWarningThreshold) {
    // Warning only: no degradation yet
    log::Warn("slot_manager", "Elevated memory pressure (" +
                                  std::to_string(pressure) +
                                  "%), monitoring but not degrading yet");
  }

  return false; // No degradation performed
#else
  return false; // Not available on non-CUDA builds
#endif
}

} // namespace scheduler
} // namespace inferflux
