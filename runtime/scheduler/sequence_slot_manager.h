#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

namespace inferflux {
namespace scheduler {

/**
 * @brief State of a sequence slot in the KV cache
 */
enum class SequenceState {
  kIdle,       // Available for new requests
  kPrefilling, // Processing prompt (prefill phase)
  kDecoding,   // Generating tokens (decode phase)
  kRetiring,   // Backend cleanup fence pending before reuse
  kCompleted,  // Finished, waiting for cleanup
  kEvicted     // Timed out and evicted
};

/**
 * @brief Fence metadata for deferred slot reuse.
 *
 * The initial implementation is time-based so the scheduler can prove
 * generation-safe delayed reuse in tests. Future native/backend integrations
 * can replace the ready_at contract with CUDA/backend completion signals while
 * keeping the same slot-manager state machine.
 */
struct SequenceRetireFence {
  std::chrono::steady_clock::time_point ready_at{};

  static SequenceRetireFence Immediate() {
    return SequenceRetireFence{std::chrono::steady_clock::now()};
  }

  static SequenceRetireFence After(std::chrono::milliseconds grace_period) {
    return SequenceRetireFence{
        std::chrono::steady_clock::now() +
        std::max(std::chrono::milliseconds(0), grace_period)};
  }

  static SequenceRetireFence Pending() {
    return SequenceRetireFence{std::chrono::steady_clock::time_point::max()};
  }
};

/**
 * @brief Represents a single KV cache slot for one sequence
 */
struct SequenceSlot {
  int slot_id;         // Slot number (0 to max_slots-1)
  int64_t request_id;  // Associated request ID
  int sequence_id;     // Backend sequence ID
  uint64_t generation; // Monotonic acquisition generation for slot reuse safety
  SequenceState state;
  std::chrono::steady_clock::time_point last_access;
  std::chrono::steady_clock::time_point acquired_at;
  std::chrono::steady_clock::time_point retire_ready_at;
  int token_count; // Tokens processed in this sequence

  /**
   * @brief Check if slot has been idle longer than timeout
   * @param timeout Idle timeout threshold
   * @return true if slot is idle (not actively generating or past timeout)
   */
  bool IsIdle(std::chrono::milliseconds timeout) const {
    if (state != SequenceState::kDecoding)
      return true; // Not actively generating
    auto idle = std::chrono::steady_clock::now() - last_access;
    return idle > timeout;
  }
};

/**
 * @brief Logical lease for a physical sequence slot.
 *
 * Generation is bumped on every acquisition so higher layers can distinguish a
 * newly acquired slot from stale references to the same physical slot_id.
 */
struct SequenceLease {
  int slot_id{-1};
  uint64_t generation{0};
  int64_t request_id{-1};

  bool IsValid() const { return slot_id >= 0; }
};

/**
 * @brief Manages KV cache slots for both inferflux_cuda and llama_cpp_cuda
 * backends
 *
 * This provides universal sequence slot management at the orchestration layer,
 * above both backend implementations. Each backend maintains its own private
 * KV cache internally, but slot acquisition/release is coordinated here.
 *
 * Key benefits:
 * - Configurable slot count (increase from 16 to 128+ for production)
 * - Timeout-based eviction to prevent slot exhaustion
 * - Statistics and metrics for monitoring
 * - Thread-safe slot allocation
 */
class SequenceSlotManager {
public:
  /**
   * @brief Construct slot manager
   * @param max_slots Maximum number of concurrent sequences (default: 128)
   */
  explicit SequenceSlotManager(size_t max_slots = 128);

  /**
   * @brief Acquire a generation-stamped lease for a request.
   * @param request_id Request ID for tracking
   * @return Lease if available, nullopt if all slots in use
   */
  std::optional<SequenceLease> AcquireLease(int64_t request_id);

  /**
   * @brief Acquire a slot for a request
   * @param request_id Request ID for tracking
   * @return Slot ID if available, nullopt if all slots in use
   */
  std::optional<int> AcquireSlot(int64_t request_id);

  /**
   * @brief Mark a lease as retiring until its fence is ready.
   * @param lease Lease to retire
   * @param fence Fence controlling when the slot may be reused
   * @return true if the live lease matched and entered retiring state
   */
  bool
  RetireLease(const SequenceLease &lease,
              SequenceRetireFence fence = SequenceRetireFence::Immediate());

  /**
   * @brief Release a slot when sequence completes
   * @param slot_id Slot ID to release
   */
  void ReleaseSlot(int slot_id);

  /**
   * @brief Release a generation-validated lease when sequence completes
   * @param lease Lease to release
   * @return true if the live lease matched and was released, false otherwise
   */
  bool ReleaseLease(const SequenceLease &lease);

  /**
   * @brief Complete a lease already in retiring state and make it reusable
   * @param lease Lease to complete
   * @return true if the retiring lease matched and was reset to idle
   */
  bool CompleteRetiredLease(const SequenceLease &lease);

  /**
   * @brief Reap retiring slots whose fence is now ready.
   * @return Number of slots transitioned back to idle
   */
  size_t ReapRetiredSlots();

  /**
   * @brief Evict idle sequences based on timeout
   * @param timeout Idle timeout threshold
   * @return Vector of evicted slot IDs with their sequence IDs
   */
  std::vector<std::pair<int, int>>
  EvictIdleSlots(std::chrono::milliseconds timeout);

  /**
   * @brief Mark slot as actively processing (update last_access time)
   * @param slot_id Slot ID to mark as processing
   */
  void MarkProcessing(int slot_id);

  /**
   * @brief Update token count for a slot
   * @param slot_id Slot ID
   * @param token_count Total tokens processed
   */
  void UpdateTokenCount(int slot_id, int token_count);

  /**
   * @brief Get the current generation for a slot
   * @param slot_id Slot ID to inspect
   * @return Generation if slot exists, nullopt otherwise
   */
  std::optional<uint64_t> CurrentGeneration(int slot_id) const;

  /**
   * @brief Check whether a lease still matches the live slot owner
   * @param lease Lease to validate
   * @return true if the slot is still owned by this generation
   */
  bool IsLiveLease(const SequenceLease &lease) const;

  /**
   * @brief Get number of retiring slots that are fenced from reuse
   * @return Retiring slot count
   */
  size_t GetRetiringSlotCount() const;

  /**
   * @brief Get number of currently used slots
   * @return Used slot count
   */
  size_t GetUsedSlotCount() const;

  /**
   * @brief Get number of available slots
   * @return Free slot count
   */
  size_t GetFreeSlotCount() const;

  /**
   * @brief Get status of all slots
   * @return Vector of slot status copies
   */
  std::vector<SequenceSlot> GetSlotStatus() const;

  /**
   * @brief Mark a live lease as retained for session/prefix reuse.
   * @param lease Lease to mark completed
   * @param token_count Tokens currently resident in the sequence
   * @return true if the lease matched and was updated
   */
  bool MarkCompleted(const SequenceLease &lease, int token_count);

  /**
   * @brief Restore a completed retained lease for a new request.
   * @param lease Existing retained lease
   * @param request_id New request ID to bind
   * @param sequence_id Backend sequence ID for the retained slot
   * @param token_count Tokens already resident in the retained sequence
   * @return true if the retained lease matched and was reactivated
   */
  bool RestoreLease(const SequenceLease &lease, int64_t request_id,
                    int sequence_id, int token_count);

  /**
   * @brief Set maximum slot count (requires reinitialization)
   * @param max_slots New maximum slot count
   */
  void SetMaxSlots(size_t max_slots);

  /**
   * @brief Set idle timeout for eviction
   * @param timeout Timeout in milliseconds
   */
  void SetIdleTimeout(std::chrono::milliseconds timeout);

  /**
   * @brief Get current idle timeout
   * @return Idle timeout in milliseconds
   */
  std::chrono::milliseconds GetIdleTimeout() const { return idle_timeout_; }

  /**
   * @brief Get maximum slot count
   * @return Maximum slots
   */
  size_t GetMaxSlots() const { return max_slots_; }

  /**
   * @brief Check if we can accept a new request based on memory pressure
   * @return true if there's sufficient memory, false if we should
   * reject/degrade
   */
  bool CanAcceptRequest() const;

  /**
   * @brief Get current memory pressure status
   * @return Memory pressure as percentage (0-100), or -1 if unavailable
   */
  int GetMemoryPressure() const;

  /**
   * @brief Perform graceful degradation if under memory pressure
   * @return true if degradation was performed, false otherwise
   */
  bool PerformGracefulDegradation();

private:
  mutable std::shared_mutex mutex_;
  std::vector<SequenceSlot> slots_;
  size_t max_slots_;
  std::chrono::milliseconds idle_timeout_{300000}; // 5 minutes default

  /**
   * @brief Find available slot
   * @return Slot ID if found, nullopt if all slots in use
   */
  std::optional<int> FindFreeSlot();

  /**
   * @brief Find slot by request ID
   * @param request_id Request ID to find
   * @return Slot ID if found, nullopt if not found
   */
  std::optional<int> FindSlotByRequestId(int64_t request_id);

  /**
   * @brief Initialize all slots to idle state
   */
  void InitializeSlots();

  /**
   * @brief Internal eviction (assumes lock already held)
   * @param timeout Idle timeout threshold
   * @return Vector of evicted slot IDs with their sequence IDs
   */
  std::vector<std::pair<int, int>>
  EvictIdleSlotsLocked(std::chrono::milliseconds timeout);

  /**
   * @brief Internal retire reap (assumes lock already held)
   * @param now Time used to evaluate ready fences
   * @return Number of slots transitioned back to idle
   */
  size_t ReapRetiredSlotsLocked(std::chrono::steady_clock::time_point now);

  /**
   * @brief Reset a slot to idle/reusable state
   * @param slot Slot to clear
   */
  void ResetSlotForReuse(SequenceSlot &slot);

  /**
   * @brief Internal free slot count (assumes lock already held)
   * @return Free slot count
   */
  size_t GetFreeSlotCountLocked() const;

  /**
   * @brief Internal used slot count (assumes lock already held)
   * @return Used slot count
   */
  size_t GetUsedSlotCountLocked() const;
};

} // namespace scheduler
} // namespace inferflux
