#include "runtime/scheduler/sequence_slot_manager.h"
#include <catch2/catch_amalgamated.hpp>
#include <chrono>
#include <thread>

namespace inferflux {
namespace scheduler {

TEST_CASE("SequenceSlotManager basic operations", "[slot_manager]") {
  SequenceSlotManager manager(16);

  SECTION("Initial state") {
    REQUIRE(manager.GetMaxSlots() == 16);
    REQUIRE(manager.GetUsedSlotCount() == 0);
    REQUIRE(manager.GetFreeSlotCount() == 16);
  }

  SECTION("Acquire and release slots") {
    auto slot1 = manager.AcquireSlot(100);
    REQUIRE(slot1.has_value());
    REQUIRE(*slot1 >= 0);
    REQUIRE(*slot1 < 16);
    REQUIRE(manager.GetUsedSlotCount() == 1);
    REQUIRE(manager.GetFreeSlotCount() == 15);

    auto slot2 = manager.AcquireSlot(101);
    REQUIRE(slot2.has_value());
    REQUIRE(*slot2 != *slot1); // Different slots
    REQUIRE(manager.GetUsedSlotCount() == 2);

    manager.ReleaseSlot(*slot1);
    REQUIRE(manager.GetUsedSlotCount() == 1);
    REQUIRE(manager.GetFreeSlotCount() == 15);

    manager.ReleaseSlot(*slot2);
    REQUIRE(manager.GetUsedSlotCount() == 0);
    REQUIRE(manager.GetFreeSlotCount() == 16);
  }

  SECTION("Slot exhaustion") {
    std::vector<int> acquired_slots;

    // Acquire all 16 slots
    for (int i = 0; i < 16; i++) {
      auto slot = manager.AcquireSlot(200 + i);
      REQUIRE(slot.has_value());
      acquired_slots.push_back(*slot);
    }

    REQUIRE(manager.GetUsedSlotCount() == 16);
    REQUIRE(manager.GetFreeSlotCount() == 0);

    // Next acquire should fail
    auto slot = manager.AcquireSlot(999);
    REQUIRE(!slot.has_value());

    // Release one slot
    manager.ReleaseSlot(acquired_slots[0]);
    REQUIRE(manager.GetUsedSlotCount() == 15);
    REQUIRE(manager.GetFreeSlotCount() == 1);

    // Now acquire should succeed
    auto new_slot = manager.AcquireSlot(1000);
    REQUIRE(new_slot.has_value());
  }
}

TEST_CASE("SequenceSlotManager lease generations invalidate stale slot reuse",
          "[slot_manager]") {
  SequenceSlotManager manager(2);

  auto first = manager.AcquireLease(100);
  REQUIRE(first.has_value());
  REQUIRE(first->slot_id >= 0);
  REQUIRE(first->generation == 1);
  REQUIRE(first->request_id == 100);
  REQUIRE(manager.IsLiveLease(*first));
  REQUIRE(manager.CurrentGeneration(first->slot_id) == first->generation);

  REQUIRE(manager.ReleaseLease(*first));
  REQUIRE_FALSE(manager.IsLiveLease(*first));

  auto reused = manager.AcquireLease(101);
  REQUIRE(reused.has_value());
  REQUIRE(reused->slot_id == first->slot_id);
  REQUIRE(reused->generation == first->generation + 1);
  REQUIRE_FALSE(manager.IsLiveLease(*first));
  REQUIRE(manager.IsLiveLease(*reused));
  REQUIRE(manager.CurrentGeneration(reused->slot_id) == reused->generation);
}

TEST_CASE("SequenceSlotManager rejects stale lease release after slot reuse",
          "[slot_manager]") {
  SequenceSlotManager manager(1);

  auto first = manager.AcquireLease(200);
  REQUIRE(first.has_value());
  REQUIRE(manager.ReleaseLease(*first));

  auto second = manager.AcquireLease(201);
  REQUIRE(second.has_value());
  REQUIRE(second->slot_id == first->slot_id);
  REQUIRE(second->generation > first->generation);

  REQUIRE_FALSE(manager.ReleaseLease(*first));
  REQUIRE(manager.IsLiveLease(*second));
  REQUIRE(manager.GetUsedSlotCount() == 1);

  REQUIRE(manager.ReleaseLease(*second));
  REQUIRE(manager.GetUsedSlotCount() == 0);
}

TEST_CASE("SequenceSlotManager defers slot reuse until retire fence is ready",
          "[slot_manager]") {
  SequenceSlotManager manager(1);

  auto lease = manager.AcquireLease(300);
  REQUIRE(lease.has_value());
  REQUIRE(manager.GetUsedSlotCount() == 1);

  REQUIRE(manager.RetireLease(
      *lease, SequenceRetireFence::After(std::chrono::milliseconds(25))));
  REQUIRE(manager.GetUsedSlotCount() == 0);
  REQUIRE(manager.GetRetiringSlotCount() == 1);
  REQUIRE_FALSE(manager.IsLiveLease(*lease));

  auto blocked = manager.AcquireLease(301);
  REQUIRE_FALSE(blocked.has_value());

  auto status = manager.GetSlotStatus();
  REQUIRE(status.size() == 1);
  REQUIRE(status[0].state == SequenceState::kRetiring);
  REQUIRE(status[0].generation == lease->generation);

  std::this_thread::sleep_for(std::chrono::milliseconds(35));
  REQUIRE(manager.ReapRetiredSlots() == 1);
  REQUIRE(manager.GetRetiringSlotCount() == 0);
  REQUIRE(manager.GetFreeSlotCount() == 1);

  auto reacquired = manager.AcquireLease(302);
  REQUIRE(reacquired.has_value());
  REQUIRE(reacquired->slot_id == lease->slot_id);
  REQUIRE(reacquired->generation == lease->generation + 1);
}

TEST_CASE("SequenceSlotManager can explicitly complete a retiring lease",
          "[slot_manager]") {
  SequenceSlotManager manager(1);

  auto lease = manager.AcquireLease(400);
  REQUIRE(lease.has_value());
  REQUIRE(manager.RetireLease(*lease, SequenceRetireFence::Pending()));
  REQUIRE(manager.GetRetiringSlotCount() == 1);

  REQUIRE(manager.CompleteRetiredLease(*lease));
  REQUIRE(manager.GetRetiringSlotCount() == 0);
  REQUIRE(manager.GetFreeSlotCount() == 1);
}

TEST_CASE("SequenceSlotManager can retain and restore completed session leases",
          "[slot_manager]") {
  SequenceSlotManager manager(1);

  auto lease = manager.AcquireLease(500);
  REQUIRE(lease.has_value());

  manager.MarkProcessing(lease->slot_id);
  manager.UpdateTokenCount(lease->slot_id, 128);
  REQUIRE(manager.MarkCompleted(*lease, 128));
  REQUIRE(manager.GetUsedSlotCount() == 0);
  REQUIRE(manager.GetFreeSlotCount() == 0);

  auto status = manager.GetSlotStatus();
  REQUIRE(status.size() == 1);
  REQUIRE(status[0].state == SequenceState::kCompleted);
  REQUIRE(status[0].token_count == 128);

  REQUIRE(manager.RestoreLease(*lease, /*request_id=*/501,
                               /*sequence_id=*/lease->slot_id,
                               /*token_count=*/96));
  status = manager.GetSlotStatus();
  REQUIRE(status[0].state == SequenceState::kPrefilling);
  REQUIRE(status[0].request_id == 501);
  REQUIRE(status[0].sequence_id == lease->slot_id);
  REQUIRE(status[0].token_count == 96);
  REQUIRE(manager.GetUsedSlotCount() == 1);
}

TEST_CASE("SequenceSlotManager timeout-based eviction", "[slot_manager]") {
  SequenceSlotManager manager(16);

  // Set short timeout for testing (100ms)
  manager.SetIdleTimeout(std::chrono::milliseconds(100));

  // Acquire a slot
  auto slot = manager.AcquireSlot(100);
  REQUIRE(slot.has_value());

  // Mark as processing (transitions to Decoding state)
  manager.MarkProcessing(*slot);

  // Wait longer than timeout
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Run eviction - should evict the idle slot
  auto evicted = manager.EvictIdleSlots(std::chrono::milliseconds(100));
  REQUIRE(evicted.size() == 1);
  REQUIRE(evicted[0].first == *slot); // Evicted slot ID matches

  // Slot should now be available
  REQUIRE(manager.GetUsedSlotCount() == 0);
}

TEST_CASE("SequenceSlotManager slot status", "[slot_manager]") {
  SequenceSlotManager manager(4);

  auto slot = manager.AcquireSlot(100);
  REQUIRE(slot.has_value());

  auto status = manager.GetSlotStatus();
  REQUIRE(status.size() == 4);

  // Find the acquired slot
  bool found = false;
  for (const auto &s : status) {
    if (s.slot_id == *slot) {
      REQUIRE(s.request_id == 100);
      REQUIRE(s.generation == 1);
      REQUIRE(s.state == SequenceState::kPrefilling);
      found = true;
      break;
    }
  }
  REQUIRE(found);
}

} // namespace scheduler
} // namespace inferflux
