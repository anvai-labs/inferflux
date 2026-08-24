#include <catch2/catch_amalgamated.hpp>

#include "scheduler/session_handle_manager.h"

#include <chrono>
#include <thread>

using namespace inferflux;

TEST_CASE("SessionHandleManager commit and reacquire existing state",
          "[session_handles]") {
  scheduler::SessionHandleManager::Config cfg;
  cfg.max_sessions = 8;
  cfg.ttl = std::chrono::milliseconds(1000);
  scheduler::SessionHandleManager manager(cfg);

  auto lease = manager.AcquireLease("sess-a");
  REQUIRE(lease.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);
  REQUIRE_FALSE(lease.has_state);

  scheduler::SessionHandleState state;
  state.model_id = "model-a";
  state.sequence_id = 7;
  state.sequence_generation = 3;
  state.prompt_tokens = {1, 2, 3};
  state.block_table = {10, 11};
  manager.CommitLease("sess-a", state);

  auto second = manager.AcquireLease("sess-a");
  REQUIRE(second.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);
  REQUIRE(second.has_state);
  REQUIRE(second.state.model_id == "model-a");
  REQUIRE(second.state.sequence_id == 7);
  REQUIRE(second.state.sequence_generation == 3);
  REQUIRE(second.state.prompt_tokens == std::vector<int>{1, 2, 3});
  REQUIRE(second.state.block_table == std::vector<int>{10, 11});
}

TEST_CASE("SessionHandleManager enforces single in-flight lease",
          "[session_handles]") {
  scheduler::SessionHandleManager manager;

  auto first = manager.AcquireLease("sess-b");
  REQUIRE(first.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);

  auto second = manager.AcquireLease("sess-b");
  REQUIRE(second.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kBusy);

  manager.ReleaseLease("sess-b");
  auto third = manager.AcquireLease("sess-b");
  REQUIRE(third.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);
}

TEST_CASE("SessionHandleManager collects expired sessions",
          "[session_handles]") {
  scheduler::SessionHandleManager::Config cfg;
  cfg.max_sessions = 4;
  cfg.ttl = std::chrono::milliseconds(10);
  scheduler::SessionHandleManager manager(cfg);

  auto lease = manager.AcquireLease("sess-c");
  REQUIRE(lease.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);

  scheduler::SessionHandleState state;
  state.model_id = "model-c";
  state.sequence_id = 3;
  state.sequence_generation = 9;
  state.block_table = {4, 5};
  manager.CommitLease("sess-c", state);

  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  auto expired = manager.CollectExpired();
  REQUIRE(expired.size() == 1);
  REQUIRE(expired[0].model_id == "model-c");
  REQUIRE(expired[0].sequence_id == 3);
  REQUIRE(expired[0].sequence_generation == 9);
}

TEST_CASE("SessionHandleManager returns cleanup state on capacity eviction",
          "[session_handles]") {
  scheduler::SessionHandleManager::Config cfg;
  cfg.max_sessions = 1;
  cfg.ttl = std::chrono::milliseconds(1000);
  scheduler::SessionHandleManager manager(cfg);

  auto lease = manager.AcquireLease("sess-d");
  REQUIRE(lease.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);

  scheduler::SessionHandleState state;
  state.model_id = "model-d";
  state.sequence_id = 9;
  state.sequence_generation = 4;
  state.block_table = {99};
  manager.CommitLease("sess-d", state);

  auto next = manager.AcquireLease("sess-e");
  REQUIRE(next.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);
  REQUIRE(next.cleanup_states.size() == 1);
  REQUIRE(next.cleanup_states[0].model_id == "model-d");
  REQUIRE(next.cleanup_states[0].sequence_id == 9);
  REQUIRE(next.cleanup_states[0].sequence_generation == 4);
}

TEST_CASE("SessionHandleManager drain all returns retained states",
          "[session_handles]") {
  scheduler::SessionHandleManager manager;

  auto lease = manager.AcquireLease("sess-f");
  REQUIRE(lease.status ==
          scheduler::SessionHandleManager::LeaseResult::Status::kAcquired);
  scheduler::SessionHandleState state;
  state.model_id = "model-f";
  state.sequence_id = 17;
  state.sequence_generation = 12;
  manager.CommitLease("sess-f", state);

  auto drained = manager.DrainAll();
  REQUIRE(drained.size() == 1);
  REQUIRE(drained[0].model_id == "model-f");
  REQUIRE(drained[0].sequence_generation == 12);
  REQUIRE(manager.SessionCount() == 0);
}
