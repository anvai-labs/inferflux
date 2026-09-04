#include "runtime/execution/parallel_context.h"
#include <catch2/catch_amalgamated.hpp>
#include <vector>

using namespace inferflux;

TEST_CASE("ParallelContext: initialization and rank tracking", "[parallel]") {
  auto &pc = ParallelContext::Get();

  // Test initialization as rank 0 of 2 (Master)
  pc.Initialize(0, 2, "stub");

  REQUIRE(pc.IsInitialized());
  REQUIRE(pc.Rank() == 0);
  REQUIRE(pc.WorldSize() == 2);
  REQUIRE(pc.IsMaster());
  REQUIRE(pc.Comm() != nullptr);
}

TEST_CASE("ParallelContext: batch synchronization stubs", "[parallel]") {
  auto &pc = ParallelContext::Get();
  // pc is already initialized from previous test case (singleton)

  SECTION("BroadcastBatch (Master)") {
    std::vector<int> ids = {1, 2};
    std::vector<int> phases = {1, 1};
    REQUIRE_NOTHROW(pc.BroadcastBatch(ids, phases));
  }

  SECTION("ReceiveBatch (Stub always returns same)") {
    std::vector<int> ids;
    std::vector<int> phases;
    // The stub implementation of ReceiveBatch uses Broadcast(0) which is a
    // no-op. It returns false if data is empty.
    REQUIRE_FALSE(pc.ReceiveBatch(ids, phases));
  }
}

TEST_CASE("ParallelContext: collective stubs", "[parallel]") {
  auto &pc = ParallelContext::Get();
  // comm_ only exists after Initialize(); under randomized test order this
  // case can run before any initializing case, so guarantee it here.
  // Initialize is idempotent (first call wins), so use the same topology
  // (rank 0, world 2) as "initialization and rank tracking" — whichever
  // case runs first initializes the singleton for both.
  pc.Initialize(0, 2);
  auto *comm = pc.Comm();
  REQUIRE(comm != nullptr);

  SECTION("AllGather stub") {
    std::vector<float> send = {1.0f, 2.0f};
    std::vector<float> recv;
    comm->AllGather(send, recv);
    // Stub AllGather just copies send to recv.
    REQUIRE(recv == send);
  }

  SECTION("Barrier stub") { REQUIRE_NOTHROW(comm->Barrier()); }
}

TEST_CASE("ParallelContext: P2P stubs", "[parallel]") {
  auto &pc = ParallelContext::Get();

  SECTION("Send/Recv activations") {
    std::vector<float> data = {0.1f, 0.2f};
    REQUIRE_NOTHROW(pc.SendActivations(data, 1));
    REQUIRE_NOTHROW(pc.RecvActivations(data, 0));
  }
}
