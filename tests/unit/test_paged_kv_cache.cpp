#include "runtime/kv_cache/paged_kv_cache.h"
#include <catch2/catch_amalgamated.hpp>
#include <vector>

using namespace inferflux;

TEST_CASE("PagedKVCache: multi-block allocation", "[paged_kv]") {
  // 10 pages total, 1KB each.
  PagedKVCache cache(10, 1024, PagedKVCache::EvictionPolicy::kLRU);

  SECTION("Reserve multiple blocks") {
    auto blocks = cache.ReserveBlocks(3);
    REQUIRE(blocks.size() == 3);
    REQUIRE(cache.NumFreeBlocks() == 7);

    // Ensure blocks are unique.
    REQUIRE(blocks[0] != blocks[1]);
    REQUIRE(blocks[1] != blocks[2]);
    REQUIRE(blocks[0] != blocks[2]);
  }

  SECTION("Exhaustion throws") {
    REQUIRE_NOTHROW(cache.ReserveBlocks(10));
    REQUIRE(cache.NumFreeBlocks() == 0);
    REQUIRE_THROWS_AS(cache.ReserveBlocks(1), std::runtime_error);
  }

  SECTION("Release blocks") {
    auto blocks = cache.ReserveBlocks(5);
    REQUIRE(cache.NumFreeBlocks() == 5);
    cache.ReleaseBlocks(blocks);
    REQUIRE(cache.NumFreeBlocks() == 10);
  }
}

TEST_CASE("PagedKVCache: host swapping", "[paged_kv]") {
  // 4 pages primary, 8 pages host (default 2x).
  PagedKVCache cache(4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  SECTION("Swap out blocks to host") {
    auto blocks = cache.ReserveBlocks(2);
    std::vector<float> data(256, 1.23f); // 1KB
    cache.Write(blocks[0], data);
    cache.Write(blocks[1], data);

    auto host_handles = cache.SwapOut(blocks);
    REQUIRE(host_handles.size() == 2);
    REQUIRE(cache.NumFreeBlocks() == 4);          // Primary blocks freed
    REQUIRE(cache.NumAvailableHostBlocks() == 6); // 2 of 8 host blocks used

    // Swap in to new blocks.
    auto target_blocks = cache.ReserveBlocks(2);
    REQUIRE_NOTHROW(cache.SwapIn(host_handles, target_blocks));

    auto read_data = cache.Read(target_blocks[0]);
    REQUIRE(read_data.size() == 256);
    REQUIRE(read_data[0] == 1.23f);

    REQUIRE(cache.NumAvailableHostBlocks() == 8); // Host blocks freed
  }

  SECTION("SwapOut invalid blocks is safe") {
    auto handles = cache.SwapOut({99, -1});
    REQUIRE(handles.empty());
  }

  SECTION("SwapIn size mismatch throws") {
    REQUIRE_THROWS_AS(cache.SwapIn({1}, {1, 2}), std::invalid_argument);
  }
}

TEST_CASE("PagedKVCache: capacity reporting", "[paged_kv]") {
  PagedKVCache cache(5, 1024, PagedKVCache::EvictionPolicy::kLRU);
  REQUIRE(cache.NumFreeBlocks() == 5);
  REQUIRE(cache.NumAvailableHostBlocks() == 10);
}

TEST_CASE("PagedKVCache: usage snapshot reports block and byte ownership",
          "[paged_kv]") {
  PagedKVCache cache(5, 1024, PagedKVCache::EvictionPolicy::kLRU);

  auto snapshot = cache.GetUsageSnapshot();
  REQUIRE(snapshot.total_blocks == 5);
  REQUIRE(snapshot.used_blocks == 0);
  REQUIRE(snapshot.free_blocks == 5);
  REQUIRE(snapshot.page_size_bytes == 1024);
  REQUIRE(snapshot.total_bytes() == 5120);
  REQUIRE(snapshot.used_bytes() == 0);
  REQUIRE(snapshot.free_bytes() == 5120);

  auto blocks = cache.ReserveBlocks(2);
  snapshot = cache.GetUsageSnapshot();
  REQUIRE(snapshot.used_blocks == 2);
  REQUIRE(snapshot.free_blocks == 3);
  REQUIRE(snapshot.used_bytes() == 2048);
  REQUIRE(snapshot.free_bytes() == 3072);

  cache.ReleaseBlocks(blocks);
  snapshot = cache.GetUsageSnapshot();
  REQUIRE(snapshot.used_blocks == 0);
  REQUIRE(snapshot.free_blocks == 5);
}
