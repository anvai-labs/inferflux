#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/kv_cache/paged_kv_cache.h"

#include <catch2/catch_amalgamated.hpp>

#include <atomic>
#include <thread>
#include <vector>

using namespace inferflux;

TEST_CASE("RadixPrefixCache: miss on empty cache", "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, RadixPrefixCacheLimits{16, 12});
  RadixLookupResult lookup;
  lookup.matched_tokens = -1;
  REQUIRE_FALSE(cache.Lookup({1, 2, 3}, nullptr, &lookup));
  REQUIRE(lookup.matched_tokens == 0);
}

TEST_CASE("RadixPrefixCache: exact hit after insert", "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, RadixPrefixCacheLimits{16, 12});
  // Assume [1, 2, 3] covered by block 100, computed in seq 5.
  cache.Insert({1, 2, 3}, {100}, 5, nullptr);
  RadixLookupResult lookup;
  bool hit = cache.Lookup({1, 2, 3}, nullptr, &lookup);
  UNSCOPED_INFO("hit=" << hit << " matched=" << lookup.matched_tokens
                       << " out_seq=" << lookup.sequence_id);
  REQUIRE(hit);
  REQUIRE(lookup.block_table == std::vector<int>{100});
  REQUIRE(lookup.sequence_id == 5);
  REQUIRE(lookup.matched_tokens == 3);
}

TEST_CASE("RadixPrefixCache: partial prefix match reports matched_tokens",
          "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, RadixPrefixCacheLimits{16, 12});
  cache.Insert({1, 2, 3}, {100}, 5, nullptr);

  RadixLookupResult lookup;
  lookup.matched_tokens = -1;
  // Query [1, 2, 4] — [1, 2] shares a prefix but divergences mid-edge [1, 2,
  // 3]. Correctness Fix (§ Item 3): matched_tokens should be 2, but hit is
  // false (no full node).
  bool hit = cache.Lookup({1, 2, 4}, nullptr, &lookup);
  REQUIRE_FALSE(hit);
  REQUIRE(lookup.matched_tokens == 2);
}

TEST_CASE("RadixPrefixCache: deep radix tree with progressive prefixes",
          "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, RadixPrefixCacheLimits{32, 12});
  // Blocks of 16 tokens each.
  cache.Insert({1, 2}, {10}, 1, nullptr);
  cache.Insert({1, 2, 3, 4}, {10, 11}, 2, nullptr);

  RadixLookupResult lookup;
  bool hit1 = cache.Lookup({1, 2}, nullptr, &lookup);
  UNSCOPED_INFO("hit1=" << hit1 << " matched=" << lookup.matched_tokens
                        << " out_seq=" << lookup.sequence_id);
  REQUIRE(hit1);
  REQUIRE(lookup.matched_tokens == 2);
  REQUIRE(lookup.sequence_id == 1);

  bool hit2 = cache.Lookup({1, 2, 3, 4}, nullptr, &lookup);
  UNSCOPED_INFO("hit2=" << hit2 << " matched=" << lookup.matched_tokens
                        << " out_seq=" << lookup.sequence_id);
  REQUIRE(hit2);
  REQUIRE(lookup.matched_tokens == 4);
  REQUIRE(lookup.sequence_id == 2);
}

TEST_CASE(
    "RadixPrefixCache: reinserting existing node keeps suffix blocks only",
    "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, RadixPrefixCacheLimits{64, 12});

  std::vector<int> prefix_tokens;
  std::vector<int> full_tokens;
  for (int i = 1; i <= 16; ++i) {
    prefix_tokens.push_back(i);
    full_tokens.push_back(i);
  }
  for (int i = 17; i <= 32; ++i) {
    full_tokens.push_back(i);
  }

  cache.Insert(prefix_tokens, {100}, 1, nullptr);
  cache.Insert(full_tokens, {100, 200}, 2, nullptr);
  cache.Insert(full_tokens, {100, 201}, 3, nullptr);

  RadixLookupResult lookup;
  bool hit = cache.Lookup(full_tokens, nullptr, &lookup);

  REQUIRE(hit);
  REQUIRE(lookup.matched_tokens == 32);
  REQUIRE(lookup.block_table == std::vector<int>{100, 201});
  REQUIRE(lookup.sequence_id == 3);
}

TEST_CASE("RadixPrefixCache: sequence slot capping and eviction",
          "[radix_cache]") {
  int evicted_seq = -1;
  RadixPrefixCache cache(
      nullptr, [&](int seq) { evicted_seq = seq; },
      RadixPrefixCacheLimits{100, 2});

  cache.Insert({1}, {10}, 101, nullptr);
  cache.Insert({2}, {11}, 102, nullptr);
  REQUIRE(cache.LiveSequences() == 2);

  // Inserting 3rd sequence should trigger eviction of the oldest (101).
  cache.Insert({3}, {12}, 103, nullptr);
  REQUIRE(cache.LiveSequences() == 2);
  REQUIRE(evicted_seq == 101);
}

TEST_CASE("RadixPrefixCache: memory snapshot reports unique retained blocks",
          "[radix_cache]") {
  auto paged_kv = std::make_shared<PagedKVCache>(
      8, 1024, PagedKVCache::EvictionPolicy::kLRU);
  RadixPrefixCache cache(paged_kv, [](int) {}, RadixPrefixCacheLimits{64, 12});

  cache.Insert({1, 2, 3}, {10}, 1001, nullptr);
  cache.Insert(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
      {10, 20}, 1002, nullptr);

  const auto snapshot = cache.MemorySnapshot();
  REQUIRE(snapshot.unique_retained_blocks == 2);
  REQUIRE(snapshot.retained_bytes == 2048);
  REQUIRE(snapshot.live_sequences == 2);
}
