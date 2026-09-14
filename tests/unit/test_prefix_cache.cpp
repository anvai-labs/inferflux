#include <catch2/catch_amalgamated.hpp>

#include "runtime/prefix_cache/prefix_cache.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"

#include <string>
#include <thread>
#include <vector>

using namespace inferflux;

TEST_CASE("PrefixCache miss on empty cache", "[prefix_cache]") {
  PrefixCache cache(16);
  std::string out;
  int tok = 0;
  REQUIRE_FALSE(cache.Lookup({1, 2, 3}, &out, &tok));
}

TEST_CASE("PrefixCache hit after insert", "[prefix_cache]") {
  PrefixCache cache(16);
  cache.Insert({1, 2, 3}, "hello", 1);
  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(out == "hello");
}

TEST_CASE("RadixPrefixCache LRU eviction", "[prefix_cache]") {
  RadixPrefixCache cache(
      nullptr, [](int, std::shared_ptr<inferflux::BackendInterface>) {},
      RadixPrefixCacheLimits{4, 12});
  // Fill capacity.
  cache.Insert({1}, {10}, 1, nullptr);
  cache.Insert({2}, {11}, 2, nullptr);
  cache.Insert({3}, {12}, 3, nullptr);
  cache.Insert({4}, {13}, 4, nullptr);
  REQUIRE(cache.Size() == 4);

  // Use {1}.
  RadixLookupResult lookup;
  // Node hit on {1}.
  REQUIRE(cache.Lookup({1}, nullptr, &lookup));

  // Insert {5} — should evict {2} (LRU).
  cache.Insert({5}, {14}, 5, nullptr);
  REQUIRE(cache.Size() == 4);
  // {2} was pruned from trie.
  REQUIRE_FALSE(cache.Lookup({2}, nullptr, &lookup));
  REQUIRE(cache.Lookup({1}, nullptr, &lookup));
}

TEST_CASE("PrefixCache thread safety", "[prefix_cache]") {
  PrefixCache cache(64);
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back(
        [&cache, i]() { cache.Insert({i, i * 2}, std::to_string(i), 1); });
  }
  for (auto &t : threads) {
    t.join();
  }
  // All should fit within capacity=64; at least some must be present.
  int hits = 0;
  for (int i = 0; i < 8; ++i) {
    std::string out;
    int tok = 0;
    if (cache.Lookup({i, i * 2}, &out, &tok)) {
      ++hits;
    }
  }
  REQUIRE(hits > 0);
}
