#include <catch2/catch_amalgamated.hpp>

#include "server/auth/rate_limiter.h"

TEST_CASE("RateLimiter disabled when limit is 0", "[ratelimit]") {
  inferflux::RateLimiter limiter(0);
  REQUIRE(!limiter.Enabled());
  REQUIRE(limiter.CurrentLimit() == 0);
  // Always allowed when disabled.
  REQUIRE(limiter.Allow("any-key"));
}

TEST_CASE("RateLimiter allows up to limit", "[ratelimit]") {
  inferflux::RateLimiter limiter(3);
  REQUIRE(limiter.Enabled());
  REQUIRE(limiter.CurrentLimit() == 3);

  // First 3 requests should succeed (initial bucket = 3).
  REQUIRE(limiter.Allow("user1"));
  REQUIRE(limiter.Allow("user1"));
  REQUIRE(limiter.Allow("user1"));

  // 4th should be denied (bucket exhausted, no time passed).
  REQUIRE(!limiter.Allow("user1"));
}

TEST_CASE("RateLimiter per-key isolation", "[ratelimit]") {
  inferflux::RateLimiter limiter(1);

  REQUIRE(limiter.Allow("user-a"));
  REQUIRE(!limiter.Allow("user-a")); // exhausted for user-a

  // user-b has its own bucket.
  REQUIRE(limiter.Allow("user-b"));
}

TEST_CASE("RateLimiter UpdateLimit resets state", "[ratelimit]") {
  inferflux::RateLimiter limiter(2);
  REQUIRE(limiter.Allow("x"));
  REQUIRE(limiter.Allow("x"));
  REQUIRE(!limiter.Allow("x"));

  limiter.UpdateLimit(5);
  REQUIRE(limiter.CurrentLimit() == 5);
  // After reset, user gets a fresh bucket.
  REQUIRE(limiter.Allow("x"));
}

#include "server/auth/rate_limit_config.h"
#include <atomic>
#include <thread>
#include <vector>
using namespace inferflux;

TEST_CASE("Endpoint admission charges shared and class buckets atomically",
          "[ratelimit]") {
  auto now = RateLimiter::Clock::time_point{};
  RateLimiter limiter(
      4, {{RequestBucketPolicy{60, 1}}, {RequestBucketPolicy{120, 2}}},
      "policy_store", [&] { return now; });
  REQUIRE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  auto denied = limiter.Admit("a", RequestClass::kGeneration);
  REQUIRE_FALSE(denied.allowed);
  REQUIRE(denied.scope == RequestClass::kGeneration);
  REQUIRE(denied.retry_after_seconds == 1);
  REQUIRE(limiter.Admit("a", RequestClass::kEmbeddings).allowed);
  REQUIRE(limiter.Admit("a", RequestClass::kEmbeddings).allowed);
  REQUIRE_FALSE(limiter.Admit("a", RequestClass::kEmbeddings).allowed);
  REQUIRE(limiter.Allow("a")); // Denials did not consume shared capacity.
  denied = limiter.Admit("a", RequestClass::kGeneration);
  REQUIRE_FALSE(denied.allowed);
  REQUIRE(denied.scope == RequestClass::kOther);
  REQUIRE(denied.retry_after_seconds == 15); // Wait for ALL applicable buckets.
  REQUIRE(limiter.Admit("b", RequestClass::kGeneration).allowed);
  now += std::chrono::seconds(15);
  REQUIRE(limiter.Admit("a", RequestClass::kGeneration).allowed);
}

TEST_CASE("Endpoint rate state survives aggregate changes and rejects "
          "disabling ceiling",
          "[ratelimit]") {
  auto now = RateLimiter::Clock::time_point{};
  RateLimiter limiter(4, {{RequestBucketPolicy{60, 1}}, {}}, "yaml",
                      [&] { return now; });
  REQUIRE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  limiter.UpdateLimit(10);
  REQUIRE_FALSE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  REQUIRE(limiter.Snapshot().source == "admin_api");
  REQUIRE(limiter.Snapshot().endpoints.generation->burst == 1);
  REQUIRE_THROWS_AS(limiter.UpdateLimit(0), std::invalid_argument);
  REQUIRE(limiter.CurrentLimit() == 10);
  now += std::chrono::milliseconds(999);
  REQUIRE_FALSE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  now += std::chrono::milliseconds(1);
  REQUIRE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  now += std::chrono::seconds(121);
  REQUIRE(limiter.Admit("a", RequestClass::kGeneration).allowed);
  REQUIRE_FALSE(limiter.Admit("a", RequestClass::kGeneration).allowed);
}

TEST_CASE("Endpoint admission serializes concurrent callers", "[ratelimit]") {
  RateLimiter limiter(100, {{}, {RequestBucketPolicy{8, 3}}});
  std::atomic<int> allowed{0};
  std::vector<std::thread> callers;
  for (int i = 0; i < 16; ++i) {
    callers.emplace_back([&] {
      if (limiter.Admit("same", RequestClass::kEmbeddings).allowed)
        ++allowed;
    });
  }
  for (auto &caller : callers)
    caller.join();
  REQUIRE(allowed == 3);
}

TEST_CASE("Endpoint config is opt in and rejects ambiguous or invalid limits",
          "[ratelimit]") {
  REQUIRE_FALSE(
      ParseEndpointRateLimits(YAML::Node(YAML::NodeType::Undefined)).Enabled());
  auto parse = [](const std::string &text) {
    YAML::Node config;
    config["auth"]["endpoint_limits"] = YAML::Load(text);
    return ParseEndpointRateLimits(config);
  };
  auto limits = parse("generation: {requests_per_minute: 120, burst: "
                      "2}\nembeddings: {requests_per_minute: 480, burst: 8}");
  REQUIRE(limits.generation.has_value());
  REQUIRE(limits.generation->requests_per_minute == 120);
  REQUIRE(limits.embeddings->burst == 8);
  for (const auto *invalid :
       {"null", "[]", "{}", "other: 3", "generation: 3",
        "generation: {requests_per_minute: 60}",
        "generation: {requests_per_minute: 0, burst: 1}",
        "generation: {requests_per_minute: 60, burst: -1}",
        "generation: {requests_per_minute: 1, burst: 2}",
        "generation: {requests_per_minute: 1.5, burst: 1}",
        "generation: {requests_per_minute: 999999999999999999, burst: 1}",
        "generation: {requests_per_minute: 60, burst: 1, typo: 1}",
        "generation: {requests_per_minute: 60, burst: 1, burst: 2}",
        "generation: {requests_per_minute: 60, burst: 1}\ngeneration: "
        "{requests_per_minute: 60, burst: 2}"}) {
    INFO(invalid);
    REQUIRE_THROWS_AS(parse(invalid), std::invalid_argument);
  }
  REQUIRE_THROWS_AS(RateLimiter(0, limits), std::invalid_argument);
  limits.generation->burst = 0;
  REQUIRE_THROWS_AS(RateLimiter(600, limits), std::invalid_argument);
}
