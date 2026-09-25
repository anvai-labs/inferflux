#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace inferflux {

enum class RequestClass { kOther, kGeneration, kEmbeddings };
const char *RequestClassName(RequestClass request_class);

struct RequestBucketPolicy {
  int requests_per_minute{0};
  int burst{0};
};
struct EndpointRateLimits {
  std::optional<RequestBucketPolicy> generation;
  std::optional<RequestBucketPolicy> embeddings;
  bool Enabled() const { return generation || embeddings; }
};
void ValidateEndpointRateLimits(const EndpointRateLimits &limits);

struct RateLimitDecision {
  bool allowed{true};
  bool endpoint_policy{false};
  RequestClass scope{RequestClass::kOther}; // Other means the shared bucket.
  int requests_per_minute{0};
  int retry_after_seconds{0};
};
struct RateLimitSnapshot {
  int requests_per_minute;
  std::string source;
  EndpointRateLimits endpoints;
};

class RateLimiter {
public:
  using Clock = std::chrono::steady_clock;
  using Now = std::function<Clock::time_point()>;
  explicit RateLimiter(int tokens_per_minute, EndpointRateLimits endpoints = {},
                       std::string source = "constructor",
                       Now now = Clock::now);
  bool Allow(const std::string &key);
  RateLimitDecision Admit(const std::string &key, RequestClass request_class);
  bool Enabled() const;
  void UpdateLimit(int tokens_per_minute, std::string source = "admin_api");
  int CurrentLimit() const;
  RateLimitSnapshot Snapshot() const;

private:
  void EvictStaleLocked(Clock::time_point now);
  struct Entry {
    double tokens{0.0};
    Clock::time_point last;
  };
  struct Principal {
    std::array<Entry, 3> buckets;
    Clock::time_point last;
  };
  int tokens_per_minute_;
  EndpointRateLimits endpoints_;
  std::string source_;
  Now now_;
  std::unordered_map<std::string, Principal> entries_;
  mutable std::mutex mutex_;
  Clock::time_point last_eviction_;
};
} // namespace inferflux
