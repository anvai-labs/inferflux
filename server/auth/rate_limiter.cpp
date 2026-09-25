#include "server/auth/rate_limiter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace inferflux {

const char *RequestClassName(RequestClass request_class) {
  switch (request_class) {
  case RequestClass::kGeneration:
    return "generation";
  case RequestClass::kEmbeddings:
    return "embeddings";
  default:
    return "aggregate";
  }
}

void ValidateEndpointRateLimits(const EndpointRateLimits &limits) {
  for (const auto &policy : {limits.generation, limits.embeddings}) {
    if (policy && (policy->requests_per_minute <= 0 || policy->burst <= 0 ||
                   policy->burst > policy->requests_per_minute)) {
      throw std::invalid_argument(
          "endpoint rate limits require 0 < burst <= requests_per_minute");
    }
  }
}

RateLimiter::RateLimiter(int tokens_per_minute, EndpointRateLimits endpoints,
                         std::string source, Now now)
    : tokens_per_minute_(tokens_per_minute), endpoints_(std::move(endpoints)),
      source_(std::move(source)), now_(std::move(now)) {
  ValidateEndpointRateLimits(endpoints_);
  if (endpoints_.Enabled() && tokens_per_minute <= 0) {
    throw std::invalid_argument(
        "endpoint rate limits require a positive aggregate request limit");
  }
}

bool RateLimiter::Allow(const std::string &key) {
  return Admit(key, RequestClass::kOther).allowed;
}

RateLimitDecision RateLimiter::Admit(const std::string &key,
                                     RequestClass request_class) {
  // Keep policy reads, refill, and both debits in one critical section. A
  // rejected endpoint request must not consume the shared request allowance.
  std::lock_guard<std::mutex> lock(mutex_);
  RateLimitDecision decision;
  decision.endpoint_policy = endpoints_.Enabled();
  if (tokens_per_minute_ <= 0) {
    return decision;
  }
  const auto now = now_();
  EvictStaleLocked(now);
  const std::array<std::optional<RequestBucketPolicy>, 3> policies = {
      RequestBucketPolicy{tokens_per_minute_, tokens_per_minute_},
      endpoints_.generation, endpoints_.embeddings};
  auto [it, inserted] = entries_.try_emplace(key);
  auto &principal = it->second;
  principal.last = now;
  if (inserted) {
    for (std::size_t i = 0; i < policies.size(); ++i) {
      principal.buckets[i] = {
          policies[i] ? static_cast<double>(policies[i]->burst) : 0.0, now};
    }
  }
  const auto endpoint = static_cast<std::size_t>(request_class);
  auto applies = [&](std::size_t i) {
    return policies[i] && (i == 0 || i == endpoint);
  };
  for (std::size_t i = 0; i < policies.size(); ++i) {
    if (!applies(i)) {
      continue;
    }
    const auto &policy = *policies[i];
    auto &bucket = principal.buckets[i];
    const double elapsed =
        std::chrono::duration<double>(now - bucket.last).count();
    const double refill = policy.requests_per_minute / 60.0;
    bucket.tokens =
        std::min<double>(policy.burst, bucket.tokens + elapsed * refill);
    bucket.last = now;
    if (bucket.tokens < 1.0) {
      const int retry =
          static_cast<int>(std::ceil((1.0 - bucket.tokens) / refill));
      if (decision.allowed || retry > decision.retry_after_seconds) {
        decision.scope = static_cast<RequestClass>(i);
        decision.requests_per_minute = policy.requests_per_minute;
        decision.retry_after_seconds = retry;
      }
      decision.allowed = false;
    }
  }
  if (decision.allowed) {
    for (std::size_t i = 0; i < policies.size(); ++i) {
      if (applies(i)) {
        principal.buckets[i].tokens -= 1.0;
      }
    }
  }
  return decision;
}

void RateLimiter::EvictStaleLocked(Clock::time_point now) {
  constexpr auto kEvictInterval = std::chrono::seconds(60);
  if (now - last_eviction_ < kEvictInterval) {
    return;
  }
  last_eviction_ = now;
  // Every configured burst refills within one minute. After two idle minutes,
  // removing the principal is equivalent to retaining its full buckets.
  const auto ttl = std::chrono::seconds(120);
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (now - it->second.last > ttl) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void RateLimiter::UpdateLimit(int tokens_per_minute, std::string source) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (endpoints_.Enabled() && tokens_per_minute <= 0) {
    throw std::invalid_argument(
        "endpoint rate limits require a positive aggregate request limit");
  }
  tokens_per_minute_ = tokens_per_minute;
  source_ = std::move(source);
  if (!endpoints_.Enabled()) {
    entries_.clear(); // Preserve the existing unconfigured reset behavior.
  } else {
    const auto now = now_();
    for (auto &[key, principal] : entries_) {
      principal.buckets[0] = {static_cast<double>(tokens_per_minute), now};
    }
  }
}

bool RateLimiter::Enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tokens_per_minute_ > 0;
}

int RateLimiter::CurrentLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tokens_per_minute_;
}

RateLimitSnapshot RateLimiter::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {tokens_per_minute_, source_, endpoints_};
}
} // namespace inferflux
