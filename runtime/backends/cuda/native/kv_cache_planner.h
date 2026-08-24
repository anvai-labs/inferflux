#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

namespace detail {
inline std::size_t SaturatingMul(std::size_t a, std::size_t b) {
  if (a != 0U && b > (std::numeric_limits<std::size_t>::max() / a)) {
    return std::numeric_limits<std::size_t>::max();
  }
  return a * b;
}
} // namespace detail

inline std::size_t
EstimateKvBytesPerTokenPerSequence(int num_layers, int num_kv_heads,
                                   int head_dim, std::size_t kv_element_bytes) {
  const std::size_t layers = static_cast<std::size_t>(std::max(0, num_layers));
  const std::size_t kv_heads =
      static_cast<std::size_t>(std::max(0, num_kv_heads));
  const std::size_t dim = static_cast<std::size_t>(std::max(0, head_dim));

  // 2x factor for K and V.
  std::size_t total =
      detail::SaturatingMul(layers, static_cast<std::size_t>(2));
  total = detail::SaturatingMul(total, kv_heads);
  total = detail::SaturatingMul(total, dim);
  total = detail::SaturatingMul(total, kv_element_bytes);
  return total;
}

inline std::size_t EstimateKvCacheBytes(int max_batch, int max_seq,
                                        std::size_t bytes_per_token_per_seq) {
  const std::size_t batch = static_cast<std::size_t>(std::max(0, max_batch));
  const std::size_t seq = static_cast<std::size_t>(std::max(0, max_seq));
  std::size_t total = detail::SaturatingMul(batch, seq);
  total = detail::SaturatingMul(total, bytes_per_token_per_seq);
  return total;
}

struct KvCachePlanInput {
  int requested_max_batch{32};
  int requested_max_seq{4096};
  int min_max_batch{32};
  int min_max_seq{128};
  int model_max_position_embeddings{0};
  std::size_t bytes_per_token_per_sequence{0};

  bool auto_tune_enabled{true};
  bool max_seq_overridden{false};

  // Explicit budget takes precedence when set (>0).
  std::size_t explicit_budget_bytes{0};

  // Fallback budget source when explicit budget is not provided.
  std::size_t free_bytes{0};
  double budget_ratio{0.30};
};

struct KvCachePlanOutput {
  int max_batch{0};
  int max_seq{0};
  std::size_t requested_bytes{0};
  std::size_t planned_bytes{0};
  std::size_t budget_bytes{0};
  bool auto_tuned_seq{false};
};

inline KvCachePlanOutput PlanKvCache(const KvCachePlanInput &input) {
  KvCachePlanOutput out;
  out.max_batch = std::max(input.requested_max_batch, input.min_max_batch);
  out.max_seq = std::max(input.requested_max_seq, input.min_max_seq);
  if (input.model_max_position_embeddings > 0) {
    out.max_seq = std::min(out.max_seq, input.model_max_position_embeddings);
  }

  out.requested_bytes = EstimateKvCacheBytes(
      out.max_batch, out.max_seq, input.bytes_per_token_per_sequence);

  if (input.explicit_budget_bytes > 0) {
    out.budget_bytes = input.explicit_budget_bytes;
  } else if (input.free_bytes > 0 && input.budget_ratio > 0.0 &&
             input.budget_ratio <= 1.0) {
    const long double scaled =
        static_cast<long double>(input.free_bytes) * input.budget_ratio;
    out.budget_bytes = static_cast<std::size_t>(scaled);
  }

  if (!input.auto_tune_enabled || input.max_seq_overridden ||
      input.bytes_per_token_per_sequence == 0 || out.budget_bytes == 0 ||
      out.requested_bytes <= out.budget_bytes) {
    out.planned_bytes = out.requested_bytes;
    return out;
  }

  const std::size_t denom =
      detail::SaturatingMul(static_cast<std::size_t>(out.max_batch),
                            input.bytes_per_token_per_sequence);
  if (denom == 0 || denom == std::numeric_limits<std::size_t>::max()) {
    out.planned_bytes = out.requested_bytes;
    return out;
  }

  std::size_t seq_by_budget = out.budget_bytes / denom;
  seq_by_budget = std::min<std::size_t>(
      seq_by_budget, static_cast<std::size_t>(std::numeric_limits<int>::max()));
  int tuned_seq = static_cast<int>(seq_by_budget);
  tuned_seq = std::max(tuned_seq, input.min_max_seq);
  if (input.model_max_position_embeddings > 0) {
    tuned_seq = std::min(tuned_seq, input.model_max_position_embeddings);
  }

  if (tuned_seq < out.max_seq) {
    out.max_seq = tuned_seq;
    out.auto_tuned_seq = true;
  }
  out.planned_bytes = EstimateKvCacheBytes(out.max_batch, out.max_seq,
                                           input.bytes_per_token_per_sequence);
  return out;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
