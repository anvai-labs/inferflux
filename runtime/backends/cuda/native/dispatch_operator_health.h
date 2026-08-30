#pragma once

// Process-lifetime operator health for native CUDA dispatch. When the
// load-time reachability probe (or repeated runtime divergence) proves an
// operator cannot execute, it is marked unhealthy and the dispatch rules
// skip it — the next-best rule wins — instead of silently degrading
// throughput. The state is deliberately process-lifetime: a down-ranked
// operator recovers only on restart, loudly (log + metric + /readyz).

#include "runtime/backends/cuda/native/dispatch_catalog.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace inferflux {

class InferfluxCudaOperatorHealth {
public:
  static InferfluxCudaOperatorHealth &Instance();

  void MarkUnhealthy(FfnProjOperator op, const char *reason);
  void MarkUnhealthy(DownProjOperator op, const char *reason);

  bool IsUnhealthy(FfnProjOperator op) const {
    return ffn_mask_.load(std::memory_order_relaxed) &
           (uint16_t{1} << static_cast<unsigned>(op));
  }
  bool IsUnhealthy(DownProjOperator op) const {
    return down_mask_.load(std::memory_order_relaxed) &
           (uint16_t{1} << static_cast<unsigned>(op));
  }

  bool AnyUnhealthy() const {
    return ffn_mask_.load(std::memory_order_relaxed) != 0 ||
           down_mask_.load(std::memory_order_relaxed) != 0;
  }

  // "ffn:q8_1_group_mmq3(reason),down:mmq(reason)" for logs, /readyz, and
  // the startup advisor.
  std::string Describe() const;

  // Test/A-B hook: CSV like "ffn:q8_1_group_mmq3,down:mmq" parsed from
  // INFERFLUX_CUDA_DISPATCH_PROBE_FORCE_UNHEALTHY. Unknown entries are
  // ignored (logged once by the caller).
  void ApplyForceList(const char *csv);

private:
  InferfluxCudaOperatorHealth() = default;

  std::atomic<uint16_t> ffn_mask_{0};
  std::atomic<uint16_t> down_mask_{0};
  mutable std::mutex describe_mutex_;
  std::vector<std::pair<std::string, std::string>> marked_;
};

} // namespace inferflux
