#pragma once
#include <cstdint>
#include <string>
namespace inferflux {
// Residency is observed from loaded weight buffers, never inferred from a
// backend label or enumeration. It does not attest simultaneous GPU execution.
struct DevicePlacement {
  std::string requested;
  std::string effective;
  std::string vendor;
  std::string name;
  std::string stable_id; // PCI identity; empty means unavailable
  std::string state{"unverified"};
  uint64_t gpu_weight_bytes{0};
  uint64_t cpu_weight_bytes{0};
  int gpu_layer_count{0};
  int requested_gpu_layers{0};
  int context_size{0};
  int max_parallel_sequences{0};
  std::string kv_cache_type;
};
} // namespace inferflux
