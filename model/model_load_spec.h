#pragma once

#include "runtime/backends/common/backend_config.h"
#include <optional>
#include <string>

namespace YAML {
class Node;
}

namespace inferflux {

// One load contract for startup YAML, watched YAML and admin JSON.
// Missing overrides inherit runtime defaults. Live specifications are
// immutable; changing one requires explicit removal after users release the
// backend.
struct ModelLoadSpec {
  std::string id;
  std::string path;
  std::string format{"auto"};
  std::string backend;
  bool make_default{false};
  std::optional<std::string> device; // cuda:N or rocm:N, vendor-local ordinal
  std::optional<int> context_size;
  std::optional<int> gpu_layers;
  std::optional<int> max_parallel_sequences;
  std::optional<std::string> kv_cache_type;

  bool HasOverrides() const;
  bool operator==(const ModelLoadSpec &other) const;
  bool operator!=(const ModelLoadSpec &other) const {
    return !(*this == other);
  }
  std::string Validate() const;
  LlamaBackendConfig Apply(const LlamaBackendConfig &defaults) const;
};

struct DeviceSelector {
  std::string vendor;
  int ordinal{0};
};
// Rejects bare ordinals, negative/overflow ordinals and unknown vendors.
std::optional<DeviceSelector> ParseDeviceSelector(const std::string &value);
ModelLoadSpec ParseModelLoadSpec(const YAML::Node &node);

} // namespace inferflux
