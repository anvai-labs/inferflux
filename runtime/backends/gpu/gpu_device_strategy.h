#pragma once

#include "runtime/backends/llama/llama_backend_traits.h"
#include "runtime/backends/llama/llama_cpp_backend.h"

#include <cstddef>
#include <string>

namespace inferflux {

struct GpuDeviceInfo {
  std::string device_name;
  std::string arch; // "sm_89", "GFX1201", "apple_m2", "vulkan_1.3"
  size_t total_memory_mb{0};
  int device_id{0};
  bool supports_flash_attention{false};
  std::string flash_attention_version; // "fa2", "fa3", "none"
};

class GpuDeviceStrategy {
public:
  virtual ~GpuDeviceStrategy() = default;
  virtual bool Initialize() = 0;
  virtual bool IsAvailable() const = 0;
  virtual GpuDeviceInfo GetDeviceInfo() const = 0;
  virtual LlamaBackendTarget Target() const = 0;
  virtual void RecordMetrics(const LlamaBackendConfig &config) = 0;
};

} // namespace inferflux
