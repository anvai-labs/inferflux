#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

namespace inferflux {

class VulkanBackend : public GpuAcceleratedBackend {
public:
  VulkanBackend();

  std::string Name() const override { return "llama_cpp_vulkan"; }
};

} // namespace inferflux
