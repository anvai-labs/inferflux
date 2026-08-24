#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

#include <memory>
#include <string>

namespace inferflux {

class RocmBackend : public GpuAcceleratedBackend {
public:
  RocmBackend();

  std::string GetBackendType() const { return "rocm"; }

  bool SupportsFlashAttention() const;
  std::string GetFlashAttentionVersion() const;
  std::string GetSelectedAttentionKernel() const;

  std::string Name() const override { return "llama_cpp_rocm"; }
};

} // namespace inferflux
