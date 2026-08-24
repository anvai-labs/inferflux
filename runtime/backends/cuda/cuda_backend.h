#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

#include <filesystem>
#include <string>

namespace inferflux {

class CudaBackend : public GpuAcceleratedBackend {
public:
  CudaBackend();

  std::string Name() const override { return "llama_cpp_cuda"; }
};

} // namespace inferflux
