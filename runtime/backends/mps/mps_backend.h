#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

namespace inferflux {

class MpsBackend : public GpuAcceleratedBackend {
public:
  MpsBackend();

  std::string Name() const override { return "llama_cpp_mps"; }
};

} // namespace inferflux
