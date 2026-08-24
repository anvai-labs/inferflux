#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

namespace inferflux {

class OpenClBackend : public GpuAcceleratedBackend {
public:
  OpenClBackend();

  std::string Name() const override { return "llama_cpp_opencl"; }
};

} // namespace inferflux
