#include "runtime/backends/cuda/inferflux_cuda_runtime.h"

#include "runtime/backends/cuda/inferflux_cuda_executor.h"

namespace inferflux {

std::unique_ptr<InferfluxCudaRuntime> CreateInferfluxCudaRuntime() {
  return std::make_unique<InferfluxCudaExecutor>();
}

} // namespace inferflux
