#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/gpu/cuda_device_strategy.h"

namespace inferflux {

CudaBackend::CudaBackend()
    : GpuAcceleratedBackend(std::make_unique<CudaDeviceStrategy>()) {}

#ifdef INFERFLUX_HAS_CUDA
static const bool kCudaRegistered =
    (BackendRegistry::Instance().Register(
         LlamaBackendTarget::kCuda, BackendProvider::kLlamaCpp,
         [] { return std::make_shared<CudaBackend>(); }),
     true);
#endif

} // namespace inferflux
