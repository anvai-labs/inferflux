#include "runtime/backends/cuda/inferflux_cuda_backend.h"

#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/cuda/inferflux_cuda_runtime.h"
#include "runtime/backends/gpu/native_cuda_device_strategy.h"
#include "server/logging/logger.h"

namespace inferflux {

InferfluxCudaBackend::InferfluxCudaBackend()
    : NativeGpuBackend(std::make_unique<NativeCudaDeviceStrategy>()) {
  native_strategy_ = static_cast<NativeCudaDeviceStrategy *>(strategy_.get());
}

InferfluxCudaBackend::~InferfluxCudaBackend() = default;

std::unique_ptr<NativeInferenceRuntime>
InferfluxCudaBackend::CreateNativeRuntime() {
  if (!native_strategy_) {
    log::Error("inferflux_cuda_backend",
               "No native CUDA strategy available for runtime creation");
    return nullptr;
  }
  return native_strategy_->CreateRuntime();
}

bool InferfluxCudaBackend::NativeKernelsReady() {
  return NativeCudaDeviceStrategy::NativeKernelsReady();
}

#ifdef INFERFLUX_HAS_CUDA
static const bool kInferfluxCudaRegistered =
    (BackendRegistry::Instance().Register(
         LlamaBackendTarget::kCuda, BackendProvider::kNative,
         [] { return std::make_shared<InferfluxCudaBackend>(); }),
     true);
#endif

} // namespace inferflux
