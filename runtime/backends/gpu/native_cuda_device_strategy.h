#pragma once

#include "runtime/backends/gpu/cuda_device_strategy.h"

#include <memory>

namespace inferflux {

class NativeInferenceRuntime;

// NativeCudaDeviceStrategy extends CudaDeviceStrategy with InferFlux native
// runtime creation.  Used by InferfluxCudaBackend when it participates in the
// GpuAcceleratedBackend Strategy pattern.
class NativeCudaDeviceStrategy : public CudaDeviceStrategy {
public:
  bool Initialize() override;
  LlamaBackendTarget Target() const override {
    return LlamaBackendTarget::kCuda;
  }

  // Returns true when InferFlux CUDA kernels are compiled and CUDA runtime is
  // available.
  static bool NativeKernelsReady();

  // Create the InferFlux CUDA runtime after device init succeeds.
  std::unique_ptr<NativeInferenceRuntime> CreateRuntime();

private:
  bool native_ready_{false};
};

} // namespace inferflux
