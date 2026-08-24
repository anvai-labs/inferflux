#pragma once

#include "runtime/backends/native/native_gpu_backend.h"

#include <memory>
#include <string>

namespace inferflux {

class NativeCudaDeviceStrategy;

// InferfluxCudaBackend is the CUDA-specific thin subclass of NativeGpuBackend.
// It provides:
//   - NativeCudaDeviceStrategy for device init
//   - CreateNativeRuntime() for CUDA runtime creation
//   - Static NativeKernelsReady() for CUDA readiness check
class InferfluxCudaBackend : public NativeGpuBackend {
public:
  InferfluxCudaBackend();
  ~InferfluxCudaBackend() override;

  std::string Name() const override { return "inferflux_cuda"; }

  std::unique_ptr<NativeInferenceRuntime> CreateNativeRuntime() override;

  static bool NativeKernelsReady();

protected:
  const char *LogTag() const override { return "inferflux_cuda_backend"; }

private:
  NativeCudaDeviceStrategy *native_strategy_{nullptr};
};

} // namespace inferflux
