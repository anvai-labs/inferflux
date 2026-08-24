#include "runtime/backends/gpu/native_cuda_device_strategy.h"

#include "runtime/backends/cuda/inferflux_cuda_runtime.h"
#include "server/logging/logger.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cctype>
#include <cstdlib>
#include <string>

namespace inferflux {

namespace {

bool ParseBoolValue(const char *raw) {
  if (!raw) {
    return false;
  }
  std::string lowered(raw);
  for (auto &ch : lowered) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

} // namespace

bool NativeCudaDeviceStrategy::Initialize() {
  if (!NativeKernelsReady()) {
    log::Error("native_cuda_strategy", "InferFlux CUDA kernels are not ready");
    return false;
  }

  if (!CudaDeviceStrategy::Initialize()) {
    return false;
  }

  native_ready_ = true;
  return true;
}

bool NativeCudaDeviceStrategy::NativeKernelsReady() {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
#ifdef INFERFLUX_HAS_CUDA
  if (ParseBoolValue(std::getenv("INFERFLUX_DISABLE_INFERFLUX_CUDA"))) {
    return false;
  }

  const auto log_probe_failure = [](const char *stage, cudaError_t err) {
    static bool logged = false;
    if (logged) {
      return;
    }
    logged = true;
    log::Warn("native_cuda_strategy",
              std::string("InferFlux CUDA readiness probe failed at ") + stage +
                  ": " + cudaGetErrorString(err));
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    return true;
  }

  const cudaError_t init_err = cudaFree(nullptr);
  if (init_err != cudaSuccess && init_err != cudaErrorCudartUnloading) {
    log_probe_failure("cudaFree(0)", init_err);
    return false;
  }

  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    log_probe_failure("cudaGetDeviceCount(retry)", err);
    return false;
  }
  return device_count > 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

std::unique_ptr<NativeInferenceRuntime>
NativeCudaDeviceStrategy::CreateRuntime() {
  if (!native_ready_) {
    log::Error("native_cuda_strategy",
               "Cannot create runtime before Initialize()");
    return nullptr;
  }
  return CreateInferfluxCudaRuntime();
}

} // namespace inferflux
