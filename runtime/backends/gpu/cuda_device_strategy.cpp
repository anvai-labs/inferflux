#include "runtime/backends/gpu/cuda_device_strategy.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace inferflux {

bool CudaDeviceStrategy::Initialize() {
#ifdef INFERFLUX_HAS_CUDA
  if (initialized_) {
    return true;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    log::Error("cuda_strategy", "No CUDA devices found");
    return false;
  }

  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    log::Error("cuda_strategy", "Failed to get CUDA device properties");
    return false;
  }

  info_.device_name = prop.name;
  info_.arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
  info_.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
  info_.device_id = 0;
  // SM 8.0+ supports FlashAttention-2
  info_.supports_flash_attention = (prop.major >= 8);
  info_.flash_attention_version =
      info_.supports_flash_attention ? "fa2" : "none";

  initialized_ = true;

  log::Info("cuda_strategy",
            "CUDA device: " + info_.device_name + " (Arch: " + info_.arch +
                ", Memory: " + std::to_string(info_.total_memory_mb) + " MB)");

  return true;
#else
  log::Error("cuda_strategy", "CUDA support not compiled in");
  return false;
#endif
}

bool CudaDeviceStrategy::IsAvailable() const {
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
#else
  return false;
#endif
}

GpuDeviceInfo CudaDeviceStrategy::GetDeviceInfo() const { return info_; }

void CudaDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  std::string attention_kernel = "standard";
  if (config.use_flash_attention) {
    if (config.cuda_attention_kernel == "auto") {
      attention_kernel = "fa2";
    } else if (config.cuda_attention_kernel == "fa2" ||
               config.cuda_attention_kernel == "fa3") {
      attention_kernel = config.cuda_attention_kernel;
    } else {
      attention_kernel = "fa2";
    }

    log::Info("cuda_strategy",
              "FlashAttention enabled (kernel=" + attention_kernel + ", tile=" +
                  std::to_string(config.flash_attention_tile) + ")");
  }

  GlobalMetrics().SetCudaAttentionKernel(attention_kernel);
  if (config.use_flash_attention) {
    GlobalMetrics().RecordFlashAttentionRequest(attention_kernel);
  }
  GlobalMetrics().SetFlashAttentionEnabled(config.use_flash_attention);

  log::Info("cuda_strategy",
            "CUDA model loaded (attention_kernel=" + attention_kernel + ")");
}

} // namespace inferflux
