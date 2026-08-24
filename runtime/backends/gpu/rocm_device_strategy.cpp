#include "runtime/backends/gpu/rocm_device_strategy.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <algorithm>
#include <cctype>
#include <string>

#ifdef INFERFLUX_HAS_ROCM
#include <hip/hip_runtime_api.h>
#endif

namespace inferflux {

bool RocmDeviceStrategy::SupportsFlashAttentionForArch(
    const std::string &arch) {
  if (arch.find("GFX9") != std::string::npos) {
    return true; // MI200/MI250X/MI300X
  }
  if (arch.find("GFX10") != std::string::npos) {
    return true; // RDNA 2
  }
  if (arch.find("GFX11") != std::string::npos) {
    return true; // RDNA 3
  }
  if (arch.find("GFX12") != std::string::npos) {
    return true; // RDNA 4
  }
  return false;
}

bool RocmDeviceStrategy::Initialize() {
#ifdef INFERFLUX_HAS_ROCM
  if (initialized_) {
    return true;
  }

  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    log::Error("rocm_strategy", "No HIP devices found");
    return false;
  }

  err = hipSetDevice(0);
  if (err != hipSuccess) {
    log::Error("rocm_strategy", "Failed to set HIP device: " +
                                    std::to_string(static_cast<int>(err)));
    return false;
  }

  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    log::Error("rocm_strategy", "Failed to get HIP device properties");
    return false;
  }

  // Get arch name and convert to uppercase
  std::string arch(prop.gcnArchName);
  std::transform(arch.begin(), arch.end(), arch.begin(), ::toupper);

  info_.device_name = prop.name;
  info_.arch = arch;
  info_.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
  info_.device_id = 0;
  info_.supports_flash_attention = SupportsFlashAttentionForArch(arch);
  info_.flash_attention_version =
      info_.supports_flash_attention ? "fa2" : "none";

  initialized_ = true;

  log::Info("rocm_strategy",
            "HIP device: " + info_.device_name + " (Arch: " + info_.arch +
                ", Memory: " + std::to_string(info_.total_memory_mb) + " MB)");

  GlobalMetrics().RecordRocmDeviceProperties(info_.device_id, info_.arch);

  return true;
#else
  log::Error("rocm_strategy", "ROCm support not compiled in");
  return false;
#endif
}

bool RocmDeviceStrategy::IsAvailable() const {
#ifdef INFERFLUX_HAS_ROCM
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  return (err == hipSuccess && device_count > 0);
#else
  return false;
#endif
}

GpuDeviceInfo RocmDeviceStrategy::GetDeviceInfo() const { return info_; }

void RocmDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  if (!config.use_flash_attention) {
    return;
  }

  if (!info_.supports_flash_attention) {
    log::Warn("rocm_strategy",
              "FlashAttention requested but not supported on " + info_.arch +
                  ", falling back to standard attention");
    GlobalMetrics().RecordCudaAttentionKernelFallback("auto", "standard",
                                                      "rocm_arch_unsupported");
    return;
  }

  log::Info("rocm_strategy",
            "FlashAttention enabled (kernel=fa2, arch=" + info_.arch + ")");
  GlobalMetrics().SetCudaAttentionKernel("fa2");
  GlobalMetrics().RecordFlashAttentionRequest("fa2");
  GlobalMetrics().SetFlashAttentionEnabled(true);
}

} // namespace inferflux
