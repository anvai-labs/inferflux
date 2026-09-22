#include "server/startup_advisor.h"
#ifdef INFERFLUX_HAS_ROCM
#include <hip/hip_runtime_api.h>
#endif

namespace inferflux {
AdvisorGpuInfo ProbeRocmGpu() {
  AdvisorGpuInfo info;
#ifdef INFERFLUX_HAS_ROCM
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return info;
  }
  info.available = true;
  info.device_count = device_count;

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
    info.device_name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  }

  std::size_t free_mem = 0, total_mem = 0;
  if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
    info.free_vram_bytes = static_cast<std::uint64_t>(free_mem);
  }

  // NEW: Calculate recommended reserve (15%)
  info.recommended_reserve_bytes =
      (info.total_vram_bytes / 100U) * 15U +
      ((info.total_vram_bytes % 100U) * 15U) / 100U;
  info.usable_vram_bytes =
      info.total_vram_bytes - info.recommended_reserve_bytes;
#endif
  return info;
}

} // namespace inferflux
