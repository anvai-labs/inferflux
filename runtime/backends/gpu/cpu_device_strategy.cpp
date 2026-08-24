#include "runtime/backends/gpu/cpu_device_strategy.h"

#include <thread>

namespace inferflux {

bool CpuDeviceStrategy::Initialize() {
  info_.device_name = "CPU";
  info_.arch = "x86_64";
  info_.total_memory_mb = 0;
  info_.device_id = 0;
  info_.supports_flash_attention = false;
  info_.flash_attention_version = "none";
  return true;
}

bool CpuDeviceStrategy::IsAvailable() const { return true; }

GpuDeviceInfo CpuDeviceStrategy::GetDeviceInfo() const { return info_; }

void CpuDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  (void)config;
}

} // namespace inferflux
