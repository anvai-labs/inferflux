#include "runtime/backends/gpu/mps_device_strategy.h"
#include "server/logging/logger.h"

namespace inferflux {

bool MpsDeviceStrategy::Initialize() {
#ifdef __APPLE__
  if (initialized_) {
    return true;
  }

  info_.device_name = "Apple Metal GPU";
  info_.arch = "apple_metal";
  info_.total_memory_mb = 0; // Unknown without Metal API queries
  info_.device_id = 0;
  info_.supports_flash_attention = true;
  info_.flash_attention_version = "fa2";

  initialized_ = true;

  log::Info("mps_strategy", "Metal device initialized");
  return true;
#else
  log::Error("mps_strategy", "MPS not available on this platform");
  return false;
#endif
}

bool MpsDeviceStrategy::IsAvailable() const {
#ifdef __APPLE__
  return true;
#else
  return false;
#endif
}

GpuDeviceInfo MpsDeviceStrategy::GetDeviceInfo() const { return info_; }

void MpsDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  (void)config;
}

} // namespace inferflux
