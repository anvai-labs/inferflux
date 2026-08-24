#include "runtime/backends/gpu/vulkan_device_strategy.h"
#include "server/logging/logger.h"

namespace inferflux {

bool VulkanDeviceStrategy::Initialize() {
#ifdef GGML_USE_VULKAN
  if (initialized_) {
    return true;
  }

  info_.device_name = "Vulkan GPU";
  info_.arch = "vulkan";
  info_.total_memory_mb = 0;
  info_.device_id = 0;
  info_.supports_flash_attention = false;
  info_.flash_attention_version = "none";

  initialized_ = true;

  log::Info("vulkan_strategy", "Vulkan device initialized");
  return true;
#else
  log::Error("vulkan_strategy", "Vulkan support not compiled in");
  return false;
#endif
}

bool VulkanDeviceStrategy::IsAvailable() const {
#ifdef GGML_USE_VULKAN
  return true;
#else
  return false;
#endif
}

GpuDeviceInfo VulkanDeviceStrategy::GetDeviceInfo() const { return info_; }

void VulkanDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  (void)config;
}

} // namespace inferflux
