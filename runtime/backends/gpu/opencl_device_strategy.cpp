#include "runtime/backends/gpu/opencl_device_strategy.h"
#include "server/logging/logger.h"

namespace inferflux {

bool OpenClDeviceStrategy::Initialize() {
  log::Error("opencl_strategy", "OpenCL backend not yet implemented");
  return false;
}

bool OpenClDeviceStrategy::IsAvailable() const { return false; }

GpuDeviceInfo OpenClDeviceStrategy::GetDeviceInfo() const { return info_; }

void OpenClDeviceStrategy::RecordMetrics(const LlamaBackendConfig &config) {
  (void)config;
}

} // namespace inferflux
