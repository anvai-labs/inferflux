#pragma once

#include "runtime/backends/gpu/gpu_device_strategy.h"

namespace inferflux {

class OpenClDeviceStrategy : public GpuDeviceStrategy {
public:
  bool Initialize() override;
  bool IsAvailable() const override;
  GpuDeviceInfo GetDeviceInfo() const override;
  LlamaBackendTarget Target() const override {
    return LlamaBackendTarget::kOpenCL;
  }
  void RecordMetrics(const LlamaBackendConfig &config) override;

private:
  GpuDeviceInfo info_;
};

} // namespace inferflux
