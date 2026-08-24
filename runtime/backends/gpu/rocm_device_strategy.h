#pragma once

#include "runtime/backends/gpu/gpu_device_strategy.h"

namespace inferflux {

class RocmDeviceStrategy : public GpuDeviceStrategy {
public:
  bool Initialize() override;
  bool IsAvailable() const override;
  GpuDeviceInfo GetDeviceInfo() const override;
  LlamaBackendTarget Target() const override {
    return LlamaBackendTarget::kRocm;
  }
  void RecordMetrics(const LlamaBackendConfig &config) override;

private:
  GpuDeviceInfo info_;
  bool initialized_{false};

  static bool SupportsFlashAttentionForArch(const std::string &arch);
};

} // namespace inferflux
