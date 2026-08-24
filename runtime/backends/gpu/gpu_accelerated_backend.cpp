#include "runtime/backends/gpu/gpu_accelerated_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"

namespace inferflux {

GpuAcceleratedBackend::GpuAcceleratedBackend(
    std::unique_ptr<GpuDeviceStrategy> strategy)
    : strategy_(std::move(strategy)) {}

bool GpuAcceleratedBackend::InitializeDevice(const LlamaBackendConfig &config,
                                             LlamaBackendConfig *tuned_out) {
  if (!strategy_) {
    log::Error("gpu_backend", "No device strategy configured");
    return false;
  }

  if (!strategy_->IsAvailable()) {
    log::Error("gpu_backend", "Device strategy reports backend not available");
    return false;
  }

  if (!strategy_->Initialize()) {
    log::Error("gpu_backend", "Device initialization failed");
    return false;
  }

  device_info_ = strategy_->GetDeviceInfo();
  device_initialized_ = true;

  LlamaBackendConfig tuned =
      TuneLlamaBackendConfig(strategy_->Target(), config);
  strategy_->RecordMetrics(tuned);

  if (tuned_out) {
    *tuned_out = tuned;
  }

  log::Info("gpu_backend", "Device initialized: " + device_info_.device_name +
                               " (arch=" + device_info_.arch + ")");
  return true;
}

bool GpuAcceleratedBackend::LoadModel(const std::filesystem::path &model_path,
                                      const LlamaBackendConfig &config) {
  LlamaBackendConfig tuned;
  if (!InitializeDevice(config, &tuned)) {
    return false;
  }

  if (!LlamaCppBackend::LoadModel(model_path, tuned)) {
    log::Error("gpu_backend", "Failed to load model at " + model_path.string());
    return false;
  }

  log::Info("gpu_backend", "Model loaded on " + device_info_.device_name +
                               " (arch=" + device_info_.arch + ")");
  return true;
}

bool GpuAcceleratedBackend::IsReady() const {
  return device_initialized_ && LlamaCppBackend::IsReady();
}

std::string GpuAcceleratedBackend::Name() const {
  if (strategy_) {
    auto target = strategy_->Target();
    auto traits = DescribeLlamaBackendTarget(target);
    return "llama_cpp_" + traits.label;
  }
  return "gpu_unknown";
}

} // namespace inferflux
