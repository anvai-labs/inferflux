#include "runtime/backends/gpu/gpu_accelerated_backend.h"
#include "model/model_load_spec.h"
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

  int ordinal = 0;
  if (!config.device.empty()) {
    const auto selector = ParseDeviceSelector(config.device);
    const auto target = strategy_->Target();
    if (!selector ||
        (selector->vendor == "cuda" && target != LlamaBackendTarget::kCuda) ||
        (selector->vendor == "rocm" && target != LlamaBackendTarget::kRocm)) {
      log::Error("gpu_backend",
                 "Device selector does not match backend vendor");
      return false;
    }
    ordinal = selector->ordinal;
  }
  if (!strategy_->Initialize(ordinal)) {
    log::Error("gpu_backend", "Device initialization failed");
    return false;
  }

  device_info_ = strategy_->GetDeviceInfo();
  device_initialized_ = true;

  LlamaBackendConfig tuned =
      TuneLlamaBackendConfig(strategy_->Target(), config);
  if (tuned.device.empty() && tuned.gpu_layers != 0) {
    if (strategy_->Target() == LlamaBackendTarget::kCuda)
      tuned.device = "cuda:" + std::to_string(ordinal);
    if (strategy_->Target() == LlamaBackendTarget::kRocm)
      tuned.device = "rocm:" + std::to_string(ordinal);
  }
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
  device_error_.clear();
  LlamaBackendConfig tuned;
  if (!InitializeDevice(config, &tuned)) {
    device_error_ =
        "placement_unavailable: cannot initialize " + Name() +
        (config.device.empty() ? " device 0" : " device " + config.device);
    return false;
  }

  if (!LlamaCppBackend::LoadModel(model_path, tuned)) {
    log::Error("gpu_backend", "Failed to load model at " + model_path.string());
    return false;
  }

  log::Info("gpu_backend", "Model loaded with device strategy " +
                               device_info_.device_name +
                               " (arch=" + device_info_.arch + ")");
  return true;
}

std::string GpuAcceleratedBackend::LoadError() const {
  return device_error_.empty() ? LlamaCppBackend::LoadError() : device_error_;
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
