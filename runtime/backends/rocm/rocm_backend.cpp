#include "runtime/backends/rocm/rocm_backend.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/gpu/rocm_device_strategy.h"

namespace inferflux {

RocmBackend::RocmBackend()
    : GpuAcceleratedBackend(std::make_unique<RocmDeviceStrategy>()) {}

bool RocmBackend::SupportsFlashAttention() const {
  return DeviceInfo().supports_flash_attention;
}

std::string RocmBackend::GetFlashAttentionVersion() const {
  return DeviceInfo().flash_attention_version;
}

std::string RocmBackend::GetSelectedAttentionKernel() const {
  if (DeviceInfo().supports_flash_attention && FlashAttentionEnabled()) {
    return "fa2";
  }
  return "standard";
}

#ifdef INFERFLUX_HAS_ROCM
static const bool kRocmRegistered =
    (BackendRegistry::Instance().Register(
         LlamaBackendTarget::kRocm, BackendProvider::kLlamaCpp,
         [] { return std::make_shared<RocmBackend>(); }),
     true);
#endif

} // namespace inferflux
