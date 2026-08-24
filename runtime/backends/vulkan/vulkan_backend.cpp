#include "runtime/backends/vulkan/vulkan_backend.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/gpu/vulkan_device_strategy.h"

namespace inferflux {

VulkanBackend::VulkanBackend()
    : GpuAcceleratedBackend(std::make_unique<VulkanDeviceStrategy>()) {}

#ifdef GGML_USE_VULKAN
static const bool kVulkanRegistered =
    (BackendRegistry::Instance().Register(
         LlamaBackendTarget::kVulkan, BackendProvider::kLlamaCpp,
         [] { return std::make_shared<VulkanBackend>(); }),
     true);
#endif

} // namespace inferflux
