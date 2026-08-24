#include "runtime/backends/mps/mps_backend.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/gpu/mps_device_strategy.h"

namespace inferflux {

MpsBackend::MpsBackend()
    : GpuAcceleratedBackend(std::make_unique<MpsDeviceStrategy>()) {}

#ifdef __APPLE__
static const bool kMpsRegistered =
    (BackendRegistry::Instance().Register(
         LlamaBackendTarget::kMps, BackendProvider::kLlamaCpp,
         [] { return std::make_shared<MpsBackend>(); }),
     true);
#endif

} // namespace inferflux
