#include "runtime/backends/opencl/opencl_backend.h"
#include "runtime/backends/gpu/opencl_device_strategy.h"

namespace inferflux {

OpenClBackend::OpenClBackend()
    : GpuAcceleratedBackend(std::make_unique<OpenClDeviceStrategy>()) {}

// OpenCL is a future extension point. Not registered in BackendRegistry
// because llama.cpp does not support OpenCL natively.

} // namespace inferflux
