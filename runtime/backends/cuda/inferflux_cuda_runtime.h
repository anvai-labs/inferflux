#pragma once

// InferfluxCudaRuntime is now a typedef for NativeInferenceRuntime.
// This header is retained for backward compatibility — all existing code that
// includes it continues to compile.  New code should prefer the device-neutral
// NativeInferenceRuntime interface directly.

#include "runtime/backends/native/native_inference_runtime.h"

namespace inferflux {

using InferfluxCudaRuntime = NativeInferenceRuntime;

std::unique_ptr<InferfluxCudaRuntime> CreateInferfluxCudaRuntime();

} // namespace inferflux
