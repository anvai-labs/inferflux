#include "runtime/backends/backend_factory.h"

#include "runtime/backends/backend_registry.h"
#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/cuda/inferflux_cuda_backend.h"
#include "runtime/backends/gpu/native_cuda_device_strategy.h"
#include "runtime/backends/mps/mps_backend.h"
#include "runtime/backends/rocm/rocm_backend.h"
#include "runtime/backends/vulkan/vulkan_backend.h"
#include "server/logging/logger.h"

#if INFERFLUX_HAS_MLX
#include "runtime/backends/mlx/mlx_backend.h"
#endif

#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace inferflux {

namespace {

BackendExposurePolicy g_policy{};
std::mutex g_policy_mutex;

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

BackendExposurePolicy LoadPolicy() {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  return g_policy;
}

bool SupportsNativeBackend(LlamaBackendTarget target) {
  // Check if a native provider is registered in the BackendRegistry AND the
  // underlying hardware/kernels are available.  For CUDA this delegates to the
  // NativeCudaDeviceStrategy readiness probe.
  if (target == LlamaBackendTarget::kCuda) {
    return NativeCudaDeviceStrategy::NativeKernelsReady();
  }
  return false;
}

bool IsExplicitNativeHint(const std::string &hint) {
  return hint == "inferflux_cuda" || hint == "cuda_native" ||
         hint == "native_cuda";
}

bool IsExplicitLlamaCppHint(const std::string &hint) {
  return hint == "llama_cpp_cuda" || hint == "cuda_llama_cpp" ||
         hint == "cuda_llama";
}

std::shared_ptr<BackendInterface>
CreateNativeBackendForTarget(LlamaBackendTarget target) {
  auto &reg = BackendRegistry::Instance();
  if (reg.Has(target, BackendProvider::kNative)) {
    return reg.Create(target, BackendProvider::kNative);
  }
  return nullptr;
}

BackendFactoryResult CpuFallback(const std::string &reason) {
  if (!reason.empty()) {
    log::Warn("backend_factory", reason);
  }
  BackendFactoryResult out;
  out.target = LlamaBackendTarget::kCpu;
  out.traits = DescribeLlamaBackendTarget(out.target);
  out.capabilities = out.traits.capabilities;
  out.backend = std::make_shared<LlamaCppBackend>();
  out.provider = BackendProvider::kLlamaCpp;
  out.backend_label =
      BackendFactory::CanonicalBackendId(out.provider, out.target);
  out.config = TuneLlamaBackendConfig(out.target, {});
  return out;
}

BackendFactoryResult LlamaCppForTarget(LlamaBackendTarget target,
                                       const std::string &hint) {
  BackendFactoryResult out;
  out.target = target;
  out.traits = DescribeLlamaBackendTarget(target);
  out.capabilities = out.traits.capabilities;
  out.provider = BackendProvider::kLlamaCpp;
  out.backend_label = BackendFactory::CanonicalBackendId(out.provider, target);
  out.config = TuneLlamaBackendConfig(target, {});

  if (hint == "cuda") {
#ifdef INFERFLUX_HAS_CUDA
    out.backend = std::make_shared<CudaBackend>();
#else
    return CpuFallback(
        "CUDA backend requested but binary was built without CUDA support. "
        "Falling back to CPU backend.");
#endif
    return out;
  }

  if (hint == "mlx") {
#if INFERFLUX_HAS_MLX
    out.target = LlamaBackendTarget::kMps;
    out.traits = DescribeLlamaBackendTarget(out.target);
    out.capabilities = out.traits.capabilities;
    out.backend = std::make_shared<MlxBackend>();
    out.backend_label = "mlx";
    out.config = TuneLlamaBackendConfig(out.target, {});
    return out;
#else
    return CpuFallback(
        "MLX backend requested but binary was built without ENABLE_MLX. "
        "Falling back to CPU backend.");
#endif
  }

  if (hint == "rocm") {
#ifdef INFERFLUX_HAS_ROCM
    out.backend = std::make_shared<RocmBackend>();
#else
    return CpuFallback(
        "ROCm backend requested but binary was built without ROCm support. "
        "Falling back to CPU backend.");
#endif
    return out;
  }

  if (hint == "mps") {
    auto &reg = BackendRegistry::Instance();
    if (reg.Has(LlamaBackendTarget::kMps, BackendProvider::kLlamaCpp)) {
      out.backend =
          reg.Create(LlamaBackendTarget::kMps, BackendProvider::kLlamaCpp);
    } else {
      out.backend = std::make_shared<MpsBackend>();
    }
    return out;
  }

  if (hint == "vulkan") {
    auto &reg = BackendRegistry::Instance();
    if (reg.Has(LlamaBackendTarget::kVulkan, BackendProvider::kLlamaCpp)) {
      out.backend =
          reg.Create(LlamaBackendTarget::kVulkan, BackendProvider::kLlamaCpp);
    } else {
      out.backend = std::make_shared<VulkanBackend>();
    }
    return out;
  }

  if (hint == "cpu") {
    return CpuFallback("");
  }

  return CpuFallback("Unknown backend hint '" + hint +
                     "'. Falling back to CPU backend.");
}

BackendFactoryResult NativeUnavailableResult(LlamaBackendTarget target,
                                             const LlamaBackendTraits &traits,
                                             const std::string &reason) {
  BackendFactoryResult out;
  out.target = target;
  out.traits = traits;
  out.capabilities = traits.capabilities;
  out.provider = BackendProvider::kNative;
  out.backend_label = BackendFactory::CanonicalBackendId(out.provider, target);
  out.fallback_reason = reason;
  log::Error("backend_factory",
             "No backend exposed for '" + traits.label + "': " + reason);
  return out;
}

} // namespace

std::string BackendFactory::NormalizeHint(const std::string &backend_hint) {
  if (backend_hint.empty()) {
    return "cpu";
  }
  const std::string lowered = ToLower(backend_hint);
  if (lowered == "auto" || lowered == "mlx" || lowered == "cpu" ||
      lowered == "cuda" || lowered == "mps" || lowered == "rocm" ||
      lowered == "vulkan" || lowered == "opencl") {
    return lowered;
  }
  if (IsExplicitNativeHint(lowered)) {
    return CanonicalBackendId(BackendProvider::kNative,
                              LlamaBackendTarget::kCuda);
  }
  if (IsExplicitLlamaCppHint(lowered)) {
    return CanonicalBackendId(BackendProvider::kLlamaCpp,
                              LlamaBackendTarget::kCuda);
  }
  return DescribeLlamaBackendTarget(ParseLlamaBackendTarget(lowered)).label;
}

std::string BackendFactory::CanonicalBackendId(BackendProvider provider,
                                               LlamaBackendTarget target) {
  if (provider == BackendProvider::kNative) {
    switch (target) {
    case LlamaBackendTarget::kCuda:
      return "inferflux_cuda";
    case LlamaBackendTarget::kMps:
      return "inferflux_mps";
    case LlamaBackendTarget::kRocm:
      return "inferflux_rocm";
    case LlamaBackendTarget::kVulkan:
      return "inferflux_vulkan";
    case LlamaBackendTarget::kOpenCL:
      return "inferflux_opencl";
    case LlamaBackendTarget::kCpu:
    default:
      return "inferflux_cpu";
    }
  }

  switch (target) {
  case LlamaBackendTarget::kCuda:
    return "llama_cpp_cuda";
  case LlamaBackendTarget::kMps:
    return "llama_cpp_mps";
  case LlamaBackendTarget::kRocm:
    return "llama_cpp_rocm";
  case LlamaBackendTarget::kVulkan:
    return "llama_cpp_vulkan";
  case LlamaBackendTarget::kOpenCL:
    return "llama_cpp_opencl";
  case LlamaBackendTarget::kCpu:
  default:
    return "llama_cpp_cpu";
  }
}

std::string BackendFactory::ProviderLabel(BackendProvider provider) {
  switch (provider) {
  case BackendProvider::kNative:
    return "inferflux";
  case BackendProvider::kLlamaCpp:
  default:
    return "llama_cpp";
  }
}

BackendProvider
BackendFactory::ParseProviderLabel(const std::string &provider_label) {
  const std::string lowered = ToLower(provider_label);
  if (lowered == "inferflux") {
    return BackendProvider::kNative;
  }
  return BackendProvider::kLlamaCpp;
}

std::vector<std::string>
BackendFactory::NormalizeHintList(const std::vector<std::string> &backend_hints,
                                  const std::string &default_hint) {
  std::vector<std::string> out;
  out.reserve(backend_hints.size() + 1);
  std::unordered_set<std::string> seen;

  const auto append_unique = [&](const std::string &raw_hint,
                                 std::vector<std::string> *dest) {
    const std::string normalized = NormalizeHint(raw_hint);
    if (normalized.empty()) {
      return;
    }
    if (seen.insert(normalized).second) {
      dest->push_back(normalized);
    }
  };

  for (const auto &hint : backend_hints) {
    append_unique(hint, &out);
  }
  if (out.empty()) {
    append_unique(default_hint.empty() ? "cpu" : default_hint, &out);
  }
  return out;
}

BackendFactoryResult BackendFactory::Create(const std::string &backend_hint) {
  std::string hint = NormalizeHint(backend_hint);
  bool force_native = false;
  bool force_llama_cpp = false;
  if (IsExplicitNativeHint(hint)) {
    hint = "cuda";
    force_native = true;
  } else if (IsExplicitLlamaCppHint(hint)) {
    hint = "cuda";
    force_llama_cpp = true;
  }
  if (hint == "auto") {
#ifdef INFERFLUX_HAS_CUDA
    hint = "cuda";
#elif INFERFLUX_HAS_MLX
    hint = "mlx";
#else
    hint = "cpu";
#endif
  }

  const auto target = ParseLlamaBackendTarget(hint);
  const auto traits = DescribeLlamaBackendTarget(target);
  const auto policy = LoadPolicy();

  if (hint == "mlx") {
    // MLX has its own backend implementation today.
    return LlamaCppForTarget(LlamaBackendTarget::kMps, hint);
  }

  if (hint == "cpu") {
    return CpuFallback("");
  }

  if (hint == "cuda" || hint == "mps" || hint == "rocm" || hint == "vulkan") {
    if (force_native) {
      if (!SupportsNativeBackend(target)) {
        if (policy.strict_inferflux_request) {
          return NativeUnavailableResult(
              target, traits,
              "backend_policy_violation: strict_inferflux_request enabled; "
              "inferflux backend explicitly requested but InferFlux CUDA "
              "kernels are not "
              "ready");
        }
        if (policy.allow_llama_cpp_fallback) {
          auto out = LlamaCppForTarget(target, hint);
          out.used_fallback = true;
          out.fallback_reason =
              "inferflux backend explicitly requested but unavailable; exposed "
              "llama.cpp backend";
          log::Warn("backend_factory", out.fallback_reason);
          return out;
        }
        return NativeUnavailableResult(
            target, traits,
            "inferflux backend explicitly requested but unavailable; "
            "llama.cpp fallback disabled");
      }

      auto backend = CreateNativeBackendForTarget(target);
      if (backend) {
        BackendFactoryResult out;
        out.target = target;
        out.traits = traits;
        out.capabilities = traits.capabilities;
        out.backend_label =
            CanonicalBackendId(BackendProvider::kNative, target);
        out.provider = BackendProvider::kNative;
        out.backend = std::move(backend);
        out.config = TuneLlamaBackendConfig(target, {});
        out.require_strict_inferflux_execution =
            policy.strict_inferflux_request;
        return out;
      }
      return NativeUnavailableResult(
          target, traits,
          "inferflux backend explicitly requested but unavailable");
    }

    if (force_llama_cpp) {
      return LlamaCppForTarget(target, hint);
    }

    if (policy.prefer_inferflux && SupportsNativeBackend(target)) {
      // Native kernels are ready - create inferflux backend
      auto backend = CreateNativeBackendForTarget(target);
      if (backend) {
        BackendFactoryResult out;
        out.target = target;
        out.traits = traits;
        out.capabilities = traits.capabilities;
        out.backend_label =
            CanonicalBackendId(BackendProvider::kNative, target);
        out.provider = BackendProvider::kNative;
        out.backend = std::move(backend);
        out.config = TuneLlamaBackendConfig(target, {});
        return out;
      }
      if (policy.allow_llama_cpp_fallback) {
        auto out = LlamaCppForTarget(target, hint);
        out.used_fallback = true;
        out.fallback_reason =
            "inferflux backend reported ready but creation failed; exposed "
            "llama.cpp backend";
        log::Warn("backend_factory", out.fallback_reason);
        return out;
      }
      return NativeUnavailableResult(
          target, traits,
          "inferflux backend indicated as ready but creation failed; "
          "llama.cpp fallback disabled");
    }

    if (policy.prefer_inferflux && !policy.allow_llama_cpp_fallback) {
      return NativeUnavailableResult(
          target, traits,
          "inferflux backend requested but unavailable; llama.cpp fallback "
          "disabled");
    }

    BackendFactoryResult out;
    out = LlamaCppForTarget(target, hint);
    if (policy.prefer_inferflux && !SupportsNativeBackend(target)) {
      out.used_fallback = true;
      out.fallback_reason =
          "inferflux backend unavailable; exposed llama.cpp backend";
      log::Warn("backend_factory",
                "Using llama.cpp backend for '" + hint +
                    "' because inferflux backend is not available yet.");
    }
    return out;
  }

  return CpuFallback("Unknown backend hint '" + backend_hint +
                     "'. Falling back to CPU backend.");
}

void BackendFactory::SetExposurePolicy(const BackendExposurePolicy &policy) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  g_policy = policy;
}

BackendExposurePolicy BackendFactory::ExposurePolicy() { return LoadPolicy(); }

LlamaBackendConfig MergeBackendConfig(const LlamaBackendConfig &defaults,
                                      const BackendFactoryResult &selection) {
  return TuneLlamaBackendConfig(selection.target, defaults);
}

} // namespace inferflux
