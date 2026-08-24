#include "runtime/backends/llama/llama_backend_traits.h"

#include <algorithm>
#include <cctype>

namespace inferflux {

namespace {

std::string NormalizeHint(const std::string &hint) {
  std::string lowered = hint;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lowered;
}

BackendCapabilities BuildCapabilitiesForTarget(LlamaBackendTarget target) {
  BackendCapabilities caps;
  (void)target;

#if defined(INFERFLUX_HAS_MTMD) && INFERFLUX_HAS_MTMD
  caps.supports_vision = true;
#else
  caps.supports_vision = false;
#endif
  return caps;
}

std::string NormalizeCudaAttentionKernel(const std::string &value) {
  std::string lowered = NormalizeHint(value);
  if (lowered == "fa3" || lowered == "fa2" || lowered == "standard") {
    return lowered;
  }
  return "auto";
}

} // namespace

LlamaBackendTarget ParseLlamaBackendTarget(const std::string &hint) {
  const std::string lowered = NormalizeHint(hint);
  if (lowered == "cuda" || lowered == "inferflux_cuda" ||
      lowered == "llama_cpp_cuda" || lowered == "cuda_native" ||
      lowered == "native_cuda" || lowered == "cuda_llama_cpp" ||
      lowered == "cuda_llama") {
    return LlamaBackendTarget::kCuda;
  }
  if (lowered == "mps") {
    return LlamaBackendTarget::kMps;
  }
  if (lowered == "rocm") {
    return LlamaBackendTarget::kRocm;
  }
  if (lowered == "vulkan") {
    return LlamaBackendTarget::kVulkan;
  }
  if (lowered == "opencl") {
    return LlamaBackendTarget::kOpenCL;
  }
  return LlamaBackendTarget::kCpu;
}

LlamaBackendTraits DescribeLlamaBackendTarget(LlamaBackendTarget target) {
  LlamaBackendTraits traits;
  traits.capabilities = BuildCapabilitiesForTarget(target);

  switch (target) {
  case LlamaBackendTarget::kCuda:
    traits.label = "cuda";
    traits.gpu_accelerated = true;
    traits.supports_flash_attention = true;
    return traits;
  case LlamaBackendTarget::kMps:
    traits.label = "mps";
    traits.gpu_accelerated = true;
    traits.supports_flash_attention = true;
    return traits;
  case LlamaBackendTarget::kRocm:
    traits.label = "rocm";
    traits.gpu_accelerated = true;
    traits.supports_flash_attention = true;
    return traits;
  case LlamaBackendTarget::kVulkan:
    traits.label = "vulkan";
    traits.gpu_accelerated = true;
    traits.supports_flash_attention = false;
    return traits;
  case LlamaBackendTarget::kOpenCL:
    traits.label = "opencl";
    traits.gpu_accelerated = true;
    traits.supports_flash_attention = false;
    return traits;
  case LlamaBackendTarget::kCpu:
  default:
    traits.label = "cpu";
    traits.gpu_accelerated = false;
    traits.supports_flash_attention = false;
    return traits;
  }
}

LlamaBackendConfig TuneLlamaBackendConfig(LlamaBackendTarget target,
                                          const LlamaBackendConfig &base) {
  LlamaBackendConfig tuned = base;
  const auto traits = DescribeLlamaBackendTarget(target);

  if (!traits.gpu_accelerated) {
    tuned.gpu_layers = 0;
    tuned.use_flash_attention = false;
  } else {
    if (tuned.gpu_layers <= 0) {
      tuned.gpu_layers = 99;
    }
    if (!traits.supports_flash_attention) {
      tuned.use_flash_attention = false;
    }
  }

  if (tuned.flash_attention_tile <= 0) {
    tuned.flash_attention_tile = 128;
  }

  if (target == LlamaBackendTarget::kCuda) {
    tuned.cuda_attention_kernel =
        NormalizeCudaAttentionKernel(tuned.cuda_attention_kernel);
    if (tuned.cuda_phase_overlap_min_prefill_tokens <= 0) {
      tuned.cuda_phase_overlap_min_prefill_tokens = 256;
    }

    // Populate structured CudaConfigExtension from flat fields.
    tuned.cuda_ext.attention_kernel = tuned.cuda_attention_kernel;
    tuned.cuda_ext.phase_overlap_scaffold = tuned.cuda_phase_overlap_scaffold;
    tuned.cuda_ext.phase_overlap_min_prefill_tokens =
        tuned.cuda_phase_overlap_min_prefill_tokens;
    tuned.cuda_ext.phase_overlap_prefill_replica =
        tuned.cuda_phase_overlap_prefill_replica;
    tuned.cuda_ext.kv_cache_dtype = tuned.inferflux_cuda_kv_cache_dtype;
    tuned.cuda_ext.dequantized_cache_policy =
        tuned.inferflux_cuda_dequantized_cache_policy;
    tuned.cuda_ext.require_fused_quantized_matmul =
        tuned.inferflux_cuda_require_fused_quantized_matmul;
  } else {
    tuned.cuda_attention_kernel = "standard";
    tuned.cuda_phase_overlap_scaffold = false;
    tuned.cuda_phase_overlap_prefill_replica = false;
  }

  return tuned;
}

} // namespace inferflux
