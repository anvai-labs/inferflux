#include "runtime/backends/cuda/kernels/flash_attention.h"
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "server/logging/logger.h"

#include <cmath>
// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif
#include <stdexcept>
#include <string>

namespace inferflux {
namespace cuda {
namespace kernels {

namespace {

// Check if GPU architecture supports FlashAttention-2
bool SupportsFlashAttention2(const cudaDeviceProp &prop) {
  // FlashAttention-2 requires compute capability 8.0+ (Ampere/Ada)
  return prop.major >= 8;
}

// Check if GPU architecture supports FlashAttention-3
bool SupportsFlashAttention3(const cudaDeviceProp &prop) {
  // FlashAttention-3 requires compute capability 9.0+ (Hopper H100+)
  return prop.major >= 9;
}

} // anonymous namespace

//=============================================================================
// GPU Capability Querying
//=============================================================================

FlashAttentionCapabilities QueryFlashAttentionCapabilities(int device_id) {
  FlashAttentionCapabilities caps{};
  cudaDeviceProp prop;

  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    log::Error("flash_attention", "Failed to query device properties: " +
                                      std::string(cudaGetErrorString(err)));
    return caps;
  }

  // Query capabilities
  caps.max_shared_mem = prop.sharedMemPerBlock;
  caps.max_threads_per_block = prop.maxThreadsPerBlock;
  caps.sm_count = prop.multiProcessorCount;
  caps.supports_flash2 = SupportsFlashAttention2(prop);
  caps.supports_flash3 = SupportsFlashAttention3(prop);

  // Determine recommended kernel
  if (caps.supports_flash3) {
    caps.recommended_kernel = AttentionKernelType::kFlash3;
  } else if (caps.supports_flash2) {
    caps.recommended_kernel = AttentionKernelType::kFlash2;
  } else {
    caps.recommended_kernel = AttentionKernelType::kStandard;
  }

  // Log capabilities
  log::Info("flash_attention", "GPU: " + std::string(prop.name));
  log::Info("flash_attention",
            "Compute Capability: " + std::to_string(prop.major) + "." +
                std::to_string(prop.minor));
  log::Info("flash_attention",
            "FlashAttention-2: " +
                std::string(caps.supports_flash2 ? "YES" : "NO"));
  log::Info("flash_attention",
            "FlashAttention-3: " +
                std::string(caps.supports_flash3 ? "YES" : "NO"));
  log::Info("flash_attention", "Recommended Kernel: " +
                                   GetKernelTypeName(caps.recommended_kernel));
  log::Info("flash_attention", "SM Count: " + std::to_string(caps.sm_count));
  log::Info("flash_attention",
            "Max Shared Mem: " + std::to_string(caps.max_shared_mem / 1024) +
                " KB");

  return caps;
}

//=============================================================================
// Kernel Selection Logic
//=============================================================================

AttentionKernelType
SelectOptimalKernel(const FlashAttentionConfig &config,
                    const FlashAttentionCapabilities &caps) {

  // Update GQA/MQA detection
  bool is_gqa =
      (config.num_kv_heads > 0 && config.num_kv_heads < config.num_heads);
  bool is_mqa = (config.num_kv_heads == 1);

  // User override takes precedence (handled at caller level)
  // This function handles "auto" selection

  // Priority order: Flash3 > Flash2 > Flash1 > Standard
  if (caps.supports_flash3) {
    return AttentionKernelType::kFlash3;
  }

  if (caps.supports_flash2) {
    // FlashAttention-2 works well with GQA/MQA
    // Check if configuration is compatible
    if (config.head_dim <= 192 && config.max_seq_len <= 32768) {
      return AttentionKernelType::kFlash2;
    }
  }

  // Fallback to standard attention
  return AttentionKernelType::kStandard;
}

//=============================================================================
// Kernel Name Utilities
//=============================================================================

std::string GetKernelTypeName(AttentionKernelType type) {
  switch (type) {
  case AttentionKernelType::kAuto:
    return "auto";
  case AttentionKernelType::kFlash3:
    return "flash3";
  case AttentionKernelType::kFlash2:
    return "flash2";
  case AttentionKernelType::kFlash1:
    return "flash1";
  case AttentionKernelType::kStandard:
    return "standard";
  default:
    return "unknown";
  }
}

//=============================================================================
// Unified Forward Entry Point
//=============================================================================

bool FlashAttentionForward(const FlashAttentionConfig &config, const void *q,
                           const void *k, const void *v, void *output,
                           void *lse, int batch_size, int seq_len,
                           bool use_fp16, AttentionKernelType kernel_type,
                           cudaStream_t stream) {

  // Query capabilities if needed
  static FlashAttentionCapabilities caps = QueryFlashAttentionCapabilities();

  // Select kernel if auto
  if (kernel_type == AttentionKernelType::kAuto) {
    kernel_type = SelectOptimalKernel(config, caps);
  }

  log::Info("flash_attention",
            "Using kernel: " + GetKernelTypeName(kernel_type));

  // All paths now use the tiled FlashAttention-2 kernel.
  // FA3 falls through to FA2 (requires SM 9.0+ TMA which we don't have yet).
  // Standard/FA1 also use the tiled FA2 — it's strictly better.
  switch (kernel_type) {
  case AttentionKernelType::kFlash3:
    log::Warn("flash_attention",
              "FlashAttention-3 requested but not supported on this GPU, "
              "using tiled FlashAttention-2");
    [[fallthrough]];

  case AttentionKernelType::kFlash2:
  case AttentionKernelType::kFlash1:
  case AttentionKernelType::kStandard:
    if (use_fp16) {
      return FlashAttention2ForwardNativeTyped<half>(
          static_cast<const half *>(q), static_cast<const half *>(k),
          static_cast<const half *>(v), static_cast<half *>(output), batch_size,
          seq_len, seq_len, config.num_heads, config.num_kv_heads,
          config.head_dim, config.use_causal_mask, stream);
    } else {
      log::Error("flash_attention",
                 "FP32 FlashAttention not supported — use FP16 or BF16");
      return false;
    }

  default:
    log::Error("flash_attention", "Unknown kernel type");
    return false;
  }
}

//=============================================================================
// FlashAttention-2 FP16 Native (direct CUDA kernel launch)
//=============================================================================

bool FlashAttention2ForwardNative(const half *Q, const half *K, const half *V,
                                  half *O, int batch_size, int query_len,
                                  int kv_len, int num_heads, int num_kv_heads,
                                  int head_dim, bool causal,
                                  cudaStream_t stream) {
  // Delegate to the templated version which uses the tiled kernel
  return FlashAttention2ForwardNativeTyped<half>(
      Q, K, V, O, batch_size, query_len, kv_len, num_heads, num_kv_heads,
      head_dim, causal, stream);
}

//=============================================================================
// Templated FlashAttention-2 Native (typed dispatch)
//=============================================================================

template <typename T>
bool FlashAttention2ForwardNativeTyped(const T *Q, const T *K, const T *V, T *O,
                                       int batch_size, int query_len,
                                       int kv_len, int num_heads,
                                       int num_kv_heads, int head_dim,
                                       bool causal, cudaStream_t stream) {

  if (!Q || !K || !V || !O) {
    log::Error("flash_attention",
               "Null pointer in FlashAttention2ForwardNativeTyped");
    return false;
  }

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  cudaError_t err = cuda_kernel::FlashAttention2Typed<T>(
      Q, K, V, O, batch_size, query_len, kv_len, num_heads, num_kv_heads,
      head_dim, scale, causal, stream);

  if (err != cudaSuccess) {
    log::Error("flash_attention", "FlashAttention2Typed kernel failed: " +
                                      std::string(cudaGetErrorString(err)));
    return false;
  }
  return true;
}

// Explicit instantiations
template bool FlashAttention2ForwardNativeTyped<half>(const half *,
                                                      const half *,
                                                      const half *, half *, int,
                                                      int, int, int, int, int,
                                                      bool, cudaStream_t);
template bool FlashAttention2ForwardNativeTyped<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *, int, int, int, int, int, int, bool, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace inferflux
