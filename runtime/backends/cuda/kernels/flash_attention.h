#pragma once

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
#include <memory>
#include <string>

namespace inferflux {
namespace cuda {
namespace kernels {

// Attention kernel types
enum class AttentionKernelType {
  kAuto,    // Automatically select best kernel
  kFlash3,  // FlashAttention-3 (Hopper H100 only)
  kFlash2,  // FlashAttention-2 (Ada/Ampere)
  kFlash1,  // FlashAttention-1 (older Volta/Turing)
  kStandard // Standard attention implementation
};

// Forward pass kernel configurations
struct FlashAttentionConfig {
  int head_dim;         // Dimension per attention head
  int num_heads;        // Number of attention heads
  int num_kv_heads;     // Number of key-value heads (for GQA/MQA)
  int max_seq_len;      // Maximum sequence length
  bool use_causal_mask; // Whether to apply causal mask
  float softmax_scale;  // Scaling factor for softmax

  // Computed values
  bool is_gqa;       // Grouped-query attention
  bool is_mqa;       // Multi-query attention
  int kv_head_ratio; // num_heads / num_kv_heads

  FlashAttentionConfig()
      : head_dim(128), num_heads(32), num_kv_heads(32), max_seq_len(2048),
        use_causal_mask(true), softmax_scale(1.0f), is_gqa(false),
        is_mqa(false), kv_head_ratio(1) {}
};

// Query GPU capabilities for FlashAttention
struct FlashAttentionCapabilities {
  bool supports_flash2;      // FlashAttention-2 (Ada/Ampere)
  bool supports_flash3;      // FlashAttention-3 (Hopper)
  int max_shared_mem;        // Max shared memory per block
  int max_threads_per_block; // Max threads per block
  int sm_count;              // Number of streaming multiprocessors

  // Recommended kernel based on hardware
  AttentionKernelType recommended_kernel;
};

// Query GPU capabilities for FlashAttention support
FlashAttentionCapabilities QueryFlashAttentionCapabilities(int device_id = 0);

// Detect best kernel type based on config and hardware
AttentionKernelType SelectOptimalKernel(const FlashAttentionConfig &config,
                                        const FlashAttentionCapabilities &caps);

// Get kernel type name for logging/metrics
std::string GetKernelTypeName(AttentionKernelType type);

// Unified forward entry point (auto-selects kernel, uses tiled FA2)
bool FlashAttentionForward(
    const FlashAttentionConfig &config, const void *q, const void *k,
    const void *v, void *output, void *lse, int batch_size, int seq_len,
    bool use_fp16 = true,
    AttentionKernelType kernel_type = AttentionKernelType::kAuto,
    cudaStream_t stream = 0);

/**
 * FlashAttention-2 FP16 forward pass with causal mask and GQA.
 *
 * This is the primary entry point for native kernel inference.
 * Launches the CUDA kernel directly (no fallback).
 *
 * @param Q          [batch, num_heads, query_len, head_dim] FP16
 * @param K          [batch, num_kv_heads, kv_len, head_dim] FP16
 * @param V          [batch, num_kv_heads, kv_len, head_dim] FP16
 * @param O          [batch, num_heads, query_len, head_dim] FP16
 * @param batch_size Batch size
 * @param query_len  Length of Q sequence
 * @param kv_len     Length of K/V sequence (includes past tokens)
 * @param num_heads  Number of Q attention heads
 * @param num_kv_heads Number of KV heads (GQA when < num_heads)
 * @param head_dim   Dimension per head
 * @param causal     Apply causal mask
 */
bool FlashAttention2ForwardNative(const half *Q, const half *K, const half *V,
                                  half *O, int batch_size, int query_len,
                                  int kv_len, int num_heads, int num_kv_heads,
                                  int head_dim, bool causal,
                                  cudaStream_t stream = 0);

// Templated FlashAttention-2 forward pass (native, typed)
template <typename T>
bool FlashAttention2ForwardNativeTyped(const T *Q, const T *K, const T *V, T *O,
                                       int batch_size, int query_len,
                                       int kv_len, int num_heads,
                                       int num_kv_heads, int head_dim,
                                       bool causal, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda
} // namespace inferflux
