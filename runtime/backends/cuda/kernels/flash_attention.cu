#include "flash_attention.cuh"
#include "flash_attention_mma.cuh"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// ============================================================================
// Tiled FlashAttention-2 kernel (BF16 + FP16)
//
// Properly tiled implementation with:
// - K/V tiles loaded into shared memory (amortized global reads)
// - Warp-shuffle dot product reduction (no per-position barriers)
// - Batched cross-warp reduction (1 barrier per tile, not per KV position)
// - Online softmax with rescaling
//
// Grid: (query_len, num_heads, batch_size)
// Block: next_pow2(head_dim) threads
// Shared memory: (2 * Bc * d + num_warps * Bc + Bc) * sizeof(float)
//
// For kv_len=1024, d=128: 128 barriers vs 8192 in the scalar version.
// ============================================================================

// KV tile size: 64 positions per tile (2x over previous 32).
// For head_dim=128, 4 warps: smem = (2*64*128 + 4*64 + 64) * 4 = ~67KB.
// Fits Ada (100KB max configurable) and Ampere (164KB). GPUs that can't
// configure enough smem will use cudaFuncSetAttribute in the host wrappers.
constexpr int FA2_TILE_KV = 64;

template <typename T>
__global__ void FlashAttention2TypedKernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, int query_len, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, float scale, bool causal) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_pos = blockIdx.x;

  if (q_pos >= query_len)
    return;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_head_ratio = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  // SHD layout strides
  const int q_stride = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  // Load this thread's Q dimension into register
  float q_reg = 0.0f;
  if (d < head_dim) {
    size_t q_offset = (size_t)batch_idx * query_len * q_stride +
                      (size_t)q_pos * q_stride + head_idx * head_dim + d;
    q_reg = DtypeTraits<T>::to_float(Q[q_offset]);
  }

  const int causal_limit = causal ? (kv_len - query_len + q_pos + 1) : kv_len;

  // Dynamic shared memory layout:
  //   s_k:         [FA2_TILE_KV * head_dim] floats — K tile
  //   s_v:         [FA2_TILE_KV * head_dim] floats — V tile
  //   s_warp_dots: [num_warps * FA2_TILE_KV] floats — per-warp partial sums
  //   s_scores:    [FA2_TILE_KV] floats — final attention scores
  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = smem + 2 * FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + num_warps * FA2_TILE_KV;

  // KV base pointer (this batch, this kv_head)
  const size_t kv_base =
      (size_t)batch_idx * kv_len * kv_stride + kv_head_idx * head_dim;

  // Online softmax state
  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  // Tile loop over KV positions
  for (int kv_start = 0; kv_start < causal_limit; kv_start += FA2_TILE_KV) {
    int tile_len = min(FA2_TILE_KV, causal_limit - kv_start);
    int total_elements = tile_len * head_dim;

    // Phase 1: Cooperatively load K and V tiles into shared memory.
    // Access is coalesced: consecutive threads load consecutive dimensions.
    for (int i = d; i < total_elements; i += num_threads) {
      int t = i / head_dim;
      int dim = i % head_dim;
      size_t kv_offset = kv_base + (size_t)(kv_start + t) * kv_stride + dim;
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads(); // BARRIER 1: tiles loaded

    // Phase 2: Compute tile_len dot products using warp-level reduction.
    // Each thread multiplies its Q dimension with the corresponding K
    // dimension, then warp shuffle reduces across 32 lanes. Lane 0 writes to
    // shared memory.
    for (int t = 0; t < tile_len; t++) {
      float partial = (d < head_dim) ? q_reg * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
      if (lane == 0)
        s_warp_dots[warp_id * FA2_TILE_KV + t] = partial;
    }
    __syncthreads(); // BARRIER 2: warp partials ready

    // Phase 3: Cross-warp reduction — sum partial dot products across warps.
    // First tile_len threads each reduce one score.
    if (d < tile_len) {
      float dot_sum = 0.0f;
      for (int w = 0; w < num_warps; w++)
        dot_sum += s_warp_dots[w * FA2_TILE_KV + d];
      s_scores[d] = dot_sum * scale;
    }
    __syncthreads(); // BARRIER 3: scores ready

    // Phase 4: Online softmax + V accumulation.
    // All threads read the same scores and independently update their
    // o_acc dimension. Since all threads see identical scores, they compute
    // identical row_max/row_sum — no broadcast needed.
    for (int t = 0; t < tile_len; t++) {
      float score = s_scores[t];
      float new_max = fmaxf(row_max, score);
      float rescale = expf(row_max - new_max);
      float exp_w = expf(score - new_max);

      if (d < head_dim)
        o_acc = o_acc * rescale + exp_w * s_v[t * head_dim + d];

      row_sum = row_sum * rescale + exp_w;
      row_max = new_max;
    }
    __syncthreads(); // BARRIER 4: safe to overwrite smem in next iteration
  }

  // Write output
  if (d < head_dim) {
    size_t o_offset = (size_t)batch_idx * query_len * q_stride +
                      (size_t)q_pos * q_stride + head_idx * head_dim + d;
    O[o_offset] =
        DtypeTraits<T>::from_float((row_sum > 0.0f) ? (o_acc / row_sum) : 0.0f);
  }
}

// ============================================================================
// GQA-grouped prefill: one block per (q_pos, kv_head, batch), processing all
// Q-heads that share that KV head. Amortizes KV tile loads by gqa_ratio.
//
// Grid: (query_len, num_kv_heads, batch_size)
// Block: next_pow2(head_dim) threads
// ============================================================================

template <typename T, int GQARatio>
__global__ void FlashAttention2TypedGQAKernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, int query_len, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, float scale, bool causal) {
  const int batch_idx = blockIdx.z;
  const int kv_head_idx = blockIdx.y;
  const int q_pos = blockIdx.x;

  if (q_pos >= query_len)
    return;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int q_stride = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  // Load Q registers for all GQARatio heads sharing this KV head
  float q_reg[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    const int head_idx = kv_head_idx * GQARatio + h;
    if (d < head_dim && head_idx < num_heads) {
      size_t q_offset = (size_t)batch_idx * query_len * q_stride +
                        (size_t)q_pos * q_stride + head_idx * head_dim + d;
      q_reg[h] = DtypeTraits<T>::to_float(Q[q_offset]);
    } else {
      q_reg[h] = 0.0f;
    }
  }

  const int causal_limit = causal ? (kv_len - query_len + q_pos + 1) : kv_len;

  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = smem + 2 * FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + GQARatio * num_warps * FA2_TILE_KV;

  const size_t kv_base =
      (size_t)batch_idx * kv_len * kv_stride + kv_head_idx * head_dim;

  float row_max[GQARatio];
  float row_sum[GQARatio];
  float o_acc[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    row_max[h] = -INFINITY;
    row_sum[h] = 0.0f;
    o_acc[h] = 0.0f;
  }

  for (int kv_start = 0; kv_start < causal_limit; kv_start += FA2_TILE_KV) {
    int tile_len = min(FA2_TILE_KV, causal_limit - kv_start);
    int total_elements = tile_len * head_dim;

    // Phase 1: Load K/V tile (ONCE for all GQARatio heads)
    for (int i = d; i < total_elements; i += num_threads) {
      int t = i / head_dim;
      int dim = i % head_dim;
      size_t kv_offset = kv_base + (size_t)(kv_start + t) * kv_stride + dim;
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    // Phase 2: Dot products for all GQARatio heads
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        float partial =
            (d < head_dim) ? q_reg[h] * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        if (lane == 0)
          s_warp_dots[(h * num_warps + warp_id) * FA2_TILE_KV + t] = partial;
      }
    }
    __syncthreads();

    // Phase 3: Cross-warp reduction per head.
    // Process each head sequentially; within each head, threads cover
    // tile positions in parallel (same pattern as the non-GQA kernel).
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      if (d < tile_len) {
        float dot_sum = 0.0f;
        for (int w = 0; w < num_warps; w++)
          dot_sum += s_warp_dots[(h * num_warps + w) * FA2_TILE_KV + d];
        s_scores[h * FA2_TILE_KV + d] = dot_sum * scale;
      }
    }
    __syncthreads();

    // Phase 4: Online softmax + V accumulation per head
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        float score = s_scores[h * FA2_TILE_KV + t];
        float new_max = fmaxf(row_max[h], score);
        float rescale = expf(row_max[h] - new_max);
        float exp_w = expf(score - new_max);

        if (d < head_dim)
          o_acc[h] = o_acc[h] * rescale + exp_w * s_v[t * head_dim + d];

        row_sum[h] = row_sum[h] * rescale + exp_w;
        row_max[h] = new_max;
      }
    }
    __syncthreads();
  }

  // Write output for all GQARatio heads
  if (d < head_dim) {
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      const int head_idx = kv_head_idx * GQARatio + h;
      if (head_idx < num_heads) {
        size_t o_offset = (size_t)batch_idx * query_len * q_stride +
                          (size_t)q_pos * q_stride + head_idx * head_dim + d;
        O[o_offset] = DtypeTraits<T>::from_float(
            (row_sum[h] > 0.0f) ? (o_acc[h] / row_sum[h]) : 0.0f);
      }
    }
  }
}

// ============================================================================
// Host wrappers
// ============================================================================

// Compute shared memory bytes for FA2 kernels given tile size.
static inline int ComputeFA2Smem(int tile_kv, int head_dim, int num_warps) {
  return (2 * tile_kv * head_dim + num_warps * tile_kv + tile_kv) *
         static_cast<int>(sizeof(float));
}

// Compute shared memory bytes for GQA-grouped FA2 kernels.
static inline int ComputeFA2SmemGQA(int tile_kv, int head_dim, int num_warps,
                                    int gqa_ratio) {
  // K tile + V tile + per-head warp partial dots + per-head scores
  return (2 * tile_kv * head_dim + num_warps * tile_kv + tile_kv) *
             static_cast<int>(sizeof(float)) +
         // Extra smem for gqa_ratio-1 additional heads' warp_dots and scores
         (gqa_ratio - 1) * (num_warps * tile_kv + tile_kv) *
             static_cast<int>(sizeof(float));
}

// Configure extended shared memory for FA2 kernels.
// cudaFuncSetAttribute is idempotent and lightweight.
template <typename Func>
static void ConfigureFA2Smem(Func kernel, int smem_bytes) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_bytes);
}

// Launch MMA-accelerated GQA prefill kernel for query_len >= 16.
// Uses tensor cores for Q*K^T, scalar V accumulation.
template <typename T, int GQARatio>
static cudaError_t
LaunchMMAGQAPrefill(const T *Q, const T *K, const T *V, T *O, int batch_size,
                    int query_len, int kv_len, int num_heads, int num_kv_heads,
                    int head_dim, float scale, bool causal,
                    cudaStream_t stream) {
  const int threads = MMA_FA2_THREADS; // 128 = 4 warps
  int smem = ComputeMMAFA2Smem(head_dim);

  ConfigureFA2Smem(FlashAttention2MMAGQAKernel<T, GQARatio>, smem);

  const int q_blocks = (query_len + MMA_FA2_BR - 1) / MMA_FA2_BR;
  dim3 grid(q_blocks, num_kv_heads, batch_size);

  FlashAttention2MMAGQAKernel<T, GQARatio><<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Launch GQA-grouped prefill kernel for a specific GQA ratio.
// Selects MMA kernel when query_len >= 16 and head_dim fits MMA tiles.
template <typename T, int GQARatio>
static cudaError_t
LaunchGQAPrefill(const T *Q, const T *K, const T *V, T *O, int batch_size,
                 int query_len, int kv_len, int num_heads, int num_kv_heads,
                 int head_dim, float scale, bool causal, cudaStream_t stream) {
  // Use MMA kernel when query_len >= 16, head_dim is multiple of 16, and
  // head_dim <= 128 (so 128 threads suffice to cover all dimensions).
  if (query_len >= MMA_FA2_BR && head_dim <= MMA_FA2_THREADS &&
      (head_dim % 16) == 0) {
    return LaunchMMAGQAPrefill<T, GQARatio>(Q, K, V, O, batch_size, query_len,
                                            kv_len, num_heads, num_kv_heads,
                                            head_dim, scale, causal, stream);
  }

  // Fallback: scalar GQA kernel (one Q position per block)
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  int num_warps = threads / 32;
  int smem = ComputeFA2SmemGQA(FA2_TILE_KV, head_dim, num_warps, GQARatio);

  ConfigureFA2Smem(FlashAttention2TypedGQAKernel<T, GQARatio>, smem);

  dim3 grid(query_len, num_kv_heads, batch_size);

  FlashAttention2TypedGQAKernel<T, GQARatio><<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Templated FlashAttention-2 host wrapper (tiled)
template <typename T>
cudaError_t FlashAttention2Typed(const T *Q, const T *K, const T *V, T *O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream) {
  // Use GQA-grouped kernel when multiple Q-heads share a KV head.
  const int gqa_ratio = (num_kv_heads > 0 && num_heads > num_kv_heads)
                            ? (num_heads / num_kv_heads)
                            : 1;

  if (gqa_ratio == 8) {
    return LaunchGQAPrefill<T, 8>(Q, K, V, O, batch_size, query_len, kv_len,
                                  num_heads, num_kv_heads, head_dim, scale,
                                  causal, stream);
  }
  if (gqa_ratio == 4) {
    return LaunchGQAPrefill<T, 4>(Q, K, V, O, batch_size, query_len, kv_len,
                                  num_heads, num_kv_heads, head_dim, scale,
                                  causal, stream);
  }
  if (gqa_ratio == 2) {
    return LaunchGQAPrefill<T, 2>(Q, K, V, O, batch_size, query_len, kv_len,
                                  num_heads, num_kv_heads, head_dim, scale,
                                  causal, stream);
  }

  // Fallback: non-GQA or unsupported ratio
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  int num_warps = threads / 32;
  int smem = ComputeFA2Smem(FA2_TILE_KV, head_dim, num_warps);

  ConfigureFA2Smem(FlashAttention2TypedKernel<T>, smem);

  dim3 grid(query_len, num_heads, batch_size);

  FlashAttention2TypedKernel<T><<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t FlashAttention2Typed<half>(const half *, const half *,
                                                const half *, half *, int, int,
                                                int, int, int, int, float, bool,
                                                cudaStream_t);
template cudaError_t FlashAttention2Typed<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *, int, int, int, int, int, int, float, bool, cudaStream_t);

// ============================================================================
// FlashDecodeMultiSeq: Batched decode attention for B sequences with
// variable KV cache lengths. Each query has query_len=1.
//
// Grid: (batch_size, num_heads)
// Block: next_pow2(head_dim) threads
// Each block handles one (batch, head) pair.
// ============================================================================

template <typename T>
__global__ void FlashDecodeMultiSeqKernel(const T *__restrict__ Q,
                                          const T *const *__restrict__ K_ptrs,
                                          const T *const *__restrict__ V_ptrs,
                                          T *__restrict__ O,
                                          const int *__restrict__ kv_lens,
                                          int num_heads, int num_kv_heads,
                                          int head_dim, float scale) {
  const int b = blockIdx.x;
  const int head_idx = blockIdx.y;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  if (kv_len <= 0)
    return;

  const int kv_head_ratio = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  // Q is [B, num_heads * head_dim]
  float q_reg = 0.0f;
  if (d < head_dim) {
    q_reg = DtypeTraits<T>::to_float(
        Q[b * num_heads * head_dim + head_idx * head_dim + d]);
  }

  // K/V pointers for this batch element: [kv_len, num_kv_heads * head_dim]
  const T *K = K_ptrs[b];
  const T *V = V_ptrs[b];
  const int kv_stride = num_kv_heads * head_dim;

  // Shared memory layout: same as FA2
  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = smem + 2 * FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + num_warps * FA2_TILE_KV;

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  for (int kv_start = 0; kv_start < kv_len; kv_start += FA2_TILE_KV) {
    int tile_len = min(FA2_TILE_KV, kv_len - kv_start);
    int total_elements = tile_len * head_dim;

    // Load K and V tiles
    for (int i = d; i < total_elements; i += num_threads) {
      int t = i / head_dim;
      int dim = i % head_dim;
      size_t kv_offset =
          (size_t)(kv_start + t) * kv_stride + kv_head_idx * head_dim + dim;
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    // Dot products with warp reduction
    for (int t = 0; t < tile_len; t++) {
      float partial = (d < head_dim) ? q_reg * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
      if (lane == 0)
        s_warp_dots[warp_id * FA2_TILE_KV + t] = partial;
    }
    __syncthreads();

    // Cross-warp reduction
    if (d < tile_len) {
      float dot_sum = 0.0f;
      for (int w = 0; w < num_warps; w++)
        dot_sum += s_warp_dots[w * FA2_TILE_KV + d];
      s_scores[d] = dot_sum * scale;
    }
    __syncthreads();

    // Online softmax + V accumulation
    for (int t = 0; t < tile_len; t++) {
      float score = s_scores[t];
      float new_max = fmaxf(row_max, score);
      float rescale = expf(row_max - new_max);
      float exp_w = expf(score - new_max);

      if (d < head_dim)
        o_acc = o_acc * rescale + exp_w * s_v[t * head_dim + d];

      row_sum = row_sum * rescale + exp_w;
      row_max = new_max;
    }
    __syncthreads();
  }

  // Write output: O is [B, num_heads * head_dim]
  if (d < head_dim) {
    O[b * num_heads * head_dim + head_idx * head_dim + d] =
        DtypeTraits<T>::from_float((row_sum > 0.0f) ? (o_acc / row_sum) : 0.0f);
  }
}

template <typename T>
cudaError_t FlashDecodeMultiSeq(const T *Q, const T *const *d_k_ptrs,
                                const T *const *d_v_ptrs, T *O,
                                const int *d_kv_lens, int batch_size,
                                int num_heads, int num_kv_heads, int head_dim,
                                float scale, cudaStream_t stream) {
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  int num_warps = threads / 32;

  int smem = ComputeFA2Smem(FA2_TILE_KV, head_dim, num_warps);

  ConfigureFA2Smem(FlashDecodeMultiSeqKernel<T>, smem);

  dim3 grid(batch_size, num_heads);

  FlashDecodeMultiSeqKernel<T><<<grid, threads, smem, stream>>>(
      Q, d_k_ptrs, d_v_ptrs, O, d_kv_lens, num_heads, num_kv_heads, head_dim,
      scale);
  return cudaGetLastError();
}

template cudaError_t FlashDecodeMultiSeq<half>(const half *,
                                               const half *const *,
                                               const half *const *, half *,
                                               const int *, int, int, int, int,
                                               float, cudaStream_t);
template cudaError_t FlashDecodeMultiSeq<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *const *,
    const __nv_bfloat16 *const *, __nv_bfloat16 *, const int *, int, int, int,
    int, float, cudaStream_t);

// ============================================================================
// GQA-grouped decode: one block per (batch, kv_head), processing all Q-heads
// that share that KV head. This amortizes KV tile loads by gqa_ratio
// (e.g., 4x for Qwen 2.5 3B with 32 Q-heads / 8 KV-heads).
//
// Grid: (batch_size, num_kv_heads)
// Block: next_pow2(head_dim) threads
// Smem: K tile + V tile + warp_dots*gqa + scores*gqa
// ============================================================================

template <typename T, int GQARatio>
__global__ void FlashDecodeMultiSeqStridedGQAKernel(
    const T *__restrict__ Q, const T *__restrict__ kv_buffer, T *__restrict__ O,
    const int *__restrict__ seq_ids, const int *__restrict__ kv_lens, int layer,
    int num_heads, int num_kv_heads, int head_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, float scale) {
  const int b = blockIdx.x;
  const int kv_head_idx = blockIdx.y;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  if (kv_len <= 0) {
    return;
  }

  // Load Q registers for all GQARatio heads sharing this KV head.
  // head_idx = kv_head_idx * GQARatio + h
  float q_reg[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    const int head_idx = kv_head_idx * GQARatio + h;
    if (d < head_dim && head_idx < num_heads) {
      q_reg[h] = DtypeTraits<T>::to_float(
          Q[b * num_heads * head_dim + head_idx * head_dim + d]);
    } else {
      q_reg[h] = 0.0f;
    }
  }

  // Derive K/V pointers for this KV head
  const size_t seq_offset = static_cast<size_t>(seq_ids[b]) * slot_stride;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const T *K = kv_buffer + seq_offset + layer_offset;
  const T *V = K + kv_stride;
  const int kv_stride_elems = num_kv_heads * head_dim;

  // Shared memory layout:
  //   s_k:         [FA2_TILE_KV * head_dim] floats — K tile (shared across
  //   heads) s_v:         [FA2_TILE_KV * head_dim] floats — V tile (shared
  //   across heads) s_warp_dots: [GQARatio * num_warps * FA2_TILE_KV] floats —
  //   per-head warp partials s_scores:    [GQARatio * FA2_TILE_KV] floats —
  //   per-head final scores
  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = s_v + FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + GQARatio * num_warps * FA2_TILE_KV;

  // Per-head online softmax state
  float row_max[GQARatio];
  float row_sum[GQARatio];
  float o_acc[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    row_max[h] = -INFINITY;
    row_sum[h] = 0.0f;
    o_acc[h] = 0.0f;
  }

  for (int kv_start = 0; kv_start < kv_len; kv_start += FA2_TILE_KV) {
    const int tile_len = min(FA2_TILE_KV, kv_len - kv_start);
    const int total_elements = tile_len * head_dim;

    // Phase 1: Load K/V tile (ONCE for all GQARatio heads)
    for (int i = d; i < total_elements; i += num_threads) {
      const int t = i / head_dim;
      const int dim = i % head_dim;
      const size_t kv_offset =
          static_cast<size_t>(kv_start + t) *
              static_cast<size_t>(kv_stride_elems) +
          static_cast<size_t>(kv_head_idx * head_dim + dim);
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    // Phase 2: Dot products — all GQARatio heads against shared K tile
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        float partial =
            (d < head_dim) ? q_reg[h] * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        if (lane == 0) {
          s_warp_dots[(h * num_warps + warp_id) * FA2_TILE_KV + t] = partial;
        }
      }
    }
    __syncthreads();

    // Phase 3: Cross-warp reduction — per head.
    // Process each head sequentially; within each head, threads cover
    // tile positions in parallel (same pattern as the non-GQA kernel).
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      if (d < tile_len) {
        float dot_sum = 0.0f;
        for (int w = 0; w < num_warps; w++) {
          dot_sum += s_warp_dots[(h * num_warps + w) * FA2_TILE_KV + d];
        }
        s_scores[h * FA2_TILE_KV + d] = dot_sum * scale;
      }
    }
    __syncthreads();

    // Phase 4: Online softmax + V accumulation (per head)
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        const float score = s_scores[h * FA2_TILE_KV + t];
        const float new_max = fmaxf(row_max[h], score);
        const float rescale = expf(row_max[h] - new_max);
        const float exp_w = expf(score - new_max);

        if (d < head_dim) {
          o_acc[h] = o_acc[h] * rescale + exp_w * s_v[t * head_dim + d];
        }

        row_sum[h] = row_sum[h] * rescale + exp_w;
        row_max[h] = new_max;
      }
    }
    __syncthreads();
  }

  // Write output for all GQARatio heads
  if (d < head_dim) {
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      const int head_idx = kv_head_idx * GQARatio + h;
      if (head_idx < num_heads) {
        O[b * num_heads * head_dim + head_idx * head_dim + d] =
            DtypeTraits<T>::from_float(
                (row_sum[h] > 0.0f) ? (o_acc[h] / row_sum[h]) : 0.0f);
      }
    }
  }
}

// ============================================================================
// KV-split FlashDecode: parallelize across KV sequence length via grid.z
//
// Each block handles a chunk of the KV sequence for one (batch, kv_head) pair.
// Partial softmax state (max, sum, weighted O) is written to an intermediate
// buffer.  A lightweight reduction kernel then combines the partials using the
// online softmax identity:
//   O = (sum_i * exp(max_i - global_max)) * O_i / global_sum
//
// This is the same split-K strategy used by llama.cpp's flash_attn_ext_vec.
// On Qwen2.5-3B (GQA 16/2, head_dim=128) it replaces 2 blocks with 2*16=32,
// recovering ~3.9 ms/token at c=1 decode.
// ============================================================================

template <typename T, int GQARatio>
__global__ void FlashDecodeGQASplitKernel(
    const T *__restrict__ Q, const T *__restrict__ kv_buffer,
    float *__restrict__ partial_O,   // [B, num_kv_heads, num_splits, GQARatio, head_dim]
    float *__restrict__ partial_max, // [B, num_kv_heads, num_splits, GQARatio]
    float *__restrict__ partial_sum, // [B, num_kv_heads, num_splits, GQARatio]
    const int *__restrict__ seq_ids, const int *__restrict__ kv_lens, int layer,
    int num_heads, int num_kv_heads, int head_dim, int num_splits,
    size_t slot_stride, size_t layer_stride, size_t kv_stride, float scale) {
  const int b = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int split_id = blockIdx.z;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  // Compute this split's KV range
  const int split_len = (kv_len + num_splits - 1) / num_splits;
  const int kv_begin = split_id * split_len;
  const int kv_end = min(kv_begin + split_len, kv_len);
  if (kv_begin >= kv_len) {
    // This split has no work — write identity state
    if (d < GQARatio) {
      const size_t base =
          ((static_cast<size_t>(b) * num_kv_heads + kv_head_idx) * num_splits +
           split_id) *
          GQARatio;
      partial_max[base + d] = -INFINITY;
      partial_sum[base + d] = 0.0f;
    }
    return;
  }

  // Load Q registers for all GQARatio heads sharing this KV head
  float q_reg[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    const int head_idx = kv_head_idx * GQARatio + h;
    if (d < head_dim && head_idx < num_heads) {
      q_reg[h] = DtypeTraits<T>::to_float(
          Q[b * num_heads * head_dim + head_idx * head_dim + d]);
    } else {
      q_reg[h] = 0.0f;
    }
  }

  // Derive K/V pointers
  const size_t seq_offset = static_cast<size_t>(seq_ids[b]) * slot_stride;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const T *K = kv_buffer + seq_offset + layer_offset;
  const T *V = K + kv_stride;
  const int kv_stride_elems = num_kv_heads * head_dim;

  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = s_v + FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + GQARatio * num_warps * FA2_TILE_KV;

  float row_max[GQARatio];
  float row_sum[GQARatio];
  float o_acc[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
    row_max[h] = -INFINITY;
    row_sum[h] = 0.0f;
    o_acc[h] = 0.0f;
  }

  for (int kv_start = kv_begin; kv_start < kv_end; kv_start += FA2_TILE_KV) {
    const int tile_len = min(FA2_TILE_KV, kv_end - kv_start);
    const int total_elements = tile_len * head_dim;

    // Load K/V tile
    for (int i = d; i < total_elements; i += num_threads) {
      const int t = i / head_dim;
      const int dim = i % head_dim;
      const size_t kv_offset = static_cast<size_t>(kv_start + t) *
                                   static_cast<size_t>(kv_stride_elems) +
                               static_cast<size_t>(kv_head_idx * head_dim + dim);
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    // Dot products
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        float partial =
            (d < head_dim) ? q_reg[h] * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        if (lane == 0) {
          s_warp_dots[(h * num_warps + warp_id) * FA2_TILE_KV + t] = partial;
        }
      }
    }
    __syncthreads();

    // Cross-warp reduction
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      if (d < tile_len) {
        float dot_sum = 0.0f;
        for (int w = 0; w < num_warps; w++) {
          dot_sum += s_warp_dots[(h * num_warps + w) * FA2_TILE_KV + d];
        }
        s_scores[h * FA2_TILE_KV + d] = dot_sum * scale;
      }
    }
    __syncthreads();

    // Online softmax + V accumulation
    for (int t = 0; t < tile_len; t++) {
#pragma unroll
      for (int h = 0; h < GQARatio; h++) {
        const float score = s_scores[h * FA2_TILE_KV + t];
        const float new_max = fmaxf(row_max[h], score);
        const float rescale = expf(row_max[h] - new_max);
        const float exp_w = expf(score - new_max);
        if (d < head_dim) {
          o_acc[h] = o_acc[h] * rescale + exp_w * s_v[t * head_dim + d];
        }
        row_sum[h] = row_sum[h] * rescale + exp_w;
        row_max[h] = new_max;
      }
    }
    __syncthreads();
  }

  // Write partial results
  const size_t split_base =
      ((static_cast<size_t>(b) * num_kv_heads + kv_head_idx) * num_splits +
       split_id) *
      GQARatio;
  if (d < head_dim) {
#pragma unroll
    for (int h = 0; h < GQARatio; h++) {
      partial_O[(split_base + h) * head_dim + d] = o_acc[h];
    }
  }
  if (d < GQARatio) {
    partial_max[split_base + d] = row_max[d];
    partial_sum[split_base + d] = row_sum[d];
  }
}

// Reduction kernel: combine partial softmax results across splits.
// Grid: (B, num_heads)  Block: head_dim threads
template <typename T>
__global__ void FlashDecodeReduceSplitsKernel(
    const float *__restrict__ partial_O,
    const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum, T *__restrict__ O,
    int num_heads, int num_kv_heads, int head_dim, int num_splits) {
  const int b = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = threadIdx.x;
  if (d >= head_dim)
    return;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int h = head_idx % gqa_ratio;

  // Find global max across splits
  float global_max = -INFINITY;
  const size_t split_base =
      (static_cast<size_t>(b) * num_kv_heads + kv_head_idx) * num_splits;
  for (int s = 0; s < num_splits; s++) {
    float m = partial_max[(split_base + s) * gqa_ratio + h];
    global_max = fmaxf(global_max, m);
  }

  // Combine weighted O and sum
  float global_sum = 0.0f;
  float o_final = 0.0f;
  for (int s = 0; s < num_splits; s++) {
    const size_t idx = (split_base + s) * gqa_ratio + h;
    float m = partial_max[idx];
    float sm = partial_sum[idx];
    float weight = expf(m - global_max) * sm;
    float o_val = partial_O[idx * head_dim + d];
    // o_val is sum-weighted (not normalized), so scale by exp(m - global_max)
    o_final += expf(m - global_max) * o_val;
    global_sum += weight;
  }

  O[b * num_heads * head_dim + head_idx * head_dim + d] =
      DtypeTraits<T>::from_float(
          (global_sum > 0.0f) ? (o_final / global_sum) : 0.0f);
}

template <typename T>
__global__ void FlashDecodeMultiSeqStridedKernel(
    const T *__restrict__ Q, const T *__restrict__ kv_buffer, T *__restrict__ O,
    const int *__restrict__ seq_ids, const int *__restrict__ kv_lens, int layer,
    int num_heads, int num_kv_heads, int head_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, float scale) {
  const int b = blockIdx.x;
  const int head_idx = blockIdx.y;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  if (kv_len <= 0) {
    return;
  }

  const int kv_head_ratio = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  float q_reg = 0.0f;
  if (d < head_dim) {
    q_reg = DtypeTraits<T>::to_float(
        Q[b * num_heads * head_dim + head_idx * head_dim + d]);
  }

  const size_t seq_offset = static_cast<size_t>(seq_ids[b]) * slot_stride;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const T *K = kv_buffer + seq_offset + layer_offset;
  const T *V = K + kv_stride;
  const int kv_stride_elems = num_kv_heads * head_dim;

  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = s_v + FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + num_warps * FA2_TILE_KV;

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  for (int kv_start = 0; kv_start < kv_len; kv_start += FA2_TILE_KV) {
    const int tile_len = min(FA2_TILE_KV, kv_len - kv_start);
    const int total_elements = tile_len * head_dim;

    for (int i = d; i < total_elements; i += num_threads) {
      const int t = i / head_dim;
      const int dim = i % head_dim;
      const size_t kv_offset =
          static_cast<size_t>(kv_start + t) *
              static_cast<size_t>(kv_stride_elems) +
          static_cast<size_t>(kv_head_idx * head_dim + dim);
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    for (int t = 0; t < tile_len; t++) {
      float partial = (d < head_dim) ? q_reg * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
      }
      if (lane == 0) {
        s_warp_dots[warp_id * FA2_TILE_KV + t] = partial;
      }
    }
    __syncthreads();

    if (d < tile_len) {
      float dot_sum = 0.0f;
      for (int w = 0; w < num_warps; w++) {
        dot_sum += s_warp_dots[w * FA2_TILE_KV + d];
      }
      s_scores[d] = dot_sum * scale;
    }
    __syncthreads();

    for (int t = 0; t < tile_len; t++) {
      const float score = s_scores[t];
      const float new_max = fmaxf(row_max, score);
      const float rescale = expf(row_max - new_max);
      const float exp_w = expf(score - new_max);

      if (d < head_dim) {
        o_acc = o_acc * rescale + exp_w * s_v[t * head_dim + d];
      }

      row_sum = row_sum * rescale + exp_w;
      row_max = new_max;
    }
    __syncthreads();
  }

  if (d < head_dim) {
    O[b * num_heads * head_dim + head_idx * head_dim + d] =
        DtypeTraits<T>::from_float((row_sum > 0.0f) ? (o_acc / row_sum) : 0.0f);
  }
}

// ---------------------------------------------------------------------------
// Warp-per-head FlashDecode: each warp owns one Q head and serially scans
// the full KV range with 32-lane dot reductions (5 shuffle rounds) — no
// shared memory, no block-wide barriers, no split workspace, no combine
// kernel. The block-cooperative split kernel pays ~232 synchronized
// block reductions per CTA at decode sizes; this layout pays none, which
// is why llama.cpp's decode attention is 4x faster at short contexts.
// One launch, shape-static (kv_len read on device): CUDA-graph safe.
// Grid: (ceil(num_heads / kWarpsPerBlock), batch).
// ---------------------------------------------------------------------------
constexpr int kFlashDecodeWarpsPerBlock = 4;

template <typename T, int GQARatio>
__global__ void FlashDecodeGQAWarpPerHeadKernel(
    const T *__restrict__ Q, const T *__restrict__ kv_buffer, T *__restrict__ O,
    const int *__restrict__ seq_ids, const int *__restrict__ kv_lens, int layer,
    int num_heads, int num_kv_heads, int head_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, float scale) {
  const int warp = (blockIdx.x * kFlashDecodeWarpsPerBlock) +
                   (threadIdx.x / 32);
  const int lane = threadIdx.x & 31;
  const int b = blockIdx.y;
  if (warp >= num_heads) {
    return;
  }
  const int head_idx = warp;
  const int kv_head_idx = head_idx / GQARatio;
  // Lanes stride head_dim: 4 consecutive elements per lane (head_dim 128).
  const int elems_per_lane = head_dim / 32;

  const int kv_len = kv_lens[b];
  if (kv_len <= 0) {
    for (int e = 0; e < elems_per_lane; ++e) {
      const int dim = lane * elems_per_lane + e;
      O[(static_cast<size_t>(b) * num_heads + head_idx) * head_dim + dim] =
          static_cast<T>(0.0f);
    }
    return;
  }

  const size_t seq_offset = static_cast<size_t>(seq_ids[b]) * slot_stride;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const T *K = kv_buffer + seq_offset + layer_offset;
  const T *V = K + kv_stride;
  const int kv_row_elems = num_kv_heads * head_dim;
  const size_t kv_head_off = static_cast<size_t>(kv_head_idx) * head_dim +
                             lane * elems_per_lane;

  // Q fragment.
  float q[8];
  const T *q_row = Q + static_cast<size_t>(b) * num_heads * head_dim +
                   static_cast<size_t>(head_idx) * head_dim +
                   lane * elems_per_lane;
#pragma unroll
  for (int e = 0; e < elems_per_lane; ++e) {
    q[e] = static_cast<float>(q_row[e]);
  }

  float m = -INFINITY;
  float l = 0.0f;
  float acc[8] = {0.0f};

  for (int t = 0; t < kv_len; ++t) {
    const size_t base = static_cast<size_t>(t) * kv_row_elems + kv_head_off;
    // Q dot K (per-lane partial, then 5-round warp reduce).
    float dot = 0.0f;
#pragma unroll
    for (int e = 0; e < elems_per_lane; ++e) {
      dot += q[e] * static_cast<float>(K[base + e]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
    }
    const float score = dot * scale;

    // Online softmax update (m/l identical across lanes after reduce).
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float p = __expf(score - m_new);
    l = l * alpha + p;
    const float inv_alpha_scale = alpha;
#pragma unroll
    for (int e = 0; e < elems_per_lane; ++e) {
      acc[e] = acc[e] * inv_alpha_scale + p * static_cast<float>(V[base + e]);
    }
    m = m_new;
  }

  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  T *o_row = O + static_cast<size_t>(b) * num_heads * head_dim +
             static_cast<size_t>(head_idx) * head_dim +
             lane * elems_per_lane;
#pragma unroll
  for (int e = 0; e < elems_per_lane; ++e) {
    o_row[e] = static_cast<T>(acc[e] * inv_l);
  }
}

template <typename T, int GQARatio>
static cudaError_t
LaunchGQADecodeWarpPerHead(const T *Q, const T *kv_buffer, T *O,
                           const int *d_seq_ids, const int *d_kv_lens, int layer,
                           int batch_size, int num_heads, int num_kv_heads,
                           int head_dim, size_t slot_stride, size_t layer_stride,
                           size_t kv_stride, float scale, cudaStream_t stream) {
  const int blocks =
      (num_heads + kFlashDecodeWarpsPerBlock - 1) / kFlashDecodeWarpsPerBlock;
  dim3 grid(blocks, batch_size);
  dim3 threads(kFlashDecodeWarpsPerBlock * 32);
  FlashDecodeGQAWarpPerHeadKernel<T, GQARatio>
      <<<grid, threads, 0, stream>>>(
          Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, num_heads,
          num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale);
  return cudaGetLastError();
}

// Number of KV-length splits for parallel FlashDecode.
// 8 splits provides sufficient SM saturation for Ada/Ampere GPUs (76-128 SMs)
// while halving the split workspace memory vs the original 16-split design.
constexpr int kFlashDecodeSplits = 8;

// Minimum KV length before enabling split-K parallelism.
// Below this threshold, a single block per KV head is fast enough.
constexpr int kFlashDecodeSplitThreshold = 128;

// Launch GQA-grouped decode kernel for a specific GQA ratio.
template <typename T, int GQARatio>
static cudaError_t LaunchGQADecode(const T *Q, const T *kv_buffer, T *O,
                                    const int *d_seq_ids, const int *d_kv_lens,
                                    int layer, int batch_size, int num_heads,
                                    int num_kv_heads, int head_dim,
                                    size_t slot_stride, size_t layer_stride,
                                    size_t kv_stride, float scale,
                                    cudaStream_t stream,
                                    void *split_workspace = nullptr,
                                    size_t split_workspace_bytes = 0) {
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  const int num_warps = threads / 32;
  int smem = ComputeFA2SmemGQA(FA2_TILE_KV, head_dim, num_warps, GQARatio);

  // Compute workspace needed for split-K
  const int num_splits = kFlashDecodeSplits;
  const size_t partial_O_bytes = static_cast<size_t>(batch_size) * num_kv_heads *
                                 num_splits * GQARatio * head_dim * sizeof(float);
  const size_t partial_scalar_bytes = static_cast<size_t>(batch_size) *
                                      num_kv_heads * num_splits * GQARatio *
                                      sizeof(float);
  const size_t total_workspace = partial_O_bytes + 2 * partial_scalar_bytes;

  // Use split-K path when workspace is available and large enough
  if (split_workspace && split_workspace_bytes >= total_workspace) {
    ConfigureFA2Smem(FlashDecodeGQASplitKernel<T, GQARatio>, smem);

    float *partial_O = static_cast<float *>(split_workspace);
    float *partial_max =
        reinterpret_cast<float *>(reinterpret_cast<char *>(split_workspace) +
                                  partial_O_bytes);
    float *partial_sum =
        reinterpret_cast<float *>(reinterpret_cast<char *>(split_workspace) +
                                  partial_O_bytes + partial_scalar_bytes);

    dim3 grid(batch_size, num_kv_heads, num_splits);
    FlashDecodeGQASplitKernel<T, GQARatio>
        <<<grid, threads, smem, stream>>>(
            Q, kv_buffer, partial_O, partial_max, partial_sum, d_seq_ids,
            d_kv_lens, layer, num_heads, num_kv_heads, head_dim, num_splits,
            slot_stride, layer_stride, kv_stride, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      return err;

    // Reduction: combine partial results across splits
    int reduce_threads = 1;
    while (reduce_threads < head_dim)
      reduce_threads <<= 1;
    reduce_threads = min(reduce_threads, 1024);
    dim3 reduce_grid(batch_size, num_heads);
    FlashDecodeReduceSplitsKernel<T>
        <<<reduce_grid, reduce_threads, 0, stream>>>(
            partial_O, partial_max, partial_sum, O, num_heads, num_kv_heads,
            head_dim, num_splits);
    return cudaGetLastError();
  }

  // Fallback: original unsplit kernel
  ConfigureFA2Smem(FlashDecodeMultiSeqStridedGQAKernel<T, GQARatio>, smem);
  dim3 grid(batch_size, num_kv_heads);
  FlashDecodeMultiSeqStridedGQAKernel<T, GQARatio>
      <<<grid, threads, smem, stream>>>(
          Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, num_heads, num_kv_heads,
          head_dim, slot_stride, layer_stride, kv_stride, scale);
  return cudaGetLastError();
}

// Warp-per-head decode entry: preferred for decode contexts where the
// full-KV scan by num_heads warps already saturates bandwidth (no split
// workspace, single launch). Shape-static for CUDA graphs.
template <typename T>
cudaError_t FlashDecodeWarpPerHead(const T *Q, const T *kv_buffer, T *O,
                                   const int *d_seq_ids, const int *d_kv_lens,
                                   int layer, int batch_size, int num_heads,
                                   int num_kv_heads, int head_dim,
                                   size_t slot_stride, size_t layer_stride,
                                   size_t kv_stride, float scale,
                                   cudaStream_t stream) {
  const int gqa_ratio = (num_kv_heads > 0 && num_heads > num_kv_heads)
                            ? (num_heads / num_kv_heads)
                            : 1;
  switch (gqa_ratio) {
  case 8:
    return LaunchGQADecodeWarpPerHead<T, 8>(
        Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, batch_size, num_heads,
        num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale,
        stream);
  case 4:
    return LaunchGQADecodeWarpPerHead<T, 4>(
        Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, batch_size, num_heads,
        num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale,
        stream);
  case 2:
    return LaunchGQADecodeWarpPerHead<T, 2>(
        Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, batch_size, num_heads,
        num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale,
        stream);
  default:
    return LaunchGQADecodeWarpPerHead<T, 1>(
        Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, batch_size, num_heads,
        num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale,
        stream);
  }
}

template <typename T>
cudaError_t FlashDecodeMultiSeqStrided(const T *Q, const T *kv_buffer, T *O,
                                       const int *d_seq_ids,
                                       const int *d_kv_lens, int layer,
                                       int batch_size, int num_heads,
                                       int num_kv_heads, int head_dim,
                                       size_t slot_stride, size_t layer_stride,
                                       size_t kv_stride, float scale,
                                       cudaStream_t stream,
                                       void *split_workspace,
                                       size_t split_workspace_bytes) {
  // Use GQA-grouped kernel when multiple Q-heads share a KV head.
  // This reduces KV memory reads by gqa_ratio (e.g., 4x for ratio=4).
  const int gqa_ratio = (num_kv_heads > 0 && num_heads > num_kv_heads)
                            ? (num_heads / num_kv_heads)
                            : 1;

  if (gqa_ratio == 8) {
    return LaunchGQADecode<T, 8>(Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer,
                                  batch_size, num_heads, num_kv_heads, head_dim,
                                  slot_stride, layer_stride, kv_stride, scale,
                                  stream, split_workspace,
                                  split_workspace_bytes);
  }
  if (gqa_ratio == 4) {
    return LaunchGQADecode<T, 4>(Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer,
                                  batch_size, num_heads, num_kv_heads, head_dim,
                                  slot_stride, layer_stride, kv_stride, scale,
                                  stream, split_workspace,
                                  split_workspace_bytes);
  }
  if (gqa_ratio == 2) {
    return LaunchGQADecode<T, 2>(Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer,
                                  batch_size, num_heads, num_kv_heads, head_dim,
                                  slot_stride, layer_stride, kv_stride, scale,
                                  stream, split_workspace,
                                  split_workspace_bytes);
  }

  // Fallback: non-GQA or unsupported ratio — use per-head kernel
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = min(threads, 1024);
  const int num_warps = threads / 32;
  int smem = ComputeFA2Smem(FA2_TILE_KV, head_dim, num_warps);

  ConfigureFA2Smem(FlashDecodeMultiSeqStridedKernel<T>, smem);

  dim3 grid(batch_size, num_heads);

  FlashDecodeMultiSeqStridedKernel<T><<<grid, threads, smem, stream>>>(
      Q, kv_buffer, O, d_seq_ids, d_kv_lens, layer, num_heads, num_kv_heads,
      head_dim, slot_stride, layer_stride, kv_stride, scale);
  return cudaGetLastError();
}

template cudaError_t FlashDecodeWarpPerHead<half>(
    const half *, const half *, half *, const int *, const int *, int, int,
    int, int, int, size_t, size_t, size_t, float, cudaStream_t);
template cudaError_t FlashDecodeWarpPerHead<float>(
    const float *, const float *, float *, const int *, const int *, int, int,
    int, int, int, size_t, size_t, size_t, float, cudaStream_t);
template cudaError_t FlashDecodeWarpPerHead<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *,
    const int *, const int *, int, int, int, int, int, size_t, size_t,
    size_t, float, cudaStream_t);

template cudaError_t FlashDecodeMultiSeqStrided<half>(
    const half *, const half *, half *, const int *, const int *, int, int,
    int, int, int, size_t, size_t, size_t, float, cudaStream_t, void *,
    size_t);
template cudaError_t FlashDecodeMultiSeqStrided<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, const int *,
    const int *, int, int, int, int, int, size_t, size_t, size_t, float,
    cudaStream_t, void *, size_t);

// ============================================================================
// FlashDecodeMultiSeqIndirect: Build K/V pointer arrays from slot_base_ptrs,
// then dispatch to existing FlashDecodeMultiSeq (pointer-based).
// Supports hybrid KV cache where slots may not be contiguous.
// ============================================================================

template <typename T>
__global__ void BuildKVPtrsFromSlotsKernel(
    T *const *__restrict__ slot_base_ptrs,
    const int *__restrict__ seq_ids, int layer, int batch_size,
    size_t layer_stride, size_t kv_stride,
    const T **__restrict__ out_k_ptrs, const T **__restrict__ out_v_ptrs) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;
  const T *slot = slot_base_ptrs[seq_ids[b]];
  const T *k_base = slot + static_cast<size_t>(layer) * layer_stride;
  out_k_ptrs[b] = k_base;
  out_v_ptrs[b] = k_base + kv_stride;
}

template <typename T>
cudaError_t FlashDecodeMultiSeqIndirect(
    const T *Q, T *const *slot_base_ptrs, T *O, const int *d_seq_ids,
    const int *d_kv_lens, int layer, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, size_t layer_stride, size_t kv_stride,
    float scale, cudaStream_t stream, void *ptr_workspace,
    size_t ptr_workspace_bytes) {
  // ptr_workspace must hold 2 * batch_size * sizeof(T*)
  const size_t required = 2 * static_cast<size_t>(batch_size) * sizeof(T *);
  if (!ptr_workspace || ptr_workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }

  const T **d_k_ptrs = static_cast<const T **>(ptr_workspace);
  const T **d_v_ptrs = d_k_ptrs + batch_size;

  // Build K/V pointer arrays from slot_base_ptrs
  const int threads = 64;
  const int blocks = (batch_size + threads - 1) / threads;
  BuildKVPtrsFromSlotsKernel<T><<<blocks, threads, 0, stream>>>(
      slot_base_ptrs, d_seq_ids, layer, batch_size, layer_stride, kv_stride,
      d_k_ptrs, d_v_ptrs);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return err;

  // Dispatch to existing pointer-based FlashDecode
  return FlashDecodeMultiSeq<T>(Q, d_k_ptrs, d_v_ptrs, O, d_kv_lens,
                                batch_size, num_heads, num_kv_heads, head_dim,
                                scale, stream);
}

template cudaError_t FlashDecodeMultiSeqIndirect<half>(
    const half *, half *const *, half *, const int *, const int *, int, int,
    int, int, int, size_t, size_t, float, cudaStream_t, void *, size_t);
template cudaError_t FlashDecodeMultiSeqIndirect<__nv_bfloat16>(
    const __nv_bfloat16 *, __nv_bfloat16 *const *, __nv_bfloat16 *,
    const int *, const int *, int, int, int, int, int, size_t, size_t, float,
    cudaStream_t, void *, size_t);

// ============================================================================
// Split decode attention: parallelizes the GQA kernel over BOTH the Q-heads
// sharing a KV head (qsplit — wins at short context where the KV working set
// is tiny but the monolithic kernel leaves 46 of 48 SMs idle) AND fixed KV
// chunks (ksplits — wins at long context where serial tile iteration
// dominates). Grid: (batch, num_kv_heads, qsplit * ksplits).
//
// Each CTA computes heads_per_cta = GQARatio/qsplit Q-heads over one KV
// chunk. With ksplits == 1 it writes normalized O directly; otherwise each
// CTA writes an unnormalized online-softmax partial {m, l, acc[head_dim]}
// and a combine pass performs the global merge.
//
// Partial layout: ((b * num_heads + head) * ksplits + chunk) * stride, with
// stride = 2 + head_dim floats ([0]=m, [1]=l, [2..]=acc).
// ============================================================================

constexpr int kFlashDecodeMaxQHeads = 8;

template <typename T, int GQARatio>
__global__ void FlashDecodeSplitStridedGQAKernel(
    const T *__restrict__ Q, const T *__restrict__ kv_buffer, T *__restrict__ O,
    float *__restrict__ partials, const int *__restrict__ seq_ids,
    const int *__restrict__ kv_lens, int layer, int num_heads, int num_kv_heads,
    int head_dim, size_t slot_stride, size_t layer_stride, size_t kv_stride,
    float scale, int qsplit, int chunk, int ksplits, int partial_stride) {
  const int b = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int split = blockIdx.z;
  const int q_group = split % qsplit;
  const int kv_chunk = split / qsplit;
  const int heads_per_cta = GQARatio / qsplit;
  const int head_base = kv_head_idx * GQARatio + q_group * heads_per_cta;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  const int kv_start = kv_chunk * chunk;
  const int kv_end = min(kv_len, kv_start + chunk);

  const size_t partial_base =
      (static_cast<size_t>(b) * num_heads + head_base) * ksplits + kv_chunk;

  if (kv_len <= 0 || kv_start >= kv_len) {
    // This CTA has no work. With a combine pass pending, emit the identity
    // partial (m=-inf, l=0, acc=0) so the merge never reads garbage.
    if (ksplits > 1) {
#pragma unroll
      for (int h = 0; h < GQARatio; ++h) {
        if (h >= heads_per_cta || head_base + h >= num_heads) {
          continue;
        }
        float *p = partials + (partial_base + h * ksplits) * partial_stride;
        if (threadIdx.x == 0) {
          p[0] = -INFINITY;
          p[1] = 0.0f;
        }
        if (d < head_dim) {
          p[2 + d] = 0.0f;
        }
      }
    }
    return;
  }

  // Q registers for this CTA's head subset. The loop is unrolled to the
  // compile-time GQARatio bound (guarded by heads_per_cta) so the arrays
  // stay in registers.
  float q_reg[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; ++h) {
    const int head_idx = head_base + h;
    if (h < heads_per_cta && d < head_dim && head_idx < num_heads) {
      q_reg[h] = DtypeTraits<T>::to_float(
          Q[b * num_heads * head_dim + head_idx * head_dim + d]);
    } else {
      q_reg[h] = 0.0f;
    }
  }

  const size_t seq_offset = static_cast<size_t>(seq_ids[b]) * slot_stride;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const T *K = kv_buffer + seq_offset + layer_offset;
  const T *V = K + kv_stride;
  const int kv_stride_elems = num_kv_heads * head_dim;

  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = s_v + FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + GQARatio * num_warps * FA2_TILE_KV;

  float row_max[GQARatio];
  float row_sum[GQARatio];
  float o_acc[GQARatio];
#pragma unroll
  for (int h = 0; h < GQARatio; ++h) {
    row_max[h] = -INFINITY;
    row_sum[h] = 0.0f;
    o_acc[h] = 0.0f;
  }

  for (int pos = kv_start; pos < kv_end; pos += FA2_TILE_KV) {
    const int tile_len = min(FA2_TILE_KV, kv_end - pos);
    const int total_elements = tile_len * head_dim;

    for (int i = d; i < total_elements; i += num_threads) {
      const int t = i / head_dim;
      const int dim = i % head_dim;
      const size_t kv_offset =
          static_cast<size_t>(pos + t) * static_cast<size_t>(kv_stride_elems) +
          static_cast<size_t>(kv_head_idx * head_dim + dim);
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    for (int t = 0; t < tile_len; ++t) {
#pragma unroll
      for (int h = 0; h < GQARatio; ++h) {
        if (h >= heads_per_cta) {
          break;
        }
        float partial =
            (d < head_dim) ? q_reg[h] * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        if (lane == 0) {
          s_warp_dots[(h * num_warps + warp_id) * FA2_TILE_KV + t] = partial;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < GQARatio; ++h) {
      if (h >= heads_per_cta) {
        break;
      }
      if (d < tile_len) {
        float dot_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
          dot_sum += s_warp_dots[(h * num_warps + w) * FA2_TILE_KV + d];
        }
        s_scores[h * FA2_TILE_KV + d] = dot_sum * scale;
      }
    }
    __syncthreads();

    for (int t = 0; t < tile_len; ++t) {
#pragma unroll
      for (int h = 0; h < GQARatio; ++h) {
        if (h >= heads_per_cta) {
          break;
        }
        const float score = s_scores[h * FA2_TILE_KV + t];
        const float new_max = fmaxf(row_max[h], score);
        const float rescale = expf(row_max[h] - new_max);
        const float exp_w = expf(score - new_max);

        if (d < head_dim) {
          o_acc[h] = o_acc[h] * rescale + exp_w * s_v[t * head_dim + d];
        }

        row_sum[h] = row_sum[h] * rescale + exp_w;
        row_max[h] = new_max;
      }
    }
    __syncthreads();
  }

  if (ksplits == 1) {
    if (d < head_dim) {
#pragma unroll
      for (int h = 0; h < GQARatio; ++h) {
        if (h >= heads_per_cta) {
          break;
        }
        const int head_idx = head_base + h;
        if (head_idx < num_heads) {
          O[b * num_heads * head_dim + head_idx * head_dim + d] =
              DtypeTraits<T>::from_float(
                  (row_sum[h] > 0.0f) ? (o_acc[h] / row_sum[h]) : 0.0f);
        }
      }
    }
    return;
  }

  // Unnormalized partial: combine pass rescales by exp(m_s - m*) and
  // normalizes by the global l.
#pragma unroll
  for (int h = 0; h < GQARatio; ++h) {
    if (h >= heads_per_cta || head_base + h >= num_heads) {
      continue;
    }
    float *p = partials + (partial_base + h * ksplits) * partial_stride;
    if (threadIdx.x == 0) {
      p[0] = row_max[h];
      p[1] = row_sum[h];
    }
    if (d < head_dim) {
      p[2 + d] = o_acc[h];
    }
  }
}

template <typename T>
__global__ void
FlashDecodeSplitCombineKernel(const float *__restrict__ partials,
                              T *__restrict__ O, int num_heads, int head_dim,
                              int ksplits, int partial_stride) {
  const int b = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = threadIdx.x;

  const size_t base = (static_cast<size_t>(b) * num_heads + head_idx) * ksplits;
  float m = -INFINITY;
  for (int s = 0; s < ksplits; ++s) {
    m = fmaxf(m, partials[(base + s) * partial_stride]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (int s = 0; s < ksplits; ++s) {
    const float *p = partials + (base + s) * partial_stride;
    const float w = (p[0] > -INFINITY && m > -INFINITY) ? expf(p[0] - m) : 0.0f;
    l += p[1] * w;
    if (d < head_dim) {
      acc += p[2 + d] * w;
    }
  }
  if (d < head_dim) {
    O[(static_cast<size_t>(b) * num_heads + head_idx) * head_dim + d] =
        DtypeTraits<T>::from_float((l > 0.0f) ? (acc / l) : 0.0f);
  }
}

template <typename T, int GQARatio>
static cudaError_t
LaunchSplitGQADecode(const T *Q, const T *kv_buffer, T *O, float *partials,
                     const int *d_seq_ids, const int *d_kv_lens, int layer,
                     int batch_size, int num_heads, int num_kv_heads,
                     int head_dim, size_t slot_stride, size_t layer_stride,
                     size_t kv_stride, float scale, int qsplit, int chunk,
                     int ksplits, cudaStream_t stream) {
  qsplit = max(1, min(qsplit, GQARatio));
  // qsplit must DIVIDE the ratio: heads_per_cta = ratio/qsplit covers
  // qsplit*heads_per_cta heads in total, and a non-divisor value would
  // silently leave the trailing Q-heads uncomputed.
  while (qsplit > 1 && GQARatio % qsplit != 0) {
    --qsplit;
  }
  const int heads_per_cta = GQARatio / qsplit;
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = min(threads, 1024);
  const int num_warps = threads / 32;
  // Arrays are declared at GQARatio width; only heads_per_cta rows are
  // touched but the allocation must cover the full declared layout.
  int smem = ComputeFA2SmemGQA(FA2_TILE_KV, head_dim, num_warps, GQARatio);

  ConfigureFA2Smem(FlashDecodeSplitStridedGQAKernel<T, GQARatio>, smem);

  const int partial_stride = 2 + head_dim;
  dim3 grid(batch_size, num_kv_heads, qsplit * ksplits);

  FlashDecodeSplitStridedGQAKernel<T, GQARatio>
      <<<grid, threads, smem, stream>>>(
          Q, kv_buffer, O, partials, d_seq_ids, d_kv_lens, layer, num_heads,
          num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride, scale,
          qsplit, chunk, ksplits, partial_stride);

  if (ksplits > 1) {
    FlashDecodeSplitCombineKernel<T>
        <<<dim3(batch_size, num_heads), threads, 0, stream>>>(
            partials, O, num_heads, head_dim, ksplits, partial_stride);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t FlashDecodeMultiSeqStridedSplit(
    const T *Q, const T *kv_buffer, T *O, float *partials, const int *d_seq_ids,
    const int *d_kv_lens, int layer, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, size_t slot_stride, size_t layer_stride,
    size_t kv_stride, float scale, int qsplit, int chunk, int ksplits,
    cudaStream_t stream) {
  const int gqa_ratio = (num_kv_heads > 0 && num_heads > num_kv_heads)
                            ? (num_heads / num_kv_heads)
                            : 1;
  // Fold qsplit into the ratio when there is no GQA grouping to split.
  if (gqa_ratio == 1) {
    qsplit = 1;
  }

  if (gqa_ratio == 8) {
    return LaunchSplitGQADecode<T, 8>(
        Q, kv_buffer, O, partials, d_seq_ids, d_kv_lens, layer, batch_size,
        num_heads, num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride,
        scale, qsplit, chunk, ksplits, stream);
  }
  if (gqa_ratio == 4) {
    return LaunchSplitGQADecode<T, 4>(
        Q, kv_buffer, O, partials, d_seq_ids, d_kv_lens, layer, batch_size,
        num_heads, num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride,
        scale, qsplit, chunk, ksplits, stream);
  }
  if (gqa_ratio == 2) {
    return LaunchSplitGQADecode<T, 2>(
        Q, kv_buffer, O, partials, d_seq_ids, d_kv_lens, layer, batch_size,
        num_heads, num_kv_heads, head_dim, slot_stride, layer_stride, kv_stride,
        scale, qsplit, chunk, ksplits, stream);
  }
  // Unsupported ratio — fall back to the monolithic dispatcher.
  return FlashDecodeMultiSeqStrided<T>(Q, kv_buffer, O, d_seq_ids, d_kv_lens,
                                       layer, batch_size, num_heads,
                                       num_kv_heads, head_dim, slot_stride,
                                       layer_stride, kv_stride, scale, stream);
}

template cudaError_t FlashDecodeMultiSeqStridedSplit<half>(
    const half *, const half *, half *, float *, const int *, const int *, int,
    int, int, int, int, size_t, size_t, size_t, float, int, int, int,
    cudaStream_t);
template cudaError_t FlashDecodeMultiSeqStridedSplit<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, float *,
    const int *, const int *, int, int, int, int, int, size_t, size_t, size_t,
    float, int, int, int, cudaStream_t);

} // namespace cuda_kernel
} // namespace inferflux
