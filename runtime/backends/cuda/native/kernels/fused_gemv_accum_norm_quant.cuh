#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// FinishNormQuantQ8_1: Fused RmsNorm + Q8_1 quantization after GEMV accum.
//
// Matches the production fused_rmsnorm_quantize_q8_1_kernel architecture:
//   Phase 1: Load FP16 residual → FP32 shared memory, sum-of-squares
//   Phase 2: Apply norm weights in-place in shared memory (FP32)
//   Phase 3: Quantize from FP32 shared memory → Q8_1 blocks
//
// Key: NO FP16 intermediate — quantization reads from FP32 shared memory
// directly, matching the production kernel's precision path.
//
// Uses 8 warps (256 threads) per block, K floats + 8 floats in shared memory.
// Grid: M blocks (one per row).
// ============================================================================

using namespace inferflux::runtime::cuda::native;

constexpr int kFinishNormWarps = 8;
constexpr int kFinishNormThreads = kFinishNormWarps * 32; // 256

__global__ void FinishNormQuantQ8_1Kernel(
    const half *__restrict__ residual,    // [M, K] input (post-accumulate)
    const half *__restrict__ norm_weight, // [K] RmsNorm weight
    half *__restrict__ norm_output,       // [M, K] normalized FP16 output
    void *__restrict__ act_q8_1_out,      // [M, K/32] Q8_1 output blocks
    int M, int K, float rms_norm_eps) {
  // Shared memory layout: [K floats for values] + [kFinishNormWarps floats for
  // warp sums]
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int row = blockIdx.x;
  if (row >= M)
    return;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const half *x = residual + row * K;

  // Phase 1: Load FP16 residual → FP32 smem, compute sum-of-squares
  // Match production kernel's half2 vectorized load + sum pattern.
  const int K2 = K / 2;
  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  float local_sum_sq = 0.0f;
  for (int i = tid; i < K2; i += kFinishNormThreads) {
    half2 v = x2[i];
    float f0 = __half2float(__low2half(v));
    float f1 = __half2float(__high2half(v));
    sx[2 * i] = f0;
    sx[2 * i + 1] = f1;
    local_sum_sq += f0 * f0 + f1 * f1;
  }
  // Warp-level reduction for sum-of-squares
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  // Final reduction by thread 0
  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kFinishNormWarps; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + rms_norm_eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  // Phase 2: Apply norm weights in-place in shared memory (FP32)
  // Also write FP16 output for downstream consumers that read FP16.
  const int K2n = K / 2;
  const half2 *nw2 = reinterpret_cast<const half2 *>(norm_weight);
  half *y = norm_output + row * K;
  for (int i = tid; i < K2n; i += kFinishNormThreads) {
    half2 w = nw2[i];
    float w0 = __half2float(__low2half(w));
    float w1 = __half2float(__high2half(w));
    float v0 = sx[2 * i] * rms * w0;
    float v1 = sx[2 * i + 1] * rms * w1;
    sx[2 * i] = v0;
    sx[2 * i + 1] = v1;
    // Write FP16 output
    y[2 * i] = __float2half(v0);
    y[2 * i + 1] = __float2half(v1);
  }
  __syncthreads();

  // Phase 3: Quantize from FP32 shared memory → Q8_1 blocks
  // Matches production kernel's dual-block warp processing.
  const int num_blocks = K / QK8_1;
  auto *q8_out = reinterpret_cast<block_q8_1 *>(act_q8_1_out);
  block_q8_1 *out_row = q8_out + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kFinishNormWarps * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;

    float val0 = sx[blk0 * QK8_1 + lane];
    float val1 = has_blk1 ? sx[blk1 * QK8_1 + lane] : 0.0f;

    // Warp-level amax reduction
    float amax0 = fabsf(val0);
    float amax1 = fabsf(val1);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
    }

    float d0 = amax0 / 127.0f;
    float inv_d0 = (amax0 > 0.0f) ? 127.0f / amax0 : 0.0f;
    int q0 = __float2int_rn(val0 * inv_d0);
    q0 = max(-128, min(127, q0));
    int sum0 = q0;

    float d1 = amax1 / 127.0f;
    float inv_d1 = (amax1 > 0.0f) ? 127.0f / amax1 : 0.0f;
    int q1 = has_blk1 ? __float2int_rn(val1 * inv_d1) : 0;
    q1 = max(-128, min(127, q1));
    int sum1 = q1;

    // Warp-level sum reduction for Q8_1 sum field
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    out_row[blk0].qs[lane] = static_cast<int8_t>(q0);
    if (lane == 0) {
      out_row[blk0].ds = __halves2half2(
          __float2half(d0), __float2half(d0 * static_cast<float>(sum0)));
    }
    if (has_blk1) {
      out_row[blk1].qs[lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        out_row[blk1].ds = __halves2half2(
            __float2half(d1), __float2half(d1 * static_cast<float>(sum1)));
      }
    }
  }
}

// RmsNormMixedQuantQ8_1: identical to FinishNormQuantQ8_1 but reads an
// FP32 residual (the production default, enable_fp32_residual) — folds
// the RmsNormMixed + QuantizeRowQ8_1 launch pair the fp32 path paid at
// three sites per layer-step.
__global__ void RmsNormMixedQuantQ8_1Kernel(
    const float *__restrict__ residual, // [M, K] FP32 input
    const half *__restrict__ norm_weight, half *__restrict__ norm_output,
    void *__restrict__ act_q8_1_out, int M, int K, float rms_norm_eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int row = blockIdx.x;
  if (row >= M)
    return;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  // Phase 1: FP32 residual → smem + sum-of-squares (float loads).
  float local_sum_sq = 0.0f;
  for (int i = tid; i < K; i += kFinishNormThreads) {
    const float f0 = residual[row * K + i];
    sx[i] = f0;
    local_sum_sq += f0 * f0;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kFinishNormWarps; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + rms_norm_eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  // Phase 2: norm weights + FP16 output.
  const int K2n = K / 2;
  const half2 *nw2 = reinterpret_cast<const half2 *>(norm_weight);
  half *y = norm_output + row * K;
  for (int i = tid; i < K2n; i += kFinishNormThreads) {
    const half2 w = nw2[i];
    const float w0 = __half2float(__low2half(w));
    const float w1 = __half2float(__high2half(w));
    const float v0 = sx[2 * i] * rms * w0;
    const float v1 = sx[2 * i + 1] * rms * w1;
    sx[2 * i] = v0;
    sx[2 * i + 1] = v1;
    y[2 * i] = __float2half(v0);
    y[2 * i + 1] = __float2half(v1);
  }
  __syncthreads();

  // Phase 3: quantize from FP32 smem → Q8_1 (dual-block per warp).
  const int num_blocks = K / QK8_1;
  auto *q8_out = reinterpret_cast<block_q8_1 *>(act_q8_1_out);
  block_q8_1 *out_row = q8_out + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kFinishNormWarps * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;

    float val0 = sx[blk0 * QK8_1 + lane];
    float val1 = has_blk1 ? sx[blk1 * QK8_1 + lane] : 0.0f;

    float amax0 = fabsf(val0);
    float amax1 = fabsf(val1);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
    }

    float d0 = amax0 / 127.0f;
    float inv_d0 = (amax0 > 0.0f) ? 127.0f / amax0 : 0.0f;
    int q0 = __float2int_rn(val0 * inv_d0);
    q0 = max(-128, min(127, q0));
    int sum0 = q0;

    float d1 = amax1 / 127.0f;
    float inv_d1 = (amax1 > 0.0f) ? 127.0f / amax1 : 0.0f;
    int q1 = has_blk1 ? __float2int_rn(val1 * inv_d1) : 0;
    q1 = max(-128, min(127, q1));
    int sum1 = q1;

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    out_row[blk0].qs[lane] = static_cast<int8_t>(q0);
    if (lane == 0) {
      out_row[blk0].ds = __halves2half2(
          __float2half(d0), __float2half(d0 * static_cast<float>(sum0)));
    }
    if (has_blk1) {
      out_row[blk1].qs[lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        out_row[blk1].ds = __halves2half2(
            __float2half(d1), __float2half(d1 * static_cast<float>(sum1)));
      }
    }
  }
}

/// Host API: FP32-residual fused RmsNorm + FP16 out + Q8_1 quantize.
inline cudaError_t RmsNormMixedQuantQ8_1(const float *residual,
                                         const half *norm_weight,
                                         half *norm_output,
                                         void *act_q8_1_out, int M, int K,
                                         float rms_norm_eps,
                                         cudaStream_t stream = 0) {
  const size_t smem = static_cast<size_t>(K) * sizeof(float) +
                      kFinishNormWarps * sizeof(float);
  RmsNormMixedQuantQ8_1Kernel<<<M, kFinishNormThreads, smem, stream>>>(
      residual, norm_weight, norm_output, act_q8_1_out, M, K, rms_norm_eps);
  return cudaGetLastError();
}

/// Host API: Launch FinishNormQuantQ8_1 after GEMV accumulate.
/// Uses 256 threads (8 warps) with K + 8 floats of shared memory,
/// matching the production fused_rmsnorm_quantize_q8_1_kernel.
inline cudaError_t FinishNormQuantQ8_1(const half *residual,
                                       const half *norm_weight,
                                       half *norm_output,
                                       void *act_q8_1_out, int M, int K,
                                       float rms_norm_eps,
                                       cudaStream_t stream) {
  const int threads = kFinishNormThreads;
  const int smem =
      static_cast<int>((K + kFinishNormWarps) * sizeof(float));

  FinishNormQuantQ8_1Kernel<<<M, threads, smem, stream>>>(
      residual, norm_weight, norm_output, act_q8_1_out, M, K, rms_norm_eps);
  return cudaGetLastError();
}

} // namespace cuda_kernel
} // namespace inferflux
