#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// Number of warps per thread block for V1 GEMV kernels.
constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

// Dp4aS8, Vsubss4, LoadPackedInt32*, PackedProjectionGroupParams, and
// get_scale_min_k4 are now defined in quant_common.cuh (included above).


// Vectorized half→float load with sum-of-squares accumulation (for RmsNorm).
__device__ __forceinline__ float
LoadHalfToSmemSumSq(const half *__restrict__ src, float *__restrict__ dst,
                    int K, int tid) {
  float local_sum_sq = 0.0f;
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 h2 = reinterpret_cast<const __half2 *>(src)[i];
    float2 f2 = __half22float2(h2);
    local_sum_sq += f2.x * f2.x + f2.y * f2.y;
    dst[2 * i] = f2.x;
    dst[2 * i + 1] = f2.y;
  }
  return local_sum_sq;
}


// Vectorized in-place RmsNorm application with half2 norm weight loads.
__device__ __forceinline__ void
ApplyNormInPlace(float *__restrict__ sx, const half *__restrict__ norm_weight,
                 float rms, int K, int tid) {
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 nw = reinterpret_cast<const __half2 *>(norm_weight)[i];
    float2 nf = __half22float2(nw);
    sx[2 * i] = sx[2 * i] * rms * nf.x;
    sx[2 * i + 1] = sx[2 * i + 1] * rms * nf.y;
  }
}


//==============================================================================
// Q6_K dp4a GEMV (SM 6.1+)
//
// Q6_K values are 6-bit integers (-32..31) — fit perfectly in int8.
// Extraction: q6 = (ql_nibble | (qh_2bits << 4)) - 32
// Matches llama.cpp's vec_dot_q6_K_q8_1 approach.
//
// Work distribution: 32 lanes process Q6_K super-blocks (256 elements).
// Each lane reads int32 from ql (4 nibbles) + extracts qh bits.
// Per sub-pattern: 32 elements, with 2 scales (16 elements each).
//
// Shared memory: K bytes (int8 activations) + workspace floats.
//==============================================================================
__global__ void
fused_dequant_gemv_q6k_dp4a_packed(const block_q6_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int act_base = blk * QK_K;

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int x_lo = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + sub_base * 32 + e_base]);
    const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int x_hi = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);
    const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * activation_scale *
           (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
            static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q6k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[Outputs] = {};
  for (int blk = 0; blk < num_blocks; ++blk) {
    const int act_base = blk * QK_K;
    const int x_lo = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + sub_base * 32 + e_base]);
    const int x_hi = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q6_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      const int vl_lo = ql4 & 0x0F0F0F0F;
      const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[i] +=
          d * activation_scale *
          (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
           static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}


__global__ void
fused_dequant_gemv_q4k_dp4a_packed(const block_q4_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const int base = blk * QK_K + pair * 64;
    const int x_lo4 = *reinterpret_cast<const int *>(&x_row[base + offs]);
    const int x_hi4 = *reinterpret_cast<const int *>(&x_row[base + 32 + offs]);

    const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
    const int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
    const int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

    acc += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                               dm_m_lo * static_cast<float>(sum_lo) +
                               d_sc_hi * static_cast<float>(dot_hi) -
                               dm_m_hi * static_cast<float>(sum_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[Outputs] = {};
  for (int blk = 0; blk < num_blocks; ++blk) {
    const int base = blk * QK_K + pair * 64;
    const int x_lo4 = *reinterpret_cast<const int *>(&x_row[base + offs]);
    const int x_hi4 = *reinterpret_cast<const int *>(&x_row[base + 32 + offs]);

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q4_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
      const int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
      const int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

      acc[i] += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                                    dm_m_lo * static_cast<float>(sum_lo) +
                                    d_sc_hi * static_cast<float>(dot_hi) -
                                    dm_m_hi * static_cast<float>(sum_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

__global__ void
fused_dequant_gemv_q8_0_dp4a_packed(const block_q8_0 *__restrict__ weight,
                                    const int8_t *__restrict__ x_q,
                                    const float *__restrict__ row_scales,
                                    half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_0 &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    int int_acc = 0;
    for (int j = 0; j < QK8_0; j += 4) {
      const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
      const int x4 = LoadPackedInt32Aligned(&x_row[blk * QK8_0 + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q8_0_dp4a_packed_group(
    PackedProjectionGroupParams<block_q8_0, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK8_0;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q8_0 *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_0 &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      int int_acc = 0;
      for (int j = 0; j < QK8_0; j += 4) {
        const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
        const int x4 = LoadPackedInt32Aligned(&x_row[blk * QK8_0 + j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d * static_cast<float>(int_acc) * activation_scale;
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

__global__ void
fused_dequant_gemv_q8k_dp4a_packed(const block_q8_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_k &b = wrow[blk];
    const float d = b.d;
    int int_acc = 0;
    for (int j = 0; j < QK_K; j += 4) {
      const int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      const int x4 = *reinterpret_cast<const int *>(&x_row[blk * QK_K + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q8k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q8_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q8_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_k &b = wrow[blk];
      const float d = b.d;
      int int_acc = 0;
      for (int j = 0; j < QK_K; j += 4) {
        const int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
        const int x4 = *reinterpret_cast<const int *>(&x_row[blk * QK_K + j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d * static_cast<float>(int_acc) * activation_scale;
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

//==============================================================================
// Q8_1 Activation Quantization Kernels
//
// Quantize FP16 activations to block_q8_1 format:
//   - Per-32-element block scale (d = max|x| / 127)
//   - Pre-computed d*sum(qs) for Q4_K dmin correction
//   - Stored in global memory, reused across sibling projections via L2 cache
//==============================================================================

// Standalone FP16 → Q8_1 quantization (for O proj, down proj inputs)
// Grid: (M,)  Block: (kGemvThreadsPerBlock,)
__global__ void quantize_row_q8_1_kernel(const half *__restrict__ input,
                                         block_q8_1 *__restrict__ output,
                                         int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int num_blocks = K / QK8_1;

  const half *x_row = input + row * K;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;

    float val0 = __half2float(x_row[blk0 * QK8_1 + lane]);
    float val1 =
        has_blk1 ? __half2float(x_row[blk1 * QK8_1 + lane]) : 0.0f;

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

__global__ void silu_mul_quantize_q8_1_kernel(const half *__restrict__ gate,
                                              const half *__restrict__ up,
                                              block_q8_1 *__restrict__ output,
                                              int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int num_blocks = K / QK8_1;

  const half *gate_row = gate + row * K;
  const half *up_row = up + row * K;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;
    const int idx0 = blk0 * QK8_1 + lane;
    const float g0 = __half2float(gate_row[idx0]);
    const float u0 = __half2float(up_row[idx0]);
    const float val0 = (g0 / (1.0f + expf(-g0))) * u0;

    float val1 = 0.0f;
    if (has_blk1) {
      const int idx1 = blk1 * QK8_1 + lane;
      const float g1 = __half2float(gate_row[idx1]);
      const float u1 = __half2float(up_row[idx1]);
      val1 = (g1 / (1.0f + expf(-g1))) * u1;
    }

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

// Fused RmsNorm + Q8_1 quantization: residual → normalized → Q8_1 blocks
// Eliminates standalone RmsNorm kernel while producing Q8_1 for reuse.
// Grid: (M,)  Block: (kGemvThreadsPerBlock,)
// Smem: K * sizeof(float) + kGemvWarpsPerBlock * sizeof(float)
__global__ void fused_rmsnorm_quantize_q8_1_kernel(
    const half *__restrict__ residual, const half *__restrict__ norm_weight,
    block_q8_1 *__restrict__ output, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  // Phase 1: Load residual → FP32 smem, compute sum-of-squares
  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  // Phase 2: Apply norm weights in-place (vectorized half2 norm loads)
  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  // Phase 3: Quantize normalized values to Q8_1 blocks
  const int num_blocks = K / QK8_1;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
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

//==============================================================================
// Q8_1 Activation GEMV Kernels
//
// Read pre-quantized Q8_1 activations from global memory (L2 cached).
// No shared memory needed for activations — all smem freed for occupancy.
template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_group(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
    int x_lo4;
    int x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));
    float s_lo = __half2float(__high2half(a_lo.ds));
    float s_hi = __half2float(__high2half(a_hi.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      int q_lo4 = qs4 & 0x0F0F0F0F;
      int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

      acc[i] +=
          d * static_cast<float>(sc_lo) * d8_lo * static_cast<float>(dot_lo) +
          d * static_cast<float>(sc_hi) * d8_hi * static_cast<float>(dot_hi);

      if ((lane & 7) == 0) {
        acc[i] -= dmin * static_cast<float>(m_lo) * s_lo +
                  dmin * static_cast<float>(m_hi) * s_hi;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_group_rowquad(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row_base = blockIdx.y * 4;
  if (row_base >= total_rows)
    return;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_rows = min(4, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_rows[4] = {
      act_q8_1 + row_base * num_q8_per_row,
      num_rows > 1 ? (act_q8_1 + (row_base + 1) * num_q8_per_row) : nullptr,
      num_rows > 2 ? (act_q8_1 + (row_base + 2) * num_q8_per_row) : nullptr,
      num_rows > 3 ? (act_q8_1 + (row_base + 3) * num_q8_per_row) : nullptr,
  };

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[4][Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = a_rows[r][blk * 8 + pair * 2];
        const block_q8_1 &a_hi = a_rows[r][blk * 8 + pair * 2 + 1];
        int x_lo4;
        int x_hi4;
        memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
        memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
        const float d8_lo = __half2float(__low2half(a_lo.ds));
        const float d8_hi = __half2float(__low2half(a_hi.ds));
        const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
        const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

        acc[r][i] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                     d_sc_hi * d8_hi * static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          acc[r][i] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][i] += __shfl_down_sync(0xFFFFFFFF, acc[r][i], offset);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= num_rows)
        continue;
#pragma unroll
      for (int i = 0; i < Outputs; ++i) {
        if (out_idx < params.output_cols[i]) {
          params.outputs[i][(row_base + r) * params.output_cols[i] + out_idx] =
              __float2half(acc[r][i]);
        }
      }
    }
  }
}
template <int Outputs>
__global__ void fused_dequant_gemv_q6k_q8_1_group(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];
    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q6_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      int vl_lo = ql4 & 0x0F0F0F0F;
      int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[i] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                         static_cast<float>(dot_lo) +
                     static_cast<float>(b.scales[sc_hi]) * d8_hi *
                         static_cast<float>(dot_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}



template <int Outputs>
__global__ void fused_dequant_gemv_q8_0_q8_1_group(
    PackedProjectionGroupParams<block_q8_0, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_blocks = K / QK8_0;
  const block_q8_1 *a_row = act_q8_1 + row * num_blocks;

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_1 &a = a_row[blk];
    const float d_a = __half2float(__low2half(a.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q8_0 *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_0 &b = wrow[blk];
      const float d_w = __half2float(*reinterpret_cast<const half *>(&b.d));
      int int_acc = 0;
      for (int j = 0; j < QK8_0; j += 4) {
        const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
        const int x4 = LoadPackedInt32Unaligned(&a.qs[j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d_w * d_a * static_cast<float>(int_acc);
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}


template <int Outputs>
__global__ void fused_dequant_gemv_q8k_q8_1_group(
    PackedProjectionGroupParams<block_q8_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_super_blocks; blk += 32) {
    for (int sub = 0; sub < 8; ++sub) {
      const block_q8_1 &a = a_row[blk * 8 + sub];
      const float d_a = __half2float(__low2half(a.ds));

#pragma unroll
      for (int i = 0; i < Outputs; ++i) {
        if (out_idx >= params.output_cols[i])
          continue;

        const block_q8_k *wrow = params.weights[i] + out_idx * num_super_blocks;
        const block_q8_k &b = wrow[blk];
        const float d_w = b.d;
        int int_acc = 0;
        for (int j = 0; j < QK8_1; j += 4) {
          int w4 = *reinterpret_cast<const int *>(&b.qs[sub * QK8_1 + j]);
          int x4;
          memcpy(&x4, &a.qs[j], sizeof(x4));
          int_acc = Dp4aS8(w4, x4, int_acc);
        }
        acc[i] += d_w * d_a * static_cast<float>(int_acc);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
