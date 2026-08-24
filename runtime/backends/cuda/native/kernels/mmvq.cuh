#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// ============================================================================
// MMVQ (Matrix-Matrix Vector-Quantized) Kernels
//
// Weight-read-first architecture: weights are read ONCE from global memory and
// amortized across all batch columns. This is the critical structural fix over
// V1/V2 GEMV kernels which re-read weights B times (once per batch element).
//
// For quantized LLM inference, weights are ~280x larger than activations per
// projection. The outer loop MUST iterate over weight blocks along K, with an
// inner loop over batch columns that dot-products against the SAME weight data.
//
// Block: 128 threads = 4 warps (same as V2 cooperative-warp)
// Grid:  dim3(N, (M + ncols - 1) / ncols)
//   - blockIdx.x: output row index (one block per weight row)
//   - blockIdx.y: batch tile index
// Smem:  float warp_sums[4 * ncols] (tiny: 16-128 bytes)
// ============================================================================

constexpr int kMmvqWarps = 4;
constexpr int kMmvqThreads = kMmvqWarps * 32; // 128

// Thread block configuration: always use kMmvqWarps warps.
// The kernel loop stride and shared memory reduction are hard-coded to
// kMmvqWarps, so the launch must match.  Adaptive warp counts would
// require a template parameter for nwarps throughout the kernel.
constexpr int calc_mmvq_warps(int /*ncols*/) { return kMmvqWarps; }

constexpr int calc_mmvq_threads(int ncols) {
  return calc_mmvq_warps(ncols) * 32;
}

// ============================================================================
// Q4_K × Q8_1 MMVQ kernel
// ============================================================================

template <int ncols>
__global__ void inferflux_mmvq_q4k(const block_q4_k *__restrict__ weight,
                                   const block_q8_1 *__restrict__ act_q8_1,
                                   half *__restrict__ output, int N, int K,
                                   int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  const int group_leader = lane & ~7;

  float acc[ncols] = {};

  // Outer loop: weight blocks along K dimension (weights read ONCE)
  for (int blk = warp_id; blk < num_super_blocks; blk += kMmvqWarps) {
    // Load weight data ONCE from global memory (use __ldg for read-only cache)
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));
    const float dmin = __half2float(__ldg(reinterpret_cast<const half *>(&b.dmin)));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    if ((lane & 7) == 0) {
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
    }
    sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
    m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
    sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
    m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    int qs4 = __ldg(reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]));
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    // Inner loop: batch columns (activation reads, reusing weight data)
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;
      const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];

      const int x_lo4 = *reinterpret_cast<const int *>(&a_lo.qs[offs]);
      const int x_hi4 = *reinterpret_cast<const int *>(&a_hi.qs[offs]);

      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));

      int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

      acc[c] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                d_sc_hi * d8_hi * static_cast<float>(dot_hi);

      if ((lane & 7) == 0) {
        float s_lo = __half2float(__high2half(a_lo.ds));
        float s_hi = __half2float(__high2half(a_hi.ds));
        acc[c] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }
  }

  // Intra-warp reduction for each column
#pragma unroll
  for (int c = 0; c < ncols; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[c] += __shfl_down_sync(0xFFFFFFFF, acc[c], offset);
    }
  }

  // Cross-warp reduction via shared memory
  __shared__ float warp_sums[kMmvqWarps * ncols];
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      warp_sums[c * kMmvqWarps + warp_id] = acc[c];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;
      float sum = 0.0f;
      for (int w = 0; w < kMmvqWarps; ++w)
        sum += warp_sums[c * kMmvqWarps + w];
      output[row * N + out_idx] = __float2half(sum);
    }
  }
}

// ============================================================================
// Q6_K × Q8_1 MMVQ kernel
// ============================================================================

template <int ncols>
__global__ void inferflux_mmvq_q6k(const block_q6_k *__restrict__ weight,
                                   const block_q8_1 *__restrict__ act_q8_1,
                                   half *__restrict__ output, int N, int K,
                                   int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[ncols] = {};

  for (int blk = warp_id; blk < num_super_blocks; blk += kMmvqWarps) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    int vl_lo = ql4 & 0x0F0F0F0F;
    int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);

    int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;
      const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];

      const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
      const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);

      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));

      int dot_lo = Dp4aS8(vi_lo, x_lo, 0);
      int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[c] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                          static_cast<float>(dot_lo) +
                      static_cast<float>(b.scales[sc_hi]) * d8_hi *
                          static_cast<float>(dot_hi));
    }
  }

  // Intra-warp reduction
#pragma unroll
  for (int c = 0; c < ncols; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[c] += __shfl_down_sync(0xFFFFFFFF, acc[c], offset);
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kMmvqWarps * ncols];
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      warp_sums[c * kMmvqWarps + warp_id] = acc[c];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;
      float sum = 0.0f;
      for (int w = 0; w < kMmvqWarps; ++w)
        sum += warp_sums[c * kMmvqWarps + w];
      output[row * N + out_idx] = __float2half(sum);
    }
  }
}

// ============================================================================
// Q8_0 × Q8_1 MMVQ kernel
// ============================================================================

template <int ncols>
__global__ void inferflux_mmvq_q8_0(const block_q8_0 *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K,
                                    int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;
  if (out_idx >= N)
    return;

  // Q8_0 and Q8_1 both have 32-element blocks: 1:1 mapping
  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;

  float acc[ncols] = {};

  // Each warp handles strided blocks. Weight data read once per block.
  for (int blk = warp_id * 32 + lane; blk < num_blocks;
       blk += kMmvqWarps * 32) {
    const block_q8_0 &b = wrow[blk];
    const float d_w = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));

    // Pre-load weight quants
    int w4[QK8_0 / 4];
    for (int j = 0; j < QK8_0; j += 4) {
      w4[j / 4] = LoadPackedInt32Unaligned(&b.qs[j]);
    }

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_blocks;
      const block_q8_1 &a = a_row[blk];
      const float d_a = __half2float(__low2half(a.ds));

      int int_acc = 0;
      for (int j = 0; j < QK8_0; j += 4) {
        const int x4 = *reinterpret_cast<const int *>(&a.qs[j]);
        int_acc = Dp4aS8(w4[j / 4], x4, int_acc);
      }
      acc[c] += d_w * d_a * static_cast<float>(int_acc);
    }
  }

  // Intra-warp reduction
#pragma unroll
  for (int c = 0; c < ncols; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[c] += __shfl_down_sync(0xFFFFFFFF, acc[c], offset);
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kMmvqWarps * ncols];
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      warp_sums[c * kMmvqWarps + warp_id] = acc[c];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;
      float sum = 0.0f;
      for (int w = 0; w < kMmvqWarps; ++w)
        sum += warp_sums[c * kMmvqWarps + w];
      output[row * N + out_idx] = __float2half(sum);
    }
  }
}

// ============================================================================
// Q8_K × Q8_1 MMVQ kernel
// ============================================================================

template <int ncols>
__global__ void inferflux_mmvq_q8k(const block_q8_k *__restrict__ weight,
                                   const block_q8_1 *__restrict__ act_q8_1,
                                   half *__restrict__ output, int N, int K,
                                   int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;
  if (out_idx >= N)
    return;

  // Q8_K super-block = 256 elements = 8 Q8_1 blocks
  const int num_super_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;

  float acc[ncols] = {};

  for (int blk = warp_id * 32 + lane; blk < num_super_blocks;
       blk += kMmvqWarps * 32) {
    const block_q8_k &b = wrow[blk];
    const float d_w = __ldg(&b.d);

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

      for (int sub = 0; sub < 8; ++sub) {
        const block_q8_1 &a = a_row[blk * 8 + sub];
        const float d_a = __half2float(__low2half(a.ds));

        int int_acc = 0;
        for (int j = 0; j < QK8_1; j += 4) {
          int w4 = __ldg(reinterpret_cast<const int *>(&b.qs[sub * QK8_1 + j]));
          const int x4 = *reinterpret_cast<const int *>(&a.qs[j]);
          int_acc = Dp4aS8(w4, x4, int_acc);
        }
        acc[c] += d_w * d_a * static_cast<float>(int_acc);
      }
    }
  }

  // Intra-warp reduction
#pragma unroll
  for (int c = 0; c < ncols; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[c] += __shfl_down_sync(0xFFFFFFFF, acc[c], offset);
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kMmvqWarps * ncols];
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      warp_sums[c * kMmvqWarps + warp_id] = acc[c];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;
      float sum = 0.0f;
      for (int w = 0; w < kMmvqWarps; ++w)
        sum += warp_sums[c * kMmvqWarps + w];
      output[row * N + out_idx] = __float2half(sum);
    }
  }
}

// ============================================================================
// Grouped MMVQ kernels (2 or 3 outputs sharing activation reads)
//
// Used for FFN gate+up (2 outputs) and Q+K+V (3 outputs).
// Weights for each projection are read independently, but all projections
// share the same activation load per K-block per batch column.
// ============================================================================

template <int ncols, int nprojs>
__global__ void inferflux_mmvq_q4k_group(
    PackedProjectionGroupParams<block_q4_k, nprojs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < nprojs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  const int group_leader = lane & ~7;

  float acc[nprojs][ncols] = {};

  for (int blk = warp_id; blk < num_super_blocks; blk += kMmvqWarps) {
    // Inner loop over batch columns — load activation once per column
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;
      const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
      const int x_lo4 = *reinterpret_cast<const int *>(&a_lo.qs[offs]);
      const int x_hi4 = *reinterpret_cast<const int *>(&a_hi.qs[offs]);
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));
      float s_lo = __half2float(__high2half(a_lo.ds));
      float s_hi = __half2float(__high2half(a_hi.ds));

      // Loop over projections — each reads its own weight row
#pragma unroll
      for (int p = 0; p < nprojs; ++p) {
        if (out_idx >= params.output_cols[p])
          continue;

        const block_q4_k *wrow =
            params.weights[p] + out_idx * num_super_blocks;
        const block_q4_k &b = wrow[blk];
        const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));
        const float dmin =
            __half2float(__ldg(reinterpret_cast<const half *>(&b.dmin)));

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        if ((lane & 7) == 0) {
          get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
          get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
        }
        sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
        m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
        sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
        m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

        int qs4 = __ldg(reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]));
        int q_lo4 = qs4 & 0x0F0F0F0F;
        int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

        int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
        int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

        acc[p][c] +=
            d * static_cast<float>(sc_lo) * d8_lo *
                static_cast<float>(dot_lo) +
            d * static_cast<float>(sc_hi) * d8_hi *
                static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          acc[p][c] -= dmin * static_cast<float>(m_lo) * s_lo +
                       dmin * static_cast<float>(m_hi) * s_hi;
        }
      }
    }
  }

  // Intra-warp reduction
#pragma unroll
  for (int p = 0; p < nprojs; ++p) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[p][c] += __shfl_down_sync(0xFFFFFFFF, acc[p][c], offset);
      }
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kMmvqWarps * nprojs * ncols];
  if (lane == 0) {
#pragma unroll
    for (int p = 0; p < nprojs; ++p) {
#pragma unroll
      for (int c = 0; c < ncols; ++c) {
        warp_sums[(p * ncols + c) * kMmvqWarps + warp_id] = acc[p][c];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < nprojs; ++p) {
      if (out_idx >= params.output_cols[p])
        continue;
#pragma unroll
      for (int c = 0; c < ncols; ++c) {
        const int row = col_base + c;
        if (row >= M)
          break;
        float sum = 0.0f;
        for (int w = 0; w < kMmvqWarps; ++w)
          sum += warp_sums[(p * ncols + c) * kMmvqWarps + w];
        params.outputs[p][row * params.output_cols[p] + out_idx] =
            __float2half(sum);
      }
    }
  }
}

template <int ncols, int nprojs>
__global__ void inferflux_mmvq_q6k_group(
    PackedProjectionGroupParams<block_q6_k, nprojs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < nprojs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[nprojs][ncols] = {};

  for (int blk = warp_id; blk < num_super_blocks; blk += kMmvqWarps) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;
      const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];
      const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
      const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));

#pragma unroll
      for (int p = 0; p < nprojs; ++p) {
        if (out_idx >= params.output_cols[p])
          continue;

        const block_q6_k *wrow =
            params.weights[p] + out_idx * num_super_blocks;
        const block_q6_k &b = wrow[blk];
        const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));

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

        acc[p][c] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                              static_cast<float>(dot_lo) +
                          static_cast<float>(b.scales[sc_hi]) * d8_hi *
                              static_cast<float>(dot_hi));
      }
    }
  }

  // Intra-warp reduction
#pragma unroll
  for (int p = 0; p < nprojs; ++p) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[p][c] += __shfl_down_sync(0xFFFFFFFF, acc[p][c], offset);
      }
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kMmvqWarps * nprojs * ncols];
  if (lane == 0) {
#pragma unroll
    for (int p = 0; p < nprojs; ++p) {
#pragma unroll
      for (int c = 0; c < ncols; ++c) {
        warp_sums[(p * ncols + c) * kMmvqWarps + warp_id] = acc[p][c];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < nprojs; ++p) {
      if (out_idx >= params.output_cols[p])
        continue;
#pragma unroll
      for (int c = 0; c < ncols; ++c) {
        const int row = col_base + c;
        if (row >= M)
          break;
        float sum = 0.0f;
        for (int w = 0; w < kMmvqWarps; ++w)
          sum += warp_sums[(p * ncols + c) * kMmvqWarps + w];
        params.outputs[p][row * params.output_cols[p] + out_idx] =
            __float2half(sum);
      }
    }
  }
}

// ============================================================================
// MMVQ dispatch helpers
//
// Select ncols template at runtime based on batch size M.
// ============================================================================

template <typename BlockT,
          void (*Kernel1)(const BlockT *, const block_q8_1 *, half *, int, int,
                          int),
          void (*Kernel2)(const BlockT *, const block_q8_1 *, half *, int, int,
                          int),
          void (*Kernel4)(const BlockT *, const block_q8_1 *, half *, int, int,
                          int),
          void (*Kernel8)(const BlockT *, const block_q8_1 *, half *, int, int,
                          int)>
bool DispatchMmvq(const void *data, const void *act_q8_1, half *output, int M,
                  int N, int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockT *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);

  auto launch = [&](int ncols, auto kernel) {
    dim3 grid(N, (M + ncols - 1) / ncols);
    size_t smem = sizeof(float) * kMmvqWarps * ncols;
    const int threads = calc_mmvq_threads(ncols);
    kernel<<<grid, threads, smem, stream>>>(w, a, output, N, K, M);
    return true;
  };

  if (M <= 1)
    return launch(1, Kernel1);
  if (M <= 2)
    return launch(2, Kernel2);
  if (M <= 4)
    return launch(4, Kernel4);
  if (M <= 8)
    return launch(8, Kernel8);
  return false; // M > 8 → MMQ or cuBLAS
}

template <typename BlockT, int nprojs,
          void (*Kernel1)(PackedProjectionGroupParams<BlockT, nprojs>,
                          const block_q8_1 *, int, int),
          void (*Kernel2)(PackedProjectionGroupParams<BlockT, nprojs>,
                          const block_q8_1 *, int, int),
          void (*Kernel4)(PackedProjectionGroupParams<BlockT, nprojs>,
                          const block_q8_1 *, int, int),
          void (*Kernel8)(PackedProjectionGroupParams<BlockT, nprojs>,
                          const block_q8_1 *, int, int)>
bool DispatchMmvqGroup(
    const std::array<const void *, nprojs> &weights, const void *act_q8_1,
    const std::array<half *, nprojs> &outputs,
    const std::array<int, nprojs> &output_cols, int M, int K,
    cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockT, nprojs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < nprojs; ++i) {
    params.weights[i] = static_cast<const BlockT *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  auto launch = [&](int ncols, auto kernel) {
    dim3 grid(max_output_cols, (M + ncols - 1) / ncols);
    size_t smem = sizeof(float) * kMmvqWarps * nprojs * ncols;
    const int threads = calc_mmvq_threads(ncols);
    kernel<<<grid, threads, smem, stream>>>(params, a, K, M);
    return true;
  };

  if (M <= 1)
    return launch(1, Kernel1);
  if (M <= 2)
    return launch(2, Kernel2);
  if (M <= 4)
    return launch(4, Kernel4);
  if (M <= 8)
    return launch(8, Kernel8);
  return false;
}

// Convenience dispatch for pair (2 projections)
template <typename BlockT,
          void (*Kernel1)(PackedProjectionGroupParams<BlockT, 2>,
                          const block_q8_1 *, int, int),
          void (*Kernel2)(PackedProjectionGroupParams<BlockT, 2>,
                          const block_q8_1 *, int, int),
          void (*Kernel4)(PackedProjectionGroupParams<BlockT, 2>,
                          const block_q8_1 *, int, int),
          void (*Kernel8)(PackedProjectionGroupParams<BlockT, 2>,
                          const block_q8_1 *, int, int)>
bool DispatchMmvqPair(const void *data0, const void *data1,
                      const void *act_q8_1, half *output0, int N0,
                      half *output1, int N1, int M, int K,
                      cudaStream_t stream) {
  return DispatchMmvqGroup<BlockT, 2, Kernel1, Kernel2, Kernel4, Kernel8>(
      {data0, data1}, act_q8_1, {output0, output1}, {N0, N1}, M, K, stream);
}

// Convenience dispatch for triple (3 projections)
template <typename BlockT,
          void (*Kernel1)(PackedProjectionGroupParams<BlockT, 3>,
                          const block_q8_1 *, int, int),
          void (*Kernel2)(PackedProjectionGroupParams<BlockT, 3>,
                          const block_q8_1 *, int, int),
          void (*Kernel4)(PackedProjectionGroupParams<BlockT, 3>,
                          const block_q8_1 *, int, int),
          void (*Kernel8)(PackedProjectionGroupParams<BlockT, 3>,
                          const block_q8_1 *, int, int)>
bool DispatchMmvqTriple(const void *data0, const void *data1,
                        const void *data2, const void *act_q8_1,
                        half *output0, int N0, half *output1, int N1,
                        half *output2, int N2, int M, int K,
                        cudaStream_t stream) {
  return DispatchMmvqGroup<BlockT, 3, Kernel1, Kernel2, Kernel4, Kernel8>(
      {data0, data1, data2}, act_q8_1, {output0, output1, output2},
      {N0, N1, N2}, M, K, stream);
}

// ============================================================================
// Fused Gate+Up+SiLU MMVQ kernel
//
// Computes SiLU(gate_proj(x)) * up_proj(x) in a single kernel pass.
// Both gate and up weights are Q4_K, activation is pre-quantized Q8_1.
// Eliminates intermediate gate/up buffer writes and the separate SiLuMul
// kernel, matching llama.cpp's has_fusion architecture.
//
// Output is FP16 (requires separate Q8_1 quantize before down_proj GEMV).
// ============================================================================

template <int ncols>
__global__ void inferflux_mmvq_q4k_fused_gate_up_silu(
    const block_q4_k *__restrict__ gate_weight,
    const block_q4_k *__restrict__ up_weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output, int N,
    int K, int M) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int col_base = blockIdx.y * ncols;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  const int group_leader = lane & ~7;

  float gate_acc[ncols] = {};
  float up_acc[ncols] = {};

  const block_q4_k *grow = gate_weight + out_idx * num_super_blocks;
  const block_q4_k *urow = up_weight + out_idx * num_super_blocks;

  // Outer loop: weight blocks along K (weights read ONCE per projection)
  for (int blk = warp_id; blk < num_super_blocks; blk += kMmvqWarps) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;

      // Load activation ONCE (shared between gate and up)
      const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;
      const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
      const int x_lo4 = *reinterpret_cast<const int *>(&a_lo.qs[offs]);
      const int x_hi4 = *reinterpret_cast<const int *>(&a_hi.qs[offs]);
      const float d8_lo = __half2float(__low2half(a_lo.ds));
      const float d8_hi = __half2float(__low2half(a_hi.ds));

      // Gate projection
      {
        const block_q4_k &b = grow[blk];
        const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));
        const float dmin =
            __half2float(__ldg(reinterpret_cast<const half *>(&b.dmin)));

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        if ((lane & 7) == 0) {
          get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
          get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
        }
        sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
        m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
        sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
        m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

        const int qs4 =
            __ldg(reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]));
        const int dot_lo = Dp4aS8(qs4 & 0x0F0F0F0F, x_lo4, 0);
        const int dot_hi = Dp4aS8((qs4 >> 4) & 0x0F0F0F0F, x_hi4, 0);

        gate_acc[c] +=
            d * static_cast<float>(sc_lo) * d8_lo *
                static_cast<float>(dot_lo) +
            d * static_cast<float>(sc_hi) * d8_hi *
                static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          gate_acc[c] -= dmin * static_cast<float>(m_lo) * s_lo +
                         dmin * static_cast<float>(m_hi) * s_hi;
        }
      }

      // Up projection (reuses activation data, loads different weights)
      {
        const block_q4_k &b = urow[blk];
        const float d = __half2float(__ldg(reinterpret_cast<const half *>(&b.d)));
        const float dmin =
            __half2float(__ldg(reinterpret_cast<const half *>(&b.dmin)));

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        if ((lane & 7) == 0) {
          get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
          get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
        }
        sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
        m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
        sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
        m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

        const int qs4 =
            __ldg(reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]));
        const int dot_lo = Dp4aS8(qs4 & 0x0F0F0F0F, x_lo4, 0);
        const int dot_hi = Dp4aS8((qs4 >> 4) & 0x0F0F0F0F, x_hi4, 0);

        up_acc[c] +=
            d * static_cast<float>(sc_lo) * d8_lo *
                static_cast<float>(dot_lo) +
            d * static_cast<float>(sc_hi) * d8_hi *
                static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          up_acc[c] -= dmin * static_cast<float>(m_lo) * s_lo +
                       dmin * static_cast<float>(m_hi) * s_hi;
        }
      }
    }
  }

  // Intra-warp reduction for both gate and up
#pragma unroll
  for (int c = 0; c < ncols; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate_acc[c] += __shfl_down_sync(0xFFFFFFFF, gate_acc[c], offset);
      up_acc[c] += __shfl_down_sync(0xFFFFFFFF, up_acc[c], offset);
    }
  }

  // Cross-warp reduction via shared memory
  __shared__ float warp_gate[kMmvqWarps * ncols];
  __shared__ float warp_up[kMmvqWarps * ncols];
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      warp_gate[c * kMmvqWarps + warp_id] = gate_acc[c];
      warp_up[c * kMmvqWarps + warp_id] = up_acc[c];
    }
  }
  __syncthreads();

  // Final reduction + fused SiLU(gate) * up
  if (tid == 0) {
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
      const int row = col_base + c;
      if (row >= M)
        break;
      float g = 0.0f, u = 0.0f;
      for (int w = 0; w < kMmvqWarps; ++w) {
        g += warp_gate[c * kMmvqWarps + w];
        u += warp_up[c * kMmvqWarps + w];
      }
      // SiLU(gate) * up = gate * sigmoid(gate) * up
      output[row * N + out_idx] = __float2half(u * g / (1.0f + expf(-g)));
    }
  }
}

// Dispatch helper for fused gate+up+SiLU MMVQ
template <typename BlockT,
          void (*Kernel1)(const BlockT *, const BlockT *, const block_q8_1 *,
                          half *, int, int, int),
          void (*Kernel2)(const BlockT *, const BlockT *, const block_q8_1 *,
                          half *, int, int, int),
          void (*Kernel4)(const BlockT *, const BlockT *, const block_q8_1 *,
                          half *, int, int, int),
          void (*Kernel8)(const BlockT *, const BlockT *, const block_q8_1 *,
                          half *, int, int, int)>
bool DispatchMmvqFusedGateUpSilu(const void *gate_data, const void *up_data,
                                 const void *act_q8_1, half *output, int N,
                                 int M, int K, cudaStream_t stream) {
  const auto *gate_w = reinterpret_cast<const BlockT *>(gate_data);
  const auto *up_w = reinterpret_cast<const BlockT *>(up_data);
  const auto *act = reinterpret_cast<const block_q8_1 *>(act_q8_1);
  int ncols;
  if (M <= 1)
    ncols = 1;
  else if (M <= 2)
    ncols = 2;
  else if (M <= 4)
    ncols = 4;
  else
    ncols = 8;
  const dim3 grid(N, (M + ncols - 1) / ncols);
  const int threads = calc_mmvq_threads(ncols);
  const dim3 block(threads);
  switch (ncols) {
  case 1:
    Kernel1<<<grid, block, 0, stream>>>(gate_w, up_w, act, output, N, K, M);
    break;
  case 2:
    Kernel2<<<grid, block, 0, stream>>>(gate_w, up_w, act, output, N, K, M);
    break;
  case 4:
    Kernel4<<<grid, block, 0, stream>>>(gate_w, up_w, act, output, N, K, M);
    break;
  case 8:
    Kernel8<<<grid, block, 0, stream>>>(gate_w, up_w, act, output, N, K, M);
    break;
  }
  return cudaGetLastError() == cudaSuccess;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
