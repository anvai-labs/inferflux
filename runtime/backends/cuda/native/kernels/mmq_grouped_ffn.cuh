#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// =============================================================================
// MMQ3: Warp-cooperative K-reduction with smem activation reuse.
//
// Identical thread layout to the generic rowquad kernel (32 lanes split K via
// dp4a, warp-level shuffle reduction) but loads activation Q8_1 blocks into
// shared memory and shares them across all kWarps output neurons per block.
//
// Grid:  (⌈max(N0,N1) / kWarps⌉, ⌈M / kRows⌉)
// Block: (kWarps * 32)  — 1D, warps identified via tid >> 5
// Smem:  kRows × 8 × sizeof(block_q8_1) bytes per super-block iteration
//
// Template parameter kRows = tokens per block (matches rowquad's 4).
// =============================================================================

template <typename BlockT, int Outputs>
struct MmqGroupParams {
  const BlockT *weights[Outputs];
  half *outputs[Outputs];
  int output_cols[Outputs];
};

constexpr int kMmq3Warps = 8;

template <int kRows, int Outputs>
__global__ void fused_grouped_ffn_mmq3_q4k_q8_1(
    MmqGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  constexpr int kQ8PerSB = 8;

  extern __shared__ char smem_raw[];
  auto *act_tile = reinterpret_cast<block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kMmq3Warps + warp_id;
  const int row_base = blockIdx.y * kRows;

  if (row_base >= total_rows)
    return;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_rows = min(kRows, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  constexpr int total_threads = kMmq3Warps * 32;

  float acc[kRows][Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Cooperative load: all threads fill act_tile[kRows][kQ8PerSB].
    constexpr int kSmemBlocks = kRows * kQ8PerSB;
    for (int idx = tid; idx < kSmemBlocks; idx += total_threads) {
      const int r = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = row_base + r;
      if (tok < total_rows) {
        act_tile[r * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k &b =
          params.weights[i][out_idx * num_super_blocks + blk];
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
      for (int r = 0; r < kRows; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = act_tile[r * kQ8PerSB + pair * 2];
        const block_q8_1 &a_hi = act_tile[r * kQ8PerSB + pair * 2 + 1];
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
    __syncthreads();
  }

  // Warp reduction.
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][i] += __shfl_down_sync(0xFFFFFFFF, acc[r][i], offset);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
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

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
