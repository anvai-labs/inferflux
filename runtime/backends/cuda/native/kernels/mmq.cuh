#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// ============================================================================
// MMQ: Tiled Quantized GEMM for batch sizes 9-64
//
// Operates directly on quantized weight blocks (Q4_K, Q6_K, Q8_0, Q8_K)
// without dequantizing to FP16. Eliminates the 3.6x bandwidth waste of the
// cuBLAS-on-FP16 fallback path.
//
// Tile: 32 output rows (M-tile) × 8 output columns (N-tile, one per warp)
// K-iteration: one super-block (256 elements for K-types, 32 for Q8_0) per step
// Grid: dim3(ceil(N/kMmqWarps), ceil(M/kMmqTileM))
// Block: 256 threads (8 warps)
// Smem: kMmqTileM × Q8_1-blocks-per-superblock × sizeof(block_q8_1)
//
// This design mirrors MMQ3 (mmq_grouped_ffn.cuh) but is a general-purpose
// single-projection GEMM rather than a grouped FFN kernel.
// ============================================================================

constexpr int kMmqWarps = 8;
constexpr int kMmqThreads = kMmqWarps * 32;

// ============================================================================
// Q4_K × Q8_1 MMQ kernel
//
// Each warp handles one output column (weight row). All warps cooperatively
// load activation tiles into shared memory, then each warp independently
// computes its dot product against its weight row.
// ============================================================================

template <int kTileM>
__global__ void inferflux_mmq_q4k(const block_q4_k *__restrict__ weight,
                                  const block_q8_1 *__restrict__ act_q8_1,
                                  half *__restrict__ output, int M, int N,
                                  int K) {
  constexpr int kQ8PerSB = 8; // Q8_1 blocks per Q4_K super-block

  extern __shared__ char smem_raw[];
  auto *act_tile = reinterpret_cast<block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kMmqWarps + warp_id;
  const int row_base = blockIdx.y * kTileM;

  if (row_base >= M)
    return;

  const int num_rows = min(kTileM, M - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[kTileM] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Cooperative load: all threads fill act_tile[kTileM][kQ8PerSB]
    constexpr int kSmemBlocks = kTileM * kQ8PerSB;
    for (int idx = tid; idx < kSmemBlocks; idx += kMmqThreads) {
      const int r = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = row_base + r;
      if (tok < M) {
        act_tile[r * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
    }
    __syncthreads();

    if (out_idx < N) {
      const block_q4_k &b = weight[out_idx * num_super_blocks + blk];
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
      for (int r = 0; r < kTileM; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = act_tile[r * kQ8PerSB + pair * 2];
        const block_q8_1 &a_hi = act_tile[r * kQ8PerSB + pair * 2 + 1];
        int x_lo4, x_hi4;
        memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
        memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
        const float d8_lo = __half2float(__low2half(a_lo.ds));
        const float d8_hi = __half2float(__low2half(a_hi.ds));
        const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
        const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

        acc[r] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                  d_sc_hi * d8_hi * static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          acc[r] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
        }
      }
    }
    __syncthreads();
  }

  // Warp reduction
  if (out_idx < N) {
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
      }
    }

    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        if (r < num_rows) {
          output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
        }
      }
    }
  }
}

// ============================================================================
// Q6_K × Q8_1 MMQ kernel
// ============================================================================

template <int kTileM>
__global__ void inferflux_mmq_q6k(const block_q6_k *__restrict__ weight,
                                  const block_q8_1 *__restrict__ act_q8_1,
                                  half *__restrict__ output, int M, int N,
                                  int K) {
  constexpr int kQ8PerSB = 8;

  extern __shared__ char smem_raw[];
  auto *act_tile = reinterpret_cast<block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kMmqWarps + warp_id;
  const int row_base = blockIdx.y * kTileM;

  if (row_base >= M)
    return;

  const int num_rows = min(kTileM, M - row_base);
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

  float acc[kTileM] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    constexpr int kSmemBlocks = kTileM * kQ8PerSB;
    for (int idx = tid; idx < kSmemBlocks; idx += kMmqThreads) {
      const int r = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = row_base + r;
      if (tok < M) {
        act_tile[r * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
    }
    __syncthreads();

    if (out_idx < N) {
      const block_q6_k &b = weight[out_idx * num_super_blocks + blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

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
      for (int r = 0; r < kTileM; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = act_tile[r * kQ8PerSB + g * 4 + sub_base];
        const block_q8_1 &a_hi =
            act_tile[r * kQ8PerSB + g * 4 + sub_base + 2];
        const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
        const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
        float d8_lo = __half2float(__low2half(a_lo.ds));
        float d8_hi = __half2float(__low2half(a_hi.ds));

        int dot_lo = Dp4aS8(vi_lo, x_lo, 0);
        int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

        acc[r] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                            static_cast<float>(dot_lo) +
                        static_cast<float>(b.scales[sc_hi]) * d8_hi *
                            static_cast<float>(dot_hi));
      }
    }
    __syncthreads();
  }

  if (out_idx < N) {
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
      }
    }

    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        if (r < num_rows) {
          output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
        }
      }
    }
  }
}

// ============================================================================
// Q8_0 × Q8_1 MMQ kernel
// ============================================================================

template <int kTileM>
__global__ void inferflux_mmq_q8_0(const block_q8_0 *__restrict__ weight,
                                   const block_q8_1 *__restrict__ act_q8_1,
                                   half *__restrict__ output, int M, int N,
                                   int K) {
  // Q8_0 blocks are 32 elements, same as Q8_1 — 1:1 mapping
  constexpr int kBlocksPerIter = 4; // Process 4 Q8_0 blocks per iteration

  extern __shared__ char smem_raw[];
  auto *act_tile = reinterpret_cast<block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kMmqWarps + warp_id;
  const int row_base = blockIdx.y * kTileM;

  if (row_base >= M)
    return;

  const int num_rows = min(kTileM, M - row_base);
  const int num_blocks = K / QK8_0;

  float acc[kTileM] = {};

  for (int blk_base = 0; blk_base < num_blocks; blk_base += kBlocksPerIter) {
    const int iter_blocks = min(kBlocksPerIter, num_blocks - blk_base);
    const int kSmemBlocks = kTileM * iter_blocks;
    for (int idx = tid; idx < kSmemBlocks; idx += kMmqThreads) {
      const int r = idx / iter_blocks;
      const int q = idx % iter_blocks;
      const int tok = row_base + r;
      if (tok < M) {
        act_tile[r * kBlocksPerIter + q] =
            act_q8_1[tok * num_blocks + blk_base + q];
      }
    }
    __syncthreads();

    if (out_idx < N) {
      for (int bi = 0; bi < iter_blocks; ++bi) {
        const int blk = blk_base + bi;
        const block_q8_0 &b = weight[out_idx * num_blocks + blk];
        const float d_w =
            __half2float(*reinterpret_cast<const half *>(&b.d));

        int w4[QK8_0 / 4];
        for (int j = 0; j < QK8_0; j += 4) {
          w4[j / 4] = LoadPackedInt32Unaligned(&b.qs[j]);
        }

#pragma unroll
        for (int r = 0; r < kTileM; ++r) {
          if (r >= num_rows)
            continue;

          const block_q8_1 &a = act_tile[r * kBlocksPerIter + bi];
          const float d_a = __half2float(__low2half(a.ds));

          int int_acc = 0;
          for (int j = 0; j < QK8_0; j += 4) {
            const int x4 = LoadPackedInt32Unaligned(&a.qs[j]);
            int_acc = Dp4aS8(w4[j / 4], x4, int_acc);
          }
          acc[r] += d_w * d_a * static_cast<float>(int_acc);
        }
      }
    }
    __syncthreads();
  }

  if (out_idx < N) {
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
      }
    }

    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        if (r < num_rows) {
          output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
        }
      }
    }
  }
}

// ============================================================================
// Q8_K × Q8_1 MMQ kernel
// ============================================================================

template <int kTileM>
__global__ void inferflux_mmq_q8k(const block_q8_k *__restrict__ weight,
                                  const block_q8_1 *__restrict__ act_q8_1,
                                  half *__restrict__ output, int M, int N,
                                  int K) {
  constexpr int kQ8PerSB = 8;

  extern __shared__ char smem_raw[];
  auto *act_tile = reinterpret_cast<block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kMmqWarps + warp_id;
  const int row_base = blockIdx.y * kTileM;

  if (row_base >= M)
    return;

  const int num_rows = min(kTileM, M - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;

  float acc[kTileM] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    constexpr int kSmemBlocks = kTileM * kQ8PerSB;
    for (int idx = tid; idx < kSmemBlocks; idx += kMmqThreads) {
      const int r = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = row_base + r;
      if (tok < M) {
        act_tile[r * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
    }
    __syncthreads();

    if (out_idx < N) {
      const block_q8_k &b = weight[out_idx * num_super_blocks + blk];
      const float d_w = b.d;

#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        if (r >= num_rows)
          continue;

        for (int sub = 0; sub < 8; ++sub) {
          const block_q8_1 &a = act_tile[r * kQ8PerSB + sub];
          const float d_a = __half2float(__low2half(a.ds));

          int int_acc = 0;
          for (int j = 0; j < QK8_1; j += 4) {
            int w4 = *reinterpret_cast<const int *>(&b.qs[sub * QK8_1 + j]);
            int x4;
            memcpy(&x4, &a.qs[j], sizeof(x4));
            int_acc = Dp4aS8(w4, x4, int_acc);
          }
          acc[r] += d_w * d_a * static_cast<float>(int_acc);
        }
      }
    }
    __syncthreads();
  }

  if (out_idx < N) {
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
      }
    }

    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        if (r < num_rows) {
          output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
        }
      }
    }
  }
}

// ============================================================================
// MMQ dispatch helpers
//
// Select kTileM at runtime. Use 16 for M<=16, 32 for M<=32, 64 for larger.
// ============================================================================

template <typename BlockT,
          void (*Kernel16)(const BlockT *, const block_q8_1 *, half *, int, int,
                           int),
          void (*Kernel32)(const BlockT *, const block_q8_1 *, half *, int, int,
                           int)>
bool DispatchMmq(const void *data, const void *act_q8_1, half *output, int M,
                 int N, int K, int q8_per_sb, cudaStream_t stream) {
  auto *w = static_cast<const BlockT *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);

  auto launch = [&](int tile_m, auto kernel) {
    dim3 grid((N + kMmqWarps - 1) / kMmqWarps, (M + tile_m - 1) / tile_m);
    size_t smem =
        static_cast<size_t>(tile_m) * q8_per_sb * sizeof(block_q8_1);
    kernel<<<grid, kMmqThreads, smem, stream>>>(w, a, output, M, N, K);
    return true;
  };

  if (M <= 16)
    return launch(16, Kernel16);
  return launch(32, Kernel32);
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
