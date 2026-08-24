#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using inferflux::FusedDispatchGeometry;
using inferflux::FusedQuantGemm;
using inferflux::NativeExecutionPolicy;
using inferflux::PackedProjectionSpec;
namespace native = inferflux::runtime::cuda::native;
namespace GGUF = native::GGUF;

namespace {

constexpr int kWarmupIters = 30;
constexpr int kBenchmarkIters = 200;
constexpr int kK = 2048;
constexpr int kN0 = 11008;
constexpr int kN1 = 11008;
constexpr int kBlocksPerRow = kK / QK_K;
constexpr int kMmqTileCols = FusedQuantGemm::kDownProjMmqTileCols;
constexpr int kQ8BlocksPerRow = kK / QK8_1;
constexpr int kMmqQ8BlocksPerPack = 4;

struct alignas(16) BenchmarkBlockQ8_1Mmq {
  half2 ds4[4];
  int8_t qs[4 * QK8_1];
};

static_assert(sizeof(BenchmarkBlockQ8_1Mmq) == 4 * sizeof(native::block_q8_1),
              "Unexpected MMQ benchmark Q8_1 block size");

struct BenchmarkResult {
  float mean_ms{0.0f};
  std::vector<half> output;
};

__device__ __forceinline__ int BenchDp4aS8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
  return __dp4a(a, b, c);
#else
  const auto *ap = reinterpret_cast<const signed char *>(&a);
  const auto *bp = reinterpret_cast<const signed char *>(&b);
  return c + static_cast<int>(ap[0]) * static_cast<int>(bp[0]) +
         static_cast<int>(ap[1]) * static_cast<int>(bp[1]) +
         static_cast<int>(ap[2]) * static_cast<int>(bp[2]) +
         static_cast<int>(ap[3]) * static_cast<int>(bp[3]);
#endif
}

template <typename BlockT>
__global__ void transform_grouped_mmq_layout(const BlockT *__restrict__ src,
                                             BlockT *__restrict__ dst, int rows,
                                             int num_super_blocks,
                                             int tile_cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile_count = (rows + tile_cols - 1) / tile_cols;
  const int total_blocks = tile_count * num_super_blocks * tile_cols;
  if (idx >= total_blocks) {
    return;
  }

  const int tile_stride = num_super_blocks * tile_cols;
  const int tile = idx / tile_stride;
  const int rem = idx % tile_stride;
  const int super_block = rem / tile_cols;
  const int tile_col = rem % tile_cols;
  const int src_row = tile * tile_cols + tile_col;
  if (src_row < rows) {
    dst[idx] = src[src_row * num_super_blocks + super_block];
  } else {
    dst[idx] = BlockT{};
  }
}

__global__ void fused_grouped_ffn_mmq_q4k_q8_1_tile(
    const native::block_q4_k *__restrict__ tiled_weight0,
    const native::block_q4_k *__restrict__ tiled_weight1,
    const native::block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output0,
    int N0, half *__restrict__ output1, int N1, int K, int tile_cols) {
  extern __shared__ unsigned char shared[];
  auto *shared_act = reinterpret_cast<native::block_q8_1 *>(shared);

  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;
  const int max_output_cols = N0 > N1 ? N0 : N1;
  if (out_idx >= max_output_cols) {
    return;
  }

  const int num_super_blocks = K / QK_K;
  const int num_q8_blocks = K / QK8_1;
  const int linear_tid = out_local * 32 + lane;
  for (int idx = linear_tid; idx < num_q8_blocks;
       idx += blockDim.x * blockDim.y) {
    shared_act[idx] = act_q8_1[row * num_q8_blocks + idx];
  }
  __syncthreads();

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  const native::block_q4_k *tile_weights0 =
      tiled_weight0 + (blockIdx.x * num_super_blocks * tile_cols);
  const native::block_q4_k *tile_weights1 =
      tiled_weight1 + (blockIdx.x * num_super_blocks * tile_cols);
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const native::block_q8_1 &a_lo = shared_act[blk * 8 + pair * 2];
    const native::block_q8_1 &a_hi = shared_act[blk * 8 + pair * 2 + 1];
    int x_lo4 = 0;
    int x_hi4 = 0;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));
    const float s_lo = __half2float(__high2half(a_lo.ds));
    const float s_hi = __half2float(__high2half(a_hi.ds));

    if (out_idx < N0) {
      const native::block_q4_k &b = tile_weights0[blk * tile_cols + out_local];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const half *>(&b.dmin));
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      native::get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      native::get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);
      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;
      const int dot_lo = BenchDp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = BenchDp4aS8(q_hi4, x_hi4, 0);

      acc0 += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
              d_sc_hi * d8_hi * static_cast<float>(dot_hi);
      if ((lane & 7) == 0) {
        acc0 -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }
    if (out_idx < N1) {
      const native::block_q4_k &b = tile_weights1[blk * tile_cols + out_local];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const half *>(&b.dmin));
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      native::get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      native::get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);
      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;
      const int dot_lo = BenchDp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = BenchDp4aS8(q_hi4, x_hi4, 0);

      acc1 += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
              d_sc_hi * d8_hi * static_cast<float>(dot_hi);
      if ((lane & 7) == 0) {
        acc1 -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    if (out_idx < N0) {
      output0[row * N0 + out_idx] = __float2half(acc0);
    }
    if (out_idx < N1) {
      output1[row * N1 + out_idx] = __float2half(acc1);
    }
  }
}

__global__ void transform_act_q8_1_mmq_layout(
    const native::block_q8_1 *__restrict__ src,
    BenchmarkBlockQ8_1Mmq *__restrict__ dst, int rows, int q8_blocks_per_row) {
  const int packed_per_row = q8_blocks_per_row / kMmqQ8BlocksPerPack;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * packed_per_row;
  if (idx >= total) {
    return;
  }

  const int row = idx / packed_per_row;
  const int packed = idx % packed_per_row;
  const native::block_q8_1 *src_blocks =
      src + row * q8_blocks_per_row + packed * kMmqQ8BlocksPerPack;
  BenchmarkBlockQ8_1Mmq &out = dst[idx];
  for (int i = 0; i < kMmqQ8BlocksPerPack; ++i) {
    out.ds4[i] = src_blocks[i].ds;
    memcpy(&out.qs[i * QK8_1], src_blocks[i].qs, QK8_1);
  }
}

__global__ void fused_grouped_ffn_mmq_q4k_q8_1_layout(
    const native::block_q4_k *__restrict__ tiled_weight0,
    const native::block_q4_k *__restrict__ tiled_weight1,
    const BenchmarkBlockQ8_1Mmq *__restrict__ act_q8_1_mmq,
    half *__restrict__ output0, int N0, half *__restrict__ output1, int N1,
    int K, int tile_cols) {
  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;
  const int max_output_cols = N0 > N1 ? N0 : N1;
  if (out_idx >= max_output_cols) {
    return;
  }

  const int num_super_blocks = K / QK_K;
  const int packed_per_row = K / (QK8_1 * kMmqQ8BlocksPerPack);
  const native::block_q4_k *tile_weights0 =
      tiled_weight0 + (blockIdx.x * num_super_blocks * tile_cols);
  const native::block_q4_k *tile_weights1 =
      tiled_weight1 + (blockIdx.x * num_super_blocks * tile_cols);

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  const BenchmarkBlockQ8_1Mmq *row_act =
      act_q8_1_mmq + row * packed_per_row;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const BenchmarkBlockQ8_1Mmq &packed = row_act[blk * 2 + pair / 2];
    const int local_base = (pair & 1) ? 2 : 0;

    int x_lo4 = 0;
    int x_hi4 = 0;
    memcpy(&x_lo4, &packed.qs[local_base * QK8_1 + offs], sizeof(x_lo4));
    memcpy(&x_hi4, &packed.qs[(local_base + 1) * QK8_1 + offs], sizeof(x_hi4));
    const float d8_lo = __half2float(__low2half(packed.ds4[local_base]));
    const float d8_hi =
        __half2float(__low2half(packed.ds4[local_base + 1]));
    const float s_lo = __half2float(__high2half(packed.ds4[local_base]));
    const float s_hi =
        __half2float(__high2half(packed.ds4[local_base + 1]));

    if (out_idx < N0) {
      const native::block_q4_k &b = tile_weights0[blk * tile_cols + out_local];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const half *>(&b.dmin));
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      native::get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      native::get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);
      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;
      const int dot_lo = BenchDp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = BenchDp4aS8(q_hi4, x_hi4, 0);

      acc0 += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
              d_sc_hi * d8_hi * static_cast<float>(dot_hi);
      if ((lane & 7) == 0) {
        acc0 -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }

    if (out_idx < N1) {
      const native::block_q4_k &b = tile_weights1[blk * tile_cols + out_local];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const half *>(&b.dmin));
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      native::get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      native::get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);
      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;
      const int dot_lo = BenchDp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = BenchDp4aS8(q_hi4, x_hi4, 0);

      acc1 += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
              d_sc_hi * d8_hi * static_cast<float>(dot_hi);
      if ((lane & 7) == 0) {
        acc1 -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    if (out_idx < N0) {
      output0[row * N0 + out_idx] = __float2half(acc0);
    }
    if (out_idx < N1) {
      output1[row * N1 + out_idx] = __float2half(acc1);
    }
  }
}

// =============================================================================
// MMQ2: True 2D-tiled kernel — kMMQX tokens × kMMQY output neurons per block.
// Activations loaded into smem once per super-block, reused across kMMQY outputs.
// No warp-level reduction: each thread accumulates the full K dimension for its
// (token, output-neuron) pairs independently.
// =============================================================================

constexpr int kMMQY = 32;   // output neurons per block (= warp width)
constexpr int kNWarps2 = 8; // warps per block → token groups

template <int kMMQX>
__global__ void fused_grouped_ffn_mmq2_q4k_q8_1(
    const native::block_q4_k *__restrict__ w0,
    const native::block_q4_k *__restrict__ w1,
    const native::block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output0,
    int N0, half *__restrict__ output1, int N1, int M, int K) {
  // kMMQX must be divisible by kNWarps2
  static_assert(kMMQX % kNWarps2 == 0, "kMMQX not divisible by kNWarps2");
  constexpr int kQ8PerSB = 8;   // QK_K / QK8_1 = 256 / 32
  constexpr int kAccPT = kMMQX / kNWarps2; // accumulators per thread per projection

  // Smem: activation tile for kMMQX tokens × kQ8PerSB Q8_1 blocks per super-block.
  // Reloaded each super-block iteration.
  extern __shared__ unsigned char smem_raw[];
  auto *act_tile = reinterpret_cast<native::block_q8_1 *>(smem_raw);

  const int lane = threadIdx.x; // 0..31 → output-neuron within tile
  const int wy = threadIdx.y;   // 0..7  → token-group within tile
  const int linear_tid = wy * 32 + lane;

  const int out_i = blockIdx.x * kMMQY + lane;
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;

  const bool active0 = (out_i < N0);
  const bool active1 = (out_i < N1);

  float acc_gate[kAccPT] = {};
  float acc_up[kAccPT] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Cooperative load: all 256 threads fill act_tile[kMMQX][kQ8PerSB].
    constexpr int kSmemBlocks = kMMQX * kQ8PerSB;
    for (int idx = linear_tid; idx < kSmemBlocks; idx += 256) {
      const int j = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = blockIdx.y * kMMQX + j;
      if (tok < M) {
        act_tile[j * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
      // act_tile entries for tok >= M are never read (jl loop guards below).
    }
    __syncthreads();

    // Each thread owns one output neuron (out_i) and kAccPT tokens.
    // Load weight block pointers for this super-block once.
    const native::block_q4_k *wb0 =
        active0 ? &w0[out_i * num_super_blocks + blk] : nullptr;
    const native::block_q4_k *wb1 =
        active1 ? &w1[out_i * num_super_blocks + blk] : nullptr;

    const float d0 = active0 ? __half2float(*reinterpret_cast<const half *>(&wb0->d)) : 0.f;
    const float dmin0 = active0 ? __half2float(*reinterpret_cast<const half *>(&wb0->dmin)) : 0.f;
    const float d1 = active1 ? __half2float(*reinterpret_cast<const half *>(&wb1->d)) : 0.f;
    const float dmin1 = active1 ? __half2float(*reinterpret_cast<const half *>(&wb1->dmin)) : 0.f;

    for (int jl = 0; jl < kAccPT; ++jl) {
      const int j = wy * kAccPT + jl;
      const int global_tok = blockIdx.y * kMMQX + j;
      if (global_tok >= M) {
        continue;
      }

      // Loop over 8 Q8_1 sub-blocks within this super-block.
      for (int sb = 0; sb < 8; ++sb) {
        const native::block_q8_1 &a = act_tile[j * kQ8PerSB + sb];
        const float d8 = __half2float(__low2half(a.ds));
        // s8 = d8 * sum(qs) — used for the zero-point correction.
        const float s8 = __half2float(__high2half(a.ds));
        const int pair = sb >> 1; // 0..3: selects 32-byte qs group
        const int is_hi = sb & 1; // lo/hi nibble selector

        if (active0) {
          unsigned char sc, m;
          native::get_scale_min_k4(sb, wb0->scales, &sc, &m);
          const float d_sc = d0 * static_cast<float>(sc);
          const float dm_m = dmin0 * static_cast<float>(m);
          // Min correction: dmin * m * sum(act) = dm_m * s8
          acc_gate[jl] -= dm_m * s8;
          // Dot product over 32 elements via 8 dp4a calls.
          for (int e = 0; e < 8; ++e) {
            const int qs4 =
                *reinterpret_cast<const int *>(&wb0->qs[pair * 32 + e * 4]);
            const int q4 =
                is_hi ? ((qs4 >> 4) & 0x0F0F0F0F) : (qs4 & 0x0F0F0F0F);
            int x4;
            memcpy(&x4, &a.qs[e * 4], sizeof(x4));
            acc_gate[jl] +=
                d_sc * d8 * static_cast<float>(BenchDp4aS8(q4, x4, 0));
          }
        }

        if (active1) {
          unsigned char sc, m;
          native::get_scale_min_k4(sb, wb1->scales, &sc, &m);
          const float d_sc = d1 * static_cast<float>(sc);
          const float dm_m = dmin1 * static_cast<float>(m);
          acc_up[jl] -= dm_m * s8;
          for (int e = 0; e < 8; ++e) {
            const int qs4 =
                *reinterpret_cast<const int *>(&wb1->qs[pair * 32 + e * 4]);
            const int q4 =
                is_hi ? ((qs4 >> 4) & 0x0F0F0F0F) : (qs4 & 0x0F0F0F0F);
            int x4;
            memcpy(&x4, &a.qs[e * 4], sizeof(x4));
            acc_up[jl] +=
                d_sc * d8 * static_cast<float>(BenchDp4aS8(q4, x4, 0));
          }
        }
      }
    }
    __syncthreads();
  }

  // Writeback: each thread writes kAccPT (token, output-neuron) pairs.
  for (int jl = 0; jl < kAccPT; ++jl) {
    const int global_tok = blockIdx.y * kMMQX + wy * kAccPT + jl;
    if (global_tok >= M) {
      break;
    }
    if (active0) {
      output0[global_tok * N0 + out_i] = __float2half(acc_gate[jl]);
    }
    if (active1) {
      output1[global_tok * N1 + out_i] = __float2half(acc_up[jl]);
    }
  }
}

// =============================================================================
// MMQ3: Warp-cooperative K-reduction WITH smem activation reuse.
// Same warp layout as rowquad (32 lanes split K), but activation data is loaded
// into shared memory and shared across kWarps3 output neurons per block.
// kRowsPerBlock tokens processed per block (like rowquad uses 4).
// =============================================================================

constexpr int kWarps3 = 8;
constexpr int kRowsPerBlock3 = 4;

template <int kRows>
__global__ void fused_grouped_ffn_mmq3_q4k_q8_1(
    const native::block_q4_k *__restrict__ w0,
    const native::block_q4_k *__restrict__ w1,
    const native::block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output0,
    int N0, half *__restrict__ output1, int N1, int M, int K) {
  constexpr int kQ8PerSB = 8;

  extern __shared__ unsigned char smem_raw[];
  // smem holds kRows × kQ8PerSB activation blocks per super-block iteration.
  auto *act_tile = reinterpret_cast<native::block_q8_1 *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kWarps3 + warp_id;
  const int row_base = blockIdx.y * kRows;

  if (row_base >= M)
    return;

  const bool active0 = (out_idx < N0);
  const bool active1 = (out_idx < N1);
  if (!active0 && !active1)
    return;

  const int num_rows = min(kRows, M - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  const int total_threads = kWarps3 * 32;

  float acc[kRows][2] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Cooperative load: all threads fill act_tile[kRows][kQ8PerSB].
    constexpr int kSmemBlocks = kRows * kQ8PerSB;
    for (int idx = tid; idx < kSmemBlocks; idx += total_threads) {
      const int r = idx / kQ8PerSB;
      const int q = idx % kQ8PerSB;
      const int tok = row_base + r;
      if (tok < M) {
        act_tile[r * kQ8PerSB + q] =
            act_q8_1[tok * num_q8_per_row + blk * kQ8PerSB + q];
      }
    }
    __syncthreads();

    // Each warp processes its output neuron across all rows.
    // Weight block is the same for all rows (same output neuron, same
    // super-block).
#pragma unroll
    for (int pi = 0; pi < 2; ++pi) {
      const int N = pi == 0 ? N0 : N1;
      const native::block_q4_k *w = pi == 0 ? w0 : w1;
      if (out_idx >= N)
        continue;

      const native::block_q4_k &b = w[out_idx * num_super_blocks + blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      native::get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      native::get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

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

        const native::block_q8_1 &a_lo =
            act_tile[r * kQ8PerSB + pair * 2];
        const native::block_q8_1 &a_hi =
            act_tile[r * kQ8PerSB + pair * 2 + 1];
        int x_lo4;
        int x_hi4;
        memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
        memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
        const float d8_lo = __half2float(__low2half(a_lo.ds));
        const float d8_hi = __half2float(__low2half(a_hi.ds));
        const int dot_lo = BenchDp4aS8(q_lo4, x_lo4, 0);
        const int dot_hi = BenchDp4aS8(q_hi4, x_hi4, 0);

        acc[r][pi] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                      d_sc_hi * d8_hi * static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          acc[r][pi] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
        }
      }
    }
    __syncthreads();
  }

  // Warp reduction.
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int pi = 0; pi < 2; ++pi) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][pi] += __shfl_down_sync(0xFFFFFFFF, acc[r][pi], offset);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      if (r >= num_rows)
        continue;
      if (active0) {
        output0[(row_base + r) * N0 + out_idx] = __float2half(acc[r][0]);
      }
      if (active1) {
        output1[(row_base + r) * N1 + out_idx] = __float2half(acc[r][1]);
      }
    }
  }
}

unsigned short EncodeHalfBits(float value) {
  const half h = __float2half(value);
  unsigned short bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

std::vector<half> MakeWaveTensor(size_t count, float scale, float bias = 0.0f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    const float value =
        bias + scale * std::sin(0.173f * static_cast<float>(i) + 0.031f);
    out[i] = __float2half(value);
  }
  return out;
}

std::vector<native::block_q4_k> MakeQ4Rows(int rows, int seed) {
  std::vector<native::block_q4_k> blocks(static_cast<size_t>(rows) *
                                         kBlocksPerRow);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * kBlocksPerRow + blk];
      block.d = EncodeHalfBits(
          0.02f * static_cast<float>(((row + blk + seed) % 4) + 1));
      block.dmin = EncodeHalfBits(
          0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] = static_cast<unsigned char>(
            (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] = static_cast<unsigned char>(
            (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
      }
    }
  }
  return blocks;
}

BenchmarkResult RunCase(int m, const NativeExecutionPolicy &policy) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * (kK / QK8_1) *
                              sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN0) * kK},
       d_out,
       kN0},
      {{d_w1, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN1) * kK},
       d_out + static_cast<size_t>(m) * kN0,
       kN1},
  }};

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kK, stream,
                                   &policy);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kK, stream,
                                   &policy);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

BenchmarkResult RunRowQuadCandidateCase(int m) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * (kK / QK8_1) *
                              sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN0) * kK},
       d_out,
       kN0},
      {{d_w1, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN1) * kK},
       d_out + static_cast<size_t>(m) * kN0,
       kN1},
  }};

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::GemvQ8_1PairRowQuadCandidate(projections, d_act_q8_1, m,
                                                   kK, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::GemvQ8_1PairRowQuadCandidate(projections, d_act_q8_1, m,
                                                   kK, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

BenchmarkResult RunMmqTileCandidateCase(int m, int tile_cols) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  native::block_q4_k *d_w0_tiled = nullptr;
  native::block_q4_k *d_w1_tiled = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * (kK / QK8_1) *
                              sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const int num_super_blocks = kK / QK_K;
  const int tile_count = (kN0 + tile_cols - 1) / tile_cols;
  const size_t transformed_blocks =
      static_cast<size_t>(tile_count) * num_super_blocks * tile_cols;
  cudaMalloc(reinterpret_cast<void **>(&d_w0_tiled),
             transformed_blocks * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1_tiled),
             transformed_blocks * sizeof(native::block_q4_k));

  constexpr int kTransformThreads = 256;
  const int transform_blocks = static_cast<int>(
      (transformed_blocks + kTransformThreads - 1) / kTransformThreads);
  transform_grouped_mmq_layout<<<transform_blocks, kTransformThreads, 0, stream>>>(
      d_w0, d_w0_tiled, kN0, num_super_blocks, tile_cols);
  transform_grouped_mmq_layout<<<transform_blocks, kTransformThreads, 0, stream>>>(
      d_w1, d_w1_tiled, kN1, num_super_blocks, tile_cols);
  cudaStreamSynchronize(stream);

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const dim3 block(32, tile_cols);
    const dim3 grid((kN0 + tile_cols - 1) / tile_cols, m);
    const size_t shared_bytes =
        static_cast<size_t>(kK / QK8_1) * sizeof(native::block_q8_1);
    for (int i = 0; i < kWarmupIters; ++i) {
      fused_grouped_ffn_mmq_q4k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
          d_w0_tiled, d_w1_tiled,
          static_cast<const native::block_q8_1 *>(d_act_q8_1), d_out, kN0,
          d_out + static_cast<size_t>(m) * kN0, kN1, kK, tile_cols);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      fused_grouped_ffn_mmq_q4k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
          d_w0_tiled, d_w1_tiled,
          static_cast<const native::block_q8_1 *>(d_act_q8_1), d_out, kN0,
          d_out + static_cast<size_t>(m) * kN0, kN1, kK, tile_cols);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_w1_tiled);
  cudaFree(d_w0_tiled);
  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

BenchmarkResult RunMmqLayoutCandidateCase(int m, int tile_cols) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  native::block_q4_k *d_w0_tiled = nullptr;
  native::block_q4_k *d_w1_tiled = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  BenchmarkBlockQ8_1Mmq *d_act_q8_1_mmq = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * kQ8BlocksPerRow *
                              sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const int num_super_blocks = kK / QK_K;
  const int tile_count = (kN0 + tile_cols - 1) / tile_cols;
  const size_t transformed_blocks =
      static_cast<size_t>(tile_count) * num_super_blocks * tile_cols;
  cudaMalloc(reinterpret_cast<void **>(&d_w0_tiled),
             transformed_blocks * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1_tiled),
             transformed_blocks * sizeof(native::block_q4_k));

  constexpr int kTransformThreads = 256;
  const int transform_blocks = static_cast<int>(
      (transformed_blocks + kTransformThreads - 1) / kTransformThreads);
  transform_grouped_mmq_layout<<<transform_blocks, kTransformThreads, 0, stream>>>(
      d_w0, d_w0_tiled, kN0, num_super_blocks, tile_cols);
  transform_grouped_mmq_layout<<<transform_blocks, kTransformThreads, 0, stream>>>(
      d_w1, d_w1_tiled, kN1, num_super_blocks, tile_cols);

  const int packed_per_row = kQ8BlocksPerRow / kMmqQ8BlocksPerPack;
  const size_t packed_count = static_cast<size_t>(m) * packed_per_row;
  cudaMalloc(reinterpret_cast<void **>(&d_act_q8_1_mmq),
             packed_count * sizeof(BenchmarkBlockQ8_1Mmq));
  const int act_blocks = static_cast<int>(
      (packed_count + kTransformThreads - 1) / kTransformThreads);
  transform_act_q8_1_mmq_layout<<<act_blocks, kTransformThreads, 0, stream>>>(
      static_cast<const native::block_q8_1 *>(d_act_q8_1), d_act_q8_1_mmq, m,
      kQ8BlocksPerRow);
  cudaStreamSynchronize(stream);

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const dim3 block(32, tile_cols);
    const dim3 grid((kN0 + tile_cols - 1) / tile_cols, m);
    for (int i = 0; i < kWarmupIters; ++i) {
      fused_grouped_ffn_mmq_q4k_q8_1_layout<<<grid, block, 0, stream>>>(
          d_w0_tiled, d_w1_tiled, d_act_q8_1_mmq, d_out, kN0,
          d_out + static_cast<size_t>(m) * kN0, kN1, kK, tile_cols);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      fused_grouped_ffn_mmq_q4k_q8_1_layout<<<grid, block, 0, stream>>>(
          d_w0_tiled, d_w1_tiled, d_act_q8_1_mmq, d_out, kN0,
          d_out + static_cast<size_t>(m) * kN0, kN1, kK, tile_cols);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_act_q8_1_mmq);
  cudaFree(d_w1_tiled);
  cudaFree(d_w0_tiled);
  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

BenchmarkResult RunMMQ2CandidateCase(int m, int mmq_x) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * kQ8BlocksPerRow *
                               sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const auto launch_mmq2 = [&](cudaStream_t s) {
    const dim3 block(kMMQY, kNWarps2);
    const int max_n = kN0 > kN1 ? kN0 : kN1;
    const auto *act = static_cast<const native::block_q8_1 *>(d_act_q8_1);
    half *out0 = d_out;
    half *out1 = d_out + static_cast<size_t>(m) * kN0;

#define LAUNCH_MMQ2(KMMQX)                                                     \
  do {                                                                          \
    const dim3 grid((max_n + kMMQY - 1) / kMMQY,                               \
                    (m + (KMMQX)-1) / (KMMQX));                                \
    const size_t smem = (KMMQX) * 8 * sizeof(native::block_q8_1);             \
    fused_grouped_ffn_mmq2_q4k_q8_1<(KMMQX)>                                  \
        <<<grid, block, smem, s>>>(d_w0, d_w1, act, out0, kN0, out1, kN1, m,  \
                                   kK);                                         \
  } while (0)

    switch (mmq_x) {
    case 8:
      LAUNCH_MMQ2(8);
      break;
    case 16:
      LAUNCH_MMQ2(16);
      break;
    case 24:
      LAUNCH_MMQ2(24);
      break;
    case 32:
      LAUNCH_MMQ2(32);
      break;
    case 48:
      LAUNCH_MMQ2(48);
      break;
    default:
      break;
    }
#undef LAUNCH_MMQ2
  };

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      launch_mmq2(stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      launch_mmq2(stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

BenchmarkResult RunMMQ3CandidateCase(int m) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0),
             w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1),
             w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * kQ8BlocksPerRow *
                               sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const auto *act = static_cast<const native::block_q8_1 *>(d_act_q8_1);
  half *out0 = d_out;
  half *out1 = d_out + static_cast<size_t>(m) * kN0;

  const auto launch_mmq3 = [&](cudaStream_t s) {
    constexpr int kRows = kRowsPerBlock3;
    const int max_n = kN0 > kN1 ? kN0 : kN1;
    const dim3 block(kWarps3 * 32);
    const dim3 grid((max_n + kWarps3 - 1) / kWarps3,
                    (m + kRows - 1) / kRows);
    const size_t smem =
        static_cast<size_t>(kRows) * 8 * sizeof(native::block_q8_1);
    fused_grouped_ffn_mmq3_q4k_q8_1<kRows>
        <<<grid, block, smem, s>>>(d_w0, d_w1, act, out0, kN0, out1, kN1, m,
                                   kK);
  };

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      launch_mmq3(stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      launch_mmq3(stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

float MaxAbsDiff(const std::vector<half> &lhs, const std::vector<half> &rhs,
                 bool verbose = false) {
  float max_diff = 0.0f;
  size_t max_idx = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(__half2float(lhs[i]) - __half2float(rhs[i]));
    if (diff > max_diff) {
      max_diff = diff;
      max_idx = i;
    }
  }
  if (verbose && max_diff > 0.0f) {
    const float lv = __half2float(lhs[max_idx]);
    const float rv = __half2float(rhs[max_idx]);
    std::printf("  max_diff at index %zu: generic=%.8f candidate=%.8f "
                "diff=%.8f\n",
                max_idx, lv, rv, max_diff);
    // Print first 5 differing elements
    int count = 0;
    for (size_t i = 0; i < lhs.size() && count < 5; ++i) {
      const float d = std::fabs(__half2float(lhs[i]) - __half2float(rhs[i]));
      if (d > 0.0f) {
        std::printf("  diff[%zu]: generic=%.8f candidate=%.8f diff=%.8f\n", i,
                    __half2float(lhs[i]), __half2float(rhs[i]), d);
        ++count;
      }
    }
  }
  return max_diff;
}

void PrintCaseSummary(const char *label, int m, const BenchmarkResult &generic,
                      const BenchmarkResult &candidate,
                      const char *candidate_name) {
  std::printf("Case: %s Q4_K M=%d N=%d/%d K=%d\n", label, m, kN0, kN1, kK);
  std::printf("  generic: %.3f ms\n", generic.mean_ms);
  std::printf("  %s: %8.3f ms\n", candidate_name, candidate.mean_ms);
  std::printf("  speedup: %.3fx\n", generic.mean_ms / candidate.mean_ms);
  std::printf("  max_abs_diff_vs_generic: %.6f\n",
              MaxAbsDiff(generic.output, candidate.output));
}

void PrintRawCaseSummary(const char *label, int m,
                         const BenchmarkResult &result, const char *mode_name) {
  std::printf("Case: %s Q4_K M=%d N=%d/%d K=%d\n", label, m, kN0, kN1, kK);
  std::printf("  %s: %.3f ms\n", mode_name, result.mean_ms);
}

} // namespace

int main(int argc, char **argv) {
#ifndef INFERFLUX_HAS_CUDA
  std::puts("benchmark_grouped_q81_pair requires InferFlux CUDA kernels");
  return 0;
#else
  bool force_v2 = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--v2") == 0) {
      force_v2 = true;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (force_v2) {
#ifdef _WIN32
    _putenv_s("INFERFLUX_GEMV_V2", "1");
#else
    setenv("INFERFLUX_GEMV_V2", "1", 1);
#endif
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::puts("No CUDA device available");
    return 0;
  }

  std::puts("========================================");
  std::puts("Grouped Q8_1 Pair Benchmark");
  std::puts("========================================");
  std::printf("Mode: %s\n", force_v2 ? "v2" : "v1");

  NativeExecutionPolicy generic_policy;
  generic_policy.enable_experimental_q81_grouped_hot_q4k = false;
  generic_policy.enable_experimental_q81_grouped_rowpair_w4 = false;

  NativeExecutionPolicy m1_hot_policy = generic_policy;
  m1_hot_policy.enable_experimental_q81_grouped_hot_q4k = true;

  NativeExecutionPolicy m2_rowpair_policy = generic_policy;
  m2_rowpair_policy.enable_experimental_q81_grouped_rowpair_w4 = true;

  const BenchmarkResult m1_generic = RunCase(1, generic_policy);
  const BenchmarkResult m2_generic = RunCase(2, generic_policy);
  const BenchmarkResult m3_generic = RunCase(3, generic_policy);
  const BenchmarkResult m4_generic = RunCase(4, generic_policy);
  const BenchmarkResult m8_generic = RunCase(8, generic_policy);
  const BenchmarkResult m16_generic = RunCase(16, generic_policy);
  const BenchmarkResult m24_generic = RunCase(24, generic_policy);
  const BenchmarkResult m48_generic = RunCase(48, generic_policy);

  if (force_v2) {
    PrintRawCaseSummary("generic", 1, m1_generic, "v2");
    PrintRawCaseSummary("generic", 2, m2_generic, "v2");
    PrintRawCaseSummary("generic", 3, m3_generic, "v2");
    PrintRawCaseSummary("generic", 4, m4_generic, "v2");
    PrintRawCaseSummary("generic", 8, m8_generic, "v2");
    PrintRawCaseSummary("generic", 16, m16_generic, "v2");
    PrintRawCaseSummary("generic", 24, m24_generic, "v2");
    return 0;
  }

  const BenchmarkResult m1_hot = RunCase(1, m1_hot_policy);
  const BenchmarkResult m2_rowpair = RunCase(2, m2_rowpair_policy);
  const BenchmarkResult m3_rowquad = RunRowQuadCandidateCase(3);
  const BenchmarkResult m4_rowquad = RunRowQuadCandidateCase(4);
  const BenchmarkResult m8_mmq = RunMmqTileCandidateCase(8, kMmqTileCols);
  const BenchmarkResult m16_mmq = RunMmqTileCandidateCase(16, kMmqTileCols);
  const BenchmarkResult m24_mmq = RunMmqTileCandidateCase(24, kMmqTileCols);
  const BenchmarkResult m24_mmq_wide = RunMmqTileCandidateCase(24, 16);
  const BenchmarkResult m48_mmq = RunMmqTileCandidateCase(48, kMmqTileCols);
  const BenchmarkResult m48_mmq_wide = RunMmqTileCandidateCase(48, 16);
  const BenchmarkResult m24_mmq_layout =
      RunMmqLayoutCandidateCase(24, kMmqTileCols);
  const BenchmarkResult m24_mmq_layout_wide = RunMmqLayoutCandidateCase(24, 16);
  const BenchmarkResult m48_mmq_layout =
      RunMmqLayoutCandidateCase(48, kMmqTileCols);
  const BenchmarkResult m48_mmq_layout_wide = RunMmqLayoutCandidateCase(48, 16);
  const BenchmarkResult m24_mmq2 = RunMMQ2CandidateCase(24, 24);
  const BenchmarkResult m24_mmq2_32 = RunMMQ2CandidateCase(24, 32);
  const BenchmarkResult m48_mmq2 = RunMMQ2CandidateCase(48, 48);
  const BenchmarkResult m48_mmq2_32 = RunMMQ2CandidateCase(48, 32);
  PrintCaseSummary("single-row hot", 1, m1_generic, m1_hot, "hot");
  std::puts("");
  PrintCaseSummary("row-pair", 2, m2_generic, m2_rowpair, "rowpair");
  std::puts("");
  PrintCaseSummary("row-quad", 3, m3_generic, m3_rowquad, "rowquad");
  std::puts("");
  PrintCaseSummary("row-quad confirmation", 4, m4_generic, m4_rowquad,
                   "rowquad");
  std::puts("");
  PrintCaseSummary("mmq-style tile", 8, m8_generic, m8_mmq, "mmq_tile");
  std::puts("");
  PrintCaseSummary("mmq-style tile", 16, m16_generic, m16_mmq, "mmq_tile");
  std::puts("");
  PrintCaseSummary("mmq-style tile", 24, m24_generic, m24_mmq, "mmq_tile");
  std::puts("");
  PrintCaseSummary("mmq-style tile wide", 24, m24_generic, m24_mmq_wide,
                   "mmq_tile16");
  std::puts("");
  PrintCaseSummary("mmq-layout tile", 24, m24_generic, m24_mmq_layout,
                   "mmq_layout");
  std::puts("");
  PrintCaseSummary("mmq-layout tile wide", 24, m24_generic,
                   m24_mmq_layout_wide, "mmq_layout16");
  std::puts("");
  PrintCaseSummary("mmq-style tile", 48, m48_generic, m48_mmq, "mmq_tile");
  std::puts("");
  PrintCaseSummary("mmq-style tile wide", 48, m48_generic, m48_mmq_wide,
                   "mmq_tile16");
  std::puts("");
  PrintCaseSummary("mmq-layout tile", 48, m48_generic, m48_mmq_layout,
                   "mmq_layout");
  std::puts("");
  PrintCaseSummary("mmq-layout tile wide", 48, m48_generic,
                   m48_mmq_layout_wide, "mmq_layout16");
  std::puts("");
  const BenchmarkResult m1_mmq3 = RunMMQ3CandidateCase(1);
  const BenchmarkResult m2_mmq3 = RunMMQ3CandidateCase(2);
  const BenchmarkResult m3_mmq3 = RunMMQ3CandidateCase(3);
  const BenchmarkResult m4_mmq3 = RunMMQ3CandidateCase(4);
  const BenchmarkResult m8_mmq3 = RunMMQ3CandidateCase(8);
  const BenchmarkResult m16_mmq3 = RunMMQ3CandidateCase(16);
  const BenchmarkResult m24_mmq3 = RunMMQ3CandidateCase(24);
  const BenchmarkResult m48_mmq3 = RunMMQ3CandidateCase(48);
  std::puts("");
  std::puts("--- MMQ3 (smem-rowquad) Results ---");
  PrintCaseSummary("mmq3 smem-rowquad", 1, m1_generic, m1_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 2, m2_generic, m2_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 3, m3_generic, m3_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 4, m4_generic, m4_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 8, m8_generic, m8_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 16, m16_generic, m16_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 24, m24_generic, m24_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq3 smem-rowquad", 48, m48_generic, m48_mmq3, "mmq3");
  std::puts("");
  PrintCaseSummary("mmq2 2D-tile (kMMQX=24)", 24, m24_generic, m24_mmq2,
                   "mmq2_24");
  std::puts("");
  PrintCaseSummary("mmq2 2D-tile (kMMQX=48)", 48, m48_generic, m48_mmq2,
                   "mmq2_48");
  return 0;
#endif
}
