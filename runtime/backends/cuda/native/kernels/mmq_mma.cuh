#pragma once

// MMQ MMA: int8 tensor-core quantized GEMM for Q6_K weights × Q8_1-mmq
// activations. This is the M>1 decode path llama.cpp uses on sm_75+; our
// dp4a MMVQ/MMQ kernels measure 4-8x off llama at M 9-16 because they
// cannot use tensor cores.
//
// Port of the TURING_MMA branch of llama.cpp's ggml-cuda/mmq.cuh (MIT,
// ggml contributors): load_tiles_q6_K, vec_dot_q6_K_q8_1_mma, and a
// conventional-tiling driver (stream-K/MoE/channel machinery deliberately
// not carried over). Activation layout block_q8_1_mmq (D4) ported from
// ggml-cuda/quantize.cu.
//
// Layout notes (match llama exactly):
//   x_qs tile row pitch = MMQ_MMA_TILE_X_K_Q6_K = 75 ints (64 quant +
//   padding to avoid bank conflicts); kq0/kq1 split each 256-elem
//   super-block into two 128-elem halves.
//   y tile pitch = MMQ_TILE_Y_K = 40 ints (32 quant + 8 scale floats).
//   dB = y_df/2 compensates Q6_K's 64-value scale groups against the
//   128-value y groups — calibrated, keep as-is.

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// ---------------------------------------------------------------------------
// Activation layout: D4 — one float scale per 32 values, 128 values per
// block (4 super-blocks of the Q8_1 32-value granularity).
// ---------------------------------------------------------------------------
struct BlockQ8_1Mmq {
  float d4[4];        // 1 32-bit scale per 32 values
  int8_t qs[4 * 32]; // 128 values
};
static_assert(sizeof(BlockQ8_1Mmq) == 144, "BlockQ8_1Mmq layout");

constexpr int kMmqTileNeK = 32;  // K slice per staged chunk (ints of quants)
constexpr int kMmqTileYK =
    kMmqTileNeK + kMmqTileNeK / 8; // 32 quants + 4 scale floats = 36
// llama formula: 2*NE_K + NE_K/QI6_K + NE_K/8 + 7 = 64+1+4+7 = 76 ints.
// 76*4 = 304 bytes % 16 == 0: every row 16B-aligned for ldmatrix (the
// +7 padding exists precisely for this).
constexpr int kMmqMmaTileXKQ6K = 76;
static_assert(kMmqMmaTileXKQ6K % 8 == 4, "pitch must be 4 mod 8");
constexpr int kMmqIterK = 256;
constexpr int kMmqY = 128;    // weight rows per block (sm >= Volta)
constexpr int kMmqMmaWarps = 8;  // warps per block (blockDim = dim3(32, 8))
constexpr int kMmqMmaMaxSplits = 8; // K-split cap (partials buffer sizing)

// Dynamic shared-memory requirement (ints) for the mmq_x=16 kernel.
constexpr int MmqSmemInts(int mmq_x) {
  const int pad = ((mmq_x * kMmqTileYK + 255) / 256) * 256;
  return pad + kMmqY * kMmqMmaTileXKQ6K;
}

// ---------------------------------------------------------------------------
// D4 quantizer: half[M, K] -> BlockQ8_1Mmq[M, K/128]. 128-thread blocks
// quantize four 128-value groups each (one warp per group); each thread
// handles 4 values, scale reduced across the 8 lanes of its 32-value
// sub-group.
// ---------------------------------------------------------------------------
static __global__ void QuantizeRowQ8_1MmqKernel(const half *__restrict__ x,
                                          BlockQ8_1Mmq *__restrict__ y, int K,
                                          int total_rows) {
  const int row = blockIdx.y;
  if (row >= total_rows)
    return;
  const int t = threadIdx.x; // 0..127
  const int groups_per_row = K / 128;
  const int group = blockIdx.x * 4 + t / 32;
  if (group >= groups_per_row) {
    return; // tail block when groups_per_row % 4 != 0
  }
  // Group-major layout [K/128][rows]: every 128-value group of all rows is
  // contiguous, so the MMQ driver stages a whole activation tile as one
  // contiguous chunk per K-group.
  BlockQ8_1Mmq &grp = y[static_cast<size_t>(group) * total_rows + row];
  const int lane = t % 32; // 4 values per lane: 32 lanes = 128 values

  const int base = row * K + group * 128 + 4 * lane;
  const half2 h01 = *reinterpret_cast<const half2 *>(&x[base + 0]);
  const half2 h23 = *reinterpret_cast<const half2 *>(&x[base + 2]);
  const float vals[4] = {__half2float(__low2half(h01)),
                         __half2float(__high2half(h01)),
                         __half2float(__low2half(h23)),
                         __half2float(__high2half(h23))};

  float amax = fabsf(vals[0]);
  amax = fmaxf(amax, fabsf(vals[1]));
  amax = fmaxf(amax, fabsf(vals[2]));
  amax = fmaxf(amax, fabsf(vals[3]));
  // Sub-group = 32 values = 8 lanes of 4.
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
  }

  const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  char4 q;
  q.x = __float2int_rn(vals[0] * d_inv);
  q.y = __float2int_rn(vals[1] * d_inv);
  q.z = __float2int_rn(vals[2] * d_inv);
  q.w = __float2int_rn(vals[3] * d_inv);
  reinterpret_cast<char4 *>(grp.qs)[lane] = q;

  // Scale slots are group-relative: lanes 0/8/16/24 of the group's warp.
  if (lane % 8 == 0) {
    grp.d4[lane / 8] = amax > 0.0f ? 1.0f / d_inv : 0.0f;
  }
}

// Fused SwiGLU producer: silu(gate) * up quantized straight to D4, same
// thread mapping and group-major layout as QuantizeRowQ8_1MmqKernel.
static __global__ void SiluMulQuantizeQ8_1MmqKernel(const half *__restrict__ gate,
                                             const half *__restrict__ up,
                                             BlockQ8_1Mmq *__restrict__ y,
                                             int K, int total_rows) {
  const int row = blockIdx.y;
  if (row >= total_rows)
    return;
  const int t = threadIdx.x; // 0..127
  const int groups_per_row = K / 128;
  const int group = blockIdx.x * 4 + t / 32;
  if (group >= groups_per_row) {
    return;
  }
  BlockQ8_1Mmq &grp = y[static_cast<size_t>(group) * total_rows + row];
  const int lane = t % 32;

  const int base = row * K + group * 128 + 4 * lane;
  float vals[4];
#pragma unroll
  for (int j = 0; j < 4; j += 2) {
    const half2 g2 = *reinterpret_cast<const half2 *>(&gate[base + j]);
    const half2 u2 = *reinterpret_cast<const half2 *>(&up[base + j]);
    const float g0 = __half2float(__low2half(g2));
    const float g1 = __half2float(__high2half(g2));
    vals[j] = g0 * __half2float(__low2half(u2)) /
              (1.0f + __expf(-g0));
    vals[j + 1] = g1 * __half2float(__high2half(u2)) /
                  (1.0f + __expf(-g1));
  }

  float amax = fabsf(vals[0]);
  amax = fmaxf(amax, fabsf(vals[1]));
  amax = fmaxf(amax, fabsf(vals[2]));
  amax = fmaxf(amax, fabsf(vals[3]));
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
  }

  const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  char4 q;
  q.x = __float2int_rn(vals[0] * d_inv);
  q.y = __float2int_rn(vals[1] * d_inv);
  q.z = __float2int_rn(vals[2] * d_inv);
  q.w = __float2int_rn(vals[3] * d_inv);
  reinterpret_cast<char4 *>(grp.qs)[lane] = q;

  if (lane % 8 == 0) {
    grp.d4[lane / 8] = amax > 0.0f ? 1.0f / d_inv : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Weight staging: Q6_K -> int8 x_qs tile + float x_df + packed int scales.
// Port of load_tiles_q6_K (mmq.cuh:2289-2360), MMA branch.
// ---------------------------------------------------------------------------
template <int mmq_y>
__device__ __forceinline__ void
LoadTilesQ6KMma(const char *__restrict__ x, int *__restrict__ x_tile,
                int kbx0, int i_max, int stride) {
  // Exact port of load_tiles_q6_K, TURING branch (mmq.cuh:2289-2360).
  // Constants: QI6_K = QK_K/(4*QR6_K) = 32; threads_per_row =
  // MMQ_ITER_K/(4*QR6_K) = 32; nrows = 1.
  constexpr int warp_size = 32;
  int *x_qs = x_tile;
  float *x_df = reinterpret_cast<float *>(x_qs + kMmqTileNeK * 2);
  int *x_sc = reinterpret_cast<int *>(x_df + kMmqTileNeK / 32);

  constexpr int QI6 = 32;
  constexpr int threads_per_row = 32;
  constexpr int nrows = warp_size / threads_per_row;
  const int txi = threadIdx.x % threads_per_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nrows * kMmqMmaWarps) {
    const int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows +
                                             threadIdx.x / threads_per_row);
    if (i > i_max) {
      break;
    }

    const block_q6_k *bxi =
        reinterpret_cast<const block_q6_k *>(x) + kbx0 + i * stride;

    // get_int_b2 semantics: 4 bytes at offset 4*i (llama reads two
    // uint16s at 2*i and 2*i+1) — NOT 2*i bytes.
    const int ql = LoadPackedInt32Unaligned(&bxi->ql[4 * txi]);
    const int ql0 = (ql >> 0) & 0x0F0F0F0F;
    const int ql1 = (ql >> 4) & 0x0F0F0F0F;

    const int qh = LoadPackedInt32Unaligned(
        &bxi->qh[4 * ((QI6 / 4) * (txi / (QI6 / 2)) + txi % (QI6 / 4))]);
    const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
    const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;

    const int kq0 = 2 * txi - txi % (QI6 / 2) + 0;
    const int kq1 = 2 * txi - txi % (QI6 / 2) + QI6 / 2;

    x_qs[i * kMmqMmaTileXKQ6K + kq0] = Vsubss4(ql0 | qh0, 0x20202020);
    x_qs[i * kMmqMmaTileXKQ6K + kq1] = Vsubss4(ql1 | qh1, 0x20202020);
  }

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += kMmqMmaWarps * warp_size) {
    const int i = (i0 + threadIdx.y * warp_size + threadIdx.x) % mmq_y;
    if (i > i_max) {
      break;
    }
    const block_q6_k *bxi =
        reinterpret_cast<const block_q6_k *>(x) + kbx0 + i * stride;
    // Per-row placement: the float lives in row i's scale region (row
    // offset 64), NOT a flat float array.
    x_df[i * kMmqMmaTileXKQ6K] =
        __half2float(*reinterpret_cast<const half *>(&bxi->d));
  }

  constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += kMmqMmaWarps * rows_per_warp) {
    const int i = (i0 + threadIdx.y * rows_per_warp +
                   threadIdx.x / (kMmqTileNeK / 8)) %
                  mmq_y;
    if (i > i_max) {
      break;
    }
    const block_q6_k *bxi =
        reinterpret_cast<const block_q6_k *>(x) + kbx0 + i * stride +
        (threadIdx.x % (kMmqTileNeK / 8)) / 4;
    x_sc[i * kMmqMmaTileXKQ6K + threadIdx.x % 4] =
        LoadPackedInt32Unaligned(&bxi->scales[4 * (threadIdx.x % 4)]);
  }
}

// ---------------------------------------------------------------------------
// MMA vec_dot: per-warp A(16x8 weights) x B(16x8 acts) -> C(16x16),
// scales applied in float outside the tensor core.
// Port of vec_dot_q6_K_q8_1_mma (mmq.cuh:2406-2477), granularity=32
// (mmq_x=16 path -> rows_per_warp = 32, ntx = 2).
// ---------------------------------------------------------------------------
template <int mmq_x>
__device__ __forceinline__ void
VecDotQ6KQ8_1Mma(const int *__restrict__ x, const int *__restrict__ y,
                 float *__restrict__ sum, int k00) {
  namespace m = mma;
  using TileA = m::Tile<16, 4, int>;
  using TileB = m::Tile<8, 4, int>;
  using TileC = m::Tile<16, 8, int>;

  // TURING granularity: 8 for mmq_x < 48 -> rows_per_warp = 16, ntx = 1.
  constexpr int granularity = 8;
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / TileC::I;

  y += (threadIdx.y % ntx) * (TileC::J * kMmqTileYK);

  const int *x_qs = x;
  const float *x_df = reinterpret_cast<const float *>(x_qs) + kMmqTileNeK * 2;
  const int *x_sc = reinterpret_cast<const int *>(x_df + kMmqTileNeK / 32);
  const int *y_qs = y + 4;
  const float *y_df = reinterpret_cast<const float *>(y);

  const int i0 = (threadIdx.y / ntx) * (ntx * TileA::I);

  TileA a[ntx][8];
  int sc_a[ntx][TileC::ne / 2][8];
  float d_a[ntx][TileC::ne / 2];

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < kMmqTileNeK; k01 += 8) {
      const int k0 = k00 + k01;
      m::LoadLdmatrix(
          a[n][k01 / 4 + 0],
          reinterpret_cast<const int *>(
              x_qs + (i0 + n * TileA::I) * kMmqMmaTileXKQ6K + (k0 + 0)),
          kMmqMmaTileXKQ6K);
      m::LoadLdmatrix(
          a[n][k01 / 4 + 1],
          reinterpret_cast<const int *>(
              x_qs + (i0 + n * TileA::I) * kMmqMmaTileXKQ6K +
              (k0 + TileA::J)),
          kMmqMmaTileXKQ6K);
    }

#pragma unroll
    for (int k01 = 0; k01 < kMmqTileNeK; k01 += 16) {
      const int k0 = k00 + k01;
#pragma unroll
      for (int l = 0; l < TileC::ne / 2; ++l) {
        const int i = i0 + n * TileC::I + TileC::get_i(2 * l);
        const int sc_packed =
            x_sc[i * kMmqMmaTileXKQ6K + k0 / 16];
        const int8_t *sc = reinterpret_cast<const int8_t *>(&sc_packed);
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) {
          sc_a[n][l][k01 / 4 + ksc] = sc[ksc];
        }
      }
    }

#pragma unroll
    for (int l = 0; l < TileC::ne / 2; ++l) {
      const int i = i0 + n * TileC::I + TileC::get_i(2 * l);
      d_a[n][l] = x_df[i * kMmqMmaTileXKQ6K];
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * TileC::J) {
    float tmp[ntx][TileC::ne] = {{0.0f}};

#pragma unroll
    for (int k01 = 0; k01 < kMmqTileNeK; k01 += 8) {
      TileB b[2];
      float db[TileC::ne / 2];

      // llama notes load_generic beats ldmatrix here for B.
      m::LoadGeneric(b[0], y_qs + j0 * kMmqTileYK + 0 + k01, kMmqTileYK);
      m::LoadGeneric(b[1], y_qs + j0 * kMmqTileYK + TileB::J + k01,
                     kMmqTileYK);

#pragma unroll
      for (int l = 0; l < TileC::ne / 2; ++l) {
        const int j = j0 + TileC::get_j(l);
        db[l] = y_df[j * kMmqTileYK + k01 / 8];
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        TileC c[2];
        m::MmaS8(c[0], a[n][k01 / 4 + 0], b[0]);
        m::MmaS8(c[1], a[n][k01 / 4 + 1], b[1]);
#pragma unroll
        for (int l = 0; l < TileC::ne; ++l) {
          tmp[n][l] += (c[0].x[l] * sc_a[n][l / 2][k01 / 4 + 0] +
                        c[1].x[l] * sc_a[n][l / 2][k01 / 4 + 1]) *
                       db[l % 2];
        }
      }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < TileC::ne; ++l) {
        sum[(j0 / TileC::J + n) * TileC::ne + l] += tmp[n][l] * d_a[n][l / 2];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Driver: conventional tiling. Grid (ceil(N/mmq_y), ceil(M/mmq_x)).
// __restrict__-clean; output half[M, N].
// ---------------------------------------------------------------------------
// Deterministic K-split reduce: partials[S][M][N] fp32 -> half[M][N],
// summed in fixed z order (no atomics, bit-stable across runs).
static __global__ void ReduceMmqKSplit(const float *__restrict__ partials,
                                half *__restrict__ out, int splits,
                                size_t mn) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= mn) {
    return;
  }
  float acc = 0.0f;
  for (int s = 0; s < splits; ++s) {
    acc += partials[static_cast<size_t>(s) * mn + idx];
  }
  out[idx] = __float2half(acc);
}

template <int mmq_x>
__global__ void __launch_bounds__(kMmqMmaWarps * 32, 1)
    InferfluxMmqQ6KMma(const char *__restrict__ w,
                       const BlockQ8_1Mmq *__restrict__ act, half *__restrict__ out,
                       int N, int K, int M, float *__restrict__ partials,
                       int ksplits) {
  constexpr int warp_size = 32;
  constexpr int QK = 256; // QK_K

  extern __shared__ int smem[];
  int *tile_y = smem;
  // llama pads the y tile to a whole wave of threads before tile_x —
  // without it the weight stage overwrites the activation tail.
  constexpr int kPad = ((mmq_x * kMmqTileYK + 255) / 256) * 256;
  int *tile_x = tile_y + kPad;

  const int blocks_per_row = K / QK;
  const int it = blockIdx.x;             // weight-row tile
  const int jt = blockIdx.y;             // activation-column tile
  const int tile_x_max_i = N - it * kMmqY - 1;
  const int tile_y_max_j = M - jt * mmq_x - 1;

  float sum[mmq_x * kMmqY / (kMmqMmaWarps * warp_size)] = {0};

  const char *x = w + static_cast<size_t>(it) * kMmqY * blocks_per_row *
                          sizeof(block_q6_k);
  constexpr int sz = sizeof(BlockQ8_1Mmq) / sizeof(int);
  static_assert(sz == kMmqTileYK, "row stride must equal tile Y pitch");
  // y layout: group-major [K/128][M rows][36 ints]; the tile stages rows
  // [jt*mmq_x, +mmq_x) of K-group kb0 as one contiguous chunk.
  const int *y_base = reinterpret_cast<const int *>(act);

  // K-split: slice z covers super-blocks [kb_begin, kb_end). ksplits == 1
  // (partials == nullptr) writes half outputs directly.
  const int sb_per_split = (blocks_per_row + ksplits - 1) / ksplits;
  const int kb_begin = blockIdx.z * sb_per_split;
  const int kb_end = min(kb_begin + sb_per_split, blocks_per_row);

  for (int kb0 = kb_begin; kb0 < kb_end; ++kb0) {
    LoadTilesQ6KMma<kMmqY>(x, tile_x, kb0, tile_x_max_i, blocks_per_row);
    // Two 128-value chunks per 256-value super-block: vec_dot consumes
    // kMmqTileNeK=32 int-cols (128 int8) per call, x_qs holds 64 — llama
    // stages and consumes the second chunk in the same kb0 iteration.
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      {
        const int *src =
            y_base +
            (static_cast<size_t>(kb0 * QK / 128 + chunk) * M + jt * mmq_x) *
                sz;
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * kMmqTileYK;
             l0 += kMmqMmaWarps * warp_size) {
          const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
          if (l < mmq_x * kMmqTileYK) {
            tile_y[l] = src[l];
          }
        }
      }
      __syncthreads();
      VecDotQ6KQ8_1Mma<mmq_x>(tile_x, tile_y, sum, chunk * kMmqTileNeK);
      __syncthreads();
    }
  }

#ifdef INFERFLUX_MMA_DEBUG_SUM
  if (blockIdx.x == 0 && blockIdx.y == 0) {
    float *dbg = reinterpret_cast<float *>(out) + M * N;
    const int slot = (threadIdx.y * 32 + threadIdx.x) * 8;
    for (int s = 0; s < 8; ++s) dbg[slot + s] = sum[s];
  }
#endif

  // Write back: warp w owns rows [w*32/ntx ... ] — mirror llama's
  // mmq_write_back_mma mapping for tile_C<16,16>, rows_per_warp=32.
  namespace m = mma;
  using TileC = m::Tile<16, 8, int>;
  constexpr int ntx = 1; // TURING granularity 8 -> rows_per_warp 16
  const int i0 = (threadIdx.y / ntx) * (ntx * TileC::I);

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * TileC::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < TileC::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * TileC::J + TileC::get_j(l);
        const int i = i0 + n * TileC::I + TileC::get_i(l);
        if (i > tile_x_max_i || j > tile_y_max_j) {
          continue;
        }
        if (ksplits == 1) {
          out[static_cast<size_t>(j) * N + it * kMmqY + i] =
              __float2half(sum[(j0 / TileC::J + n) * TileC::ne + l]);
        } else {
          partials[(static_cast<size_t>(blockIdx.z) * M + j) * N +
                   it * kMmqY + i] =
              sum[(j0 / TileC::J + n) * TileC::ne + l];
        }
      }
    }
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
