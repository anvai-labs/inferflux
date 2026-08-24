#pragma once

// MMA tensor-core flash attention kernel for prefill (query_len >= 16).
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 for the Q*K^T
// dot product (the dominant cost for long sequences). V accumulation remains
// scalar — avoids the V-transpose complexity while capturing the MMA benefit.
//
// Grid:  (ceil(query_len / 16), num_kv_heads, batch_size)
// Block: 128 threads = 4 warps
// Each block processes Br=16 Q positions against all KV tiles (Bc=64).

#include "runtime/backends/cuda/common/dtype_traits.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// Tile sizes
static constexpr int MMA_FA2_BR = 16;    // Q rows per block
static constexpr int MMA_FA2_BC = 64;    // KV positions per tile
static constexpr int MMA_FA2_WARPS = 4;  // 128 threads
static constexpr int MMA_FA2_THREADS = MMA_FA2_WARPS * 32;

// ============================================================================
// MMA m16n8k16 inline assembly wrapper
// ============================================================================

// Compute D += A * B using m16n8k16 tensor core MMA.
// A: 4 x 32-bit registers (16x16 half values, row-major)
// B: 2 x 32-bit registers (8x16 half values, column-major)
// D: 4 x 32-bit registers (16x8 f32 values, accumulator)
__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    int (&D)[4], const int (&A)[4], const int (&B)[2]) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#elif __CUDA_ARCH__ >= 750
  // Turing fallback: 2x m16n8k8
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
      : "r"(A[2]), "r"(A[3]), "r"(B[1]));
#else
  // Volta and earlier: no MMA support. This path should never execute
  // because the host wrapper selects scalar kernels for SM < 7.5.
  (void)A;
  (void)B;
#endif
}

// ============================================================================
// MMA fragment <-> shared memory helpers
//
// For m16n8k16 I_MAJOR layout (Turing/Ampere/Ada):
//   tile<16, 8, float>: ne = 4 floats per thread
//     get_i(l) = ((l / 2) * 8) + (threadIdx.x / 4)    (within warp: % 32)
//     get_j(l) = ((threadIdx.x % 4) * 2) + (l % 2)
//
//   tile<16, 8, half2>: ne = 4 half2 per thread (= 8 half values)
//     Same index pattern as float tile.
// ============================================================================

// Extract (row, col) for the lth fragment element in an m16n8 output tile.
__device__ __forceinline__ int mma_frag_row(int l, int lane) {
  return ((l / 2) * 8) + (lane / 4);
}

__device__ __forceinline__ int mma_frag_col(int l, int lane) {
  return ((lane % 4) * 2) + (l % 2);
}

// Load a [16 x 16] half sub-matrix from shared memory into 4 MMA A registers.
// src: shared memory pointer to [rows x stride] half array (row-major).
// row0: starting row in src. k0: starting column (K dim offset).
// stride: number of half elements per row.
__device__ __forceinline__ void load_mma_a_16x16(
    int (&A)[4], const half *src, int row0, int k0, int stride) {
  const int lane = threadIdx.x % 32;
  // A fragment layout for m16n8k16, I_MAJOR:
  //   A has 4 half2 per thread (8 half values covering 16x16 logical).
  //   Register l holds half2 at logical position:
  //     row = ((l / 2) * 8) + (lane / 4)
  //     col = ((lane % 4) * 2) + (l % 2) * ... but for A (K dim), it's packed
  // Use the NVIDIA-specified layout: thread (lane) holds:
  //   A[0] = {src[row0 + lane/4, k0 + (lane%4)*2], src[row0 + lane/4, k0 + (lane%4)*2 + 1]}
  //   A[1] = {src[row0 + lane/4, k0 + (lane%4)*2 + 8], src[row0 + lane/4, k0 + (lane%4)*2 + 9]}
  //   A[2] = {src[row0 + lane/4 + 8, k0 + (lane%4)*2], ...}
  //   A[3] = {src[row0 + lane/4 + 8, k0 + (lane%4)*2 + 8], ...}
  const int row_lo = row0 + lane / 4;
  const int row_hi = row_lo + 8;
  const int col_base = k0 + (lane % 4) * 2;

  const half *r0 = src + row_lo * stride + col_base;
  const half *r1 = src + row_lo * stride + col_base + 8;
  const half *r2 = src + row_hi * stride + col_base;
  const half *r3 = src + row_hi * stride + col_base + 8;

  A[0] = *reinterpret_cast<const int *>(r0);
  A[1] = *reinterpret_cast<const int *>(r1);
  A[2] = *reinterpret_cast<const int *>(r2);
  A[3] = *reinterpret_cast<const int *>(r3);
}

// Load a [16 x 8] half sub-matrix (column-major B operand) from shared memory.
// For Q*K^T: B = K^T, so we read K in row-major [kv_pos, head_dim] and need
// the [K=16, N=8] column-major view. K is stored row-major with stride=head_dim.
// We want B[k, n] = K[kv_base + n, k_offset + k] — transposed load.
__device__ __forceinline__ void load_mma_b_transposed_16x8(
    int (&B)[2], const half *k_smem, int kv_base, int k_offset, int stride) {
  const int lane = threadIdx.x % 32;
  // B fragment layout for m16n8k16, I_MAJOR:
  //   B has 2 half2 per thread.
  //   B[0] = {K[kv_base + lane/4, k_offset + (lane%4)*2],
  //           K[kv_base + lane/4, k_offset + (lane%4)*2 + 1]}
  //   B[1] = {K[kv_base + lane/4, k_offset + (lane%4)*2 + 8],
  //           K[kv_base + lane/4, k_offset + (lane%4)*2 + 9]}
  // Wait — that's wrong. B is column-major [K=16, N=8].
  // B[k, n] corresponds to K^T[k, n] = K[n, k].
  // So B fragment loads: the PTX spec says for B column-major:
  //   lane holds elements at specific (k, n) positions.
  // For B (col-major) with m16n8k16:
  //   B[0] = packed half2 at: row = (lane % 4)*2 .. (lane%4)*2+1, col = lane/4
  //   B[1] = same but row += 8

  // K^T[k, n] = K[n, k_offset + k]. Read from K row-major smem:
  const int n = lane / 4;       // 0..7 (column in B = KV position)
  const int k_lo = (lane % 4) * 2;
  const int k_hi = k_lo + 8;

  const half *src0_a = k_smem + (kv_base + n) * stride + (k_offset + k_lo);
  const half *src1_a = k_smem + (kv_base + n) * stride + (k_offset + k_hi);

  B[0] = *reinterpret_cast<const int *>(src0_a);
  B[1] = *reinterpret_cast<const int *>(src1_a);
}

// ============================================================================
// MMA FlashAttention-2 prefill kernel
//
// Computes Q*K^T using MMA tensor cores, then scalar online softmax + V
// accumulation. Each block handles Br=16 Q rows.
// ============================================================================

template <typename T, int GQARatio>
__global__ void FlashAttention2MMAGQAKernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, int query_len, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, float scale, bool causal) {
  const int batch_idx = blockIdx.z;
  const int kv_head_idx = blockIdx.y;
  const int q_block = blockIdx.x; // Which block of 16 Q positions
  const int q_start = q_block * MMA_FA2_BR;

  if (q_start >= query_len)
    return;
  const int q_end = min(q_start + MMA_FA2_BR, query_len);
  const int q_count = q_end - q_start; // May be < 16 for last block

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid % 32;

  const int q_stride = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  // ---- Shared memory layout ----
  // s_q:      [MMA_FA2_BR * head_dim] half   — Q tile (loaded once)
  // s_k:      [MMA_FA2_BC * head_dim] half   — K tile (per KV tile)
  // s_v:      [MMA_FA2_BC * head_dim] float  — V tile as float (per KV tile)
  // s_scores: [GQARatio * MMA_FA2_BR * MMA_FA2_BC] float — attention scores
  extern __shared__ char smem_raw[];
  half *s_q = reinterpret_cast<half *>(smem_raw);
  half *s_k = s_q + MMA_FA2_BR * head_dim;
  float *s_v = reinterpret_cast<float *>(s_k + MMA_FA2_BC * head_dim);
  float *s_scores =
      s_v + MMA_FA2_BC * head_dim; // [GQARatio][MMA_FA2_BR][MMA_FA2_BC]

  // ---- Per-head output accumulators ----
  // Each thread handles one dimension d (tid < head_dim) for ALL Q rows and
  // ALL GQA heads. Register pressure: GQARatio * MMA_FA2_BR floats.
  float o_acc[GQARatio][MMA_FA2_BR];
  float row_max[GQARatio][MMA_FA2_BR];
  float row_sum[GQARatio][MMA_FA2_BR];
#pragma unroll
  for (int h = 0; h < GQARatio; h++) {
#pragma unroll
    for (int q = 0; q < MMA_FA2_BR; q++) {
      o_acc[h][q] = 0.0f;
      row_max[h][q] = -INFINITY;
      row_sum[h][q] = 0.0f;
    }
  }

  // ---- Load Q tile into shared memory [MMA_FA2_BR * head_dim] ----
  // Load for all GQA heads. Q data for different heads loaded on demand from
  // s_q_head (we share Q smem across heads since we process heads sequentially
  // within the same softmax/V-accumulation pass).
  // Actually, for GQA: all heads sharing a KV head have DIFFERENT Q vectors.
  // We need per-head Q. To fit in smem, we load Q per-head on demand.
  //
  // For the MMA path, Q needs to be in half format in smem for fragment loads.
  // We'll load Q for each head into s_q before the MMA phase.

  const size_t kv_base_offset =
      (size_t)batch_idx * kv_len * kv_stride + kv_head_idx * head_dim;

  // ---- Main KV tile loop ----
  for (int h = 0; h < GQARatio; h++) {
    const int head_idx = kv_head_idx * GQARatio + h;
    if (head_idx >= num_heads)
      break;

    // Determine causal limit for each Q row (the max applies to all rows in
    // the block, but per-row masking is applied during score consumption).
    const int block_causal_limit =
        causal ? (kv_len - query_len + q_end) : kv_len;

    // Load Q for this head into s_q
    const int q_total_elems = q_count * head_dim;
    for (int i = tid; i < q_total_elems; i += MMA_FA2_THREADS) {
      const int qr = i / head_dim;
      const int qd = i % head_dim;
      const size_t q_offset = (size_t)batch_idx * query_len * q_stride +
                              (size_t)(q_start + qr) * q_stride +
                              head_idx * head_dim + qd;
      s_q[qr * head_dim + qd] =
          __float2half(DtypeTraits<T>::to_float(Q[q_offset]));
    }
    // Zero-pad if q_count < MMA_FA2_BR
    if (q_count < MMA_FA2_BR) {
      for (int i = tid; i < (MMA_FA2_BR - q_count) * head_dim;
           i += MMA_FA2_THREADS) {
        s_q[q_count * head_dim + i] = __float2half(0.0f);
      }
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < block_causal_limit;
         kv_start += MMA_FA2_BC) {
      const int tile_len = min(MMA_FA2_BC, block_causal_limit - kv_start);
      const int kv_total_elems = tile_len * head_dim;

      // Phase 1: Load K tile [tile_len, head_dim] as half
      for (int i = tid; i < kv_total_elems; i += MMA_FA2_THREADS) {
        const int t = i / head_dim;
        const int dim = i % head_dim;
        const size_t kv_offset = kv_base_offset +
                                 (size_t)(kv_start + t) * kv_stride + dim;
        s_k[t * head_dim + dim] =
            __float2half(DtypeTraits<T>::to_float(K[kv_offset]));
      }
      // Zero-pad K if tile_len < MMA_FA2_BC
      if (tile_len < MMA_FA2_BC) {
        for (int i = tid; i < (MMA_FA2_BC - tile_len) * head_dim;
             i += MMA_FA2_THREADS) {
          s_k[tile_len * head_dim + i] = __float2half(0.0f);
        }
      }
      // Load V tile as float for scalar accumulation
      for (int i = tid; i < kv_total_elems; i += MMA_FA2_THREADS) {
        const int t = i / head_dim;
        const int dim = i % head_dim;
        const size_t kv_offset = kv_base_offset +
                                 (size_t)(kv_start + t) * kv_stride + dim;
        s_v[t * head_dim + dim] = DtypeTraits<T>::to_float(V[kv_offset]);
      }
      __syncthreads();

      // Phase 2: MMA Q*K^T → s_scores[MMA_FA2_BR, MMA_FA2_BC]
      //
      // Each warp handles 16 KV positions (2 MMA N-tiles of 8).
      // warp_id → KV positions [warp_id*16 .. warp_id*16+15]
      // For each N-tile of 8 KV positions, iterate over K in steps of 16.
      {
        const int kv_warp_base = warp_id * 16;

        // Two N-tiles per warp
        for (int nt = 0; nt < 2; nt++) {
          const int kv_n_base = kv_warp_base + nt * 8;

          // MMA accumulator [16x8] — 4 floats per thread
          int D[4] = {0, 0, 0, 0};

          // Iterate K dimension in chunks of 16
          for (int k = 0; k < head_dim; k += 16) {
            int A[4], B[2];

            // Load Q fragment [16x16] from s_q
            load_mma_a_16x16(A, s_q, 0, k, head_dim);

            // Load K^T fragment [16x8] — transposed from K[kv, head_dim]
            load_mma_b_transposed_16x8(B, s_k, kv_n_base, k, head_dim);

            // MMA: D += A * B
            mma_m16n8k16_f32_f16(D, A, B);
          }

          // Write MMA result to s_scores
          // Fragment layout: each thread holds 4 floats at specific (row, col)
          const float *Df = reinterpret_cast<const float *>(D);
#pragma unroll
          for (int l = 0; l < 4; l++) {
            const int row = mma_frag_row(l, lane);
            const int col = mma_frag_col(l, lane) + kv_n_base;
            if (row < q_count && col < tile_len) {
              s_scores[row * MMA_FA2_BC + col] = Df[l] * scale;
            }
          }
        }
      }
      __syncthreads();

      // Phase 3: Apply causal mask (set masked positions to -INFINITY)
      if (causal) {
        const int mask_elems = MMA_FA2_BR * MMA_FA2_BC;
        for (int i = tid; i < mask_elems; i += MMA_FA2_THREADS) {
          const int qr = i / MMA_FA2_BC;
          const int kvp = i % MMA_FA2_BC;
          const int q_pos_abs = q_start + qr;
          const int kv_pos_abs = kv_start + kvp;
          const int per_row_limit = kv_len - query_len + q_pos_abs + 1;
          if (qr >= q_count || kvp >= tile_len ||
              kv_pos_abs >= per_row_limit) {
            s_scores[qr * MMA_FA2_BC + kvp] = -INFINITY;
          }
        }
      }
      __syncthreads();

      // Phase 4: Online softmax + scalar V accumulation
      // Each thread handles dimension tid (if tid < head_dim) for all Q rows.
      if (tid < head_dim) {
        const int d = tid;
        for (int qr = 0; qr < q_count; qr++) {
          for (int t = 0; t < tile_len; t++) {
            const float score = s_scores[qr * MMA_FA2_BC + t];
            if (score == -INFINITY)
              continue;

            const float new_max = fmaxf(row_max[h][qr], score);
            const float rescale = expf(row_max[h][qr] - new_max);
            const float exp_w = expf(score - new_max);

            o_acc[h][qr] = o_acc[h][qr] * rescale + exp_w * s_v[t * head_dim + d];
            row_sum[h][qr] = row_sum[h][qr] * rescale + exp_w;
            row_max[h][qr] = new_max;
          }
        }
      }
      __syncthreads();
    } // kv tile loop
  }   // GQA head loop

  // ---- Write output ----
  if (tid < head_dim) {
    const int d = tid;
    for (int h = 0; h < GQARatio; h++) {
      const int head_idx = kv_head_idx * GQARatio + h;
      if (head_idx >= num_heads)
        break;
      for (int qr = 0; qr < q_count; qr++) {
        const size_t o_offset = (size_t)batch_idx * query_len * q_stride +
                                (size_t)(q_start + qr) * q_stride +
                                head_idx * head_dim + d;
        O[o_offset] = DtypeTraits<T>::from_float(
            (row_sum[h][qr] > 0.0f) ? (o_acc[h][qr] / row_sum[h][qr]) : 0.0f);
      }
    }
  }
}

// Compute shared memory for MMA FA2 kernel.
static inline int ComputeMMAFA2Smem(int head_dim) {
  // s_q: [16 * head_dim] half
  // s_k: [64 * head_dim] half
  // s_v: [64 * head_dim] float
  // s_scores: [16 * 64] float
  const int q_bytes = MMA_FA2_BR * head_dim * sizeof(half);
  const int k_bytes = MMA_FA2_BC * head_dim * sizeof(half);
  const int v_bytes = MMA_FA2_BC * head_dim * sizeof(float);
  const int scores_bytes = MMA_FA2_BR * MMA_FA2_BC * sizeof(float);
  return q_bytes + k_bytes + v_bytes + scores_bytes;
}

} // namespace cuda_kernel
} // namespace inferflux
