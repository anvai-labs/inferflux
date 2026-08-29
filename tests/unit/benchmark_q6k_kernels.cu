// Isolated micro-benchmark for the Q6_K down-projection kernels against the
// bandwidth floor, plus llama.cpp-style structural variants. S2 of the perf
// parity plan: mmvq_q6k measured 194µs at (M=8, N=2048, K=11008) in-server
// vs a 51µs floor (2048×11008×6.5625/8 bytes at 360 GB/s) — 3.8x off.
//
// Variants benchmarked:
//   v1   — the production kernel (4 warps per row striding super-blocks)
//   warp — llama.cpp mul_mat_vec_q structure: one warp per (row, batch),
//          full K loop serially, no cross-warp reduction
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace native = inferflux::runtime::cuda::native;
using native::block_q6_k;
using native::block_q8_1;
using native::Dp4aS8;
using native::LoadPackedInt32Unaligned;
using native::Vsubss4;

namespace {

constexpr int kIters = 200;
constexpr int kWarmup = 30;
constexpr int kN = 2048;
constexpr int kK = 11008;
constexpr int kBlocksPerRow = kK / QK_K;

unsigned short EncodeHalf(float v) {
  const half h = __float2half(v);
  unsigned short b;
  std::memcpy(&b, &h, sizeof(b));
  return b;
}

std::vector<block_q6_k> MakeQ6(int rows, int seed) {
  std::vector<block_q6_k> w(static_cast<size_t>(rows) * kBlocksPerRow);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      auto &blk = w[static_cast<size_t>(r) * kBlocksPerRow + b];
      for (int i = 0; i < QK_K / 2; ++i)
        blk.ql[i] = (seed * 11 + r * 13 + b * 7 + i * 3) & 0xFF;
      for (int i = 0; i < QK_K / 4; ++i)
        blk.qh[i] = (seed * 5 + r * 17 + b * 9 + i * 11) & 0xFF;
      for (int i = 0; i < QK_K / 16; ++i)
        blk.scales[i] = (((seed + r + b) * 5 + i * 7) % 31) - 15;
      blk.d = EncodeHalf(0.008f * (((r + b + seed) % 5) + 1));
    }
  }
  return w;
}

std::vector<half> MakeActs(int m) {
  std::vector<half> a(static_cast<size_t>(m) * kK);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = __float2half(0.009f * std::sin(0.173f * i) - 0.001f);
  return a;
}

// ---------------------------------------------------------------------------
// Variant: one warp per (row, batch) — llama mul_mat_vec_q structure. Each
// warp walks the full K of its weight row; lane-parallel over the 256-elem
// super-block, shuffle-reduce, no smem.
// ---------------------------------------------------------------------------
template <int Batch>
__global__ void q6k_warp_per_row(const block_q6_k *__restrict__ weight,
                                 const block_q8_1 *__restrict__ act_q8_1,
                                 half *__restrict__ output, int N, int K,
                                 int M) {
  static_assert(Batch >= 1 && Batch <= 8, "batch tile");
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  // Batch=1 in "full" mode: one warp per (row, m) pair across the whole M.
  const int row = warp / M;
  const int m = warp % M;
  if (row >= N)
    return;
  (void)Batch;

  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + static_cast<size_t>(row) * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;

  // Lane mapping over one 256-elem super-block: 8 sub-blocks of 32, each
  // lane covers 8 consecutive elements as two 4-elem dp4a groups (matching
  // the v1 packing).
  const int sub = lane >> 2;        // 0..7 sub-block (32 elems)
  const int quad = (lane & 3) << 3; // 0,8,16,24 within the sub-block

  float acc[Batch] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    // Q6_K: sub-block i covers elements [i*32, i*32+32). ql holds lower 4
    // bits packed 2-per-byte (128B), qh upper 2 bits 1-per-byte (64B).
    // Lane reads 8 elements = 4 ql bytes + 8 qh bits.
    const int qlo0 = (sub * 16) + (quad >> 1); // ql byte for elems 0..7 (4B)
    const int ql4 = LoadPackedInt32Unaligned(&b.ql[qlo0]);
    // qh: one byte per element; 8 elements = 8 bytes. First 4 bytes cover
    // elements 0..3 (paired with ql even nibbles), bytes 4..7 cover 4..7.
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[sub * 8 + (quad >> 1)]);
    const int qh4b = LoadPackedInt32Unaligned(&b.qh[sub * 8 + (quad >> 1) + 4]);

    const int sc = b.scales[sub];

    // Elements 0..3: even nibbles of ql, high 2 bits from qh4 bytes 0..3.
    int vl0 = ql4 & 0x0F0F0F0F;
    int vh0 = (qh4 << 4) & 0x30303030;
    int vi0 = Vsubss4(vl0 | vh0, 0x20202020);
    // Elements 4..7: odd nibbles of ql, high 2 bits from qh4b bytes 0..3.
    int vl1 = (ql4 >> 4) & 0x0F0F0F0F;
    int vh1 = (qh4b << 4) & 0x30303030;
    int vi1 = Vsubss4(vl1 | vh1, 0x20202020);

#pragma unroll
    for (int c = 0; c < Batch; ++c) {
      if (m + c >= M)
        break; // pad rows: skip
      // activation sub-block sub, elements quad..quad+7
      const block_q8_1 &a =
          act_q8_1[static_cast<size_t>(m + c) * num_q8_per_row + blk * 8 + sub];
      const int x0 = LoadPackedInt32Unaligned(&a.qs[quad]);
      const int x1 = LoadPackedInt32Unaligned(&a.qs[quad + 4]);
      const float d8 = __half2float(__low2half(a.ds));
      const float wgt = d * static_cast<float>(sc) * d8;
      acc[c] =
          fmaf(wgt, static_cast<float>(Dp4aS8(vi0, x0, 0) + Dp4aS8(vi1, x1, 0)),
               acc[c]);
    }
  }

  // Warp reduce each batch column.
#pragma unroll
  for (int c = 0; c < Batch; ++c) {
    float v = acc[c];
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xFFFFFFFF, v, off);
    if (lane == 0 && m + c < M)
      output[static_cast<size_t>(m + c) * N + row] = __float2half(v);
  }
}

// ---------------------------------------------------------------------------
// Variant: llama.cpp mmvq structure for ncols_dst=8 — 2 warps, 2 rows/block,
// vec_dot_q6_K_q8_1 per (row, token, k-block) with their exact index math.
// QI6_K=128, QR6_K=2, VDR(Q6_K,MMVQ)=1 -> each thread handles one 4-elem
// int per row per strided block iteration.
// ---------------------------------------------------------------------------
constexpr int kQI6 = 128; // quants per block per "int-pair" index space
constexpr int kQR6 = 2;

__device__ __forceinline__ int get_int_b2(const unsigned char *q, int i) {
  // llama get_int_b2: 4 bytes at 2-byte-aligned offset (unaligned safe).
  int v;
  memcpy(&v, q + 2 * i, sizeof(v));
  return v;
}

__device__ __forceinline__ int get_int_b4(const char *q, int iqs) {
  // llama get_int_b4 on q8 qs: offset iqs*4 within the 32-byte qs array.
  int v;
  memcpy(&v, q + 4 * iqs, sizeof(v));
  return v;
}

__device__ __forceinline__ float vd_q6k(const block_q6_k *__restrict__ bq,
                                        const block_q8_1 *__restrict__ y,
                                        int kbx, int iqs) {
  const int bq8_offset =
      2 * kQR6 * (iqs / (kQI6 / 2)) + (iqs % (kQI6 / 2)) / (kQI6 / 4);
  const int scale_offset =
      (kQI6 / 4) * (iqs / (kQI6 / 2)) + (iqs % (kQI6 / 2)) / (kQI6 / 8);
  const int vh_shift = 2 * ((iqs % (kQI6 / 2)) / (kQI6 / 4));

  const int vl = get_int_b2(bq->ql, iqs);
  const int vh =
      get_int_b2(bq->qh, (kQI6 / 4) * (iqs / (kQI6 / 2)) + iqs % (kQI6 / 4)) >>
      vh_shift;

  const char *scales = bq->scales + scale_offset;

  int u[kQR6];
  float d8[kQR6];
#pragma unroll
  for (int i = 0; i < kQR6; ++i) {
    u[i] = get_int_b4(reinterpret_cast<const char *>(y[bq8_offset + 2 * i].qs),
                      iqs % 32);
    d8[i] = __half2float(__low2half(y[bq8_offset + 2 * i].ds));
  }

  // Exact port of vec_dot_q6_K_q8_1_impl_mmvq (vecdotq.cuh:589).
  const float d = __half2float(*reinterpret_cast<const half *>(&bq->d));
  float sumf = 0;
#pragma unroll
  for (int i = 0; i < kQR6; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

template <int ncols_dst>
__global__ void q6k_llama_mmvq(const block_q6_k *__restrict__ w,
                               const block_q8_1 *__restrict__ act,
                               half *__restrict__ out, int N, int K, int M) {
  // ncols_dst = tokens; 2 warps, 2 rows per block (llama GENERIC table).
  constexpr int nwarps = (ncols_dst <= 4) ? 4 : 2;
  constexpr int rows_per_block = (ncols_dst == 1) ? 1 : 2;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int row0 = rows_per_block * blockIdx.x;
  const int blocks_per_row = K / QK_K;
  constexpr int vdr = 1; // VDR_Q6_K_Q8_1_MMVQ
  constexpr int blocks_per_iter = vdr * nwarps * 32 / (kQI6 / 2);

  float tmp[ncols_dst][rows_per_block] = {{0}};

  for (int kbx = tid / (kQI6 / 2 / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
    const int kby = kbx * (QK_K / 32);
    const int kqs = vdr * (tid % ((kQI6 / 2) / vdr));
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      const block_q8_1 *y = act + static_cast<size_t>(j) * (K / 32) + kby;
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
        const int row = row0 + i;
        if (row < N)
          tmp[j][i] += vd_q6k(
              &w[static_cast<size_t>(row) * blocks_per_row + kbx], y, kbx, kqs);
      }
    }
  }

  __shared__ float sh[nwarps > 1 ? nwarps - 1 : 1][ncols_dst][rows_per_block]
                     [32];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i)
        sh[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
  }
  __syncthreads();
  if (threadIdx.y == 0) {
#pragma unroll
    for (int w2 = 1; w2 < nwarps; ++w2)
#pragma unroll
      for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i)
          tmp[j][i] += sh[w2 - 1][j][i][threadIdx.x];
          // warp reduce
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
        float v = tmp[j][i];
        for (int off = 16; off > 0; off >>= 1)
          v += __shfl_down_sync(0xFFFFFFFF, v, off);
        const int row = row0 + i;
        if (threadIdx.x == 0 && row < N && j < M)
          out[static_cast<size_t>(j) * N + row] = __float2half(v);
      }
    }
  }
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost);
  return host;
}

float BenchUs(cudaEvent_t a, cudaEvent_t b, int iters) {
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.0f / iters;
}

} // namespace

int main() {
  const int cases[] = {1, 2, 4, 8};
  auto w = MakeQ6(kN, 7);

  block_q6_k *d_w;
  cudaMalloc(&d_w, w.size() * sizeof(block_q6_k));
  cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q6_k),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t s;
  cudaStreamCreate(&s);

  printf("Q6_K down-proj isolated (N=%d K=%d), floor ~%.0f us\n", kN, kK,
         kN * kK * 6.5625f / 8 / 360e9f * 1e6f);

  for (int M : cases) {
    auto acts = MakeActs(M);
    block_q8_1 *d_a;
    cudaMalloc(&d_a,
               static_cast<size_t>(M) * (kK / QK8_1) * sizeof(block_q8_1));
    // Quantize acts on device via the production helper is header-private;
    // do a host quantize instead (values only feed timing, not correctness).
    {
      std::vector<block_q8_1> hq(static_cast<size_t>(M) * (kK / QK8_1));
      for (int r = 0; r < M; ++r)
        for (int blk = 0; blk < kK / QK8_1; ++blk) {
          auto &b = hq[static_cast<size_t>(r) * (kK / QK8_1) + blk];
          float amax = 0, sum = 0;
          float vals[32];
          for (int i = 0; i < 32; ++i) {
            const int idx = blk * 32 + i;
            const float v =
                __half2float(acts[static_cast<size_t>(r) * kK + idx]);
            vals[i] = v;
            amax = std::fmax(amax, std::fabs(v));
            sum += v;
          }
          const float d8 = amax / 127.0f;
          const half2 ds =
              __halves2half2(__float2half(d8 ? d8 : 1e-9f), __float2half(sum));
          std::memcpy(&b.ds, &ds, sizeof(b.ds));
          for (int i = 0; i < 32; ++i)
            b.qs[i] = static_cast<char>(vals[i] / (d8 ? d8 : 1e-9f));
        }
      cudaMemcpy(d_a, hq.data(), hq.size() * sizeof(block_q8_1),
                 cudaMemcpyHostToDevice);
    }

    half *d_out;
    cudaMalloc(&d_out, static_cast<size_t>(M) * kN * sizeof(half));

    // v1-col1: the ncols=1 kernel with grid(N, M) — each block handles one
    // (row, token) pair with 4 warps. Tests whether the fix is dispatch-level.
    {
      constexpr int threads = 128;
      dim3 grid(kN, M);
      auto k = native::inferflux_mmvq_q6k<1>;
      for (int i = 0; i < kWarmup; ++i)
        k<<<grid, threads, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
      cudaEventRecord(start, s);
      for (int i = 0; i < kIters; ++i)
        k<<<grid, threads, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
      cudaEventRecord(stop, s);
      cudaEventSynchronize(stop);
      printf("M=%-2d  v1<1>(N,M)  : %7.1f us\n", M,
             BenchUs(start, stop, kIters));
    }

    // v1: production dispatch (per M template)
    {
      constexpr int threads = 128;
      dim3 grid(kN, (M + 7) / 8);
      void (*v1_1)(const block_q6_k *, const block_q8_1 *, half *, int, int,
                   int) = native::inferflux_mmvq_q6k<1>;
      void (*v1_2)(const block_q6_k *, const block_q8_1 *, half *, int, int,
                   int) = native::inferflux_mmvq_q6k<2>;
      void (*v1_4)(const block_q6_k *, const block_q8_1 *, half *, int, int,
                   int) = native::inferflux_mmvq_q6k<4>;
      void (*v1_8)(const block_q6_k *, const block_q8_1 *, half *, int, int,
                   int) = native::inferflux_mmvq_q6k<8>;
      void (*v1)(const block_q6_k *, const block_q8_1 *, half *, int, int,
                 int) =
          M == 1 ? v1_1 : (M == 2 ? v1_2 : (M == 4 ? v1_4 : v1_8));
      for (int i = 0; i < kWarmup; ++i)
        v1<<<grid, threads, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
      cudaEventRecord(start, s);
      for (int i = 0; i < kIters; ++i)
        v1<<<grid, threads, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
      cudaEventRecord(stop, s);
      cudaEventSynchronize(stop);
      printf("M=%-2d  v1(4warp/row): %7.1f us\n", M,
             BenchUs(start, stop, kIters));
    }

    // warp-per-row variants
    struct {
      const char *name;
      int batch;
    } warp_cases[] = {
        {"warp b1", 1}, {"warp b2", 2}, {"warp b4", 4}, {"warp b8", 8}};
    for (auto &wc : warp_cases) {
      if (wc.batch != 1)
        continue; // only the corrected full-coverage variant
      if (M > wc.batch * ((kN * wc.batch + 31) / 32))
        continue;
      const int warps_total = kN * M;
      const int blocks = (warps_total * 32 + 255) / 256;
      auto launch = [&](auto kern) {
        for (int i = 0; i < kWarmup; ++i)
          kern<<<blocks, 256, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
        cudaEventRecord(start, s);
        for (int i = 0; i < kIters; ++i)
          kern<<<blocks, 256, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        printf("M=%-2d  %-11s : %7.1f us\n", M, wc.name,
               BenchUs(start, stop, kIters));
      };
      switch (wc.batch) {
      case 1:
        launch(q6k_warp_per_row<1>);
        break;
      case 2:
        launch(q6k_warp_per_row<2>);
        break;
      case 4:
        launch(q6k_warp_per_row<4>);
        break;
      case 8:
        launch(q6k_warp_per_row<8>);
        break;
      }
    }
    {
      struct {
        const char *nm;
        int nc;
      } cs[] = {
          {"llama n1", 1}, {"llama n2", 2}, {"llama n4", 4}, {"llama n8", 8}};
      for (auto &e : cs) {
        if (e.nc < M)
          continue; // ncols must cover M (tokens)
        dim3 block(32, (e.nc <= 4) ? 4 : 2);
        const int rpb = (e.nc == 1) ? 1 : 2;
        dim3 grid((kN + rpb - 1) / rpb);
        auto go = [&]() {
          for (int i = 0; i < kWarmup; ++i) {
            if (e.nc == 1)
              q6k_llama_mmvq<1>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else if (e.nc == 2)
              q6k_llama_mmvq<2>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else if (e.nc == 4)
              q6k_llama_mmvq<4>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else
              q6k_llama_mmvq<8>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
          }
          cudaEventRecord(start, s);
          for (int i = 0; i < kIters; ++i) {
            if (e.nc == 1)
              q6k_llama_mmvq<1>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else if (e.nc == 2)
              q6k_llama_mmvq<2>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else if (e.nc == 4)
              q6k_llama_mmvq<4>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
            else
              q6k_llama_mmvq<8>
                  <<<grid, block, 0, s>>>(d_w, d_a, d_out, kN, kK, M);
          }
          cudaEventRecord(stop, s);
          cudaEventSynchronize(stop);
        };
        go();
        printf("M=%-2d  %-11s : %7.1f us\n", M, e.nm,
               BenchUs(start, stop, kIters));
      }
    }
    printf("\n");
    cudaFree(d_a);
    cudaFree(d_out);
  }
  // ===================================================================
  // MMA validation ladder
  // ===================================================================
  {
    using native::BlockQ8_1Mmq;
    printf("\n--- MMA validation (vs FPU reference) ---\n");
    for (int M : {1, 8, 16}) {
      auto acts = MakeActs(M);
      // Quantize to D4 on host.
      std::vector<BlockQ8_1Mmq> hq(static_cast<size_t>(M) * (kK / 128));
      for (int r = 0; r < M; ++r) {
        for (int grp = 0; grp < kK / 128; ++grp) {
          auto &g = hq[static_cast<size_t>(r) * (kK / 128) + grp];
          for (int sub = 0; sub < 4; ++sub) {
            float amax = 0;
            float vals[32];
            for (int i = 0; i < 32; ++i) {
              const float v = __half2float(
                  acts[static_cast<size_t>(r) * kK + grp * 128 + sub * 32 + i]);
              vals[i] = v;
              amax = std::fmax(amax, std::fabs(v));
            }
            const float d_inv = amax > 0 ? 127.0f / amax : 0.0f;
            g.d4[sub] = amax > 0 ? 1.0f / d_inv : 0.0f;
            for (int i = 0; i < 32; ++i)
              g.qs[sub * 32 + i] =
                  static_cast<int8_t>(std::lround(vals[i] * d_inv));
          }
        }
      }
      BlockQ8_1Mmq *d_a;
      cudaMalloc(&d_a, hq.size() * sizeof(BlockQ8_1Mmq));
      cudaMemcpy(d_a, hq.data(), hq.size() * sizeof(BlockQ8_1Mmq),
                 cudaMemcpyHostToDevice);
      half *d_out_mma;
      cudaMalloc(&d_out_mma, static_cast<size_t>(M) * kN * sizeof(half));

      // Timing
      {
        dim3 grid((kN + native::kMmqY - 1) / native::kMmqY, (M + 15) / 16);
        const size_t smem = native::MmqSmemInts(16) * sizeof(int);
        cudaFuncSetAttribute(native::InferfluxMmqQ6KMma<16>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        for (int i = 0; i < kWarmup; ++i)
          native::InferfluxMmqQ6KMma<16>
              <<<grid, 256, smem, s>>>(
                  reinterpret_cast<const char *>(d_w), d_a, d_out_mma, kN, kK, M);
        cudaEventRecord(start, s);
        for (int i = 0; i < kIters; ++i)
          native::InferfluxMmqQ6KMma<16>
              <<<grid, 256, smem, s>>>(
                  reinterpret_cast<const char *>(d_w), d_a, d_out_mma, kN, kK, M);
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        printf("M=%-2d  mma q6k     : %7.1f us\n", M,
               BenchUs(start, stop, kIters));
      }

      // Correctness vs FPU reference
      {
        auto out = CopyDeviceHalfs(d_out_mma, static_cast<size_t>(M) * kN);
        // FPU ref: for each row n, col m: sum_k deq(w[n][k]) * act[m][k]
        const auto &wv = w;
        double max_rel = 0;
        int n_checked = 0;
        for (int n = 0; n < kN; n += 97) { // sample rows
          const auto &b0 = wv[static_cast<size_t>(n) * kBlocksPerRow];
          (void)b0;
          for (int m = 0; m < M; ++m) {
            double ref = 0;
            for (int blk = 0; blk < kBlocksPerRow; ++blk) {
              const auto &b = wv[static_cast<size_t>(n) * kBlocksPerRow + blk];
              const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
              for (int e = 0; e < 256; ++e) {
                const int lo = b.ql[e / 2] & 0xF;
                const int hi = b.ql[e / 2] >> 4;
                const int l = (e % 2 == 0) ? lo : hi;
                const int h = (b.qh[e] & 3);
                const int q = ((l | (h << 4)) - 32);
                const float sc =
                    static_cast<float>(b.scales[e / 16]);
                const int k = blk * 256 + e;
                const auto &g =
                    hq[static_cast<size_t>(m) * (kK / 128) + k / 128];
                const int8_t aq = g.qs[k % 128];
                const float ad = g.d4[(k % 128) / 32];
                ref += static_cast<double>(d) * sc * q * ad * aq;
              }
            }
            const float got = __half2float(
                out[static_cast<size_t>(m) * kN + n]);
            const double rel = std::fabs(ref) > 1e-9 ? std::fabs((got - ref) / ref) : std::fabs(got - ref);
            max_rel = std::fmax(max_rel, rel);
            ++n_checked;
            if (n_checked >= 64)
              break;
          }
          if (n_checked >= 64)
            break;
        }
        printf("M=%-2d  max rel err (64 samples): %.3e\n", M, max_rel);
      }
      cudaFree(d_a);
      cudaFree(d_out_mma);
    }
  }
  return 0;
}
