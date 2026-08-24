/**
 * Performance benchmark: baseline vs vectorized GEMV
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 --std=c++17 \
 *       -I/home/vsingh/code/inferflux/runtime/backends/cuda/native/kernels \
 *       -o benchmark_vectorized_perf \
 *       benchmark_vectorized_perf.cu
 *
 * Run:
 *   ./benchmark_vectorized_perf
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define QK_K 256

typedef struct {
  half d;
  half dmin;
  unsigned char scales[QK_K / 32];
  unsigned char qs[QK_K / 2];
} block_q4_k;

constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

__device__ __forceinline__ void get_scale_min_k4(int i, const unsigned char *scales,
                                                 unsigned char *d, unsigned char *m) {
  const unsigned char q = scales[i >> 1];
  *d = (q & 0xF) | ((q & 0xF0) << 4);
  *m = (q & 0xF0) | ((q & 0x0F) << 4);
}

__device__ __forceinline__ void LoadHalfToSmem(const half *__restrict__ src,
                                               float *__restrict__ dst, int K,
                                               int tid) {
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 h2 = reinterpret_cast<const __half2 *>(src)[i];
    float2 f2 = __half22float2(h2);
    dst[2 * i] = f2.x;
    dst[2 * i + 1] = f2.y;
  }
}

// Baseline kernel (scalar loads)
__global__ void fused_dequant_gemv_q4k_baseline(const block_q4_k *__restrict__ weight,
                                                const half *__restrict__ x,
                                                half *__restrict__ output,
                                                int N, int K) {
  extern __shared__ float sx[];
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int sb_lo = pair * 2;
      const int sb_hi = pair * 2 + 1;
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(sb_lo, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(sb_hi, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const unsigned char qbyte = b.qs[pair * 32 + lane];
      const float q_lo = static_cast<float>(qbyte & 0x0F);
      const float q_hi = static_cast<float>(qbyte >> 4);

      const int base = blk * QK_K + pair * 64;
      acc += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
      acc += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

// Vectorized kernel (scales loaded with memcpy)
__global__ void fused_dequant_gemv_q4k_vectorized(const block_q4_k *__restrict__ weight,
                                                   const half *__restrict__ x,
                                                   half *__restrict__ output,
                                                   int N, int K) {
  extern __shared__ float sx[];
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    // Vectorized scales load using memcpy (handles alignment)
    uint64_t scales_packed = 0;
    memcpy(&scales_packed, b.scales, 8);
    const unsigned char *scales_bytes =
        reinterpret_cast<const unsigned char *>(&scales_packed);

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int sb_lo = pair * 2;
      const int sb_hi = pair * 2 + 1;
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(sb_lo, scales_bytes, &sc_lo, &m_lo);
      get_scale_min_k4(sb_hi, scales_bytes, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const unsigned char qbyte = b.qs[pair * 32 + lane];
      const float q_lo = static_cast<float>(qbyte & 0x0F);
      const float q_hi = static_cast<float>(qbyte >> 4);

      const int base = blk * QK_K + pair * 64;
      acc += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
      acc += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

int main() {
  printf("========================================\n");
  printf("Vectorized GEMV Performance Benchmark\n");
  printf("========================================\n\n");

  const int K = 2048;
  const int N = 512;
  const int M = 1;
  const int num_iterations = 1000;

  printf("Configuration:\n");
  printf("  K (input dim): %d\n", K);
  printf("  N (output dim): %d\n", N);
  printf("  M (batch): %d\n", M);
  printf("  Iterations: %d\n\n", num_iterations);

  // Allocate device memory
  const int weight_size = N * (K / QK_K);
  const int x_size = M * K;
  const int output_size = M * N;

  block_q4_k *d_weight;
  half *d_x, *d_output;

  cudaMalloc(&d_weight, weight_size * sizeof(block_q4_k));
  cudaMalloc(&d_x, x_size * sizeof(half));
  cudaMalloc(&d_output, output_size * sizeof(half));

  // Configure kernel
  const int nwarps = kGemvWarpsPerBlock;
  const int threads_per_block = kGemvThreadsPerBlock;
  const int nblocks_x = (N + nwarps - 1) / nwarps;
  dim3 grid(nblocks_x, M);
  dim3 block(threads_per_block);
  size_t smem_size = K * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  for (int i = 0; i < 10; ++i) {
    fused_dequant_gemv_q4k_baseline<<<grid, block, smem_size>>>(
        d_weight, d_x, d_output, N, K);
    fused_dequant_gemv_q4k_vectorized<<<grid, block, smem_size>>>(
        d_weight, d_x, d_output, N, K);
  }
  cudaDeviceSynchronize();

  // Benchmark baseline
  printf("Benchmarking baseline kernel...\n");
  cudaEventRecord(start);
  for (int i = 0; i < num_iterations; ++i) {
    fused_dequant_gemv_q4k_baseline<<<grid, block, smem_size>>>(
        d_weight, d_x, d_output, N, K);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float baseline_time;
  cudaEventElapsedTime(&baseline_time, start, stop);
  float baseline_avg = baseline_time / num_iterations;

  // Benchmark vectorized
  printf("Benchmarking vectorized kernel...\n");
  cudaEventRecord(start);
  for (int i = 0; i < num_iterations; ++i) {
    fused_dequant_gemv_q4k_vectorized<<<grid, block, smem_size>>>(
        d_weight, d_x, d_output, N, K);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float vectorized_time;
  cudaEventElapsedTime(&vectorized_time, start, stop);
  float vectorized_avg = vectorized_time / num_iterations;

  // Calculate speedup
  float speedup = baseline_avg / vectorized_avg;
  float improvement = ((baseline_avg - vectorized_avg) / baseline_avg) * 100.0f;

  printf("\n========================================\n");
  printf("Results (averaged over %d iterations):\n", num_iterations);
  printf("========================================\n");
  printf("  Baseline:    %.6f ms\n", baseline_avg);
  printf("  Vectorized:  %.6f ms\n", vectorized_avg);
  printf("  Speedup:     %.3fx\n", speedup);
  printf("  Improvement: %+.2f%%\n", improvement);
  printf("========================================\n\n");

  if (improvement > 0) {
    printf("✅ Vectorized kernel is %.1f%% FASTER\n", improvement);
  } else if (improvement < -1.0f) {
    printf("❌ Vectorized kernel is %.1f%% SLOWER (unexpected!)\n", -improvement);
    printf("   memcpy overhead may dominate benefit\n");
  } else {
    printf("➖ Vectorized kernel is ~same speed (within 1%%)\n");
    printf("   Benefit may be visible in full model context\n");
  }

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_weight);
  cudaFree(d_x);
  cudaFree(d_output);

  return 0;
}
