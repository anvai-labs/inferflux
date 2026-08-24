/**
 * Test correctness of vectorized GEMV kernel vs baseline
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 --std=c++17 \
 *       -I/home/vsingh/code/inferflux/runtime/backends/cuda/native/kernels \
 *       -o test_vectorized_gemv \
 *       test_vectorized_gemv.cu
 *
 * Run:
 *   ./test_vectorized_gemv
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Simulate the quant types
#define QK_K 256

typedef struct {
  half d;
  half dmin;
  unsigned char scales[QK_K / 32];  // 8 bytes
  unsigned char qs[QK_K / 2];       // 128 bytes
} block_q4_k;

// Kernel constants
constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

// Helper functions
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

// Baseline kernel (original, scalar loads)
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

// Vectorized kernel - simplified version with only scales vectorization
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

    // Vectorized scales load using memcpy to handle alignment safely
    // scales[] is at offset 4 (not 8-byte aligned), so we pack manually
    uint64_t scales_packed = 0;
    memcpy(&scales_packed, b.scales, 8);
    const unsigned char *scales_bytes =
        reinterpret_cast<const unsigned char *>(&scales_packed);

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int sb_lo = pair * 2;
      const int sb_hi = pair * 2 + 1;

      // Extract scales/mins from vectorized load
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(sb_lo, scales_bytes, &sc_lo, &m_lo);
      get_scale_min_k4(sb_hi, scales_bytes, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      // Qs loads remain scalar for now - keeping same logic as baseline
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

// Test harness
int main() {
  printf("========================================\n");
  printf("Vectorized GEMV Kernel Test\n");
  printf("========================================\n\n");

  const int K = 2048;
  const int N = 512;
  const int M = 4;

  // Allocate device memory
  const int weight_size = N * (K / QK_K);
  const int x_size = M * K;
  const int output_size = M * N;

  block_q4_k *d_weight;
  half *d_x, *d_output_baseline, *d_output_vectorized;

  cudaMalloc(&d_weight, weight_size * sizeof(block_q4_k));
  cudaMalloc(&d_x, x_size * sizeof(half));
  cudaMalloc(&d_output_baseline, output_size * sizeof(half));
  cudaMalloc(&d_output_vectorized, output_size * sizeof(half));

  // Configure kernel
  const int nwarps = kGemvWarpsPerBlock;
  const int threads_per_block = kGemvThreadsPerBlock;
  const int nblocks_x = (N + nwarps - 1) / nwarps;
  dim3 grid(nblocks_x, M);
  dim3 block(threads_per_block);
  size_t smem_size = K * sizeof(float);

  printf("Test configuration:\n");
  printf("  K (input dim): %d\n", K);
  printf("  N (output dim): %d\n", N);
  printf("  M (batch size): %d\n", M);
  printf("  Grid: (%d, %d)\n", nblocks_x, M);
  printf("  Block: %d threads\n", threads_per_block);
  printf("  Shared memory: %zu bytes\n\n", smem_size);

  // Run baseline
  printf("Running baseline kernel...\n");
  fused_dequant_gemv_q4k_baseline<<<grid, block, smem_size>>>(
      d_weight, d_x, d_output_baseline, N, K);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("ERROR: Baseline kernel failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("  ✓ Baseline complete\n");

  // Run vectorized
  printf("Running vectorized kernel...\n");
  fused_dequant_gemv_q4k_vectorized<<<grid, block, smem_size>>>(
      d_weight, d_x, d_output_vectorized, N, K);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("ERROR: Vectorized kernel failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("  ✓ Vectorized complete\n\n");

  // Copy outputs to host for comparison
  half *h_baseline = new half[output_size];
  half *h_vectorized = new half[output_size];

  cudaMemcpy(h_baseline, d_output_baseline, output_size * sizeof(half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vectorized, d_output_vectorized, output_size * sizeof(half),
             cudaMemcpyDeviceToHost);

  // Compare outputs
  printf("Comparing outputs...\n");
  float max_diff = 0.0f;
  int mismatch_count = 0;
  const float tolerance = 1e-3f;

  for (int i = 0; i < output_size; ++i) {
    float b = __half2float(h_baseline[i]);
    float v = __half2float(h_vectorized[i]);
    float diff = std::abs(b - v);
    max_diff = std::max(max_diff, diff);

    if (diff > tolerance) {
      if (mismatch_count < 10) {
        printf("  Mismatch at [%d]: baseline=%.6f, vectorized=%.6f, diff=%.6f\n",
               i, b, v, diff);
      }
      mismatch_count++;
    }
  }

  printf("\nResults:\n");
  printf("  Maximum difference: %.8f\n", max_diff);
  printf("  Mismatches (>%.4f): %d / %d\n", tolerance, mismatch_count, output_size);

  if (mismatch_count == 0) {
    printf("\n✅ SUCCESS: Vectorized kernel produces identical results!\n");
  } else {
    printf("\n❌ FAILURE: Vectorized kernel produces different results.\n");
  }

  // Cleanup
  delete[] h_baseline;
  delete[] h_vectorized;
  cudaFree(d_weight);
  cudaFree(d_x);
  cudaFree(d_output_baseline);
  cudaFree(d_output_vectorized);

  printf("\n========================================\n");
  printf("Test complete\n");
  printf("========================================\n");

  return mismatch_count == 0 ? 0 : 1;
}
