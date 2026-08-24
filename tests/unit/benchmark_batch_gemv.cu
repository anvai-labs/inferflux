/**
 * Standalone benchmark for template-based batch GEMV kernels
 *
 * Compiles independently of the InferFlux test suite for easier iteration.
 *
 * Build:
 *   nvcc -O3 -arch=native --std=c++17 \
 *       -I/home/vsingh/code/inferflux/runtime/backends/cuda/native/kernels \
 *       -o benchmark_batch_gemv \
 *       benchmark_batch_gemv.cu
 *
 * Run:
 *   ./benchmark_batch_gemv
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Simulate the quant types (simplified for standalone build)
#define QK_K 256

typedef struct {
  half d;
  half dmin;
  unsigned char scales[QK_K / 32];
  unsigned char qs[QK_K / 2];
} block_q4_k;

// Kernel constants
constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

// Simplified helper functions (copied from kernel file)
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

// Baseline kernel (current cuda_native)
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

  if (out_idx >= N) return;

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

// Batch kernel prototype
template <int BatchSize>
__global__ void fused_dequant_gemv_q4k_batched(const block_q4_k *__restrict__ weight,
                                                const half *__restrict__ x,
                                                half *__restrict__ output,
                                                int N, int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  float batch_acc[BatchSize];
  #pragma unroll
  for (int b = 0; b < BatchSize; ++b) {
    batch_acc[b] = 0.0f;
  }

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  // Process each sequence in batch
  #pragma unroll
  for (int b = 0; b < BatchSize; ++b) {
    const half *x_row = x + b * K * blockDim.y + row * K;
    LoadHalfToSmem(x_row, sx, K, tid);
    __syncthreads();

    for (int blk = 0; blk < num_blocks; ++blk) {
      const block_q4_k &bq = wrow[blk];

      const float d = __half2float(*reinterpret_cast<const half *>(&bq.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&bq.dmin));

      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int sb_lo = pair * 2;
        const int sb_hi = pair * 2 + 1;

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4(sb_lo, bq.scales, &sc_lo, &m_lo);
        get_scale_min_k4(sb_hi, bq.scales, &sc_hi, &m_hi);

        const float d_sc_lo = d * static_cast<float>(sc_lo);
        const float dm_m_lo = dmin * static_cast<float>(m_lo);
        const float d_sc_hi = d * static_cast<float>(sc_hi);
        const float dm_m_hi = dmin * static_cast<float>(m_hi);

        const unsigned char qbyte = bq.qs[pair * 32 + lane];
        const float q_lo = static_cast<float>(qbyte & 0x0F);
        const float q_hi = static_cast<float>(qbyte >> 4);

        const int base = blk * QK_K + pair * 64;
        batch_acc[b] += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
        batch_acc[b] += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
      }
    }

    __syncthreads();
  }

  // Warp reduction for each sequence
  #pragma unroll
  for (int b = 0; b < BatchSize; ++b) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      batch_acc[b] += __shfl_down_sync(0xFFFFFFFF, batch_acc[b], offset);
    }
  }

  if (lane == 0 && out_idx < N) {
    #pragma unroll
    for (int b = 0; b < BatchSize; ++b) {
      output[b * N * blockDim.y + row * N + out_idx] = __float2half(batch_acc[b]);
    }
  }
}

int main() {
  printf("========================================\n");
  printf("Batch GEMV Kernel Benchmark\n");
  printf("========================================\n\n");

  // Test configuration
  const int K = 2048;
  const int N = 512;
  const int batch_sizes[] = {1, 2, 4, 8};
  const int num_iterations = 100;

  for (int M : batch_sizes) {
    printf("Batch size: %d\n", M);
    printf("  Output dim (N): %d\n", N);
    printf("  Input dim (K): %d\n", K);
    printf("  ---------------------------\n");

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

    // Warmup
    for (int i = 0; i < 10; ++i) {
      fused_dequant_gemv_q4k_baseline<<<grid, block, smem_size>>>(
          d_weight, d_x, d_output, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark baseline
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
      fused_dequant_gemv_q4k_baseline<<<grid, block, smem_size>>>(
          d_weight, d_x, d_output, N, K);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float baseline_time;
    cudaEventElapsedTime(&baseline_time, start, stop);
    baseline_time /= num_iterations;  // Average in ms

    // Warmup batch kernel
    for (int i = 0; i < 10; ++i) {
      switch (M) {
        case 1: fused_dequant_gemv_q4k_batched<1><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 2: fused_dequant_gemv_q4k_batched<2><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 4: fused_dequant_gemv_q4k_batched<4><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 8: fused_dequant_gemv_q4k_batched<8><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
      }
    }
    cudaDeviceSynchronize();

    // Benchmark batch kernel
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
      switch (M) {
        case 1: fused_dequant_gemv_q4k_batched<1><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 2: fused_dequant_gemv_q4k_batched<2><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 4: fused_dequant_gemv_q4k_batched<4><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
        case 8: fused_dequant_gemv_q4k_batched<8><<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K); break;
      }
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float batch_time;
    cudaEventElapsedTime(&batch_time, start, stop);
    batch_time /= num_iterations;

    // Print results
    float speedup = baseline_time / batch_time;
    float improvement = ((baseline_time - batch_time) / baseline_time) * 100.0f;

    printf("  Baseline kernel: %.3f ms\n", baseline_time);
    printf("  Batch kernel:    %.3f ms\n", batch_time);
    printf("  Speedup:         %.2fx\n", speedup);
    printf("  Improvement:     %+.1f%%\n", improvement);
    printf("\n");

    // Cleanup
    cudaFree(d_weight);
    cudaFree(d_x);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  printf("========================================\n");
  printf("Benchmark complete\n");
  printf("========================================\n");

  return 0;
}
