/**
 * Test harness for template-based batch GEMV kernels
 *
 * Validates correctness and measures performance of batched kernels
 * compared to the baseline single-sequence implementation.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh"

#include <algorithm>
#include <chrono>
#include <vector>
#include <random>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace test {

// =============================================================================
// Test Configuration
// =============================================================================

constexpr int K = 2048;           // Input dimension (hidden size)
constexpr int N = 4096;           // Output dimension (intermediate size)
constexpr int QK_K = 256;         // Q4_K block size

// =============================================================================
// Utility Functions
// =============================================================================

void PrintTestInfo(const std::string &test_name, int M, int N, int K) {
  std::cout << "\n============================================\n";
  std::cout << "Test: " << test_name << "\n";
  std::cout << "Batch size (M): " << M << "\n";
  std::cout << "Output dim (N): " << N << "\n";
  std::cout << "Input dim (K): " << K << "\n";
  std::cout << "============================================\n";
}

void CheckCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << ": " << cudaGetErrorString(err) << "\n";
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

#define CHECK_CUDA(x) CheckCudaError(x, __FILE__, __LINE__)

// =============================================================================
// Reference Implementation (Single-Sequence Kernel)
// =============================================================================

/**
 * Run the baseline single-sequence GEMV kernel
 * Returns output data for verification
 */
std::vector<half> RunBaselineKernel(int M, int N, int K,
                                    const std::vector<block_q4_k> &weight,
                                    const std::vector<half> &x) {
  std::vector<half> output(M * N);

  // Allocate device memory
  block_q4_k *d_weight;
  half *d_x, *d_output;
  CHECK_CUDA(cudaMalloc(&d_weight, weight.size() * sizeof(block_q4_k)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_output, output.size() * sizeof(half)));

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(block_q4_k),
                       cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half),
                       cudaMemcpyHostToDevice));

  // Configure kernel launch
  const int nwarps = kGemvWarpsPerBlock;
  const int threads_per_block = kGemvThreadsPerBlock;
  const int nblocks_x = (N + nwarps - 1) / nwarps;

  dim3 grid(nblocks_x, M);
  dim3 block(threads_per_block);

  size_t smem_size = K * sizeof(float);

  // Launch kernel
  fused_dequant_gemv_q4k<<<grid, block, smem_size>>>(d_weight, d_x, d_output, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half),
                       cudaMemcpyDeviceToHost));

  // Cleanup
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_output));

  return output;
}

// =============================================================================
// Batch Kernel Template Instantiations
// =============================================================================

/**
 * Launch template-based batch kernel for specific batch size
 */
template <int BatchSize>
std::vector<half> RunBatchKernel(int M, int N, int K,
                                  const std::vector<block_q4_k> &weight,
                                  const std::vector<half> &x) {
  std::vector<half> output(M * N);

  // Allocate device memory
  block_q4_k *d_weight;
  half *d_x, *d_output;
  CHECK_CUDA(cudaMalloc(&d_weight, weight.size() * sizeof(block_q4_k)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_output, output.size() * sizeof(half)));

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(block_q4_k),
                       cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half),
                       cudaMemcpyHostToDevice));

  // Configure kernel launch
  const int nwarps = kGemvWarpsPerBlock;
  const int threads_per_block = kGemvThreadsPerBlock;
  const int nblocks_x = (N + nwarps - 1) / nwarps;

  dim3 grid(nblocks_x, M);
  dim3 block(threads_per_block);

  size_t smem_size = K * sizeof(float);

  // Launch batch kernel
  fused_dequant_gemv_q4k_batched<BatchSize><<<grid, block, smem_size>>>(
      d_weight, d_x, d_output, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half),
                       cudaMemcpyDeviceToHost));

  // Cleanup
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_output));

  return output;
}

// =============================================================================
// Data Generation Utilities
// =============================================================================

/**
 * Generate random Q4_K quantized weights
 */
std::vector<block_q4_k> GenerateRandomQ4KWeights(int N, int K) {
  std::vector<block_q4_k> weight(N * (K / QK_K));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d_dist(0, 255);  // 8-bit delta scale
  std::uniform_int_distribution<> m_dist(0, 255);  // 8-bit delta min
  std::uniform_int_distribution<> q_dist(0, 15);   // 4-bit quantized

  for (auto &block : weight) {
    // Delta-scale and delta-min as half2 (but we use half for simplicity)
    unsigned short d_val = static_cast<unsigned short>(d_dist(gen));
    unsigned short dm_val = static_cast<unsigned short>(m_dist(gen));
    memcpy(&block.d, &d_val, sizeof(half));
    memcpy(&block.dmin, &dm_val, sizeof(half));

    // Scales (packed per 32 elements)
    for (int i = 0; i < QK_K / 32; ++i) {
      block.scales[i] = static_cast<unsigned char>(d_dist(gen));
    }

    // Quantized values (nibble-packed)
    for (int i = 0; i < QK_K / 2; ++i) {
      unsigned char q_lo = q_dist(gen);
      unsigned char q_hi = q_dist(gen);
      block.qs[i] = (q_hi << 4) | q_lo;
    }
  }

  return weight;
}

/**
 * Generate random activation values
 */
std::vector<half> GenerateRandomActivations(int M, int K) {
  std::vector<half> x(M * K);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : x) {
    val = __float2half(dist(gen));
  }

  return x;
}

// =============================================================================
// Correctness Validation
// =============================================================================

/**
 * Compare two output vectors with tolerance
 */
bool CompareOutputs(const std::vector<half> &output1,
                    const std::vector<half> &output2,
                    float tolerance = 1e-3f) {
  if (output1.size() != output2.size()) {
    std::cerr << "Size mismatch: " << output1.size() << " vs " << output2.size() << "\n";
    return false;
  }

  int mismatches = 0;
  const int max_mismatches = 10;  // Only report first 10

  for (size_t i = 0; i < output1.size(); ++i) {
    float v1 = __half2float(output1[i]);
    float v2 = __half2float(output2[i]);
    float diff = std::abs(v1 - v2);

    if (diff > tolerance) {
      if (mismatches < max_mismatches) {
        std::cerr << "Mismatch at index " << i
                  << ": " << v1 << " vs " << v2
                  << " (diff=" << diff << ")\n";
      }
      mismatches++;
    }
  }

  if (mismatches > max_mismatches) {
    std::cerr << " ... and " << (mismatches - max_mismatches) << " more\n";
  }

  if (mismatches > 0) {
    std::cerr << "Total mismatches: " << mismatches << " / " << output1.size() << "\n";
    return false;
  }

  return true;
}

// =============================================================================
// Correctness Tests
// =============================================================================

TEST(BatchGEMVKernelTest, BatchSize1_Correctness) {
  constexpr int M = 1;
  constexpr int N = 512;
  constexpr int K = 2048;

  PrintTestInfo("BatchSize=1 Correctness", M, N, K);

  // Generate test data
  auto weight = GenerateRandomQ4KWeights(N, K);
  auto x = GenerateRandomActivations(M, K);

  // Run baseline and batch kernels
  auto output_baseline = RunBaselineKernel(M, N, K, weight, x);
  auto output_batch = RunBatchKernel<1>(M, N, K, weight, x);

  // Verify correctness
  EXPECT_TRUE(CompareOutputs(output_baseline, output_batch))
      << "Batch kernel (BatchSize=1) should match baseline";
}

TEST(BatchGEMVKernelTest, BatchSize2_Correctness) {
  constexpr int M = 2;
  constexpr int N = 512;
  constexpr int K = 2048;

  PrintTestInfo("BatchSize=2 Correctness", M, N, K);

  auto weight = GenerateRandomQ4KWeights(N, K);
  auto x = GenerateRandomActivations(M, K);

  // Run baseline (multiple launches for M=2)
  auto output_baseline = RunBaselineKernel(M, N, K, weight, x);

  // Run batch kernel (single launch for M=2)
  auto output_batch = RunBatchKernel<2>(M, N, K, weight, x);

  EXPECT_TRUE(CompareOutputs(output_baseline, output_batch))
      << "Batch kernel (BatchSize=2) should match baseline";
}

TEST(BatchGEMVKernelTest, BatchSize4_Correctness) {
  constexpr int M = 4;
  constexpr int N = 512;
  constexpr int K = 2048;

  PrintTestInfo("BatchSize=4 Correctness", M, N, K);

  auto weight = GenerateRandomQ4KWeights(N, K);
  auto x = GenerateRandomActivations(M, K);

  auto output_baseline = RunBaselineKernel(M, N, K, weight, x);
  auto output_batch = RunBatchKernel<4>(M, N, K, weight, x);

  EXPECT_TRUE(CompareOutputs(output_baseline, output_batch))
      << "Batch kernel (BatchSize=4) should match baseline";
}

TEST(BatchGEMVKernelTest, BatchSize8_Correctness) {
  constexpr int M = 8;
  constexpr int N = 512;
  constexpr int K = 2048;

  PrintTestInfo("BatchSize=8 Correctness", M, N, K);

  auto weight = GenerateRandomQ4KWeights(N, K);
  auto x = GenerateRandomActivations(M, K);

  auto output_baseline = RunBaselineKernel(M, N, K, weight, x);
  auto output_batch = RunBatchKernel<8>(M, N, K, weight, x);

  EXPECT_TRUE(CompareOutputs(output_baseline, output_batch))
      << "Batch kernel (BatchSize=8) should match baseline";
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

/**
 * Measure kernel execution time
 */
template <typename KernelFunc>
float MeasureKernelTime(KernelFunc kernel_func, int iterations = 10) {
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    kernel_func();
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> diff = end - start;
  return diff.count() / iterations;  // Average time in ms
}

TEST(BatchGEMVKernelTest, Performance_BaselineVsBatch) {
  constexpr int M_values[] = {1, 2, 4, 8};
  constexpr int N = 512;
  constexpr int K = 2048;

  std::cout << "\n============================================\n";
  std::cout << "Performance Comparison: Baseline vs Batch\n";
  std::cout << "============================================\n";

  for (int M : M_values) {
    PrintTestInfo("", M, N, K);

    // Generate test data
    auto weight = GenerateRandomQ4KWeights(N, K);
    auto x = GenerateRandomActivations(M, K);

    // Allocate device memory once for both kernels
    block_q4_k *d_weight;
    half *d_x, *d_output_baseline, *d_output_batch;
    CHECK_CUDA(cudaMalloc(&d_weight, weight.size() * sizeof(block_q4_k)));
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output_baseline, M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output_batch, M * N * sizeof(half)));

    // Copy data once
    CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(block_q4_k),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half),
                         cudaMemcpyHostToDevice));

    // Configure kernel
    const int nwarps = kGemvWarpsPerBlock;
    const int threads_per_block = kGemvThreadsPerBlock;
    const int nblocks_x = (N + nwarps - 1) / nwarps;
    dim3 grid(nblocks_x, M);
    dim3 block(threads_per_block);
    size_t smem_size = K * sizeof(float);

    // Measure baseline kernel
    auto baseline_kernel = [&]() {
      fused_dequant_gemv_q4k<<<grid, block, smem_size>>>(
          d_weight, d_x, d_output_baseline, N, K);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    float baseline_time = MeasureKernelTime(baseline_kernel);

    // Measure batch kernel
    auto batch_kernel = [&]() {
      switch (M) {
        case 1:
          fused_dequant_gemv_q4k_batched<1><<<grid, block, smem_size>>>(
              d_weight, d_x, d_output_batch, N, K);
          break;
        case 2:
          fused_dequant_gemv_q4k_batched<2><<<grid, block, smem_size>>>(
              d_weight, d_x, d_output_batch, N, K);
          break;
        case 4:
          fused_dequant_gemv_q4k_batched<4><<<grid, block, smem_size>>>(
              d_weight, d_x, d_output_batch, N, K);
          break;
        case 8:
          fused_dequant_gemv_q4k_batched<8><<<grid, block, smem_size>>>(
              d_weight, d_x, d_output_batch, N, K);
          break;
        default:
          GTEST_FAIL() << "Unsupported batch size";
      }
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    float batch_time = MeasureKernelTime(batch_kernel);

    // Print results
    float speedup = baseline_time / batch_time;
    float improvement = ((baseline_time - batch_time) / baseline_time) * 100.0f;

    std::cout << "Batch size " << M << ":\n";
    std::cout << "  Baseline: " << baseline_time << " ms\n";
    std::cout << "  Batched:  " << batch_time << " ms\n";
    std::cout << "  Speedup:  " << speedup << "x\n";
    std::cout << "  Change:   " << (improvement >= 0 ? "+" : "") << improvement << "%\n";
    std::cout << "\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_output_baseline));
    CHECK_CUDA(cudaFree(d_output_batch));
  }

  SUCCEED();
}

} // namespace test
} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
