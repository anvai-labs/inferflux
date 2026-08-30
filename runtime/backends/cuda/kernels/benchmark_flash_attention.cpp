#include "runtime/backends/cuda/kernels/flash_attention.h"
#include "server/logging/logger.h"

#include <chrono>
// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif
#include <iostream>
#include <vector>

namespace inferflux {
namespace cuda {
namespace kernels {
namespace benchmark {

// Benchmark configuration
struct BenchmarkConfig {
  int head_dim = 128;
  int num_heads = 32;
  int num_kv_heads = 32;
  int max_seq_len = 2048;
  int batch_size = 1;
  bool use_fp16 = true;
  int num_iterations = 100;
  bool use_causal_mask = true;

  void Print() const {
    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Head Dim: " << head_dim << "\n";
    std::cout << "  Num Heads: " << num_heads << "\n";
    std::cout << "  Num KV Heads: " << num_kv_heads << "\n";
    std::cout << "  Max Seq Len: " << max_seq_len << "\n";
    std::cout << "  Batch Size: " << batch_size << "\n";
    std::cout << "  Use FP16: " << (use_fp16 ? "Yes" : "No") << "\n";
    std::cout << "  Num Iterations: " << num_iterations << "\n";
    std::cout << "  Causal Mask: " << (use_causal_mask ? "Yes" : "No") << "\n";
  }
};

// Benchmark results
struct BenchmarkResults {
  float avg_time_ms;
  float min_time_ms;
  float max_time_ms;
  float tokens_per_second;
  float memory_mb;

  void Print() const {
    std::cout << "\nBenchmark Results:\n";
    std::cout << "  Average Time: " << avg_time_ms << " ms\n";
    std::cout << "  Min Time: " << min_time_ms << " ms\n";
    std::cout << "  Max Time: " << max_time_ms << " ms\n";
    std::cout << "  Throughput: " << tokens_per_second << " tokens/sec\n";
    std::cout << "  GPU Memory: " << memory_mb << " MB\n";
  }
};

// Allocate GPU memory with error checking
template <typename T> T *AllocateGPU(size_t count) {
  T *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
  if (err != cudaSuccess) {
    log::Error("benchmark", "Failed to allocate GPU memory: " +
                                std::string(cudaGetErrorString(err)));
    return nullptr;
  }
  return ptr;
}

// Free GPU memory
template <typename T> void FreeGPU(T *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

// Run benchmark for a specific kernel type
BenchmarkResults RunBenchmark(const BenchmarkConfig &config,
                              AttentionKernelType kernel_type) {

  BenchmarkResults results{};
  results.min_time_ms = std::numeric_limits<float>::max();
  results.max_time_ms = 0.0f;

  // Setup FlashAttention config
  FlashAttentionConfig fa_config;
  fa_config.head_dim = config.head_dim;
  fa_config.num_heads = config.num_heads;
  fa_config.num_kv_heads = config.num_kv_heads;
  fa_config.max_seq_len = config.max_seq_len;
  fa_config.use_causal_mask = config.use_causal_mask;
  fa_config.softmax_scale = 1.0f / std::sqrt(float(config.head_dim));

  // Detect GQA/MQA
  fa_config.is_gqa =
      (config.num_kv_heads > 0 && config.num_kv_heads < config.num_heads);
  fa_config.is_mqa = (config.num_kv_heads == 1);
  fa_config.kv_head_ratio =
      fa_config.is_gqa ? config.num_heads / config.num_kv_heads : 1;

  std::cout << "\nRunning benchmark with kernel: "
            << GetKernelTypeName(kernel_type) << "\n";

  // Allocate GPU memory
  size_t seq_len = config.max_seq_len;
  size_t head_dim = config.head_dim;
  size_t num_heads = config.num_heads;
  size_t num_kv_heads = config.num_kv_heads;
  size_t batch_size = config.batch_size;

  // Calculate sizes
  size_t q_size = batch_size * seq_len * num_heads * head_dim;
  size_t k_size = batch_size * seq_len * num_kv_heads * head_dim;
  size_t v_size = batch_size * seq_len * num_kv_heads * head_dim;
  size_t out_size = batch_size * seq_len * num_heads * head_dim;
  size_t lse_size = batch_size * num_heads * seq_len;

  // Memory tracking
  size_t total_memory = (q_size + k_size + v_size + out_size + lse_size);
  if (config.use_fp16) {
    total_memory = total_memory * sizeof(half) + lse_size * sizeof(float);
  } else {
    total_memory = total_memory * sizeof(float);
  }
  results.memory_mb = total_memory / (1024.0f * 1024.0f);

  std::cout << "  Allocating " << results.memory_mb << " MB GPU memory\n";

  // Allocate device memory
  void *d_q = nullptr;
  void *d_k = nullptr;
  void *d_v = nullptr;
  void *d_output = nullptr;
  float *d_lse = nullptr;

  if (config.use_fp16) {
    d_q = AllocateGPU<half>(q_size);
    d_k = AllocateGPU<half>(k_size);
    d_v = AllocateGPU<half>(v_size);
    d_output = AllocateGPU<half>(out_size);
  } else {
    d_q = AllocateGPU<float>(q_size);
    d_k = AllocateGPU<float>(k_size);
    d_v = AllocateGPU<float>(v_size);
    d_output = AllocateGPU<float>(out_size);
  }
  d_lse = AllocateGPU<float>(lse_size);

  if (!d_q || !d_k || !d_v || !d_output || !d_lse) {
    log::Error("benchmark", "Failed to allocate GPU memory");
    goto cleanup;
  }

  // Initialize with dummy data (zeros for now)
  // TODO: Initialize with actual test data for correctness validation
  cudaMemset(d_q, 0,
             config.use_fp16 ? q_size * sizeof(half) : q_size * sizeof(float));
  cudaMemset(d_k, 0,
             config.use_fp16 ? k_size * sizeof(half) : k_size * sizeof(float));
  cudaMemset(d_v, 0,
             config.use_fp16 ? v_size * sizeof(half) : v_size * sizeof(float));

  // Warmup run
  FlashAttentionForward(fa_config, d_q, d_k, d_v, d_output, d_lse, batch_size,
                        seq_len, config.use_fp16, kernel_type, 0);
  cudaDeviceSynchronize();

  // Benchmark loop
  std::vector<float> timings;
  timings.reserve(config.num_iterations);

  std::cout << "  Running " << config.num_iterations << " iterations...\n";

  for (int i = 0; i < config.num_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    bool success = FlashAttentionForward(fa_config, d_q, d_k, d_v, d_output,
                                         d_lse, batch_size, seq_len,
                                         config.use_fp16, kernel_type, 0);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
      log::Error("benchmark", "FlashAttention forward failed");
      goto cleanup;
    }

    float duration_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    timings.push_back(duration_ms);

    results.min_time_ms = std::min(results.min_time_ms, duration_ms);
    results.max_time_ms = std::max(results.max_time_ms, duration_ms);
  }

  // Calculate statistics
  float total_time = 0.0f;
  for (float t : timings) {
    total_time += t;
  }
  results.avg_time_ms = total_time / timings.size();

  // Calculate throughput (tokens per second)
  float total_tokens = batch_size * seq_len;
  results.tokens_per_second = (total_tokens / (results.avg_time_ms / 1000.0f));

cleanup:
  FreeGPU(static_cast<half *>(d_q));
  FreeGPU(static_cast<half *>(d_k));
  FreeGPU(static_cast<half *>(d_v));
  FreeGPU(static_cast<half *>(d_output));
  FreeGPU(d_lse);

  return results;
}

// Compare multiple kernel types
void CompareKernels(const BenchmarkConfig &config) {
  std::cout << "\n========================================\n";
  std::cout << "FlashAttention Benchmark on Ada RTX 4000\n";
  std::cout << "========================================\n";

  config.Print();

  // Query GPU capabilities
  FlashAttentionCapabilities caps = QueryFlashAttentionCapabilities();
  std::cout << "\nGPU Capabilities:\n";
  std::cout << "  FlashAttention-2: " << (caps.supports_flash2 ? "YES" : "NO")
            << "\n";
  std::cout << "  FlashAttention-3: " << (caps.supports_flash3 ? "YES" : "NO")
            << "\n";
  std::cout << "  Recommended: " << GetKernelTypeName(caps.recommended_kernel)
            << "\n";

  // Benchmark each available kernel
  std::vector<AttentionKernelType> kernels_to_test;

  if (caps.supports_flash2) {
    kernels_to_test.push_back(AttentionKernelType::kFlash2);
  }

  kernels_to_test.push_back(AttentionKernelType::kStandard);

  // Run benchmarks
  std::cout << "\n========================================\n";

  for (auto kernel_type : kernels_to_test) {
    BenchmarkResults results = RunBenchmark(config, kernel_type);
    results.Print();
    std::cout << "----------------------------------------\n";
  }
}

// Test different sequence lengths
void BenchmarkSequenceLengths() {
  std::cout << "\n========================================\n";
  std::cout << "Sequence Length Scaling Benchmark\n";
  std::cout << "========================================\n";

  std::vector<int> seq_lengths = {512, 1024, 2048, 4096};

  for (int seq_len : seq_lengths) {
    BenchmarkConfig config;
    config.max_seq_len = seq_len;
    config.num_iterations = 50;

    std::cout << "\n--- Sequence Length: " << seq_len << " ---\n";

    BenchmarkResults results = RunBenchmark(config, AttentionKernelType::kAuto);
    results.Print();
  }
}

// Test different batch sizes
void BenchmarkBatchSizes() {
  std::cout << "\n========================================\n";
  std::cout << "Batch Size Scaling Benchmark\n";
  std::cout << "========================================\n";

  std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

  for (int batch_size : batch_sizes) {
    BenchmarkConfig config;
    config.batch_size = batch_size;
    config.num_iterations = 50;

    std::cout << "\n--- Batch Size: " << batch_size << " ---\n";

    BenchmarkResults results = RunBenchmark(config, AttentionKernelType::kAuto);
    results.Print();
  }
}

// Test GQA (Grouped-Query Attention)
void BenchmarkGQA() {
  std::cout << "\n========================================\n";
  std::cout << "GQA (Grouped-Query Attention) Benchmark\n";
  std::cout << "========================================\n";

  std::vector<std::pair<int, int>> gqa_configs = {
      {32, 32}, // MHA (Multi-Head Attention)
      {32, 8},  // GQA (4:1 ratio)
      {32, 1},  // MQA (Multi-Query Attention)
  };

  for (auto [num_heads, num_kv_heads] : gqa_configs) {
    BenchmarkConfig config;
    config.num_heads = num_heads;
    config.num_kv_heads = num_kv_heads;
    config.num_iterations = 50;

    std::cout << "\n--- Config: " << num_heads << " heads, " << num_kv_heads
              << " KV heads ---\n";

    BenchmarkResults results = RunBenchmark(config, AttentionKernelType::kAuto);
    results.Print();
  }
}

} // namespace benchmark
} // namespace kernels
} // namespace cuda
} // namespace inferflux

// Main entry point
int main(int argc, char **argv) {
  using namespace inferflux::cuda::kernels::benchmark;

  std::cout << "FlashAttention Benchmark for Ada RTX 4000\n";
  std::cout << "==========================================\n\n";

  // Run comparison benchmark
  BenchmarkConfig default_config;
  default_config.max_seq_len = 2048;
  default_config.num_heads = 32;
  default_config.num_kv_heads = 32;
  default_config.use_fp16 = true;
  default_config.num_iterations = 100;

  CompareKernels(default_config);

  // Uncomment to run additional benchmarks:
  // BenchmarkSequenceLengths();
  // BenchmarkBatchSizes();
  // BenchmarkGQA();

  return 0;
}
