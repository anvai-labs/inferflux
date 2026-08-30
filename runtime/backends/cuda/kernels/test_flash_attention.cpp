/**
 * FlashAttention Test Program for Ada RTX 4000
 *
 * This program tests if your GPU supports FlashAttention-2
 * and provides information about your CUDA capabilities.
 *
 * Build: g++ -std=c++17 -o test_flash_attention test_flash_attention.cpp
 * -lcudart Run: ./test_flash_attention
 */

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
#include <string>
#include <vector>

struct GpuInfo {
  std::string name;
  int compute_major;
  int compute_minor;
  size_t total_memory_mb;
  int sm_count;
  int max_threads_per_block;
  size_t shared_mem_per_block_kb;
  bool supports_fa2;
  bool supports_fa3;
  std::string arch_name;
};

GpuInfo GetGpuInfo(int device_id) {
  GpuInfo info{};
  cudaDeviceProp prop;

  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return info;
  }

  info.name = prop.name;
  info.compute_major = prop.major;
  info.compute_minor = prop.minor;
  info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
  info.sm_count = prop.multiProcessorCount;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.shared_mem_per_block_kb = prop.sharedMemPerBlock / 1024;

  // FlashAttention support
  info.supports_fa2 = (prop.major >= 8); // Ampere (8.0), Ada (8.6, 8.9)
  info.supports_fa3 = (prop.major >= 9); // Hopper (9.0+)

  // Architecture name
  if (prop.major == 9) {
    info.arch_name = "Hopper (H100/H200)";
  } else if (prop.major == 8) {
    if (prop.minor == 0) {
      info.arch_name = "Ampere (A100)";
    } else if (prop.minor == 6) {
      info.arch_name = "Ada (RTX 4090)";
    } else if (prop.minor == 9) {
      info.arch_name = "Ada (RTX 4000/4060/4070/4080)";
    } else {
      info.arch_name = "Ampere/Ada";
    }
  } else if (prop.major == 7) {
    if (prop.minor == 5) {
      info.arch_name = "Turing (RTX 20-series, T4)";
    } else {
      info.arch_name = "Volta (V100)";
    }
  } else {
    info.arch_name = "Unknown/Older";
  }

  return info;
}

void PrintGpuInfo(const GpuInfo &info) {
  std::cout << "=== GPU Information ===" << std::endl;
  std::cout << "GPU Name: " << info.name << std::endl;
  std::cout << "Compute Capability: " << info.compute_major << "."
            << info.compute_minor << std::endl;
  std::cout << "Architecture: " << info.arch_name << std::endl;
  std::cout << "Total Memory: " << info.total_memory_mb << " MB" << std::endl;
  std::cout << "Streaming Multiprocessors: " << info.sm_count << std::endl;
  std::cout << "Max Threads Per Block: " << info.max_threads_per_block
            << std::endl;
  std::cout << "Shared Memory Per Block: " << info.shared_mem_per_block_kb
            << " KB" << std::endl;
  std::cout << std::endl;
}

void PrintFlashAttentionSupport(const GpuInfo &info) {
  std::cout << "=== FlashAttention Support ===" << std::endl;

  std::cout << "FlashAttention-2: ";
  if (info.supports_fa2) {
    std::cout << "YES ✓" << std::endl;
    std::cout << "  Your GPU supports FlashAttention-2 optimizations!"
              << std::endl;
  } else {
    std::cout << "NO ✗" << std::endl;
    std::cout
        << "  Your GPU does not meet the requirements for FlashAttention-2."
        << std::endl;
    std::cout << "  Requires: Compute Capability 8.0+ (Ampere/Ada/Hopper)"
              << std::endl;
  }

  std::cout << std::endl;

  std::cout << "FlashAttention-3: ";
  if (info.supports_fa3) {
    std::cout << "YES ✓" << std::endl;
    std::cout << "  Your GPU supports FlashAttention-3 (Hopper optimizations)!"
              << std::endl;
  } else {
    std::cout << "NO ✗" << std::endl;
    std::cout << "  Your GPU does not support FlashAttention-3." << std::endl;
    std::cout << "  Requires: Compute Capability 9.0+ (Hopper H100+)"
              << std::endl;
  }
  std::cout << std::endl;
}

void PrintPerformanceInfo(const GpuInfo &info) {
  if (!info.supports_fa2 && !info.supports_fa3) {
    return;
  }

  std::cout << "=== Performance Information ===" << std::endl;

  std::cout << "With FlashAttention-2, you can expect:" << std::endl;
  std::cout << "  • 1.5-2.0x speedup for long sequences (2048+ tokens)"
            << std::endl;
  std::cout << "  • 1.2-1.5x speedup for medium sequences (512-1024 tokens)"
            << std::endl;
  std::cout << "  • Lower memory usage through efficient memory access"
            << std::endl;
  std::cout << "  • Better scaling with batch size" << std::endl;
  std::cout << std::endl;

  std::cout << "Optimal use cases:" << std::endl;
  std::cout << "  • Large batch sizes (8+ requests)" << std::endl;
  std::cout << "  • Long sequences (1024+ tokens)" << std::endl;
  std::cout << "  • GQA models (Llama 2/3, Mistral, etc.)" << std::endl;
  std::cout << std::endl;

  std::cout << "Less beneficial for:" << std::endl;
  std::cout << "  • Very small batches (1-2 requests)" << std::endl;
  std::cout << "  • Short sequences (< 256 tokens)" << std::endl;
  std::cout << "  • Single request with short prompts" << std::endl;
  std::cout << std::endl;
}

void PrintLlamaCppInfo() {
  std::cout << "=== llama.cpp FlashAttention Status ===" << std::endl;

  // Check if llama.cpp has FlashAttention files
  std::vector<std::string> fa_files = {
      "external/llama.cpp/ggml/src/ggml-cuda/fattn.cu",
      "external/llama.cpp/ggml/src/ggml-cuda/fattn.cuh",
      "external/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cu",
      "external/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cu"};

  bool all_exist = true;
  for (const auto &file : fa_files) {
    if (FILE *f = fopen(file.c_str(), "r")) {
      fclose(f);
      std::cout << "✓ " << file << std::endl;
    } else {
      std::cout << "✗ " << file << " (missing)" << std::endl;
      all_exist = false;
    }
  }

  if (all_exist) {
    std::cout << std::endl;
    std::cout << "✓ llama.cpp FlashAttention source files are present!"
              << std::endl;
    std::cout << "  These will be compiled if GGML_CUDA_FA=ON during build."
              << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "✗ Some FlashAttention files are missing." << std::endl;
    std::cout << "  You may need to update llama.cpp submodule." << std::endl;
  }
  std::cout << std::endl;
}

void PrintNextSteps() {
  std::cout << "=== Next Steps ===" << std::endl;
  std::cout << "1. Verify llama.cpp was built with FlashAttention:"
            << std::endl;
  std::cout << "   grep GGML_CUDA_FA build/CMakeCache.txt" << std::endl;
  std::cout << "   Should show: GGML_CUDA_FA:BOOL=ON" << std::endl;
  std::cout << std::endl;
  std::cout << "2. Build InferFlux (if not already built):" << std::endl;
  std::cout << "   ./scripts/build.sh" << std::endl;
  std::cout << std::endl;
  std::cout << "3. Start the server with debug logging:" << std::endl;
  std::cout << "   INFERFLUX_LOG_LEVEL=debug ./build/inferfluxd --config "
               "config/server.cuda.yaml"
            << std::endl;
  std::cout << std::endl;
  std::cout << "4. Make a test request:" << std::endl;
  std::cout << "   curl -X POST http://localhost:8080/v1/completions \\"
            << std::endl;
  std::cout << "     -H 'Content-Type: application/json' \\" << std::endl;
  std::cout << "     -H 'Authorization: Bearer dev-key-123' \\" << std::endl;
  std::cout << "     -d '{\"prompt\": \"Hello world!\", \"max_tokens\": 50, "
               "\"model\": \"tinyllama\"}'"
            << std::endl;
  std::cout << std::endl;
  std::cout << "5. Check logs for FlashAttention usage:" << std::endl;
  std::cout << "   tail -f logs/server.log | grep -i flash" << std::endl;
  std::cout << std::endl;
}

void PrintLearningResources() {
  std::cout << "=== Learning Resources ===" << std::endl;
  std::cout << "Documentation:" << std::endl;
  std::cout << "  • docs/FLASHATTENTION_QUICKSTART.md     - Quick start guide"
            << std::endl;
  std::cout << "  • docs/FLASHATTENTION_IMPLEMENTATION_GUIDE.md - Full "
               "implementation guide"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Source code to study:" << std::endl;
  std::cout << "  • external/llama.cpp/ggml/src/ggml-cuda/fattn.cu"
            << std::endl;
  std::cout << "  • runtime/backends/cuda/kernels/flash_attention.h"
            << std::endl;
  std::cout << "  • runtime/backends/cuda/cuda_backend.cpp" << std::endl;
  std::cout << std::endl;
  std::cout << "Research papers:" << std::endl;
  std::cout << "  • FlashAttention-2: https://arxiv.org/abs/2307.08691"
            << std::endl;
  std::cout << "  • FlashAttention-3: https://arxiv.org/abs/2407.16863"
            << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "FlashAttention Test for Ada RTX 4000" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  // Check CUDA is available
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cerr << "Error: CUDA not available - " << cudaGetErrorString(err)
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Please ensure:" << std::endl;
    std::cerr << "  1. NVIDIA drivers are installed" << std::endl;
    std::cerr << "  2. CUDA toolkit is installed" << std::endl;
    std::cerr << "  3. Your GPU is properly detected" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Check with: nvidia-smi" << std::endl;
    return 1;
  }

  if (device_count == 0) {
    std::cerr << "Error: No CUDA devices found" << std::endl;
    return 1;
  }

  std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
  std::cout << std::endl;

  // Get info for device 0 (primary GPU)
  GpuInfo info = GetGpuInfo(0);
  if (info.name.empty()) {
    std::cerr << "Error: Failed to get GPU information" << std::endl;
    return 1;
  }

  // Print information
  PrintGpuInfo(info);
  PrintFlashAttentionSupport(info);
  PrintPerformanceInfo(info);
  PrintLlamaCppInfo();
  PrintNextSteps();
  PrintLearningResources();

  std::cout << "========================================" << std::endl;
  std::cout << "Test complete!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
