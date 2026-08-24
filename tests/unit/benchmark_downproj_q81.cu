#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using inferflux::FusedDispatchGeometry;
using inferflux::FusedQuantGemm;
using inferflux::NativeExecutionPolicy;
using inferflux::QuantizedWeightInfo;
namespace native = inferflux::runtime::cuda::native;
namespace GGUF = native::GGUF;

namespace {

constexpr int kWarmupIters = 30;
constexpr int kBenchmarkIters = 200;
constexpr int kN = 2048;
constexpr int kK = 11008;
constexpr int kBlocksPerRow = kK / QK_K;

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

std::vector<native::block_q4_k> MakeQ4Rows(int rows, int blocks_per_row, int seed) {
  std::vector<native::block_q4_k> blocks(static_cast<size_t>(rows) *
                                         blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
      block.d =
          EncodeHalfBits(0.012f * static_cast<float>(((row + blk + seed) % 5) + 1));
      block.dmin = EncodeHalfBits(
          0.006f * static_cast<float>(((row + blk + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] = static_cast<unsigned char>(
            (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] = static_cast<unsigned char>(
            (seed * 13 + row * 19 + blk * 23 + i * 3) & 0xFF);
      }
    }
  }
  return blocks;
}

std::vector<native::block_q6_k> MakeQ6Rows(int rows, int blocks_per_row, int seed) {
  std::vector<native::block_q6_k> blocks(static_cast<size_t>(rows) *
                                         blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] = static_cast<unsigned char>(
            (seed * 11 + row * 13 + blk * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] = static_cast<unsigned char>(
            (seed * 5 + row * 17 + blk * 9 + i * 11) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>((((seed + row + blk) * 5 + i * 7) % 31) - 15);
      }
      block.d =
          EncodeHalfBits(0.008f * static_cast<float>(((row + blk + seed) % 5) + 1));
    }
  }
  return blocks;
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost);
  return host;
}

template <typename Block>
float RunBenchmark(const std::vector<Block> &weights, int quant_type,
                   const char *label, int m,
                   bool enable_row_hot_fixed,
                   bool enable_rowpair_hot_fixed) {
  std::printf("\nCase: %s M=%d N=%d K=%d\n", label, m, kN, kK);

  Block *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out_generic = nullptr;
  half *d_out_hot = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto input =
      MakeWaveTensor(static_cast<size_t>(m) * kK, 0.009f, -0.001f);
  cudaMalloc(reinterpret_cast<void **>(&d_w), weights.size() * sizeof(Block));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1,
             static_cast<size_t>(m) * (kK / QK8_1) * sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out_generic),
             static_cast<size_t>(m) * kN * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&d_out_hot),
             static_cast<size_t>(m) * kN * sizeof(half));

  cudaMemcpy(d_w, weights.data(), weights.size() * sizeof(Block),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(kN) * kK};
  NativeExecutionPolicy generic_policy;
  NativeExecutionPolicy hot_policy;
  hot_policy.enable_experimental_q81_downproj_hot_fixed = enable_row_hot_fixed;
  hot_policy.enable_experimental_q81_downproj_rowpair_hot_fixed =
      enable_rowpair_hot_fixed;

  const auto generic_op = FusedQuantGemm::SelectDownProjOperator(
      quant_type, FusedDispatchGeometry{m, kN, kK, 1, true, false}, true, true,
      true, &generic_policy);
  const auto hot_op = FusedQuantGemm::SelectDownProjOperator(
      quant_type, FusedDispatchGeometry{m, kN, kK, 1, true, false}, true, true,
      true, &hot_policy);
  std::printf("  generic operator: %s\n",
              FusedQuantGemm::DownProjOperatorName(generic_op));
  std::printf("  hot operator:     %s\n",
              FusedQuantGemm::DownProjOperatorName(hot_op));

  auto bench_ms = [&](half *out, const NativeExecutionPolicy *policy) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, out, m, kN, kK, stream,
                               policy);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, out, m, kN, kK, stream,
                               policy);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float generic_ms = bench_ms(d_out_generic, &generic_policy);
  const float hot_ms = bench_ms(d_out_hot, &hot_policy);

  const auto generic = CopyDeviceHalfs(d_out_generic, static_cast<size_t>(m) * kN);
  const auto hot = CopyDeviceHalfs(d_out_hot, static_cast<size_t>(m) * kN);
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < generic.size(); ++i) {
    max_abs_diff = std::max(
        max_abs_diff, std::fabs(__half2float(generic[i]) - __half2float(hot[i])));
  }

  std::printf("  generic: %.3f ms\n", generic_ms);
  std::printf("  hot:     %.3f ms\n", hot_ms);
  std::printf("  speedup: %.3fx\n", generic_ms / hot_ms);
  std::printf("  max abs diff: %.6f\n", max_abs_diff);

  cudaFree(d_out_hot);
  cudaFree(d_out_generic);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w);
  cudaStreamDestroy(stream);
  return generic_ms / hot_ms;
}

} // namespace

int main() {
  std::puts("========================================");
  std::puts("Down-Proj Q8_1 Hot-Path Benchmark");
  std::puts("========================================");

#ifndef INFERFLUX_HAS_CUDA
  std::puts("benchmark_downproj_q81 requires InferFlux CUDA kernels");
  return 0;
#else
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::puts("No CUDA device available");
    return 0;
  }

  const float q4_m1_speedup = RunBenchmark(
      MakeQ4Rows(kN, kBlocksPerRow, 3),
      static_cast<int>(GGUF::TensorType::Q4_K), "Q4_K single-row", 1, true,
      false);
  const float q6_m1_speedup = RunBenchmark(
      MakeQ6Rows(kN, kBlocksPerRow, 7),
      static_cast<int>(GGUF::TensorType::Q6_K), "Q6_K single-row", 1, true,
      false);
  const float q4_m2_speedup = RunBenchmark(
      MakeQ4Rows(kN, kBlocksPerRow, 11),
      static_cast<int>(GGUF::TensorType::Q4_K), "Q4_K row-pair", 2, false,
      true);
  const float q6_m2_speedup = RunBenchmark(
      MakeQ6Rows(kN, kBlocksPerRow, 13),
      static_cast<int>(GGUF::TensorType::Q6_K), "Q6_K row-pair", 2, false,
      true);

  std::puts("\nSummary:");
  std::printf("  Q4_K M=1 speedup: %.3fx\n", q4_m1_speedup);
  std::printf("  Q6_K M=1 speedup: %.3fx\n", q6_m1_speedup);
  std::printf("  Q4_K M=2 speedup: %.3fx\n", q4_m2_speedup);
  std::printf("  Q6_K M=2 speedup: %.3fx\n", q6_m2_speedup);
  return 0;
#endif
}
