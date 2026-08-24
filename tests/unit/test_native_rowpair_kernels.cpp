#include <catch2/catch_amalgamated.hpp>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace native = inferflux::runtime::cuda::native;
namespace GGUF = native::GGUF;
using inferflux::NativeExecutionPolicy;
using inferflux::FusedQuantGemm;
using inferflux::PackedProjectionSpec;
using inferflux::QuantizedWeightInfo;

namespace {

unsigned short EncodeHalfBits(float value) {
  const half h = __float2half(value);
  unsigned short bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

std::vector<half> MakeWaveTensor(size_t count, float scale, float bias = 0.0f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    const float value = bias + scale * std::sin(0.173f * static_cast<float>(i) +
                                                0.031f);
    out[i] = __float2half(value);
  }
  return out;
}

std::vector<native::block_q4_k> MakeQ4Rows(int rows, int blocks_per_row,
                                           int seed) {
  std::vector<native::block_q4_k> blocks(static_cast<size_t>(rows) *
                                         blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
      block.d = EncodeHalfBits(0.02f * static_cast<float>(((row + blk + seed) % 4) + 1));
      block.dmin = EncodeHalfBits(0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] = static_cast<unsigned char>((seed * 17 + row * 7 +
                                                     blk * 11 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] = static_cast<unsigned char>((seed * 11 + row * 13 +
                                                  blk * 19 + i * 3) & 0xFF);
      }
    }
  }
  return blocks;
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  REQUIRE(cudaMemcpy(host.data(), device, count * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

} // namespace

#ifdef INFERFLUX_HAS_CUDA

TEST_CASE("Native grouped row-pair FFN path matches generic grouping",
          "[native_forward][cuda_runtime_contract][native_rowpair]") {
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count <= 0) {
    SUCCEED("No CUDA device available; skipping row-pair FFN parity test.");
    return;
  }

  constexpr int kM = 2;
  constexpr int kK = 2048;
  constexpr int kN0 = 11008;
  constexpr int kN1 = 11008;
  const int blocks_per_row = kK / QK_K;

  const auto w0 = MakeQ4Rows(kN0, blocks_per_row, 3);
  const auto w1 = MakeQ4Rows(kN1, blocks_per_row, 7);
  const auto input = MakeWaveTensor(static_cast<size_t>(kM) * kK, 0.015f, -0.002f);

  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out_generic = nullptr;
  half *d_out_rowpair = nullptr;
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     w0.size() * sizeof(native::block_q4_k)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     w1.size() * sizeof(native::block_q4_k)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     input.size() * sizeof(half)) == cudaSuccess);
      REQUIRE(cudaMalloc(&d_act_q8_1,
                     static_cast<size_t>(kM) * (kK / QK8_1) *
                         sizeof(native::block_q8_1)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out_generic),
                     static_cast<size_t>(kM) * (kN0 + kN1) * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out_rowpair),
                     static_cast<size_t>(kM) * (kN0 + kN1) * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, kM, kK, stream);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN0) * kK},
       d_out_generic,
       kN0},
      {{d_w1, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN1) * kK},
       d_out_generic + static_cast<size_t>(kM) * kN0,
       kN1},
  }};
  const std::array<PackedProjectionSpec, 2> projections_rowpair = {{
      {{d_w0, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN0) * kK},
       d_out_rowpair,
       kN0},
      {{d_w1, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN1) * kK},
       d_out_rowpair + static_cast<size_t>(kM) * kN0,
       kN1},
  }};

  NativeExecutionPolicy default_policy;
  NativeExecutionPolicy rowpair_policy;
  rowpair_policy.enable_experimental_q81_grouped_rowpair_w4 = true;

  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, kM, kK,
                                       stream, &default_policy));
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections_rowpair, d_act_q8_1, kM, kK,
                                       stream, &rowpair_policy));
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const auto generic_output =
      CopyDeviceHalfs(d_out_generic, static_cast<size_t>(kM) * (kN0 + kN1));
  const auto rowpair_output =
      CopyDeviceHalfs(d_out_rowpair, static_cast<size_t>(kM) * (kN0 + kN1));
  for (size_t i = 0; i < generic_output.size(); ++i) {
    const float diff = std::fabs(__half2float(generic_output[i]) -
                                __half2float(rowpair_output[i]));
    REQUIRE(diff < 1e-3f);
  }

  cudaFree(d_out_rowpair);
  cudaFree(d_out_generic);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
}

TEST_CASE("Native down-proj row-pair kernel matches reference math",
          "[native_forward][cuda_runtime_contract][native_rowpair]") {
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count <= 0) {
    SUCCEED("No CUDA device available; skipping down-proj parity test.");
    return;
  }

  constexpr int kM = 2;
  constexpr int kK = 11008;
  constexpr int kN = 2048;
  const int blocks_per_row = kK / QK_K;

  const auto weights = MakeQ4Rows(kN, blocks_per_row, 11);
  const auto input = MakeWaveTensor(static_cast<size_t>(kM) * kK, 0.009f, -0.001f);

  native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     weights.size() * sizeof(native::block_q4_k)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_act_q8_1,
                     static_cast<size_t>(kM) * (kK / QK8_1) *
                         sizeof(native::block_q8_1)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out),
                     static_cast<size_t>(kM) * kN * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq),
                     static_cast<size_t>(kN) * kK * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, weights.data(),
                     weights.size() * sizeof(native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, kM, kK, stream);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  QuantizedWeightInfo info{d_w, static_cast<int>(GGUF::TensorType::Q4_K),
                           static_cast<int64_t>(kN) * kK};
  NativeExecutionPolicy policy;
  REQUIRE(
      FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, kM, kN, kK, stream,
                                &policy));
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  REQUIRE(native::dequantize_q4_k(d_w, d_deq, static_cast<size_t>(kN) * kK,
                                  stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const auto actual = CopyDeviceHalfs(d_out, static_cast<size_t>(kM) * kN);
  const auto deq_weights = CopyDeviceHalfs(d_deq, static_cast<size_t>(kN) * kK);
  std::vector<native::block_q8_1> act_blocks(static_cast<size_t>(kM) *
                                             (kK / QK8_1));
  REQUIRE(cudaMemcpy(act_blocks.data(), d_act_q8_1,
                     act_blocks.size() * sizeof(native::block_q8_1),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(static_cast<size_t>(kM) * kK, 0.0f);
  for (int m = 0; m < kM; ++m) {
    for (int blk = 0; blk < kK / QK8_1; ++blk) {
      const auto &a = act_blocks[m * (kK / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * kK + blk * QK8_1 + j] =
            scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> reference(static_cast<size_t>(kM) * kN, 0.0f);
  for (int m = 0; m < kM; ++m) {
    for (int row = 0; row < kN; ++row) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k) {
        const float activation = act_ref[m * kK + k];
        const float weight = __half2float(deq_weights[row * kK + k]);
        acc += activation * weight;
      }
      reference[m * kN + row] = acc;
    }
  }

  for (size_t idx = 0; idx < actual.size(); ++idx) {
    const float diff = std::fabs(__half2float(actual[idx]) - reference[idx]);
    REQUIRE(diff < 1e-1f);
  }

  cudaFree(d_deq);
  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w);
  cudaStreamDestroy(stream);
}

#else

TEST_CASE("Native row-pair kernel parity requires CUDA",
          "[native_rowpair]") {
  SUCCEED("Native row-pair kernel parity tests require CUDA.");
}

#endif
#endif // INFERFLUX_NATIVE_KERNELS_READY
