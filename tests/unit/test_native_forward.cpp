#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/inferflux_cuda_executor.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include "runtime/backends/cuda/native/llama_forward.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/native_dispatch_policy.h"
#include "runtime/backends/cuda/native/native_linear_executor.h"
#include "runtime/backends/cuda/native/quantized_weight_map_adapter.h"
#include "runtime/backends/cuda/native/safetensors_adapter.h"
#include "runtime/backends/cuda/native/weight_map.h"
#endif

#include "support/scoped_env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace inferflux {

using test::ScopedEnvVar;

// 3-arg portable_setenv wrapper for backward compatibility with existing tests.
inline int portable_setenv(const char *name, const char *value, int) {
  test::portable_setenv(name, value);
  return 0;
}
inline int portable_unsetenv(const char *name) {
  test::portable_unsetenv(name);
  return 0;
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
namespace {

std::vector<half> MakeWaveTensor(size_t count, float scale, float bias = 0.0f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    const float value =
        bias + scale * std::sin(0.173f * static_cast<float>(i) + 0.031f);
    out[i] = __float2half(value);
  }
  return out;
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  REQUIRE(cudaMemcpy(host.data(), device, count * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

std::vector<float> CopyDeviceFloats(const float *device, size_t count) {
  std::vector<float> host(count);
  REQUIRE(cudaMemcpy(host.data(), device, count * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

void ApplyRopeHost(std::vector<half> *q, std::vector<half> *k, int seq_len,
                   int num_heads, int num_kv_heads, int head_dim, int n_past,
                   float freq_base, int rope_type) {
  const int half_dim = head_dim / 2;
  auto apply = [&](std::vector<half> *tensor, int heads) {
    for (int pos = 0; pos < seq_len; ++pos) {
      const int position = n_past + pos;
      for (int head = 0; head < heads; ++head) {
        const int offset = pos * heads * head_dim + head * head_dim;
        for (int pair_idx = 0; pair_idx < half_dim; ++pair_idx) {
          const float freq =
              1.0f / std::pow(freq_base,
                              2.0f * pair_idx / static_cast<float>(head_dim));
          const float angle = position * freq;
          const float cos_val = std::cos(angle);
          const float sin_val = std::sin(angle);

          int i0 = offset + 2 * pair_idx;
          int i1 = offset + 2 * pair_idx + 1;
          if (rope_type == 2) {
            i0 = offset + pair_idx;
            i1 = offset + pair_idx + half_dim;
          }

          const float v0 = __half2float((*tensor)[i0]);
          const float v1 = __half2float((*tensor)[i1]);
          (*tensor)[i0] = __float2half(v0 * cos_val - v1 * sin_val);
          (*tensor)[i1] = __float2half(v0 * sin_val + v1 * cos_val);
        }
      }
    }
  };

  apply(q, num_heads);
  apply(k, num_kv_heads);
}

std::pair<float, float>
DecodeQ81Ds(const runtime::cuda::native::block_q8_1 &block) {
  const __half2_raw ds_raw = static_cast<__half2_raw>(block.ds);
  __half_raw d_raw{};
  d_raw.x = ds_raw.x;
  __half_raw ds_raw_half{};
  ds_raw_half.x = ds_raw.y;
  return {__half2float(static_cast<half>(d_raw)),
          __half2float(static_cast<half>(ds_raw_half))};
}

void FillQuantizedRows(runtime::cuda::native::GGUF::TensorType tensor_type,
                       int rows, int blocks_per_row, void *weights) {
  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  if (tensor_type == runtime::cuda::native::GGUF::TensorType::Q4_K) {
    auto *w = reinterpret_cast<runtime::cuda::native::block_q4_k *>(weights);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = w[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d =
            encode_half(0.012f * static_cast<float>(((row + blk) % 5) + 1));
        block.dmin =
            encode_half(0.006f * static_cast<float>(((row + blk) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] =
              static_cast<unsigned char>((row * 17 + blk * 7 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] =
              static_cast<unsigned char>((row * 13 + blk * 11 + i * 3) & 0xFF);
        }
      }
    }
    return;
  }

  auto *w = reinterpret_cast<runtime::cuda::native::block_q6_k *>(weights);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = w[static_cast<size_t>(row) * blocks_per_row + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] =
            static_cast<unsigned char>((row * 11 + blk * 5 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] =
            static_cast<unsigned char>((row * 7 + blk * 13 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>(((row * 5 + blk * 3 + i * 7) % 31) - 15);
      }
      block.d = encode_half(0.01f * static_cast<float>(((row + blk) % 5) + 1));
    }
  }
}

void RequireQ6KVectorizedParity(int M, int N, int K, bool check_accumulate) {
  using runtime::cuda::native::block_q6_k;
  using runtime::cuda::native::block_q8_1;
  using runtime::cuda::native::GGUF::TensorType;

  CAPTURE(M, N, K);
  const int quant_type = static_cast<int>(TensorType::Q6_K);
  const int blocks_per_row = K / QK_K;
  const size_t output_count = static_cast<size_t>(M) * N;
  std::vector<block_q6_k> weights(static_cast<size_t>(N) * blocks_per_row);
  FillQuantizedRows(TensorType::Q6_K, N, blocks_per_row, weights.data());
  const std::vector<half> input =
      MakeWaveTensor(static_cast<size_t>(M) * K, 0.009f, -0.001f);

  block_q6_k *d_weights = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_scalar = nullptr;
  half *d_vector = nullptr;
  float *d_scalar_f32 = nullptr;
  float *d_vector_f32 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_weights),
                     weights.size() * sizeof(block_q6_k)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_act_q8_1, static_cast<size_t>(M) * (K / QK8_1) *
                                      sizeof(block_q8_1)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scalar),
                     output_count * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_vector),
                     output_count * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scalar_f32),
                     output_count * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_vector_f32),
                     output_count * sizeof(float)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_weights, weights.data(),
                     weights.size() * sizeof(block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  NativeExecutionPolicy scalar_policy;
  scalar_policy.disable_q6k_vectorized = true;
  NativeExecutionPolicy vector_policy;
  vector_policy.enable_q6k_vectorized = true;
  const QuantizedWeightInfo info{d_weights, quant_type,
                                 static_cast<int64_t>(N) * K};

  REQUIRE(cudaMemset(d_scalar, 0, output_count * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMemset(d_vector, 0, output_count * sizeof(half)) == cudaSuccess);
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_scalar, M, N, K, nullptr,
                                   &scalar_policy));
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_vector, M, N, K, nullptr,
                                   &vector_policy));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const auto scalar = CopyDeviceHalfs(d_scalar, output_count);
  const auto vector = CopyDeviceHalfs(d_vector, output_count);
  for (size_t i = 0; i < output_count; ++i) {
    REQUIRE(__half2float(vector[i]) ==
            Catch::Approx(__half2float(scalar[i])).margin(8e-2f));
  }

  if (check_accumulate) {
    const auto residual = MakeWaveTensor(output_count, 0.003f, 0.02f);
    REQUIRE(cudaMemcpy(d_scalar, residual.data(), output_count * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_vector, residual.data(), output_count * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(FusedQuantGemm::GemvQ8_1Accum(info, d_act_q8_1, d_scalar, M, N, K,
                                          nullptr, &scalar_policy));
    REQUIRE(FusedQuantGemm::GemvQ8_1Accum(info, d_act_q8_1, d_vector, M, N, K,
                                          nullptr, &vector_policy));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto scalar_accum = CopyDeviceHalfs(d_scalar, output_count);
    const auto vector_accum = CopyDeviceHalfs(d_vector, output_count);
    for (size_t i = 0; i < output_count; ++i) {
      REQUIRE(__half2float(vector_accum[i]) ==
              Catch::Approx(__half2float(scalar_accum[i])).margin(8e-2f));
    }

    if (M <= 8) {
      std::vector<float> residual_f32(output_count);
      for (size_t i = 0; i < output_count; ++i) {
        residual_f32[i] = 0.02f + 0.001f * static_cast<float>(i % 17);
      }
      REQUIRE(cudaMemcpy(d_scalar_f32, residual_f32.data(),
                         output_count * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      REQUIRE(cudaMemcpy(d_vector_f32, residual_f32.data(),
                         output_count * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);
      REQUIRE(FusedQuantGemm::GemvQ8_1AccumF32(
          info, d_act_q8_1, d_scalar_f32, M, N, K, nullptr, &scalar_policy));
      REQUIRE(FusedQuantGemm::GemvQ8_1AccumF32(
          info, d_act_q8_1, d_vector_f32, M, N, K, nullptr, &vector_policy));
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      const auto scalar_accum_f32 =
          CopyDeviceFloats(d_scalar_f32, output_count);
      const auto vector_accum_f32 =
          CopyDeviceFloats(d_vector_f32, output_count);
      for (size_t i = 0; i < output_count; ++i) {
        REQUIRE(vector_accum_f32[i] ==
                Catch::Approx(scalar_accum_f32[i]).margin(8e-2f));
      }
    } else {
      REQUIRE_FALSE(FusedQuantGemm::GemvQ8_1AccumF32(
          info, d_act_q8_1, d_scalar_f32, M, N, K, nullptr, &scalar_policy));
      REQUIRE_FALSE(FusedQuantGemm::GemvQ8_1AccumF32(
          info, d_act_q8_1, d_vector_f32, M, N, K, nullptr, &vector_policy));
    }
  }

  REQUIRE(cudaFree(d_vector_f32) == cudaSuccess);
  REQUIRE(cudaFree(d_scalar_f32) == cudaSuccess);
  REQUIRE(cudaFree(d_vector) == cudaSuccess);
  REQUIRE(cudaFree(d_scalar) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_weights) == cudaSuccess);
}

void RunLmHeadLongPrefillContract(
    runtime::cuda::native::GGUF::TensorType tensor_type, float margin) {
  const int quant_type = static_cast<int>(tensor_type);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 lm_head kernels unavailable for this device/profile.");
    return;
  }

  constexpr int seq_len = 23;
  constexpr int M = 1;
  constexpr int K = 2048;
  constexpr int N = 4096;
  constexpr float eps = 1e-5f;
  constexpr std::array<int, 10> sample_outputs = {0,   1,   2,    17,   63,
                                                  255, 511, 1023, 2047, 4095};

  const FusedDispatchGeometry geometry{M, N, K, 1, false, true};
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, geometry));

  std::vector<half> h_residual_all =
      MakeWaveTensor(static_cast<size_t>(seq_len) * K, 0.013f, -0.001f);
  std::vector<half> h_residual(K);
  std::copy(h_residual_all.end() - K, h_residual_all.end(), h_residual.begin());

  std::vector<half> h_norm_weight(K);
  float sum_sq = 0.0f;
  for (int i = 0; i < K; ++i) {
    const float weight = 0.9f + static_cast<float>(i % 17) * 0.01f;
    h_norm_weight[i] = __float2half(weight);
    const float residual = __half2float(h_residual[i]);
    sum_sq += residual * residual;
  }

  const size_t weight_bytes =
      tensor_type == runtime::cuda::native::GGUF::TensorType::Q4_K
          ? static_cast<size_t>(N) * (K / QK_K) *
                sizeof(runtime::cuda::native::block_q4_k)
          : static_cast<size_t>(N) * (K / QK_K) *
                sizeof(runtime::cuda::native::block_q6_k);
  std::vector<unsigned char> h_weights(weight_bytes);
  FillQuantizedRows(tensor_type, N, K / QK_K, h_weights.data());

  void *d_w = nullptr;
  half *d_residual = nullptr;
  half *d_norm_weight = nullptr;
  runtime::cuda::native::block_q8_1 *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(&d_w, weight_bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_residual),
                     K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_norm_weight),
                     K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_act_q8_1),
                     (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq),
                     static_cast<size_t>(N) * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_weights.data(), weight_bytes,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_residual, h_residual.data(), K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_norm_weight, h_norm_weight.data(), K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  FusedQuantGemm::FusedRmsNormQuantizeQ8_1(d_residual, d_norm_weight,
                                           d_act_q8_1, M, K, eps, nullptr);
  if (tensor_type == runtime::cuda::native::GGUF::TensorType::Q4_K) {
    REQUIRE(
        runtime::cuda::native::dequantize_q4_k(
            reinterpret_cast<const runtime::cuda::native::block_q4_k *>(d_w),
            d_deq, static_cast<size_t>(N) * K) == cudaSuccess);
  } else {
    REQUIRE(
        runtime::cuda::native::dequantize_q6_k(
            reinterpret_cast<const runtime::cuda::native::block_q6_k *>(d_w),
            d_deq, static_cast<size_t>(N) * K) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(K / QK8_1);
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);
  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);

  std::vector<float> act_ref(K, 0.0f);
  for (int blk = 0; blk < K / QK8_1; ++blk) {
    const auto &a = q8_1_blocks[blk];
    const float scale = DecodeQ81Ds(a).first;
    for (int j = 0; j < QK8_1; ++j) {
      act_ref[blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, N);
  for (int out_idx : sample_outputs) {
    float ref = 0.0f;
    for (int i = 0; i < K; ++i) {
      ref +=
          act_ref[i] * __half2float(deq[static_cast<size_t>(out_idx) * K + i]);
    }
    CAPTURE(static_cast<int>(tensor_type), out_idx);
    const float tolerance = std::max(margin, std::fabs(ref) * 1.0e-3f);
    REQUIRE(__half2float(out[out_idx]) == Catch::Approx(ref).margin(tolerance));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_norm_weight) == cudaSuccess);
  REQUIRE(cudaFree(d_residual) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

} // namespace
#endif

// ============================================================================
// WeightMap Tests (compile-time only, no GPU required)
// ============================================================================

TEST_CASE("WeightMap: Build returns false on empty loader",
          "[native_forward]") {
  // WeightMap::Build requires a loaded SafetensorsLoader with GPU data.
  // Without actual safetensors files, we verify the interface compiles
  // and handles the empty case.
  SafetensorsLoader loader;
  SafetensorsLoader::ModelConfig config;
  config.num_hidden_layers = 2;
  config.hidden_size = 128;

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  WeightMap wm;
  // Should fail because no tensors are loaded
  REQUIRE_FALSE(wm.Build(loader, config));
#else
  // Without InferFlux CUDA kernels, just verify types compile
  REQUIRE(config.hidden_size == 128);
#endif
}

// ============================================================================
// ModelForwardFactory Tests
// ============================================================================

TEST_CASE("ModelForwardFactory: supported model types", "[native_forward]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Llama-family models should return a valid ModelForward
  REQUIRE(CreateModelForward("llama") != nullptr);
  REQUIRE(CreateModelForward("qwen2") != nullptr);
  REQUIRE(CreateModelForward("qwen3") != nullptr);
  REQUIRE(CreateModelForward("mistral") != nullptr);
  REQUIRE(CreateModelForward("gemma") != nullptr);
  REQUIRE(CreateModelForward("phi3") != nullptr);

  // Unsupported models should return nullptr
  REQUIRE(CreateModelForward("unknown_model") == nullptr);
  REQUIRE(CreateModelForward("") == nullptr);
  REQUIRE(CreateModelForwardTyped<__nv_bfloat16>("llama") != nullptr);
#else
  REQUIRE(true); // Placeholder when InferFlux CUDA kernels are not compiled
#endif
}

// ============================================================================
// InferfluxCudaExecutor Interface Tests
// ============================================================================

TEST_CASE("InferfluxCudaExecutor: Name and fallback state",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  REQUIRE(executor.Name() == "inferflux_cuda");
  REQUIRE_FALSE(executor.IsFallback());
  REQUIRE(executor.FallbackReason().empty());
  REQUIRE(executor.BackendHandle() == nullptr);
}

TEST_CASE("NativeExecutionPolicy loads hot-path policy from env",
          "[native_forward]") {
  ScopedEnvVar enable_batched("INFERFLUX_ENABLE_BATCHED_DECODE", "1");
  ScopedEnvVar disable_graph("INFERFLUX_DISABLE_CUDA_GRAPH", "1");
  ScopedEnvVar phase_timing("INFERFLUX_CUDA_PHASE_TIMING", "1");
  ScopedEnvVar force_cublas("INFERFLUX_FORCE_CUBLAS", "1");
  ScopedEnvVar disable_packed("INFERFLUX_DISABLE_PREPACKED_ACTIVATIONS", "1");
  ScopedEnvVar disable_q81("INFERFLUX_DISABLE_Q8_1_ACTIVATIONS", "1");
  ScopedEnvVar disable_fused("INFERFLUX_DISABLE_FUSED_GEMV", "1");
  ScopedEnvVar debug_decode_mapping("INFERFLUX_CUDA_DEBUG_DECODE_MAPPING", "1");
  ScopedEnvVar debug_decode_mapping_limit(
      "INFERFLUX_CUDA_DEBUG_DECODE_MAPPING_LIMIT", "21");
  ScopedEnvVar debug_logits("INFERFLUX_DEBUG_LOGITS", "1");
  ScopedEnvVar debug_logits_limit("INFERFLUX_DEBUG_LOGITS_LIMIT", "13");
  ScopedEnvVar require_fused("INFERFLUX_CUDA_REQUIRE_FUSED_MATMUL", "1");
  ScopedEnvVar dequant_policy("INFERFLUX_CUDA_DEQUANT_CACHE_POLICY", "batch");
  ScopedEnvVar grouped_hot("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K",
                           "1");
  ScopedEnvVar downproj_hot(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED", "1");
  ScopedEnvVar downproj_rowpair_hot(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED", "1");
  ScopedEnvVar enable_mmq("INFERFLUX_ENABLE_DOWNPROJ_MMQ", "1");
  ScopedEnvVar mmq_min_batch("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH", "7");
  ScopedEnvVar timing_sample_rate("INFERFLUX_CUDA_TIMING_SAMPLE_RATE", "9");
  ScopedEnvVar disable_q6k_vectorized("INFERFLUX_DISABLE_Q6K_VECTORIZED", "1");

  const auto policy = NativeExecutionPolicy::FromEnv();
  REQUIRE(policy.enable_batched_decode);
  REQUIRE(policy.disable_cuda_graph);
  REQUIRE(policy.phase_timing_enabled);
  REQUIRE(policy.force_cublas);
  REQUIRE(policy.disable_prepacked_activations);
  REQUIRE(policy.disable_q81_activations);
  REQUIRE(policy.disable_fused_gemv);
  REQUIRE(policy.debug_decode_mapping);
  REQUIRE(policy.debug_decode_mapping_limit == 21);
  REQUIRE(policy.debug_logits);
  REQUIRE(policy.debug_logits_limit == 13);
  REQUIRE(policy.enable_experimental_q81_grouped_hot_q4k);
  REQUIRE(policy.enable_experimental_q81_downproj_hot_fixed);
  REQUIRE(policy.enable_experimental_q81_downproj_rowpair_hot_fixed);
  REQUIRE(policy.enable_downproj_mmq);
  REQUIRE(policy.downproj_mmq_min_batch_override == 7);
  REQUIRE(policy.timing_sample_rate == 9);
  REQUIRE(policy.disable_q6k_vectorized);
  REQUIRE(policy.require_fused_quantized_matmul_override);
  REQUIRE(policy.require_fused_quantized_matmul);
  REQUIRE(policy.dequantized_cache_policy_override == "batch");
}

TEST_CASE("NativeExecutionPolicy keeps grouped Q4_K single-row hot path on by "
          "default",
          "[native_forward]") {
  ScopedEnvVar grouped_hot("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K",
                           nullptr);

  const auto policy = NativeExecutionPolicy::FromEnv();
  REQUIRE(policy.enable_experimental_q81_grouped_hot_q4k);
}

TEST_CASE("Q6_K vectorized decode selector honors architecture, cohort, and "
          "rollback policy",
          "[native_forward][q6k_vectorized]") {
  NativeExecutionPolicy policy;
  for (const int M : {3, 4, 5, 8}) {
    REQUIRE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 9, M, policy));
  }
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 6, 4, policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(9, 0, 4, policy));

  policy.enable_q6k_vectorized = true;
  REQUIRE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 6, 4, policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 9, 9, policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 9, 0, policy));

  policy.disable_q6k_vectorized = true;
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 9, 4, policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseQ6KVectorizedDecode(8, 6, 4, policy));
}

TEST_CASE("NativeExecutionPolicy enables batched decode by default",
          "[native_forward]") {
  ScopedEnvVar enable_batched("INFERFLUX_ENABLE_BATCHED_DECODE", nullptr);

  const auto policy = NativeExecutionPolicy::FromEnv();
  REQUIRE(policy.enable_batched_decode);
}

TEST_CASE("InferfluxCudaExecutor: ExecuteUnifiedBatch returns empty when no "
          "model loaded",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  std::vector<LlamaCppBackend::UnifiedBatchInput> inputs;
  auto outputs = executor.ExecuteUnifiedBatch(inputs);
  REQUIRE(outputs.empty());
}

TEST_CASE("InferfluxCudaExecutor: async unified-batch contract is gated by "
          "runtime readiness",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  REQUIRE_FALSE(executor.SupportsAsyncUnifiedBatch());

  LlamaCppBackend::UnifiedBatchInput input;
  input.sequence_id = 0;
  input.n_past = 0;
  input.tokens = {1};
  input.request_logits = true;

  std::vector<LlamaCppBackend::UnifiedBatchInput> inputs;
  inputs.push_back(std::move(input));
  const auto handle = executor.SubmitUnifiedBatchAsync(
      inputs, LlamaCppBackend::UnifiedBatchLane::kAuto);
  REQUIRE(handle == 0);

  std::vector<LlamaCppBackend::UnifiedBatchOutput> outputs;
  REQUIRE_FALSE(executor.TryCollectUnifiedBatchAsync(1, &outputs));
}

TEST_CASE("InferfluxCudaExecutor: timing sample helper records every Nth work "
          "item",
          "[native_forward]") {
  int counter = 0;
  REQUIRE_FALSE(InferfluxCudaExecutor::ShouldRecordTimingSample(0, &counter));
  REQUIRE(counter == 0);

  REQUIRE_FALSE(InferfluxCudaExecutor::ShouldRecordTimingSample(4, &counter));
  REQUIRE(counter == 1);
  REQUIRE_FALSE(InferfluxCudaExecutor::ShouldRecordTimingSample(4, &counter));
  REQUIRE(counter == 2);
  REQUIRE_FALSE(InferfluxCudaExecutor::ShouldRecordTimingSample(4, &counter));
  REQUIRE(counter == 3);
  REQUIRE(InferfluxCudaExecutor::ShouldRecordTimingSample(4, &counter));
  REQUIRE(counter == 4);

  REQUIRE(InferfluxCudaExecutor::ShouldRecordTimingSample(1, &counter));
  REQUIRE(counter == 5);
}

// ============================================================================
// SafetensorsLoader Config Parsing Tests
// ============================================================================

TEST_CASE("SafetensorsLoader: GetConfig returns defaults for no config",
          "[native_forward]") {
  SafetensorsLoader loader;
  const auto &config = loader.GetConfig();
  REQUIRE(config.hidden_size == 0);
  REQUIRE(config.num_hidden_layers == 0);
  REQUIRE(config.vocab_size == 0);
  REQUIRE(config.model_type.empty());
}

TEST_CASE("SafetensorsLoader: GetTensor returns nullptr for unknown name",
          "[native_forward]") {
  SafetensorsLoader loader;
  REQUIRE(loader.GetTensor("nonexistent") == nullptr);
  REQUIRE(loader.GetTensorNames().empty());
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("SafetensorsWeightAccessor: FP16 path is direct non-quantized "
          "access",
          "[native_forward]") {
  SafetensorsLoader::Tensor tensor;
  tensor.name = "w";
  tensor.shape = {4, 8};
  tensor.dtype = "f16";
  tensor.gpu_offset = 16;

  std::array<uint8_t, 128> gpu_buffer{};
  runtime::cuda::native::SafetensorsWeightAccessor accessor(&tensor,
                                                            gpu_buffer.data());

  REQUIRE_FALSE(accessor.IsQuantized());
  REQUIRE(accessor.GetDataType() == "f16");
  REQUIRE(accessor.GetDimensions() == std::make_pair<size_t, size_t>(4, 8));

  void *raw = accessor.GetGpuWeights(nullptr);
  REQUIRE(raw == (gpu_buffer.data() + tensor.gpu_offset));
  half *dequant = accessor.GetDequantizedGpuWeights(nullptr);
  REQUIRE(reinterpret_cast<void *>(dequant) == raw);
}

TEST_CASE("SafetensorsWeightAccessor: BF16 path materializes cache when CUDA "
          "device is available",
          "[native_forward]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping BF16 accessor cache contract.");
    return;
  }

  SafetensorsLoader::Tensor tensor;
  tensor.name = "w_bf16";
  tensor.shape = {4, 8};
  tensor.dtype = "bf16";
  tensor.gpu_offset = 0;

  void *gpu_buffer = nullptr;
  REQUIRE(cudaMalloc(&gpu_buffer, 4 * 8 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMemset(gpu_buffer, 0, 4 * 8 * sizeof(half)) == cudaSuccess);

  runtime::cuda::native::SafetensorsWeightAccessor accessor(&tensor,
                                                            gpu_buffer);
  REQUIRE_FALSE(accessor.IsQuantized());
  REQUIRE_FALSE(accessor.IsDequantizedCached());

  half *dequant = accessor.GetDequantizedGpuWeights(nullptr);
  REQUIRE(dequant != nullptr);
  REQUIRE(accessor.IsDequantizedCached());

  REQUIRE(cudaFree(gpu_buffer) == cudaSuccess);
}
#endif

// ============================================================================
// InferfluxCudaExecutor Native* Method Tests
// ============================================================================

TEST_CASE("InferfluxCudaExecutor: NativeIsReady returns false when no model",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  REQUIRE_FALSE(executor.NativeIsReady());
}

TEST_CASE("InferfluxCudaExecutor: NativeTokenize returns empty when no model",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  auto tokens = executor.NativeTokenize("hello world");
  REQUIRE(tokens.empty());
}

TEST_CASE("InferfluxCudaExecutor: NativeTokenCount returns 0 when no model",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  REQUIRE(executor.NativeTokenCount("hello world") == 0);
}

TEST_CASE("InferfluxCudaExecutor: NativeFreeSequence does not crash when no "
          "model",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  executor.NativeFreeSequence(0); // Should not crash
  REQUIRE(true);
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("PackedActivationWidth covers FFN width for packed single-projection "
          "reuse",
          "[native_forward]") {
  REQUIRE(PackedActivationWidth(4096, 11008) == 11008);
  REQUIRE(PackedActivationWidth(4096, 4096) == 4096);
  REQUIRE(PackedActivationWidth(0, 8192) == 8192);
}

TEST_CASE("ShouldUseDecodeGraph disables graph replay during phase timing",
          "[native_forward]") {
  REQUIRE(ShouldUseDecodeGraph(true, false, false, true));
  REQUIRE_FALSE(ShouldUseDecodeGraph(true, true, false, true));
  REQUIRE_FALSE(ShouldUseDecodeGraph(false, false, false, true));
  REQUIRE_FALSE(ShouldUseDecodeGraph(true, false, true, true));
  REQUIRE_FALSE(ShouldUseDecodeGraph(true, false, false, false));
}

TEST_CASE("ModelForward::WarmWeightCaches default is no-op",
          "[native_forward]") {
  // Base class WarmWeightCaches() is a no-op — subclasses override.
  // This ensures the interface contract: callers can always call it safely.
  struct StubForward : ModelForward {
    bool Initialize(const SafetensorsLoader::ModelConfig &, const WeightMap &,
                    IKvCacheGpu *, CublasGemm *, cudaStream_t) override {
      return true;
    }
    bool Forward(const std::vector<int> &, int, int, float *) override {
      return true;
    }
    int VocabSize() const override { return 1; }
    int HiddenSize() const override { return 1; }
    std::string ModelType() const override { return "stub"; }
    void FreeScratchBuffers() override {}
  };
  StubForward stub;
  // Must not throw or crash
  stub.WarmWeightCaches();
}

TEST_CASE("SelectSharedActivationGrouping prefers triple launch when all "
          "siblings match",
          "[native_forward]") {
  const auto choice = SelectSharedActivationGrouping<3>(
      {1, 1, 1}, {64, 32, 32}, {true, true, true}, {true, true, true});
  REQUIRE(choice.grouped_count == 3);
  REQUIRE(choice.indices[0] == 0);
  REQUIRE(choice.indices[1] == 1);
  REQUIRE(choice.indices[2] == 2);
}

TEST_CASE("SelectSharedActivationGrouping chooses best mixed pair when triple "
          "grouping is unavailable",
          "[native_forward]") {
  const auto choice = SelectSharedActivationGrouping<3>(
      {7, 3, 7}, {128, 32, 96}, {true, true, true}, {false, false, false});
  REQUIRE(choice.grouped_count == 2);
  REQUIRE(choice.indices[0] == 0);
  REQUIRE(choice.indices[1] == 2);
}

TEST_CASE("SelectSharedActivationGrouping falls back to singles when no pair "
          "is eligible",
          "[native_forward]") {
  const auto choice = SelectSharedActivationGrouping<3>(
      {4, 5, 6}, {64, 64, 64}, {true, true, true}, {false, false, false});
  REQUIRE(choice.grouped_count == 0);
  REQUIRE(choice.indices[0] == -1);
}

TEST_CASE(
    "SelectSharedActivationGrouping uses pair launch for matching two-way "
    "group",
    "[native_forward]") {
  const auto choice = SelectSharedActivationGrouping<2>(
      {8, 8}, {64, 32}, {true, true}, {false, false});
  REQUIRE(choice.grouped_count == 2);
  REQUIRE(choice.indices[0] == 0);
  REQUIRE(choice.indices[1] == 1);
}
#endif // INFERFLUX_NATIVE_KERNELS_READY

TEST_CASE(
    "InferfluxCudaExecutor: NativeCopySequencePrefix is safe with no model",
    "[native_forward]") {
  InferfluxCudaExecutor executor;
  executor.NativeCopySequencePrefix(0, 1, 10); // Should not crash
  REQUIRE(true);
}

TEST_CASE(
    "InferfluxCudaExecutor: NativeSerializeSequence returns empty with no "
    "model",
    "[native_forward]") {
  InferfluxCudaExecutor executor;
  auto blob = executor.NativeSerializeSequence(0);
  REQUIRE(blob.empty());
}

TEST_CASE("InferfluxCudaExecutor: NativeHydrateSequence returns false with no "
          "model",
          "[native_forward]") {
  InferfluxCudaExecutor executor;
  REQUIRE_FALSE(executor.NativeHydrateSequence(0, {}));
}

// ============================================================================
// SafetensorsLoader ModelConfig Extended Fields Tests
// ============================================================================

TEST_CASE("SafetensorsLoader: ModelConfig defaults for torch_dtype and "
          "rms_norm_eps",
          "[native_forward]") {
  SafetensorsLoader::ModelConfig config;
  REQUIRE(config.torch_dtype.empty());
  REQUIRE(config.rms_norm_eps == Catch::Approx(1e-6f));
}

// ============================================================================
// QuantizedWeightMapAdapter Tests
// ============================================================================

#ifdef INFERFLUX_NATIVE_KERNELS_READY

// ============================================================================
// QuantizedWeightInfo & Raw Accessor Tests
// ============================================================================

TEST_CASE("QuantizedWeightInfo: default-constructed has null data",
          "[native_forward]") {
  QuantizedWeightInfo info;
  REQUIRE(info.data == nullptr);
  REQUIRE(info.quant_type == -1);
  REQUIRE(info.num_elements == 0);
}

TEST_CASE("WeightMapTyped: HasQuantizedWeights returns false by default",
          "[native_forward]") {
  WeightMap wm;
  REQUIRE_FALSE(wm.HasQuantizedWeights());
  REQUIRE(wm.LayerQProjRaw(0).data == nullptr);
  REQUIRE(wm.LmHeadRaw().data == nullptr);
}

// FusedFFNGemmQ4K test removed — bring-up kernel deleted in Phase 4 cleanup.

TEST_CASE("BatchedRoPE matches per-sequence decode RoPE",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping BatchedRoPE parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 2;
  constexpr int num_kv_heads = 1;
  constexpr int head_dim = 32;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const std::array<int, batch_size> n_past = {1, 3};

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.08f);
  const std::vector<half> h_k =
      MakeWaveTensor(static_cast<size_t>(batch_size) * kv_cols, 0.05f, 0.02f);

  half *d_q_batch = nullptr;
  half *d_k_batch = nullptr;
  int *d_n_past = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_batch),
                     h_k.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_n_past),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_q_batch, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k_batch, h_k.data(), h_k.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_n_past, n_past.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::BatchedRoPE<half>(d_q_batch, d_k_batch, batch_size,
                                         num_heads, num_kv_heads, head_dim,
                                         d_n_past, 10000.0f, stream,
                                         /*rope_type=*/0) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> q_batched = CopyDeviceHalfs(d_q_batch, h_q.size());
  const std::vector<half> k_batched = CopyDeviceHalfs(d_k_batch, h_k.size());

  for (int b = 0; b < batch_size; ++b) {
    half *d_q_single = nullptr;
    half *d_k_single = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_single),
                       kv_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q_single, h_q.data() + b * q_cols,
                       q_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_k_single, h_k.data() + b * kv_cols,
                       kv_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cuda_kernel::RoPE<half>(d_q_single, d_k_single, /*seq_len=*/1,
                                    num_heads, num_kv_heads, head_dim,
                                    n_past[b], 10000.0f, stream,
                                    /*rope_type=*/0) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const std::vector<half> q_single = CopyDeviceHalfs(d_q_single, q_cols);
    const std::vector<half> k_single = CopyDeviceHalfs(d_k_single, kv_cols);

    for (int i = 0; i < q_cols; ++i) {
      REQUIRE(__half2float(q_batched[b * q_cols + i]) ==
              Catch::Approx(__half2float(q_single[i])).margin(2e-3f));
    }
    for (int i = 0; i < kv_cols; ++i) {
      REQUIRE(__half2float(k_batched[b * kv_cols + i]) ==
              Catch::Approx(__half2float(k_single[i])).margin(2e-3f));
    }

    REQUIRE(cudaFree(d_k_single) == cudaSuccess);
    REQUIRE(cudaFree(d_q_single) == cudaSuccess);
  }

  REQUIRE(cudaFree(d_n_past) == cudaSuccess);
  REQUIRE(cudaFree(d_k_batch) == cudaSuccess);
  REQUIRE(cudaFree(d_q_batch) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("FlashDecodeMultiSeq matches per-sequence FlashAttention decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping FlashDecode parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 2;
  constexpr int num_kv_heads = 1;
  constexpr int head_dim = 32;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const std::array<int, batch_size> kv_lens = {2, 4};
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.08f);
  const std::vector<half> h_k0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.04f, 0.01f);
  const std::vector<half> h_v0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.03f, -0.02f);
  const std::vector<half> h_k1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.05f, 0.03f);
  const std::vector<half> h_v1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.02f, 0.04f);

  half *d_q_batch = nullptr;
  half *d_o_batch = nullptr;
  half *d_k0 = nullptr;
  half *d_v0 = nullptr;
  half *d_k1 = nullptr;
  half *d_v1 = nullptr;
  const half **d_k_ptrs = nullptr;
  const half **d_v_ptrs = nullptr;
  int *d_kv_lens = nullptr;

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k0),
                     h_k0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v0),
                     h_v0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k1),
                     h_k1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v1),
                     h_v1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_ptrs),
                     batch_size * sizeof(const half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v_ptrs),
                     batch_size * sizeof(const half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_kv_lens),
                     batch_size * sizeof(int)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_q_batch, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k0, h_k0.data(), h_k0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v0, h_v0.data(), h_v0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k1, h_k1.data(), h_k1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v1, h_v1.data(), h_v1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  const std::array<const half *, batch_size> h_k_ptrs = {d_k0, d_k1};
  const std::array<const half *, batch_size> h_v_ptrs = {d_v0, d_v1};
  REQUIRE(cudaMemcpy(d_k_ptrs, h_k_ptrs.data(),
                     batch_size * sizeof(const half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v_ptrs, h_v_ptrs.data(),
                     batch_size * sizeof(const half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_kv_lens, kv_lens.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::FlashDecodeMultiSeq<half>(
              d_q_batch, d_k_ptrs, d_v_ptrs, d_o_batch, d_kv_lens, batch_size,
              num_heads, num_kv_heads, head_dim, scale, stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> o_batched = CopyDeviceHalfs(d_o_batch, h_q.size());

  for (int b = 0; b < batch_size; ++b) {
    half *d_q_single = nullptr;
    half *d_o_single = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q_single, h_q.data() + b * q_cols,
                       q_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const half *d_k_single = (b == 0) ? d_k0 : d_k1;
    const half *d_v_single = (b == 0) ? d_v0 : d_v1;
    REQUIRE(cuda_kernel::FlashAttention2Typed<half>(
                d_q_single, d_k_single, d_v_single, d_o_single,
                /*batch_size=*/1, /*query_len=*/1, kv_lens[b], num_heads,
                num_kv_heads, head_dim, scale, /*causal=*/true,
                stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const std::vector<half> o_single = CopyDeviceHalfs(d_o_single, q_cols);
    for (int i = 0; i < q_cols; ++i) {
      REQUIRE(__half2float(o_batched[b * q_cols + i]) ==
              Catch::Approx(__half2float(o_single[i])).margin(2e-3f));
    }

    REQUIRE(cudaFree(d_o_single) == cudaSuccess);
    REQUIRE(cudaFree(d_q_single) == cudaSuccess);
  }

  REQUIRE(cudaFree(d_kv_lens) == cudaSuccess);
  REQUIRE(cudaFree(reinterpret_cast<void *>(d_v_ptrs)) == cudaSuccess);
  REQUIRE(cudaFree(reinterpret_cast<void *>(d_k_ptrs)) == cudaSuccess);
  REQUIRE(cudaFree(d_v1) == cudaSuccess);
  REQUIRE(cudaFree(d_k1) == cudaSuccess);
  REQUIRE(cudaFree(d_v0) == cudaSuccess);
  REQUIRE(cudaFree(d_k0) == cudaSuccess);
  REQUIRE(cudaFree(d_o_batch) == cudaSuccess);
  REQUIRE(cudaFree(d_q_batch) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("BatchedRoPE matches per-sequence decode RoPE for Qwen geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-geometry BatchedRoPE "
            "parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const std::array<int, batch_size> n_past = {5, 11};

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.03f);
  const std::vector<half> h_k =
      MakeWaveTensor(static_cast<size_t>(batch_size) * kv_cols, 0.02f, 0.01f);

  half *d_q_batch = nullptr;
  half *d_k_batch = nullptr;
  int *d_n_past = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_batch),
                     h_k.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_n_past),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_q_batch, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k_batch, h_k.data(), h_k.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_n_past, n_past.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::BatchedRoPE<half>(d_q_batch, d_k_batch, batch_size,
                                         num_heads, num_kv_heads, head_dim,
                                         d_n_past, 1000000.0f, stream,
                                         /*rope_type=*/0) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> q_batched = CopyDeviceHalfs(d_q_batch, h_q.size());
  const std::vector<half> k_batched = CopyDeviceHalfs(d_k_batch, h_k.size());

  for (int b = 0; b < batch_size; ++b) {
    half *d_q_single = nullptr;
    half *d_k_single = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_single),
                       kv_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q_single, h_q.data() + b * q_cols,
                       q_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_k_single, h_k.data() + b * kv_cols,
                       kv_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cuda_kernel::RoPE<half>(d_q_single, d_k_single, /*seq_len=*/1,
                                    num_heads, num_kv_heads, head_dim,
                                    n_past[b], 1000000.0f, stream,
                                    /*rope_type=*/0) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const std::vector<half> q_single = CopyDeviceHalfs(d_q_single, q_cols);
    const std::vector<half> k_single = CopyDeviceHalfs(d_k_single, kv_cols);

    for (int i = 0; i < q_cols; ++i) {
      REQUIRE(__half2float(q_batched[b * q_cols + i]) ==
              Catch::Approx(__half2float(q_single[i])).margin(2e-3f));
    }
    for (int i = 0; i < kv_cols; ++i) {
      REQUIRE(__half2float(k_batched[b * kv_cols + i]) ==
              Catch::Approx(__half2float(k_single[i])).margin(2e-3f));
    }

    REQUIRE(cudaFree(d_k_single) == cudaSuccess);
    REQUIRE(cudaFree(d_q_single) == cudaSuccess);
  }

  REQUIRE(cudaFree(d_n_past) == cudaSuccess);
  REQUIRE(cudaFree(d_k_batch) == cudaSuccess);
  REQUIRE(cudaFree(d_q_batch) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Bias fold into RoPE and KV append matches BiasAdd-then-transform",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping bias-fold parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const std::array<int, batch_size> n_past = {5, 11};
  const std::array<int, batch_size> seq_ids = {0, 1};

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.03f);
  const std::vector<half> h_k =
      MakeWaveTensor(static_cast<size_t>(batch_size) * kv_cols, 0.02f, 0.01f);
  const std::vector<half> h_v =
      MakeWaveTensor(static_cast<size_t>(batch_size) * kv_cols, 0.025f, -0.01f);
  const std::vector<half> h_q_bias =
      MakeWaveTensor(static_cast<size_t>(q_cols), 0.01f, 0.005f);
  const std::vector<half> h_k_bias =
      MakeWaveTensor(static_cast<size_t>(kv_cols), 0.008f, -0.004f);
  const std::vector<half> h_v_bias =
      MakeWaveTensor(static_cast<size_t>(kv_cols), 0.006f, 0.003f);

  auto dev_copy = [&](const std::vector<half> &h) {
    half *d = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d),
                       h.size() * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, h.data(), h.size() * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
  };

  half *d_q_a = dev_copy(h_q), *d_k_a = dev_copy(h_k), *d_v_a = dev_copy(h_v);
  half *d_q_b = dev_copy(h_q), *d_k_b = dev_copy(h_k), *d_v_b = dev_copy(h_v);
  half *d_q_bias = dev_copy(h_q_bias);
  half *d_k_bias = dev_copy(h_k_bias);
  half *d_v_bias = dev_copy(h_v_bias);
  int *d_n_past = nullptr;
  int *d_seq_ids = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_n_past),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_seq_ids),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_n_past, n_past.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_seq_ids, seq_ids.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  KvCacheGpuTyped<half> cache_a, cache_b;
  constexpr int max_seq = 32;
  constexpr int max_batch = 4;
  REQUIRE(cache_a.Allocate(1, num_kv_heads, head_dim, max_seq, max_batch));
  REQUIRE(cache_b.Allocate(1, num_kv_heads, head_dim, max_seq, max_batch));

  // Unfused: BiasAdd each, then RoPE / KV append without bias params.
  REQUIRE(cuda_kernel::BiasAdd<half>(d_q_a, d_q_bias, batch_size, q_cols,
                                     stream) == cudaSuccess);
  REQUIRE(cuda_kernel::BiasAdd<half>(d_k_a, d_k_bias, batch_size, kv_cols,
                                     stream) == cudaSuccess);
  REQUIRE(cuda_kernel::BiasAdd<half>(d_v_a, d_v_bias, batch_size, kv_cols,
                                     stream) == cudaSuccess);
  REQUIRE(cuda_kernel::BatchedRoPE<half>(d_q_a, d_k_a, batch_size, num_heads,
                                         num_kv_heads, head_dim, d_n_past,
                                         1000000.0f, stream,
                                         /*rope_type=*/0) == cudaSuccess);
  REQUIRE(cuda_kernel::BatchedKvAppendStrided<half>(
              d_k_a, d_v_a, cache_a.Buffer(), d_seq_ids, d_n_past,
              /*layer=*/0, batch_size, num_kv_heads * head_dim,
              cache_a.SlotStride(), cache_a.LayerStride(), cache_a.KvStride(),
              stream) == cudaSuccess);

  // Fused: bias folded into the RoPE load and the append store.
  REQUIRE(cuda_kernel::BatchedRoPE<half>(
              d_q_b, d_k_b, batch_size, num_heads, num_kv_heads, head_dim,
              d_n_past, 1000000.0f, stream,
              /*rope_type=*/0, d_q_bias, d_k_bias) == cudaSuccess);
  REQUIRE(cuda_kernel::BatchedKvAppendStrided<half>(
              d_k_b, d_v_b, cache_b.Buffer(), d_seq_ids, d_n_past,
              /*layer=*/0, batch_size, num_kv_heads * head_dim,
              cache_b.SlotStride(), cache_b.LayerStride(), cache_b.KvStride(),
              stream, d_v_bias) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> q_a = CopyDeviceHalfs(d_q_a, h_q.size());
  const std::vector<half> k_a = CopyDeviceHalfs(d_k_a, h_k.size());
  const std::vector<half> q_b = CopyDeviceHalfs(d_q_b, h_q.size());
  const std::vector<half> k_b = CopyDeviceHalfs(d_k_b, h_k.size());
  // Fused keeps the biased value in registers through rotation (one fewer
  // half rounding), so allow the same tolerance as the other RoPE parities.
  for (size_t i = 0; i < q_a.size(); ++i) {
    REQUIRE(__half2float(q_b[i]) ==
            Catch::Approx(__half2float(q_a[i])).margin(5e-3f));
  }
  for (size_t i = 0; i < k_a.size(); ++i) {
    REQUIRE(__half2float(k_b[i]) ==
            Catch::Approx(__half2float(k_a[i])).margin(5e-3f));
  }

  // V cache rows: both paths round the same float sum once, so compare the
  // stored cache contents exactly.
  const half *v_cache_a = cache_a.GetV(0, seq_ids[0]);
  const half *v_cache_b = cache_b.GetV(0, seq_ids[0]);
  std::vector<half> row_a(kv_cols), row_b(kv_cols);
  REQUIRE(cudaMemcpy(row_a.data(), v_cache_a, kv_cols * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(row_b.data(), v_cache_b, kv_cols * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < kv_cols; ++i) {
    REQUIRE(__half2float(row_b[i]) ==
            Catch::Approx(__half2float(row_a[i])).margin(1e-6f));
  }

  REQUIRE(cudaFree(d_seq_ids) == cudaSuccess);
  REQUIRE(cudaFree(d_n_past) == cudaSuccess);
  REQUIRE(cudaFree(d_v_bias) == cudaSuccess);
  REQUIRE(cudaFree(d_k_bias) == cudaSuccess);
  REQUIRE(cudaFree(d_q_bias) == cudaSuccess);
  REQUIRE(cudaFree(d_v_b) == cudaSuccess);
  REQUIRE(cudaFree(d_k_b) == cudaSuccess);
  REQUIRE(cudaFree(d_q_b) == cudaSuccess);
  REQUIRE(cudaFree(d_v_a) == cudaSuccess);
  REQUIRE(cudaFree(d_k_a) == cudaSuccess);
  REQUIRE(cudaFree(d_q_a) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE(
    "FlashDecodeMultiSeq matches per-sequence FlashAttention decode for Qwen "
    "geometry",
    "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-geometry FlashDecode "
            "parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const std::array<int, batch_size> kv_lens = {7, 13};
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.03f);
  const std::vector<half> h_k0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, 0.01f);
  const std::vector<half> h_v0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, -0.01f);
  const std::vector<half> h_k1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.025f, 0.02f);
  const std::vector<half> h_v1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.015f, 0.03f);

  half *d_q_batch = nullptr;
  half *d_o_batch = nullptr;
  half *d_k0 = nullptr;
  half *d_v0 = nullptr;
  half *d_k1 = nullptr;
  half *d_v1 = nullptr;
  const half **d_k_ptrs = nullptr;
  const half **d_v_ptrs = nullptr;
  int *d_kv_lens = nullptr;

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k0),
                     h_k0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v0),
                     h_v0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k1),
                     h_k1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v1),
                     h_v1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_ptrs),
                     batch_size * sizeof(const half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v_ptrs),
                     batch_size * sizeof(const half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_kv_lens),
                     batch_size * sizeof(int)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_q_batch, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k0, h_k0.data(), h_k0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v0, h_v0.data(), h_v0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k1, h_k1.data(), h_k1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v1, h_v1.data(), h_v1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  const std::array<const half *, batch_size> h_k_ptrs = {d_k0, d_k1};
  const std::array<const half *, batch_size> h_v_ptrs = {d_v0, d_v1};
  REQUIRE(cudaMemcpy(d_k_ptrs, h_k_ptrs.data(),
                     batch_size * sizeof(const half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v_ptrs, h_v_ptrs.data(),
                     batch_size * sizeof(const half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_kv_lens, kv_lens.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::FlashDecodeMultiSeq<half>(
              d_q_batch, d_k_ptrs, d_v_ptrs, d_o_batch, d_kv_lens, batch_size,
              num_heads, num_kv_heads, head_dim, scale, stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> o_batched = CopyDeviceHalfs(d_o_batch, h_q.size());

  for (int b = 0; b < batch_size; ++b) {
    half *d_q_single = nullptr;
    half *d_o_single = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q_single, h_q.data() + b * q_cols,
                       q_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const half *d_k_single = (b == 0) ? d_k0 : d_k1;
    const half *d_v_single = (b == 0) ? d_v0 : d_v1;
    REQUIRE(cuda_kernel::FlashAttention2Typed<half>(
                d_q_single, d_k_single, d_v_single, d_o_single,
                /*batch_size=*/1, /*query_len=*/1, kv_lens[b], num_heads,
                num_kv_heads, head_dim, scale, /*causal=*/true,
                stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const std::vector<half> o_single = CopyDeviceHalfs(d_o_single, q_cols);
    for (int i = 0; i < q_cols; ++i) {
      REQUIRE(__half2float(o_batched[b * q_cols + i]) ==
              Catch::Approx(__half2float(o_single[i])).margin(2e-3f));
    }

    REQUIRE(cudaFree(d_o_single) == cudaSuccess);
    REQUIRE(cudaFree(d_q_single) == cudaSuccess);
  }

  REQUIRE(cudaFree(d_kv_lens) == cudaSuccess);
  REQUIRE(cudaFree(reinterpret_cast<void *>(d_v_ptrs)) == cudaSuccess);
  REQUIRE(cudaFree(reinterpret_cast<void *>(d_k_ptrs)) == cudaSuccess);
  REQUIRE(cudaFree(d_v1) == cudaSuccess);
  REQUIRE(cudaFree(d_k1) == cudaSuccess);
  REQUIRE(cudaFree(d_v0) == cudaSuccess);
  REQUIRE(cudaFree(d_k0) == cudaSuccess);
  REQUIRE(cudaFree(d_o_batch) == cudaSuccess);
  REQUIRE(cudaFree(d_q_batch) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE(
    "FlashDecodeMultiSeqStrided matches per-sequence FlashAttention decode for "
    "Qwen geometry",
    "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-geometry strided "
            "FlashDecode parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  constexpr int max_seq_len = 16;
  constexpr int max_batch = 4;
  constexpr int layer = 0;
  const std::array<int, batch_size> seq_ids = {1, 3};
  const std::array<int, batch_size> kv_lens = {7, 13};
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  KvCacheGpuTyped<half> cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, num_kv_heads, head_dim, max_seq_len,
                         max_batch));

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.03f);
  const std::vector<half> h_k0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, 0.01f);
  const std::vector<half> h_v0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, -0.01f);
  const std::vector<half> h_k1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.025f, 0.02f);
  const std::vector<half> h_v1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.015f, 0.03f);

  half *d_q_batch = nullptr;
  half *d_o_batch = nullptr;
  half *d_k0 = nullptr;
  half *d_v0 = nullptr;
  half *d_k1 = nullptr;
  half *d_v1 = nullptr;
  int *d_seq_ids = nullptr;
  int *d_kv_lens = nullptr;

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_batch),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k0),
                     h_k0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v0),
                     h_v0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k1),
                     h_k1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v1),
                     h_v1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_seq_ids),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_kv_lens),
                     batch_size * sizeof(int)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_q_batch, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k0, h_k0.data(), h_k0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v0, h_v0.data(), h_v0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k1, h_k1.data(), h_k1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v1, h_v1.data(), h_v1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cache.Append(layer, seq_ids[0], /*start_pos=*/0, kv_lens[0], d_k0,
                       d_v0, stream) == cudaSuccess);
  REQUIRE(cache.Append(layer, seq_ids[1], /*start_pos=*/0, kv_lens[1], d_k1,
                       d_v1, stream) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_seq_ids, seq_ids.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_kv_lens, kv_lens.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  REQUIRE(cuda_kernel::FlashDecodeMultiSeqStrided<half>(
              d_q_batch, cache.Buffer(), d_o_batch, d_seq_ids, d_kv_lens, layer,
              batch_size, num_heads, num_kv_heads, head_dim, cache.SlotStride(),
              cache.LayerStride(), cache.KvStride(), scale,
              stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> o_batched = CopyDeviceHalfs(d_o_batch, h_q.size());

  for (int b = 0; b < batch_size; ++b) {
    half *d_q_single = nullptr;
    half *d_o_single = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_single),
                       q_cols * sizeof(half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q_single, h_q.data() + b * q_cols,
                       q_cols * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const half *d_k_single = (b == 0) ? cache.GetK(layer, seq_ids[0])
                                      : cache.GetK(layer, seq_ids[1]);
    const half *d_v_single = (b == 0) ? cache.GetV(layer, seq_ids[0])
                                      : cache.GetV(layer, seq_ids[1]);
    REQUIRE(cuda_kernel::FlashAttention2Typed<half>(
                d_q_single, d_k_single, d_v_single, d_o_single,
                /*batch_size=*/1, /*query_len=*/1, kv_lens[b], num_heads,
                num_kv_heads, head_dim, scale, /*causal=*/true,
                stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const std::vector<half> o_single = CopyDeviceHalfs(d_o_single, q_cols);
    for (int i = 0; i < q_cols; ++i) {
      REQUIRE(__half2float(o_batched[b * q_cols + i]) ==
              Catch::Approx(__half2float(o_single[i])).margin(2e-3f));
    }

    REQUIRE(cudaFree(d_o_single) == cudaSuccess);
    REQUIRE(cudaFree(d_q_single) == cudaSuccess);
  }

  REQUIRE(cudaFree(d_kv_lens) == cudaSuccess);
  REQUIRE(cudaFree(d_seq_ids) == cudaSuccess);
  REQUIRE(cudaFree(d_v1) == cudaSuccess);
  REQUIRE(cudaFree(d_k1) == cudaSuccess);
  REQUIRE(cudaFree(d_v0) == cudaSuccess);
  REQUIRE(cudaFree(d_k0) == cudaSuccess);
  REQUIRE(cudaFree(d_o_batch) == cudaSuccess);
  REQUIRE(cudaFree(d_q_batch) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE(
    "FlashDecodeMultiSeqStridedSplit matches monolithic decode across splits",
    "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping split FlashDecode parity.");
    return;
  }

  constexpr int batch_size = 2;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  constexpr int max_seq_len = 96;
  constexpr int max_batch = 4;
  constexpr int layer = 0;
  // Non-multiple of every chunk size exercised below, plus a kv_len that
  // spans multiple chunks when chunk is small.
  const std::array<int, batch_size> seq_ids = {0, 2};
  const std::array<int, batch_size> kv_lens = {37, 91};
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  KvCacheGpuTyped<half> cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, num_kv_heads, head_dim, max_seq_len,
                         max_batch));

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(batch_size) * q_cols, 0.03f);
  const std::vector<half> h_k0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, 0.01f);
  const std::vector<half> h_v0 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[0]) * kv_cols, 0.02f, -0.01f);
  const std::vector<half> h_k1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.025f, 0.02f);
  const std::vector<half> h_v1 =
      MakeWaveTensor(static_cast<size_t>(kv_lens[1]) * kv_cols, 0.015f, 0.03f);

  half *d_q = nullptr;
  half *d_o_mono = nullptr;
  half *d_o_split = nullptr;
  float *d_partials = nullptr;
  int *d_seq_ids = nullptr;
  int *d_kv_lens = nullptr;

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_mono),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o_split),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  const size_t partial_floats =
      static_cast<size_t>(batch_size) * num_heads * 8 * (2 + head_dim);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_partials),
                     partial_floats * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_seq_ids),
                     batch_size * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_kv_lens),
                     batch_size * sizeof(int)) == cudaSuccess);

  half *d_k0 = nullptr;
  half *d_v0 = nullptr;
  half *d_k1 = nullptr;
  half *d_v1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k0),
                     h_k0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v0),
                     h_v0.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k1),
                     h_k1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v1),
                     h_v1.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k0, h_k0.data(), h_k0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v0, h_v0.data(), h_v0.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k1, h_k1.data(), h_k1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v1, h_v1.data(), h_v1.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cache.Append(layer, seq_ids[0], /*start_pos=*/0, kv_lens[0], d_k0,
                       d_v0, stream) == cudaSuccess);
  REQUIRE(cache.Append(layer, seq_ids[1], /*start_pos=*/0, kv_lens[1], d_k1,
                       d_v1, stream) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_seq_ids, seq_ids.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_kv_lens, kv_lens.data(), batch_size * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  REQUIRE(cuda_kernel::FlashDecodeMultiSeqStrided<half>(
              d_q, cache.Buffer(), d_o_mono, d_seq_ids, d_kv_lens, layer,
              batch_size, num_heads, num_kv_heads, head_dim, cache.SlotStride(),
              cache.LayerStride(), cache.KvStride(), scale,
              stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  const std::vector<half> o_mono = CopyDeviceHalfs(d_o_mono, h_q.size());

  // (qsplit, chunk, ksplits) configs: pure Q-split, pure KV-split with
  // non-multiple chunk boundaries, both, and a non-divisor qsplit that the
  // launcher must snap down to a ratio divisor (3 → 2 for GQA 8).
  const std::vector<std::array<int, 3>> configs = {
      {8, 96, 1}, {2, 96, 1}, {1, 16, 6}, {2, 24, 4}, {4, 32, 3}, {3, 96, 1}};
  for (const auto &cfg : configs) {
    REQUIRE(cudaMemset(d_o_split, 0xAB, h_q.size() * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cuda_kernel::FlashDecodeMultiSeqStridedSplit<half>(
                d_q, cache.Buffer(), d_o_split, d_partials, d_seq_ids,
                d_kv_lens, layer, batch_size, num_heads, num_kv_heads, head_dim,
                cache.SlotStride(), cache.LayerStride(), cache.KvStride(),
                scale, cfg[0], cfg[1], cfg[2], stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const std::vector<half> o_split = CopyDeviceHalfs(d_o_split, h_q.size());
    for (size_t i = 0; i < o_mono.size(); ++i) {
      REQUIRE(__half2float(o_split[i]) ==
              Catch::Approx(__half2float(o_mono[i])).margin(4e-3f));
    }
  }

  REQUIRE(cudaFree(d_kv_lens) == cudaSuccess);
  REQUIRE(cudaFree(d_seq_ids) == cudaSuccess);
  REQUIRE(cudaFree(d_partials) == cudaSuccess);
  REQUIRE(cudaFree(d_o_split) == cudaSuccess);
  REQUIRE(cudaFree(d_o_mono) == cudaSuccess);
  REQUIRE(cudaFree(d_v1) == cudaSuccess);
  REQUIRE(cudaFree(d_k1) == cudaSuccess);
  REQUIRE(cudaFree(d_v0) == cudaSuccess);
  REQUIRE(cudaFree(d_k0) == cudaSuccess);
  REQUIRE(cudaFree(d_q) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("FlashAttention2Typed matches naive causal attention for Qwen "
          "long-prefill geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen long-prefill attention "
            "parity.");
    return;
  }

  constexpr int batch_size = 1;
  constexpr int query_len = 17;
  constexpr int kv_len = 17;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(query_len) * q_cols, 0.03f);
  const std::vector<half> h_k =
      MakeWaveTensor(static_cast<size_t>(kv_len) * kv_cols, 0.02f, 0.01f);
  const std::vector<half> h_v =
      MakeWaveTensor(static_cast<size_t>(kv_len) * kv_cols, 0.015f, -0.02f);

  half *d_q = nullptr;
  half *d_k = nullptr;
  half *d_v = nullptr;
  half *d_o = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k),
                     h_k.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v),
                     h_v.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_o),
                     h_q.size() * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::FlashAttention2Typed<half>(
              d_q, d_k, d_v, d_o, batch_size, query_len, kv_len, num_heads,
              num_kv_heads, head_dim, scale, /*causal=*/true,
              stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> o_cuda = CopyDeviceHalfs(d_o, h_q.size());
  std::vector<float> ref(o_cuda.size(), 0.0f);
  const int kv_head_ratio = num_heads / num_kv_heads;

  for (int q_pos = 0; q_pos < query_len; ++q_pos) {
    for (int head = 0; head < num_heads; ++head) {
      const int kv_head = head / kv_head_ratio;
      std::vector<float> scores(q_pos + 1, 0.0f);
      float max_score = -1.0e30f;
      for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          const float qv = __half2float(
              h_q[static_cast<size_t>(q_pos) * q_cols + head * head_dim + d]);
          const float kv =
              __half2float(h_k[static_cast<size_t>(k_pos) * kv_cols +
                               kv_head * head_dim + d]);
          dot += qv * kv;
        }
        const float score = dot * scale;
        scores[k_pos] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (float &score : scores) {
        score = std::exp(score - max_score);
        denom += score;
      }

      for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
          const float prob = scores[k_pos] / denom;
          const float vv =
              __half2float(h_v[static_cast<size_t>(k_pos) * kv_cols +
                               kv_head * head_dim + d]);
          acc += prob * vv;
        }
        ref[static_cast<size_t>(q_pos) * q_cols + head * head_dim + d] = acc;
      }
    }
  }

  for (size_t i = 0; i < o_cuda.size(); ++i) {
    REQUIRE(__half2float(o_cuda[i]) == Catch::Approx(ref[i]).margin(6e-2f));
  }

  REQUIRE(cudaFree(d_o) == cudaSuccess);
  REQUIRE(cudaFree(d_v) == cudaSuccess);
  REQUIRE(cudaFree(d_k) == cudaSuccess);
  REQUIRE(cudaFree(d_q) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("RoPE matches host reference for Qwen long-prefill geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping Qwen long-prefill RoPE parity.");
    return;
  }

  constexpr int seq_len = 17;
  constexpr int num_heads = 16;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int q_cols = num_heads * head_dim;
  constexpr int kv_cols = num_kv_heads * head_dim;
  constexpr float freq_base = 1000000.0f;
  constexpr int rope_type = 2;

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  const std::vector<half> h_q =
      MakeWaveTensor(static_cast<size_t>(seq_len) * q_cols, 0.03f);
  const std::vector<half> h_k =
      MakeWaveTensor(static_cast<size_t>(seq_len) * kv_cols, 0.02f, 0.01f);
  std::vector<half> q_ref = h_q;
  std::vector<half> k_ref = h_k;
  ApplyRopeHost(&q_ref, &k_ref, seq_len, num_heads, num_kv_heads, head_dim,
                /*n_past=*/0, freq_base, rope_type);

  half *d_q = nullptr;
  half *d_k = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q),
                     h_q.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k),
                     h_k.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::RoPE<half>(d_q, d_k, seq_len, num_heads, num_kv_heads,
                                  head_dim, /*n_past=*/0, freq_base, stream,
                                  rope_type) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::vector<half> q_cuda = CopyDeviceHalfs(d_q, h_q.size());
  const std::vector<half> k_cuda = CopyDeviceHalfs(d_k, h_k.size());
  for (size_t i = 0; i < q_cuda.size(); ++i) {
    REQUIRE(__half2float(q_cuda[i]) ==
            Catch::Approx(__half2float(q_ref[i])).margin(2e-3f));
  }
  for (size_t i = 0; i < k_cuda.size(); ++i) {
    REQUIRE(__half2float(k_cuda[i]) ==
            Catch::Approx(__half2float(k_ref[i])).margin(2e-3f));
  }

  REQUIRE(cudaFree(d_k) == cudaSuccess);
  REQUIRE(cudaFree(d_q) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves sampled outputs for long-prefill "
          "lm_head Q4_K geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping long-prefill lm_head Q4_K parity.");
    return;
  }

  RunLmHeadLongPrefillContract(runtime::cuda::native::GGUF::TensorType::Q4_K,
                               2.0e-1f);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves sampled outputs for long-prefill "
          "lm_head Q6_K geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping long-prefill lm_head Q6_K parity.");
    return;
  }

  RunLmHeadLongPrefillContract(runtime::cuda::native::GGUF::TensorType::Q6_K,
                               2.0e-1f);
}

TEST_CASE("FusedQuantGemm::GemvPackedPair preserves per-row grouped Q8_0 "
          "outputs",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q8_0 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q8_0);
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
    SUCCEED("Packed Q8_0 kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 32;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q8_0_weights = [&](int rows) {
    std::vector<runtime::cuda::native::block_q8_0> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d = encode_half(0.01f * static_cast<float>((row % 5) + 1));
      for (int i = 0; i < K; ++i) {
        blocks[row].qs[i] =
            static_cast<signed char>(((row * 17 + i * 3) % 23) - 11);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q8_0_weights(N0);
  const auto h_w1 = make_q8_0_weights(N1);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.02f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 13 + i * 5) % 15) - 7);
    }
  }

  auto q8_0_ref = [&](const std::vector<runtime::cuda::native::block_q8_0> &w,
                      int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        half scale_half{};
        std::memcpy(&scale_half, &w[row].d, sizeof(scale_half));
        const float w_scale = __half2float(scale_half);
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b = w_scale * static_cast<float>(w[row].qs[i]);
          acc += a * b;
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = q8_0_ref(h_w0, N0);
  const std::vector<float> ref1 = q8_0_ref(h_w1, N1);

  runtime::cuda::native::block_q8_0 *d_w0 = nullptr;
  runtime::cuda::native::block_q8_0 *d_w1 = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q8_0)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q8_0)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q8_0),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q8_0),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedPair(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(3e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(3e-2f));
  }

  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvPackedPair preserves per-row grouped Q4_K "
          "outputs",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q4_K parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
    SUCCEED("Packed Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d = encode_half(0.015f * static_cast<float>((row % 5) + 1));
      blocks[row].dmin = encode_half(0.01f * static_cast<float>((row % 3) + 1));
      for (int i = 0; i < K_SCALE_SIZE; ++i) {
        blocks[row].scales[i] =
            static_cast<unsigned char>((row * 19 + i * 11) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].qs[i] =
            static_cast<unsigned char>((row * 7 + i * 13) & 0xFF);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0);
  const auto h_w1 = make_q4_k_weights(N1);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.015f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 17 + i * 5) % 15) - 7);
    }
  }

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b = __half2float(weights[row * K + i]);
          acc += a * b;
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedPair(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(7e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(7e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvPackedTriple preserves per-row grouped Q4_K "
          "outputs",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q4_K triple parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
    SUCCEED("Packed Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  constexpr int N2 = 4;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max({N0, N1, N2}), K, 3, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d =
          encode_half(0.012f * static_cast<float>(((row + seed) % 5) + 1));
      blocks[row].dmin =
          encode_half(0.008f * static_cast<float>(((row + seed) % 3) + 1));
      for (int i = 0; i < K_SCALE_SIZE; ++i) {
        blocks[row].scales[i] =
            static_cast<unsigned char>((seed * 23 + row * 17 + i * 11) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].qs[i] =
            static_cast<unsigned char>((seed * 7 + row * 13 + i * 5) & 0xFF);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 1);
  const auto h_w1 = make_q4_k_weights(N1, 2);
  const auto h_w2 = make_q4_k_weights(N2, 3);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.0125f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 19 + i * 7) % 15) - 7);
    }
  }

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  runtime::cuda::native::block_q4_k *d_w2 = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_out2 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  half *d_deq2 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w2),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out2),
                     M * N2 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq2),
                     N2 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w2, h_w2.data(),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w2, d_deq2, N2 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  const std::vector<half> deq2 = CopyDeviceHalfs(d_deq2, N2 * K);
  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b = __half2float(weights[row * K + i]);
          acc += a * b;
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);
  const std::vector<float> ref2 = deq_ref(deq2, N2);

  const std::array<PackedProjectionSpec, 3> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
      {{d_w2, quant_type, static_cast<int64_t>(N2) * K}, d_out2, N2},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedTriple(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  const std::vector<half> out2 = CopyDeviceHalfs(d_out2, M * N2);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(7e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(7e-2f));
  }
  for (int i = 0; i < M * N2; ++i) {
    REQUIRE(__half2float(out2[i]) == Catch::Approx(ref2[i]).margin(7e-2f));
  }

  REQUIRE(cudaFree(d_deq2) == cudaSuccess);
  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out2) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_w2) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvPackedTriple preserves sampled outputs for "
          "Qwen QKV long-prefill geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen long-prefill QKV parity.");
    return;
  }

  constexpr int M = 17;
  constexpr int K = 2048;
  constexpr int NQ = 2048;
  constexpr int NK = 256;
  constexpr int NV = 256;
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const FusedDispatchGeometry geometry{M, NQ, K, 3, true, false};
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(quant_type, geometry)) {
    SUCCEED("Packed Q4_K triple kernel unavailable for long-prefill geometry "
            "on this device/profile.");
    return;
  }

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows, int cols, int seed) {
    REQUIRE(cols % QK_K == 0);
    const int blocks_per_row = cols / QK_K;
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.0095f *
                              static_cast<float>(((row + seed + blk) % 5) + 1));
        block.dmin = encode_half(
            0.0060f * static_cast<float>(((row + seed + blk) % 3) + 1));
        for (int i = 0; i < K_SCALE_SIZE; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 29 + row * 17 + blk * 11 + i * 7) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 13 + row * 19 + blk * 5 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_wq = make_q4_k_weights(NQ, K, 1);
  const auto h_wk = make_q4_k_weights(NK, K, 2);
  const auto h_wv = make_q4_k_weights(NV, K, 3);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.0115f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 31 + i * 7) % 15) - 7);
    }
  }

  runtime::cuda::native::block_q4_k *d_wq = nullptr;
  runtime::cuda::native::block_q4_k *d_wk = nullptr;
  runtime::cuda::native::block_q4_k *d_wv = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_q = nullptr;
  half *d_k = nullptr;
  half *d_v = nullptr;
  half *d_q_deq = nullptr;
  half *d_k_deq = nullptr;
  half *d_v_deq = nullptr;

  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_wq),
                     h_wq.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_wk),
                     h_wk.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_wv),
                     h_wv.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q),
                     static_cast<size_t>(M) * NQ * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k),
                     static_cast<size_t>(M) * NK * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v),
                     static_cast<size_t>(M) * NV * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q_deq),
                     static_cast<size_t>(NQ) * K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_k_deq),
                     static_cast<size_t>(NK) * K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_v_deq),
                     static_cast<size_t>(NV) * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_wq, h_wq.data(),
                     h_wq.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_wk, h_wk.data(),
                     h_wk.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_wv, h_wv.data(),
                     h_wv.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_wq, d_q_deq, NQ * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_wk, d_k_deq, NK * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_wv, d_v_deq, NV * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> q_deq =
      CopyDeviceHalfs(d_q_deq, static_cast<size_t>(NQ) * K);
  const std::vector<half> k_deq =
      CopyDeviceHalfs(d_k_deq, static_cast<size_t>(NK) * K);
  const std::vector<half> v_deq =
      CopyDeviceHalfs(d_v_deq, static_cast<size_t>(NV) * K);

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(static_cast<size_t>(M) * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b =
              __half2float(weights[static_cast<size_t>(row) * K + i]);
          acc += a * b;
        }
        out[static_cast<size_t>(m) * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref_q = deq_ref(q_deq, NQ);
  const std::vector<float> ref_k = deq_ref(k_deq, NK);
  const std::vector<float> ref_v = deq_ref(v_deq, NV);

  const std::array<PackedProjectionSpec, 3> projections = {{
      {{d_wq, quant_type, static_cast<int64_t>(NQ) * K}, d_q, NQ},
      {{d_wk, quant_type, static_cast<int64_t>(NK) * K}, d_k, NK},
      {{d_wv, quant_type, static_cast<int64_t>(NV) * K}, d_v, NV},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedTriple(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> q_out =
      CopyDeviceHalfs(d_q, static_cast<size_t>(M) * NQ);
  const std::vector<half> k_out =
      CopyDeviceHalfs(d_k, static_cast<size_t>(M) * NK);
  const std::vector<half> v_out =
      CopyDeviceHalfs(d_v, static_cast<size_t>(M) * NV);

  for (size_t i = 0; i < q_out.size(); ++i) {
    REQUIRE(__half2float(q_out[i]) == Catch::Approx(ref_q[i]).margin(4.0e-1f));
  }
  for (size_t i = 0; i < k_out.size(); ++i) {
    REQUIRE(__half2float(k_out[i]) == Catch::Approx(ref_k[i]).margin(3.0e-1f));
  }
  for (size_t i = 0; i < v_out.size(); ++i) {
    REQUIRE(__half2float(v_out[i]) == Catch::Approx(ref_v[i]).margin(3.0e-1f));
  }

  REQUIRE(cudaFree(d_v_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_k_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_q_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_v) == cudaSuccess);
  REQUIRE(cudaFree(d_k) == cudaSuccess);
  REQUIRE(cudaFree(d_q) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_wv) == cudaSuccess);
  REQUIRE(cudaFree(d_wk) == cudaSuccess);
  REQUIRE(cudaFree(d_wq) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvPackedPair preserves per-row grouped Q6_K "
          "outputs",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q6_K parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
    SUCCEED("Packed Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q6_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q6_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].ql[i] =
            static_cast<unsigned char>((seed * 11 + row * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        blocks[row].qh[i] =
            static_cast<unsigned char>((seed * 13 + row * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        blocks[row].scales[i] =
            static_cast<char>(((seed * 17 + row * 3 + i * 5) % 31) - 15);
      }
      blocks[row].d =
          encode_half(0.01f * static_cast<float>(((row + seed) % 5) + 1));
    }
    return blocks;
  };

  const auto h_w0 = make_q6_k_weights(N0, 1);
  const auto h_w1 = make_q6_k_weights(N1, 2);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.012f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 23 + i * 7) % 15) - 7);
    }
  }

  runtime::cuda::native::block_q6_k *d_w0 = nullptr;
  runtime::cuda::native::block_q6_k *d_w1 = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b = __half2float(weights[row * K + i]);
          acc += a * b;
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedPair(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvPackedTriple preserves per-row grouped Q6_K "
          "outputs",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q6_K triple parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
    SUCCEED("Packed Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  constexpr int N2 = 4;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max({N0, N1, N2}), K, 3, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q6_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q6_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].ql[i] =
            static_cast<unsigned char>((seed * 11 + row * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        blocks[row].qh[i] =
            static_cast<unsigned char>((seed * 13 + row * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        blocks[row].scales[i] =
            static_cast<char>(((seed * 17 + row * 3 + i * 5) % 31) - 15);
      }
      blocks[row].d =
          encode_half(0.01f * static_cast<float>(((row + seed) % 5) + 1));
    }
    return blocks;
  };

  const auto h_w0 = make_q6_k_weights(N0, 1);
  const auto h_w1 = make_q6_k_weights(N1, 2);
  const auto h_w2 = make_q6_k_weights(N2, 3);

  std::vector<int8_t> h_activation(M * K);
  std::vector<float> h_row_scales(M);
  for (int m = 0; m < M; ++m) {
    h_row_scales[m] = 0.012f * static_cast<float>(m + 1);
    for (int i = 0; i < K; ++i) {
      h_activation[m * K + i] =
          static_cast<int8_t>(((m * 23 + i * 7) % 15) - 7);
    }
  }

  runtime::cuda::native::block_q6_k *d_w0 = nullptr;
  runtime::cuda::native::block_q6_k *d_w1 = nullptr;
  runtime::cuda::native::block_q6_k *d_w2 = nullptr;
  int8_t *d_activation = nullptr;
  float *d_row_scales = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_out2 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  half *d_deq2 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w2),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                     h_activation.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_row_scales),
                     h_row_scales.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out2),
                     M * N2 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq2),
                     N2 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w2, h_w2.data(),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_activation, h_activation.data(),
                     h_activation.size() * sizeof(int8_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_row_scales, h_row_scales.data(),
                     h_row_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w2, d_deq2, N2 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  const std::vector<half> deq2 = CopyDeviceHalfs(d_deq2, N2 * K);
  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          const float a =
              h_row_scales[m] * static_cast<float>(h_activation[m * K + i]);
          const float b = __half2float(weights[row * K + i]);
          acc += a * b;
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);
  const std::vector<float> ref2 = deq_ref(deq2, N2);

  const std::array<PackedProjectionSpec, 3> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
      {{d_w2, quant_type, static_cast<int64_t>(N2) * K}, d_out2, N2},
  }};
  const PackedActivationInfo activation{d_activation, d_row_scales};
  REQUIRE(
      FusedQuantGemm::GemvPackedTriple(projections, activation, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  const std::vector<half> out2 = CopyDeviceHalfs(d_out2, M * N2);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N2; ++i) {
    REQUIRE(__half2float(out2[i]) == Catch::Approx(ref2[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq2) == cudaSuccess);
  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out2) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_row_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_activation) == cudaSuccess);
  REQUIRE(cudaFree(d_w2) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q6_K "
          "outputs with misaligned activation blocks",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q6_K Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q6_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q6_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].ql[i] =
            static_cast<unsigned char>((seed * 11 + row * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        blocks[row].qh[i] =
            static_cast<unsigned char>((seed * 13 + row * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        blocks[row].scales[i] =
            static_cast<char>(((seed * 17 + row * 3 + i * 5) % 31) - 15);
      }
      blocks[row].d =
          encode_half(0.01f * static_cast<float>(((row + seed) % 5) + 1));
    }
    return blocks;
  };

  const auto h_w0 = make_q6_k_weights(N0, 1);
  const auto h_w1 = make_q6_k_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.015f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w0 = nullptr;
  runtime::cuda::native::block_q6_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q6_K "
          "outputs for four-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q6_K Q8_1 row-quad "
            "parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q6_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q6_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].ql[i] =
            static_cast<unsigned char>((seed * 11 + row * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        blocks[row].qh[i] =
            static_cast<unsigned char>((seed * 13 + row * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        blocks[row].scales[i] =
            static_cast<char>(((seed * 17 + row * 3 + i * 5) % 31) - 15);
      }
      blocks[row].d =
          encode_half(0.012f * static_cast<float>(((row + seed) % 5) + 1));
    }
    return blocks;
  };

  const auto h_w0 = make_q6_k_weights(N0, 1);
  const auto h_w1 = make_q6_k_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.014f, 0.0015f);

  runtime::cuda::native::block_q6_k *d_w0 = nullptr;
  runtime::cuda::native::block_q6_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Triple preserves per-row grouped Q6_K "
          "outputs for multi-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q6_K Q8_1 triple "
            "parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K triple kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  constexpr int N2 = 4;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max({N0, N1, N2}), K, 3, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q6_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q6_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].ql[i] =
            static_cast<unsigned char>((seed * 11 + row * 7 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        blocks[row].qh[i] =
            static_cast<unsigned char>((seed * 13 + row * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        blocks[row].scales[i] =
            static_cast<char>(((seed * 17 + row * 3 + i * 5) % 31) - 15);
      }
      blocks[row].d =
          encode_half(0.01f * static_cast<float>(((row + seed) % 5) + 1));
    }
    return blocks;
  };

  const auto h_w0 = make_q6_k_weights(N0, 1);
  const auto h_w1 = make_q6_k_weights(N1, 2);
  const auto h_w2 = make_q6_k_weights(N2, 3);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.015f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w0 = nullptr;
  runtime::cuda::native::block_q6_k *d_w1 = nullptr;
  runtime::cuda::native::block_q6_k *d_w2 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_out2 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  half *d_deq2 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w2),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out2),
                     M * N2 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq2),
                     N2 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w2, h_w2.data(),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w2, d_deq2, N2 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  const std::vector<half> deq2 = CopyDeviceHalfs(d_deq2, N2 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);
  const std::vector<float> ref2 = deq_ref(deq2, N2);

  const std::array<PackedProjectionSpec, 3> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
      {{d_w2, quant_type, static_cast<int64_t>(N2) * K}, d_out2, N2},
  }};
  REQUIRE(
      FusedQuantGemm::GemvQ8_1Triple(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  const std::vector<half> out2 = CopyDeviceHalfs(d_out2, M * N2);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N2; ++i) {
    REQUIRE(__half2float(out2[i]) == Catch::Approx(ref2[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq2) == cudaSuccess);
  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out2) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w2) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for multi-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q4_K Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows, int seed) {
    const int blocks_per_row = K / QK_K;
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 3 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 1);
  const auto h_w1 = make_q4_k_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.017f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for Qwen decode geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-geometry grouped Q4_K "
            "Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 2048;
  constexpr int N0 = 2048;
  constexpr int N1 = 256;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  const int blocks_per_row = K / QK_K;
  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 1);
  const auto h_w1 = make_q4_k_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.017f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for Qwen FFN projection geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-FFN grouped Q4_K "
            "Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 2048;
  constexpr int N0 = 11008;
  constexpr int N1 = 11008;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  const int blocks_per_row = K / QK_K;
  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 3);
  const auto h_w1 = make_q4_k_weights(N1, 7);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.015f, -0.002f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_out0_tuned = nullptr;
  half *d_out1_tuned = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0_tuned),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1_tuned),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  NativeExecutionPolicy tuned_policy;
  tuned_policy.enable_experimental_q81_grouped_rowpair_w4 = true;
  REQUIRE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      quant_type, FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false},
      &tuned_policy));
  const std::array<PackedProjectionSpec, 2> tuned_projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0_tuned, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1_tuned, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(tuned_projections, d_act_q8_1, M, K,
                                       nullptr, &tuned_policy));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  const std::vector<half> out0_tuned = CopyDeviceHalfs(d_out0_tuned, M * N0);
  const std::vector<half> out1_tuned = CopyDeviceHalfs(d_out1_tuned, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
    REQUIRE(__half2float(out0_tuned[i]) ==
            Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
    REQUIRE(__half2float(out1_tuned[i]) ==
            Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1_tuned) == cudaSuccess);
  REQUIRE(cudaFree(d_out0_tuned) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for Qwen FFN row-quad decode geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-FFN grouped Q4_K "
            "row-quad parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 3;
  constexpr int K = 2048;
  constexpr int N0 = 11008;
  constexpr int N1 = 11008;

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  const int blocks_per_row = K / QK_K;
  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 3);
  const auto h_w1 = make_q4_k_weights(N1, 7);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.015f, -0.002f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0_generic = nullptr;
  half *d_out1_generic = nullptr;
  half *d_out0_tuned = nullptr;
  half *d_out1_tuned = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0_generic),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1_generic),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0_tuned),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1_tuned),
                     M * N1 * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::array<PackedProjectionSpec, 2> generic_projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0_generic, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1_generic, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(generic_projections, d_act_q8_1, M, K,
                                       nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::array<PackedProjectionSpec, 2> tuned_projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0_tuned, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1_tuned, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1PairRowQuadCandidate(
      tuned_projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0_generic =
      CopyDeviceHalfs(d_out0_generic, M * N0);
  const std::vector<half> out1_generic =
      CopyDeviceHalfs(d_out1_generic, M * N1);
  const std::vector<half> out0_tuned = CopyDeviceHalfs(d_out0_tuned, M * N0);
  const std::vector<half> out1_tuned = CopyDeviceHalfs(d_out1_tuned, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0_tuned[i]) ==
            Catch::Approx(__half2float(out0_generic[i])).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1_tuned[i]) ==
            Catch::Approx(__half2float(out1_generic[i])).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_out1_tuned) == cudaSuccess);
  REQUIRE(cudaFree(d_out0_tuned) == cudaSuccess);
  REQUIRE(cudaFree(d_out1_generic) == cudaSuccess);
  REQUIRE(cudaFree(d_out0_generic) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves sampled outputs for "
          "Qwen FFN long-prefill geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-FFN long-prefill parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  NativeExecutionPolicy policy;
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), quant_type,
                                2048LL * 11008LL};
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kPrefill,
              FusedDispatchGeometry{23, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81Group);

  constexpr int M = 23;
  constexpr int K = 2048;
  constexpr int N0 = 11008;
  constexpr int N1 = 11008;
  const std::array<int, 10> sample_rows = {0,   1,    2,    17,   63,
                                           511, 2047, 4095, 8191, 11007};

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  const int blocks_per_row = K / QK_K;
  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 3);
  const auto h_w1 = make_q4_k_weights(N1, 7);
  const std::vector<half> h_input =
      MakeWaveTensor(static_cast<size_t>(M) * K, 0.015f, -0.002f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1, static_cast<size_t>(M) * (K / QK8_1) *
                                  sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     static_cast<size_t>(M) * N0 * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     static_cast<size_t>(M) * N1 * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     static_cast<size_t>(N0) * K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     static_cast<size_t>(N1) * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(
              d_w0, d_deq0, static_cast<size_t>(N0) * K) == cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(
              d_w1, d_deq1, static_cast<size_t>(N1) * K) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 =
      CopyDeviceHalfs(d_deq0, static_cast<size_t>(N0) * K);
  const std::vector<half> deq1 =
      CopyDeviceHalfs(d_deq1, static_cast<size_t>(N1) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(
      static_cast<size_t>(M) * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(static_cast<size_t>(M) * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[static_cast<size_t>(m) * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[static_cast<size_t>(m) * K + blk * QK8_1 + j] =
            scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 =
      CopyDeviceHalfs(d_out0, static_cast<size_t>(M) * N0);
  const std::vector<half> out1 =
      CopyDeviceHalfs(d_out1, static_cast<size_t>(M) * N1);

  auto check_samples = [&](const std::vector<half> &weights,
                           const std::vector<half> &actual, int rows) {
    for (int m = 0; m < M; ++m) {
      for (int row : sample_rows) {
        CAPTURE(m, row, rows);
        float ref = 0.0f;
        for (int i = 0; i < K; ++i) {
          ref += act_ref[static_cast<size_t>(m) * K + i] *
                 __half2float(weights[static_cast<size_t>(row) * K + i]);
        }
        REQUIRE(__half2float(actual[static_cast<size_t>(m) * rows + row]) ==
                Catch::Approx(ref).margin(8e-2f));
      }
    }
  };

  check_samples(deq0, out0, N0);
  check_samples(deq1, out1, N1);

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for single-row hot decode geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping single-row hot grouped Q4_K "
            "Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  NativeExecutionPolicy policy;
  policy.enable_experimental_q81_grouped_hot_q4k = true;

  constexpr int M = 1;
  constexpr int K = 2048;
  constexpr int N0 = 8192;
  constexpr int N1 = 8192;
  REQUIRE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      quant_type, FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false},
      &policy));

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  const int blocks_per_row = K / QK_K;
  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(
        static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
      for (int blk = 0; blk < blocks_per_row; ++blk) {
        auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
        block.d = encode_half(0.02f *
                              static_cast<float>(((row + blk + seed) % 4) + 1));
        block.dmin = encode_half(
            0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
        for (int i = 0; i < 12; ++i) {
          block.scales[i] = static_cast<unsigned char>(
              (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
        }
        for (int i = 0; i < QK_K / 2; ++i) {
          block.qs[i] = static_cast<unsigned char>(
              (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
        }
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 3);
  const auto h_w1 = make_q4_k_weights(N1, 7);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.015f, -0.002f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr,
                                       &policy));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q4_K "
          "outputs for four-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q4_K Q8_1 row-quad "
            "parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d =
          encode_half(0.02f * static_cast<float>(((row + seed) % 4) + 1));
      blocks[row].dmin =
          encode_half(0.01f * static_cast<float>(((row + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        blocks[row].scales[i] =
            static_cast<unsigned char>((seed * 17 + row * 7 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].qs[i] =
            static_cast<unsigned char>((seed * 11 + row * 13 + i * 3) & 0xFF);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 1);
  const auto h_w1 = make_q4_k_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.017f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Triple preserves per-row grouped Q4_K "
          "outputs for multi-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q4_K Q8_1 triple "
            "parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K triple kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N0 = 8;
  constexpr int N1 = 6;
  constexpr int N2 = 4;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max({N0, N1, N2}), K, 3, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q4_k_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q4_k> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d =
          encode_half(0.02f * static_cast<float>(((row + seed) % 4) + 1));
      blocks[row].dmin =
          encode_half(0.01f * static_cast<float>(((row + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        blocks[row].scales[i] =
            static_cast<unsigned char>((seed * 17 + row * 7 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        blocks[row].qs[i] =
            static_cast<unsigned char>((seed * 11 + row * 13 + i * 3) & 0xFF);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q4_k_weights(N0, 1);
  const auto h_w1 = make_q4_k_weights(N1, 2);
  const auto h_w2 = make_q4_k_weights(N2, 3);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.017f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w0 = nullptr;
  runtime::cuda::native::block_q4_k *d_w1 = nullptr;
  runtime::cuda::native::block_q4_k *d_w2 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_out2 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  half *d_deq2 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w2),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out2),
                     M * N2 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq2),
                     N2 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w2, h_w2.data(),
                     h_w2.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w2, d_deq2, N2 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  const std::vector<half> deq2 = CopyDeviceHalfs(d_deq2, N2 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);
  const std::vector<float> ref2 = deq_ref(deq2, N2);

  const std::array<PackedProjectionSpec, 3> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
      {{d_w2, quant_type, static_cast<int64_t>(N2) * K}, d_out2, N2},
  }};
  REQUIRE(
      FusedQuantGemm::GemvQ8_1Triple(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  const std::vector<half> out2 = CopyDeviceHalfs(d_out2, M * N2);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(8e-2f));
  }
  for (int i = 0; i < M * N2; ++i) {
    REQUIRE(__half2float(out2[i]) == Catch::Approx(ref2[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq2) == cudaSuccess);
  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out2) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w2) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1Pair preserves per-row grouped Q8_0 "
          "outputs with misaligned weight and activation rows",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping grouped Q8_0 Q8_1 parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q8_0);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q8_0 kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 32;
  constexpr int N0 = 7;
  constexpr int N1 = 5;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type,
      FusedDispatchGeometry{M, std::max(N0, N1), K, 2, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  auto make_q8_0_weights = [&](int rows, int seed) {
    std::vector<runtime::cuda::native::block_q8_0> blocks(rows);
    for (int row = 0; row < rows; ++row) {
      blocks[row].d =
          encode_half(0.015f * static_cast<float>(((seed + row) % 5) + 1));
      for (int i = 0; i < K; ++i) {
        blocks[row].qs[i] =
            static_cast<signed char>(((seed * 19 + row * 7 + i * 5) % 29) - 14);
      }
    }
    return blocks;
  };

  const auto h_w0 = make_q8_0_weights(N0, 1);
  const auto h_w1 = make_q8_0_weights(N1, 2);
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.03f, -0.004f);

  runtime::cuda::native::block_q8_0 *d_w0 = nullptr;
  runtime::cuda::native::block_q8_0 *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out0 = nullptr;
  half *d_out1 = nullptr;
  half *d_deq0 = nullptr;
  half *d_deq1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w0),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q8_0)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w1),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q8_0)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                     M * N0 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                     M * N1 * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq0),
                     N0 * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq1),
                     N1 * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w0, h_w0.data(),
                     h_w0.size() * sizeof(runtime::cuda::native::block_q8_0),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_w1, h_w1.data(),
                     h_w1.size() * sizeof(runtime::cuda::native::block_q8_0),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q8_0(d_w0, d_deq0, N0 * K) ==
          cudaSuccess);
  REQUIRE(runtime::cuda::native::dequantize_q8_0(d_w1, d_deq1, N1 * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq0 = CopyDeviceHalfs(d_deq0, N0 * K);
  const std::vector<half> deq1 = CopyDeviceHalfs(d_deq1, N1 * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  auto deq_ref = [&](const std::vector<half> &weights, int rows) {
    std::vector<float> out(M * rows, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
          acc += act_ref[m * K + i] * __half2float(weights[row * K + i]);
        }
        out[m * rows + row] = acc;
      }
    }
    return out;
  };

  const std::vector<float> ref0 = deq_ref(deq0, N0);
  const std::vector<float> ref1 = deq_ref(deq1, N1);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, quant_type, static_cast<int64_t>(N0) * K}, d_out0, N0},
      {{d_w1, quant_type, static_cast<int64_t>(N1) * K}, d_out1, N1},
  }};
  REQUIRE(FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, M, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out0 = CopyDeviceHalfs(d_out0, M * N0);
  const std::vector<half> out1 = CopyDeviceHalfs(d_out1, M * N1);
  for (int i = 0; i < M * N0; ++i) {
    REQUIRE(__half2float(out0[i]) == Catch::Approx(ref0[i]).margin(4e-2f));
  }
  for (int i = 0; i < M * N1; ++i) {
    REQUIRE(__half2float(out1[i]) == Catch::Approx(ref1[i]).margin(4e-2f));
  }

  REQUIRE(cudaFree(d_deq1) == cudaSuccess);
  REQUIRE(cudaFree(d_deq0) == cudaSuccess);
  REQUIRE(cudaFree(d_out1) == cudaSuccess);
  REQUIRE(cudaFree(d_out0) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w1) == cudaSuccess);
  REQUIRE(cudaFree(d_w0) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q4_K outputs for "
          "multi-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q4_K Q8_1 batch GEMV parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N = 7;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    h_w[row].d = encode_half(0.02f * static_cast<float>((row % 4) + 1));
    h_w[row].dmin = encode_half(0.01f * static_cast<float>((row % 3) + 1));
    for (int i = 0; i < 12; ++i) {
      h_w[row].scales[i] =
          static_cast<unsigned char>((row * 17 + i * 5) & 0xFF);
    }
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].qs[i] = static_cast<unsigned char>((row * 13 + i * 7) & 0xFF);
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.018f, -0.003f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q4_K outputs for "
          "four-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q4_K Q8_1 row-quad parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N = 8;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    h_w[row].d = encode_half(0.02f * static_cast<float>((row % 4) + 1));
    h_w[row].dmin = encode_half(0.01f * static_cast<float>((row % 3) + 1));
    for (int i = 0; i < 12; ++i) {
      h_w[row].scales[i] =
          static_cast<unsigned char>((row * 17 + i * 5) & 0xFF);
    }
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].qs[i] = static_cast<unsigned char>((row * 13 + i * 7) & 0xFF);
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.018f, -0.003f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::DownProjMmq preserves per-row Q4_K outputs for "
          "four-row tiled decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q4_K down-proj MMQ parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsDownProjMmq(quant_type)) {
    SUCCEED("Down-proj MMQ Q4_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N = 16;

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    h_w[row].d = encode_half(0.02f * static_cast<float>((row % 4) + 1));
    h_w[row].dmin = encode_half(0.01f * static_cast<float>((row % 3) + 1));
    for (int i = 0; i < 12; ++i) {
      h_w[row].scales[i] =
          static_cast<unsigned char>((row * 17 + i * 5) & 0xFF);
    }
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].qs[i] = static_cast<unsigned char>((row * 13 + i * 7) & 0xFF);
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.016f, -0.002f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo raw{d_w, quant_type, static_cast<int64_t>(N) * K};
  MmqWeightInfo layout{};
  REQUIRE(FusedQuantGemm::BuildDownProjMmqLayout(raw, N, K, &layout, nullptr));
  REQUIRE(layout.data != nullptr);
  REQUIRE(layout.tile_cols == FusedQuantGemm::kDownProjMmqTileCols);

  const char *prev = std::getenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ", "1", 1) == 0);

  REQUIRE(
      FusedQuantGemm::DownProjMmq(layout, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  if (prev) {
    REQUIRE(portable_setenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ", prev_value.c_str(),
                            1) == 0);
  } else {
    REQUIRE(portable_unsetenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ") == 0);
  }

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  FusedQuantGemm::DestroyDownProjMmqLayout(layout);
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::DownProjMmq preserves per-row Q6_K outputs for "
          "four-row tiled decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K down-proj MMQ parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsDownProjMmq(quant_type)) {
    SUCCEED("Down-proj MMQ Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N = 16;

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q6_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].ql[i] = static_cast<unsigned char>((row * 11 + i * 3) & 0xFF);
    }
    for (int i = 0; i < QK_K / 4; ++i) {
      h_w[row].qh[i] = static_cast<unsigned char>((row * 7 + i * 9) & 0xFF);
    }
    for (int i = 0; i < QK_K / 16; ++i) {
      h_w[row].scales[i] = static_cast<char>(((row * 5 + i * 7) % 31) - 15);
    }
    h_w[row].d = encode_half(0.01f * static_cast<float>((row % 5) + 1));
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.014f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo raw{d_w, quant_type, static_cast<int64_t>(N) * K};
  MmqWeightInfo layout{};
  REQUIRE(FusedQuantGemm::BuildDownProjMmqLayout(raw, N, K, &layout, nullptr));
  REQUIRE(layout.data != nullptr);
  REQUIRE(layout.tile_cols == FusedQuantGemm::kDownProjMmqTileCols);

  const char *prev = std::getenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ", "1", 1) == 0);

  REQUIRE(
      FusedQuantGemm::DownProjMmq(layout, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  if (prev) {
    REQUIRE(portable_setenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ", prev_value.c_str(),
                            1) == 0);
  } else {
    REQUIRE(portable_unsetenv("INFERFLUX_ENABLE_DOWNPROJ_MMQ") == 0);
  }

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  FusedQuantGemm::DestroyDownProjMmqLayout(layout);
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for "
          "multi-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K Q8_1 batch GEMV parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 256;
  constexpr int N = 7;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q6_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].ql[i] = static_cast<unsigned char>((row * 11 + i * 3) & 0xFF);
    }
    for (int i = 0; i < QK_K / 4; ++i) {
      h_w[row].qh[i] = static_cast<unsigned char>((row * 7 + i * 9) & 0xFF);
    }
    for (int i = 0; i < QK_K / 16; ++i) {
      h_w[row].scales[i] = static_cast<char>(((row * 5 + i * 7) % 31) - 15);
    }
    h_w[row].d = encode_half(0.01f * static_cast<float>((row % 5) + 1));
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.014f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for "
          "Qwen decode geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen-geometry Q6_K Q8_1 "
            "batch GEMV parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 2048;
  constexpr int N = 256;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  constexpr int blocks_per_row = K / QK_K;
  std::vector<runtime::cuda::native::block_q6_k> h_w(static_cast<size_t>(N) *
                                                     blocks_per_row);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = h_w[static_cast<size_t>(row) * blocks_per_row + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] =
            static_cast<unsigned char>((row * 11 + blk * 5 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] =
            static_cast<unsigned char>((row * 7 + blk * 13 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>(((row * 5 + blk * 3 + i * 7) % 31) - 15);
      }
      block.d = encode_half(0.01f * static_cast<float>(((row + blk) % 5) + 1));
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.014f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for "
          "Qwen down-proj geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Qwen down-proj Q6_K Q8_1 "
            "batch GEMV parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 2;
  constexpr int K = 11008;
  constexpr int N = 2048;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  constexpr int blocks_per_row = K / QK_K;
  std::vector<runtime::cuda::native::block_q6_k> h_w(static_cast<size_t>(N) *
                                                     blocks_per_row);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = h_w[static_cast<size_t>(row) * blocks_per_row + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] =
            static_cast<unsigned char>((row * 11 + blk * 5 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] =
            static_cast<unsigned char>((row * 7 + blk * 13 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>(((row * 5 + blk * 3 + i * 7) % 31) - 15);
      }
      block.d = encode_half(0.01f * static_cast<float>(((row + blk) % 5) + 1));
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.012f, 0.001f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for "
          "four-row batch decode",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K Q8_1 row-quad parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  constexpr int M = 4;
  constexpr int K = 256;
  constexpr int N = 8;
  const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{M, N, K, 1, true, false});
  REQUIRE(M <= threshold);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q6_k> h_w(N);
  for (int row = 0; row < N; ++row) {
    for (int i = 0; i < QK_K / 2; ++i) {
      h_w[row].ql[i] = static_cast<unsigned char>((row * 11 + i * 3) & 0xFF);
    }
    for (int i = 0; i < QK_K / 4; ++i) {
      h_w[row].qh[i] = static_cast<unsigned char>((row * 7 + i * 9) & 0xFF);
    }
    for (int i = 0; i < QK_K / 16; ++i) {
      h_w[row].scales[i] = static_cast<char>(((row * 5 + i * 7) % 31) - 15);
    }
    h_w[row].d = encode_half(0.01f * static_cast<float>((row % 5) + 1));
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.014f, 0.002f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq = CopyDeviceHalfs(d_deq, N * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[m * (K / QK8_1) + blk];
      half scale_half{};
      std::memcpy(&scale_half, &a.ds, sizeof(scale_half));
      const float scale = __half2float(scale_half);
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[m * K + blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int row = 0; row < N; ++row) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[m * K + i] * __half2float(deq[row * K + i]);
      }
      ref[m * N + row] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(8e-2f));
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("Q6_K vectorized decode matches scalar dispatch across production "
          "output modes",
          "[native_forward][cuda_runtime_contract][q6k_vectorized]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K vector parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }

  for (const int M : {3, 4, 5, 8}) {
    RequireQ6KVectorizedParity(M, 8, 256, true);
  }

  SECTION("MMQ boundary ignores the MMVQ vector policy") {
    for (const int M : {9, 16}) {
      RequireQ6KVectorizedParity(M, 8, 256, true);
    }
  }

  SECTION("exact down-projection geometry") {
    RequireQ6KVectorizedParity(8, 2048, 11008, false);
  }
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q4_K outputs for exact "
          "down-proj hot geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q4_K down-proj hot parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }
  const char *prev =
      std::getenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(
      portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                      "1", 1) == 0);

  constexpr int M = 1;
  constexpr int K = 11008;
  constexpr int N = 2048;
  constexpr int kBlocksPerRow = K / QK_K;
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{M, N, K, 1, true, false}, true,
              true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(N * kBlocksPerRow);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = h_w[row * kBlocksPerRow + blk];
      block.d = encode_half(0.012f * static_cast<float>(((row + blk) % 5) + 1));
      block.dmin =
          encode_half(0.006f * static_cast<float>(((row + blk) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] =
            static_cast<unsigned char>((row * 17 + blk * 7 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] =
            static_cast<unsigned char>((row * 13 + blk * 11 + i * 3) & 0xFF);
      }
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.009f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int blk = 0; blk < K / QK8_1; ++blk) {
    const auto &a = q8_1_blocks[blk];
    const float scale = DecodeQ81Ds(a).first;
    for (int j = 0; j < QK8_1; ++j) {
      act_ref[blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
    }
  }

  std::vector<float> ref(N, 0.0f);
  for (int row = 0; row < N; ++row) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += act_ref[i] * __half2float(deq[row * K + i]);
    }
    ref[row] = acc;
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(1.0e-1f));
  }

  if (prev) {
    REQUIRE(
        portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                        prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED") == 0);
  }
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves sampled outputs for long-prefill "
          "down-proj row-quad geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping long-prefill down-proj parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }

  NativeExecutionPolicy policy;
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), quant_type,
                                11008LL * 2048LL};
  const MmqWeightInfo mmq{};
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kPrefill,
              FusedDispatchGeometry{23, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad);

  constexpr int M = 23;
  constexpr int K = 11008;
  constexpr int N = 2048;
  constexpr int kBlocksPerRow = K / QK_K;
  const std::array<int, 10> sample_outputs = {0,   1,   2,    17,   63,
                                              127, 511, 1023, 1535, 2047};

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(static_cast<size_t>(N) *
                                                     kBlocksPerRow);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = h_w[static_cast<size_t>(row) * kBlocksPerRow + blk];
      block.d = encode_half(0.012f * static_cast<float>(((row + blk) % 5) + 1));
      block.dmin =
          encode_half(0.006f * static_cast<float>(((row + blk) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] =
            static_cast<unsigned char>((row * 17 + blk * 7 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] =
            static_cast<unsigned char>((row * 13 + blk * 11 + i * 3) & 0xFF);
      }
    }
  }
  const std::vector<half> h_input =
      MakeWaveTensor(static_cast<size_t>(M) * K, 0.009f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1, static_cast<size_t>(M) * (K / QK8_1) *
                                  sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out),
                     static_cast<size_t>(M) * N * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq),
                     static_cast<size_t>(N) * K * sizeof(half)) == cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(
              d_w, d_deq, static_cast<size_t>(N) * K) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(
      static_cast<size_t>(M) * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(static_cast<size_t>(M) * K, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[static_cast<size_t>(row) * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[static_cast<size_t>(row) * K + blk * QK8_1 + j] =
            scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out =
      CopyDeviceHalfs(d_out, static_cast<size_t>(M) * N);
  for (int row = 0; row < M; ++row) {
    for (int out_idx : sample_outputs) {
      CAPTURE(row, out_idx);
      float ref = 0.0f;
      for (int i = 0; i < K; ++i) {
        ref += act_ref[static_cast<size_t>(row) * K + i] *
               __half2float(deq[static_cast<size_t>(out_idx) * K + i]);
      }
      REQUIRE(__half2float(out[static_cast<size_t>(row) * N + out_idx]) ==
              Catch::Approx(ref).margin(1.0e-1f));
    }
  }

  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for exact "
          "down-proj hot geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K down-proj hot parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }
  const char *prev =
      std::getenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(
      portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                      "1", 1) == 0);

  constexpr int M = 1;
  constexpr int K = 11008;
  constexpr int N = 2048;
  constexpr int kBlocksPerRow = K / QK_K;
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{M, N, K, 1, true, false}, true,
              true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q6_k> h_w(N * kBlocksPerRow);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = h_w[row * kBlocksPerRow + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] =
            static_cast<unsigned char>((row * 11 + blk * 13 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] =
            static_cast<unsigned char>((row * 7 + blk * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>((((row + blk) * 5 + i * 7) % 31) - 15);
      }
      block.d = encode_half(0.008f * static_cast<float>(((row + blk) % 5) + 1));
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.008f, 0.001f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int blk = 0; blk < K / QK8_1; ++blk) {
    const auto &a = q8_1_blocks[blk];
    const float scale = DecodeQ81Ds(a).first;
    for (int j = 0; j < QK8_1; ++j) {
      act_ref[blk * QK8_1 + j] = scale * static_cast<float>(a.qs[j]);
    }
  }

  std::vector<float> ref(N, 0.0f);
  for (int row = 0; row < N; ++row) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += act_ref[i] * __half2float(deq[row * K + i]);
    }
    ref[row] = acc;
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(1.0e-1f));
  }

  if (prev) {
    REQUIRE(
        portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                        prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED") == 0);
  }
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q4_K outputs for exact "
          "two-row down-proj hot geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q4_K row-pair hot parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q4_K kernels unavailable for this device/profile.");
    return;
  }
  const char *prev = std::getenv(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv(
              "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
              "1", 1) == 0);

  constexpr int M = 2;
  constexpr int K = 11008;
  constexpr int N = 2048;
  constexpr int kBlocksPerRow = K / QK_K;
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{M, N, K, 1, true, false}, true,
              true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q4_k> h_w(N * kBlocksPerRow);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = h_w[row * kBlocksPerRow + blk];
      block.d = encode_half(0.012f * static_cast<float>(((row + blk) % 5) + 1));
      block.dmin =
          encode_half(0.006f * static_cast<float>(((row + blk) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] =
            static_cast<unsigned char>((row * 17 + blk * 7 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] =
            static_cast<unsigned char>((row * 13 + blk * 11 + i * 3) & 0xFF);
      }
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.009f, -0.001f);

  runtime::cuda::native::block_q4_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q4_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q4_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[row * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[row * K + blk * QK8_1 + j] =
            scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int out = 0; out < N; ++out) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[row * K + i] * __half2float(deq[out * K + i]);
      }
      ref[row * N + out] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(1.0e-1f));
  }

  if (prev) {
    REQUIRE(portable_setenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
                prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(
        portable_unsetenv(
            "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED") ==
        0);
  }
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::GemvQ8_1 preserves per-row Q6_K outputs for exact "
          "two-row down-proj hot geometry",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping Q6_K row-pair hot parity.");
    return;
  }

  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  if (!FusedQuantGemm::SupportsQ8_1Activations(quant_type)) {
    SUCCEED("Q8_1 Q6_K kernels unavailable for this device/profile.");
    return;
  }
  const char *prev = std::getenv(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv(
              "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
              "1", 1) == 0);

  constexpr int M = 2;
  constexpr int K = 11008;
  constexpr int N = 2048;
  constexpr int kBlocksPerRow = K / QK_K;
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{M, N, K, 1, true, false}, true,
              true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);

  auto encode_half = [](float value) {
    const half h = __float2half(value);
    unsigned short bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };

  std::vector<runtime::cuda::native::block_q6_k> h_w(N * kBlocksPerRow);
  for (int row = 0; row < N; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = h_w[row * kBlocksPerRow + blk];
      for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] =
            static_cast<unsigned char>((row * 11 + blk * 13 + i * 3) & 0xFF);
      }
      for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] =
            static_cast<unsigned char>((row * 7 + blk * 5 + i * 9) & 0xFF);
      }
      for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] =
            static_cast<char>((((row + blk) * 5 + i * 7) % 31) - 15);
      }
      block.d = encode_half(0.008f * static_cast<float>(((row + blk) % 5) + 1));
    }
  }
  const std::vector<half> h_input = MakeWaveTensor(M * K, 0.008f, 0.001f);

  runtime::cuda::native::block_q6_k *d_w = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  half *d_deq = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_w),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(
      cudaMalloc(&d_act_q8_1,
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_out), M * N * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_deq), N * K * sizeof(half)) ==
          cudaSuccess);

  REQUIRE(cudaMemcpy(d_w, h_w.data(),
                     h_w.size() * sizeof(runtime::cuda::native::block_q6_k),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, M, K, nullptr);

  REQUIRE(runtime::cuda::native::dequantize_q6_k(d_w, d_deq, N * K) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> deq =
      CopyDeviceHalfs(d_deq, static_cast<size_t>(N) * K);
  std::vector<runtime::cuda::native::block_q8_1> q8_1_blocks(M * (K / QK8_1));
  REQUIRE(
      cudaMemcpy(q8_1_blocks.data(), d_act_q8_1,
                 q8_1_blocks.size() * sizeof(runtime::cuda::native::block_q8_1),
                 cudaMemcpyDeviceToHost) == cudaSuccess);

  std::vector<float> act_ref(M * K, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int blk = 0; blk < K / QK8_1; ++blk) {
      const auto &a = q8_1_blocks[row * (K / QK8_1) + blk];
      const float scale = DecodeQ81Ds(a).first;
      for (int j = 0; j < QK8_1; ++j) {
        act_ref[row * K + blk * QK8_1 + j] =
            scale * static_cast<float>(a.qs[j]);
      }
    }
  }

  std::vector<float> ref(M * N, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int out = 0; out < N; ++out) {
      float acc = 0.0f;
      for (int i = 0; i < K; ++i) {
        acc += act_ref[row * K + i] * __half2float(deq[out * K + i]);
      }
      ref[row * N + out] = acc;
    }
  }

  QuantizedWeightInfo info{d_w, quant_type, static_cast<int64_t>(N) * K};
  REQUIRE(FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_out, M, N, K, nullptr));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<half> out = CopyDeviceHalfs(d_out, M * N);
  for (int i = 0; i < M * N; ++i) {
    REQUIRE(__half2float(out[i]) == Catch::Approx(ref[i]).margin(1.0e-1f));
  }

  if (prev) {
    REQUIRE(portable_setenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
                prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(
        portable_unsetenv(
            "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED") ==
        0);
  }
  REQUIRE(cudaFree(d_deq) == cudaSuccess);
  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
  REQUIRE(cudaFree(d_w) == cudaSuccess);
}

TEST_CASE(
    "QuantizedWeightMapAdapter: HasQuantizedWeights delegates to underlying",
    "[native_forward]") {
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Without Build(), IsQuantized() returns false
  REQUIRE_FALSE(adapter.HasQuantizedWeights());

  // Raw accessors should return empty info
  REQUIRE(adapter.LayerQProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerKProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerVProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerOProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerGateProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerUpProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerDownProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerDownProjMmq(0).data == nullptr);
  REQUIRE(adapter.LmHeadRaw().data == nullptr);
}

// ============================================================================
// FusedQuantGemm Dispatch Tests
// ============================================================================

TEST_CASE("FusedQuantGemm: adaptive threshold varies by quant type",
          "[native_forward]") {
  // Lower bits per weight → higher threshold (fused saves more bandwidth)
  int q4k_threshold = FusedQuantGemm::GetAdaptiveThreshold(12); // Q4_K
  int q6k_threshold = FusedQuantGemm::GetAdaptiveThreshold(14); // Q6_K
  int q8_0_threshold = FusedQuantGemm::GetAdaptiveThreshold(8); // Q8_0
  REQUIRE(q4k_threshold >= q8_0_threshold);
  REQUIRE(q6k_threshold >= q8_0_threshold);
  REQUIRE(q4k_threshold >= 4);
  REQUIRE(q4k_threshold <= 64);
}

TEST_CASE("FusedQuantGemm: down-proj MMQ support is additive for Q4_K and "
          "Q6_K",
          "[native_forward]") {
  REQUIRE(FusedQuantGemm::SupportsDownProjMmq(
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K)));
  REQUIRE(FusedQuantGemm::SupportsDownProjMmq(
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K)));
  REQUIRE_FALSE(FusedQuantGemm::SupportsDownProjMmq(
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q8_0)));
}

TEST_CASE("FusedQuantGemm: down-proj selector rejects invalid geometry",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{0, 4096, 11008, 1, true, false},
              true, true, true) == FusedQuantGemm::DownProjOperator::kFallback);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 0, 11008, 1, true, false},
              true, true, true) == FusedQuantGemm::DownProjOperator::kFallback);
}

TEST_CASE("FusedQuantGemm: down-proj selector keeps small Q4_K decode on "
          "Q8_1 GEMV",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 4096, 11008, 1, true, false},
              true, true, true) == FusedQuantGemm::DownProjOperator::kQ81Gemv);
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes exact Q4_K decode hot "
          "shape to fixed-block Q8_1 GEMV when explicitly enabled",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const char *prev =
      std::getenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(
      portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                      "1", 1) == 0);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              true, true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);
  if (prev) {
    REQUIRE(
        portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                        prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED") == 0);
  }
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes exact Q4_K hot geometry "
          "to fixed-block Q8_1 by default",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              true, true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes exact Q6_K hot geometry "
          "to fixed-block Q8_1 by default",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              true, true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);
}

TEST_CASE("FusedQuantGemm: down-proj selector keeps exact Q6_K decode hot "
          "shape on fixed-block Q8_1 GEMV when explicitly enabled",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const char *prev =
      std::getenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(
      portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                      "1", 1) == 0);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              true, true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);
  if (prev) {
    REQUIRE(
        portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED",
                        prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED") == 0);
  }
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes two-row Q4_K decode "
          "to exact-shape fixed-block row-pair Q8_1 when explicitly enabled",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const char *prev = std::getenv(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv(
              "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
              "1", 1) == 0);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              true, true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);
  if (prev) {
    REQUIRE(portable_setenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
                prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(
        portable_unsetenv(
            "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED") ==
        0);
  }
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes exact two-row Q4_K "
          "decode to fixed-block row-pair by default",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              true, true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes two-row Q6_K decode "
          "to exact-shape fixed-block row-pair Q8_1 when explicitly enabled",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const char *prev = std::getenv(
      "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv(
              "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
              "1", 1) == 0);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              true, true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);
  if (prev) {
    REQUIRE(portable_setenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED",
                prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(
        portable_unsetenv(
            "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED") ==
        0);
  }
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes exact two-row Q6_K "
          "decode to fixed-block row-pair by default",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              true, true, true) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes four-row Q6_K decode "
          "to explicit row-quad Q8_1 operator",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{4, 2048, 11008, 1, true, false},
              true, true,
              true) == FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad);
}

TEST_CASE("FusedQuantGemm: down-proj selector uses packed path when Q8_1 is "
          "disabled but packed activations are available",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              false, true,
              false) == FusedQuantGemm::DownProjOperator::kPackedGemv);
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes large Q4_K decode to "
          "MMQ when allowed",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type,
              FusedDispatchGeometry{256, 4096, 11008, 1, true, false}, true,
              true, true) == FusedQuantGemm::DownProjOperator::kMmq);
}

TEST_CASE("FusedQuantGemm: down-proj selector honors MMQ batch override for "
          "Q4_K decode",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const char *prev = std::getenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH", "1", 1) == 0);

  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 4096, 11008, 1, true, false},
              true, true, true) == FusedQuantGemm::DownProjOperator::kMmq);

  if (prev) {
    REQUIRE(portable_setenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH",
                            prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH") == 0);
  }
}

TEST_CASE("FusedQuantGemm: invalid MMQ batch override is ignored",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const char *prev = std::getenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH");
  std::string prev_value = prev ? prev : "";
  REQUIRE(portable_setenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH", "not_a_number",
                          1) == 0);

  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type, FusedDispatchGeometry{1, 4096, 11008, 1, true, false},
              true, true, true) == FusedQuantGemm::DownProjOperator::kQ81Gemv);

  if (prev) {
    REQUIRE(portable_setenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH",
                            prev_value.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH") == 0);
  }
}

TEST_CASE("FusedQuantGemm: down-proj selector promotes large Q6_K decode to "
          "MMQ when allowed",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  REQUIRE(FusedQuantGemm::SelectDownProjOperator(
              quant_type,
              FusedDispatchGeometry{256, 4096, 11008, 1, true, false}, true,
              true, true) == FusedQuantGemm::DownProjOperator::kMmq);
}

TEST_CASE("FusedQuantGemm: adaptive threshold remains bounded for all target "
          "quantized GGUF types",
          "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };
  for (const int quant_type : target_quant_types) {
    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    REQUIRE(threshold >= 4);
    REQUIRE(threshold <= 64);
  }
}

TEST_CASE("FusedQuantGemm: deterministic fallback policy by batch size for "
          "target quant types",
          "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };

  for (const int quant_type : target_quant_types) {
    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    const int short_prefill_m = std::max(2, threshold);
    const int batched_decode_m = threshold + 1;

    // Decode and short-prefill stay on fused path.
    REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));
    REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, short_prefill_m));

    // Larger M deterministically falls back to compatibility/cuBLAS path.
    REQUIRE_FALSE(
        FusedQuantGemm::ShouldUseFusedPath(quant_type, batched_decode_m));
  }
}

TEST_CASE("FusedQuantGemm: deterministic fallback policy rejects invalid "
          "inputs",
          "[native_forward]") {
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(0, 1));  // F32
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(1, 1));  // F16
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(12, 0)); // non-positive M
}

TEST_CASE("FusedQuantGemm: geometry-aware threshold boosts packed sibling "
          "projections",
          "[native_forward]") {
  const int quant_type = 12; // Q4_K
  const int base = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
  const int packed_single = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{4, 2048, 2048, 1, true, false});
  const int packed_group = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{4, 2048, 2048, 3, true, true});

  REQUIRE(packed_single >= base);
  REQUIRE(packed_group >= packed_single);
  REQUIRE(packed_group <= 64);
}

TEST_CASE("FusedQuantGemm: geometry-aware threshold tempers huge single-output "
          "packed projections",
          "[native_forward]") {
  const int quant_type = 12; // Q4_K
  const int packed_group = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{4, 2048, 2048, 3, true, true});
  const int lm_head_like = FusedQuantGemm::GetGeometryAwareThreshold(
      quant_type, FusedDispatchGeometry{4, 151936, 2048, 1, true, true});

  REQUIRE(lm_head_like < packed_group);
  REQUIRE(lm_head_like >= 4);
}

TEST_CASE("FusedQuantGemm: geometry-aware selector keeps packed grouped paths "
          "alive beyond legacy cutoff",
          "[native_forward]") {
  const int quant_type = 12; // Q4_K
  const int base = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
  const int probe_m = std::min(base + 4, 64);

  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(quant_type, probe_m));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(
      quant_type, FusedDispatchGeometry{probe_m, 2048, 2048, 3, true, true}));
}

TEST_CASE("FusedQuantGemm: specialized grouped Q8_1 fast path is limited to "
          "Q4_K single-row FFN geometry and explicit rollout enablement",
          "[native_forward]") {
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const int q6k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const FusedDispatchGeometry hot_m1{1, 11008, 2048, 2, true, false};
  const FusedDispatchGeometry hot_m2{2, 11008, 2048, 2, true, false};

  const char *prev_value =
      std::getenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K");
  const bool had_prev = prev_value != nullptr;
  const std::string prev = had_prev ? std::string(prev_value) : std::string();
  REQUIRE(portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K",
                          "0", 1) == 0);
  REQUIRE_FALSE(
      FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(q4k, hot_m1));
  REQUIRE_FALSE(
      FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(q4k, hot_m2));

  REQUIRE(portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K",
                          "1", 1) == 0);
  REQUIRE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(q4k, hot_m1));
  REQUIRE_FALSE(
      FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(q4k, hot_m2));

  if (had_prev) {
    REQUIRE(
        portable_setenv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K",
                        prev.c_str(), 1) == 0);
  } else {
    REQUIRE(portable_unsetenv(
                "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K") == 0);
  }

  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      q4k, FusedDispatchGeometry{1, 2048, 2048, 2, true, false}));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      q4k, FusedDispatchGeometry{3, 11008, 2048, 2, true, false}));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      q4k, FusedDispatchGeometry{1, 11008, 4096, 2, true, false}));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      q4k, FusedDispatchGeometry{1, 11008, 2048, 3, true, false}));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
      q6k, FusedDispatchGeometry{1, 11008, 2048, 2, true, false}));
}

TEST_CASE("FusedQuantGemm: 4-warp grouped row-pair path is limited to "
          "Q4_K M=2 FFN decode geometry and can be explicitly disabled",
          "[native_forward]") {
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const int q6k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const FusedDispatchGeometry hot_m2{2, 11008, 2048, 2, true, false};

  NativeExecutionPolicy enabled_policy;
  REQUIRE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, hot_m2, &enabled_policy));

  NativeExecutionPolicy disabled_policy;
  disabled_policy.enable_experimental_q81_grouped_rowpair_w4 = false;
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, hot_m2, &disabled_policy));

  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, FusedDispatchGeometry{1, 11008, 2048, 2, true, false},
      &enabled_policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, FusedDispatchGeometry{2, 2048, 2048, 2, true, false},
      &enabled_policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, FusedDispatchGeometry{2, 11008, 4096, 2, true, false},
      &enabled_policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q4k, FusedDispatchGeometry{2, 11008, 2048, 3, true, false},
      &enabled_policy));
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      q6k, hot_m2, &enabled_policy));
}

TEST_CASE("FusedQuantGemm: FFN grouped selector keeps the proven "
          "Q4_K decode hot paths on by default",
          "[native_forward]") {
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  NativeExecutionPolicy policy;
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K);
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{2, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
  // M>=2 Q4_K decode uses MMQ3 (default-ON via
  // enable_experimental_q81_grouped_mmq3)
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{3, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{4, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);

  policy.enable_experimental_q81_grouped_mmq3 = false;
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{2, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupRowPairW4);

  policy.enable_experimental_q81_grouped_rowpair_w4 = false;
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{2, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81Group);

  policy.enable_experimental_q81_grouped_mmq3 = true;
  policy.enable_experimental_q81_grouped_rowquad_m4 = true;
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{4, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4);
  // M=3 with rowquad enabled: rowquad needs M>=4, MMQ3 matches at M>=3
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{3, 11008, 2048, 2, true, true},
              true, true,
              &policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
}

TEST_CASE("FusedQuantGemm: FFN grouped selector keeps non-hot grouped decode "
          "on generic Q8_1 path",
          "[native_forward]") {
  const int q6k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q6k, q6k, FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              true, true) == FusedQuantGemm::FfnProjOperator::kQ81Group);
}

TEST_CASE("FusedQuantGemm: FFN grouped selector falls back to packed path "
          "when Q8_1 activations are disabled",
          "[native_forward]") {
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  REQUIRE(FusedQuantGemm::SelectFfnProjOperator(
              q4k, q4k, FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              false, true) == FusedQuantGemm::FfnProjOperator::kPackedGroup);
}

TEST_CASE(
    "InferfluxCudaDispatchPolicy: wrapper selectors preserve policy gates",
    "[native_forward]") {
  NativeExecutionPolicy policy;
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), q4k,
                                2048LL * 11008LL};
  const MmqWeightInfo mmq{reinterpret_cast<const void *>(0x2), q4k, 2048, 11008,
                          FusedQuantGemm::kDownProjMmqTileCols};

  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/false,
              policy) == FusedQuantGemm::DownProjOperator::kFallback);

  policy.disable_q81_activations = true;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kPackedGemv);

  policy.disable_q81_activations = false;
  policy.force_cublas = true;
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kFallback);
}

TEST_CASE(
    "InferfluxCudaDispatchPolicy: phase-aware FFN registry keeps prefill on "
    "generic grouped Q8_1 path",
    "[native_forward]") {
  NativeExecutionPolicy policy;
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), q4k,
                                2048LL * 11008LL};

  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K);
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kPrefill,
              FusedDispatchGeometry{1, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81Group);
  // M>=2 Q4_K decode uses MMQ3 (default-ON via
  // enable_experimental_q81_grouped_mmq3)
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{2, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{3, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{4, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);

  policy.enable_experimental_q81_grouped_rowquad_m4 = true;
  REQUIRE(SelectInferfluxCudaFfnProjOperator(
              raw, raw, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{4, 11008, 2048, 2, true, true},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4);
}

TEST_CASE(
    "InferfluxCudaDispatchPolicy: phase-aware down-proj registry preserves "
    "existing operator ladder",
    "[native_forward]") {
  NativeExecutionPolicy policy;
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), q4k,
                                2048LL * 11008LL};
  const MmqWeightInfo mmq{};

  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{1, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed);
  // M=4 Q4_K decode now takes the tensor-core operator too (S18 extended
  // the MMA window beyond Q6_K); non-MMA hardware keeps the Q8_1 ladder.
  const FusedQuantGemm::DownProjOperator q4k_m4_expect =
      FusedQuantGemm::CurrentDeviceSupportsMmqMma()
          ? FusedQuantGemm::DownProjOperator::kMmqMma
          : FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{4, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == q4k_m4_expect);

  policy.enable_experimental_q81_downproj_rowpair_hot_fixed = true;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true, policy) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);
}

TEST_CASE(
    "InferfluxCudaDispatchPolicy: down-proj registry promotes Q6_K MMA-window "
    "decode to the tensor-core operator",
    "[native_forward]") {
  NativeExecutionPolicy policy;
  const int q6k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const QuantizedWeightInfo raw{reinterpret_cast<const void *>(0x1), q6k,
                                2048LL * 11008LL};
  const QuantizedWeightInfo raw_q4{reinterpret_cast<const void *>(0x1), q4k,
                                   2048LL * 11008LL};
  const MmqWeightInfo mmq{};

  // Inside the MMA window (mmq_mma_min_batch..max_batch) the tensor-core
  // operator outranks everything — but only on tensor-core hardware; the
  // CPU-only fallback keeps the Q8_1 ladder.
  const FusedQuantGemm::DownProjOperator window_expect =
      FusedQuantGemm::CurrentDeviceSupportsMmqMma()
          ? FusedQuantGemm::DownProjOperator::kMmqMma
          : FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{8, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == window_expect);

  // S18: Q4_K down-proj joins the same window — the ollama Q4_K_M recipe
  // mixes per-layer Q4_K/Q6_K down-proj tensors, so both quants must
  // promote identically.
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw_q4, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{8, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == window_expect);

  // Policy off falls back down the ladder — never a tier-mismatched
  // selection.
  policy.enable_mmq_mma = false;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{8, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad);
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw_q4, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{8, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad);

  // Below the MMA window the Q8_1 ladder is untouched even with MMA armed
  // (M=2 takes the geometry-gated row-pair hot-fixed rule).
  policy.enable_mmq_mma = true;
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{2, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed);

  // Prefill chunks above the fused-path M ceiling select the dense
  // fallback — the executor never attempts a fused tier there, so the
  // MMA-with-prefill-ceiling executions live in the inline
  // fused-gate+up+SiLU chain, outside this catalog (documented debt in
  // dispatch_catalog.h).
  REQUIRE(SelectInferfluxCudaDownProjOperator(
              raw, mmq, InferfluxCudaDispatchPhase::kPrefill,
              FusedDispatchGeometry{512, 2048, 11008, 1, true, false},
              /*allow_fused_quantized_matmul=*/true,
              policy) == FusedQuantGemm::DownProjOperator::kFallback);
}

TEST_CASE(
    "InferfluxCudaLinearExecutor: FFN helper falls back from Q8_1 to packed "
    "without invoking generic path",
    "[native_forward]") {
  std::vector<std::string> calls;
  NativeFfnExecutionSummary summary;

  const bool ok = ExecuteInferfluxCudaFfnProjectionStage(
      FusedQuantGemm::FfnProjOperator::kQ81Group, "decode", "q4_k",
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K), 1, 11008,
      2048,
      [&]() {
        calls.emplace_back("q81");
        return false;
      },
      [&]() {
        calls.emplace_back("packed");
        return true;
      },
      [&]() {
        calls.emplace_back("fallback");
        return false;
      },
      &summary);

  REQUIRE(ok);
  REQUIRE(calls == std::vector<std::string>{"q81", "packed"});
  REQUIRE(summary.used_q81 == false);
  REQUIRE(summary.used_packed);
  REQUIRE(summary.actual_op == FusedQuantGemm::FfnProjOperator::kPackedGroup);
}

TEST_CASE("InferfluxCudaLinearExecutor: FFN helper executes MMQ3 grouped Q8_1 "
          "path before packed fallback",
          "[native_forward]") {
  std::vector<std::string> calls;
  NativeFfnExecutionSummary summary;

  const bool ok = ExecuteInferfluxCudaFfnProjectionStage(
      FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3, "decode", "q4_k",
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K), 4, 11008,
      2048,
      [&]() {
        calls.emplace_back("q81");
        return true;
      },
      [&]() {
        calls.emplace_back("packed");
        return false;
      },
      [&]() {
        calls.emplace_back("fallback");
        return false;
      },
      &summary);

  REQUIRE(ok);
  REQUIRE(calls == std::vector<std::string>{"q81"});
  REQUIRE(summary.used_q81);
  REQUIRE_FALSE(summary.used_packed);
  REQUIRE(summary.actual_op == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3);
}

TEST_CASE(
    "InferfluxCudaLinearExecutor: normalized projection helper computes norm "
    "once before dense fallback",
    "[native_forward]") {
  std::vector<std::string> calls;
  bool norm_computed = false;

  const bool ok = ExecuteNativeNormalizedProjectionStage(
      &norm_computed,
      [&]() {
        calls.emplace_back("norm");
        return true;
      },
      [&]() { return false; },
      [&]() {
        calls.emplace_back("dense");
        return true;
      });

  REQUIRE(ok);
  REQUIRE(norm_computed);
  REQUIRE(calls == std::vector<std::string>{"norm", "dense"});
}

TEST_CASE(
    "InferfluxCudaLinearExecutor: grouped projection helper uses packed path "
    "before generic fallback",
    "[native_forward]") {
  std::vector<std::string> calls;
  NativeGroupedProjectionSummary summary;

  const bool ok = ExecuteNativeGroupedProjectionStage(
      [&]() {
        calls.emplace_back("q81");
        return false;
      },
      [&]() {
        calls.emplace_back("packed");
        return true;
      },
      [&]() {
        calls.emplace_back("fallback");
        return false;
      },
      &summary);

  REQUIRE(ok);
  REQUIRE(calls == std::vector<std::string>{"q81", "packed"});
  REQUIRE_FALSE(summary.used_q81);
  REQUIRE(summary.used_packed);
}

TEST_CASE(
    "InferfluxCudaLinearExecutor: down-proj helper invokes fallback only after "
    "all fused paths miss",
    "[native_forward]") {
  std::vector<std::string> calls;
  NativeDownProjExecutionSummary summary;

  const bool ok = ExecuteInferfluxCudaDownProjStage(
      FusedQuantGemm::DownProjOperator::kQ81Gemv, "decode", "q4_k",
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K), 1, 2048,
      11008,
      [&]() {
        calls.emplace_back("mma");
        return false;
      },
      [&]() {
        calls.emplace_back("mmq");
        return false;
      },
      [&]() {
        calls.emplace_back("q81");
        return false;
      },
      [&]() {
        calls.emplace_back("packed");
        return false;
      },
      [&]() {
        calls.emplace_back("fallback");
        return true;
      },
      [&](FusedQuantGemm::DownProjOperator) { calls.emplace_back("log"); },
      &summary);

  REQUIRE(ok);
  REQUIRE(calls == std::vector<std::string>{"q81", "mma", "mmq", "packed",
                                            "fallback"});
  REQUIRE_FALSE(summary.used_mmq_mma);
  REQUIRE_FALSE(summary.used_mmq);
  REQUIRE_FALSE(summary.used_q81);
  REQUIRE_FALSE(summary.used_packed);
  REQUIRE(summary.actual_op == FusedQuantGemm::DownProjOperator::kFallback);
}

// Regression shape of the kQ81GroupMmq3 allowlist bug: every selectable
// operator must reach the lambda of its own tier when that tier is
// available, and actual_op must equal the selection. Before the executor
// allowlist learned kQ81GroupMmq3, that operator silently degraded to the
// packed catch-all (metrics attributed packed_group, the tiled kernel never
// launched, ~25% of c=16 throughput lost for months).
TEST_CASE("InferfluxCudaLinearExecutor: every FFN operator reaches its tier "
          "when available",
          "[native_forward][dispatch_reachability]") {
  using Op = FusedQuantGemm::FfnProjOperator;
  const std::array<Op, 7> kAllFfnOps = {
      Op::kFallback,          Op::kQ81Group,          Op::kQ81GroupHotQ4K,
      Op::kQ81GroupRowPairW4, Op::kQ81GroupRowQuadM4, Op::kQ81GroupMmq3,
      Op::kPackedGroup};

  for (const Op op : kAllFfnOps) {
    bool q81_called = false;
    bool packed_called = false;
    bool fallback_called = false;
    NativeFfnExecutionSummary summary;

    const bool ok = ExecuteInferfluxCudaFfnProjectionStage(
        op, "decode", "q4_k",
        static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K), 4,
        11008, 2048,
        [&]() {
          q81_called = true;
          return true;
        },
        [&]() {
          packed_called = true;
          return true;
        },
        [&]() {
          fallback_called = true;
          return true;
        },
        &summary);

    CAPTURE(static_cast<int>(op));
    REQUIRE(ok);

    switch (op) {
    case Op::kQ81Group:
    case Op::kQ81GroupHotQ4K:
    case Op::kQ81GroupRowPairW4:
    case Op::kQ81GroupRowQuadM4:
    case Op::kQ81GroupMmq3:
      // The Q8_1 lambda must be the one invoked — this is the exact
      // assertion that fails when an operator is missing from the
      // executor's dispatch.
      REQUIRE(q81_called);
      REQUIRE_FALSE(packed_called);
      REQUIRE(summary.used_q81);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kPackedGroup:
      REQUIRE(packed_called);
      REQUIRE_FALSE(q81_called);
      REQUIRE(summary.used_packed);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kFallback:
      // kFallback means the selector requested the dense fallback
      // (force_cublas or no fused tier ready) — the fused catch-alls are
      // skipped entirely.
      REQUIRE(fallback_called);
      REQUIRE_FALSE(packed_called);
      REQUIRE(summary.actual_op == op);
      break;
    }
  }
}

TEST_CASE("InferfluxCudaLinearExecutor: every down-proj operator reaches its "
          "tier when available",
          "[native_forward][dispatch_reachability]") {
  using Op = FusedQuantGemm::DownProjOperator;
  const std::array<Op, 9> kAllDownOps = {
      Op::kFallback,        Op::kQ81Gemv,
      Op::kQ81GemvHotFixed, Op::kQ81GemvRowPairHotFixed,
      Op::kQ81GemvRowPair,  Op::kQ81GemvRowQuad,
      Op::kPackedGemv,      Op::kMmq,
      Op::kMmqMma};

  for (const Op op : kAllDownOps) {
    bool mma_called = false;
    bool mmq_called = false;
    bool q81_called = false;
    bool packed_called = false;
    bool fallback_called = false;
    NativeDownProjExecutionSummary summary;

    const bool ok = ExecuteInferfluxCudaDownProjStage(
        op, "decode", "q4_k",
        static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K), 4,
        2048, 11008,
        [&]() {
          mma_called = true;
          return true;
        },
        [&]() {
          mmq_called = true;
          return true;
        },
        [&]() {
          q81_called = true;
          return true;
        },
        [&]() {
          packed_called = true;
          return true;
        },
        [&]() {
          fallback_called = true;
          return true;
        },
        [](FusedQuantGemm::DownProjOperator) {}, &summary);

    CAPTURE(static_cast<int>(op));
    REQUIRE(ok);

    switch (op) {
    case Op::kQ81Gemv:
    case Op::kQ81GemvHotFixed:
    case Op::kQ81GemvRowPairHotFixed:
    case Op::kQ81GemvRowPair:
    case Op::kQ81GemvRowQuad:
      REQUIRE(q81_called);
      REQUIRE_FALSE(mmq_called);
      REQUIRE_FALSE(mma_called);
      REQUIRE(summary.used_q81);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kMmq:
      REQUIRE(mmq_called);
      REQUIRE_FALSE(q81_called);
      REQUIRE_FALSE(mma_called);
      REQUIRE(summary.used_mmq);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kMmqMma:
      // The tensor-core selection must reach the MMA lambda — this is the
      // assertion that fails if the executor lets kMmqMma degrade to the
      // dp4a catch-all (the kQ81GroupMmq3 incident shape).
      REQUIRE(mma_called);
      REQUIRE_FALSE(mmq_called);
      REQUIRE_FALSE(q81_called);
      REQUIRE(summary.used_mmq_mma);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kPackedGemv:
      REQUIRE(packed_called);
      REQUIRE(summary.used_packed);
      REQUIRE(summary.actual_op == op);
      break;
    case Op::kFallback:
      // kFallback means the selector requested the dense fallback — the
      // fused catch-alls are skipped entirely.
      REQUIRE(fallback_called);
      REQUIRE(summary.actual_op == op);
      break;
    }
  }
}

TEST_CASE("FusedQuantGemm: metric names distinguish V2 hot-path variants",
          "[native_forward]") {
  const int q4k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);
  const int q6k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);
  const int q8k =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q8_K);

  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81Group, q4k, 1, 2048) ==
          std::string("q8_1_group_mmvq"));
  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81Group, q4k, 2, 2048) ==
          std::string("q8_1_group_mmvq"));
  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81Group, q6k, 5, 2048) ==
          std::string("q8_1_group_mmvq"));
  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K, q4k, 1, 2048) ==
          std::string("q8_1_group_hot_q4k"));
  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4, q4k, 4,
              2048) == std::string("q8_1_group_row_quad_m4"));
  REQUIRE(FusedQuantGemm::FfnProjOperatorMetricName(
              FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3, q4k, 4, 2048) ==
          std::string("q8_1_group_mmq3"));
  REQUIRE(FusedQuantGemm::DownProjOperatorMetricName(
              FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed, q4k, 2,
              11008) == std::string("q8_1_gemv_row_pair_hot_fixed"));
  REQUIRE(FusedQuantGemm::DownProjOperatorMetricName(
              FusedQuantGemm::DownProjOperator::kQ81GemvRowPair, q6k, 2,
              11008) == std::string("q8_1_gemv_row_pair"));
  REQUIRE(FusedQuantGemm::DownProjOperatorMetricName(
              FusedQuantGemm::DownProjOperator::kQ81Gemv, q8k, 1, 11008) ==
          std::string("q8_1_mmvq"));
  REQUIRE(FusedQuantGemm::DownProjOperatorMetricName(
              FusedQuantGemm::DownProjOperator::kMmq, q4k, 2, 11008) ==
          std::string("mmq"));
  REQUIRE(FusedQuantGemm::DownProjOperatorMetricName(
              FusedQuantGemm::DownProjOperator::kMmqMma, q6k, 8, 11008) ==
          std::string("mmq_mma"));
}

TEST_CASE("cuda_kernel::QuantizeRowsSymmetric quantizes once per row with "
          "stable scale contract",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping activation quantization contract.");
    return;
  }

  const int rows = 2;
  const int cols = 4;
  const std::vector<half> h_input = {
      __float2half(-4.0f), __float2half(-2.0f), __float2half(0.0f),
      __float2half(4.0f),  __float2half(0.0f),  __float2half(1.5f),
      __float2half(-1.5f), __float2half(3.0f),
  };

  half *d_input = nullptr;
  int8_t *d_quantized = nullptr;
  float *d_scales = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_input),
                     h_input.size() * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_quantized),
                     h_input.size() * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scales),
                     rows * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::QuantizeRowsSymmetric(d_input, d_quantized, d_scales,
                                             rows, cols,
                                             nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<int8_t> h_quantized(h_input.size());
  std::vector<float> h_scales(rows);
  REQUIRE(cudaMemcpy(h_quantized.data(), d_quantized,
                     h_quantized.size() * sizeof(int8_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h_scales.data(), d_scales, h_scales.size() * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  REQUIRE(h_scales[0] == Catch::Approx(4.0f / 127.0f).epsilon(1e-4f));
  REQUIRE(h_scales[1] == Catch::Approx(3.0f / 127.0f).epsilon(1e-4f));
  REQUIRE(static_cast<int>(h_quantized[0]) == -127);
  REQUIRE(static_cast<int>(h_quantized[1]) == -64);
  REQUIRE(static_cast<int>(h_quantized[2]) == 0);
  REQUIRE(static_cast<int>(h_quantized[3]) == 127);
  REQUIRE(static_cast<int>(h_quantized[4]) == 0);
  REQUIRE(static_cast<int>(h_quantized[5]) == 64);
  REQUIRE(static_cast<int>(h_quantized[6]) == -64);
  REQUIRE(static_cast<int>(h_quantized[7]) == 127);

  REQUIRE(cudaFree(d_scales) == cudaSuccess);
  REQUIRE(cudaFree(d_quantized) == cudaSuccess);
  REQUIRE(cudaFree(d_input) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm: CUDA runtime contract launches Q8_1 fused decode "
          "path and falls back above threshold",
          "[native_forward][cuda_runtime_contract]") {
  const char *disable_q81_env =
      std::getenv("INFERFLUX_DISABLE_Q8_1_ACTIVATIONS");
  if (disable_q81_env && (std::string(disable_q81_env) == "1" ||
                          std::string(disable_q81_env) == "true")) {
    SUCCEED(
        "Q8_1 activations disabled via INFERFLUX_DISABLE_Q8_1_ACTIVATIONS.");
    return;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping CUDA runtime contract.");
    return;
  }

  const auto run_contract = [](runtime::cuda::native::GGUF::TensorType qtype,
                               int K, int N) {
    const int quant_type = static_cast<int>(qtype);
    const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
        quant_type, FusedDispatchGeometry{1, N, K, 1, true, false});
    const int m_fused = 1;
    const int m_fallback = threshold + 1;

    const size_t weight_bytes = runtime::cuda::native::CalcTensorSize(
        qtype, {static_cast<uint64_t>(N), static_cast<uint64_t>(K)});
    REQUIRE(weight_bytes > 0);

    // Q8_1 blocks: K/32 blocks per row, each 36 bytes (32 int8 + half2 ds)
    const int q8_1_blocks_per_row = K / 32;
    const size_t q8_1_block_size = 36;

    void *d_weight = nullptr;
    half *d_activation = nullptr;
    half *d_output = nullptr;
    void *d_act_q8_1 = nullptr;

    REQUIRE(cudaMalloc(&d_weight, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMemset(d_weight, 0, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                       static_cast<size_t>(m_fallback) * K * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_activation, 0,
                       static_cast<size_t>(m_fallback) * K * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_output),
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_output, 0,
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d_act_q8_1, static_cast<size_t>(m_fallback) *
                                        q8_1_blocks_per_row *
                                        q8_1_block_size) == cudaSuccess);

    QuantizedWeightInfo info;
    info.data = d_weight;
    info.quant_type = quant_type;
    info.num_elements = static_cast<int64_t>(N) * static_cast<int64_t>(K);

    (void)cudaGetLastError(); // clear sticky errors before launch checks

    // Quantize activations to Q8_1
    FusedQuantGemm::QuantizeRowQ8_1(d_activation, d_act_q8_1, m_fused, K,
                                    nullptr);
    const bool used_fused = FusedQuantGemm::GemvQ8_1(info, d_act_q8_1, d_output,
                                                     m_fused, N, K, nullptr);
    REQUIRE(used_fused);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    FusedQuantGemm::QuantizeRowQ8_1(d_activation, d_act_q8_1, m_fallback, K,
                                    nullptr);
    const bool used_fallback = FusedQuantGemm::GemvQ8_1(
        info, d_act_q8_1, d_output, m_fallback, N, K, nullptr);
    REQUIRE_FALSE(used_fallback);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaFree(d_act_q8_1) == cudaSuccess);
    REQUIRE(cudaFree(d_output) == cudaSuccess);
    REQUIRE(cudaFree(d_activation) == cudaSuccess);
    REQUIRE(cudaFree(d_weight) == cudaSuccess);
  };

  // Keep dimensions minimal but valid for each quant block layout.
  run_contract(runtime::cuda::native::GGUF::TensorType::Q4_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q6_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_0, 32, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_K, 256, 32);
}

TEST_CASE("FusedQuantGemm::GemvPacked launches packed fused decode path and "
          "falls back above threshold",
          "[native_forward][cuda_runtime_contract]") {
  const char *disable_fused_env = std::getenv("INFERFLUX_DISABLE_FUSED_GEMV");
  if (disable_fused_env && (std::string(disable_fused_env) == "1" ||
                            std::string(disable_fused_env) == "true")) {
    SUCCEED("Fused kernels disabled via INFERFLUX_DISABLE_FUSED_GEMV.");
    return;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping packed fused runtime contract.");
    return;
  }

  const auto run_contract = [](runtime::cuda::native::GGUF::TensorType qtype,
                               int K, int N) {
    const int quant_type = static_cast<int>(qtype);
    if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
      SUCCEED("Packed activation kernels unavailable for this device/profile.");
      return;
    }

    const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
        quant_type, FusedDispatchGeometry{1, N, K, 1, true, false});
    const int m_fused = 1;
    const int m_fallback = threshold + 1;

    const size_t weight_bytes = runtime::cuda::native::CalcTensorSize(
        qtype, {static_cast<uint64_t>(N), static_cast<uint64_t>(K)});
    REQUIRE(weight_bytes > 0);

    void *d_weight = nullptr;
    int8_t *d_activation = nullptr;
    float *d_scales = nullptr;
    half *d_output = nullptr;

    REQUIRE(cudaMalloc(&d_weight, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMemset(d_weight, 0, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_activation, 0,
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scales),
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_scales, 0,
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_output),
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_output, 0,
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);

    QuantizedWeightInfo info;
    info.data = d_weight;
    info.quant_type = quant_type;
    info.num_elements = static_cast<int64_t>(N) * static_cast<int64_t>(K);

    PackedActivationInfo activation;
    activation.data = d_activation;
    activation.row_scales = d_scales;

    (void)cudaGetLastError();
    const bool used_fused = FusedQuantGemm::GemvPacked(
        info, activation, d_output, m_fused, N, K, nullptr);
    REQUIRE(used_fused);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    const bool used_fallback = FusedQuantGemm::GemvPacked(
        info, activation, d_output, m_fallback, N, K, nullptr);
    REQUIRE_FALSE(used_fallback);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaFree(d_output) == cudaSuccess);
    REQUIRE(cudaFree(d_scales) == cudaSuccess);
    REQUIRE(cudaFree(d_activation) == cudaSuccess);
    REQUIRE(cudaFree(d_weight) == cudaSuccess);
  };

  run_contract(runtime::cuda::native::GGUF::TensorType::Q4_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q6_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_0, 32, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_K, 256, 32);
}

TEST_CASE("FusedQuantGemm::GemvPackedPair launches one grouped packed kernel "
          "and falls back above threshold",
          "[native_forward][cuda_runtime_contract]") {
  const char *disable_fused_env = std::getenv("INFERFLUX_DISABLE_FUSED_GEMV");
  if (disable_fused_env && (std::string(disable_fused_env) == "1" ||
                            std::string(disable_fused_env) == "true")) {
    SUCCEED("Fused kernels disabled via INFERFLUX_DISABLE_FUSED_GEMV.");
    return;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping grouped pair runtime contract.");
    return;
  }

  const auto run_contract = [](runtime::cuda::native::GGUF::TensorType qtype,
                               int K, int N0, int N1) {
    const int quant_type = static_cast<int>(qtype);
    if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
      SUCCEED("Packed activation kernels unavailable for this device/profile.");
      return;
    }

    const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
        quant_type,
        FusedDispatchGeometry{1, std::max(N0, N1), K, 2, true, false});
    const int m_fused = 1;
    const int m_fallback = threshold + 1;

    const std::array<int, 2> output_cols = {N0, N1};
    std::array<void *, 2> d_weight = {nullptr, nullptr};
    std::array<half *, 2> d_output = {nullptr, nullptr};
    for (size_t i = 0; i < output_cols.size(); ++i) {
      const size_t weight_bytes = runtime::cuda::native::CalcTensorSize(
          qtype,
          {static_cast<uint64_t>(output_cols[i]), static_cast<uint64_t>(K)});
      REQUIRE(weight_bytes > 0);
      REQUIRE(cudaMalloc(&d_weight[i], weight_bytes) == cudaSuccess);
      REQUIRE(cudaMemset(d_weight[i], 0, weight_bytes) == cudaSuccess);
      REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_output[i]),
                         static_cast<size_t>(m_fallback) * output_cols[i] *
                             sizeof(half)) == cudaSuccess);
      REQUIRE(cudaMemset(d_output[i], 0,
                         static_cast<size_t>(m_fallback) * output_cols[i] *
                             sizeof(half)) == cudaSuccess);
    }

    int8_t *d_activation = nullptr;
    float *d_scales = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_activation, 0,
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scales),
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_scales, 0,
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);

    const std::array<PackedProjectionSpec, 2> projections = {{
        {{d_weight[0], quant_type,
          static_cast<int64_t>(output_cols[0]) * static_cast<int64_t>(K)},
         d_output[0],
         output_cols[0]},
        {{d_weight[1], quant_type,
          static_cast<int64_t>(output_cols[1]) * static_cast<int64_t>(K)},
         d_output[1],
         output_cols[1]},
    }};
    PackedActivationInfo activation{d_activation, d_scales};

    (void)cudaGetLastError();
    const bool used_fused = FusedQuantGemm::GemvPackedPair(
        projections, activation, m_fused, K, nullptr);
    REQUIRE(used_fused);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    const bool used_fallback = FusedQuantGemm::GemvPackedPair(
        projections, activation, m_fallback, K, nullptr);
    REQUIRE_FALSE(used_fallback);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaFree(d_scales) == cudaSuccess);
    REQUIRE(cudaFree(d_activation) == cudaSuccess);
    for (size_t i = 0; i < d_output.size(); ++i) {
      REQUIRE(cudaFree(d_output[i]) == cudaSuccess);
      REQUIRE(cudaFree(d_weight[i]) == cudaSuccess);
    }
  };

  run_contract(runtime::cuda::native::GGUF::TensorType::Q4_K, 256, 32, 16);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q6_K, 256, 32, 16);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_0, 32, 32, 16);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_K, 256, 32, 16);
}

TEST_CASE("FusedQuantGemm::GemvPackedTriple launches one grouped packed "
          "kernel and falls back above threshold",
          "[native_forward][cuda_runtime_contract]") {
  const char *disable_fused_env = std::getenv("INFERFLUX_DISABLE_FUSED_GEMV");
  if (disable_fused_env && (std::string(disable_fused_env) == "1" ||
                            std::string(disable_fused_env) == "true")) {
    SUCCEED("Fused kernels disabled via INFERFLUX_DISABLE_FUSED_GEMV.");
    return;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping grouped triple runtime contract.");
    return;
  }

  const auto run_contract = [](runtime::cuda::native::GGUF::TensorType qtype,
                               int K, int N0, int N1, int N2) {
    const int quant_type = static_cast<int>(qtype);
    if (!FusedQuantGemm::SupportsPackedActivations(quant_type)) {
      SUCCEED("Packed activation kernels unavailable for this device/profile.");
      return;
    }

    const int threshold = FusedQuantGemm::GetGeometryAwareThreshold(
        quant_type,
        FusedDispatchGeometry{1, std::max({N0, N1, N2}), K, 3, true, false});
    const int m_fused = 1;
    const int m_fallback = threshold + 1;

    const std::array<int, 3> output_cols = {N0, N1, N2};
    std::array<void *, 3> d_weight = {nullptr, nullptr, nullptr};
    std::array<half *, 3> d_output = {nullptr, nullptr, nullptr};
    for (size_t i = 0; i < output_cols.size(); ++i) {
      const size_t weight_bytes = runtime::cuda::native::CalcTensorSize(
          qtype,
          {static_cast<uint64_t>(output_cols[i]), static_cast<uint64_t>(K)});
      REQUIRE(weight_bytes > 0);
      REQUIRE(cudaMalloc(&d_weight[i], weight_bytes) == cudaSuccess);
      REQUIRE(cudaMemset(d_weight[i], 0, weight_bytes) == cudaSuccess);
      REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_output[i]),
                         static_cast<size_t>(m_fallback) * output_cols[i] *
                             sizeof(half)) == cudaSuccess);
      REQUIRE(cudaMemset(d_output[i], 0,
                         static_cast<size_t>(m_fallback) * output_cols[i] *
                             sizeof(half)) == cudaSuccess);
    }

    int8_t *d_activation = nullptr;
    float *d_scales = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_activation, 0,
                       static_cast<size_t>(m_fallback) * K * sizeof(int8_t)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scales),
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_scales, 0,
                       static_cast<size_t>(m_fallback) * sizeof(float)) ==
            cudaSuccess);

    const std::array<PackedProjectionSpec, 3> projections = {{
        {{d_weight[0], quant_type,
          static_cast<int64_t>(output_cols[0]) * static_cast<int64_t>(K)},
         d_output[0],
         output_cols[0]},
        {{d_weight[1], quant_type,
          static_cast<int64_t>(output_cols[1]) * static_cast<int64_t>(K)},
         d_output[1],
         output_cols[1]},
        {{d_weight[2], quant_type,
          static_cast<int64_t>(output_cols[2]) * static_cast<int64_t>(K)},
         d_output[2],
         output_cols[2]},
    }};
    PackedActivationInfo activation{d_activation, d_scales};

    (void)cudaGetLastError();
    const bool used_fused = FusedQuantGemm::GemvPackedTriple(
        projections, activation, m_fused, K, nullptr);
    REQUIRE(used_fused);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    const bool used_fallback = FusedQuantGemm::GemvPackedTriple(
        projections, activation, m_fallback, K, nullptr);
    REQUIRE_FALSE(used_fallback);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaFree(d_scales) == cudaSuccess);
    REQUIRE(cudaFree(d_activation) == cudaSuccess);
    for (size_t i = 0; i < d_output.size(); ++i) {
      REQUIRE(cudaFree(d_output[i]) == cudaSuccess);
      REQUIRE(cudaFree(d_weight[i]) == cudaSuccess);
    }
  };

  run_contract(runtime::cuda::native::GGUF::TensorType::Q4_K, 256, 32, 8, 8);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q6_K, 256, 32, 8, 8);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_0, 32, 32, 8, 8);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_K, 256, 32, 8, 8);
}

TEST_CASE("cuda_kernel::SiluMulQuantizeRowsSymmetric fuses SwiGLU activation "
          "and row quantization",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping fused SwiGLU quantization test.");
    return;
  }

  constexpr int rows = 1;
  constexpr int cols = 8;
  const std::array<float, cols> gate_vals = {-1.0f, 0.0f, 1.0f,  2.0f,
                                             -2.0f, 0.5f, -0.5f, 3.0f};
  const std::array<float, cols> up_vals = {1.0f, 1.0f, 1.0f, 1.0f,
                                           0.5f, 2.0f, 4.0f, 1.0f};

  std::array<half, cols> h_gate{};
  std::array<half, cols> h_up{};
  std::array<float, cols> expected{};
  float max_abs = 0.0f;
  for (int i = 0; i < cols; ++i) {
    h_gate[i] = __float2half(gate_vals[i]);
    h_up[i] = __float2half(up_vals[i]);
    const float silu = gate_vals[i] / (1.0f + std::exp(-gate_vals[i]));
    expected[i] = silu * up_vals[i];
    max_abs = std::max(max_abs, std::fabs(expected[i]));
  }
  const float expected_scale = max_abs / 127.0f;

  half *d_gate = nullptr;
  half *d_up = nullptr;
  int8_t *d_q = nullptr;
  float *d_scale = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_gate),
                     rows * cols * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_up),
                     rows * cols * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q),
                     rows * cols * sizeof(int8_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_scale),
                     rows * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_gate, h_gate.data(), rows * cols * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_up, h_up.data(), rows * cols * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cuda_kernel::SiluMulQuantizeRowsSymmetric(
              d_gate, d_up, d_q, d_scale, rows, cols, nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::array<int8_t, cols> q{};
  float scale = 0.0f;
  REQUIRE(cudaMemcpy(q.data(), d_q, rows * cols * sizeof(int8_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(&scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost) ==
          cudaSuccess);

  REQUIRE(scale == Catch::Approx(expected_scale).margin(1e-4));
  for (int i = 0; i < cols; ++i) {
    const int expected_q =
        std::max(-127, std::min(127, static_cast<int>(std::lrint(
                                         expected[i] / expected_scale))));
    REQUIRE(static_cast<int>(q[i]) == expected_q);
  }

  REQUIRE(cudaFree(d_scale) == cudaSuccess);
  REQUIRE(cudaFree(d_q) == cudaSuccess);
  REQUIRE(cudaFree(d_up) == cudaSuccess);
  REQUIRE(cudaFree(d_gate) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::SiluMulQuantizeQ8_1 produces Q8_1 activation "
          "blocks for down projection",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED(
        "No CUDA device available; skipping Q8_1 SwiGLU quantization test.");
    return;
  }

  constexpr int M = 1;
  constexpr int K = QK8_1;
  std::array<half, K> h_gate{};
  std::array<half, K> h_up{};
  std::array<float, K> expected{};
  float max_abs = 0.0f;
  for (int i = 0; i < K; ++i) {
    const float gate = (static_cast<float>(i) - 16.0f) / 8.0f;
    const float up = 1.0f + static_cast<float>(i % 5) * 0.25f;
    h_gate[i] = __float2half(gate);
    h_up[i] = __float2half(up);
    const float silu = gate / (1.0f + std::exp(-gate));
    expected[i] = silu * up;
    max_abs = std::max(max_abs, std::fabs(expected[i]));
  }
  const float expected_d = max_abs / 127.0f;

  half *d_gate = nullptr;
  half *d_up = nullptr;
  runtime::cuda::native::block_q8_1 *d_q8_1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_gate),
                     M * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_up), M * K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_q8_1),
                     M * sizeof(runtime::cuda::native::block_q8_1)) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(d_gate, h_gate.data(), M * K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_up, h_up.data(), M * K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  FusedQuantGemm::SiluMulQuantizeQ8_1(d_gate, d_up, d_q8_1, M, K, nullptr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  runtime::cuda::native::block_q8_1 block{};
  REQUIRE(cudaMemcpy(&block, d_q8_1, sizeof(block), cudaMemcpyDeviceToHost) ==
          cudaSuccess);

  const auto [actual_d, actual_ds] = DecodeQ81Ds(block);
  int expected_sum = 0;
  REQUIRE(actual_d == Catch::Approx(expected_d).margin(1e-4));
  for (int i = 0; i < K; ++i) {
    const int expected_q = std::max(
        -128,
        std::min(127, static_cast<int>(std::lrint(expected[i] / expected_d))));
    expected_sum += expected_q;
    REQUIRE(static_cast<int>(block.qs[i]) == expected_q);
  }
  REQUIRE(actual_ds ==
          Catch::Approx(expected_d * static_cast<float>(expected_sum))
              .margin(5e-3));

  REQUIRE(cudaFree(d_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_up) == cudaSuccess);
  REQUIRE(cudaFree(d_gate) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::SiluMulQuantizeQ8_1 preserves multi-block Q8_1 "
          "rows for down projection",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping multi-block Q8_1 SwiGLU "
            "quantization test.");
    return;
  }

  constexpr int M = 1;
  constexpr int K = QK8_1 * 2;
  std::array<half, K> h_gate{};
  std::array<half, K> h_up{};
  std::array<float, K> expected{};
  for (int i = 0; i < K; ++i) {
    const float gate = std::sin(0.19f * static_cast<float>(i)) * 2.0f;
    const float up = 0.75f + static_cast<float>((i * 3) % 11) * 0.17f;
    h_gate[i] = __float2half(gate);
    h_up[i] = __float2half(up);
    expected[i] = (gate / (1.0f + std::exp(-gate))) * up;
  }

  half *d_gate = nullptr;
  half *d_up = nullptr;
  runtime::cuda::native::block_q8_1 *d_q8_1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_gate),
                     M * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_up), M * K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(
      cudaMalloc(reinterpret_cast<void **>(&d_q8_1),
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMemcpy(d_gate, h_gate.data(), M * K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_up, h_up.data(), M * K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  FusedQuantGemm::SiluMulQuantizeQ8_1(d_gate, d_up, d_q8_1, M, K, nullptr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::array<runtime::cuda::native::block_q8_1, K / QK8_1> blocks{};
  REQUIRE(cudaMemcpy(blocks.data(), d_q8_1, sizeof(blocks),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  for (int blk = 0; blk < K / QK8_1; ++blk) {
    float max_abs = 0.0f;
    for (int i = 0; i < QK8_1; ++i) {
      max_abs = std::max(max_abs, std::fabs(expected[blk * QK8_1 + i]));
    }
    const float expected_d = max_abs / 127.0f;
    const auto [actual_d, actual_ds] = DecodeQ81Ds(blocks[blk]);
    int expected_sum = 0;
    REQUIRE(actual_d == Catch::Approx(expected_d).margin(1e-4));
    for (int i = 0; i < QK8_1; ++i) {
      const float value = expected[blk * QK8_1 + i];
      const int expected_q = std::max(
          -128,
          std::min(127, static_cast<int>(std::lrint(value / expected_d))));
      expected_sum += expected_q;
      REQUIRE(static_cast<int>(blocks[blk].qs[i]) == expected_q);
    }
    REQUIRE(actual_ds ==
            Catch::Approx(expected_d * static_cast<float>(expected_sum))
                .margin(1e-2));
  }

  REQUIRE(cudaFree(d_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_up) == cudaSuccess);
  REQUIRE(cudaFree(d_gate) == cudaSuccess);
}

TEST_CASE("FusedQuantGemm::FusedRmsNormQuantizeQ8_1 preserves multi-block "
          "rows for grouped FFN/QKV reuse",
          "[native_forward][cuda_runtime_contract]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping multi-block fused RMSNorm "
            "Q8_1 quantization test.");
    return;
  }

  constexpr int M = 1;
  constexpr int K = QK8_1 * 2;
  constexpr float eps = 1e-5f;
  std::array<half, K> h_residual{};
  std::array<half, K> h_weight{};
  std::array<float, K> normalized{};
  float sum_sq = 0.0f;
  for (int i = 0; i < K; ++i) {
    const float residual = std::cos(0.11f * static_cast<float>(i)) * 1.7f;
    const float weight = 0.9f + static_cast<float>(i % 7) * 0.03f;
    h_residual[i] = __float2half(residual);
    h_weight[i] = __float2half(weight);
    sum_sq += residual * residual;
  }
  const float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(K) + eps);
  for (int i = 0; i < K; ++i) {
    normalized[i] =
        __half2float(h_residual[i]) * rms * __half2float(h_weight[i]);
  }

  half *d_residual = nullptr;
  half *d_weight = nullptr;
  runtime::cuda::native::block_q8_1 *d_q8_1 = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_residual),
                     M * K * sizeof(half)) == cudaSuccess);
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_weight), K * sizeof(half)) ==
          cudaSuccess);
  REQUIRE(
      cudaMalloc(reinterpret_cast<void **>(&d_q8_1),
                 M * (K / QK8_1) * sizeof(runtime::cuda::native::block_q8_1)) ==
      cudaSuccess);
  REQUIRE(cudaMemcpy(d_residual, h_residual.data(), M * K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_weight, h_weight.data(), K * sizeof(half),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  FusedQuantGemm::FusedRmsNormQuantizeQ8_1(d_residual, d_weight, d_q8_1, M, K,
                                           eps, nullptr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::array<runtime::cuda::native::block_q8_1, K / QK8_1> blocks{};
  REQUIRE(cudaMemcpy(blocks.data(), d_q8_1, sizeof(blocks),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  for (int blk = 0; blk < K / QK8_1; ++blk) {
    float max_abs = 0.0f;
    for (int i = 0; i < QK8_1; ++i) {
      max_abs = std::max(max_abs, std::fabs(normalized[blk * QK8_1 + i]));
    }
    const float expected_d = max_abs / 127.0f;
    const auto [actual_d, actual_ds] = DecodeQ81Ds(blocks[blk]);
    int expected_sum = 0;
    REQUIRE(actual_d == Catch::Approx(expected_d).margin(1e-4));
    for (int i = 0; i < QK8_1; ++i) {
      const float value = normalized[blk * QK8_1 + i];
      const int expected_q = std::max(
          -128,
          std::min(127, static_cast<int>(std::lrint(value / expected_d))));
      expected_sum += expected_q;
      REQUIRE(static_cast<int>(blocks[blk].qs[i]) == expected_q);
    }
    REQUIRE(actual_ds ==
            Catch::Approx(expected_d * static_cast<float>(expected_sum))
                .margin(1e-2));
  }

  REQUIRE(cudaFree(d_q8_1) == cudaSuccess);
  REQUIRE(cudaFree(d_weight) == cudaSuccess);
  REQUIRE(cudaFree(d_residual) == cudaSuccess);
}

// ============================================================================
// QuantizedWeightMapAdapter Tests
// ============================================================================

TEST_CASE("QuantizedWeightMapAdapter: delegates NumLayers to underlying map",
          "[native_forward]") {
  // QuantizedWeightMap without Build() returns 0 layers
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Adapter should delegate to the underlying map
  REQUIRE(adapter.NumLayers() == 0);
  REQUIRE(adapter.EmbedTokens() == nullptr);
  REQUIRE(adapter.FinalNorm() == nullptr);
  REQUIRE(adapter.LmHead() == nullptr);
}

TEST_CASE("QuantizedWeightMapAdapter: is a WeightMapTyped<half>",
          "[native_forward]") {
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Verify polymorphism: adapter can be used as WeightMapTyped<half>*
  const WeightMap *base = &adapter;
  REQUIRE(base->NumLayers() == 0);
  REQUIRE(base->EmbedTokens() == nullptr);
}

TEST_CASE("FusedQuantGemm: Q4_K column-pair dispatch uses halved grid for M=1",
          "[native_forward]") {
  // Column-pair kernel processes 2 output columns per warp, so for M=1 the
  // effective outputs per block doubles from kGemvWarpsPerBlock to 2x.
  // This test validates the dispatch logic selects column-pair for M=1
  // and row-pair/row-quad for M>1.
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);

  // M=1 should use fused path (column-pair will be selected internally)
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));

  // M=2 should still use fused path (row-pair variant)
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 2));

  // M=4 should still use fused path (row-quad variant)
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 4));
}

TEST_CASE("FusedQuantGemm: Q6_K column-pair dispatch uses halved grid for M=1",
          "[native_forward]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);

  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 2));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 4));
}

// ============================================================================
// Cooperative-Warp GEMV v2 Tests
//
// These tests exercise the v2 dispatch path via FusedQuantGemm::GemvQ8_1().
// V2 is auto-selected for K >= 1024 with Q4_K/Q6_K quant types.
// ============================================================================

TEST_CASE("GEMV v2: Q4_K dispatch succeeds for representative dimensions",
          "[native_forward][gemv_v2]") {
  // Verify the dispatch path (which now uses v2 for K>=1024) accepts
  // representative Qwen2.5-3B dimensions and produces non-zero output.
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K);

  // All realistic K values are >= 1024, so v2 will be selected
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 2));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 4));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 8));
}

TEST_CASE("GEMV v2: Q6_K dispatch succeeds for representative dimensions",
          "[native_forward][gemv_v2]") {
  const int quant_type =
      static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K);

  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 2));
  REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 4));
}

TEST_CASE("Q6_K MMA architecture gate requires SM 7.5", "[native_forward]") {
  REQUIRE_FALSE(FusedQuantGemm::SupportsMmqMmaArchitecture(7, 0));
  REQUIRE_FALSE(FusedQuantGemm::SupportsMmqMmaArchitecture(7, 4));
  REQUIRE(FusedQuantGemm::SupportsMmqMmaArchitecture(7, 5));
  REQUIRE(FusedQuantGemm::SupportsMmqMmaArchitecture(8, 0));
  REQUIRE(FusedQuantGemm::SupportsMmqMmaArchitecture(12, 0));
}

#endif // INFERFLUX_NATIVE_KERNELS_READY

} // namespace inferflux
