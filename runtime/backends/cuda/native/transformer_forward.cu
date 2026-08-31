#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "runtime/backends/cuda/native/cuda_copy_trace.h"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/dispatch_operator_health.h"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/kernels/fused_gemv_accum_norm_quant.cuh"
#include "runtime/backends/cuda/native/kernels/fused_rope_kv_append.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "runtime/backends/cuda/native/llama_forward.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/native_dispatch_policy.h"
#include "runtime/backends/cuda/native/native_linear_executor.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"

#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace inferflux {

namespace {

template <typename T> bool AllocatePinnedHostBuffer(T **ptr, size_t count) {
  void *storage = nullptr;
  if (cudaMallocHost(&storage, count * sizeof(T)) != cudaSuccess) {
    return false;
  }
  *ptr = static_cast<T *>(storage);
  return true;
}

template <typename T> void FreePinnedHostBuffer(T **ptr) {
  if (!ptr || !*ptr) {
    return;
  }
  cudaFreeHost(const_cast<void *>(reinterpret_cast<const void *>(*ptr)));
  *ptr = nullptr;
}

template <typename T>
void AssignBatchMetadataViews(T *base, size_t batch_size, T **token_ids,
                              T **n_past, T **seq_ids, T **kv_lens) {
  if (!base || batch_size == 0) {
    *token_ids = nullptr;
    *n_past = nullptr;
    *seq_ids = nullptr;
    *kv_lens = nullptr;
    return;
  }
  *token_ids = base;
  *n_past = base + batch_size;
  *seq_ids = base + (batch_size * 2);
  *kv_lens = base + (batch_size * 3);
}

// Phase timing: sync-based per-phase breakdown when
// INFERFLUX_CUDA_PHASE_TIMING=1 Serializes GPU pipeline — for
// debugging/profiling only, not production.
struct PhaseTiming {
  bool enabled{false};
  cudaStream_t stream{nullptr};
  std::chrono::steady_clock::time_point last;
  double embed_ms{0}, qkv_ms{0}, rope_ms{0}, kv_ms{0}, attn_ms{0};
  double o_proj_ms{0}, ffn_proj_ms{0}, ffn_silu_ms{0}, ffn_down_ms{0},
      lm_head_ms{0};
  int forward_count{0};
  int kernel_launches{0};

  void Begin(cudaStream_t s, bool should_enable) {
    if (!should_enable)
      return;
    enabled = true;
    stream = s;
    embed_ms = qkv_ms = rope_ms = kv_ms = attn_ms = 0;
    o_proj_ms = ffn_proj_ms = ffn_silu_ms = ffn_down_ms = lm_head_ms = 0;
    kernel_launches = 0;
    cudaStreamSynchronize(stream);
    last = std::chrono::steady_clock::now();
  }

  void CountKernel(int count = 1) {
    if (enabled) {
      kernel_launches += count;
    }
  }

  double Mark() {
    if (!enabled)
      return 0;
    cudaStreamSynchronize(stream);
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    return ms;
  }

  void Report(int num_layers, int token_count) {
    if (!enabled)
      return;
    ++forward_count;
    const double ffn_ms = ffn_proj_ms + ffn_silu_ms + ffn_down_ms;
    double total = embed_ms + qkv_ms + rope_ms + kv_ms + attn_ms + o_proj_ms +
                   ffn_ms + lm_head_ms;
    // Print every forward pass for first 5, then every 10th
    if (forward_count <= 5 || forward_count % 10 == 0) {
      fprintf(stderr,
              "[phase_timing] #%d L=%d tokens=%d kernels=%d embed=%.2f "
              "qkv=%.2f rope=%.2f kv=%.2f attn=%.2f o_proj=%.2f "
              "ffn_proj=%.2f ffn_silu=%.2f ffn_down=%.2f ffn=%.2f "
              "lm_head=%.2f total=%.2f ms\n",
              forward_count, num_layers, token_count, kernel_launches, embed_ms,
              qkv_ms, rope_ms, kv_ms, attn_ms, o_proj_ms, ffn_proj_ms,
              ffn_silu_ms, ffn_down_ms, ffn_ms, lm_head_ms, total);
    }
  }
};

// Debug: dump top-K logits to stderr when INFERFLUX_DEBUG_LOGITS=1
void DebugDumpLogits(const float *d_logits, int vocab_size,
                     const std::vector<int> &token_ids, int n_past,
                     cudaStream_t stream,
                     const NativeExecutionPolicy *policy = nullptr) {
  if (!ResolveInferfluxCudaExecutionPolicy(policy).debug_logits)
    return;
  constexpr int TOP_N = 10;
  std::vector<float> h_logits(vocab_size);
  cudaError_t err =
      cudaMemcpyAsync(h_logits.data(), d_logits, vocab_size * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[DEBUG_LOGITS] cudaMemcpyAsync failed: %s\n",
            cudaGetErrorString(err));
    return;
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[DEBUG_LOGITS] cudaStreamSynchronize failed: %s\n",
            cudaGetErrorString(err));
    return;
  }

  // Find top-N by value
  std::vector<std::pair<float, int>> scored(vocab_size);
  for (int i = 0; i < vocab_size; ++i)
    scored[i] = {h_logits[i], i};
  std::partial_sort(scored.begin(), scored.begin() + TOP_N, scored.end(),
                    [](auto &a, auto &b) { return a.first > b.first; });

  fprintf(stderr, "[DEBUG_LOGITS] tokens=[");
  for (size_t i = 0; i < token_ids.size(); ++i)
    fprintf(stderr, "%s%d", i ? "," : "", token_ids[i]);
  fprintf(stderr, "] n_past=%d top-%d:", n_past, TOP_N);
  for (int i = 0; i < TOP_N; ++i)
    fprintf(stderr, " [%d]=%.4f", scored[i].second, scored[i].first);

  // Also check for NaN/Inf
  int nan_count = 0, inf_count = 0, zero_count = 0;
  for (int i = 0; i < vocab_size; ++i) {
    if (std::isnan(h_logits[i]))
      nan_count++;
    if (std::isinf(h_logits[i]))
      inf_count++;
    if (h_logits[i] == 0.0f)
      zero_count++;
  }
  fprintf(stderr, " (nan=%d inf=%d zero=%d/%d)\n", nan_count, inf_count,
          zero_count, vocab_size);
}

// Debug: dump hidden state stats
void DebugDumpHidden(const char *label, const void *d_data, int count,
                     cudaStream_t stream,
                     const NativeExecutionPolicy *policy = nullptr) {
  if (!ResolveInferfluxCudaExecutionPolicy(policy).debug_logits)
    return;
  // Read as half, convert to float
  std::vector<half> h_data(count);
  cudaError_t err = cudaMemcpyAsync(h_data.data(), d_data, count * sizeof(half),
                                    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[DEBUG_HIDDEN] %s: cudaMemcpyAsync failed: %s\n", label,
            cudaGetErrorString(err));
    return;
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[DEBUG_HIDDEN] %s: cudaStreamSynchronize failed: %s\n",
            label, cudaGetErrorString(err));
    return;
  }

  float min_v = 1e30f, max_v = -1e30f, sum = 0.0f;
  int nan_count = 0;
  for (int i = 0; i < count; ++i) {
    float v = __half2float(h_data[i]);
    if (std::isnan(v)) {
      nan_count++;
      continue;
    }
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    sum += v;
  }
  fprintf(stderr,
          "[DEBUG_HIDDEN] %s: count=%d min=%.6f max=%.6f mean=%.6f nan=%d\n",
          label, count, min_v, max_v, sum / count, nan_count);
}

void LogPackedGemmPath(const char *proj_name, const char *label) {
  static std::unordered_map<std::string, bool> logged;
  const std::string key =
      std::string(proj_name ? proj_name : "") + "|" + (label ? label : "");
  if (logged.count(key)) {
    return;
  }
  logged[key] = true;
  log::Info("llama_forward", std::string(proj_name) + ": " + label);
}

thread_local bool g_allow_fused_quantized_matmul = true;

class ScopedFusedMatmulPolicy {
public:
  explicit ScopedFusedMatmulPolicy(bool allow)
      : prev_(g_allow_fused_quantized_matmul) {
    g_allow_fused_quantized_matmul = allow;
  }
  ~ScopedFusedMatmulPolicy() { g_allow_fused_quantized_matmul = prev_; }

private:
  bool prev_{true};
};

template <typename T>
bool TryPackedGemv(const QuantizedWeightInfo &, const T *, T *, int8_t *,
                   float *, int, int, int, cudaStream_t, const char * = nullptr,
                   const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedGemv<half>(const QuantizedWeightInfo &raw, const half *input,
                         half *output, int8_t *packed_activation,
                         float *packed_scales, int M, int N, int K,
                         cudaStream_t stream, const char *proj_name,
                         const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !packed_activation || !packed_scales || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  if (cuda_kernel::QuantizeRowsSymmetric(input, packed_activation,
                                         packed_scales, M, K,
                                         stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using packed-activation fused dequant-GEMV");
  }
  return ok;
}

template <typename T>
bool TryPackedRmsNormGemv(const QuantizedWeightInfo &, const T *, const T *,
                          T *, int8_t *, float *, T *, int, int, int, float,
                          cudaStream_t, const char * = nullptr,
                          const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedRmsNormGemv<half>(const QuantizedWeightInfo &raw,
                                const half *residual, const half *norm_weight,
                                half *normalized, int8_t *packed_activation,
                                float *packed_scales, half *output, int M,
                                int N, int K, float eps, cudaStream_t stream,
                                const char *proj_name,
                                const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !residual ||
      !norm_weight || !normalized || !packed_activation || !packed_scales ||
      !output || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, true})) {
    return false;
  }

  if (cuda_kernel::RmsNorm<half>(residual, norm_weight, normalized, M, K, eps,
                                 stream) != cudaSuccess) {
    return false;
  }

  if (cuda_kernel::QuantizeRowsSymmetric(normalized, packed_activation,
                                         packed_scales, M, K,
                                         stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using packed-activation fused RmsNorm+dequant-GEMV");
  }
  return ok;
}

template <typename T> struct PackedProjectionPlan {
  QuantizedWeightInfo raw;
  T *output{nullptr};
  int output_cols{0};
};

template <size_t GroupSize, typename T>
bool TryPackedProjectionGroup(
    const std::array<PackedProjectionPlan<T>, GroupSize> &, const T *,
    const T *, T *, int8_t *, float *, int, int, float, cudaStream_t,
    const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <size_t GroupSize>
bool TryPackedProjectionGroup(
    const std::array<PackedProjectionPlan<half>, GroupSize> &plans,
    const half *residual, const half *norm_weight, half *normalized,
    int8_t *packed_activation, float *packed_scales, int M, int K, float eps,
    cudaStream_t stream, const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !residual ||
      !packed_activation || !packed_scales || M <= 0 || K <= 0) {
    return false;
  }

  std::array<int, GroupSize> quant_types{};
  std::array<int, GroupSize> output_cols{};
  std::array<bool, GroupSize> pair_ready{};
  std::array<bool, GroupSize> triple_ready{};
  const bool has_norm = (norm_weight != nullptr);
  for (size_t i = 0; i < GroupSize; ++i) {
    const auto &plan = plans[i];
    if (!plan.raw.data || plan.output == nullptr || plan.output_cols <= 0 ||
        !FusedQuantGemm::SupportsPackedActivations(plan.raw.quant_type)) {
      return false;
    }
    quant_types[i] = plan.raw.quant_type;
    output_cols[i] = plan.output_cols;
    if (!FusedQuantGemm::ShouldUseFusedPath(
            plan.raw.quant_type,
            FusedDispatchGeometry{M, plan.output_cols, K, 1, true, has_norm})) {
      return false;
    }
    pair_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
        plan.raw.quant_type,
        FusedDispatchGeometry{M, plan.output_cols, K, 2, true, has_norm});
    if constexpr (GroupSize == 3) {
      triple_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
          plan.raw.quant_type,
          FusedDispatchGeometry{M, plan.output_cols, K, 3, true, has_norm});
    }
  }

  if (norm_weight) {
    if (!normalized) {
      return false;
    }
    if (cuda_kernel::RmsNorm<half>(residual, norm_weight, normalized, M, K, eps,
                                   stream) != cudaSuccess) {
      return false;
    }
    if (cuda_kernel::QuantizeRowsSymmetric(normalized, packed_activation,
                                           packed_scales, M, K,
                                           stream) != cudaSuccess) {
      return false;
    }
  } else {
    // Norm precomputed — residual already holds the normalized data
    if (cuda_kernel::QuantizeRowsSymmetric(residual, packed_activation,
                                           packed_scales, M, K,
                                           stream) != cudaSuccess) {
      return false;
    }
  }

  PackedActivationInfo packed;
  packed.data = packed_activation;
  packed.row_scales = packed_scales;

  const auto grouping = SelectSharedActivationGrouping(
      quant_types, output_cols, pair_ready, triple_ready);
  std::array<bool, GroupSize> completed{};
  bool used_grouped = false;

  if constexpr (GroupSize == 3) {
    if (grouping.grouped_count == 3) {
      const std::array<PackedProjectionSpec, 3> grouped = {{
          {plans[0].raw, plans[0].output, plans[0].output_cols},
          {plans[1].raw, plans[1].output, plans[1].output_cols},
          {plans[2].raw, plans[2].output, plans[2].output_cols},
      }};
      if (FusedQuantGemm::GemvPackedTriple(grouped, packed, M, K, stream)) {
        LogPackedGemmPath("packed_triple", "using packed grouped triple GEMV");
        return true;
      }
    }
  }

  if (grouping.grouped_count == 2) {
    const int i = grouping.indices[0];
    const int j = grouping.indices[1];
    const std::array<PackedProjectionSpec, 2> grouped = {{
        {plans[i].raw, plans[i].output, plans[i].output_cols},
        {plans[j].raw, plans[j].output, plans[j].output_cols},
    }};
    if (FusedQuantGemm::GemvPackedPair(grouped, packed, M, K, stream)) {
      completed[i] = true;
      completed[j] = true;
      used_grouped = true;
    }
  }

  for (size_t i = 0; i < GroupSize; ++i) {
    if (completed[i]) {
      continue;
    }
    const auto &plan = plans[i];
    if (!FusedQuantGemm::GemvPacked(plan.raw, packed, plan.output, M,
                                    plan.output_cols, K, stream, policy)) {
      return false;
    }
  }
  if constexpr (GroupSize > 1) {
    if (used_grouped && grouping.grouped_count < static_cast<int>(GroupSize)) {
      LogPackedGemmPath("packed_mixed", "using packed grouped+individual GEMV");
    } else if (used_grouped) {
      LogPackedGemmPath("packed_pair", "using packed grouped pair GEMV");
    } else {
      LogPackedGemmPath("packed_individual",
                        "using packed individual GEMV (shared quantization)");
    }
  }
  return true;
}

// Q8_1 pre-quantized projection group dispatch.
// Quantizes activations once (fused with RmsNorm when norm_weight != nullptr),
// then dispatches grouped or individual Q8_1 GEMV kernels.
// Q8_1 grouped projection dispatch: quantize activations once, then run
// grouped or individual MMVQ kernels.  Only the half (FP16) path has
// MMVQ kernels; other types return false.
template <size_t GroupSize, typename T>
bool TryQ8_1ProjectionGroup(
    const std::array<PackedProjectionPlan<T>, GroupSize> &plans, const T *input,
    const T *norm_weight, void *act_q8_1, int M, int K, float eps,
    cudaStream_t stream, const NativeExecutionPolicy *policy = nullptr,
    FusedQuantGemm::FfnProjOperator selected_op =
        FusedQuantGemm::FfnProjOperator::kFallback,
    bool q81_precomputed = false) {
  if constexpr (!std::is_same_v<T, half>) {
    (void)input;
    (void)norm_weight;
    (void)act_q8_1;
    (void)M;
    (void)K;
    (void)eps;
    (void)stream;
    (void)policy;
    (void)selected_op;
    return false;
  } else {
    const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
    if (Q81ActivationsDisabled(policy_ref) || !input || !act_q8_1 || M <= 0 ||
        K <= 0) {
      return false;
    }

    std::array<int, GroupSize> quant_types{};
    std::array<int, GroupSize> output_cols{};
    std::array<bool, GroupSize> pair_ready{};
    std::array<bool, GroupSize> triple_ready{};
    for (size_t i = 0; i < GroupSize; ++i) {
      const auto &plan = plans[i];
      if (!plan.raw.data || plan.output == nullptr || plan.output_cols <= 0 ||
          !FusedQuantGemm::SupportsQ8_1Activations(plan.raw.quant_type)) {
        return false;
      }
      quant_types[i] = plan.raw.quant_type;
      output_cols[i] = plan.output_cols;
      if (!FusedQuantGemm::ShouldUseFusedPath(
              plan.raw.quant_type,
              FusedDispatchGeometry{M, plan.output_cols, K, 1, true,
                                    norm_weight != nullptr})) {
        return false;
      }
      pair_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
          plan.raw.quant_type,
          FusedDispatchGeometry{M, plan.output_cols, K, 2, true,
                                norm_weight != nullptr});
      if constexpr (GroupSize == 3) {
        triple_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
            plan.raw.quant_type,
            FusedDispatchGeometry{M, plan.output_cols, K, 3, true,
                                  norm_weight != nullptr});
      }
    }

    // Quantize activations: fused RmsNorm+Quantize or standalone quantize.
    // Skipped when a producer epilogue already wrote act_q8_1 for this
    // exact input (P2 pattern) — the standalone re-quantize was measured
    // as 2 of the 5 per-layer-step quantize launches at c1.
    if (!q81_precomputed) {
      if (norm_weight) {
        FusedQuantGemm::FusedRmsNormQuantizeQ8_1(input, norm_weight, act_q8_1,
                                                 M, K, eps, stream);
      } else {
        FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
      }
    }

    const auto grouping = SelectSharedActivationGrouping(
        quant_types, output_cols, pair_ready, triple_ready);
    std::array<bool, GroupSize> completed{};
    bool used_grouped = false;

    if constexpr (GroupSize == 3) {
      if (grouping.grouped_count == 3) {
        const std::array<PackedProjectionSpec, 3> grouped = {{
            {plans[0].raw, plans[0].output, plans[0].output_cols},
            {plans[1].raw, plans[1].output, plans[1].output_cols},
            {plans[2].raw, plans[2].output, plans[2].output_cols},
        }};
        if (FusedQuantGemm::GemvQ8_1Triple(grouped, act_q8_1, M, K, stream,
                                           policy)) {
          LogPackedGemmPath("q8_1_triple", "using Q8_1 grouped triple GEMV");
          return true;
        }
        // Q8_1 triple requires all same quant type. For Q4_K_M models
        // where V uses Q6_K, this falls through to pair (Q+K) + individual (V).
      }
    }

    if (grouping.grouped_count == 2) {
      const int i = grouping.indices[0];
      const int j = grouping.indices[1];
      const std::array<PackedProjectionSpec, 2> grouped = {{
          {plans[i].raw, plans[i].output, plans[i].output_cols},
          {plans[j].raw, plans[j].output, plans[j].output_cols},
      }};
      if (FusedQuantGemm::GemvQ8_1Pair(grouped, act_q8_1, M, K, stream, policy,
                                       selected_op)) {
        completed[i] = true;
        completed[j] = true;
        used_grouped = true;
      }
    }

    for (size_t i = 0; i < GroupSize; ++i) {
      if (completed[i]) {
        continue;
      }
      const auto &plan = plans[i];
      if (!FusedQuantGemm::GemvQ8_1(plan.raw, act_q8_1, plan.output, M,
                                    plan.output_cols, K, stream, policy)) {
        return false;
      }
    }
    if constexpr (GroupSize > 1) {
      if (used_grouped &&
          grouping.grouped_count < static_cast<int>(GroupSize)) {
        LogPackedGemmPath("q8_1_mixed", "using Q8_1 grouped+individual GEMV");
      } else if (used_grouped) {
        LogPackedGemmPath("q8_1_pair", "using Q8_1 grouped pair GEMV");
      } else {
        LogPackedGemmPath("q8_1_individual",
                          "using Q8_1 individual GEMV (shared quantization)");
      }
    }
    return true;
  } // if constexpr (std::is_same_v<T, half>)
}

// Standalone Q8_1 GEMV for single projections without norm fusion.
template <typename T>
bool TryMmqGemv(const MmqWeightInfo &, const T *, T *, void *, int, int, int,
                cudaStream_t, const char * = nullptr,
                const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqGemv<half>(const MmqWeightInfo &weight, const half *input,
                      half *output, void *act_q8_1, int M, int N, int K,
                      cudaStream_t stream, const char *proj_name,
                      const NativeExecutionPolicy *policy) {
  if (!weight.data || !input || !output || !act_q8_1 || M <= 0 || N <= 0 ||
      K <= 0 || N != weight.rows || K != weight.cols) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  const bool ok = FusedQuantGemm::DownProjMmq(weight, act_q8_1, output, M, N, K,
                                              stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using MMQ-style tiled Q8_1 down-proj");
  }
  return ok;
}

template <typename T>
bool TryMmqSiluMulGemv(const MmqWeightInfo &, const T *, const T *, T *, void *,
                       int, int, int, cudaStream_t, const char * = nullptr,
                       const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqSiluMulGemv<half>(const MmqWeightInfo &weight, const half *gate,
                             const half *up, half *output, void *act_q8_1,
                             int M, int N, int K, cudaStream_t stream,
                             const char *proj_name,
                             const NativeExecutionPolicy *policy) {
  if (!weight.data || !gate || !up || !output || !act_q8_1 || M <= 0 ||
      N <= 0 || K <= 0 || N != weight.rows || K != weight.cols) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  const bool ok = FusedQuantGemm::DownProjMmq(weight, act_q8_1, output, M, N, K,
                                              stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using MMQ-style tiled fused SwiGLU down-proj");
  }
  return ok;
}

// MMA tensor-core down-proj (S7): Q6_K only, raw row-major weights, D4
// activations. Returns false (unchanged dispatch) unless
// INFERFLUX_CUDA_MMQ_MMA=1 — callers fall through to the dp4a MMQ tier.
template <typename T>
bool TryMmqMmaSiluMul(const QuantizedWeightInfo &, const T *, const T *, T *,
                      void *, float *, int, int, int, cudaStream_t,
                      const char * = nullptr,
                      const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqMmaSiluMul<half>(const QuantizedWeightInfo &weight, const half *gate,
                            const half *up, half *output, void *act_mmq,
                            float *partials, int M, int N, int K,
                            cudaStream_t stream, const char *proj_name,
                            const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_mmq_mma || !weight.data || !gate || !up || !output ||
      !act_mmq || !partials || !FusedQuantGemm::CurrentDeviceSupportsMmqMma() ||
      InferfluxCudaOperatorHealth::Instance().IsUnhealthy(
          FusedQuantGemm::DownProjOperator::kMmq) ||
      M <= 0 || N <= 0 || K <= 0 || M < policy_ref.mmq_mma_min_batch ||
      M > policy_ref.mmq_mma_max_batch ||
      static_cast<size_t>(N) * K != static_cast<size_t>(weight.num_elements) ||
      weight.quant_type !=
          static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K)) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1Mmq(
      gate, up, static_cast<runtime::cuda::native::BlockQ8_1Mmq *>(act_mmq), M,
      K, stream);
  const bool ok = FusedQuantGemm::DownProjMmqMma(
      weight, static_cast<const runtime::cuda::native::BlockQ8_1Mmq *>(act_mmq),
      output, M, N, K, partials, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using MMA tensor-core Q6_K down-proj");
  }
  return ok;
}

// MMA down-proj for already-activated input (the fused gate+up+SiLU
// kernel produced d_ffn_gate_): quantize to D4 then MMA. Q6_K only,
// INFERFLUX_CUDA_MMQ_MMA=1, M <= mmq_mma_max_batch.
template <typename T>
bool TryMmqMmaGemv(const QuantizedWeightInfo &, const T *, T *, void *, float *,
                   int, int, int, cudaStream_t, const char * = nullptr,
                   const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqMmaGemv<half>(const QuantizedWeightInfo &weight, const half *input,
                         half *output, void *act_mmq, float *partials, int M,
                         int N, int K, cudaStream_t stream,
                         const char *proj_name,
                         const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_mmq_mma || !weight.data || !input || !output ||
      !act_mmq || !partials || !FusedQuantGemm::CurrentDeviceSupportsMmqMma() ||
      InferfluxCudaOperatorHealth::Instance().IsUnhealthy(
          FusedQuantGemm::DownProjOperator::kMmq) ||
      M <= 0 || N <= 0 || K <= 0 || M < policy_ref.mmq_mma_min_batch ||
      M > policy_ref.mmq_mma_max_batch ||
      static_cast<size_t>(N) * K != static_cast<size_t>(weight.num_elements) ||
      weight.quant_type !=
          static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K)) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1Mmq(
      input, static_cast<runtime::cuda::native::BlockQ8_1Mmq *>(act_mmq), M, K,
      stream);
  const bool ok = FusedQuantGemm::DownProjMmqMma(
      weight, static_cast<const runtime::cuda::native::BlockQ8_1Mmq *>(act_mmq),
      output, M, N, K, partials, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using MMA tensor-core Q6_K down-proj");
  }
  return ok;
}

// Generic Q4_K MMA projection (S8): tensor-core path for gate/up/o and
// other single Q4_K projections at M >= 2. Falls through (unchanged
// dispatch) unless INFERFLUX_CUDA_MMQ_MMA=1.
template <typename T>
bool TryQ8_1MmaGemv(const QuantizedWeightInfo &, const T *, T *, void *,
                    float *, int, int, int, cudaStream_t, const char * = nullptr,
                    const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1MmaGemv<half>(const QuantizedWeightInfo &weight, const half *input,
                          half *output, void *ds_act, float *partials, int M,
                          int N, int K, cudaStream_t stream,
                          const char *proj_name,
                          const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  // M >= 2 for every projection family — measured three ways at c1:
  // QKV+o (-12%), gate/up+o (-4%). The M=1 incumbents' floor gaps
  // (fused gate/up 1.5x DRAM cold, o accum 3.5x) are real but smaller
  // than the masked-tile + per-projection quantize/reduce launch
  // overhead at M=1. Improving M=1 requires kernel work on the
  // incumbents, not dispatch changes.
  if (!policy_ref.enable_mmq_mma || M < 2) {
    return false;
  }
  const bool ok = FusedQuantGemm::GemvMmqMma(
      weight, input, output,
      static_cast<runtime::cuda::native::BlockQ8_1MmqDs *>(ds_act), partials,
      M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q4_K MMA projection");
  }
  return ok;
}

template <typename T>
bool TryQ8_1Gemv(const QuantizedWeightInfo &, const T *, T *, void *, int, int,
                 int, cudaStream_t, const char * = nullptr,
                 const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1Gemv<half>(const QuantizedWeightInfo &raw, const half *input,
                       half *output, void *act_q8_1, int M, int N, int K,
                       cudaStream_t stream, const char *proj_name,
                       const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  bool ok =
      FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 pre-quantized GEMV");
  }
  return ok;
}

// Accumulate-mode Q8_1 GEMV: output[i] += gemv(input)[i]
// Used for O-proj and down-proj to eliminate separate ResidualAdd.
template <typename T>
bool TryQ8_1GemvAccum(const QuantizedWeightInfo &, const T *, T *, void *, int,
                      int, int, cudaStream_t, const char * = nullptr,
                      const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1GemvAccum<half>(const QuantizedWeightInfo &raw, const half *input,
                            half *output, void *act_q8_1, int M, int N, int K,
                            cudaStream_t stream, const char *proj_name,
                            const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  bool ok = FusedQuantGemm::GemvQ8_1Accum(raw, act_q8_1, output, M, N, K,
                                          stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 accumulate GEMV");
  }
  return ok;
}

// FP32 accumulate: quantize half input, GEMV-accum into float* output.
// Used for O-proj and down-proj when FP32 residual stream is active.
bool TryQ8_1GemvAccumF32(const QuantizedWeightInfo &raw, const half *input,
                         float *output, void *act_q8_1, int M, int N, int K,
                         cudaStream_t stream, const char *proj_name = nullptr,
                         const NativeExecutionPolicy *policy = nullptr) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  bool ok = FusedQuantGemm::GemvQ8_1AccumF32(raw, act_q8_1, output, M, N, K,
                                             stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 FP32 accumulate GEMV");
  }
  return ok;
}

template <typename T>
bool TryQ8_1GemvPrequantized(const QuantizedWeightInfo &, const void *, T *,
                             int, int, int, cudaStream_t,
                             const char * = nullptr,
                             const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1GemvPrequantized<half>(const QuantizedWeightInfo &raw,
                                   const void *act_q8_1, half *output, int M,
                                   int N, int K, cudaStream_t stream,
                                   const char *proj_name,
                                   const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !raw.data || !act_q8_1 || !output ||
      M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  const bool ok =
      FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using pre-quantized Q8_1 GEMV from epilogue");
  }
  return ok;
}

template <typename T>
bool TryQ8_1GemvPrequantizedAccum(const QuantizedWeightInfo &, const void *,
                                  T *, int, int, int, cudaStream_t,
                                  const char * = nullptr,
                                  const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1GemvPrequantizedAccum<half>(const QuantizedWeightInfo &raw,
                                        const void *act_q8_1, half *output,
                                        int M, int N, int K,
                                        cudaStream_t stream,
                                        const char *proj_name,
                                        const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !act_q8_1 || !output ||
      M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  const bool ok = FusedQuantGemm::GemvQ8_1Accum(raw, act_q8_1, output, M, N, K,
                                                stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using pre-quantized Q8_1 accumulate GEMV from epilogue");
  }
  return ok;
}

// FP32 variant of pre-quantized accumulate
bool TryQ8_1GemvPrequantizedAccumF32(
    const QuantizedWeightInfo &raw, const void *act_q8_1, float *output, int M,
    int N, int K, cudaStream_t stream, const char *proj_name = nullptr,
    const NativeExecutionPolicy *policy = nullptr) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !act_q8_1 || !output ||
      M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  const bool ok = FusedQuantGemm::GemvQ8_1AccumF32(raw, act_q8_1, output, M, N,
                                                   K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using pre-quantized Q8_1 FP32 accumulate GEMV");
  }
  return ok;
}

// Accumulate-mode Q8_1 SiLU+Mul+GEMV for down-proj: residual +=
// down(silu(gate)*up)
template <typename T>
bool TryQ8_1SiluMulGemvAccum(const QuantizedWeightInfo &, const T *, const T *,
                             T *, void *, int, int, int, cudaStream_t,
                             const char * = nullptr,
                             const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1SiluMulGemvAccum<half>(const QuantizedWeightInfo &raw,
                                   const half *gate, const half *up,
                                   half *output, void *act_q8_1, int M, int N,
                                   int K, cudaStream_t stream,
                                   const char *proj_name,
                                   const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  bool ok = FusedQuantGemm::GemvQ8_1Accum(raw, act_q8_1, output, M, N, K,
                                          stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 accumulate SiLU+Mul+GEMV");
  }
  return ok;
}

// FP32 accumulate SiLU+Mul+GEMV: silu(gate)*up → Q8_1 → GEMV-accum into float*
bool TryQ8_1SiluMulGemvAccumF32(const QuantizedWeightInfo &raw,
                                const half *gate, const half *up, float *output,
                                void *act_q8_1, int M, int N, int K,
                                cudaStream_t stream,
                                const char *proj_name = nullptr,
                                const NativeExecutionPolicy *policy = nullptr) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (!policy_ref.enable_gemv_accumulate ||
      Q81ActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  bool ok = FusedQuantGemm::GemvQ8_1AccumF32(raw, act_q8_1, output, M, N, K,
                                             stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 FP32 accumulate SiLU+Mul+GEMV");
  }
  return ok;
}

template <typename T>
bool TryMmqGemvPrequantized(const MmqWeightInfo &, const void *, T *, int, int,
                            int, cudaStream_t, const char * = nullptr,
                            const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqGemvPrequantized<half>(const MmqWeightInfo &weight,
                                  const void *act_q8_1, half *output, int M,
                                  int N, int K, cudaStream_t stream,
                                  const char *proj_name,
                                  const NativeExecutionPolicy *policy) {
  if (!weight.data || !act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0 ||
      N != weight.rows || K != weight.cols) {
    return false;
  }

  const bool ok = FusedQuantGemm::DownProjMmq(weight, act_q8_1, output, M, N, K,
                                              stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using MMQ-style tiled Q8_1 down-proj from epilogue");
  }
  return ok;
}

template <typename T>
bool TryPackedSiluMulGemv(const QuantizedWeightInfo &, const T *, const T *,
                          T *, int8_t *, float *, int, int, int, cudaStream_t,
                          const char * = nullptr,
                          const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedSiluMulGemv<half>(const QuantizedWeightInfo &raw,
                                const half *gate, const half *up, half *output,
                                int8_t *packed_activation, float *packed_scales,
                                int M, int N, int K, cudaStream_t stream,
                                const char *proj_name,
                                const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !packed_activation || !packed_scales || M <= 0 || N <= 0 ||
      K <= 0 || !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  if (cuda_kernel::SiluMulQuantizeRowsSymmetric(gate, up, packed_activation,
                                                packed_scales, M, K,
                                                stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using fused SwiGLU + packed-activation GEMV");
  }
  return ok;
}

template <typename T>
bool TryQ8_1SiluMulGemv(const QuantizedWeightInfo &, const T *, const T *, T *,
                        void *, int, int, int, cudaStream_t,
                        const char * = nullptr,
                        const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1SiluMulGemv<half>(const QuantizedWeightInfo &raw, const half *gate,
                              const half *up, half *output, void *act_q8_1,
                              int M, int N, int K, cudaStream_t stream,
                              const char *proj_name,
                              const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveInferfluxCudaExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  const bool ok =
      FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using fused SwiGLU + Q8_1 GEMV");
  }
  return ok;
}

} // namespace

template <typename T> LlamaForwardTyped<T>::~LlamaForwardTyped() {
  FreeScratchBuffers();
}

template <typename T> bool LlamaForwardTyped<T>::AllocateScratch() {
  device_workspace_bytes_ = 0;
  host_workspace_bytes_ = 0;
  auto alloc = [&](T **ptr, size_t count) -> bool {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    if (err == cudaSuccess) {
      device_workspace_bytes_ += count * sizeof(T);
    }
    return err == cudaSuccess;
  };
  cudaError_t err;

  // Scratch buffers must fit both:
  //   - Single long sequence: max_seq_len_ tokens (prefill)
  //   - Batched decode: max_batch_size_ sequences x 1 token each
  size_t rows = static_cast<size_t>(std::max(max_seq_len_, max_batch_size_));

  // Always-live buffers (cannot alias).
  if (!alloc(&d_hidden_, rows * hidden_size_))
    return false;
  if (!alloc(&d_residual_, rows * hidden_size_))
    return false;
  if (!alloc(&d_norm_out_, rows * hidden_size_))
    return false;
  // Memory pool: attention and FFN buffers never overlap in the layer loop.
  // Attention phase uses d_q_, d_k_new_, d_v_new_, d_attn_out_.
  // FFN phase uses d_ffn_gate_, d_ffn_up_, d_ffn_down_.
  // We allocate the larger set (FFN gate+up) and alias attention into it.
  //
  //   d_ffn_gate_ [rows * intermediate_size]:
  //     ├── d_q_      [rows * num_heads * head_dim]     (aliased)
  //     ├── d_k_new_  [rows * num_kv_heads * head_dim]  (aliased, offset)
  //     └── d_v_new_  [rows * num_kv_heads * head_dim]  (aliased, offset)
  //   d_ffn_up_   [rows * intermediate_size]:
  //     └── d_attn_out_ [rows * num_heads * head_dim]   (aliased)
  //   d_ffn_down_ = d_norm_out_ (aliased, same size: rows * hidden_size)
  //     d_norm_out_ is consumed before down-proj writes.

  const size_t q_elems = rows * static_cast<size_t>(num_heads_ * head_dim_);
  const size_t k_elems = rows * static_cast<size_t>(num_kv_heads_ * head_dim_);
  const size_t v_elems = k_elems;
  const size_t attn_out_elems = q_elems;
  const size_t gate_elems = rows * static_cast<size_t>(intermediate_size_);
  const size_t up_elems = gate_elems;

  // Verify aliasing fits: attention buffers must fit inside FFN buffers.
  if (q_elems + k_elems + v_elems > gate_elems || attn_out_elems > up_elems) {
    // Fallback: allocate separately (no aliasing possible for this model).
    if (!alloc(&d_q_, q_elems))
      return false;
    if (!alloc(&d_k_new_, k_elems))
      return false;
    if (!alloc(&d_v_new_, v_elems))
      return false;
    if (!alloc(&d_attn_out_, attn_out_elems))
      return false;
    if (!alloc(&d_ffn_gate_, gate_elems))
      return false;
    if (!alloc(&d_ffn_up_, up_elems))
      return false;
    if (!alloc(&d_ffn_down_, rows * hidden_size_))
      return false;
  } else {
    // Allocate FFN gate+up (the larger pair) and alias attention into them.
    if (!alloc(&d_ffn_gate_, gate_elems))
      return false;
    if (!alloc(&d_ffn_up_, up_elems))
      return false;
    // Alias attention buffers into d_ffn_gate_ (contiguous sub-regions).
    d_q_ = d_ffn_gate_;
    d_k_new_ = d_ffn_gate_ + q_elems;
    d_v_new_ = d_ffn_gate_ + q_elems + k_elems;
    // Alias d_attn_out_ into d_ffn_up_.
    d_attn_out_ = d_ffn_up_;
    // Alias d_ffn_down_ into d_norm_out_ (same size, non-overlapping lifetime).
    d_ffn_down_ = d_norm_out_;
    aliased_attn_ffn_ = true;
  }

  // Split-attention partials: [max_batch][num_heads][kMaxAttnKSplits]
  // [2 + head_dim] floats (identity-partial merge state per KV chunk).
  {
    const size_t floats = static_cast<size_t>(max_batch_size_) * num_heads_ *
                          kMaxAttnKSplits * (2 + head_dim_);
    err = cudaMalloc(&d_attn_partials_, floats * sizeof(float));
    if (err != cudaSuccess) {
      return false;
    }
    device_workspace_bytes_ += floats * sizeof(float);
  }
  const size_t packed_width = static_cast<size_t>(
      PackedActivationWidth(hidden_size_, intermediate_size_));
  err = cudaMalloc(&d_packed_activation_, rows * packed_width * sizeof(int8_t));
  if (err != cudaSuccess)
    return false;
  device_workspace_bytes_ += rows * packed_width * sizeof(int8_t);
  err = cudaMalloc(&d_packed_activation_scales_, rows * sizeof(float));
  if (err != cudaSuccess)
    return false;
  device_workspace_bytes_ += rows * sizeof(float);
  // Pre-quantized Q8_1 activation buffer: ceil(max_dim/32) blocks per row
  {
    const int max_dim = std::max(hidden_size_, intermediate_size_);
    const size_t blocks_per_row = (max_dim + 31) / 32;
    // block_q8_1 = 36 bytes (sizeof(half2) + 32)
    const size_t q8_1_block_size = 36;
    err = cudaMalloc(&d_act_q8_1_, rows * blocks_per_row * q8_1_block_size);
    if (err != cudaSuccess)
      return false;
    device_workspace_bytes_ += rows * blocks_per_row * q8_1_block_size;
  }
  {
    const size_t blocks_per_row =
        (static_cast<size_t>(intermediate_size_) + 31) / 32;
    const size_t q8_1_block_size = 36;
    err = cudaMalloc(&d_ffn_act_q8_1_, rows * blocks_per_row * q8_1_block_size);
    if (err != cudaSuccess)
      return false;
    device_workspace_bytes_ += rows * blocks_per_row * q8_1_block_size;
  }
  // MMA down-proj (S7) buffers. This tier is opt-in and rejects rows above
  // mmq_mma_max_batch, so do not charge every model for max-sequence-sized
  // scratch while the feature is disabled (or for rows it cannot consume).
  if (execution_policy_.enable_mmq_mma &&
      FusedQuantGemm::CurrentDeviceSupportsMmqMma()) {
    using namespace runtime::cuda::native;
    const size_t mma_rows = std::min(
        rows, static_cast<size_t>(execution_policy_.mmq_mma_max_batch));
    const size_t blocks_per_row =
        (static_cast<size_t>(intermediate_size_) + 127) / 128;
    err = cudaMalloc(&d_act_q8_1_mmq_,
                     blocks_per_row * mma_rows * sizeof(BlockQ8_1Mmq));
    if (err != cudaSuccess)
      return false;
    device_workspace_bytes_ += blocks_per_row * mma_rows * sizeof(BlockQ8_1Mmq);
    err = cudaMalloc(&d_mma_partials_, static_cast<size_t>(kMmqMmaMaxSplits) *
                                           mma_rows * hidden_size_ *
                                           sizeof(float));
    if (err != cudaSuccess)
      return false;
    device_workspace_bytes_ += static_cast<size_t>(kMmqMmaMaxSplits) *
                               mma_rows * hidden_size_ * sizeof(float);
  }
  // Logits buffer sized for batched decode: [max_batch_size, vocab_size]
  if (!alloc(&d_logits_typed_,
             static_cast<size_t>(max_batch_size_) * vocab_size_))
    return false;

  err = cudaMalloc(&d_token_ids_, rows * sizeof(int));
  if (err != cudaSuccess)
    return false;
  device_workspace_bytes_ += rows * sizeof(int);

  // Batch metadata buffers for batched decode
  size_t bsz = static_cast<size_t>(max_batch_size_);
  const size_t batch_meta_elements = bsz * 4;
  err = cudaMalloc(&d_batch_meta_, batch_meta_elements * sizeof(int));
  if (err != cudaSuccess)
    return false;
  device_workspace_bytes_ += batch_meta_elements * sizeof(int);
  AssignBatchMetadataViews(d_batch_meta_, bsz, &d_batch_token_ids_,
                           &d_batch_n_past_, &d_batch_seq_ids_,
                           &d_batch_kv_lens_);
  if (!AllocatePinnedHostBuffer(&h_batch_meta_, batch_meta_elements))
    return false;
  host_workspace_bytes_ += batch_meta_elements * sizeof(int);
  AssignBatchMetadataViews(h_batch_meta_, bsz, &h_batch_token_ids_,
                           &h_batch_n_past_, &h_batch_seq_ids_,
                           &h_batch_kv_lens_);

  // FlashDecode KV-split workspace for parallel attention.
  // Layout: partial_O [B * num_kv_heads * splits * gqa_ratio * head_dim] float
  //       + partial_max [B * num_kv_heads * splits * gqa_ratio] float
  //       + partial_sum [B * num_kv_heads * splits * gqa_ratio] float
  {
    constexpr int kSplits =
        8; // must match kFlashDecodeSplits in flash_attention.cu
    const int gqa_ratio = (num_kv_heads_ > 0 && num_heads_ > num_kv_heads_)
                              ? (num_heads_ / num_kv_heads_)
                              : 1;
    const size_t split_entries =
        bsz * static_cast<size_t>(num_kv_heads_) * kSplits * gqa_ratio;
    const size_t partial_O_bytes = split_entries * head_dim_ * sizeof(float);
    const size_t partial_scalar_bytes = split_entries * sizeof(float);
    attn_split_workspace_bytes_ = partial_O_bytes + 2 * partial_scalar_bytes;
    err = cudaMalloc(&d_attn_split_workspace_, attn_split_workspace_bytes_);
    if (err != cudaSuccess) {
      // Non-fatal: fall back to unsplit attention
      d_attn_split_workspace_ = nullptr;
      attn_split_workspace_bytes_ = 0;
    } else {
      device_workspace_bytes_ += attn_split_workspace_bytes_;
    }
  }

  // FP32 residual stream: avoids FP16 quantization error compounding across
  // layers during multi-token generation.  Same size as d_residual_ but 4
  // bytes per element instead of 2.
  if (execution_policy_.enable_fp32_residual) {
    const size_t f32_bytes = rows * hidden_size_ * sizeof(float);
    err = cudaMalloc(&d_residual_f32_, f32_bytes);
    if (err != cudaSuccess) {
      log::Warn("llama_forward", "FP32 residual alloc failed (" +
                                     std::to_string(f32_bytes) +
                                     " bytes), falling back to FP16 residual");
      d_residual_f32_ = nullptr;
      fp32_residual_active_ = false;
    } else {
      // Initialize to zero to prevent garbage accumulation in FP32 kernels
      err = cudaMemset(d_residual_f32_, 0, f32_bytes);
      if (err != cudaSuccess) {
        log::Warn("llama_forward",
                  "FP32 residual memset failed (" + std::to_string(f32_bytes) +
                      " bytes), falling back to FP16 residual");
        cudaFree(d_residual_f32_);
        d_residual_f32_ = nullptr;
        fp32_residual_active_ = false;
      } else {
        fp32_residual_active_ = true;
        device_workspace_bytes_ += f32_bytes;
        log::Info("llama_forward", "FP32 residual enabled (" +
                                       std::to_string(f32_bytes / 1024) +
                                       " KB)");
      }
    }
  }

  return true;
}

template <typename T> void LlamaForwardTyped<T>::FreeScratchBuffers() {
  auto free_buf = [](T **ptr) {
    if (*ptr) {
      cudaFree(*ptr);
      *ptr = nullptr;
    }
  };

  free_buf(&d_hidden_);
  free_buf(&d_residual_);
  if (aliased_attn_ffn_) {
    // Attention buffers alias d_ffn_gate_/d_ffn_up_ — only free the FFN pair.
    // d_ffn_down_ aliases d_norm_out_ — only free d_norm_out_.
    d_q_ = nullptr;
    d_k_new_ = nullptr;
    d_v_new_ = nullptr;
    d_attn_out_ = nullptr;
    d_ffn_down_ = nullptr;
    free_buf(&d_ffn_gate_);
    free_buf(&d_ffn_up_);
    free_buf(&d_norm_out_);
  } else {
    free_buf(&d_norm_out_);
    free_buf(&d_q_);
    free_buf(&d_k_new_);
    free_buf(&d_v_new_);
    free_buf(&d_attn_out_);
    free_buf(&d_ffn_gate_);
    free_buf(&d_ffn_up_);
    free_buf(&d_ffn_down_);
  }
  if (d_attn_partials_) {
    cudaFree(d_attn_partials_);
    d_attn_partials_ = nullptr;
  }
  free_buf(&d_logits_typed_);
  if (d_packed_activation_) {
    cudaFree(d_packed_activation_);
    d_packed_activation_ = nullptr;
  }
  if (d_packed_activation_scales_) {
    cudaFree(d_packed_activation_scales_);
    d_packed_activation_scales_ = nullptr;
  }
  if (d_act_q8_1_) {
    cudaFree(d_act_q8_1_);
    d_act_q8_1_ = nullptr;
  }
  if (d_ffn_act_q8_1_) {
    cudaFree(d_ffn_act_q8_1_);
    d_ffn_act_q8_1_ = nullptr;
  }
  if (d_attn_split_workspace_) {
    cudaFree(d_attn_split_workspace_);
    d_attn_split_workspace_ = nullptr;
    attn_split_workspace_bytes_ = 0;
  }
  if (d_residual_f32_) {
    cudaFree(d_residual_f32_);
    d_residual_f32_ = nullptr;
    fp32_residual_active_ = false;
  }
  if (d_act_q8_1_mmq_) {
    cudaFree(d_act_q8_1_mmq_);
    d_act_q8_1_mmq_ = nullptr;
  }
  if (d_mma_partials_) {
    cudaFree(d_mma_partials_);
    d_mma_partials_ = nullptr;
  }

  if (d_token_ids_) {
    cudaFree(d_token_ids_);
    d_token_ids_ = nullptr;
  }

  if (d_batch_meta_) {
    cudaFree(d_batch_meta_);
    d_batch_meta_ = nullptr;
  }
  d_batch_token_ids_ = nullptr;
  d_batch_n_past_ = nullptr;
  d_batch_seq_ids_ = nullptr;
  d_batch_kv_lens_ = nullptr;
  if (h_batch_meta_) {
    FreePinnedHostBuffer(&h_batch_meta_);
  }
  h_batch_token_ids_ = nullptr;
  h_batch_n_past_ = nullptr;
  h_batch_seq_ids_ = nullptr;
  h_batch_kv_lens_ = nullptr;

  if (decode_graph_exec_) {
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
  }
  if (decode_graph_) {
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
  }
  device_workspace_bytes_ = 0;
  host_workspace_bytes_ = 0;
}

template <typename T>
bool LlamaForwardTyped<T>::Initialize(
    const SafetensorsLoader::ModelConfig &config, const WeightMap &weights,
    IKvCacheGpu *kv_cache, CublasGemm *gemm, cudaStream_t stream) {
  hidden_size_ = config.hidden_size;
  num_layers_ = config.num_hidden_layers;
  num_heads_ = config.num_attention_heads;
  num_kv_heads_ = config.num_key_value_heads;
  head_dim_ = config.head_dim;
  intermediate_size_ = config.intermediate_size;
  vocab_size_ = config.vocab_size;
  max_seq_len_ = config.max_position_embeddings;
  rope_freq_base_ = config.rope_freq_base;
  rms_norm_eps_ = config.rms_norm_eps;
  rope_type_ =
      static_cast<int>(runtime::cuda::native::InferRopeType(config.model_type));

  if (max_seq_len_ > 4096) {
    max_seq_len_ = 4096;
  }

  weights_ = &weights;
  kv_cache_ = kv_cache;
  gemm_ = gemm;
  stream_ = stream;

  // Check if attention tensor capture is enabled for debugging
  const char *capture_env = std::getenv("INFERFLUX_DEBUG_ATTENTION_TENSORS");
  if (capture_env && std::string(capture_env) == "1") {
    capture_attention_tensors_ = true;
    log::Info("llama_forward", "Attention tensor capture enabled - performance "
                               "will be severely degraded");
  }

  // Match scratch buffer sizing to KV cache limits (not model max)
  if (kv_cache) {
    max_batch_size_ = kv_cache->MaxBatchSize();
    if (kv_cache->MaxSeqLen() > 0 && kv_cache->MaxSeqLen() < max_seq_len_) {
      max_seq_len_ = kv_cache->MaxSeqLen();
    }
  }

  if (!AllocateScratch()) {
    log::Error("llama_forward", "Failed to allocate scratch buffers");
    FreeScratchBuffers();
    return false;
  }

  log::Info("llama_forward",
            "Initialized (" + std::string(DtypeTraits<T>::name) +
                "): hidden=" + std::to_string(hidden_size_) +
                ", layers=" + std::to_string(num_layers_) +
                ", heads=" + std::to_string(num_heads_) + "/" +
                std::to_string(num_kv_heads_) +
                ", head_dim=" + std::to_string(head_dim_) +
                ", vocab=" + std::to_string(vocab_size_) +
                ", max_seq=" + std::to_string(max_seq_len_) +
                ", rope_type=" + (rope_type_ == 2 ? "neox" : "norm") +
                ", model=" + config.model_type);
  return true;
}

template <typename T>
bool LlamaForwardTyped<T>::Forward(const std::vector<int> &token_ids,
                                   int n_past, int sequence_id,
                                   float *d_logits) {
  NVTX_SCOPE("Forward");
  int seq_len = static_cast<int>(token_ids.size());
  if (seq_len == 0)
    return false;
  if (seq_len > max_seq_len_) {
    log::Error("llama_forward", "seq_len " + std::to_string(seq_len) +
                                    " exceeds max " +
                                    std::to_string(max_seq_len_));
    return false;
  }
  const bool allow_fused_quantized_matmul =
      !weights_ || weights_->AllowFusedQuantizedMatmul();
  ScopedFusedMatmulPolicy fused_policy(allow_fused_quantized_matmul);

  int kv_len = n_past + seq_len;
  cudaError_t err;
  PhaseTiming pt;
  pt.Begin(stream_, execution_policy_.phase_timing_enabled);

  // Step 1: Upload token_ids to GPU
  err = runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kForwardTokenIdsH2D, d_token_ids_,
      token_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "Failed to upload token_ids");
    return false;
  }

  // Step 2: Embedding lookup
  // WeightMap is always WeightMapTyped<half> currently, but the embed_tokens
  // pointer points to the same GPU data regardless of type. We cast it.
  const T *embed = reinterpret_cast<const T *>(weights_->EmbedTokens());
  // Embed directly to residual stream, eliminating the D2D copy.
  {
    NVTX_SCOPE("Embedding");
    if (fp32_residual_active_) {
      err = cuda_kernel::EmbeddingLookupF32<T>(
          embed, d_token_ids_, d_residual_f32_, seq_len, hidden_size_, stream_);
    } else {
      err = cuda_kernel::EmbeddingLookup<T>(embed, d_token_ids_, d_residual_,
                                            seq_len, hidden_size_, stream_);
    }
  }
  if (err != cudaSuccess) {
    log::Error("llama_forward", "EmbeddingLookup failed");
    return false;
  }

  DebugDumpHidden("after_embedding", d_residual_, seq_len * hidden_size_,
                  stream_, &execution_policy_);
  pt.embed_ms += pt.Mark();

  // Step 4: Transformer layers
  bool input_norm_precomputed = false;
  for (int layer = 0; layer < num_layers_; layer++) {
    NVTX_SCOPE("Layer");
    // Norm weights are small (F32/F16), always fetch eagerly
    const T *input_norm =
        reinterpret_cast<const T *>(weights_->LayerInputNorm(layer));
    const T *post_attn_norm =
        reinterpret_cast<const T *>(weights_->LayerPostAttnNorm(layer));

    // 4a-d: RMSNorm + Q/K/V projections + optional bias
    // When inter-layer ResidualAddRmsNorm pre-computed the input norm,
    // d_norm_out_ already has the normalized result — skip the norm step.
    // FP32 residual: pre-normalize from FP32 → half d_norm_out_ so all
    // downstream projection paths see pre-computed half input.
    if (fp32_residual_active_ && !input_norm_precomputed) {
      err = cuda_kernel::RmsNormMixed<T>(d_residual_f32_, input_norm,
                                         d_norm_out_, seq_len, hidden_size_,
                                         rms_norm_eps_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward",
                   "FP32 RmsNormMixed (pre-attn) failed at layer " +
                       std::to_string(layer));
        return false;
      }
      input_norm_precomputed = true;
    }
    const T *qkv_input = input_norm_precomputed
                             ? static_cast<const T *>(d_norm_out_)
                             : static_cast<const T *>(d_residual_);
    const T *qkv_norm = input_norm_precomputed ? nullptr : input_norm;
    input_norm_precomputed = false;
    {
      NVTX_SCOPE("QKV_Projection");
      auto q_raw = weights_->LayerQProjRaw(layer);
      auto k_raw = weights_->LayerKProjRaw(layer);
      auto v_raw = weights_->LayerVProjRaw(layer);
      const std::array<PackedProjectionPlan<T>, 3> qkv_plans = {
          {{q_raw, d_q_, num_heads_ * head_dim_},
           {k_raw, d_k_new_, num_kv_heads_ * head_dim_},
           {v_raw, d_v_new_, num_kv_heads_ * head_dim_}}};
      // Priority: Q8_1 > packed per-row > fused RmsNorm+GEMV > cuBLAS
      if (!ExecuteNativeGroupedProjectionStage(
              [&]() {
                return TryQ8_1ProjectionGroup(
                    qkv_plans, qkv_input, qkv_norm, d_act_q8_1_, seq_len,
                    hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
              },
              [&]() {
                return TryPackedProjectionGroup(
                    qkv_plans, qkv_input, qkv_norm, d_norm_out_,
                    d_packed_activation_, d_packed_activation_scales_, seq_len,
                    hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
              },
              [&]() {
                bool norm_computed = (qkv_norm == nullptr);

                if (!ExecuteNativeNormalizedProjectionStage(
                        &norm_computed,
                        [&]() {
                          err = cuda_kernel::RmsNorm<T>(
                              d_residual_, input_norm, d_norm_out_, seq_len,
                              hidden_size_, rms_norm_eps_, stream_);
                          if (err != cudaSuccess) {
                            log::Error("llama_forward",
                                       "RmsNorm failed at layer " +
                                           std::to_string(layer));
                            return false;
                          }
                          return true;
                        },
                        [&]() {
                          return TryQ8_1Gemv<T>(
                              q_raw, d_norm_out_, d_q_, d_act_q8_1_, seq_len,
                              num_heads_ * head_dim_, hidden_size_, stream_,
                              "q_proj", &execution_policy_);
                        },
                        [&]() {
                          const T *q_proj = reinterpret_cast<const T *>(
                              weights_->LayerQProj(layer));
                          if (!gemm_->GemmTyped<T>(
                                  seq_len, num_heads_ * head_dim_, hidden_size_,
                                  d_norm_out_, q_proj, d_q_)) {
                            log::Error("llama_forward", "Q projection failed");
                            return false;
                          }
                          return true;
                        })) {
                  return false;
                }

                if (!ExecuteNativeNormalizedProjectionStage(
                        &norm_computed,
                        [&]() {
                          err = cuda_kernel::RmsNorm<T>(
                              d_residual_, input_norm, d_norm_out_, seq_len,
                              hidden_size_, rms_norm_eps_, stream_);
                          if (err != cudaSuccess) {
                            log::Error("llama_forward",
                                       "RmsNorm failed at layer " +
                                           std::to_string(layer));
                            return false;
                          }
                          return true;
                        },
                        [&]() {
                          return TryQ8_1MmaGemv<T>(
                                     k_raw, d_norm_out_, d_k_new_,
                                     d_act_q8_1_mmq_, d_mma_partials_,
                                     seq_len, num_kv_heads_ * head_dim_,
                                     hidden_size_, stream_, "k_proj",
                                     &execution_policy_) ||
                                 TryQ8_1Gemv<T>(
                                     k_raw, d_norm_out_, d_k_new_,
                                     d_act_q8_1_, seq_len,
                                     num_kv_heads_ * head_dim_, hidden_size_,
                                     stream_, "k_proj", &execution_policy_);
                        },
                        [&]() {
                          const T *k_proj = reinterpret_cast<const T *>(
                              weights_->LayerKProj(layer));
                          if (!gemm_->GemmTyped<T>(seq_len,
                                                   num_kv_heads_ * head_dim_,
                                                   hidden_size_, d_norm_out_,
                                                   k_proj, d_k_new_)) {
                            log::Error("llama_forward", "K projection failed");
                            return false;
                          }
                          return true;
                        })) {
                  return false;
                }

                if (!ExecuteNativeNormalizedProjectionStage(
                        &norm_computed,
                        [&]() {
                          err = cuda_kernel::RmsNorm<T>(
                              d_residual_, input_norm, d_norm_out_, seq_len,
                              hidden_size_, rms_norm_eps_, stream_);
                          if (err != cudaSuccess) {
                            log::Error("llama_forward",
                                       "RmsNorm failed at layer " +
                                           std::to_string(layer));
                            return false;
                          }
                          return true;
                        },
                        [&]() {
                          return TryQ8_1MmaGemv<T>(
                                     v_raw, d_norm_out_, d_v_new_,
                                     d_act_q8_1_mmq_, d_mma_partials_,
                                     seq_len, num_kv_heads_ * head_dim_,
                                     hidden_size_, stream_, "v_proj",
                                     &execution_policy_) ||
                                 TryQ8_1Gemv<T>(
                                     v_raw, d_norm_out_, d_v_new_,
                                     d_act_q8_1_, seq_len,
                                     num_kv_heads_ * head_dim_, hidden_size_,
                                     stream_, "v_proj", &execution_policy_);
                        },
                        [&]() {
                          const T *v_proj = reinterpret_cast<const T *>(
                              weights_->LayerVProj(layer));
                          if (!gemm_->GemmTyped<T>(seq_len,
                                                   num_kv_heads_ * head_dim_,
                                                   hidden_size_, d_norm_out_,
                                                   v_proj, d_v_new_)) {
                            log::Error("llama_forward", "V projection failed");
                            return false;
                          }
                          return true;
                        })) {
                  return false;
                }
                return true;
              })) {
        return false;
      }

      // Add biases if present (Qwen2 has q/k/v biases)
      const T *q_bias =
          reinterpret_cast<const T *>(weights_->LayerQProjBias(layer));
      const T *k_bias =
          reinterpret_cast<const T *>(weights_->LayerKProjBias(layer));
      const T *v_bias =
          reinterpret_cast<const T *>(weights_->LayerVProjBias(layer));
      if (q_bias && k_bias && v_bias &&
          execution_policy_.enable_fused_bias_add) {
        // Fused triple bias add: 1 kernel instead of 3
        err = cuda_kernel::BiasAddTriple<T>(
            d_q_, d_k_new_, d_v_new_, q_bias, k_bias, v_bias, seq_len,
            num_heads_ * head_dim_, num_kv_heads_ * head_dim_,
            num_kv_heads_ * head_dim_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "Fused QKV bias add failed");
          return false;
        }
      } else {
        if (q_bias) {
          err = cuda_kernel::BiasAdd<T>(d_q_, q_bias, seq_len,
                                        num_heads_ * head_dim_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "Q bias add failed");
            return false;
          }
        }
        if (k_bias) {
          err = cuda_kernel::BiasAdd<T>(d_k_new_, k_bias, seq_len,
                                        num_kv_heads_ * head_dim_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "K bias add failed");
            return false;
          }
        }
        if (v_bias) {
          err = cuda_kernel::BiasAdd<T>(d_v_new_, v_bias, seq_len,
                                        num_kv_heads_ * head_dim_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "V bias add failed");
            return false;
          }
        }
      }
    }

    pt.qkv_ms += pt.Mark();

    if (layer == 0) {
      DebugDumpHidden("layer0_after_q_proj", d_q_,
                      seq_len * num_heads_ * head_dim_, stream_,
                      &execution_policy_);
      DebugDumpHidden("layer0_after_k_proj", d_k_new_,
                      seq_len * num_kv_heads_ * head_dim_, stream_,
                      &execution_policy_);
    }

    // Capture QKV tensors for debugging
    if (capture_attention_tensors_) {
      SnapshotTensorToDevice("qkv_q", layer, d_q_,
                             {seq_len, num_heads_, head_dim_});
      SnapshotTensorToDevice("qkv_k", layer, d_k_new_,
                             {seq_len, num_kv_heads_, head_dim_});
      SnapshotTensorToDevice("qkv_v", layer, d_v_new_,
                             {seq_len, num_kv_heads_, head_dim_});
    }

    // 4e: RoPE in-place
    {
      NVTX_SCOPE("RoPE");
      err = cuda_kernel::RoPE<T>(d_q_, d_k_new_, seq_len, num_heads_,
                                 num_kv_heads_, head_dim_, n_past,
                                 rope_freq_base_, stream_, rope_type_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "RoPE failed");
        return false;
      }
    }

    pt.rope_ms += pt.Mark();

    // Capture Q,K after RoPE for debugging
    if (capture_attention_tensors_) {
      SnapshotTensorToDevice("rope_q", layer, d_q_,
                             {seq_len, num_heads_, head_dim_});
      SnapshotTensorToDevice("rope_k", layer, d_k_new_,
                             {seq_len, num_kv_heads_, head_dim_});
    }

    // 4f: KV cache append (works for both dense and hybrid KV cache)
    {
      NVTX_SCOPE("KV_Append");
      err = kv_cache_->AppendVoid(layer, sequence_id, n_past, seq_len, d_k_new_,
                                  d_v_new_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "KV cache append failed");
        return false;
      }
    }

    pt.kv_ms += pt.Mark();

    // 4g: FlashAttention-2
    {
      NVTX_SCOPE("FlashAttention2");
      T *k_cache = static_cast<T *>(kv_cache_->GetKVoid(layer, sequence_id));
      T *v_cache = static_cast<T *>(kv_cache_->GetVVoid(layer, sequence_id));

      float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim_));

      err = cuda_kernel::FlashAttention2Typed<T>(
          d_q_, k_cache, v_cache, d_attn_out_, /*batch_size=*/1, seq_len,
          kv_len, num_heads_, num_kv_heads_, head_dim_, attn_scale,
          /*causal=*/true, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "FlashAttention2 failed");
        return false;
      }
    }

    pt.attn_ms += pt.Mark();

    // Capture attention output for debugging
    if (capture_attention_tensors_) {
      SnapshotTensorToDevice("attn_out", layer, d_attn_out_,
                             {seq_len, hidden_size_});
    }

    // 4h: O projection + residual accumulation
    bool ffn_norm_precomputed = false;
    {
      NVTX_SCOPE("O_Projection");
      auto o_raw = weights_->LayerOProjRaw(layer);
      bool o_accumulated = false;
      // Try accumulate mode: write directly to residual, skip ResidualAdd
      if constexpr (std::is_same_v<T, half>) {
        if (fp32_residual_active_ &&
            TryQ8_1GemvAccumF32(o_raw, d_attn_out_, d_residual_f32_,
                                d_act_q8_1_, seq_len, hidden_size_,
                                num_heads_ * head_dim_, stream_, "o_proj",
                                &execution_policy_)) {
          o_accumulated = true;
        }
      }
      if (!o_accumulated && !fp32_residual_active_ &&
          TryQ8_1GemvAccum<T>(o_raw, d_attn_out_, d_residual_, d_act_q8_1_,
                              seq_len, hidden_size_, num_heads_ * head_dim_,
                              stream_, "o_proj", &execution_policy_)) {
        o_accumulated = true;
      }
      if (!o_accumulated &&
          !TryQ8_1Gemv<T>(o_raw, d_attn_out_, d_norm_out_, d_act_q8_1_, seq_len,
                          hidden_size_, num_heads_ * head_dim_, stream_,
                          "o_proj", &execution_policy_) &&
          !TryPackedGemv<T>(o_raw, d_attn_out_, d_norm_out_,
                            d_packed_activation_, d_packed_activation_scales_,
                            seq_len, hidden_size_, num_heads_ * head_dim_,
                            stream_, "o_proj", &execution_policy_)) {
        // cuBLAS fallback
        const T *o_proj =
            reinterpret_cast<const T *>(weights_->LayerOProj(layer));
        if (fp32_residual_active_) {
          // Write to d_norm_out_ (half), then ResidualAddMixed into FP32
          if (!gemm_->GemmTyped<T>(seq_len, hidden_size_,
                                   num_heads_ * head_dim_, d_attn_out_, o_proj,
                                   d_norm_out_)) {
            log::Error("llama_forward", "O projection failed");
            return false;
          }
          // Non-accumulated — residual add handled below
        } else {
          // Accumulate directly into half residual (beta=1.0)
          if (!gemm_->GemmTypedAccum<T>(seq_len, hidden_size_,
                                        num_heads_ * head_dim_, d_attn_out_,
                                        o_proj, d_residual_)) {
            log::Error("llama_forward", "O projection failed");
            return false;
          }
          o_accumulated = true;
        }
      }

      // 4i: residual += O (skip if accumulated directly)
      if (!o_accumulated) {
        if (fp32_residual_active_) {
          if (execution_policy_.enable_fused_residual_norm) {
            err = cuda_kernel::ResidualAddRmsNormMixed<T>(
                d_residual_f32_, d_norm_out_, post_attn_norm, d_norm_out_,
                seq_len, hidden_size_, rms_norm_eps_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "FP32 fused ResidualAddRmsNormMixed (attn) failed");
              return false;
            }
            ffn_norm_precomputed = true;
          } else {
            err = cuda_kernel::ResidualAddMixed<T>(
                d_residual_f32_, d_norm_out_, seq_len * hidden_size_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "FP32 ResidualAddMixed (attn) failed");
              return false;
            }
          }
        } else {
          if (execution_policy_.enable_fused_residual_norm) {
            err = cuda_kernel::ResidualAddRmsNorm<T>(
                d_residual_, d_norm_out_, post_attn_norm, d_norm_out_, seq_len,
                hidden_size_, rms_norm_eps_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "Fused ResidualAddRmsNorm (attn) failed");
              return false;
            }
            ffn_norm_precomputed = true;
          } else {
            err = cuda_kernel::ResidualAdd<T>(d_residual_, d_norm_out_,
                                              seq_len * hidden_size_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward", "Residual add (attn) failed");
              return false;
            }
          }
        }
      }
    }

    pt.o_proj_ms += pt.Mark();

    // Capture residual after attention path for debugging
    if (capture_attention_tensors_) {
      if (fp32_residual_active_) {
        SnapshotTensorToDevice("attn_residual_fp32", layer, d_residual_f32_,
                               {seq_len, hidden_size_});
      } else {
        SnapshotTensorToDevice("attn_residual_fp16", layer, d_residual_,
                               {seq_len, hidden_size_});
      }
    }

    // 4j-n: FFN block
    // When ffn_norm_precomputed, d_norm_out_ already has post-attn norm result.
    // FP32 residual: pre-normalize if needed
    if (fp32_residual_active_ && !ffn_norm_precomputed) {
      err = cuda_kernel::RmsNormMixed<T>(d_residual_f32_, post_attn_norm,
                                         d_norm_out_, seq_len, hidden_size_,
                                         rms_norm_eps_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward",
                   "FP32 RmsNormMixed (post-attn) failed at layer " +
                       std::to_string(layer));
        return false;
      }
      ffn_norm_precomputed = true;
    }
    const T *ffn_input = ffn_norm_precomputed
                             ? static_cast<const T *>(d_norm_out_)
                             : static_cast<const T *>(d_residual_);
    const T *ffn_norm_weight = ffn_norm_precomputed ? nullptr : post_attn_norm;
    bool down_accumulated = false;
    {
      NVTX_SCOPE("FFN");
      // Gate projection: try fused RmsNorm+GEMV (post-attn norm)
      auto gate_raw = weights_->LayerGateProjRaw(layer);
      // Up projection: try fused RmsNorm+GEMV (post-attn norm)
      auto up_raw = weights_->LayerUpProjRaw(layer);
      const std::array<PackedProjectionPlan<T>, 2> ffn_plans = {
          {{gate_raw, d_ffn_gate_, intermediate_size_},
           {up_raw, d_ffn_up_, intermediate_size_}}};
      const char *ffn_phase = seq_len == 1 ? "decode" : "prefill";
      const auto ffn_selected_op = SelectInferfluxCudaFfnProjOperator(
          gate_raw, up_raw, ParseInferfluxCudaDispatchPhase(ffn_phase),
          FusedDispatchGeometry{seq_len, intermediate_size_, hidden_size_, 2,
                                true, !ffn_norm_precomputed},
          g_allow_fused_quantized_matmul, execution_policy_);
      const std::string ffn_quant = ProjectionGroupQuantLabel(gate_raw, up_raw);
      NativeFfnExecutionSummary ffn_summary;
      if (!ExecuteInferfluxCudaFfnProjectionStage(
              ffn_selected_op, ffn_phase, ffn_quant, gate_raw.quant_type,
              seq_len, intermediate_size_, hidden_size_,
              [&]() {
                return TryQ8_1ProjectionGroup(
                    ffn_plans, ffn_input, ffn_norm_weight, d_act_q8_1_, seq_len,
                    hidden_size_, rms_norm_eps_, stream_, &execution_policy_,
                    ffn_selected_op);
              },
              [&]() {
                return TryPackedProjectionGroup(
                    ffn_plans, ffn_input, ffn_norm_weight, d_norm_out_,
                    d_packed_activation_, d_packed_activation_scales_, seq_len,
                    hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
              },
              [&]() {
                bool ffn_norm_computed = ffn_norm_precomputed;

                if (!ExecuteNativeNormalizedProjectionStage(
                        &ffn_norm_computed,
                        [&]() {
                          err = cuda_kernel::RmsNorm<T>(
                              d_residual_, post_attn_norm, d_norm_out_, seq_len,
                              hidden_size_, rms_norm_eps_, stream_);
                          if (err != cudaSuccess) {
                            log::Error("llama_forward",
                                       "Post-attn RmsNorm failed");
                            return false;
                          }
                          return true;
                        },
                        [&]() {
                          return TryQ8_1MmaGemv<T>(
                                     gate_raw, d_norm_out_, d_ffn_gate_,
                                     d_act_q8_1_mmq_, d_mma_partials_,
                                     seq_len, intermediate_size_, hidden_size_,
                                     stream_, "gate_proj",
                                     &execution_policy_) ||
                                 TryQ8_1Gemv<T>(
                                     gate_raw, d_norm_out_, d_ffn_gate_,
                                     d_act_q8_1_, seq_len, intermediate_size_,
                                     hidden_size_, stream_, "gate_proj",
                                     &execution_policy_);
                        },
                        [&]() {
                          const T *gate_proj = reinterpret_cast<const T *>(
                              weights_->LayerGateProj(layer));
                          if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_,
                                                   hidden_size_, d_norm_out_,
                                                   gate_proj, d_ffn_gate_)) {
                            log::Error("llama_forward",
                                       "Gate projection failed");
                            return false;
                          }
                          return true;
                        })) {
                  return false;
                }

                if (!ExecuteNativeNormalizedProjectionStage(
                        &ffn_norm_computed,
                        [&]() {
                          err = cuda_kernel::RmsNorm<T>(
                              d_residual_, post_attn_norm, d_norm_out_, seq_len,
                              hidden_size_, rms_norm_eps_, stream_);
                          if (err != cudaSuccess) {
                            log::Error("llama_forward",
                                       "Post-attn RmsNorm failed");
                            return false;
                          }
                          return true;
                        },
                        [&]() {
                          return TryQ8_1MmaGemv<T>(
                                     up_raw, d_norm_out_, d_ffn_up_,
                                     d_act_q8_1_mmq_, d_mma_partials_,
                                     seq_len, intermediate_size_, hidden_size_,
                                     stream_, "up_proj", &execution_policy_) ||
                                 TryQ8_1Gemv<T>(
                                     up_raw, d_norm_out_, d_ffn_up_,
                                     d_act_q8_1_, seq_len, intermediate_size_,
                                     hidden_size_, stream_, "up_proj",
                                     &execution_policy_);
                        },
                        [&]() {
                          const T *up_proj = reinterpret_cast<const T *>(
                              weights_->LayerUpProj(layer));
                          if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_,
                                                   hidden_size_, d_norm_out_,
                                                   up_proj, d_ffn_up_)) {
                            log::Error("llama_forward", "Up projection failed");
                            return false;
                          }
                          return true;
                        })) {
                  return false;
                }
                return true;
              },
              &ffn_summary, &execution_policy_)) {
        return false;
      }
      pt.ffn_proj_ms += pt.Mark();

      auto down_raw = weights_->LayerDownProjRaw(layer);
      auto down_mmq = weights_->LayerDownProjMmq(layer);
      const auto down_selected_op = SelectInferfluxCudaDownProjOperator(
          down_raw, down_mmq, ParseInferfluxCudaDispatchPhase(ffn_phase),
          FusedDispatchGeometry{seq_len, hidden_size_, intermediate_size_, 1,
                                true, false},
          g_allow_fused_quantized_matmul, execution_policy_);
      NativeDownProjExecutionSummary down_summary;
      if (!ExecuteInferfluxCudaDownProjStage(
              down_selected_op, ffn_phase, ProjectionQuantLabel(down_raw),
              down_raw.quant_type, seq_len, hidden_size_, intermediate_size_,
              [&]() {
                return TryMmqMmaSiluMul<T>(
                           down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                           d_act_q8_1_mmq_, d_mma_partials_, seq_len,
                           hidden_size_, intermediate_size_, stream_,
                           "down_proj", &execution_policy_) ||
                       TryMmqSiluMulGemv<T>(down_mmq, d_ffn_gate_, d_ffn_up_,
                                            d_ffn_down_, d_act_q8_1_, seq_len,
                                            hidden_size_, intermediate_size_,
                                            stream_, "down_proj",
                                            &execution_policy_);
              },
              [&]() {
                // Q8_1 can fuse the residual accumulation; keep it in the
                // Q8_1 tier so selected-vs-actual telemetry remains truthful.
                if constexpr (std::is_same_v<T, half>) {
                  if (fp32_residual_active_ &&
                      TryQ8_1SiluMulGemvAccumF32(
                          down_raw, d_ffn_gate_, d_ffn_up_, d_residual_f32_,
                          d_act_q8_1_, seq_len, hidden_size_,
                          intermediate_size_, stream_, "down_proj",
                          &execution_policy_)) {
                    down_accumulated = true;
                    return true;
                  }
                }
                if (!fp32_residual_active_ &&
                    TryQ8_1SiluMulGemvAccum<T>(
                        down_raw, d_ffn_gate_, d_ffn_up_, d_residual_,
                        d_act_q8_1_, seq_len, hidden_size_, intermediate_size_,
                        stream_, "down_proj", &execution_policy_)) {
                  down_accumulated = true;
                  return true;
                }
                return TryQ8_1SiluMulGemv<T>(
                    down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
                    seq_len, hidden_size_, intermediate_size_, stream_,
                    "down_proj", &execution_policy_);
              },
              [&]() {
                return TryPackedSiluMulGemv<T>(
                    down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                    d_packed_activation_, d_packed_activation_scales_, seq_len,
                    hidden_size_, intermediate_size_, stream_, "down_proj",
                    &execution_policy_);
              },
              [&]() {
                // SwiGLU
                err = cuda_kernel::SiluMul<T>(
                    d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                    seq_len * intermediate_size_, stream_);
                if (err != cudaSuccess) {
                  log::Error("llama_forward", "SiluMul failed");
                  return false;
                }
                pt.ffn_silu_ms += pt.Mark();

                // Down projection (input is activation, not normalized — no
                // fusion)
                bool down_ok = false;
                if (down_selected_op ==
                    FusedQuantGemm::DownProjOperator::kMmq) {
                  down_ok = TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                          d_act_q8_1_, seq_len, hidden_size_,
                                          intermediate_size_, stream_,
                                          "down_proj", &execution_policy_) ||
                            TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                           d_act_q8_1_, seq_len, hidden_size_,
                                           intermediate_size_, stream_,
                                           "down_proj", &execution_policy_);
                } else {
                  down_ok = TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                           d_act_q8_1_, seq_len, hidden_size_,
                                           intermediate_size_, stream_,
                                           "down_proj", &execution_policy_) ||
                            TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                          d_act_q8_1_, seq_len, hidden_size_,
                                          intermediate_size_, stream_,
                                          "down_proj", &execution_policy_);
                }
                if (!down_ok &&
                    !TryPackedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                      d_packed_activation_,
                                      d_packed_activation_scales_, seq_len,
                                      hidden_size_, intermediate_size_, stream_,
                                      "down_proj", &execution_policy_)) {
                  // cuBLAS fallback
                  const T *down_proj = reinterpret_cast<const T *>(
                      weights_->LayerDownProj(layer));
                  if (fp32_residual_active_) {
                    // Write to d_ffn_down_ (half), residual add handled below
                    if (!gemm_->GemmTyped<T>(seq_len, hidden_size_,
                                             intermediate_size_, d_ffn_gate_,
                                             down_proj, d_ffn_down_)) {
                      log::Error("llama_forward", "Down projection failed");
                      return false;
                    }
                  } else {
                    if (!gemm_->GemmTypedAccum<T>(
                            seq_len, hidden_size_, intermediate_size_,
                            d_ffn_gate_, down_proj, d_residual_)) {
                      log::Error("llama_forward", "Down projection failed");
                      return false;
                    }
                    down_accumulated = true;
                  }
                }
                return true;
              },
              [&](FusedQuantGemm::DownProjOperator actual_op) {
                const std::string down_label =
                    std::string("selected down-proj operator: ") +
                    FusedQuantGemm::DownProjOperatorName(actual_op);
                LogPackedGemmPath("down_proj", down_label.c_str());
              },
              &down_summary, &execution_policy_)) {
        return false;
      }
    }

    // 4o: residual += down (skip if accumulated directly)
    if (!down_accumulated) {
      if (fp32_residual_active_) {
        if (execution_policy_.enable_fused_residual_norm &&
            layer < num_layers_ - 1) {
          const T *next_input_norm =
              reinterpret_cast<const T *>(weights_->LayerInputNorm(layer + 1));
          err = cuda_kernel::ResidualAddRmsNormMixed<T>(
              d_residual_f32_, d_ffn_down_, next_input_norm, d_norm_out_,
              seq_len, hidden_size_, rms_norm_eps_, stream_);
          if (err != cudaSuccess) {
            log::Error(
                "llama_forward",
                "FP32 fused ResidualAddRmsNormMixed (FFN inter-layer) failed");
            return false;
          }
          input_norm_precomputed = true;
        } else {
          err = cuda_kernel::ResidualAddMixed<T>(
              d_residual_f32_, d_ffn_down_, seq_len * hidden_size_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "FP32 ResidualAddMixed (FFN) failed");
            return false;
          }
        }
      } else {
        if (execution_policy_.enable_fused_residual_norm &&
            layer < num_layers_ - 1) {
          const T *next_input_norm =
              reinterpret_cast<const T *>(weights_->LayerInputNorm(layer + 1));
          err = cuda_kernel::ResidualAddRmsNorm<T>(
              d_residual_, d_ffn_down_, next_input_norm, d_norm_out_, seq_len,
              hidden_size_, rms_norm_eps_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward",
                       "Fused ResidualAddRmsNorm (FFN inter-layer) failed");
            return false;
          }
          input_norm_precomputed = true;
        } else {
          err = cuda_kernel::ResidualAdd<T>(d_residual_, d_ffn_down_,
                                            seq_len * hidden_size_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "Residual add (FFN) failed");
            return false;
          }
        }
      }
    }
    pt.ffn_down_ms += pt.Mark();
  }

  // Step 5: Final RMSNorm + LM head (last token only)
  {
    NVTX_SCOPE("LM_Head");
    T *last_hidden = d_residual_ + (seq_len - 1) * hidden_size_;
    float *last_hidden_f32 =
        fp32_residual_active_ ? d_residual_f32_ + (seq_len - 1) * hidden_size_
                              : nullptr;
    const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());

    // Try Q8_1 fused RmsNorm+Quantize+GEMV for LM head first
    auto lm_raw = weights_->LmHeadRaw();
    bool lm_q8_1_ok = false;
    {
      if constexpr (std::is_same_v<T, half>) {
        if (!Q81ActivationsDisabled(execution_policy_) && lm_raw.data &&
            d_act_q8_1_ &&
            FusedQuantGemm::SupportsQ8_1Activations(lm_raw.quant_type) &&
            FusedQuantGemm::ShouldUseFusedPath(
                lm_raw.quant_type,
                FusedDispatchGeometry{1, vocab_size_, hidden_size_, 1, true,
                                      true})) {
          if (fp32_residual_active_) {
            FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
                last_hidden_f32, final_norm, d_act_q8_1_, 1, hidden_size_,
                rms_norm_eps_, stream_);
          } else {
            FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
                last_hidden, final_norm, d_act_q8_1_, 1, hidden_size_,
                rms_norm_eps_, stream_);
          }
          lm_q8_1_ok = FusedQuantGemm::GemvQ8_1(
              lm_raw, d_act_q8_1_, d_logits_typed_, 1, vocab_size_,
              hidden_size_, stream_, &execution_policy_);
          if (lm_q8_1_ok) {
            LogPackedGemmPath("lm_head",
                              "using Q8_1 fused RmsNorm+Quantize+GEMV");
          }
        }
      }
    }
    if (!lm_q8_1_ok) {
      // For FP32 residual, pre-normalize to half d_norm_out_ for downstream
      if (fp32_residual_active_) {
        err = cuda_kernel::RmsNormMixed<T>(last_hidden_f32, final_norm,
                                           d_norm_out_, 1, hidden_size_,
                                           rms_norm_eps_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "FP32 final RmsNormMixed failed");
          return false;
        }
      }
    }
    if (!lm_q8_1_ok &&
        (fp32_residual_active_ ||
         !TryPackedRmsNormGemv<T>(lm_raw, last_hidden, final_norm, d_norm_out_,
                                  d_packed_activation_,
                                  d_packed_activation_scales_, d_logits_typed_,
                                  1, vocab_size_, hidden_size_, rms_norm_eps_,
                                  stream_, "lm_head", &execution_policy_))) {
      // Fallback: standalone RmsNorm + GEMV/cuBLAS
      if (!fp32_residual_active_) {
        err = cuda_kernel::RmsNorm<T>(last_hidden, final_norm, d_norm_out_, 1,
                                      hidden_size_, rms_norm_eps_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "Final RmsNorm failed");
          return false;
        }
      }
      // d_norm_out_ now has the normalized half output (both paths)
      if (!TryQ8_1Gemv<T>(lm_raw, d_norm_out_, d_logits_typed_, d_act_q8_1_, 1,
                          vocab_size_, hidden_size_, stream_, "lm_head",
                          &execution_policy_)) {
        const T *lm_head = reinterpret_cast<const T *>(weights_->LmHead());
        if (!gemm_->GemmTyped<T>(1, vocab_size_, hidden_size_, d_norm_out_,
                                 lm_head, d_logits_typed_)) {
          log::Error("llama_forward", "LM head projection failed");
          return false;
        }
      }
    }

    // Step 7: Typed -> FP32 conversion (always float* output)
    err = cuda_kernel::HalfToFloat<T>(d_logits_typed_, d_logits, vocab_size_,
                                      stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "HalfToFloat failed");
      return false;
    }

    DebugDumpLogits(d_logits, vocab_size_, token_ids, n_past, stream_,
                    &execution_policy_);
    pt.lm_head_ms += pt.Mark();
  }

  pt.Report(num_layers_, seq_len);
  return true;
}

template <typename T>
void LlamaForwardTyped<T>::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

template <typename T>
void LlamaForwardTyped<T>::SetExecutionPolicy(
    const NativeExecutionPolicy &policy) {
  execution_policy_ = policy;
}

template <typename T> void LlamaForwardTyped<T>::WarmWeightCaches() {
  // Trigger lazy F32→FP16 dequantization for all norm weights, embedding
  // tables, attention biases, and LM head.  These are stored as F32 in GGUF
  // and converted via cudaStreamSynchronize on first access — which is
  // illegal inside a CUDA graph capture region.  Calling this at model-load
  // time ensures the caches are populated before the first BatchForward().
  weights_->EmbedTokens();
  for (int l = 0; l < num_layers_; ++l) {
    weights_->LayerInputNorm(l);
    weights_->LayerPostAttnNorm(l);
    weights_->LayerQProjBias(l);
    weights_->LayerKProjBias(l);
    weights_->LayerVProjBias(l);
  }
  weights_->FinalNorm();
  // Pre-warm LM head to avoid first-token TTFT penalty from lazy dequant.
  weights_->LmHead();
  // Clear any CUDA errors from pre-warm (e.g., missing bias tensors return
  // nullptr without error, but some edge-case allocations may fail).
  cudaGetLastError();
  log::Info("llama_forward", "Weight caches pre-warmed (" +
                                 std::to_string(num_layers_) +
                                 " layers, including LM head)");
}

template <typename T>
bool LlamaForwardTyped<T>::BatchForwardReplay(float *d_logits, int batch_size) {
  if (!decode_graph_exec_ || graph_batch_size_ != batch_size) {
    return false; // Graph not captured or batch size mismatch
  }
  // Skip H2D metadata upload — DeviceTokenRelay already updated d_batch_meta_
  // on device. Just replay the graph.
  cudaError_t err = cudaGraphLaunch(decode_graph_exec_, stream_);
  return err == cudaSuccess;
}

template <typename T> int *LlamaForwardTyped<T>::GetBatchMetaDevice() {
  return d_batch_meta_;
}

template <typename T> int LlamaForwardTyped<T>::GetMaxBatchSize() const {
  return max_batch_size_;
}

template <typename T> std::string LlamaForwardTyped<T>::ProbeDispatchPaths() {
  auto &health = InferfluxCudaOperatorHealth::Instance();
  if (const char *force =
          std::getenv("INFERFLUX_CUDA_DISPATCH_PROBE_FORCE_UNHEALTHY");
      force && *force) {
    health.ApplyForceList(force);
  }

  // Layer-0 weights, production executor stages and lambdas — the probe
  // exercises exactly the dispatch that decode will take, so it cannot
  // drift from reality. Outputs land in forward scratch buffers; weights
  // are read-only.
  const int layer = 0;
  const auto gate_raw = weights_->LayerGateProjRaw(layer);
  const auto up_raw = weights_->LayerUpProjRaw(layer);
  const auto down_raw = weights_->LayerDownProjRaw(layer);
  const auto down_mmq = weights_->LayerDownProjMmq(layer);
  int divergences = 0;

  const int m_values[] = {1, 2, 4, 8, 16};
  for (const int M : m_values) {
    const std::array<PackedProjectionPlan<half>, 2> ffn_plans = {
        {{gate_raw, reinterpret_cast<half *>(d_ffn_gate_), intermediate_size_},
         {up_raw, reinterpret_cast<half *>(d_ffn_up_), intermediate_size_}}};
    const auto selected = SelectInferfluxCudaFfnProjOperator(
        gate_raw, up_raw, InferfluxCudaDispatchPhase::kDecode,
        FusedDispatchGeometry{M, intermediate_size_, hidden_size_, 2, true,
                              true},
        true, execution_policy_);
    NativeFfnExecutionSummary summary;
    ExecuteInferfluxCudaFfnProjectionStage(
        selected, "probe", "probe", gate_raw.quant_type, M, intermediate_size_,
        hidden_size_,
        [&]() {
          return TryQ8_1ProjectionGroup<2>(
              ffn_plans, reinterpret_cast<const half *>(d_residual_),
              reinterpret_cast<const half *>(
                  weights_->LayerPostAttnNorm(layer)),
              d_act_q8_1_, M, hidden_size_, rms_norm_eps_, stream_,
              &execution_policy_, selected);
        },
        [&]() {
          return TryPackedProjectionGroup<2>(
              ffn_plans, reinterpret_cast<const half *>(d_residual_),
              reinterpret_cast<const half *>(
                  weights_->LayerPostAttnNorm(layer)),
              reinterpret_cast<half *>(d_norm_out_), d_packed_activation_,
              d_packed_activation_scales_, M, hidden_size_, rms_norm_eps_,
              stream_, &execution_policy_);
        },
        []() { return false; }, &summary);
    if (TierOf(selected) != TierOf(summary.actual_op) ||
        (summary.actual_op == FfnProjOperator::kFallback &&
         selected != FfnProjOperator::kFallback)) {
      ++divergences;
      health.MarkUnhealthy(selected, "probe_divergence");
    }

    const auto selected_down = SelectInferfluxCudaDownProjOperator(
        down_raw, down_mmq, InferfluxCudaDispatchPhase::kDecode,
        FusedDispatchGeometry{M, hidden_size_, intermediate_size_, 1, true,
                              false},
        true, execution_policy_);
    NativeDownProjExecutionSummary down_summary;
    ExecuteInferfluxCudaDownProjStage(
        selected_down, "probe", ProjectionQuantLabel(down_raw),
        down_raw.quant_type, M, hidden_size_, intermediate_size_,
        [&]() {
          return TryMmqMmaSiluMul<T>(down_raw, d_ffn_gate_, d_ffn_up_,
                                     d_ffn_down_, d_act_q8_1_mmq_,
                                     d_mma_partials_, M, hidden_size_,
                                     intermediate_size_, stream_,
                                     "down_proj_probe", &execution_policy_) ||
                 TryMmqSiluMulGemv<T>(down_mmq, d_ffn_gate_, d_ffn_up_,
                                      d_ffn_down_, d_act_q8_1_, M, hidden_size_,
                                      intermediate_size_, stream_,
                                      "down_proj_probe", &execution_policy_);
        },
        [&]() {
          if constexpr (std::is_same_v<T, half>) {
            if (fp32_residual_active_ &&
                TryQ8_1SiluMulGemvAccumF32(
                    down_raw, d_ffn_gate_, d_ffn_up_, d_residual_f32_,
                    d_act_q8_1_, M, hidden_size_, intermediate_size_, stream_,
                    "down_proj_probe", &execution_policy_)) {
              return true;
            }
          }
          if (!fp32_residual_active_ &&
              TryQ8_1SiluMulGemvAccum<T>(
                  down_raw, d_ffn_gate_, d_ffn_up_, d_residual_, d_act_q8_1_, M,
                  hidden_size_, intermediate_size_, stream_, "down_proj_probe",
                  &execution_policy_)) {
            return true;
          }
          return TryQ8_1SiluMulGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_, M,
              hidden_size_, intermediate_size_, stream_, "down_proj_probe",
              &execution_policy_);
        },
        [&]() {
          return TryPackedSiluMulGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
              d_packed_activation_, d_packed_activation_scales_, M,
              hidden_size_, intermediate_size_, stream_, "down_proj_probe",
              &execution_policy_);
        },
        []() { return false; }, [](DownProjOperator) {}, &down_summary,
        &execution_policy_);
    if (TierOf(selected_down) != TierOf(down_summary.actual_op) ||
        (down_summary.actual_op == DownProjOperator::kFallback &&
         selected_down != DownProjOperator::kFallback)) {
      ++divergences;
      health.MarkUnhealthy(selected_down, "probe_divergence");
    }
  }

  cudaStreamSynchronize(stream_);
  cudaGetLastError();
  if (divergences > 0 || health.AnyUnhealthy()) {
    log::Warn("llama_forward",
              "dispatch probe: " + std::to_string(divergences) +
                  " divergent operator(s); down-ranked for this process: " +
                  (health.Describe().empty() ? "(forced list)" //
                                             : health.Describe()));
  } else {
    log::Info("llama_forward", "dispatch probe: all paths reachable");
  }
  return health.Describe();
}

template <typename T>
bool LlamaForwardTyped<T>::BatchForward(const std::vector<int> &token_ids,
                                        const std::vector<int> &n_past,
                                        const std::vector<int> &sequence_ids,
                                        float *d_logits, int batch_size) {
  NVTX_SCOPE("BatchForward");
  if (batch_size <= 0)
    return false;
  const auto &policy = execution_policy_;
  const bool use_batched = policy.enable_batched_decode;
  if (!use_batched) {
    bool all_ok = true;
    for (int b = 0; b < batch_size; ++b) {
      std::vector<int> single = {token_ids[b]};
      float *out = d_logits + b * vocab_size_;
      if (!Forward(single, n_past[b], sequence_ids[b], out))
        all_ok = false;
    }
    return all_ok;
  }

  int B = batch_size;
  if (B > max_batch_size_) {
    log::Error("llama_forward",
               "BatchForward: batch size exceeds preallocated staging buffers");
    return false;
  }
  const bool allow_fused_quantized_matmul =
      !weights_ || weights_->AllowFusedQuantizedMatmul();
  ScopedFusedMatmulPolicy fused_policy(allow_fused_quantized_matmul);
  cudaError_t err;

  // ===== Phase 1: Upload metadata to fixed device addresses =====
  // All H2D copies happen BEFORE any graph-captured region so that
  // graph replay reads updated data from the same device addresses.
  std::copy_n(token_ids.data(), B, h_batch_token_ids_);
  std::copy_n(n_past.data(), B, h_batch_n_past_);
  std::copy_n(sequence_ids.data(), B, h_batch_seq_ids_);
  for (int b = 0; b < B; ++b) {
    h_batch_kv_lens_[b] = h_batch_n_past_[b] + 1;
  }

  // The device metadata slab is laid out as four max_batch_size segments:
  // [token_ids][n_past][seq_ids][kv_lens]. Because the views are based on the
  // reserved max batch size rather than the active batch size, we must upload
  // the full reserved slab here; copying only B * 4 ints would leave the
  // later segments stale whenever B < max_batch_size_.
  err = runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kBatchMetaH2D, d_batch_meta_,
      h_batch_meta_, static_cast<size_t>(max_batch_size_) * 4 * sizeof(int),
      cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "BatchForward: batch metadata upload failed");
    return false;
  }

  return BatchForwardDevice(B, d_logits);
}

template <typename T>
bool LlamaForwardTyped<T>::BatchForwardDevice(int batch_size, float *d_logits) {
  const auto &policy = execution_policy_;
  if (!policy.enable_batched_decode) {
    // Device-fed decode requires the batched path; per-sequence Forward()
    // needs host token vectors by construction.
    return false;
  }
  if (batch_size <= 0 || batch_size > max_batch_size_) {
    log::Error("llama_forward",
               "BatchForwardDevice: batch size exceeds preallocated staging "
               "buffers");
    return false;
  }
  const bool allow_fused_quantized_matmul =
      !weights_ || weights_->AllowFusedQuantizedMatmul();
  ScopedFusedMatmulPolicy fused_policy(allow_fused_quantized_matmul);
  cudaError_t err;
  const int B = batch_size;

  // Small-batch tier switch: the whole compute section must see ONE
  // consistent tier decision (operator selection, dispatch guards, fused
  // path checks), so tiny batches run on a step-local policy copy with the
  // Q8_1 tier disabled rather than gating individual sites — which desyncs
  // selection and execution onto cuBLAS fallbacks. Tier depends only on B,
  // so per-B graph capture/recapture stays consistent by construction.
  NativeExecutionPolicy step_policy_storage;
  const NativeExecutionPolicy *active_policy = &execution_policy_;
  if (execution_policy_.packed_tier_max_batch >= 0 &&
      B <= execution_policy_.packed_tier_max_batch) {
    step_policy_storage = execution_policy_;
    step_policy_storage.disable_q81_activations = true;
    active_policy = &step_policy_storage;
  }

  // ===== Phase 2: metadata upload only =====
  // Use strided kernels for contiguous (dense) cache, indirect for hybrid.
  const bool kv_contiguous = kv_cache_->IsContiguous();
  T *const kv_buffer =
      kv_contiguous ? static_cast<T *>(kv_cache_->ContiguousBuffer()) : nullptr;
  T *const *d_slot_base_ptrs =
      static_cast<T *const *>(kv_cache_->SlotBasePtrsDevice());

  // ===== Phase 3: CUDA graph replay or capture =====
  // CUDA graph capture eliminates per-kernel launch overhead
  // (~5-10μs × 250+ kernels = 1-2ms/token). Enabled by default;
  // disable with INFERFLUX_DISABLE_CUDA_GRAPH=1.
  const bool phase_timing_enabled = policy.phase_timing_enabled;
  const bool graph_disabled = policy.disable_cuda_graph;
  const bool capture_safe = DecodeGraphCaptureSafe(
      weights_, num_layers_, B, hidden_size_, num_heads_, num_kv_heads_,
      head_dim_, intermediate_size_, vocab_size_,
      g_allow_fused_quantized_matmul, policy);
  {
    static bool logged_once = false;
    if (!logged_once) {
      logged_once = true;
      log::Info("llama_forward",
                "decode graph decision: graph_enabled=" +
                    std::string(graph_enabled_ ? "1" : "0") + " disabled=" +
                    std::string(graph_disabled ? "1" : "0") + " phase_timing=" +
                    std::string(phase_timing_enabled ? "1" : "0") +
                    " capture_safe=" + std::string(capture_safe ? "1" : "0") +
                    " warmup_left=" + std::to_string(graph_warmup_remaining_));
    }
  }
  if (graph_enabled_ && !graph_disabled && !phase_timing_enabled &&
      !capture_safe) {
    static bool logged_graph_fallback = false;
    if (!logged_graph_fallback) {
      logged_graph_fallback = true;
      log::Info("llama_forward",
                "CUDA graph disabled for batched decode because at least one "
                "projection requires non-graph-safe fallback");
    }
  }
  bool use_graph = ShouldUseDecodeGraph(graph_enabled_, graph_disabled,
                                        phase_timing_enabled, capture_safe);
  // Graph warmup: skip capture for the first N calls to let any remaining
  // lazy first-use allocations settle.  With WarmWeightCaches() called at
  // init time, this should normally be 0.
  if (use_graph && graph_warmup_remaining_ > 0 && !decode_graph_exec_) {
    --graph_warmup_remaining_;
    use_graph = false;
  }
  PhaseTiming pt;
  pt.Begin(stream_, phase_timing_enabled);

  // Fast path: replay existing graph if batch size matches
  if (use_graph && decode_graph_exec_ && graph_batch_size_ == B) {
    err = cudaGraphLaunch(decode_graph_exec_, stream_);
    if (err == cudaSuccess)
      return true;
    log::Warn("llama_forward", "CUDA graph replay failed, disabling");
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
    graph_enabled_ = false;
    // Fall through to non-graph path
  }

  // Destroy stale graph if batch size changed (graph topology depends on B)
  if (decode_graph_exec_ && graph_batch_size_ != B) {
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
  }

  // Begin graph capture if enabled
  bool capturing = false;
  if (use_graph) {
    // Drain any sticky CUDA error from prior operations (e.g., prefill
    // Forward(), weight dequantization, or pre-warm failures).
    cudaGetLastError();
    cudaStreamSynchronize(stream_); // Drain compute stream before capture
    err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed);
    if (err == cudaSuccess) {
      capturing = true;
    } else {
      if (--graph_retry_remaining_ <= 0) {
        log::Warn("llama_forward",
                  "CUDA graph capture begin permanently failed");
        graph_enabled_ = false;
      } else {
        graph_warmup_remaining_ = 2;
        log::Warn("llama_forward",
                  "CUDA graph capture begin failed, will retry (" +
                      std::to_string(graph_retry_remaining_) +
                      " attempts remaining)");
      }
    }
  }

  // ===== Compute section (captured into graph or executed directly) =====
  // All operations below use fixed device addresses. During graph replay,
  // the kernels read updated data uploaded in Phases 1-2.
  //
  // IMPORTANT: cuBLAS calls during CUDA graph capture corrupt the host heap
  // because cuBLAS internally allocates workspace that becomes stale on
  // replay. All cuBLAS fallback paths are guarded with `if (!capturing)`.
  // If fused GEMV can't handle a projection during capture, we set
  // `capture_abort` and fall back to direct execution after ending capture.
  bool capture_abort = false;
  auto RunCompute = [&]() -> bool {
    // Embedding [B, hidden_size] — write directly to residual stream.
    {
      NVTX_SCOPE("Embedding");
      const T *embed = reinterpret_cast<const T *>(weights_->EmbedTokens());
      if (fp32_residual_active_) {
        err = cuda_kernel::EmbeddingLookupF32<T>(embed, d_batch_token_ids_,
                                                 d_residual_f32_, B,
                                                 hidden_size_, stream_);
      } else {
        err = cuda_kernel::EmbeddingLookup<T>(
            embed, d_batch_token_ids_, d_residual_, B, hidden_size_, stream_);
      }
      if (err != cudaSuccess)
        return false;
    }
    pt.embed_ms += pt.Mark();

    // Transformer layers
    bool input_norm_precomputed = false;
    // Loop-carry: the previous layer's P2 epilogue quantized d_norm_out_
    // into d_act_q8_1_ — the QKV group must not re-quantize it.
    bool input_q81_precomputed = false;
    for (int layer = 0; layer < num_layers_; layer++) {
      NVTX_SCOPE("Layer");
      // Bias-fold plumbing: assigned after the QKV projection below, read
      // by the RoPE and KV-append stages.
      const T *rope_q_bias = nullptr;
      const T *rope_k_bias = nullptr;
      const T *append_v_bias = nullptr;
      const T *input_norm =
          reinterpret_cast<const T *>(weights_->LayerInputNorm(layer));
      const T *post_attn_norm =
          reinterpret_cast<const T *>(weights_->LayerPostAttnNorm(layer));

      // Batched Q/K/V projections with fused RmsNorm
      // FP32 residual: pre-normalize from FP32 → half d_norm_out_
      if (fp32_residual_active_ && !input_norm_precomputed) {
        cuda_kernel::RmsNormMixed<T>(d_residual_f32_, input_norm, d_norm_out_,
                                     B, hidden_size_, rms_norm_eps_, stream_);
        input_norm_precomputed = true;
      }
      const T *qkv_input = input_norm_precomputed
                               ? static_cast<const T *>(d_norm_out_)
                               : static_cast<const T *>(d_residual_);
      const T *qkv_norm = input_norm_precomputed ? nullptr : input_norm;
      input_norm_precomputed = false;
      {
        NVTX_SCOPE("QKV_Projection");
        auto q_raw = weights_->LayerQProjRaw(layer);
        auto k_raw = weights_->LayerKProjRaw(layer);
        auto v_raw = weights_->LayerVProjRaw(layer);
        const std::array<PackedProjectionPlan<T>, 3> qkv_plans = {
            {{q_raw, d_q_, num_heads_ * head_dim_},
             {k_raw, d_k_new_, num_kv_heads_ * head_dim_},
             {v_raw, d_v_new_, num_kv_heads_ * head_dim_}}};
        if (!ExecuteNativeGroupedProjectionStage(
                [&]() {
                  // S8: Q4_K MMA QKV triple at M >= 2 — replaces the
                  // grouped dp4a family (1.14s of c16 kernel time).
                  if constexpr (std::is_same_v<T, half>) {
                    if (B >= 2) {
                      if (qkv_norm != nullptr) {
                        cuda_kernel::RmsNorm<T>(d_residual_, qkv_norm,
                                                d_norm_out_, B, hidden_size_,
                                                rms_norm_eps_, stream_);
                      }
                      const half *mma_input = static_cast<const half *>(
                          qkv_norm != nullptr ? d_norm_out_ : qkv_input);
                      if (TryQ8_1MmaGemv<T>(
                              q_raw, mma_input, d_q_, d_act_q8_1_mmq_,
                              d_mma_partials_, B, num_heads_ * head_dim_,
                              hidden_size_, stream_, "q_proj",
                              active_policy) &&
                          TryQ8_1MmaGemv<T>(
                              k_raw, mma_input, d_k_new_, d_act_q8_1_mmq_,
                              d_mma_partials_, B, num_kv_heads_ * head_dim_,
                              hidden_size_, stream_, "k_proj",
                              active_policy) &&
                          TryQ8_1MmaGemv<T>(
                              v_raw, mma_input, d_v_new_, d_act_q8_1_mmq_,
                              d_mma_partials_, B, num_kv_heads_ * head_dim_,
                              hidden_size_, stream_, "v_proj",
                              active_policy)) {
                        return true;
                      }
                    }
                  }
                  return TryQ8_1ProjectionGroup(
                      qkv_plans, qkv_input, qkv_norm, d_act_q8_1_, B,
                      hidden_size_, rms_norm_eps_, stream_, active_policy,
                      FusedQuantGemm::FfnProjOperator::kFallback,
                      input_q81_precomputed);
                },
                [&]() {
                  return TryPackedProjectionGroup(
                      qkv_plans, qkv_input, qkv_norm, d_norm_out_,
                      d_packed_activation_, d_packed_activation_scales_, B,
                      hidden_size_, rms_norm_eps_, stream_, active_policy);
                },
                [&]() {
                  bool norm_computed = (qkv_norm == nullptr);

                  if (!ExecuteNativeNormalizedProjectionStage(
                          &norm_computed,
                          [&]() {
                            cuda_kernel::RmsNorm<T>(
                                d_residual_, input_norm, d_norm_out_, B,
                                hidden_size_, rms_norm_eps_, stream_);
                            return true;
                          },
                          [&]() {
                            return TryQ8_1MmaGemv<T>(
                                       q_raw, d_norm_out_, d_q_,
                                       d_act_q8_1_mmq_, d_mma_partials_, B,
                                       num_heads_ * head_dim_, hidden_size_,
                                       stream_, "q_proj", active_policy) ||
                                   TryQ8_1Gemv<T>(
                                       q_raw, d_norm_out_, d_q_, d_act_q8_1_,
                                       B, num_heads_ * head_dim_, hidden_size_,
                                       stream_, "q_proj", active_policy);
                          },
                          [&]() {
                            if (capturing) {
                              capture_abort = true;
                              return false;
                            }
                            const T *q_proj = reinterpret_cast<const T *>(
                                weights_->LayerQProj(layer));
                            gemm_->GemmTyped<T>(B, num_heads_ * head_dim_,
                                                hidden_size_, d_norm_out_,
                                                q_proj, d_q_);
                            return true;
                          })) {
                    return false;
                  }

                  if (!ExecuteNativeNormalizedProjectionStage(
                          &norm_computed,
                          [&]() {
                            cuda_kernel::RmsNorm<T>(
                                d_residual_, input_norm, d_norm_out_, B,
                                hidden_size_, rms_norm_eps_, stream_);
                            return true;
                          },
                          [&]() {
                            return TryQ8_1MmaGemv<T>(
                                       k_raw, d_norm_out_, d_k_new_,
                                       d_act_q8_1_mmq_, d_mma_partials_, B,
                                       num_kv_heads_ * head_dim_, hidden_size_,
                                       stream_, "k_proj", active_policy) ||
                                   TryQ8_1Gemv<T>(
                                       k_raw, d_norm_out_, d_k_new_,
                                       d_act_q8_1_, B,
                                       num_kv_heads_ * head_dim_, hidden_size_,
                                       stream_, "k_proj", active_policy);
                          },
                          [&]() {
                            if (capturing) {
                              capture_abort = true;
                              return false;
                            }
                            const T *k_proj = reinterpret_cast<const T *>(
                                weights_->LayerKProj(layer));
                            gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_,
                                                hidden_size_, d_norm_out_,
                                                k_proj, d_k_new_);
                            return true;
                          })) {
                    return false;
                  }

                  if (!ExecuteNativeNormalizedProjectionStage(
                          &norm_computed,
                          [&]() {
                            cuda_kernel::RmsNorm<T>(
                                d_residual_, input_norm, d_norm_out_, B,
                                hidden_size_, rms_norm_eps_, stream_);
                            return true;
                          },
                          [&]() {
                            return TryQ8_1MmaGemv<T>(
                                       v_raw, d_norm_out_, d_v_new_,
                                       d_act_q8_1_mmq_, d_mma_partials_, B,
                                       num_kv_heads_ * head_dim_, hidden_size_,
                                       stream_, "v_proj", active_policy) ||
                                   TryQ8_1Gemv<T>(
                                       v_raw, d_norm_out_, d_v_new_,
                                       d_act_q8_1_, B,
                                       num_kv_heads_ * head_dim_, hidden_size_,
                                       stream_, "v_proj", active_policy);
                          },
                          [&]() {
                            if (capturing) {
                              capture_abort = true;
                              return false;
                            }
                            const T *v_proj = reinterpret_cast<const T *>(
                                weights_->LayerVProj(layer));
                            gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_,
                                                hidden_size_, d_norm_out_,
                                                v_proj, d_v_new_);
                            return true;
                          })) {
                    return false;
                  }
                  return true;
                })) {
          return false;
        }

        // Biases (if present): either folded into the RoPE register load /
        // KV-append store (one fewer kernel + one fewer half-rounding), or
        // applied by standalone BiasAdd launches.
        const T *q_bias =
            reinterpret_cast<const T *>(weights_->LayerQProjBias(layer));
        const T *k_bias =
            reinterpret_cast<const T *>(weights_->LayerKProjBias(layer));
        const T *v_bias =
            reinterpret_cast<const T *>(weights_->LayerVProjBias(layer));
        const bool fold_bias =
            active_policy->enable_bias_rope_fusion && kv_contiguous;
        if (!fold_bias && q_bias && k_bias && v_bias &&
            active_policy->enable_fused_bias_add) {
          cuda_kernel::BiasAddTriple<T>(
              d_q_, d_k_new_, d_v_new_, q_bias, k_bias, v_bias, B,
              num_heads_ * head_dim_, num_kv_heads_ * head_dim_,
              num_kv_heads_ * head_dim_, stream_);
        } else if (!fold_bias) {
          if (q_bias)
            cuda_kernel::BiasAdd<T>(d_q_, q_bias, B, num_heads_ * head_dim_,
                                    stream_);
          if (k_bias)
            cuda_kernel::BiasAdd<T>(d_k_new_, k_bias, B,
                                    num_kv_heads_ * head_dim_, stream_);
          if (v_bias)
            cuda_kernel::BiasAdd<T>(d_v_new_, v_bias, B,
                                    num_kv_heads_ * head_dim_, stream_);
        } else {
          rope_q_bias = q_bias;
          rope_k_bias = k_bias;
          append_v_bias = v_bias;
        }
      }
      pt.qkv_ms += pt.Mark();

      // RoPE + KV append (fused or separate)
      if (active_policy->enable_fused_rope_kv_append &&
          rope_q_bias == nullptr && rope_k_bias == nullptr &&
          append_v_bias == nullptr) {
        NVTX_SCOPE("FusedRoPE_KV_Append");
        if (kv_contiguous) {
          err = cuda_kernel::FusedRoPEKvAppendStrided<T>(
              d_q_, d_k_new_, d_v_new_, kv_buffer, d_batch_seq_ids_,
              d_batch_n_past_, layer, B, num_heads_, num_kv_heads_, head_dim_,
              kv_cache_->KvDim(), kv_cache_->SlotStride(),
              kv_cache_->LayerStride(), kv_cache_->KvStride(), rope_freq_base_,
              stream_, rope_type_);
        } else {
          err = cuda_kernel::FusedRoPEKvAppendIndirect<T>(
              d_q_, d_k_new_, d_v_new_, d_slot_base_ptrs, d_batch_seq_ids_,
              d_batch_n_past_, layer, B, num_heads_, num_kv_heads_, head_dim_,
              kv_cache_->KvDim(), kv_cache_->LayerStride(),
              kv_cache_->KvStride(), rope_freq_base_, stream_, rope_type_);
        }
        if (err != cudaSuccess) {
          log::Error("llama_forward", "FusedRoPEKvAppend launch failed: " +
                                          std::string(cudaGetErrorString(err)));
          return false;
        }
        pt.rope_ms += pt.Mark();
        pt.kv_ms = 0;
      } else {
        // Unfused path: separate RoPE + KvAppend
        {
          NVTX_SCOPE("RoPE");
          err = cuda_kernel::BatchedRoPE<T>(
              d_q_, d_k_new_, B, num_heads_, num_kv_heads_, head_dim_,
              d_batch_n_past_, rope_freq_base_, stream_, rope_type_,
              rope_q_bias, rope_k_bias);
          if (err != cudaSuccess) {
            log::Error("llama_forward",
                       "BatchedRoPE launch failed: " +
                           std::string(cudaGetErrorString(err)));
            return false;
          }
        }
        pt.rope_ms += pt.Mark();

        {
          NVTX_SCOPE("KV_Append");
          if (kv_contiguous) {
            err = cuda_kernel::BatchedKvAppendStrided<T>(
                d_k_new_, d_v_new_, kv_buffer, d_batch_seq_ids_,
                d_batch_n_past_, layer, B, kv_cache_->KvDim(),
                kv_cache_->SlotStride(), kv_cache_->LayerStride(),
                kv_cache_->KvStride(), stream_, append_v_bias);
          } else {
            err = cuda_kernel::BatchedKvAppendIndirect<T>(
                d_k_new_, d_v_new_, d_slot_base_ptrs, d_batch_seq_ids_,
                d_batch_n_past_, layer, B, kv_cache_->KvDim(),
                kv_cache_->LayerStride(), kv_cache_->KvStride(), stream_);
          }
          if (err != cudaSuccess) {
            log::Error("llama_forward",
                       "BatchedKvAppend launch failed: " +
                           std::string(cudaGetErrorString(err)));
            return false;
          }
        }
        pt.kv_ms += pt.Mark();
      }

      // FlashDecode: strided for contiguous cache, indirect for hybrid.
      {
        NVTX_SCOPE("FlashAttention2");
        float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
        if (kv_contiguous && policy.enable_attn_split_kv && d_attn_partials_) {
          // Split geometry is fixed at capture time from the configured
          // context length so graph replays stay shape-static; per-sequence
          // kv_lens early-exit empty chunks on device.
          int qsplit = policy.attn_split_qsplit_override;
          // Measured on RTX 4000 Ada (Qwen2.5-3B, GQA 8): Q-head
          // parallelism dominates at decode contexts — qsplit=8 with a
          // single KV split gains ~15% at batch 1. Two constraints shape
          // the heuristic: enough CTAs to cover the SMs (~2 per SM), and no
          // more KV re-read parallelism than that needs — at batch 16 the
          // unsplit grid already spans 32 CTAs and 8x re-reads plus 256
          // tiny CTAs regress. Graphs are per-B, so B is static here.
          int chunk = policy.attn_split_chunk;
          if (qsplit <= 0) {
            const int kTargetCtas = 96; // ~2x SM count on Ada
            qsplit = kTargetCtas / std::max(1, B * num_kv_heads_);
            qsplit = std::max(1, std::min(qsplit, 8));
            if (max_seq_len_ > 2048) {
              // Long contexts: KV re-reads get expensive as kv grows; lean
              // on KV chunking instead.
              qsplit = std::min(qsplit, 2);
            } else {
              // Keep ksplits=1 below 2k: chunking adds a combine pass and
              // identity partials that only pay at long context.
              chunk = std::max(chunk, max_seq_len_);
            }
          }
          // Chunk floor of 64 positions matches the FA2 KV tile size.
          chunk = std::max(chunk, 64);
          const int ksplits = std::min(
              kMaxAttnKSplits, std::max(1, (max_seq_len_ + chunk - 1) / chunk));
          err = cuda_kernel::FlashDecodeMultiSeqStridedSplit<T>(
              d_q_, kv_buffer, d_attn_out_, d_attn_partials_, d_batch_seq_ids_,
              d_batch_kv_lens_, layer, B, num_heads_, num_kv_heads_, head_dim_,
              kv_cache_->SlotStride(), kv_cache_->LayerStride(),
              kv_cache_->KvStride(), attn_scale, qsplit, chunk, ksplits,
              stream_);
        } else if (kv_contiguous) {
          // Warp-per-head decode: no block barriers, no split workspace,
          // no combine launch — the block-cooperative split path pays
          // ~232 synchronized reductions per CTA at decode sizes. Use it
          // while the full-KV scan by num_heads warps keeps CTAs and KV
          // re-reads bounded (num_heads warps >= 2x SMs at B>=6 for 16
          // heads; at small B the single scan still beats split+reduce).
          // Opt-in (INFERFLUX_CUDA_ATTN_WPH=1): measured -16% at c1 —
          // each warp's serial online-softmax chain has no latency hiding
          // with only num_heads/4 blocks resident. Kept for tile-ILP rework.
          static const bool wph_enabled = [] {
            const char *raw = std::getenv("INFERFLUX_CUDA_ATTN_WPH");
            return raw && (std::string(raw) == "1" || std::string(raw) == "true");
          }();
          if (wph_enabled) {
            err = cuda_kernel::FlashDecodeWarpPerHead<T>(
                d_q_, kv_buffer, d_attn_out_, d_batch_seq_ids_,
                d_batch_kv_lens_, layer, B, num_heads_, num_kv_heads_,
                head_dim_, kv_cache_->SlotStride(), kv_cache_->LayerStride(),
                kv_cache_->KvStride(), attn_scale, stream_);
          } else {
            err = cuda_kernel::FlashDecodeMultiSeqStrided<T>(
                d_q_, kv_buffer, d_attn_out_, d_batch_seq_ids_,
                d_batch_kv_lens_, layer, B, num_heads_, num_kv_heads_,
                head_dim_, kv_cache_->SlotStride(), kv_cache_->LayerStride(),
                kv_cache_->KvStride(), attn_scale, stream_,
                d_attn_split_workspace_, attn_split_workspace_bytes_);
          }
        } else {
          err = cuda_kernel::FlashDecodeMultiSeqIndirect<T>(
              d_q_, d_slot_base_ptrs, d_attn_out_, d_batch_seq_ids_,
              d_batch_kv_lens_, layer, B, num_heads_, num_kv_heads_, head_dim_,
              kv_cache_->LayerStride(), kv_cache_->KvStride(), attn_scale,
              stream_, d_attn_split_workspace_, attn_split_workspace_bytes_);
        }
        if (err != cudaSuccess) {
          log::Error("llama_forward", "FlashDecode launch failed: " +
                                          std::string(cudaGetErrorString(err)));
          return false;
        }
      }
      pt.attn_ms += pt.Mark();

      // O projection + residual accumulation
      bool ffn_norm_precomputed = false;
      bool ffn_q81_precomputed = false;
      {
        NVTX_SCOPE("O_Projection");
        auto o_raw = weights_->LayerOProjRaw(layer);
        bool o_accumulated = false;
        // S8: Q4_K MMA o_proj at M >= 2 — writes d_norm_out_ (the
        // ResidualAdd path below consumes it); dp4a MMQ measured ~106us
        // vs ~20us for MMA at this geometry.
        const bool o_mma_done = [&, this]() {
          if constexpr (std::is_same_v<T, half>) {
            return B >= 2 &&
                   TryQ8_1MmaGemv<T>(o_raw, d_attn_out_, d_norm_out_,
                                     d_act_q8_1_mmq_, d_mma_partials_, B,
                                     hidden_size_, num_heads_ * head_dim_,
                                     stream_, "o_proj", active_policy);
          } else {
            return false;
          }
        }();
        // Try accumulate mode: write directly to residual, skip ResidualAdd
        if constexpr (std::is_same_v<T, half>) {
          if (!o_mma_done && fp32_residual_active_ &&
              TryQ8_1GemvAccumF32(o_raw, d_attn_out_, d_residual_f32_,
                                  d_act_q8_1_, B, hidden_size_,
                                  num_heads_ * head_dim_, stream_, "o_proj",
                                  active_policy)) {
            o_accumulated = true;
          }
        }
        if (!o_mma_done && !o_accumulated && !fp32_residual_active_ &&
            TryQ8_1GemvAccum<T>(o_raw, d_attn_out_, d_residual_, d_act_q8_1_, B,
                                hidden_size_, num_heads_ * head_dim_, stream_,
                                "o_proj", active_policy)) {
          o_accumulated = true;
        }
        if (!o_mma_done && !o_accumulated &&
            !TryQ8_1Gemv<T>(o_raw, d_attn_out_, d_norm_out_, d_act_q8_1_, B,
                            hidden_size_, num_heads_ * head_dim_, stream_,
                            "o_proj", active_policy) &&
            !TryPackedGemv<T>(o_raw, d_attn_out_, d_norm_out_,
                              d_packed_activation_, d_packed_activation_scales_,
                              B, hidden_size_, num_heads_ * head_dim_, stream_,
                              "o_proj", active_policy)) {
          if (capturing) {
            capture_abort = true;
            return false;
          }
          const T *o_proj =
              reinterpret_cast<const T *>(weights_->LayerOProj(layer));
          if (fp32_residual_active_) {
            gemm_->GemmTyped<T>(B, hidden_size_, num_heads_ * head_dim_,
                                d_attn_out_, o_proj, d_norm_out_);
          } else {
            gemm_->GemmTypedAccum<T>(B, hidden_size_, num_heads_ * head_dim_,
                                     d_attn_out_, o_proj, d_residual_);
            o_accumulated = true;
          }
        }

        // Residual add (skip if accumulated directly)
        if (!o_accumulated) {
          if (fp32_residual_active_) {
            if (active_policy->enable_fused_residual_norm) {
              cuda_kernel::ResidualAddRmsNormMixed<T>(
                  d_residual_f32_, d_norm_out_, post_attn_norm, d_norm_out_, B,
                  hidden_size_, rms_norm_eps_, stream_);
              ffn_norm_precomputed = true;
            } else {
              cuda_kernel::ResidualAddMixed<T>(d_residual_f32_, d_norm_out_,
                                               B * hidden_size_, stream_);
            }
          } else {
            if (active_policy->enable_fused_residual_norm) {
              cuda_kernel::ResidualAddRmsNorm<T>(
                  d_residual_, d_norm_out_, post_attn_norm, d_norm_out_, B,
                  hidden_size_, rms_norm_eps_, stream_);
              ffn_norm_precomputed = true;
            } else {
              cuda_kernel::ResidualAdd<T>(d_residual_, d_norm_out_,
                                          B * hidden_size_, stream_);
            }
          }
        }

        // P2 epilogue: after O-proj accumulate, fuse norm+quant into one kernel
        if constexpr (std::is_same_v<T, half>) {
          if (o_accumulated && !ffn_norm_precomputed &&
              active_policy->enable_fused_gemv_norm_quant_epilogue &&
              d_act_q8_1_) {
            // FP32 residual: FinishNormQuantQ8_1 reads half residual, so we
            // pre-normalize from FP32 instead and set flags.
            if (fp32_residual_active_) {
              cuda_kernel::RmsNormMixed<T>(d_residual_f32_, post_attn_norm,
                                           d_norm_out_, B, hidden_size_,
                                           rms_norm_eps_, stream_);
              FusedQuantGemm::QuantizeRowQ8_1(d_norm_out_, d_act_q8_1_, B,
                                              hidden_size_, stream_);
            } else {
              cuda_kernel::FinishNormQuantQ8_1(
                  d_residual_, post_attn_norm, d_norm_out_, d_act_q8_1_, B,
                  hidden_size_, rms_norm_eps_, stream_);
            }
            ffn_norm_precomputed = true;
            ffn_q81_precomputed = true;
          }
        }
      }
      pt.o_proj_ms += pt.Mark();

      // FFN block
      // FP32 residual: pre-normalize if needed
      if (fp32_residual_active_ && !ffn_norm_precomputed) {
        cuda_kernel::RmsNormMixed<T>(d_residual_f32_, post_attn_norm,
                                     d_norm_out_, B, hidden_size_,
                                     rms_norm_eps_, stream_);
        ffn_norm_precomputed = true;
      }
      const T *ffn_input = ffn_norm_precomputed
                               ? static_cast<const T *>(d_norm_out_)
                               : static_cast<const T *>(d_residual_);
      const T *ffn_norm_weight =
          ffn_norm_precomputed ? nullptr : post_attn_norm;
      bool down_accumulated = false;
      bool fused_gate_up_silu = false;
      bool gate_up_q81_epilogue = false;
      {
        NVTX_SCOPE("FFN");
        auto gate_raw = weights_->LayerGateProjRaw(layer);
        auto up_raw = weights_->LayerUpProjRaw(layer);
        const auto ffn_selected_op = SelectInferfluxCudaFfnProjOperator(
            gate_raw, up_raw, InferfluxCudaDispatchPhase::kDecode,
            FusedDispatchGeometry{B, intermediate_size_, hidden_size_, 2, true,
                                  !ffn_norm_precomputed},
            g_allow_fused_quantized_matmul, *active_policy);

        // Try fused gate+up+SiLU MMVQ: single kernel reads both weight
        // matrices and applies SiLU activation at write-back, halving
        // weight memory traffic vs separate gate+up projections. However,
        // measured decode throughput on the live grouped FFN path shows the
        // MMQ3/row-quad grouped kernels overtaking fusion at M>=2, so keep
        // those specialized grouped operators on the critical path.
        if constexpr (std::is_same_v<T, half>) {
          const bool prefer_grouped_q81_pair_fast_path =
              ffn_selected_op ==
                  FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3 ||
              ffn_selected_op ==
                  FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4;
          if (active_policy->enable_fused_gate_up_silu &&
              (active_policy->fused_gate_up_silu_max_batch < 0 ||
               B <= active_policy->fused_gate_up_silu_max_batch) &&
              !prefer_grouped_q81_pair_fast_path &&
              !Q81ActivationsDisabled(*active_policy) && d_act_q8_1_ &&
              gate_raw.data && up_raw.data &&
              gate_raw.quant_type == up_raw.quant_type &&
              FusedQuantGemm::SupportsQ8_1Activations(gate_raw.quant_type) &&
              FusedQuantGemm::ShouldUseFusedPath(
                  gate_raw.quant_type,
                  FusedDispatchGeometry{B, intermediate_size_, hidden_size_, 2,
                                        true, !ffn_norm_precomputed})) {
            NVTX_SCOPE("FusedGateUpSiLU");
            if (ffn_q81_precomputed) {
              // P2 already produced valid Q8_1 — skip re-quantization
            } else if (ffn_norm_precomputed) {
              FusedQuantGemm::QuantizeRowQ8_1(d_norm_out_, d_act_q8_1_, B,
                                              hidden_size_, stream_);
            } else {
              FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
                  d_residual_, post_attn_norm, d_act_q8_1_, B, hidden_size_,
                  rms_norm_eps_, stream_);
            }
            // P5: Try gate+up+SiLU with Q8_1 epilogue first — produces
            // pre-quantized Q8_1 output for down-proj, saving a separate
            // QuantizeRowQ8_1 kernel launch.
            if (active_policy->enable_gate_up_silu_q81_epilogue) {
              fused_gate_up_silu =
                  FusedQuantGemm::FusedGateUpSiluGemvQ8_1WithEpilogue(
                      gate_raw, up_raw, d_act_q8_1_, d_ffn_gate_,
                      d_ffn_act_q8_1_, B, intermediate_size_, hidden_size_,
                      stream_);
              if (fused_gate_up_silu) {
                gate_up_q81_epilogue = true;
                LogPackedGemmPath(
                    "ffn_gate_up",
                    "using fused gate+up+SiLU+Q8_1 epilogue MMVQ kernel");
              }
            }
            if (!fused_gate_up_silu) {
              fused_gate_up_silu = FusedQuantGemm::FusedGateUpSiluGemvQ8_1(
                  gate_raw, up_raw, d_act_q8_1_, d_ffn_gate_, B,
                  intermediate_size_, hidden_size_, stream_);
              if (fused_gate_up_silu) {
                LogPackedGemmPath("ffn_gate_up",
                                  "using fused gate+up+SiLU MMVQ kernel");
              }
            }
          }
        }

        if (!fused_gate_up_silu) {
          const std::array<PackedProjectionPlan<T>, 2> ffn_plans = {
              {{gate_raw, d_ffn_gate_, intermediate_size_},
               {up_raw, d_ffn_up_, intermediate_size_}}};
          const std::string ffn_quant =
              ProjectionGroupQuantLabel(gate_raw, up_raw);
          NativeFfnExecutionSummary ffn_summary;
          if (!ExecuteInferfluxCudaFfnProjectionStage(
                  ffn_selected_op, "decode", ffn_quant, gate_raw.quant_type, B,
                  intermediate_size_, hidden_size_,
                  [&]() {
                    // S8: Q4_K MMA gate+up pair at M >= 2 (12x over the
                    // grouped families at M=16 in the isolated rig).
                    // Produces raw gate/up into d_ffn_gate_/d_ffn_up_ — the
                    // down-proj stage's SiluMul consumers take exactly that.
                    if constexpr (std::is_same_v<T, half>) {
                      if (B >= 2) {
                        if (!ffn_norm_precomputed) {
                          cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm,
                                                  d_norm_out_, B, hidden_size_,
                                                  rms_norm_eps_, stream_);
                        }
                        if (TryQ8_1MmaGemv<T>(
                                gate_raw, d_norm_out_, d_ffn_gate_,
                                d_act_q8_1_mmq_, d_mma_partials_, B,
                                intermediate_size_, hidden_size_, stream_,
                                "gate_proj", active_policy) &&
                            TryQ8_1MmaGemv<T>(
                                up_raw, d_norm_out_, d_ffn_up_,
                                d_act_q8_1_mmq_, d_mma_partials_, B,
                                intermediate_size_, hidden_size_, stream_,
                                "up_proj", active_policy)) {
                          return true;
                        }
                      }
                    }
                    return TryQ8_1ProjectionGroup(
                        ffn_plans, ffn_input, ffn_norm_weight, d_act_q8_1_, B,
                        hidden_size_, rms_norm_eps_, stream_, active_policy,
                        ffn_selected_op, ffn_q81_precomputed);
                  },
                  [&]() {
                    return TryPackedProjectionGroup(
                        ffn_plans, ffn_input, ffn_norm_weight, d_norm_out_,
                        d_packed_activation_, d_packed_activation_scales_, B,
                        hidden_size_, rms_norm_eps_, stream_, active_policy);
                  },
                  [&]() {
                    bool ffn_norm_computed = ffn_norm_precomputed;

                    if (!ExecuteNativeNormalizedProjectionStage(
                            &ffn_norm_computed,
                            [&]() {
                              cuda_kernel::RmsNorm<T>(
                                  d_residual_, post_attn_norm, d_norm_out_, B,
                                  hidden_size_, rms_norm_eps_, stream_);
                              return true;
                            },
                            [&]() {
                              return TryQ8_1MmaGemv<T>(
                                         gate_raw, d_norm_out_, d_ffn_gate_,
                                         d_act_q8_1_mmq_, d_mma_partials_, B,
                                         intermediate_size_, hidden_size_,
                                         stream_, "gate_proj",
                                         active_policy) ||
                                     TryQ8_1Gemv<T>(gate_raw, d_norm_out_,
                                                    d_ffn_gate_, d_act_q8_1_, B,
                                                    intermediate_size_,
                                                    hidden_size_, stream_,
                                                    "gate_proj", active_policy);
                            },
                            [&]() {
                              if (capturing) {
                                capture_abort = true;
                                return false;
                              }
                              const T *gate_proj = reinterpret_cast<const T *>(
                                  weights_->LayerGateProj(layer));
                              gemm_->GemmTyped<T>(B, intermediate_size_,
                                                  hidden_size_, d_norm_out_,
                                                  gate_proj, d_ffn_gate_);
                              return true;
                            })) {
                      return false;
                    }

                    if (!ExecuteNativeNormalizedProjectionStage(
                            &ffn_norm_computed,
                            [&]() {
                              cuda_kernel::RmsNorm<T>(
                                  d_residual_, post_attn_norm, d_norm_out_, B,
                                  hidden_size_, rms_norm_eps_, stream_);
                              return true;
                            },
                            [&]() {
                              return TryQ8_1MmaGemv<T>(
                                         up_raw, d_norm_out_, d_ffn_up_,
                                         d_act_q8_1_mmq_, d_mma_partials_, B,
                                         intermediate_size_, hidden_size_,
                                         stream_, "up_proj", active_policy) ||
                                     TryQ8_1Gemv<T>(
                                         up_raw, d_norm_out_, d_ffn_up_,
                                         d_act_q8_1_, B, intermediate_size_,
                                         hidden_size_, stream_, "up_proj",
                                         active_policy);
                            },
                            [&]() {
                              if (capturing) {
                                capture_abort = true;
                                return false;
                              }
                              const T *up_proj = reinterpret_cast<const T *>(
                                  weights_->LayerUpProj(layer));
                              gemm_->GemmTyped<T>(B, intermediate_size_,
                                                  hidden_size_, d_norm_out_,
                                                  up_proj, d_ffn_up_);
                              return true;
                            })) {
                      return false;
                    }
                    return true;
                  },
                  &ffn_summary, active_policy)) {
            return false;
          }
        } // end if (!fused_gate_up_silu)
        pt.ffn_proj_ms += pt.Mark();

        auto down_raw = weights_->LayerDownProjRaw(layer);
        auto down_mmq = weights_->LayerDownProjMmq(layer);

        if (fused_gate_up_silu) {
          // SiLU already applied by fused kernel — d_ffn_gate_ has the
          // post-activation FP16 result.  When P5 is active, consume the
          // epilogue's pre-quantized Q8_1 buffer directly before falling back
          // to the legacy re-quantization path.
          const bool mma_down_ok = TryMmqMmaGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_down_, d_act_q8_1_mmq_,
              d_mma_partials_, B, hidden_size_, intermediate_size_, stream_,
              "down_proj", active_policy);
          bool down_ok = mma_down_ok;
          if constexpr (std::is_same_v<T, half>) {
            if (!down_ok && fp32_residual_active_) {
              if (gate_up_q81_epilogue) {
                down_ok = TryQ8_1GemvPrequantizedAccumF32(
                    down_raw, d_ffn_act_q8_1_, d_residual_f32_, B, hidden_size_,
                    intermediate_size_, stream_, "down_proj", active_policy);
              }
              if (!down_ok) {
                down_ok = TryQ8_1GemvAccumF32(
                    down_raw, d_ffn_gate_, d_residual_f32_, d_act_q8_1_, B,
                    hidden_size_, intermediate_size_, stream_, "down_proj",
                    active_policy);
              }
            }
          }
          if (!down_ok && !fp32_residual_active_) {
            if (gate_up_q81_epilogue) {
              down_ok = TryQ8_1GemvPrequantizedAccum<T>(
                  down_raw, d_ffn_act_q8_1_, d_residual_, B, hidden_size_,
                  intermediate_size_, stream_, "down_proj", active_policy);
            }
            if (!down_ok) {
              down_ok = TryQ8_1GemvAccum<T>(down_raw, d_ffn_gate_, d_residual_,
                                            d_act_q8_1_, B, hidden_size_,
                                            intermediate_size_, stream_,
                                            "down_proj", active_policy);
            }
          }
          if (down_ok && !mma_down_ok) {
            down_accumulated = true;
          }
          if (!down_ok) {
            if (gate_up_q81_epilogue) {
              down_ok = TryQ8_1GemvPrequantized<T>(
                  down_raw, d_ffn_act_q8_1_, d_ffn_down_, B, hidden_size_,
                  intermediate_size_, stream_, "down_proj", active_policy);
            }
            if (!down_ok) {
              down_ok = TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                       d_act_q8_1_, B, hidden_size_,
                                       intermediate_size_, stream_, "down_proj",
                                       active_policy);
            }
          }
          if (!down_ok) {
            if (gate_up_q81_epilogue) {
              down_ok = TryMmqGemvPrequantized<T>(
                  down_mmq, d_ffn_act_q8_1_, d_ffn_down_, B, hidden_size_,
                  intermediate_size_, stream_, "down_proj", active_policy);
            }
            if (!down_ok) {
              down_ok =
                  TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_, d_act_q8_1_,
                                B, hidden_size_, intermediate_size_, stream_,
                                "down_proj", active_policy);
            }
          }
          if (!down_ok) {
            if (capturing) {
              capture_abort = true;
              return false;
            }
            const T *down_proj =
                reinterpret_cast<const T *>(weights_->LayerDownProj(layer));
            if (fp32_residual_active_) {
              gemm_->GemmTyped<T>(B, hidden_size_, intermediate_size_,
                                  d_ffn_gate_, down_proj, d_ffn_down_);
              // Non-accumulated — residual add handled below
            } else {
              gemm_->GemmTypedAccum<T>(B, hidden_size_, intermediate_size_,
                                       d_ffn_gate_, down_proj, d_residual_);
              down_accumulated = true;
            }
          }
        } else {
          const auto down_selected_op = SelectInferfluxCudaDownProjOperator(
              down_raw, down_mmq, InferfluxCudaDispatchPhase::kDecode,
              FusedDispatchGeometry{B, hidden_size_, intermediate_size_, 1,
                                    true, false},
              g_allow_fused_quantized_matmul, *active_policy);
          NativeDownProjExecutionSummary down_summary;
          if (!ExecuteInferfluxCudaDownProjStage(
                  down_selected_op, "decode", ProjectionQuantLabel(down_raw),
                  down_raw.quant_type, B, hidden_size_, intermediate_size_,
                  [&]() {
                    return TryMmqMmaSiluMul<T>(down_raw, d_ffn_gate_, d_ffn_up_,
                                               d_ffn_down_, d_act_q8_1_mmq_,
                                               d_mma_partials_, B, hidden_size_,
                                               intermediate_size_, stream_,
                                               "down_proj", active_policy) ||
                           TryMmqSiluMulGemv<T>(
                               down_mmq, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                               d_act_q8_1_, B, hidden_size_, intermediate_size_,
                               stream_, "down_proj", active_policy);
                  },
                  [&]() {
                    if constexpr (std::is_same_v<T, half>) {
                      if (fp32_residual_active_ &&
                          TryQ8_1SiluMulGemvAccumF32(
                              down_raw, d_ffn_gate_, d_ffn_up_, d_residual_f32_,
                              d_act_q8_1_, B, hidden_size_, intermediate_size_,
                              stream_, "down_proj", active_policy)) {
                        down_accumulated = true;
                        return true;
                      }
                    }
                    if (!fp32_residual_active_ &&
                        TryQ8_1SiluMulGemvAccum<T>(
                            down_raw, d_ffn_gate_, d_ffn_up_, d_residual_,
                            d_act_q8_1_, B, hidden_size_, intermediate_size_,
                            stream_, "down_proj", active_policy)) {
                      down_accumulated = true;
                      return true;
                    }
                    return TryQ8_1SiluMulGemv<T>(
                        down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                        d_act_q8_1_, B, hidden_size_, intermediate_size_,
                        stream_, "down_proj", active_policy);
                  },
                  [&]() {
                    return TryPackedSiluMulGemv<T>(
                        down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                        d_packed_activation_, d_packed_activation_scales_, B,
                        hidden_size_, intermediate_size_, stream_, "down_proj",
                        active_policy);
                  },
                  [&]() {
                    cuda_kernel::SiluMul<T>(d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                                            B * intermediate_size_, stream_);
                    pt.ffn_silu_ms += pt.Mark();

                    bool down_ok = false;
                    if (down_selected_op ==
                        FusedQuantGemm::DownProjOperator::kMmq) {
                      down_ok =
                          TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                        d_act_q8_1_, B, hidden_size_,
                                        intermediate_size_, stream_,
                                        "down_proj", active_policy) ||
                          TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                         d_act_q8_1_, B, hidden_size_,
                                         intermediate_size_, stream_,
                                         "down_proj", active_policy);
                    } else {
                      down_ok =
                          TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                         d_act_q8_1_, B, hidden_size_,
                                         intermediate_size_, stream_,
                                         "down_proj", active_policy) ||
                          TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                        d_act_q8_1_, B, hidden_size_,
                                        intermediate_size_, stream_,
                                        "down_proj", active_policy);
                    }
                    if (!down_ok &&
                        !TryPackedGemv<T>(
                            down_raw, d_ffn_gate_, d_ffn_down_,
                            d_packed_activation_, d_packed_activation_scales_,
                            B, hidden_size_, intermediate_size_, stream_,
                            "down_proj", active_policy)) {
                      if (capturing) {
                        capture_abort = true;
                        return false;
                      }
                      const T *down_proj = reinterpret_cast<const T *>(
                          weights_->LayerDownProj(layer));
                      if (fp32_residual_active_) {
                        gemm_->GemmTyped<T>(B, hidden_size_, intermediate_size_,
                                            d_ffn_gate_, down_proj,
                                            d_ffn_down_);
                      } else {
                        gemm_->GemmTypedAccum<T>(
                            B, hidden_size_, intermediate_size_, d_ffn_gate_,
                            down_proj, d_residual_);
                        down_accumulated = true;
                      }
                    }
                    return true;
                  },
                  [&](FusedQuantGemm::DownProjOperator actual_op) {
                    const std::string down_label =
                        std::string("selected down-proj operator: ") +
                        FusedQuantGemm::DownProjOperatorName(actual_op);
                    LogPackedGemmPath("down_proj", down_label.c_str());
                  },
                  &down_summary, active_policy)) {
            return false;
          }
        } // end else (!fused_gate_up_silu)
      }

      // Residual add (skip if accumulated directly)
      if (!down_accumulated) {
        if (fp32_residual_active_) {
          if (active_policy->enable_fused_residual_norm &&
              layer < num_layers_ - 1) {
            const T *next_input_norm = reinterpret_cast<const T *>(
                weights_->LayerInputNorm(layer + 1));
            cuda_kernel::ResidualAddRmsNormMixed<T>(
                d_residual_f32_, d_ffn_down_, next_input_norm, d_norm_out_, B,
                hidden_size_, rms_norm_eps_, stream_);
            input_norm_precomputed = true;
          } else {
            cuda_kernel::ResidualAddMixed<T>(d_residual_f32_, d_ffn_down_,
                                             B * hidden_size_, stream_);
          }
        } else {
          if (active_policy->enable_fused_residual_norm &&
              layer < num_layers_ - 1) {
            const T *next_input_norm = reinterpret_cast<const T *>(
                weights_->LayerInputNorm(layer + 1));
            cuda_kernel::ResidualAddRmsNorm<T>(
                d_residual_, d_ffn_down_, next_input_norm, d_norm_out_, B,
                hidden_size_, rms_norm_eps_, stream_);
            input_norm_precomputed = true;
          } else {
            cuda_kernel::ResidualAdd<T>(d_residual_, d_ffn_down_,
                                        B * hidden_size_, stream_);
          }
        }
      }

      // P2 epilogue: after down-proj accumulate, fuse norm+quant for next layer
      if constexpr (std::is_same_v<T, half>) {
        if (down_accumulated && !input_norm_precomputed &&
            active_policy->enable_fused_gemv_norm_quant_epilogue &&
            d_act_q8_1_ && layer < num_layers_ - 1) {
          const T *next_input_norm =
              reinterpret_cast<const T *>(weights_->LayerInputNorm(layer + 1));
          if (fp32_residual_active_) {
            cuda_kernel::RmsNormMixed<T>(d_residual_f32_, next_input_norm,
                                         d_norm_out_, B, hidden_size_,
                                         rms_norm_eps_, stream_);
            FusedQuantGemm::QuantizeRowQ8_1(d_norm_out_, d_act_q8_1_, B,
                                            hidden_size_, stream_);
          } else {
            cuda_kernel::FinishNormQuantQ8_1(
                d_residual_, next_input_norm, d_norm_out_, d_act_q8_1_, B,
                hidden_size_, rms_norm_eps_, stream_);
          }
          input_norm_precomputed = true;
          input_q81_precomputed = true;
        }
      }
      pt.ffn_down_ms += pt.Mark();
    }

    // Final RMSNorm + LM head
    {
      NVTX_SCOPE("LM_Head");
      const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());
      auto lm_raw = weights_->LmHeadRaw();
      // Q8_1 fused RmsNorm+Quantize+GEMV for LM head
      bool lm_q8_1_ok = false;
      if constexpr (std::is_same_v<T, half>) {
        if (!Q81ActivationsDisabled(*active_policy) && lm_raw.data &&
            d_act_q8_1_ &&
            FusedQuantGemm::SupportsQ8_1Activations(lm_raw.quant_type) &&
            FusedQuantGemm::ShouldUseFusedPath(
                lm_raw.quant_type,
                FusedDispatchGeometry{B, vocab_size_, hidden_size_, 1, true,
                                      true})) {
          if (fp32_residual_active_) {
            FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
                d_residual_f32_, final_norm, d_act_q8_1_, B, hidden_size_,
                rms_norm_eps_, stream_);
          } else {
            FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
                d_residual_, final_norm, d_act_q8_1_, B, hidden_size_,
                rms_norm_eps_, stream_);
          }
          lm_q8_1_ok = FusedQuantGemm::GemvQ8_1(
              lm_raw, d_act_q8_1_, d_logits_typed_, B, vocab_size_,
              hidden_size_, stream_, active_policy);
        }
      }
      if (!lm_q8_1_ok) {
        // Pre-normalize from FP32 for fallback paths
        if (fp32_residual_active_) {
          cuda_kernel::RmsNormMixed<T>(d_residual_f32_, final_norm, d_norm_out_,
                                       B, hidden_size_, rms_norm_eps_, stream_);
        }
      }
      if (!lm_q8_1_ok &&
          (fp32_residual_active_ ||
           !TryPackedRmsNormGemv<T>(
               lm_raw, d_residual_, final_norm, d_norm_out_,
               d_packed_activation_, d_packed_activation_scales_,
               d_logits_typed_, B, vocab_size_, hidden_size_, rms_norm_eps_,
               stream_, "lm_head", active_policy))) {
        if (!fp32_residual_active_) {
          cuda_kernel::RmsNorm<T>(d_residual_, final_norm, d_norm_out_, B,
                                  hidden_size_, rms_norm_eps_, stream_);
        }
        // d_norm_out_ now has the normalized half output (both paths)
        if (!TryQ8_1Gemv<T>(lm_raw, d_norm_out_, d_logits_typed_, d_act_q8_1_,
                            B, vocab_size_, hidden_size_, stream_, "lm_head",
                            active_policy)) {
          if (capturing) {
            capture_abort = true;
            return false;
          }
          const T *lm_head = reinterpret_cast<const T *>(weights_->LmHead());
          gemm_->GemmTyped<T>(B, vocab_size_, hidden_size_, d_norm_out_,
                              lm_head, d_logits_typed_);
        }
      }

      cuda_kernel::HalfToFloat<T>(d_logits_typed_, d_logits, B * vocab_size_,
                                  stream_);
    }
    pt.lm_head_ms += pt.Mark();
    pt.Report(num_layers_, 1);

    return true;
  };

  // Execute compute (captured into graph or run directly)
  bool compute_ok = RunCompute();

  if (capturing) {
    // End capture regardless of compute success — CUDA requires it.
    err = cudaStreamEndCapture(stream_, &decode_graph_);
    // Clear any sticky CUDA errors from failed operations during capture
    // (e.g., lazy weight dequantization calling cudaStreamSynchronize).
    // Without this, the re-execution path below inherits the error state.
    cudaGetLastError();

    if (capture_abort) {
      // Fused kernel fallback was needed but skipped during capture.
      // Discard the incomplete graph and retry on next call.
      if (decode_graph_) {
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
      }
      GlobalMetrics().RecordInferfluxCudaDispatchDivergence(
          "graph", "fused", "cublas", "graph_capture_abort");
      if (--graph_retry_remaining_ <= 0) {
        log::Warn("llama_forward",
                  "CUDA graph permanently disabled after capture aborts");
        graph_enabled_ = false;
      } else {
        graph_warmup_remaining_ = 2;
        log::Info("llama_forward", "CUDA graph capture aborted, will retry (" +
                                       std::to_string(graph_retry_remaining_) +
                                       " attempts remaining)");
      }
      return RunCompute();
    }

    if (err == cudaSuccess && decode_graph_) {
      err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr,
                                 nullptr, 0);
      if (err == cudaSuccess) {
        graph_batch_size_ = B;
        size_t num_nodes = 0;
        cudaGraphGetNodes(decode_graph_, nullptr, &num_nodes);
        log::Info("llama_forward",
                  "CUDA graph captured for B=" + std::to_string(B) + " (" +
                      std::to_string(num_layers_) + " layers, " +
                      std::to_string(num_nodes) + " nodes)");
        err = cudaGraphLaunch(decode_graph_exec_, stream_);
        return err == cudaSuccess;
      }
      log::Warn("llama_forward", "CUDA graph instantiate failed: " +
                                     std::string(cudaGetErrorString(err)));
      cudaGraphDestroy(decode_graph_);
      decode_graph_ = nullptr;
    }
    // Graph capture failed — retry on next call
    if (--graph_retry_remaining_ <= 0) {
      log::Warn("llama_forward",
                "CUDA graph permanently disabled after capture failures");
      graph_enabled_ = false;
    } else {
      graph_warmup_remaining_ = 2;
      log::Warn("llama_forward", "CUDA graph capture failed, will retry (" +
                                     std::to_string(graph_retry_remaining_) +
                                     " attempts remaining)");
    }
    return RunCompute();
  }

  return true;
}

// ============================================================================
// EmbedForward: mean-pooled embedding extraction
// ============================================================================

template <typename T>
bool LlamaForwardTyped<T>::EmbedForward(const std::vector<int> &token_ids,
                                        int sequence_id, float *d_output) {
  NVTX_SCOPE("EmbedForward");
  const int seq_len = static_cast<int>(token_ids.size());
  if (seq_len <= 0 || !d_output)
    return false;

  // Run the full forward pass (embedding + all layers + LM head).
  // We discard the logits but keep d_residual_ which holds the final
  // hidden states after all transformer layers.
  // Allocate a temporary FP32 logits buffer for the throwaway output.
  float *d_throwaway = nullptr;
  cudaError_t alloc_err = cudaMalloc(&d_throwaway, vocab_size_ * sizeof(float));
  if (alloc_err != cudaSuccess) {
    log::Error("llama_forward", "EmbedForward: cudaMalloc for logits failed");
    return false;
  }

  bool fwd_ok = Forward(token_ids, 0, sequence_id, d_throwaway);
  cudaFree(d_throwaway);
  if (!fwd_ok) {
    log::Error("llama_forward", "EmbedForward: forward pass failed");
    return false;
  }

  // d_residual_ now holds [seq_len, hidden_size] after all layers.
  // Apply final RmsNorm to ALL positions and mean-pool.
  {
    NVTX_SCOPE("EmbedPool");
    const T *final_norm_w = reinterpret_cast<const T *>(weights_->FinalNorm());

    // RmsNorm all positions: d_residual_ -> d_norm_out_
    cudaError_t err;
    if (fp32_residual_active_) {
      err = cuda_kernel::RmsNormMixed<T>(d_residual_f32_, final_norm_w,
                                         d_norm_out_, seq_len, hidden_size_,
                                         rms_norm_eps_, stream_);
    } else {
      err = cuda_kernel::RmsNorm<T>(d_residual_, final_norm_w, d_norm_out_,
                                    seq_len, hidden_size_, rms_norm_eps_,
                                    stream_);
    }
    if (err != cudaSuccess) {
      log::Error("llama_forward", "EmbedForward: final norm failed");
      return false;
    }

    // Mean-pool across token positions: [seq_len, hidden_size] -> [hidden_size]
    err = cuda_kernel::MeanPool<T>(d_norm_out_, d_output, seq_len, hidden_size_,
                                   stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "EmbedForward: mean pool failed");
      return false;
    }
  }

  // Free the KV cache slot used for the forward pass.
  if (kv_cache_) {
    kv_cache_->ClearSequenceAsync(sequence_id, stream_);
  }

  return true;
}

// Snapshot a tensor to device memory for later download
template <typename T>
template <typename TensorT>
void LlamaForwardTyped<T>::SnapshotTensorToDevice(
    const std::string &operation, int layer_idx, const TensorT *device_ptr,
    const std::vector<int> &shape) {

  if (!capture_attention_tensors_) {
    return;
  }

  // Calculate total number of elements
  size_t num_elements = 1;
  for (int dim : shape) {
    num_elements *= dim;
  }

  if (num_elements == 0) {
    return;
  }

  // Allocate device memory for float storage
  float *d_snapshot = nullptr;
  cudaError_t err = cudaMalloc(&d_snapshot, num_elements * sizeof(float));
  if (err != cudaSuccess) {
    log::Error("llama_forward",
               "Failed to allocate device memory for tensor snapshot");
    return;
  }

  // Convert and copy tensor data
  if constexpr (std::is_same_v<TensorT, half> ||
                std::is_same_v<TensorT, __nv_bfloat16>) {
    // Convert half/bfloat16 to float
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    cuda_kernel::ConvertHalfToFloat(device_ptr, d_snapshot, num_elements,
                                    stream_);
  } else if constexpr (std::is_same_v<TensorT, float>) {
    // Already float, just copy
    cudaMemcpy(d_snapshot, device_ptr, num_elements * sizeof(float),
               cudaMemcpyDeviceToDevice);
  } else {
    // Unsupported type
    cudaFree(d_snapshot);
    return;
  }

  // Store snapshot info for later download
  DeviceTensorSnapshot snapshot;
  snapshot.layer_idx = layer_idx;
  snapshot.operation = operation;
  snapshot.d_data = d_snapshot;
  snapshot.num_elements = num_elements;
  snapshot.shape = shape;

  device_snapshots_.push_back(snapshot);
}

template <typename T>
AttentionTensorData LlamaForwardTyped<T>::CaptureAttentionTensors() {
  AttentionTensorData result;
  result.ok = false;

  // Check if attention tensor capture is enabled
  const char *env_var = std::getenv("INFERFLUX_DEBUG_ATTENTION_TENSORS");
  if (!env_var || std::string(env_var) != "1") {
    result.error = "Attention tensor capture not enabled. Set "
                   "INFERFLUX_DEBUG_ATTENTION_TENSORS=1";
    return result;
  }

  // Download all snapshots to host
  result.snapshots.reserve(device_snapshots_.size());
  for (const auto &device_snap : device_snapshots_) {
    AttentionTensorSnapshot host_snap;
    host_snap.layer_idx = device_snap.layer_idx;
    host_snap.operation = device_snap.operation;
    host_snap.shape = device_snap.shape;
    host_snap.data.resize(device_snap.num_elements);

    // Copy from device to host
    cudaError_t err = cudaMemcpy(host_snap.data.data(), device_snap.d_data,
                                 device_snap.num_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      result.error = "Failed to download tensor snapshot from device";
      // Clean up remaining device memory
      for (const auto &snap : device_snapshots_) {
        cudaFree(snap.d_data);
      }
      device_snapshots_.clear();
      return result;
    }

    result.snapshots.push_back(host_snap);

    // Free device memory
    cudaFree(device_snap.d_data);
  }

  device_snapshots_.clear();
  result.ok = true;
  return result;
}

// Explicit template instantiations
template class LlamaForwardTyped<half>;
template class LlamaForwardTyped<__nv_bfloat16>;

} // namespace inferflux
