#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/native_dispatch_registry.h"
// V1 kernels still live: packed dispatch, Q8_1 group dispatch
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_grouped_ffn.cuh"
// MMVQ: weight-read-first batch-amortized kernels (Phase 0/1 of kernel rewrite)
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"
// MMQ: tiled quantized GEMM for batch 9-64 (Phase 2 of kernel rewrite)
#include "runtime/backends/cuda/native/kernels/mmq.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "server/logging/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>

namespace inferflux {

using namespace runtime::cuda::native;

namespace {

thread_local const NativeExecutionPolicy *g_execution_policy_override = nullptr;

const NativeExecutionPolicy &
ResolveExecutionPolicy(const NativeExecutionPolicy *policy) {
  if (policy) {
    return *policy;
  }
  if (g_execution_policy_override) {
    return *g_execution_policy_override;
  }
  static thread_local NativeExecutionPolicy env_policy;
  env_policy = NativeExecutionPolicy::FromEnv();
  return env_policy;
}

class ScopedExecutionPolicyOverride {
public:
  explicit ScopedExecutionPolicyOverride(const NativeExecutionPolicy *policy)
      : prev_(g_execution_policy_override) {
    if (policy) {
      g_execution_policy_override = policy;
    }
  }

  ~ScopedExecutionPolicyOverride() { g_execution_policy_override = prev_; }

private:
  const NativeExecutionPolicy *prev_{nullptr};
};

// ============================================================================
// GPU-adaptive threshold computation
// ============================================================================

struct GpuProfile {
  int sm_major{0};
  int sm_minor{0};
  float memory_bandwidth_gb_s{0}; // GB/s
  int sm_count{0};
  bool initialized{false};
  bool has_dp4a{false}; // SM 6.1+ supports __dp4a int8x4 dot product

  // Base M threshold before scaling by bits-per-weight.
  // This reflects how much faster cuBLAS tensor cores are than our scalar
  // fused kernels — lower means cuBLAS wins sooner.
  int base_threshold() const {
    if (sm_major < 7) {
      // Pascal and older: no tensor cores. cuBLAS uses scalar FP16 GEMM,
      // no significant compute advantage over fused kernels.
      // Fused wins on memory bandwidth savings up to high M.
      return 16;
    }
    if (sm_major == 7 && sm_minor < 5) {
      // Volta (V100): first-gen FP16 tensor cores.
      // cuBLAS gets ~8x compute throughput boost over scalar.
      return 8;
    }
    if (sm_major == 7) {
      // Turing (RTX 20xx): FP16 tensor cores, lower bandwidth than V100.
      return 6;
    }
    if (sm_major == 8 && sm_minor == 0) {
      // Ampere A100: enhanced tensor cores, 2 TB/s HBM2e.
      // Very fast cuBLAS — lower threshold.
      return 4;
    }
    if (sm_major == 8) {
      // Ampere consumer (RTX 30xx) / Ada Lovelace (RTX 40xx, SM 8.9).
      // Good tensor cores but lower memory bandwidth than A100.
      return 5;
    }
    if (sm_major >= 9) {
      // Hopper (H100) and beyond: FP8 tensor cores, 3+ TB/s HBM3.
      // cuBLAS extremely fast — fused advantage shrinks quickly.
      return 3;
    }
    return 4; // Conservative default for unknown future GPUs
  }
};

GpuProfile &GetGpuProfile() {
  static GpuProfile profile;
  static std::once_flag flag;
  std::call_once(flag, [&] {
    cudaDeviceProp prop{};
    int device = 0;
    cudaGetDevice(&device);
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
      profile.sm_major = prop.major;
      profile.sm_minor = prop.minor;
      // memoryClockRate removed in CUDA 13+; use cudaDeviceGetAttribute.
      int mem_clock_khz = 0, mem_bus_width = 0;
      cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate,
                             device);
      cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth,
                             device);
      profile.memory_bandwidth_gb_s = static_cast<float>(mem_clock_khz) * 2.0f *
                                      static_cast<float>(mem_bus_width) / 8.0f /
                                      1e6f;
      profile.sm_count = prop.multiProcessorCount;
      profile.has_dp4a =
          (prop.major > 6) || (prop.major == 6 && prop.minor >= 1);
      profile.initialized = true;

      log::Info(
          "fused_quant_gemm",
          "GPU profile: SM " + std::to_string(prop.major) + "." +
              std::to_string(prop.minor) + ", " +
              std::to_string(profile.sm_count) + " SMs, " +
              std::to_string(static_cast<int>(profile.memory_bandwidth_gb_s)) +
              " GB/s bandwidth, base_threshold=" +
              std::to_string(profile.base_threshold()) +
              ", dp4a=" + (profile.has_dp4a ? "yes" : "no"));
    }
  });
  return profile;
}

// Effective bits per weight for each quant type.
// Used to scale the M threshold: lower bits = fused reads less memory =
// fused advantage extends to higher M.
float BitsPerWeight(GGUF::TensorType qtype) {
  switch (qtype) {
  case GGUF::TensorType::Q4_K:
    return 4.5f; // 4-bit + scales/mins overhead
  case GGUF::TensorType::Q6_K:
    return 6.5625f; // 6-bit + scales overhead
  case GGUF::TensorType::Q8_0:
    return 8.5f; // 8-bit + FP16 scale per 32 elements
  case GGUF::TensorType::Q8_K:
    return 8.5f; // 8-bit + FP32 scale per 256 elements
  case GGUF::TensorType::F16:
    return 16.0f;
  case GGUF::TensorType::F32:
    return 32.0f;
  default:
    return 16.0f; // Conservative: assume FP16-like
  }
}

// Compute the adaptive M threshold for a given quant type.
//
// Formula: threshold = base_threshold * (16.0 / bpw)
//
// Intuition: cuBLAS reads dequantized FP16 weights (16 bits each).
// Fused reads raw quantized weights (bpw bits each).
// The memory bandwidth ratio is 16/bpw.
// For Q4_K (4.5 bpw): fused reads 3.6x less → threshold = base * 3.6
// For Q8_0 (8.5 bpw): fused reads 1.9x less → threshold = base * 1.9
//
// Clamped to [4, kFusedGemmMaxM] — lower bound ensures fused GEMV (M=1)
// always runs, upper bound is the kernel's shared memory capacity.
int ComputeThreshold(int base, float bpw) {
  float raw = static_cast<float>(base) * (16.0f / bpw);
  return std::max(4, std::min(static_cast<int>(kFusedGemmMaxM),
                              static_cast<int>(std::round(raw))));
}

// ============================================================================
// Table-driven kernel dispatch
// ============================================================================

using PackedDispatchFn = bool (*)(const void *data, const int8_t *activation,
                                  const float *row_scales, half *output, int M,
                                  int N, int K, cudaStream_t stream);
using PackedDispatchPairFn = bool (*)(const void *data0, const void *data1,
                                      const int8_t *activation,
                                      const float *row_scales, half *output0,
                                      int N0, half *output1, int N1, int M,
                                      int K, cudaStream_t stream);
using PackedDispatchTripleFn = bool (*)(const void *data0, const void *data1,
                                        const void *data2,
                                        const int8_t *activation,
                                        const float *row_scales, half *output0,
                                        int N0, half *output1, int N1,
                                        half *output2, int N2, int M, int K,
                                        cudaStream_t stream);

template <typename BlockType,
          void (*PackedKernel)(const BlockType *, const int8_t *, const float *,
                               half *, int, int)>
bool DispatchPackedDp4a(const void *data, const int8_t *activation,
                        const float *row_scales, half *output, int M, int N,
                        int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  PackedKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(
      w, activation, row_scales, output, N, K);
  return true;
}

template <typename BlockType, int Outputs,
          void (*PackedKernel)(PackedProjectionGroupParams<BlockType, Outputs>,
                               const int8_t *, const float *, int)>
bool DispatchPackedDp4aGroup(const std::array<const void *, Outputs> &weights,
                             const int8_t *activation, const float *row_scales,
                             const std::array<half *, Outputs> &outputs,
                             const std::array<int, Outputs> &output_cols, int M,
                             int K, cudaStream_t stream) {
  PackedProjectionGroupParams<BlockType, Outputs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < Outputs; ++i) {
    params.weights[i] = static_cast<const BlockType *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  int grid_x = (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  PackedKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, activation,
                                                          row_scales, K);
  return true;
}

template <typename BlockType,
          void (*PackedKernel)(PackedProjectionGroupParams<BlockType, 2>,
                               const int8_t *, const float *, int)>
bool DispatchPackedDp4aPair(const void *data0, const void *data1,
                            const int8_t *activation, const float *row_scales,
                            half *output0, int N0, half *output1, int N1, int M,
                            int K, cudaStream_t stream) {
  return DispatchPackedDp4aGroup<BlockType, 2, PackedKernel>(
      {data0, data1}, activation, row_scales, {output0, output1}, {N0, N1}, M,
      K, stream);
}

template <typename BlockType,
          void (*PackedKernel)(PackedProjectionGroupParams<BlockType, 3>,
                               const int8_t *, const float *, int)>
bool DispatchPackedDp4aTriple(const void *data0, const void *data1,
                              const void *data2, const int8_t *activation,
                              const float *row_scales, half *output0, int N0,
                              half *output1, int N1, half *output2, int N2,
                              int M, int K, cudaStream_t stream) {
  return DispatchPackedDp4aGroup<BlockType, 3, PackedKernel>(
      {data0, data1, data2}, activation, row_scales,
      {output0, output1, output2}, {N0, N1, N2}, M, K, stream);
}

// ============================================================================
// Dispatch entry types (packed activations)
// ============================================================================

struct PackedDispatchEntry {
  PackedDispatchFn fn;
  const char *name;
};

struct PackedDispatchPairEntry {
  PackedDispatchPairFn fn;
  const char *name;
};

struct PackedDispatchTripleEntry {
  PackedDispatchTripleFn fn;
  const char *name;
};

constexpr int kMaxTensorType = 16;
constexpr int kQ4KGroupedHotPathK = 2048;
constexpr int kDownProjHotPathK = 11008;
constexpr int kDownProjHotPathN = 2048;

int ClampThreshold(int threshold) {
  return std::max(4, std::min(static_cast<int>(kFusedGemmMaxM), threshold));
}

bool OperatorSelectionDebugEnabled() {
  return ResolveExecutionPolicy(nullptr).debug_operator_selection;
}

int OperatorSelectionDebugLimit() {
  return ResolveExecutionPolicy(nullptr).debug_operator_selection_limit;
}

bool ConsumeOperatorSelectionBudget() {
  static thread_local int last_limit = -1;
  static thread_local int budget = 0;
  const int limit = OperatorSelectionDebugLimit();
  if (limit != last_limit) {
    last_limit = limit;
    budget = limit;
  }
  if (budget <= 0) {
    return false;
  }
  --budget;
  return true;
}

std::string QuantTypeToString(int quant_type) {
  switch (static_cast<GGUF::TensorType>(quant_type)) {
  case GGUF::TensorType::Q4_K:
    return "q4_k";
  case GGUF::TensorType::Q6_K:
    return "q6_k";
  case GGUF::TensorType::Q8_0:
    return "q8_0";
  case GGUF::TensorType::Q8_K:
    return "q8_k";
  default:
    return std::to_string(quant_type);
  }
}

// ============================================================================
// Vectorized load wrapper: selects baseline or vectorized kernel based on
// policy
// ============================================================================

void LogOperatorSelection(std::string_view op_kind, std::string_view selection,
                          const FusedDispatchGeometry &geometry,
                          std::string_view detail) {
  if (!OperatorSelectionDebugEnabled() || !ConsumeOperatorSelectionBudget()) {
    return;
  }
  inferflux::log::Info(
      "fused_quant_gemm",
      std::string("operator_select[") + std::string(op_kind) + "]: chosen=" +
          std::string(selection) + ", M=" + std::to_string(geometry.M) +
          ", N=" + std::to_string(geometry.N) +
          ", K=" + std::to_string(geometry.K) + ", grouped_outputs=" +
          std::to_string(geometry.grouped_outputs) + ", packed_activation=" +
          std::string(geometry.packed_activation ? "true" : "false") +
          ", includes_rmsnorm=" +
          std::string(geometry.includes_rmsnorm ? "true" : "false") +
          (detail.empty() ? std::string() : ", " + std::string(detail)));
}

bool ShouldUseSpecializedQ8_1GroupedFastPathImpl(
    int quant_type, const FusedDispatchGeometry &geometry) {
  return quant_type == static_cast<int>(GGUF::TensorType::Q4_K) &&
         geometry.grouped_outputs == 2 && geometry.M == 1 &&
         geometry.N >= 8192 && geometry.K == kQ4KGroupedHotPathK;
}

bool ShouldUseSpecializedQ8_1DownProjHotPathImpl(
    int quant_type, const FusedDispatchGeometry &geometry) {
  const auto &policy = ResolveExecutionPolicy(nullptr);
  const bool q4k = quant_type == static_cast<int>(GGUF::TensorType::Q4_K);
  const bool q6k = quant_type == static_cast<int>(GGUF::TensorType::Q6_K);
  if ((!q4k && !q6k) ||
      (q6k && !policy.enable_experimental_q81_downproj_hot_fixed)) {
    return false;
  }
  return geometry.grouped_outputs == 1 && geometry.M == 1 &&
         geometry.N == kDownProjHotPathN && geometry.K == kDownProjHotPathK;
}

bool ShouldUseSpecializedQ8_1DownProjRowPairHotPathImpl(
    int quant_type, const FusedDispatchGeometry &geometry) {
  const auto &policy = ResolveExecutionPolicy(nullptr);
  const bool q4k = quant_type == static_cast<int>(GGUF::TensorType::Q4_K);
  const bool q6k = quant_type == static_cast<int>(GGUF::TensorType::Q6_K);
  if ((!q4k && !q6k) ||
      (q6k && !policy.enable_experimental_q81_downproj_rowpair_hot_fixed)) {
    return false;
  }
  return geometry.grouped_outputs == 1 && geometry.M == 2 &&
         geometry.N == kDownProjHotPathN && geometry.K == kDownProjHotPathK;
}

bool ExperimentalQ8_1GroupedHotQ4KEnabled() {
  return ResolveExecutionPolicy(nullptr)
      .enable_experimental_q81_grouped_hot_q4k;
}

bool ExperimentalQ8_1GroupedRowPairW4Enabled() {
  return ResolveExecutionPolicy(nullptr)
      .enable_experimental_q81_grouped_rowpair_w4;
}

const PackedDispatchEntry &GetPackedDispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a
           ? DispatchPackedDp4a<block_q8_0, fused_dequant_gemv_q8_0_dp4a_packed>
           : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchPackedDp4a<block_q4_k, fused_dequant_gemv_q4k_dp4a_packed>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchPackedDp4a<block_q6_k, fused_dequant_gemv_q6k_dp4a_packed>
            : nullptr,
       "Q6_K"}, // 14
      {dp4a ? DispatchPackedDp4a<block_q8_k, fused_dequant_gemv_q8k_dp4a_packed>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const PackedDispatchPairEntry &
GetPackedDispatchPairEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchPairEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchPackedDp4aPair<
                  block_q8_0, fused_dequant_gemv_q8_0_dp4a_packed_group<2>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a
           ? DispatchPackedDp4aPair<block_q4_k,
                                    fused_dequant_gemv_q4k_dp4a_packed_group<2>>
           : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a
           ? DispatchPackedDp4aPair<block_q6_k,
                                    fused_dequant_gemv_q6k_dp4a_packed_group<2>>
           : nullptr,
       "Q6_K"}, // 14
      {dp4a
           ? DispatchPackedDp4aPair<block_q8_k,
                                    fused_dequant_gemv_q8k_dp4a_packed_group<2>>
           : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchPairEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const PackedDispatchTripleEntry &
GetPackedDispatchTripleEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchTripleEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchPackedDp4aTriple<
                  block_q8_0, fused_dequant_gemv_q8_0_dp4a_packed_group<3>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchPackedDp4aTriple<
                  block_q4_k, fused_dequant_gemv_q4k_dp4a_packed_group<3>>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchPackedDp4aTriple<
                  block_q6_k, fused_dequant_gemv_q6k_dp4a_packed_group<3>>
            : nullptr,
       "Q6_K"}, // 14
      {dp4a ? DispatchPackedDp4aTriple<
                  block_q8_k, fused_dequant_gemv_q8k_dp4a_packed_group<3>>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchTripleEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

template <size_t Count>
bool ValidatePackedProjectionSpecs(
    const std::array<PackedProjectionSpec, Count> &projections,
    const PackedActivationInfo &activation, int M, int K) {
  if (!activation.data || !activation.row_scales || M <= 0) {
    return false;
  }
  const int quant_type = projections[0].weight.quant_type;
  int max_output_cols = 0;
  for (const auto &projection : projections) {
    max_output_cols = std::max(max_output_cols, projection.output_cols);
  }
  const FusedDispatchGeometry geometry{
      M, max_output_cols, K, static_cast<int>(Count), true, false,
  };
  if (quant_type < 0 ||
      !FusedQuantGemm::SupportsPackedActivations(quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(quant_type, geometry)) {
    return false;
  }

  for (const auto &projection : projections) {
    if (!projection.weight.data || projection.output == nullptr ||
        projection.output_cols <= 0 ||
        projection.weight.quant_type != quant_type) {
      return false;
    }
  }
  return true;
}

bool DownProjMmqEnabled() {
  return ResolveExecutionPolicy(nullptr).enable_downproj_mmq;
}

int GetDownProjMmqThresholdOverride() {
  return ResolveExecutionPolicy(nullptr).downproj_mmq_min_batch_override;
}

int GetDownProjMmqThreshold(int quant_type, int M, int N, int K) {
  const int override = GetDownProjMmqThresholdOverride();
  if (override >= 0) {
    return override;
  }
  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  return ClampThreshold(
      FusedQuantGemm::GetGeometryAwareThreshold(quant_type, geometry) + 12);
}

using DownProjMmqDispatchFn = bool (*)(const void *weight, const void *act_q8_1,
                                       half *output, int M, int N, int K,
                                       int tile_cols, cudaStream_t stream);

template <typename BlockT>
__global__ void transform_downproj_mmq_layout(const BlockT *__restrict__ src,
                                              BlockT *__restrict__ dst,
                                              int rows, int num_super_blocks,
                                              int tile_cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile_count = (rows + tile_cols - 1) / tile_cols;
  const int total_blocks = tile_count * num_super_blocks * tile_cols;
  if (idx >= total_blocks) {
    return;
  }

  const int tile_stride = num_super_blocks * tile_cols;
  const int tile = idx / tile_stride;
  const int rem = idx % tile_stride;
  const int super_block = rem / tile_cols;
  const int tile_col = rem % tile_cols;
  const int src_row = tile * tile_cols + tile_col;
  if (src_row < rows) {
    dst[idx] = src[src_row * num_super_blocks + super_block];
  } else {
    dst[idx] = BlockT{};
  }
}

__global__ void
fused_downproj_mmq_q4k_q8_1_tile(const block_q4_k *__restrict__ tiled_weight,
                                 const block_q8_1 *__restrict__ act_q8_1,
                                 half *__restrict__ output, int N, int K,
                                 int tile_cols) {
  extern __shared__ unsigned char shared[];
  auto *shared_act = reinterpret_cast<block_q8_1 *>(shared);

  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;

  const int num_super_blocks = K / QK_K;
  const int num_q8_blocks = K / QK8_1;
  const int linear_tid = out_local * 32 + lane;
  for (int idx = linear_tid; idx < num_q8_blocks;
       idx += blockDim.x * blockDim.y) {
    shared_act[idx] = act_q8_1[row * num_q8_blocks + idx];
  }
  __syncthreads();

  if (out_idx >= N) {
    return;
  }

  const block_q4_k *tile_weights =
      tiled_weight + (blockIdx.x * num_super_blocks * tile_cols);
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q4_k &b = tile_weights[blk * tile_cols + out_local];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const block_q8_1 &a_lo = shared_act[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = shared_act[blk * 8 + pair * 2 + 1];
    int x_lo4 = 0;
    int x_hi4 = 0;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));

    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));
    const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
    acc += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
           d_sc_hi * d8_hi * static_cast<float>(dot_hi);

    if ((lane & 7) == 0) {
      const float s_lo = __half2float(__high2half(a_lo.ds));
      const float s_hi = __half2float(__high2half(a_hi.ds));
      acc -= dm_m_lo * s_lo + dm_m_hi * s_hi;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

bool DispatchDownProjMmqQ4K(const void *weight, const void *act_q8_1,
                            half *output, int M, int N, int K, int tile_cols,
                            cudaStream_t stream) {
  if (!weight || !act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0 ||
      tile_cols <= 0 || K % QK_K != 0 || K % QK8_1 != 0) {
    return false;
  }

  const auto *w = static_cast<const block_q4_k *>(weight);
  const auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const dim3 block(32, tile_cols);
  const dim3 grid((N + tile_cols - 1) / tile_cols, M);
  const size_t shared_bytes =
      static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
  fused_downproj_mmq_q4k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
      w, a, output, N, K, tile_cols);
  return true;
}

__global__ void
fused_downproj_mmq_q6k_q8_1_tile(const block_q6_k *__restrict__ tiled_weight,
                                 const block_q8_1 *__restrict__ act_q8_1,
                                 half *__restrict__ output, int N, int K,
                                 int tile_cols) {
  extern __shared__ unsigned char shared[];
  auto *shared_act = reinterpret_cast<block_q8_1 *>(shared);

  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;

  const int num_super_blocks = K / QK_K;
  const int num_q8_blocks = K / QK8_1;
  const int linear_tid = out_local * 32 + lane;
  for (int idx = linear_tid; idx < num_q8_blocks;
       idx += blockDim.x * blockDim.y) {
    shared_act[idx] = act_q8_1[row * num_q8_blocks + idx];
  }
  __syncthreads();

  if (out_idx >= N) {
    return;
  }

  const block_q6_k *tile_weights =
      tiled_weight + (blockIdx.x * num_super_blocks * tile_cols);
  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = tile_weights[blk * tile_cols + out_local];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const block_q8_1 &a_lo = shared_act[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = shared_act[blk * 8 + g * 4 + sub_base + 2];

    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                    static_cast<float>(dot_lo) +
                static_cast<float>(b.scales[sc_hi]) * d8_hi *
                    static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

bool DispatchDownProjMmqQ6K(const void *weight, const void *act_q8_1,
                            half *output, int M, int N, int K, int tile_cols,
                            cudaStream_t stream) {
  if (!weight || !act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0 ||
      tile_cols <= 0 || K % QK_K != 0 || K % QK8_1 != 0) {
    return false;
  }

  const auto *w = static_cast<const block_q6_k *>(weight);
  const auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const dim3 block(32, tile_cols);
  const dim3 grid((N + tile_cols - 1) / tile_cols, M);
  const size_t shared_bytes =
      static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
  fused_downproj_mmq_q6k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
      w, a, output, N, K, tile_cols);
  return true;
}

struct DownProjMmqDispatchEntry {
  DownProjMmqDispatchFn fn;
  const char *name;
};

const DownProjMmqDispatchEntry &
GetDownProjMmqDispatchEntry(GGUF::TensorType qtype) {
  static const DownProjMmqDispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr},               // 0: F32
      {nullptr, nullptr},               // 1: F16
      {nullptr, nullptr},               // 2: Q4_0
      {nullptr, nullptr},               // 3: Q4_1
      {nullptr, nullptr},               // 4: (unused)
      {nullptr, nullptr},               // 5: (unused)
      {nullptr, nullptr},               // 6: Q5_0
      {nullptr, nullptr},               // 7: Q5_1
      {nullptr, nullptr},               // 8: Q8_0
      {nullptr, nullptr},               // 9: Q8_1
      {nullptr, nullptr},               // 10: Q2_K
      {nullptr, nullptr},               // 11: Q3_K
      {DispatchDownProjMmqQ4K, "Q4_K"}, // 12
      {nullptr, nullptr},               // 13: Q5_K
      {DispatchDownProjMmqQ6K, "Q6_K"}, // 14
      {nullptr, nullptr},               // 15: Q8_K
  };

  static const DownProjMmqDispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType) {
    return empty;
  }
  return table[idx];
}

} // namespace

int FusedQuantGemm::GetAdaptiveThreshold(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  auto &gpu = GetGpuProfile();
  float bpw = BitsPerWeight(qtype);
  int base = gpu.initialized ? gpu.base_threshold() : 8;
  // dp4a kernels use 4x less shared memory (int8 vs FP32 activations) and
  // hardware integer dot products, so the fused advantage extends to higher M.
  // This matches llama.cpp's MMQ_DP4A_MAX_BATCH_SIZE=64 crossover point.
  if (gpu.has_dp4a &&
      (qtype == GGUF::TensorType::Q4_K || qtype == GGUF::TensorType::Q6_K ||
       qtype == GGUF::TensorType::Q8_0 || qtype == GGUF::TensorType::Q8_K)) {
    base = std::max(base * 2, 14);
  }
  return ComputeThreshold(base, bpw);
}

int FusedQuantGemm::GetGeometryAwareThreshold(
    int quant_type, const FusedDispatchGeometry &geometry) {
  int threshold = GetAdaptiveThreshold(quant_type);
  if (geometry.M <= 0) {
    return 0;
  }

  const bool packed = geometry.packed_activation;
  const int grouped_outputs = std::max(1, geometry.grouped_outputs);
  const int n = std::max(0, geometry.N);
  const int k = std::max(0, geometry.K);

  if (packed) {
    // Packed activations remove repeated half->int8 preparation from each GEMV
    // launch. Grouped sibling projections amortize that cost further.
    threshold += 8;
    threshold += 4 * (grouped_outputs - 1);
    if (k >= 2048) {
      threshold += 4;
    }
    if (k >= 4096) {
      threshold += 4;
    }
    // Very large single-output surfaces (notably lm_head) still favor the
    // compatibility path earlier than sibling projection groups do.
    if (grouped_outputs == 1 && n >= 32768) {
      threshold -= 8;
    } else if (grouped_outputs == 1 && n >= 8192) {
      threshold -= 2;
    }
  } else {
    // Legacy shared-memory activation paths lose relative advantage as output
    // surface grows.
    if (n >= 8192) {
      threshold -= 2;
    }
    if (geometry.includes_rmsnorm) {
      threshold -= 2;
    }
  }

  return ClampThreshold(threshold);
}

bool FusedQuantGemm::ShouldUseFusedPath(int quant_type, int M) {
  const FusedDispatchGeometry geometry{M, 0, 0, 1, false, false};
  return ShouldUseFusedPath(quant_type, geometry);
}

bool FusedQuantGemm::ShouldUseFusedPath(int quant_type,
                                        const FusedDispatchGeometry &geometry) {
  if (geometry.M <= 0) {
    return false;
  }
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  if (geometry.grouped_outputs > 1) {
    if (geometry.grouped_outputs == 2) {
      return GetPackedDispatchPairEntry(qtype).fn != nullptr &&
             geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
    }
    if (geometry.grouped_outputs == 3) {
      return GetPackedDispatchTripleEntry(qtype).fn != nullptr &&
             geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
    }
    return false;
  }
  if (geometry.packed_activation) {
    if (GetPackedDispatchEntry(qtype).fn == nullptr) {
      return false;
    }
  } else if (!SupportsQ8_1Activations(quant_type) &&
             GetPackedDispatchEntry(qtype).fn == nullptr) {
    // A quant type is fused-eligible if it has Q8_1 (MMVQ/MMQ) or packed
    // dispatch support.
    return false;
  }
  return geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
}

bool FusedQuantGemm::SupportsPackedActivations(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetPackedDispatchEntry(qtype).fn != nullptr;
}

bool FusedQuantGemm::SupportsDownProjMmq(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetDownProjMmqDispatchEntry(qtype).fn != nullptr;
}

int FusedQuantGemm::GetDownProjMmqThreshold(int quant_type, int m, int n,
                                            int k) {
  return ::inferflux::GetDownProjMmqThreshold(quant_type, m, n, k);
}

bool FusedQuantGemm::IsDownProjMmqEnabled(const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return DownProjMmqEnabled();
}

bool FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
    int quant_type, const FusedDispatchGeometry &geometry,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return ExperimentalQ8_1GroupedHotQ4KEnabled() &&
         ShouldUseSpecializedQ8_1GroupedFastPathImpl(quant_type, geometry);
}

bool FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
    int quant_type, const FusedDispatchGeometry &geometry,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return ExperimentalQ8_1GroupedRowPairW4Enabled() &&
         quant_type == static_cast<int>(GGUF::TensorType::Q4_K) &&
         geometry.grouped_outputs == 2 && geometry.M == 2 &&
         geometry.N >= 8192 && geometry.K == kQ4KGroupedHotPathK;
}

bool FusedQuantGemm::ShouldUseSpecializedQ8_1DownProjHotPath(
    int quant_type, const FusedDispatchGeometry &geometry,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return ShouldUseSpecializedQ8_1DownProjHotPathImpl(quant_type, geometry);
}

bool FusedQuantGemm::ShouldUseSpecializedQ8_1DownProjRowPairHotPath(
    int quant_type, const FusedDispatchGeometry &geometry,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return ShouldUseSpecializedQ8_1DownProjRowPairHotPathImpl(quant_type,
                                                            geometry);
}

FusedQuantGemm::FfnProjOperator FusedQuantGemm::SelectFfnProjOperator(
    int quant_type0, int quant_type1, const FusedDispatchGeometry &geometry,
    bool allow_q81, bool allow_packed, const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  const NativeExecutionPolicy &resolved_policy = ResolveExecutionPolicy(policy);
  FfnProjOperator selected = FusedQuantGemm::FfnProjOperator::kFallback;
  if (geometry.M <= 0 || geometry.N <= 0 || geometry.K <= 0 ||
      geometry.grouped_outputs != 2) {
    LogOperatorSelection("ffn_proj", FfnProjOperatorName(selected), geometry,
                         "invalid_geometry");
    return selected;
  }

  const FusedDispatchGeometry single_geometry{geometry.M,
                                              geometry.N,
                                              geometry.K,
                                              1,
                                              geometry.packed_activation,
                                              geometry.includes_rmsnorm};
  const bool q81_ready = allow_q81 && SupportsQ8_1Activations(quant_type0) &&
                         SupportsQ8_1Activations(quant_type1) &&
                         ShouldUseFusedPath(quant_type0, single_geometry) &&
                         ShouldUseFusedPath(quant_type1, single_geometry);

  const bool packed_ready = allow_packed &&
                            SupportsPackedActivations(quant_type0) &&
                            SupportsPackedActivations(quant_type1) &&
                            ShouldUseFusedPath(quant_type0, single_geometry) &&
                            ShouldUseFusedPath(quant_type1, single_geometry);
  const auto profile = BuildInferfluxCudaFfnDispatchProfile(
      InferfluxCudaDispatchPhase::kDecode, quant_type0, quant_type1, geometry,
      q81_ready, packed_ready);
  const auto decision =
      SelectInferfluxCudaFfnDispatchDecision(profile, resolved_policy);
  selected = decision.op;
  LogOperatorSelection(
      "ffn_proj", FfnProjOperatorName(selected), geometry,
      DescribeInferfluxCudaFfnDispatchDecision(
          profile, decision.reason ? decision.reason : "fallback"));
  return selected;
}

FusedQuantGemm::DownProjOperator FusedQuantGemm::SelectDownProjOperator(
    int quant_type, const FusedDispatchGeometry &geometry, bool allow_q81,
    bool allow_packed, bool allow_mmq, const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  const NativeExecutionPolicy &resolved_policy = ResolveExecutionPolicy(policy);
  DownProjOperator selected = FusedQuantGemm::DownProjOperator::kFallback;
  if (geometry.M <= 0 || geometry.N <= 0 || geometry.K <= 0) {
    LogOperatorSelection("down_proj", DownProjOperatorName(selected), geometry,
                         "invalid_geometry");
    return selected;
  }

  const bool q81_ready =
      allow_q81 && SupportsQ8_1Activations(quant_type) &&
      ShouldUseFusedPath(quant_type,
                         FusedDispatchGeometry{geometry.M, geometry.N,
                                               geometry.K, 1, true, false});

  const bool packed_ready =
      allow_packed && SupportsPackedActivations(quant_type) &&
      ShouldUseFusedPath(quant_type,
                         FusedDispatchGeometry{geometry.M, geometry.N,
                                               geometry.K, 1, true, false});

  const bool mmq_ready =
      allow_mmq && SupportsDownProjMmq(quant_type) &&
      geometry.M >= GetDownProjMmqThreshold(quant_type, geometry.M, geometry.N,
                                            geometry.K);
  const auto profile = BuildInferfluxCudaDownProjDispatchProfile(
      InferfluxCudaDispatchPhase::kUnknown, quant_type, geometry, q81_ready,
      packed_ready, mmq_ready);
  const auto decision =
      SelectInferfluxCudaDownProjDispatchDecision(profile, resolved_policy);
  selected = decision.op;
  LogOperatorSelection(
      "down_proj", DownProjOperatorName(selected), geometry,
      DescribeInferfluxCudaDownProjDispatchDecision(
          profile, decision.reason ? decision.reason : "fallback"));
  return selected;
}

bool FusedQuantGemm::GemvPacked(const QuantizedWeightInfo &weight,
                                const PackedActivationInfo &activation,
                                half *output, int M, int N, int K,
                                cudaStream_t stream,
                                const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0 || !activation.data ||
      !activation.row_scales) {
    return false;
  }
  if (ResolveExecutionPolicy(policy).disable_fused_gemv) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetPackedDispatchEntry(qtype);
  if (!entry.fn) {
    return false;
  }
  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  if (!ShouldUseFusedPath(weight.quant_type, geometry)) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed-activation fused kernel for ") +
                  entry.name + " (M=" + std::to_string(M) + ", N=" +
                  std::to_string(N) + ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(weight.data, activation.data, activation.row_scales, output,
                  M, N, K, stream);
}

bool FusedQuantGemm::GemvPackedPair(
    const std::array<PackedProjectionSpec, 2> &projections,
    const PackedActivationInfo &activation, int M, int K, cudaStream_t stream) {
  if (!ValidatePackedProjectionSpecs(projections, activation, M, K)) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(projections[0].weight.quant_type);
  const auto &entry = GetPackedDispatchPairEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed grouped pair kernel for ") +
                  entry.name + " (M=" + std::to_string(M) +
                  ", N=" + std::to_string(projections[0].output_cols) + "/" +
                  std::to_string(projections[1].output_cols) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  activation.data, activation.row_scales, projections[0].output,
                  projections[0].output_cols, projections[1].output,
                  projections[1].output_cols, M, K, stream);
}

bool FusedQuantGemm::GemvPackedTriple(
    const std::array<PackedProjectionSpec, 3> &projections,
    const PackedActivationInfo &activation, int M, int K, cudaStream_t stream) {
  if (!ValidatePackedProjectionSpecs(projections, activation, M, K)) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(projections[0].weight.quant_type);
  const auto &entry = GetPackedDispatchTripleEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed grouped triple kernel for ") +
                  entry.name + " (M=" + std::to_string(M) +
                  ", N=" + std::to_string(projections[0].output_cols) + "/" +
                  std::to_string(projections[1].output_cols) + "/" +
                  std::to_string(projections[2].output_cols) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  projections[2].weight.data, activation.data,
                  activation.row_scales, projections[0].output,
                  projections[0].output_cols, projections[1].output,
                  projections[1].output_cols, projections[2].output,
                  projections[2].output_cols, M, K, stream);
}

// ============================================================================
// Q8_1 activation quantization and dispatch
// ============================================================================

namespace {

using Q8_1DispatchFn = bool (*)(const void *data, const void *act_q8_1,
                                half *output, int M, int N, int K,
                                cudaStream_t stream);
using Q8_1DispatchPairFn = bool (*)(const void *data0, const void *data1,
                                    const void *act_q8_1, half *output0, int N0,
                                    half *output1, int N1, int M, int K,
                                    cudaStream_t stream);
using Q8_1DispatchTripleFn = bool (*)(const void *data0, const void *data1,
                                      const void *data2, const void *act_q8_1,
                                      half *output0, int N0, half *output1,
                                      int N1, half *output2, int N2, int M,
                                      int K, cudaStream_t stream);

template <
    typename BlockType, int Outputs,
    void (*Q8_1GroupKernel)(PackedProjectionGroupParams<BlockType, Outputs>,
                            const block_q8_1 *, int)>
bool DispatchQ8_1GemvGroup(const std::array<const void *, Outputs> &weights,
                           const void *act_q8_1,
                           const std::array<half *, Outputs> &outputs,
                           const std::array<int, Outputs> &output_cols, int M,
                           int K, cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, Outputs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < Outputs; ++i) {
    params.weights[i] = static_cast<const BlockType *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  int grid_x = (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  Q8_1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(PackedProjectionGroupParams<BlockType, 2>,
                                  const block_q8_1 *, int)>
bool DispatchQ8_1GemvPair(const void *data0, const void *data1,
                          const void *act_q8_1, half *output0, int N0,
                          half *output1, int N1, int M, int K,
                          cudaStream_t stream) {
  return DispatchQ8_1GemvGroup<BlockType, 2, Q8_1GroupKernel>(
      {data0, data1}, act_q8_1, {output0, output1}, {N0, N1}, M, K, stream);
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(PackedProjectionGroupParams<BlockType, 3>,
                                  const block_q8_1 *, int)>
bool DispatchQ8_1GemvTriple(const void *data0, const void *data1,
                            const void *data2, const void *act_q8_1,
                            half *output0, int N0, half *output1, int N1,
                            half *output2, int N2, int M, int K,
                            cudaStream_t stream) {
  return DispatchQ8_1GemvGroup<BlockType, 3, Q8_1GroupKernel>(
      {data0, data1, data2}, act_q8_1, {output0, output1, output2},
      {N0, N1, N2}, M, K, stream);
}

// ============================================================================
// MMVQ (weight-read-first) dispatch helpers
// ============================================================================

// MMVQ single-output dispatch for Q4_K
bool DispatchQ8_1MmvqQ4K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  return DispatchMmvq<block_q4_k, inferflux_mmvq_q4k<1>, inferflux_mmvq_q4k<2>,
                      inferflux_mmvq_q4k<4>, inferflux_mmvq_q4k<8>>(
      data, act_q8_1, output, M, N, K, stream);
}

// MMVQ single-output dispatch for Q6_K
bool DispatchQ8_1MmvqQ6K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  return DispatchMmvq<block_q6_k, inferflux_mmvq_q6k<1>, inferflux_mmvq_q6k<2>,
                      inferflux_mmvq_q6k<4>, inferflux_mmvq_q6k<8>>(
      data, act_q8_1, output, M, N, K, stream);
}

// MMVQ single-output dispatch for Q8_0
bool DispatchQ8_1MmvqQ8_0(const void *data, const void *act_q8_1, half *output,
                          int M, int N, int K, cudaStream_t stream) {
  return DispatchMmvq<block_q8_0, inferflux_mmvq_q8_0<1>,
                      inferflux_mmvq_q8_0<2>, inferflux_mmvq_q8_0<4>,
                      inferflux_mmvq_q8_0<8>>(data, act_q8_1, output, M, N, K,
                                              stream);
}

// MMVQ single-output dispatch for Q8_K
bool DispatchQ8_1MmvqQ8K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  return DispatchMmvq<block_q8_k, inferflux_mmvq_q8k<1>, inferflux_mmvq_q8k<2>,
                      inferflux_mmvq_q8k<4>, inferflux_mmvq_q8k<8>>(
      data, act_q8_1, output, M, N, K, stream);
}

// ============================================================================
// MMQ dispatch functions (batch 9-64)
// ============================================================================

bool DispatchMmqQ4K(const void *data, const void *act_q8_1, half *output, int M,
                    int N, int K, cudaStream_t stream) {
  return DispatchMmq<block_q4_k, inferflux_mmq_q4k<16>, inferflux_mmq_q4k<32>>(
      data, act_q8_1, output, M, N, K, 8, stream);
}

bool DispatchMmqQ6K(const void *data, const void *act_q8_1, half *output, int M,
                    int N, int K, cudaStream_t stream) {
  return DispatchMmq<block_q6_k, inferflux_mmq_q6k<16>, inferflux_mmq_q6k<32>>(
      data, act_q8_1, output, M, N, K, 8, stream);
}

bool DispatchMmqQ8_0(const void *data, const void *act_q8_1, half *output,
                     int M, int N, int K, cudaStream_t stream) {
  return DispatchMmq<block_q8_0, inferflux_mmq_q8_0<16>,
                     inferflux_mmq_q8_0<32>>(data, act_q8_1, output, M, N, K, 4,
                                             stream);
}

bool DispatchMmqQ8K(const void *data, const void *act_q8_1, half *output, int M,
                    int N, int K, cudaStream_t stream) {
  return DispatchMmq<block_q8_k, inferflux_mmq_q8k<16>, inferflux_mmq_q8k<32>>(
      data, act_q8_1, output, M, N, K, 8, stream);
}

// MMVQ pair dispatch for Q4_K
bool DispatchQ8_1MmvqPairQ4K(const void *data0, const void *data1,
                             const void *act_q8_1, half *output0, int N0,
                             half *output1, int N1, int M, int K,
                             cudaStream_t stream) {
  return DispatchMmvqPair<block_q4_k, inferflux_mmvq_q4k_group<1, 2>,
                          inferflux_mmvq_q4k_group<2, 2>,
                          inferflux_mmvq_q4k_group<4, 2>,
                          inferflux_mmvq_q4k_group<8, 2>>(
      data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
}

// MMVQ pair dispatch for Q6_K
bool DispatchQ8_1MmvqPairQ6K(const void *data0, const void *data1,
                             const void *act_q8_1, half *output0, int N0,
                             half *output1, int N1, int M, int K,
                             cudaStream_t stream) {
  return DispatchMmvqPair<block_q6_k, inferflux_mmvq_q6k_group<1, 2>,
                          inferflux_mmvq_q6k_group<2, 2>,
                          inferflux_mmvq_q6k_group<4, 2>,
                          inferflux_mmvq_q6k_group<8, 2>>(
      data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
}

// MMVQ triple dispatch for Q4_K
bool DispatchQ8_1MmvqTripleQ4K(const void *data0, const void *data1,
                               const void *data2, const void *act_q8_1,
                               half *output0, int N0, half *output1, int N1,
                               half *output2, int N2, int M, int K,
                               cudaStream_t stream) {
  return DispatchMmvqTriple<block_q4_k, inferflux_mmvq_q4k_group<1, 3>,
                            inferflux_mmvq_q4k_group<2, 3>,
                            inferflux_mmvq_q4k_group<4, 3>,
                            inferflux_mmvq_q4k_group<8, 3>>(
      data0, data1, data2, act_q8_1, output0, N0, output1, N1, output2, N2, M,
      K, stream);
}

// MMVQ triple dispatch for Q6_K
bool DispatchQ8_1MmvqTripleQ6K(const void *data0, const void *data1,
                               const void *data2, const void *act_q8_1,
                               half *output0, int N0, half *output1, int N1,
                               half *output2, int N2, int M, int K,
                               cudaStream_t stream) {
  return DispatchMmvqTriple<block_q6_k, inferflux_mmvq_q6k_group<1, 3>,
                            inferflux_mmvq_q6k_group<2, 3>,
                            inferflux_mmvq_q6k_group<4, 3>,
                            inferflux_mmvq_q6k_group<8, 3>>(
      data0, data1, data2, act_q8_1, output0, N0, output1, N1, output2, N2, M,
      K, stream);
}

// ============================================================================
// MMVQ/MMQ dispatch wrappers (sole Q8_1 dispatch path)
//
// M <= 8:  MMVQ (weight-read-first batch-amortized)
// M <= 64: MMQ  (tiled quantized GEMM)
// M > 64:  return false → cuBLAS fallback
// ============================================================================

bool DispatchQ8_1GemvQ4K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqQ4K(data, act_q8_1, output, M, N, K, stream);
  if (M <= 64)
    return DispatchMmqQ4K(data, act_q8_1, output, M, N, K, stream);
  return false;
}

bool DispatchQ8_1GemvQ6K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqQ6K(data, act_q8_1, output, M, N, K, stream);
  if (M <= 64)
    return DispatchMmqQ6K(data, act_q8_1, output, M, N, K, stream);
  return false;
}

bool DispatchQ8_1GemvQ8_0(const void *data, const void *act_q8_1, half *output,
                          int M, int N, int K, cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqQ8_0(data, act_q8_1, output, M, N, K, stream);
  if (M <= 64)
    return DispatchMmqQ8_0(data, act_q8_1, output, M, N, K, stream);
  return false;
}

bool DispatchQ8_1GemvQ8K(const void *data, const void *act_q8_1, half *output,
                         int M, int N, int K, cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqQ8K(data, act_q8_1, output, M, N, K, stream);
  if (M <= 64)
    return DispatchMmqQ8K(data, act_q8_1, output, M, N, K, stream);
  return false;
}

// Grouped pair dispatch: MMVQ for M<=8, V1 group fallback for M>8
// (grouped MMQ not yet implemented)
bool DispatchQ8_1GemvPairQ4K(const void *data0, const void *data1,
                             const void *act_q8_1, half *output0, int N0,
                             half *output1, int N1, int M, int K,
                             cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqPairQ4K(data0, data1, act_q8_1, output0, N0, output1,
                                   N1, M, K, stream);
  return DispatchQ8_1GemvPair<block_q4_k, fused_dequant_gemv_q4k_q8_1_group<2>>(
      data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
}

bool DispatchQ8_1GemvPairQ6K(const void *data0, const void *data1,
                             const void *act_q8_1, half *output0, int N0,
                             half *output1, int N1, int M, int K,
                             cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqPairQ6K(data0, data1, act_q8_1, output0, N0, output1,
                                   N1, M, K, stream);
  return DispatchQ8_1GemvPair<block_q6_k, fused_dequant_gemv_q6k_q8_1_group<2>>(
      data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
}

// Grouped triple dispatch: MMVQ for M<=8, V1 group fallback for M>8
bool DispatchQ8_1GemvTripleQ4K(const void *data0, const void *data1,
                               const void *data2, const void *act_q8_1,
                               half *output0, int N0, half *output1, int N1,
                               half *output2, int N2, int M, int K,
                               cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqTripleQ4K(data0, data1, data2, act_q8_1, output0, N0,
                                     output1, N1, output2, N2, M, K, stream);
  return DispatchQ8_1GemvTriple<block_q4_k,
                                fused_dequant_gemv_q4k_q8_1_group<3>>(
      data0, data1, data2, act_q8_1, output0, N0, output1, N1, output2, N2, M,
      K, stream);
}

bool DispatchQ8_1GemvTripleQ6K(const void *data0, const void *data1,
                               const void *data2, const void *act_q8_1,
                               half *output0, int N0, half *output1, int N1,
                               half *output2, int N2, int M, int K,
                               cudaStream_t stream) {
  if (M <= 8)
    return DispatchQ8_1MmvqTripleQ6K(data0, data1, data2, act_q8_1, output0, N0,
                                     output1, N1, output2, N2, M, K, stream);
  return DispatchQ8_1GemvTriple<block_q6_k,
                                fused_dequant_gemv_q6k_q8_1_group<3>>(
      data0, data1, data2, act_q8_1, output0, N0, output1, N1, output2, N2, M,
      K, stream);
}

struct Q8_1DispatchEntry {
  Q8_1DispatchFn fn;
  const char *name;
};

struct Q8_1DispatchPairEntry {
  Q8_1DispatchPairFn fn;
  const char *name;
};

struct Q8_1DispatchTripleEntry {
  Q8_1DispatchTripleFn fn;
  const char *name;
};

const Q8_1DispatchEntry &GetQ8_1DispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr},                              // 0: F32
      {nullptr, nullptr},                              // 1: F16
      {nullptr, nullptr},                              // 2: Q4_0
      {nullptr, nullptr},                              // 3: Q4_1
      {nullptr, nullptr},                              // 4: (unused)
      {nullptr, nullptr},                              // 5: (unused)
      {nullptr, nullptr},                              // 6: Q5_0
      {nullptr, nullptr},                              // 7: Q5_1
      {dp4a ? DispatchQ8_1GemvQ8_0 : nullptr, "Q8_0"}, // 8
      {nullptr, nullptr},                              // 9: Q8_1
      {nullptr, nullptr},                              // 10: Q2_K
      {nullptr, nullptr},                              // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvQ4K : nullptr, "Q4_K"},  // 12
      {nullptr, nullptr},                              // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvQ6K : nullptr, "Q6_K"},  // 14
      {dp4a ? DispatchQ8_1GemvQ8K : nullptr, "Q8_K"},  // 15
  };

  static const Q8_1DispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const Q8_1DispatchPairEntry &GetQ8_1DispatchPairEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchPairEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchQ8_1GemvPair<block_q8_0,
                                   fused_dequant_gemv_q8_0_q8_1_group<2>>
            : nullptr,
       "Q8_0"},                                           // 8
      {nullptr, nullptr},                                 // 9: Q8_1
      {nullptr, nullptr},                                 // 10: Q2_K
      {nullptr, nullptr},                                 // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvPairQ4K : nullptr, "Q4_K"}, // 12
      {nullptr, nullptr},                                 // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvPairQ6K : nullptr, "Q6_K"}, // 14
      {dp4a ? DispatchQ8_1GemvPair<block_q8_k,
                                   fused_dequant_gemv_q8k_q8_1_group<2>>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const Q8_1DispatchPairEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const Q8_1DispatchTripleEntry &
GetQ8_1DispatchTripleEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchTripleEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchQ8_1GemvTriple<block_q8_0,
                                     fused_dequant_gemv_q8_0_q8_1_group<3>>
            : nullptr,
       "Q8_0"},                                             // 8
      {nullptr, nullptr},                                   // 9: Q8_1
      {nullptr, nullptr},                                   // 10: Q2_K
      {nullptr, nullptr},                                   // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvTripleQ4K : nullptr, "Q4_K"}, // 12
      {nullptr, nullptr},                                   // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvTripleQ6K : nullptr, "Q6_K"}, // 14
      {dp4a ? DispatchQ8_1GemvTriple<block_q8_k,
                                     fused_dequant_gemv_q8k_q8_1_group<3>>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const Q8_1DispatchTripleEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

} // namespace

void FusedQuantGemm::QuantizeRowQ8_1(const half *input, void *output_q8_1,
                                     int M, int K, cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  quantize_row_q8_1_kernel<<<M, kGemvThreadsPerBlock, 0, stream>>>(input,
                                                                   output, K);
}

void FusedQuantGemm::SiluMulQuantizeQ8_1(const half *gate, const half *up,
                                         void *output_q8_1, int M, int K,
                                         cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  silu_mul_quantize_q8_1_kernel<<<M, kGemvThreadsPerBlock, 0, stream>>>(
      gate, up, output, K);
}

void FusedQuantGemm::FusedRmsNormQuantizeQ8_1(const half *residual,
                                              const half *norm_weight,
                                              void *output_q8_1, int M, int K,
                                              float rms_norm_eps,
                                              cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  size_t smem = static_cast<size_t>(K) * sizeof(float) +
                kGemvWarpsPerBlock * sizeof(float);
  fused_rmsnorm_quantize_q8_1_kernel<<<M, kGemvThreadsPerBlock, smem, stream>>>(
      residual, norm_weight, output, K, rms_norm_eps);
}

bool FusedQuantGemm::SupportsQ8_1Activations(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetQ8_1DispatchEntry(qtype).fn != nullptr;
}

bool FusedQuantGemm::BuildDownProjMmqLayout(const QuantizedWeightInfo &weight,
                                            int rows, int cols,
                                            MmqWeightInfo *layout,
                                            cudaStream_t stream) {
  if (!layout || !weight.data || rows <= 0 || cols <= 0 || cols % QK_K != 0) {
    return false;
  }
  if (!SupportsDownProjMmq(weight.quant_type)) {
    return false;
  }

  const int tile_cols = FusedQuantGemm::kDownProjMmqTileCols;
  const int num_super_blocks = cols / QK_K;
  const int tile_count = (rows + tile_cols - 1) / tile_cols;
  const size_t transformed_blocks =
      static_cast<size_t>(tile_count) * num_super_blocks * tile_cols;

  constexpr int kThreads = 256;
  const int blocks =
      static_cast<int>((transformed_blocks + kThreads - 1) / kThreads);

  void *transformed = nullptr;
  const auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  switch (qtype) {
  case GGUF::TensorType::Q4_K: {
    block_q4_k *typed = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&typed),
                   transformed_blocks * sizeof(block_q4_k)) != cudaSuccess) {
      return false;
    }
    transform_downproj_mmq_layout<<<blocks, kThreads, 0, stream>>>(
        static_cast<const block_q4_k *>(weight.data), typed, rows,
        num_super_blocks, tile_cols);
    transformed = typed;
    break;
  }
  case GGUF::TensorType::Q6_K: {
    block_q6_k *typed = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&typed),
                   transformed_blocks * sizeof(block_q6_k)) != cudaSuccess) {
      return false;
    }
    transform_downproj_mmq_layout<<<blocks, kThreads, 0, stream>>>(
        static_cast<const block_q6_k *>(weight.data), typed, rows,
        num_super_blocks, tile_cols);
    transformed = typed;
    break;
  }
  default:
    return false;
  }

  if (cudaPeekAtLastError() != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaFree(transformed);
    return false;
  }

  *layout = {transformed, weight.quant_type, rows, cols, tile_cols};
  return true;
}

void FusedQuantGemm::DestroyDownProjMmqLayout(const MmqWeightInfo &layout) {
  if (layout.data) {
    cudaFree(const_cast<void *>(layout.data));
  }
}

bool FusedQuantGemm::DownProjMmq(const MmqWeightInfo &weight,
                                 const void *act_q8_1, half *output, int M,
                                 int N, int K, cudaStream_t stream,
                                 const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!DownProjMmqEnabled() || !weight.data || !act_q8_1 || !output || M <= 0 ||
      N <= 0 || K <= 0 || N != weight.rows || K != weight.cols ||
      weight.tile_cols <= 0 || !SupportsDownProjMmq(weight.quant_type)) {
    return false;
  }

  const int threshold = GetDownProjMmqThreshold(weight.quant_type, M, N, K);
  if (M > threshold || N < weight.tile_cols || K % QK8_1 != 0) {
    return false;
  }

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  const size_t shared_bytes =
      static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return false;
  }
  const size_t shared_limit = static_cast<size_t>(
      std::max(prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin));
  if (shared_bytes > shared_limit) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetDownProjMmqDispatchEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using MMQ-style down-proj kernel for ") +
                  entry.name + " (M=" + std::to_string(M) +
                  ", N=" + std::to_string(N) + ", K=" + std::to_string(K) +
                  ", tile_cols=" + std::to_string(weight.tile_cols) + ")");
  }

  return entry.fn(weight.data, act_q8_1, output, M, N, K, weight.tile_cols,
                  stream);
}

void FusedQuantGemm::QuantizeRowQ8_1Mmq(
    const half *input, runtime::cuda::native::BlockQ8_1Mmq *output, int M, int K,
    cudaStream_t stream) {
  using namespace runtime::cuda::native;
  dim3 grid((K / 128 + 3) / 4, M);
  QuantizeRowQ8_1MmqKernel<<<grid, 128, 0, stream>>>(input, output, K, M);
}

void FusedQuantGemm::SiluMulQuantizeQ8_1Mmq(
    const half *gate, const half *up, runtime::cuda::native::BlockQ8_1Mmq *output,
    int M, int K, cudaStream_t stream) {
  using namespace runtime::cuda::native;
  dim3 grid((K / 128 + 3) / 4, M);
  SiluMulQuantizeQ8_1MmqKernel<<<grid, 128, 0, stream>>>(gate, up, output, K, M);
}

bool FusedQuantGemm::DownProjMmqMma(
    const QuantizedWeightInfo &weight,
    const runtime::cuda::native::BlockQ8_1Mmq *act_mmq, half *output, int M,
    int N, int K, float *partials, cudaStream_t stream,
    const NativeExecutionPolicy *policy) {
  using namespace runtime::cuda::native;
  const auto &p = ResolveExecutionPolicy(policy);
  if (!p.enable_mmq_mma || !weight.data || !act_mmq || !output || !partials ||
      M <= 0 || N <= 0 || K <= 0 ||
      static_cast<size_t>(N) * K !=
          static_cast<size_t>(weight.num_elements)) {
    return false;
  }
  if (static_cast<GGUF::TensorType>(weight.quant_type) !=
          GGUF::TensorType::Q6_K ||
      K % QK_K != 0 || M > p.mmq_mma_max_batch) {
    return false;
  }

  static bool smem_configured = false;
  static size_t configured_smem = 0;
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  if (!smem_configured || configured_smem != smem) {
    if (cudaFuncSetAttribute(InferfluxMmqQ6KMma<16>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)) != cudaSuccess) {
      return false;
    }
    smem_configured = true;
    configured_smem = smem;
  }

  // K-split only when the N-tile grid alone under-occupies the device:
  // one block per super-block slice keeps the deterministic reduce cheap.
  cudaDeviceProp prop{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return false;
  }
  const int n_tiles = (N + kMmqY - 1) / kMmqY;
  int splits = 1;
  if (n_tiles < prop.multiProcessorCount) {
    splits = std::min((prop.multiProcessorCount + n_tiles - 1) / n_tiles,
                      static_cast<int>(kMmqMmaMaxSplits));
  }

  dim3 grid(n_tiles, (M + 15) / 16, splits);
  InferfluxMmqQ6KMma<16><<<grid, dim3(32, kMmqMmaWarps, 1), smem, stream>>>(
      static_cast<const char *>(weight.data), act_mmq, output, N, K, M,
      splits > 1 ? partials : nullptr, splits);
  if (splits > 1) {
    const size_t mn = static_cast<size_t>(M) * N;
    const int rthreads = 256;
    const size_t rblocks = (mn + rthreads - 1) / rthreads;
    ReduceMmqKSplit<<<rblocks, rthreads, 0, stream>>>(partials, output, splits,
                                                      mn);
  }
  if (cudaPeekAtLastError() != cudaSuccess) {
    return false;
  }

  static bool logged = false;
  if (!logged) {
    logged = true;
    log::Info("fused_quant_gemm",
              std::string("Using MMA tensor-core Q6_K down-proj (M=") +
                  std::to_string(M) + ", N=" + std::to_string(N) + ", K=" +
                  std::to_string(K) + ", ksplit=" + std::to_string(splits) +
                  ")");
  }
  return true;
}

bool FusedQuantGemm::GemvQ8_1(const QuantizedWeightInfo &weight,
                              const void *act_q8_1, half *output, int M, int N,
                              int K, cudaStream_t stream,
                              const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0 || !act_q8_1)
    return false;
  if (ResolveExecutionPolicy(policy).disable_fused_gemv)
    return false;

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetQ8_1DispatchEntry(qtype);
  if (!entry.fn)
    return false;

  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  if (!ShouldUseFusedPath(weight.quant_type, geometry))
    return false;

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    const char *kernel_name = (M <= 8)    ? "MMVQ weight-read-first"
                              : (M <= 64) ? "MMQ tiled GEMM"
                                          : "Q8_1 activation";
    log::Info("fused_quant_gemm",
              std::string("Using ") + kernel_name + " kernel for " +
                  entry.name + " (M=" + std::to_string(M) + ", N=" +
                  std::to_string(N) + ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(weight.data, act_q8_1, output, M, N, K, stream);
}

static bool
GemvQ8_1PairMmq3(const std::array<PackedProjectionSpec, 2> &projections,
                 const void *act_q8_1, int M, int K, cudaStream_t stream);

bool FusedQuantGemm::GemvQ8_1Pair(
    const std::array<PackedProjectionSpec, 2> &projections,
    const void *act_q8_1, int M, int K, cudaStream_t stream,
    const NativeExecutionPolicy *policy, FfnProjOperator selected_op) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!act_q8_1 || M <= 0 || K <= 0)
    return false;

  const int quant_type = projections[0].weight.quant_type;
  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type)
      return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  const auto &entry = GetQ8_1DispatchPairEntry(qtype);
  if (!entry.fn)
    return false;

  auto idx = static_cast<uint32_t>(qtype);
  const FusedDispatchGeometry geometry{
      M,    std::max(projections[0].output_cols, projections[1].output_cols),
      K,    2,
      true, false};
  if (selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3) {
    static bool mmq3_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !mmq3_logged[idx] && entry.name) {
      mmq3_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using MMQ3 grouped Q8_1 pair kernel for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
    return GemvQ8_1PairMmq3(projections, act_q8_1, M, K, stream);
  }

  if (selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4) {
    static bool rowquad_m4_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !rowquad_m4_logged[idx] && entry.name) {
      rowquad_m4_logged[idx] = true;
      log::Info(
          "fused_quant_gemm",
          std::string("Using exact-M4 grouped Q8_1 row-quad kernel for ") +
              entry.name + " (M=" + std::to_string(M) +
              ", N=" + std::to_string(projections[0].output_cols) + "/" +
              std::to_string(projections[1].output_cols) +
              ", K=" + std::to_string(K) + ")");
    }
    return GemvQ8_1PairRowQuadCandidate(projections, act_q8_1, M, K, stream);
  }

  if (ShouldUseSpecializedQ8_1GroupedFastPath(quant_type, geometry)) {
    static bool specialized_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !specialized_logged[idx] && entry.name) {
      specialized_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using fixed-block grouped Q8_1 pair kernel for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
  } else if (ShouldUseSpecializedQ8_1GroupedRowPairW4Path(quant_type,
                                                          geometry)) {
    static bool w4_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !w4_logged[idx] && entry.name) {
      w4_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using 4-warp grouped Q8_1 row-pair kernel for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
  } else if (M >= 4 && (qtype == GGUF::TensorType::Q4_K ||
                        qtype == GGUF::TensorType::Q6_K)) {
    static bool rowquad_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !rowquad_logged[idx] && entry.name) {
      rowquad_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using batched row-quad grouped Q8_1 pair kernel "
                            "for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
  } else if (M > 1 && (qtype == GGUF::TensorType::Q4_K ||
                       qtype == GGUF::TensorType::Q6_K)) {
    static bool rowpair_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !rowpair_logged[idx] && entry.name) {
      rowpair_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using batched row-pair grouped Q8_1 pair kernel "
                            "for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  act_q8_1, projections[0].output, projections[0].output_cols,
                  projections[1].output, projections[1].output_cols, M, K,
                  stream);
}

bool FusedQuantGemm::FusedGateUpSiluGemvQ8_1(
    const QuantizedWeightInfo &gate_raw, const QuantizedWeightInfo &up_raw,
    const void *act_q8_1, half *output, int M, int N, int K,
    cudaStream_t stream) {
  if (!act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0) {
    return false;
  }
  if (!gate_raw.data || !up_raw.data) {
    return false;
  }
  // Both projections must be Q4_K
  if (gate_raw.quant_type != static_cast<int>(GGUF::TensorType::Q4_K) ||
      up_raw.quant_type != static_cast<int>(GGUF::TensorType::Q4_K)) {
    return false;
  }

  using namespace runtime::cuda::native;
  static bool logged = false;
  if (!logged) {
    logged = true;
    log::Info(
        "fused_quant_gemm",
        "Using fused gate+up+SiLU Q4_K MMVQ kernel (M=" + std::to_string(M) +
            ", N=" + std::to_string(N) + ", K=" + std::to_string(K) + ")");
  }

  return DispatchMmvqFusedGateUpSilu<block_q4_k,
                                     inferflux_mmvq_q4k_fused_gate_up_silu<1>,
                                     inferflux_mmvq_q4k_fused_gate_up_silu<2>,
                                     inferflux_mmvq_q4k_fused_gate_up_silu<4>,
                                     inferflux_mmvq_q4k_fused_gate_up_silu<8>>(
      gate_raw.data, up_raw.data, act_q8_1, output, N, M, K, stream);
}

bool FusedQuantGemm::GemvQ8_1PairRowQuadCandidate(
    const std::array<PackedProjectionSpec, 2> &projections,
    const void *act_q8_1, int M, int K, cudaStream_t stream) {
  if (!act_q8_1 || M < 3 || M > 4 || K <= 0) {
    return false;
  }

  const int quant_type = projections[0].weight.quant_type;
  if (quant_type != static_cast<int>(GGUF::TensorType::Q4_K)) {
    return false;
  }

  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type) {
      return false;
    }
  }

  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<block_q4_k, 2> params{};
  params.weights[0] =
      static_cast<const block_q4_k *>(projections[0].weight.data);
  params.weights[1] =
      static_cast<const block_q4_k *>(projections[1].weight.data);
  params.outputs[0] = projections[0].output;
  params.outputs[1] = projections[1].output;
  params.output_cols[0] = projections[0].output_cols;
  params.output_cols[1] = projections[1].output_cols;
  const int max_output_cols =
      std::max(projections[0].output_cols, projections[1].output_cols);
  const int grid_x =
      (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, (M + 3) / 4);
  fused_dequant_gemv_q4k_q8_1_group_rowquad<2>
      <<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K, M);
  return cudaGetLastError() == cudaSuccess;
}

static bool
GemvQ8_1PairMmq3(const std::array<PackedProjectionSpec, 2> &projections,
                 const void *act_q8_1, int M, int K, cudaStream_t stream) {
  if (!act_q8_1 || M < 3 || K <= 0) {
    return false;
  }

  const int quant_type = projections[0].weight.quant_type;
  if (quant_type != static_cast<int>(GGUF::TensorType::Q4_K)) {
    return false;
  }

  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type) {
      return false;
    }
  }

  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  MmqGroupParams<block_q4_k, 2> params{};
  params.weights[0] =
      static_cast<const block_q4_k *>(projections[0].weight.data);
  params.weights[1] =
      static_cast<const block_q4_k *>(projections[1].weight.data);
  params.outputs[0] = projections[0].output;
  params.outputs[1] = projections[1].output;
  params.output_cols[0] = projections[0].output_cols;
  params.output_cols[1] = projections[1].output_cols;
  const int max_output_cols =
      std::max(projections[0].output_cols, projections[1].output_cols);

  const int grid_x = (max_output_cols + kMmq3Warps - 1) / kMmq3Warps;
  if (M >= 9) {
    constexpr int kRows = 8;
    const int grid_y = (M + kRows - 1) / kRows;
    dim3 grid(grid_x, grid_y);
    dim3 block(kMmq3Warps * 32);
    const size_t smem = kRows * 8 * sizeof(block_q8_1);
    fused_grouped_ffn_mmq3_q4k_q8_1<kRows, 2>
        <<<grid, block, smem, stream>>>(params, a, K, M);
  } else {
    constexpr int kRows = 4;
    const int grid_y = (M + kRows - 1) / kRows;
    dim3 grid(grid_x, grid_y);
    dim3 block(kMmq3Warps * 32);
    const size_t smem = kRows * 8 * sizeof(block_q8_1);
    fused_grouped_ffn_mmq3_q4k_q8_1<kRows, 2>
        <<<grid, block, smem, stream>>>(params, a, K, M);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool FusedQuantGemm::GemvQ8_1Triple(
    const std::array<PackedProjectionSpec, 3> &projections,
    const void *act_q8_1, int M, int K, cudaStream_t stream,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!act_q8_1 || M <= 0 || K <= 0)
    return false;

  const int quant_type = projections[0].weight.quant_type;
  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type)
      return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  const auto &entry = GetQ8_1DispatchTripleEntry(qtype);
  if (!entry.fn)
    return false;

  auto idx = static_cast<uint32_t>(qtype);
  {
    static bool triple_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !triple_logged[idx] && entry.name) {
      triple_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using MMVQ grouped Q8_1 triple for ") +
                    entry.name + " (M=" + std::to_string(M) +
                    ", N=" + std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) + "/" +
                    std::to_string(projections[2].output_cols) +
                    ", K=" + std::to_string(K) + ")");
    }
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  projections[2].weight.data, act_q8_1, projections[0].output,
                  projections[0].output_cols, projections[1].output,
                  projections[1].output_cols, projections[2].output,
                  projections[2].output_cols, M, K, stream);
}

} // namespace inferflux
