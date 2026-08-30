#pragma once

#include <array>

#include "runtime/backends/cuda/native/dispatch_catalog.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "runtime/backends/cuda/native/weight_map.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cstdint>
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
#endif

namespace inferflux {

// Defined in kernels/mmq_mma.cuh (CUDA-only header): forward-declared here
// so CPU-only TUs can compile this header without device intrinsics.
namespace runtime {
namespace cuda {
namespace native {
struct BlockQ8_1Mmq;
struct BlockQ8_1MmqDs;
} // namespace native
} // namespace cuda
} // namespace runtime

struct PackedActivationInfo {
  const int8_t *data{nullptr};
  const float *row_scales{nullptr};
};

struct PackedProjectionSpec {
  QuantizedWeightInfo weight;
  half *output{nullptr};
  int output_cols{0};
};

struct FusedDispatchGeometry {
  int M{0};
  int N{0};
  int K{0};
  int grouped_outputs{1};
  bool packed_activation{false};
  bool includes_rmsnorm{false};
};

/**
 * Adaptive dispatch for fused dequant-GEMV/GEMM kernels.
 *
 * For small M (decode and short prefill), launches fused kernels that read
 * raw quantized blocks and dequantize in registers — avoiding full FP16
 * weight materialization.
 *
 * For large M, returns false so the caller falls back to cuBLAS with tensor
 * cores. The crossover threshold is adaptive: computed from GPU compute
 * capability (tensor core generation) and quantization bits per weight.
 */
class FusedQuantGemm {
public:
  static constexpr int kDownProjMmqTileCols = 8;

  // Operator enums live at namespace scope in dispatch_catalog.h (the
  // single source of truth shared with the CPU-compiled decision layer);
  // these aliases keep every historical FusedQuantGemm::-qualified use
  // compiling unchanged.
  using FfnProjOperator = ::inferflux::FfnProjOperator;
  using DownProjOperator = ::inferflux::DownProjOperator;

  static const char *FfnProjOperatorName(FfnProjOperator op) {
    return FfnSelectionLabel(op);
  }
  static const char *FfnProjOperatorMetricName(FfnProjOperator op,
                                               int quant_type, int m,
                                               int k = 0) {
    (void)k;
    return FfnMetricLabel(op, quant_type, m);
  }
  static const char *DownProjOperatorName(DownProjOperator op) {
    return DownSelectionLabel(op);
  }
  static const char *DownProjOperatorMetricName(DownProjOperator op,
                                                int quant_type, int m,
                                                int k = 0) {
    (void)k;
    return DownMetricLabel(op, quant_type, m);
  }

  /**
   * Attempt a fused dequant-GEMV using pre-quantized int8 activations packed
   * once per row and shared across sibling projections.
   *
   * This is the memory-first/native-throughput path for dp4a-capable quant
   * types where repeated activation preparation dominates decode cost.
   */
  static bool GemvPacked(const QuantizedWeightInfo &weight,
                         const PackedActivationInfo &activation, half *output,
                         int M, int N, int K, cudaStream_t stream,
                         const NativeExecutionPolicy *policy = nullptr);

  /**
   * Attempt a single packed-activation launch for two sibling projections.
   *
   * All projections must share the same quant type. Returns false when the
   * grouped path is unsupported so the caller can fall back to per-projection
   * launches without losing the shared activation pack.
   */
  static bool
  GemvPackedPair(const std::array<PackedProjectionSpec, 2> &projections,
                 const PackedActivationInfo &activation, int M, int K,
                 cudaStream_t stream);

  /**
   * Attempt a single packed-activation launch for three sibling projections.
   *
   * All projections must share the same quant type. Returns false when the
   * grouped path is unsupported so the caller can fall back to per-projection
   * launches without losing the shared activation pack.
   */
  static bool
  GemvPackedTriple(const std::array<PackedProjectionSpec, 3> &projections,
                   const PackedActivationInfo &activation, int M, int K,
                   cudaStream_t stream);

  /**
   * Quantize FP16 activation rows to Q8_1 block format in global memory.
   * Each 32-element block gets its own scale and pre-computed sum.
   *
   * @param input  FP16 activations [M, K] (half, device)
   * @param output_q8_1  Output Q8_1 blocks [M, K/32] (block_q8_1, device)
   * @param M  Number of rows
   * @param K  Number of columns (must be multiple of 32)
   * @param stream  CUDA stream
   */
  static void QuantizeRowQ8_1(const half *input, void *output_q8_1, int M,
                              int K, cudaStream_t stream);

  /**
   * Fuse SwiGLU activation and Q8_1 quantization for down-projection input.
   *
   * Avoids materializing the post-SwiGLU activation in FP16 before the
   * immediate quantized GEMV that consumes it.
   */
  static void SiluMulQuantizeQ8_1(const half *gate, const half *up,
                                  void *output_q8_1, int M, int K,
                                  cudaStream_t stream);

  /**
   * Fused RmsNorm + Q8_1 quantization: normalizes and quantizes in one pass.
   * Produces Q8_1 blocks for reuse across sibling projections.
   *
   * @param residual  Raw residual [M, K] (half, device)
   * @param norm_weight  RmsNorm weight vector [K] (half, device)
   * @param output_q8_1  Output Q8_1 blocks [M, K/32] (block_q8_1, device)
   * @param M  Number of rows
   * @param K  Number of columns (must be multiple of 32)
   * @param rms_norm_eps  RmsNorm epsilon
   * @param stream  CUDA stream
   */
  static void FusedRmsNormQuantizeQ8_1(const half *residual,
                                       const half *norm_weight,
                                       void *output_q8_1, int M, int K,
                                       float rms_norm_eps, cudaStream_t stream);

  /**
   * FP32 residual variant: reads float* residual, normalizes, quantizes to
   * Q8_1. No FP16 intermediate — reads FP32 directly into shared memory.
   */
  static void FusedRmsNormQuantizeQ8_1(const float *residual,
                                       const half *norm_weight,
                                       void *output_q8_1, int M, int K,
                                       float rms_norm_eps, cudaStream_t stream);

  /**
   * GEMV using pre-quantized Q8_1 activations.
   * Per-32-element block scales provide better precision than per-row.
   * No shared memory needed for activations (reads from L2 cache).
   *
   * @param weight  Raw quantized weight info (GPU pointer + quant type)
   * @param act_q8_1  Pre-quantized Q8_1 activations (block_q8_1, device)
   * @param output  Output matrix [M, N] (half, device)
   * @param M  Number of rows
   * @param N  Number of output columns
   * @param K  Inner dimension (must be multiple of 32)
   * @param stream  CUDA stream
   * @return true if Q8_1 kernel was launched, false if unsupported
   */
  static bool GemvQ8_1(const QuantizedWeightInfo &weight, const void *act_q8_1,
                       half *output, int M, int N, int K, cudaStream_t stream,
                       const NativeExecutionPolicy *policy = nullptr);

  /**
   * GEMV using pre-quantized Q8_1 activations with accumulate mode.
   * output[i] += gemv_result[i] instead of output[i] = gemv_result[i].
   * Used to accumulate O-proj and down-proj directly into the residual stream,
   * eliminating a separate ResidualAdd kernel launch.
   *
   * @param weight  Raw quantized weight info (GPU pointer + quant type)
   * @param act_q8_1  Pre-quantized Q8_1 activations (block_q8_1, device)
   * @param output  Output matrix [M, N] (half, device) — read-modify-write
   * @param M  Number of rows
   * @param N  Number of output columns
   * @param K  Inner dimension (must be multiple of 32)
   * @param stream  CUDA stream
   * @return true if accumulate kernel was launched, false if unsupported
   */
  static bool GemvQ8_1Accum(const QuantizedWeightInfo &weight,
                            const void *act_q8_1, half *output, int M, int N,
                            int K, cudaStream_t stream,
                            const NativeExecutionPolicy *policy = nullptr);

  /**
   * GEMV using pre-quantized Q8_1 activations with FP32 accumulate mode.
   * output(float)[i] += gemv_result[i] instead of output(half)[i] += ...
   * Used to accumulate O-proj and down-proj directly into the FP32 residual
   * stream, eliminating FP16 truncation at layer boundaries.
   * Currently supports MMVQ path only (M<=8, covers all decode).
   */
  static bool GemvQ8_1AccumF32(const QuantizedWeightInfo &weight,
                               const void *act_q8_1, float *output, int M,
                               int N, int K, cudaStream_t stream,
                               const NativeExecutionPolicy *policy = nullptr);

  /**
   * MMQ-style tiled down-projection path for transformed GGUF weights.
   *
   * This is an additive migration path used only for native GGUF decode on
   * projections that have an MMQ-ready transformed layout.
   */
  static bool DownProjMmq(const MmqWeightInfo &weight, const void *act_q8_1,
                          half *output, int M, int N, int K,
                          cudaStream_t stream,
                          const NativeExecutionPolicy *policy = nullptr);

  static bool
  IsDownProjMmqEnabled(const NativeExecutionPolicy *policy = nullptr);

  /// True when the CUDA architecture provides the signed-int8 mma.sync
  /// primitive used by the Q6_K MMA path (Turing / SM 7.5 or newer).
  static bool SupportsMmqMmaArchitecture(int sm_major, int sm_minor);

  /// Runtime form of SupportsMmqMmaArchitecture for the active CUDA device.
  static bool CurrentDeviceSupportsMmqMma();

  // -----------------------------------------------------------------------
  // MMA tensor-core path (S7): Q6_K down-projection via mma.sync int8
  // tensor cores with deterministic K-split. Consumes the RAW row-major
  // weight layout (no MmqWeightInfo transform) and the group-major D4
  // activation buffer produced by QuantizeRowQ8_1Mmq /
  // SiluMulQuantizeQ8_1Mmq. Gated on NativeExecutionPolicy::enable_mmq_mma
  // (INFERFLUX_CUDA_MMQ_MMA, default off).
  // -----------------------------------------------------------------------

  /**
   * Quantize FP16 activation rows to the D4 (BlockQ8_1Mmq) layout:
   * group-major [K/128][M] blocks of 128 values with four float scales.
   */
  static void QuantizeRowQ8_1Mmq(const half *input,
                                 runtime::cuda::native::BlockQ8_1Mmq *output,
                                 int M, int K, cudaStream_t stream);

  /** Fused SwiGLU + D4 quantization for down-projection input. */
  static void
  SiluMulQuantizeQ8_1Mmq(const half *gate, const half *up,
                         runtime::cuda::native::BlockQ8_1Mmq *output, int M,
                         int K, cudaStream_t stream);

  /**
   * MMA down-projection: out[M, N] = act[M, K] x W[N, K]^T for Q6_K W in
   * raw row-major super-block layout. K-split fills the SM array when the
   * N-tile grid alone under-occupies the device; partials must have room
   * for kMmqMmaMaxSplits * M * N floats.
   *
   * @return true if the kernel launched, false when unsupported (caller
   * falls back to the dp4a MMQ / Q8_1 tiers).
   */
  /**
   * Generic Q4_K single-projection MMA (S8): out[M, N] = act[M, K] x
   * W[N, K]^T via the int8 tensor-core kernel with DS activations.
   * Gated on NativeExecutionPolicy::enable_mmq_mma for Q4_K, M in
   * [mmq_mma_min_batch_exclusive..max]; M=1 and non-Q4_K fall through.
   */
  static bool GemvMmqMma(const QuantizedWeightInfo &weight, const half *input,
                         half *output,
                         runtime::cuda::native::BlockQ8_1MmqDs *ds_act,
                         float *partials, int M, int N, int K,
                         cudaStream_t stream,
                         const NativeExecutionPolicy *policy = nullptr);

  static bool DownProjMmqMma(const QuantizedWeightInfo &weight,
                             const runtime::cuda::native::BlockQ8_1Mmq *act_mmq,
                             half *output, int M, int N, int K, float *partials,
                             cudaStream_t stream,
                             const NativeExecutionPolicy *policy = nullptr);

  /**
   * Build a tile-major MMQ layout for a quantized tensor.
   *
   * The current incremental slice supports native GGUF down-proj weights for
   * the quant types with explicit MMQ dispatch entries.
   */
  static bool BuildDownProjMmqLayout(const QuantizedWeightInfo &weight,
                                     int rows, int cols, MmqWeightInfo *layout,
                                     cudaStream_t stream);

  static void DestroyDownProjMmqLayout(const MmqWeightInfo &layout);

  static bool SupportsDownProjMmq(int quant_type);
  static int GetDownProjMmqThreshold(int quant_type, int m, int n, int k);

  /**
   * Hybrid FFN gate+up grouped operator selector.
   *
   * Starts with the live native decode hot path: small-batch Q8_1 grouped
   * kernels for Q4_K, then falls back to the generic Q8_1 shared-quantization
   * path, then packed activations, then the compatibility path.
   */
  static FfnProjOperator
  SelectFfnProjOperator(int quant_type0, int quant_type1,
                        const FusedDispatchGeometry &geometry, bool allow_q81,
                        bool allow_packed,
                        const NativeExecutionPolicy *policy = nullptr);

  /**
   * Hybrid down-proj operator selector.
   *
   * Mirrors the vendored llama.cpp split conceptually: keep the MMVQ-like
   * Q8_1 path for smaller decode envelopes and promote MMQ only when the
   * caller allows it and geometry crosses the tiled-kernel threshold.
   */
  static DownProjOperator
  SelectDownProjOperator(int quant_type, const FusedDispatchGeometry &geometry,
                         bool allow_q81, bool allow_packed, bool allow_mmq,
                         const NativeExecutionPolicy *policy = nullptr);

  /**
   * Grouped Q8_1 GEMV for two sibling projections (single kernel launch).
   */
  static bool
  GemvQ8_1Pair(const std::array<PackedProjectionSpec, 2> &projections,
               const void *act_q8_1, int M, int K, cudaStream_t stream,
               const NativeExecutionPolicy *policy = nullptr,
               FfnProjOperator selected_op = FfnProjOperator::kFallback);

  /**
   * Explicit grouped row-quad FFN benchmark candidate.
   *
   * This bypasses runtime selection and exists only so benchmarks/tests can
   * measure the Q4_K grouped row-quad candidate on exact live geometries
   * without exposing it as a serving policy.
   */
  static bool GemvQ8_1PairRowQuadCandidate(
      const std::array<PackedProjectionSpec, 2> &projections,
      const void *act_q8_1, int M, int K, cudaStream_t stream);

  /**
   * Fused gate+up+SiLU MMVQ: computes SiLU(gate_proj(x)) * up_proj(x) in one
   * kernel pass. Both projections must be the same quant type (Q4_K).
   * Output is FP16; caller must quantize to Q8_1 before down_proj GEMV.
   */
  static bool FusedGateUpSiluGemvQ8_1(const QuantizedWeightInfo &gate_raw,
                                      const QuantizedWeightInfo &up_raw,
                                      const void *act_q8_1, half *output, int M,
                                      int N, int K, cudaStream_t stream);
  /**
   * Fused gate+up+SiLU MMVQ with Q8_1 quantization epilogue.
   * Produces both FP16 output and Q8_1 quantized output in a single kernel,
   * eliminating the separate QuantizeQ8_1 step before down-proj GEMV.
   * When act_q8_1_out is non-null, writes Q8_1 blocks alongside FP16.
   */
  static bool FusedGateUpSiluGemvQ8_1WithEpilogue(
      const QuantizedWeightInfo &gate_raw, const QuantizedWeightInfo &up_raw,
      const void *act_q8_1, half *output, void *act_q8_1_out, int M, int N,
      int K, cudaStream_t stream);

  /**
   * Grouped Q8_1 GEMV for three sibling projections (single kernel launch).
   */
  static bool
  GemvQ8_1Triple(const std::array<PackedProjectionSpec, 3> &projections,
                 const void *act_q8_1, int M, int K, cudaStream_t stream,
                 const NativeExecutionPolicy *policy = nullptr);

  /**
   * GEMV with bias epilogue: output = dot(W, x) + bias.
   * Fuses the bias addition into the MMVQ writeback, eliminating a
   * separate BiasAdd kernel launch. Bias may be nullptr (no-op).
   * Currently supports Q4_K only.
   */
  static bool GemvQ8_1WithBias(const QuantizedWeightInfo &weight,
                               const void *act_q8_1, half *output,
                               const half *bias, int M, int N, int K,
                               cudaStream_t stream,
                               const NativeExecutionPolicy *policy = nullptr);

  /**
   * Returns true when a quant type supports Q8_1 activation GEMV.
   */
  static bool SupportsQ8_1Activations(int quant_type);

  /**
   * Small-batch grouped Q8_1 fast-path selector.
   *
   * This is an additive hot-shape optimization for native decode kernels.
   * The current staged rollout promotes only the measured grouped Q4_K FFN
   * single-row decode envelope and falls back to the generic grouped kernels
   * for every other geometry.
   */
  static bool ShouldUseSpecializedQ8_1GroupedFastPath(
      int quant_type, const FusedDispatchGeometry &geometry,
      const NativeExecutionPolicy *policy = nullptr);

  /**
   * Experimental 4-warp grouped row-pair selector.
   *
   * This targets the measured M=2 FFN grouped decode envelope for Q4_K using
   * a smaller 4-warp CTA, following llama.cpp's MMVQ small-batch launch shape
   * more closely than the generic one-warp-per-output row-pair path.
   */
  static bool ShouldUseSpecializedQ8_1GroupedRowPairW4Path(
      int quant_type, const FusedDispatchGeometry &geometry,
      const NativeExecutionPolicy *policy = nullptr);

  static bool ShouldUseSpecializedQ8_1DownProjHotPath(
      int quant_type, const FusedDispatchGeometry &geometry,
      const NativeExecutionPolicy *policy = nullptr);

  static bool ShouldUseSpecializedQ8_1DownProjRowPairHotPath(
      int quant_type, const FusedDispatchGeometry &geometry,
      const NativeExecutionPolicy *policy = nullptr);

  /**
   * Get the adaptive M threshold for a given quant type.
   * Above this M, cuBLAS with tensor cores is expected to be faster.
   * Queries GPU properties once on first call (thread-safe).
   */
  static int GetAdaptiveThreshold(int quant_type);

  /**
   * Geometry-aware fused dispatch ceiling.
   *
   * The base threshold is quant/GPU dependent. Packed activations, grouped
   * sibling projections, and very large output surfaces shift the crossover.
   */
  static int GetGeometryAwareThreshold(int quant_type,
                                       const FusedDispatchGeometry &geometry);

  /**
   * Deterministic dispatch policy helper used by runtime paths and tests.
   *
   * Returns true when a quant type is fused-kernel supported and batch size M
   * is within the adaptive threshold for fused execution.
   */
  static bool ShouldUseFusedPath(int quant_type, int M);

  static bool ShouldUseFusedPath(int quant_type,
                                 const FusedDispatchGeometry &geometry);

  /**
   * Returns true when a quant type can consume pre-quantized int8 activation
   * rows directly from global memory.
   */
  static bool SupportsPackedActivations(int quant_type);
};

} // namespace inferflux
