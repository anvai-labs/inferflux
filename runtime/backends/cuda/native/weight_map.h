#pragma once

#include "runtime/backends/cuda/inferflux_cuda_executor.h"

// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>))
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
struct __nv_bfloat16;
typedef __nv_bfloat16 __nv_bfloat16; // opaque in CPU builds
#endif
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

/**
 * Info about a raw quantized weight tensor on GPU.
 * Used by fused dequant-GEMV kernels to bypass full dequantization.
 */
struct QuantizedWeightInfo {
  const void *data{nullptr}; // Raw quantized GPU pointer
  int quant_type{-1};        // GGUF::TensorType enum value (-1 = unknown)
  int64_t num_elements{0};   // Logical element count (rows * cols)
};

/**
 * Tile-major quantized layout for MMQ-style kernels.
 *
 * Unlike QuantizedWeightInfo, this points at a transformed layout optimized
 * for multi-output tiled execution rather than the original GGUF row-major
 * tensor blocks.
 */
struct MmqWeightInfo {
  const void *data{nullptr};
  int quant_type{-1};
  int rows{0};      // Logical output rows / columns in the destination matrix
  int cols{0};      // Logical K dimension
  int tile_cols{0}; // Number of output rows packed per layout tile
};

/**
 * WeightMapTyped<T>: typed GPU pointer accessors for safetensors model weights.
 *
 * Wraps SafetensorsLoader to provide const T* accessors organized by
 * model architecture (Llama/Qwen tensor naming conventions).
 */
template <typename T> class WeightMapTyped {
public:
  WeightMapTyped() = default;
  virtual ~WeightMapTyped() = default;

  bool Build(const SafetensorsLoader &loader,
             const SafetensorsLoader::ModelConfig &config);

  // --- Per-layer accessors ---
  virtual const T *LayerQProj(int layer) const;
  virtual const T *LayerKProj(int layer) const;
  virtual const T *LayerVProj(int layer) const;
  virtual const T *LayerOProj(int layer) const;
  virtual const T *LayerInputNorm(int layer) const;
  virtual const T *LayerPostAttnNorm(int layer) const;
  virtual const T *LayerGateProj(int layer) const;
  virtual const T *LayerUpProj(int layer) const;
  virtual const T *LayerDownProj(int layer) const;

  // --- Per-layer bias accessors (nullptr if model has no biases) ---
  virtual const T *LayerQProjBias(int layer) const;
  virtual const T *LayerKProjBias(int layer) const;
  virtual const T *LayerVProjBias(int layer) const;

  // --- Global accessors ---
  virtual const T *EmbedTokens() const { return embed_tokens_; }
  virtual const T *FinalNorm() const { return final_norm_; }
  virtual const T *LmHead() const { return lm_head_; }

  virtual int NumLayers() const { return num_layers_; }

  // --- Raw quantized weight accessors (for fused dequant-GEMV) ---
  // Default implementations return empty info (safetensors models).
  // Overridden by QuantizedWeightMapAdapter for GGUF models.
  virtual QuantizedWeightInfo LayerQProjRaw(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LayerKProjRaw(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LayerVProjRaw(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LayerOProjRaw(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LayerGateProjRaw(int /*layer*/) const {
    return {};
  }
  virtual QuantizedWeightInfo LayerUpProjRaw(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LayerDownProjRaw(int /*layer*/) const {
    return {};
  }
  virtual MmqWeightInfo LayerDownProjMmq(int /*layer*/) const { return {}; }
  virtual QuantizedWeightInfo LmHeadRaw() const { return {}; }
  virtual bool HasQuantizedWeights() const { return false; }
  // Strategy-driven policy switch used by native CUDA forward kernels to
  // force compatibility GEMM path when fused dequant kernels are disallowed.
  virtual bool AllowFusedQuantizedMatmul() const { return true; }

private:
  struct LayerWeights {
    const T *q_proj{nullptr};
    const T *k_proj{nullptr};
    const T *v_proj{nullptr};
    const T *o_proj{nullptr};
    const T *input_norm{nullptr};
    const T *post_attn_norm{nullptr};
    const T *gate_proj{nullptr};
    const T *up_proj{nullptr};
    const T *down_proj{nullptr};
    // Optional biases (Qwen2 has q/k/v biases)
    const T *q_proj_bias{nullptr};
    const T *k_proj_bias{nullptr};
    const T *v_proj_bias{nullptr};
  };

  const T *Resolve(const SafetensorsLoader &loader,
                   const std::string &name) const;

  std::vector<LayerWeights> layers_;
  const T *embed_tokens_{nullptr};
  const T *final_norm_{nullptr};
  const T *lm_head_{nullptr};
  int num_layers_{0};
};

// Backward-compatible alias
using WeightMap = WeightMapTyped<half>;

} // namespace inferflux
