#pragma once

#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/weight_map.h"

#ifdef INFERFLUX_HAS_CUDA
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

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace inferflux {

// Bring in the types from the nested namespace
using runtime::cuda::native::IModelLoader;
using runtime::cuda::native::IWeightAccessor;
using runtime::cuda::native::ModelInfo;

/**
 * @brief QuantizedWeightMap for GGUF models
 *
 * Provides similar interface to WeightMap but works with IModelLoader
 * and IWeightAccessor. Handles lazy dequantization and caching.
 *
 * Key features:
 * - Works with both quantized and non-quantized models
 * - Lazy dequantization (only when accessed)
 * - GPU caching of dequantized weights
 * - Compatible with existing forward pass code
 */
class QuantizedWeightMap {
#ifdef INFERFLUX_TESTING
  friend class WeightMapTestAccess;
#endif
public:
  QuantizedWeightMap() = default;
  ~QuantizedWeightMap();

  // Non-copyable, non-movable (manages GPU resources)
  QuantizedWeightMap(const QuantizedWeightMap &) = delete;
  QuantizedWeightMap &operator=(const QuantizedWeightMap &) = delete;

  /**
   * @brief Build weight map from IModelLoader
   * @param loader Model loader (GGUF or safetensors adapter)
   * @param config Model configuration
   * @param stream CUDA stream for operations
   * @return true on success
   */
  bool Build(IModelLoader *loader, const ModelInfo &config,
             cudaStream_t stream);

  /** Switch the CUDA stream used for dequantization operations. */
  void SetStream(cudaStream_t stream) { stream_ = stream; }

  // --- Per-layer accessors (returns dequantized FP16 weights) ---

  /**
   * @brief Get Q projection weights for a layer
   * @return Pointer to GPU memory with FP16 weights, or nullptr if not found
   */
  const half *LayerQProj(int layer) const;

  /**
   * @brief Get K projection weights for a layer
   */
  const half *LayerKProj(int layer) const;

  /**
   * @brief Get V projection weights for a layer
   */
  const half *LayerVProj(int layer) const;

  /**
   * @brief Get O projection weights for a layer
   */
  const half *LayerOProj(int layer) const;

  /**
   * @brief Get input layer norm weights for a layer
   */
  const half *LayerInputNorm(int layer) const;

  /**
   * @brief Get post-attention layer norm weights for a layer
   */
  const half *LayerPostAttnNorm(int layer) const;

  /**
   * @brief Get gate projection weights for a layer (FFN)
   */
  const half *LayerGateProj(int layer) const;

  /**
   * @brief Get up projection weights for a layer (FFN)
   */
  const half *LayerUpProj(int layer) const;

  /**
   * @brief Get down projection weights for a layer (FFN)
   */
  const half *LayerDownProj(int layer) const;

  // --- Bias accessors (may return nullptr for models without biases) ---

  const half *LayerQProjBias(int layer) const;
  const half *LayerKProjBias(int layer) const;
  const half *LayerVProjBias(int layer) const;

  // --- Global accessors ---

  const half *EmbedTokens() const;
  const half *FinalNorm() const;
  const half *LmHead() const;

  // --- Metadata ---

  int NumLayers() const { return num_layers_; }
  bool IsQuantized() const { return is_quantized_; }
  std::string GetQuantizationType() const { return quantization_type_; }
  void SetAllowFusedQuantizedMatmul(bool allow) {
    allow_fused_quantized_matmul_ = allow;
  }
  bool AllowFusedQuantizedMatmul() const {
    return allow_fused_quantized_matmul_;
  }

  // When enabled, projection weights use permanent per-tensor caching
  // instead of the shared scratch buffer.  Trades GPU memory for prefill
  // performance by eliminating dequantization kernel launches per layer.
  void SetBatchDequantCacheEnabled(bool enable) {
    batch_dequant_cache_enabled_ = enable;
  }
  bool BatchDequantCacheEnabled() const { return batch_dequant_cache_enabled_; }

  // --- Raw quantized weight accessors (for fused dequant-GEMV) ---

  QuantizedWeightInfo GetRawLayerQProj(int layer) const;
  QuantizedWeightInfo GetRawLayerKProj(int layer) const;
  QuantizedWeightInfo GetRawLayerVProj(int layer) const;
  QuantizedWeightInfo GetRawLayerOProj(int layer) const;
  QuantizedWeightInfo GetRawLayerGateProj(int layer) const;
  QuantizedWeightInfo GetRawLayerUpProj(int layer) const;
  QuantizedWeightInfo GetRawLayerDownProj(int layer) const;
  MmqWeightInfo GetMmqLayerDownProj(int layer) const;
  QuantizedWeightInfo GetRawLmHead() const;

  /**
   * @brief Check if a weight tensor exists
   */
  bool HasTensor(const std::string &name) const;

  /**
   * @brief Get raw weight accessor for a tensor (advanced use)
   */
  std::shared_ptr<IWeightAccessor> GetWeightAccessor(const std::string &name);

  /**
   * @brief Free all cached dequantized weights
   * Use to reclaim GPU memory
   */
  void ClearCache();

  std::size_t ScratchReservedBytes() const;
  std::size_t ScratchInUseBytes() const;
  std::size_t ScratchHighWaterBytes() const {
    return scratch_high_water_bytes_;
  }
  bool HasScratchBuffer() const { return scratch_buffer_ != nullptr; }
  void ReleaseScratchBuffer();

private:
  /**
   * @brief Get tensor name for a layer component
   *
   * Generates HuggingFace-style tensor names:
   * - With type (3-4 params):
   * "model.layers.{layer}.{component}.{type}.{suffix}"
   * - Without type (2 params): "model.layers.{layer}.{component}.{suffix}"
   *
   * @param layer Layer index
   * @param component Component name (e.g., "self_attn", "mlp",
   * "input_layernorm")
   * @param type Optional type (e.g., "q_proj", "gate_proj"). Empty for simple
   * components
   * @param suffix Suffix (default "weight", or "bias")
   */
  std::string GetLayerTensorName(int layer, const std::string &component,
                                 const std::string &type = "",
                                 const std::string &suffix = "weight") const;
  struct LayerWeights {
    mutable const half *q_proj{nullptr};
    mutable const half *k_proj{nullptr};
    mutable const half *v_proj{nullptr};
    mutable const half *o_proj{nullptr};
    mutable const half *input_norm{nullptr};
    mutable const half *post_attn_norm{nullptr};
    mutable const half *gate_proj{nullptr};
    mutable const half *up_proj{nullptr};
    mutable const half *down_proj{nullptr};
    mutable MmqWeightInfo down_proj_mmq{};
    // Biases
    mutable const half *q_proj_bias{nullptr};
    mutable const half *k_proj_bias{nullptr};
    mutable const half *v_proj_bias{nullptr};

    // Keep weight accessor references for lazy dequantization
    std::shared_ptr<IWeightAccessor> q_proj_accessor;
    std::shared_ptr<IWeightAccessor> k_proj_accessor;
    std::shared_ptr<IWeightAccessor> v_proj_accessor;
    std::shared_ptr<IWeightAccessor> o_proj_accessor;
    std::shared_ptr<IWeightAccessor> input_norm_accessor;
    std::shared_ptr<IWeightAccessor> post_attn_norm_accessor;
    std::shared_ptr<IWeightAccessor> gate_proj_accessor;
    std::shared_ptr<IWeightAccessor> up_proj_accessor;
    std::shared_ptr<IWeightAccessor> down_proj_accessor;
    std::shared_ptr<IWeightAccessor> q_proj_bias_accessor;
    std::shared_ptr<IWeightAccessor> k_proj_bias_accessor;
    std::shared_ptr<IWeightAccessor> v_proj_bias_accessor;
  };

  // Helper to get dequantized weights (lazy evaluation, permanent cache).
  // Guarded by dequant_cache_mu_ to prevent races when multiple threads
  // (e.g. prefill/decode overlap lanes) lazily populate cache pointers.
  const half *GetDequantizedWeights(std::shared_ptr<IWeightAccessor> accessor,
                                    const half *&cache_ptr) const;

  // Dequantize into shared scratch buffer (no per-tensor caching)
  const half *
  DequantizeToScratch(std::shared_ptr<IWeightAccessor> accessor) const;

  IModelLoader *loader_{nullptr};
  cudaStream_t stream_{nullptr};
  int num_layers_{0};
  bool is_quantized_{false};
  std::string quantization_type_;

  std::vector<LayerWeights> layers_;
  mutable const half *embed_tokens_{nullptr};
  mutable const half *final_norm_{nullptr};
  mutable const half *lm_head_{nullptr};

  // Scratch buffer for on-demand projection dequantization
  mutable half *scratch_buffer_{nullptr};
  mutable size_t scratch_buffer_elements_{0};
  mutable size_t scratch_high_water_bytes_{0};
  mutable std::mutex dequant_cache_mu_;
  mutable std::mutex mmq_cache_mu_;
  bool allow_fused_quantized_matmul_{true};
  bool batch_dequant_cache_enabled_{false};

  // Global weight accessors
  std::shared_ptr<IWeightAccessor> embed_tokens_accessor;
  std::shared_ptr<IWeightAccessor> final_norm_accessor;
  std::shared_ptr<IWeightAccessor> lm_head_accessor;
};

} // namespace inferflux
