#include "runtime/backends/cuda/native/quantized_weight_map.h"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/quantization_handler.h"
#include "server/logging/logger.h"

namespace inferflux {

QuantizedWeightMap::~QuantizedWeightMap() {
  // Note: per-tensor GPU memory is managed by IModelLoader, we don't own it
#ifdef INFERFLUX_HAS_CUDA
  for (auto &lw : layers_) {
    FusedQuantGemm::DestroyDownProjMmqLayout(lw.down_proj_mmq);
    lw.down_proj_mmq = {};
  }
  ReleaseScratchBuffer();
#endif
}

bool QuantizedWeightMap::Build(IModelLoader *loader, const ModelInfo &config,
                               cudaStream_t stream) {
  if (!loader) {
    log::Error("quantized_weight_map", "Null loader");
    return false;
  }

  loader_ = loader;
  stream_ = stream;
  num_layers_ = config.num_hidden_layers;
  is_quantized_ = loader->IsQuantized();
  quantization_type_ = loader->GetQuantizationType();
  scratch_high_water_bytes_ = 0;

  log::Info("quantized_weight_map",
            "Building weight map: layers=" + std::to_string(num_layers_) +
                " quantized=" + std::string(is_quantized_ ? "true" : "false") +
                " type=" + quantization_type_);

  // Allocate layer structures
  layers_.resize(num_layers_);

  // Build tensor names for each layer
  for (int layer = 0; layer < num_layers_; ++layer) {
    auto &lw = layers_[layer];

    // Get weight accessors for all layer tensors
    lw.q_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "q_proj"));
    lw.k_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "k_proj"));
    lw.v_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "v_proj"));
    lw.o_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "o_proj"));

    lw.input_norm_accessor =
        loader->GetWeightAccessor(GetLayerTensorName(layer, "input_layernorm"));
    lw.post_attn_norm_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "post_attention_layernorm"));

    lw.gate_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "mlp", "gate_proj"));
    lw.up_proj_accessor =
        loader->GetWeightAccessor(GetLayerTensorName(layer, "mlp", "up_proj"));
    lw.down_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "mlp", "down_proj"));

    // Biases (optional)
    lw.q_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "q_proj", "bias"));
    lw.k_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "k_proj", "bias"));
    lw.v_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "v_proj", "bias"));
  }

  // Global tensors
  embed_tokens_accessor =
      loader->GetWeightAccessor("model.embed_tokens.weight");
  final_norm_accessor = loader->GetWeightAccessor("model.norm.weight");
  lm_head_accessor = loader->GetWeightAccessor("lm_head.weight");
  if (!lm_head_accessor && embed_tokens_accessor) {
    log::Info("quantized_weight_map",
              "lm_head.weight not found, using tied embeddings");
    lm_head_accessor = embed_tokens_accessor;
  }

#ifdef INFERFLUX_HAS_CUDA
  // Compute scratch buffer size for cuBLAS fallback dequantization.
  // Allocation is deferred to first use — decode-only workloads (fused GEMV)
  // never need it, saving up to hundreds of MB for large-vocab models.
  if (is_quantized_) {
    size_t max_elements = 0;
    for (int layer = 0; layer < num_layers_; ++layer) {
      auto &lw = layers_[layer];
      for (auto &acc :
           {lw.q_proj_accessor, lw.k_proj_accessor, lw.v_proj_accessor,
            lw.o_proj_accessor, lw.gate_proj_accessor, lw.up_proj_accessor,
            lw.down_proj_accessor}) {
        if (acc && acc->IsQuantized()) {
          auto dims = acc->GetDimensions();
          max_elements = std::max(max_elements, dims.first * dims.second);
        }
      }
    }
    // Intentionally exclude lm_head from scratch sizing.
    // Large vocab projection can dominate scratch reservation while regular
    // projection fallback only needs per-layer matrices. When lm_head falls
    // back and exceeds scratch, we use accessor-level dequant cache instead.
    scratch_buffer_elements_ = max_elements; // Size known, allocation deferred
    if (max_elements > 0) {
      log::Info(
          "quantized_weight_map",
          "Scratch buffer: " +
              std::to_string(max_elements * sizeof(half) / 1024 / 1024) +
              " MiB (deferred allocation, used only for cuBLAS fallback)");
    }
  }
#endif

  log::Info("quantized_weight_map", "Weight map built successfully");
  return true;
}

std::string
QuantizedWeightMap::GetLayerTensorName(int layer, const std::string &component,
                                       const std::string &type,
                                       const std::string &suffix) const {

  // HuggingFace style naming:
  // model.layers.{layer}.{component}[.{type}].{suffix} Examples:
  // - model.layers.0.self_attn.q_proj.weight (with type)
  // - model.layers.0.input_layernorm.weight (without type)
  // - model.layers.0.self_attn.q_proj.bias (with bias suffix)

  if (type.empty()) {
    // Simple component: model.layers.{layer}.{component}.{suffix}
    return "model.layers." + std::to_string(layer) + "." + component + "." +
           suffix;
  }
  // Complex component: model.layers.{layer}.{component}.{type}.{suffix}
  return "model.layers." + std::to_string(layer) + "." + component + "." +
         type + "." + suffix;
}

const half *QuantizedWeightMap::GetDequantizedWeights(
    std::shared_ptr<IWeightAccessor> accessor, const half *&cache_ptr) const {

  if (!accessor) {
    return nullptr;
  }

  // Fast path: return cached if already populated (read is safe once written).
  if (cache_ptr) {
    return cache_ptr;
  }

  // Slow path: lazy dequantization under lock to prevent concurrent races
  // when multiple threads populate the same cache slot.
  std::lock_guard<std::mutex> lock(dequant_cache_mu_);

  // Double-check after acquiring lock.
  if (cache_ptr) {
    return cache_ptr;
  }

  cache_ptr = accessor->GetDequantizedGpuWeights(stream_);

  if (!cache_ptr) {
    log::Warn("quantized_weight_map", "Failed to get dequantized weights");
  }

  return cache_ptr;
}

const half *QuantizedWeightMap::DequantizeToScratch(
    std::shared_ptr<IWeightAccessor> accessor) const {
#ifdef INFERFLUX_HAS_CUDA
  if (!accessor) {
    return nullptr;
  }

  // Non-quantized weights have permanent GPU pointers — return directly
  if (!accessor->IsQuantized()) {
    return accessor->GetDequantizedGpuWeights(stream_);
  }

  // Quantized: dequantize into shared scratch buffer (lazy allocation)
  auto dims = accessor->GetDimensions();
  size_t num_elements = dims.first * dims.second;
  if (num_elements > scratch_buffer_elements_ ||
      scratch_buffer_elements_ == 0) {
    log::Warn("quantized_weight_map",
              "Tensor too large for scratch buffer, falling back to cache");
    return accessor->GetDequantizedGpuWeights(stream_);
  }

  // Lazy allocate on first use (decode-only workloads skip this entirely)
  if (!scratch_buffer_) {
    auto err = cudaMalloc(reinterpret_cast<void **>(&scratch_buffer_),
                          scratch_buffer_elements_ * sizeof(half));
    if (err != cudaSuccess) {
      log::Warn("quantized_weight_map",
                "Failed to allocate scratch buffer, falling back to cache");
      scratch_buffer_elements_ = 0;
      return accessor->GetDequantizedGpuWeights(stream_);
    }
    log::Info("quantized_weight_map",
              "Scratch buffer allocated: " +
                  std::to_string(scratch_buffer_elements_ * sizeof(half) /
                                 1024 / 1024) +
                  " MiB (cuBLAS fallback triggered)");
    scratch_high_water_bytes_ =
        std::max(scratch_high_water_bytes_, ScratchReservedBytes());
  }

  // Get the quantization handler and dequantize directly into scratch
  void *raw_gpu = accessor->GetGpuWeights(stream_);
  if (!raw_gpu) {
    return nullptr;
  }

  auto type_str = accessor->GetDataType();
  auto handler =
      runtime::cuda::native::QuantizationHandlerRegistry::Instance().Create(
          type_str);
  if (!handler) {
    log::Warn("quantized_weight_map",
              "No handler for " + type_str + ", falling back to cache");
    return accessor->GetDequantizedGpuWeights(stream_);
  }

  handler->DequantizeGpuToGpu(raw_gpu, scratch_buffer_, num_elements, stream_);
  return scratch_buffer_;
#else
  (void)accessor;
  return nullptr;
#endif // INFERFLUX_HAS_CUDA
}

// --- Per-layer accessors ---
// Projection weights use scratch buffer (no per-tensor caching).
// Norms/embeddings use permanent cache (small, accessed repeatedly).

const half *QuantizedWeightMap::LayerQProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].q_proj_accessor,
                                 layers_[layer].q_proj);
  return DequantizeToScratch(layers_[layer].q_proj_accessor);
}

const half *QuantizedWeightMap::LayerKProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].k_proj_accessor,
                                 layers_[layer].k_proj);
  return DequantizeToScratch(layers_[layer].k_proj_accessor);
}

const half *QuantizedWeightMap::LayerVProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].v_proj_accessor,
                                 layers_[layer].v_proj);
  return DequantizeToScratch(layers_[layer].v_proj_accessor);
}

const half *QuantizedWeightMap::LayerOProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].o_proj_accessor,
                                 layers_[layer].o_proj);
  return DequantizeToScratch(layers_[layer].o_proj_accessor);
}

const half *QuantizedWeightMap::LayerInputNorm(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].input_norm_accessor,
                               layers_[layer].input_norm);
}

const half *QuantizedWeightMap::LayerPostAttnNorm(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].post_attn_norm_accessor,
                               layers_[layer].post_attn_norm);
}

const half *QuantizedWeightMap::LayerGateProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].gate_proj_accessor,
                                 layers_[layer].gate_proj);
  return DequantizeToScratch(layers_[layer].gate_proj_accessor);
}

const half *QuantizedWeightMap::LayerUpProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].up_proj_accessor,
                                 layers_[layer].up_proj);
  return DequantizeToScratch(layers_[layer].up_proj_accessor);
}

const half *QuantizedWeightMap::LayerDownProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return nullptr;
  if (batch_dequant_cache_enabled_)
    return GetDequantizedWeights(layers_[layer].down_proj_accessor,
                                 layers_[layer].down_proj);
  return DequantizeToScratch(layers_[layer].down_proj_accessor);
}

// --- Bias accessors ---

const half *QuantizedWeightMap::LayerQProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].q_proj_bias_accessor,
                               layers_[layer].q_proj_bias);
}

const half *QuantizedWeightMap::LayerKProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].k_proj_bias_accessor,
                               layers_[layer].k_proj_bias);
}

const half *QuantizedWeightMap::LayerVProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].v_proj_bias_accessor,
                               layers_[layer].v_proj_bias);
}

// --- Global accessors ---

const half *QuantizedWeightMap::EmbedTokens() const {
  return GetDequantizedWeights(embed_tokens_accessor, embed_tokens_);
}

const half *QuantizedWeightMap::FinalNorm() const {
  return GetDequantizedWeights(final_norm_accessor, final_norm_);
}

const half *QuantizedWeightMap::LmHead() const {
  if (!lm_head_accessor) {
    return nullptr;
  }

  // In memory-first modes, avoid persisting a large dequantized lm_head cache.
  // Keep model-lifetime behavior unchanged for throughput-first deployments.
  if (loader_ && lm_head_accessor->IsQuantized() &&
      loader_->GetDequantizedCachePolicy() !=
          runtime::cuda::native::DequantizedCachePolicy::kModelLifetime) {
    return DequantizeToScratch(lm_head_accessor);
  }

  return GetDequantizedWeights(lm_head_accessor, lm_head_);
}

// --- Raw quantized weight accessors ---

namespace {
QuantizedWeightInfo MakeRawInfo(std::shared_ptr<IWeightAccessor> accessor,
                                cudaStream_t stream) {
  if (!accessor || !accessor->IsQuantized()) {
    return {};
  }
  auto dims = accessor->GetDimensions();
  auto type_str = accessor->GetDataType();
  auto tensor_type = runtime::cuda::native::StringToTensorType(type_str);
  QuantizedWeightInfo info;
  info.data = accessor->GetGpuWeights(stream);
  info.quant_type = static_cast<int>(tensor_type);
  info.num_elements = static_cast<int64_t>(dims.first * dims.second);
  return info;
}
} // namespace

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerQProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].q_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerKProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].k_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerVProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].v_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerOProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].o_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerGateProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].gate_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerUpProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].up_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerDownProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].down_proj_accessor, stream_);
}

MmqWeightInfo QuantizedWeightMap::GetMmqLayerDownProj(int layer) const {
#ifndef INFERFLUX_HAS_CUDA
  (void)layer;
  return {};
#else
  if (layer < 0 || layer >= num_layers_) {
    return {};
  }

  auto &lw = layers_[layer];
  if (lw.down_proj_mmq.data) {
    return lw.down_proj_mmq;
  }
  if (!lw.down_proj_accessor || !lw.down_proj_accessor->IsQuantized()) {
    return {};
  }

  const auto dims = lw.down_proj_accessor->GetDimensions();
  if (dims.first == 0 || dims.second == 0) {
    return {};
  }

  QuantizedWeightInfo raw = MakeRawInfo(lw.down_proj_accessor, stream_);
  if (!FusedQuantGemm::SupportsDownProjMmq(raw.quant_type)) {
    return {};
  }

  std::lock_guard<std::mutex> lock(mmq_cache_mu_);
  if (lw.down_proj_mmq.data) {
    return lw.down_proj_mmq;
  }

  MmqWeightInfo layout{};
  if (!FusedQuantGemm::BuildDownProjMmqLayout(raw, static_cast<int>(dims.first),
                                              static_cast<int>(dims.second),
                                              &layout, stream_)) {
    return {};
  }

  lw.down_proj_mmq = layout;
  return lw.down_proj_mmq;
#endif
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLmHead() const {
  return MakeRawInfo(lm_head_accessor, stream_);
}

// --- Utility ---

bool QuantizedWeightMap::HasTensor(const std::string &name) const {
  return loader_->GetWeightAccessor(name) != nullptr;
}

std::shared_ptr<IWeightAccessor>
QuantizedWeightMap::GetWeightAccessor(const std::string &name) {
  return loader_->GetWeightAccessor(name);
}

void QuantizedWeightMap::ClearCache() {
  log::Debug("quantized_weight_map", "Clearing dequantized weight cache");

  // Nullify the PROJECTION cache slots too: with the batch dequant cache
  // enabled (INFERFLUX_BATCH_DEQUANT_CACHE=1) these route through
  // GetDequantizedWeights and hold loader-level dequantized buffers — the
  // per-batch cleanup frees exactly those in the loader, and a stale map
  // pointer here meant the next batch's fast path handed freed device
  // memory to the cuBLAS/dequant fallback GEMM (use-after-free). The
  // scratch-buffer comment below predates that path.
  for (auto &lw : layers_) {
    lw.q_proj = nullptr;
    lw.k_proj = nullptr;
    lw.v_proj = nullptr;
    lw.o_proj = nullptr;
    lw.gate_proj = nullptr;
    lw.up_proj = nullptr;
    lw.down_proj = nullptr;
    lw.input_norm = nullptr;
    lw.post_attn_norm = nullptr;
    lw.q_proj_bias = nullptr;
    lw.k_proj_bias = nullptr;
    lw.v_proj_bias = nullptr;
  }

  // Keep token embeddings and final norm resident. They are reused by every
  // request and dequantizing quantized embeddings again is prohibitively
  // expensive.

  // lm_head may be explicitly memory-first, so drop the cached pointer unless
  // the loader is in model-lifetime mode or lm_head is tied to embeddings.
  if (!loader_ ||
      loader_->GetDequantizedCachePolicy() !=
          runtime::cuda::native::DequantizedCachePolicy::kModelLifetime) {
    if (lm_head_accessor != embed_tokens_accessor) {
      lm_head_ = nullptr;
    }
  }
}

std::size_t QuantizedWeightMap::ScratchReservedBytes() const {
#ifdef INFERFLUX_HAS_CUDA
  if (!scratch_buffer_ || scratch_buffer_elements_ == 0) {
    return 0;
  }
  return scratch_buffer_elements_ * sizeof(half);
#else
  return 0;
#endif
}

std::size_t QuantizedWeightMap::ScratchInUseBytes() const {
  return ScratchReservedBytes();
}

void QuantizedWeightMap::ReleaseScratchBuffer() {
#ifdef INFERFLUX_HAS_CUDA
  if (!scratch_buffer_) {
    return;
  }
  if (cudaFree(scratch_buffer_) != cudaSuccess) {
    log::Warn("quantized_weight_map", "Failed to release scratch buffer");
    return;
  }
  scratch_buffer_ = nullptr;
#endif
}

} // namespace inferflux
