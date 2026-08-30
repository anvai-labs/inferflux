#include "runtime/backends/cuda/native/safetensors_adapter.h"

// sizeof_half on CPU builds: the guarded headers make __half opaque.
#ifndef INFERFLUX_HAS_CUDA
#define sizeof_half 2
#else
#define sizeof_half sizeof(half)
#endif

#include <cstring>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

SafetensorsWeightAccessor::SafetensorsWeightAccessor(
    const SafetensorsLoader::Tensor *tensor, void *gpu_base)
    : tensor_(tensor), gpu_base_(gpu_base) {}

SafetensorsWeightAccessor::~SafetensorsWeightAccessor() = default;

std::pair<size_t, size_t> SafetensorsWeightAccessor::GetDimensions() const {
  if (!tensor_ || tensor_->shape.size() < 2) {
    return {0, 0};
  }
  return {tensor_->shape[0], tensor_->shape[1]};
}

std::string SafetensorsWeightAccessor::GetDataType() const {
  if (!tensor_) {
    return "unknown";
  }
  return tensor_->dtype;
}

bool SafetensorsWeightAccessor::IsQuantized() const { return false; }

void *SafetensorsWeightAccessor::GetGpuWeights(cudaStream_t) {
  (void)gpu_base_;
  return nullptr;
}

half *SafetensorsWeightAccessor::GetDequantizedGpuWeights(cudaStream_t) {
  return nullptr;
}

bool SafetensorsWeightAccessor::IsDequantizedCached() const { return false; }

SafetensorsLoaderAdapter::SafetensorsLoaderAdapter()
    : loader_(std::make_unique<SafetensorsLoader>()) {}

SafetensorsLoaderAdapter::~SafetensorsLoaderAdapter() = default;

bool SafetensorsLoaderAdapter::Load(const std::filesystem::path &model_path) {
  if (!loader_) {
    return false;
  }
  if (!loader_->LoadModel(model_path.string())) {
    return false;
  }
  ConvertModelInfo();
  return true;
}

const ModelInfo &SafetensorsLoaderAdapter::GetModelInfo() const {
  return model_info_;
}

std::string SafetensorsLoaderAdapter::GetFormat() const {
  return "safetensors";
}

bool SafetensorsLoaderAdapter::IsQuantized() const { return false; }

std::string SafetensorsLoaderAdapter::GetQuantizationType() const { return ""; }

bool SafetensorsLoaderAdapter::UploadToGPU(cudaStream_t) { return false; }

void SafetensorsLoaderAdapter::FreeCPUMemory() {
  if (loader_) {
    loader_->FreeCPUMemory();
  }
}

void SafetensorsLoaderAdapter::FreeGPUMemory() {
  if (loader_) {
    loader_->FreeGPUMemory();
  }
}

void *SafetensorsLoaderAdapter::GetGPUBuffer() const { return nullptr; }

size_t SafetensorsLoaderAdapter::GetGPUSize() const { return 0; }

void SafetensorsLoaderAdapter::SetDequantizedCachePolicy(
    DequantizedCachePolicy policy) {
  dequantized_cache_policy_ = policy;
}

DequantizedCachePolicy
SafetensorsLoaderAdapter::GetDequantizedCachePolicy() const {
  return dequantized_cache_policy_;
}

void SafetensorsLoaderAdapter::ClearDequantizedCache() {}
bool SafetensorsLoaderAdapter::HasDequantizedCache() const { return false; }

std::shared_ptr<IWeightAccessor>
SafetensorsLoaderAdapter::GetWeightAccessor(const std::string &) {
  return nullptr;
}

std::vector<std::string> SafetensorsLoaderAdapter::GetTensorNames() const {
  if (!loader_) {
    return {};
  }
  return loader_->GetTensorNames();
}

void SafetensorsLoaderAdapter::ConvertModelInfo() {
  model_info_ = ModelInfo{};
  if (!loader_) {
    return;
  }
  const auto &config = loader_->GetConfig();
  model_info_.hidden_size = config.hidden_size;
  model_info_.num_hidden_layers = config.num_hidden_layers;
  model_info_.num_attention_heads = config.num_attention_heads;
  model_info_.num_key_value_heads = config.num_key_value_heads;
  model_info_.head_dim = config.head_dim;
  model_info_.intermediate_size = config.intermediate_size;
  model_info_.vocab_size = config.vocab_size;
  model_info_.max_position_embeddings = config.max_position_embeddings;
  model_info_.rope_freq_base = config.rope_freq_base;
  model_info_.rope_freq_scale = config.rope_freq_scale;
  model_info_.rope_dim = config.rope_dim;
  model_info_.model_type = config.model_type;
  model_info_.activation = config.activation;
  model_info_.torch_dtype = config.torch_dtype;
  model_info_.rms_norm_eps = config.rms_norm_eps;
}

void SafetensorsQuantizationHandler::DequantizeGpuToGpu(const void *quantized,
                                                        half *dequantized,
                                                        size_t num_elements,
                                                        cudaStream_t) {
  if (!quantized || !dequantized || num_elements == 0) {
    return;
  }
  std::memcpy(dequantized, quantized, num_elements * sizeof_half);
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
