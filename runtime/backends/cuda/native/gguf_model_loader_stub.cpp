#include "runtime/backends/cuda/native/gguf_model_loader.h"

#include <cstdio>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

std::shared_ptr<IQuantizationHandler>
GGUFTensorData::GetQuantizationHandler() const {
  if (!info.is_quantized()) {
    return CreateQuantizationHandler("none");
  }
  return CreateQuantizationHandler(GetQuantizationType(info.type));
}

GGUFModelLoader::GGUFModelLoader() = default;

GGUFModelLoader::~GGUFModelLoader() { FreeGPUMemoryImpl(); }

bool GGUFModelLoader::Load(const std::filesystem::path &model_path) {
  model_path_ = model_path;
  tensors_.clear();
  gguf_to_internal_name_map_.clear();
  internal_to_gguf_name_map_.clear();
  weight_accessor_cache_.clear();
  tokenizer_pieces_.clear();
  tokenizer_eos_token_id_ = -1;
  tokenizer_bos_token_id_ = -1;
  model_info_ = ModelInfo{};
  model_type_.clear();
  quantization_type_.clear();
  data_section_offset_ = 0;

  std::FILE *file = std::fopen(model_path.c_str(), "rb");
  if (!file) {
    return false;
  }

  const bool ok = ParseHeader(file);
  std::fclose(file);
  return ok;
}

const ModelInfo &GGUFModelLoader::GetModelInfo() const { return model_info_; }

std::string GGUFModelLoader::GetFormat() const { return "gguf"; }

bool GGUFModelLoader::IsQuantized() const {
  return !quantization_type_.empty();
}

std::string GGUFModelLoader::GetQuantizationType() const {
  return quantization_type_;
}

bool GGUFModelLoader::UploadToGPU(cudaStream_t) { return true; }

void GGUFModelLoader::FreeCPUMemory() {
  for (auto &[name, tensor] : tensors_) {
    (void)name;
    tensor.ClearCPUMemory();
  }
}

void GGUFModelLoader::FreeGPUMemory() { FreeGPUMemoryImpl(); }

void *GGUFModelLoader::GetGPUBuffer() const { return d_quantized_buffer_; }

size_t GGUFModelLoader::GetGPUSize() const {
  return quantized_buffer_size_ + dequantized_buffer_size_;
}

void GGUFModelLoader::SetDequantizedCachePolicy(DequantizedCachePolicy policy) {
  dequantized_cache_policy_ = policy;
}

DequantizedCachePolicy GGUFModelLoader::GetDequantizedCachePolicy() const {
  return dequantized_cache_policy_;
}

void GGUFModelLoader::ClearDequantizedCache() {
  for (auto &[name, tensor] : tensors_) {
    (void)name;
    tensor.dequantized_gpu = nullptr;
  }
}

std::shared_ptr<IWeightAccessor>
GGUFModelLoader::GetWeightAccessor(const std::string &) {
  return nullptr;
}

const GGUFTensorData *
GGUFModelLoader::GetTensorByGGUFName(const std::string &gguf_name) const {
  auto it = tensors_.find(gguf_name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::string> GGUFModelLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(internal_to_gguf_name_map_.size());
  for (const auto &entry : internal_to_gguf_name_map_) {
    names.push_back(entry.first);
  }
  return names;
}

void GGUFModelLoader::FreeGPUMemoryImpl() {
  d_quantized_buffer_ = nullptr;
  d_dequantized_buffer_ = nullptr;
  quantized_buffer_size_ = 0;
  dequantized_buffer_size_ = 0;
  for (auto &[name, tensor] : tensors_) {
    (void)name;
    tensor.gpu_data = nullptr;
    tensor.dequantized_gpu = nullptr;
    tensor.gpu_offset = 0;
  }
}

bool GGUFModelLoader::ParseHeader(FILE *file) {
  if (!file) {
    return false;
  }
  if (std::fread(&header_.magic, sizeof(header_.magic), 1, file) != 1) {
    return false;
  }
  if (std::fread(&header_.version, sizeof(header_.version), 1, file) != 1) {
    return false;
  }
  if (std::fread(&header_.tensor_count, sizeof(header_.tensor_count), 1,
                 file) != 1) {
    return false;
  }
  if (std::fread(&header_.kv_count, sizeof(header_.kv_count), 1, file) != 1) {
    return false;
  }
  return ValidateGGUFHeader(header_);
}

bool GGUFModelLoader::ParseKeyValuePairs(FILE *) { return true; }

bool GGUFModelLoader::ParseTensorInfo(FILE *) { return true; }

bool GGUFModelLoader::LoadTensorData(FILE *) { return true; }

bool GGUFModelLoader::ExtractModelInfo() { return true; }

void GGUFModelLoader::BuildTensorNameMapping() {}

bool GGUFModelLoader::UploadQuantizedToGPU(cudaStream_t) { return true; }

bool GGUFModelLoader::UploadDequantizedToGPU(cudaStream_t) { return true; }

size_t GGUFModelLoader::CalcGPUBufferSize() const { return 0; }

GGUFWeightAccessor::GGUFWeightAccessor(GGUFTensorData *tensor,
                                       bool *dequant_dirty_flag)
    : tensor_(tensor), dequant_dirty_flag_(dequant_dirty_flag) {}

std::pair<size_t, size_t> GGUFWeightAccessor::GetDimensions() const {
  if (!tensor_ || tensor_->info.shape.size() < 2) {
    return {0, 0};
  }
  return {static_cast<size_t>(tensor_->info.shape[0]),
          static_cast<size_t>(tensor_->info.shape[1])};
}

std::string GGUFWeightAccessor::GetDataType() const {
  if (!tensor_) {
    return "unknown";
  }
  return TensorTypeToString(tensor_->info.type);
}

bool GGUFWeightAccessor::IsQuantized() const {
  return tensor_ && tensor_->info.is_quantized();
}

void *GGUFWeightAccessor::GetGpuWeights(cudaStream_t) {
  if (!tensor_) {
    return nullptr;
  }
  return tensor_->gpu_data;
}

half *GGUFWeightAccessor::GetDequantizedGpuWeights(cudaStream_t) {
  if (!tensor_) {
    return nullptr;
  }
  return tensor_->dequantized_gpu;
}

bool GGUFWeightAccessor::IsDequantizedCached() const {
  return tensor_ && tensor_->dequantized_gpu != nullptr;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
