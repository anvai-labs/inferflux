#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "server/diagnostics/crash_handler.h"
#include "server/logging/logger.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

bool ReadU32(FILE *file, uint32_t *out) {
  if (!file || !out) {
    return false;
  }
  return fread(out, sizeof(uint32_t), 1, file) == 1;
}

bool ReadU64(FILE *file, uint64_t *out) {
  if (!file || !out) {
    return false;
  }
  return fread(out, sizeof(uint64_t), 1, file) == 1;
}

bool SizeToLong(std::size_t value, long *out) {
  if (!out) {
    return false;
  }
  if (value > static_cast<std::size_t>(LONG_MAX)) {
    return false;
  }
  *out = static_cast<long>(value);
  return true;
}

int SaturatingToInt(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

bool IsPowerOfTwo(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t rem = value % alignment;
  if (rem == 0) {
    return value;
  }
  return value + (alignment - rem);
}

bool CheckCudaStatus(cudaError_t status, const char *component,
                     const std::string &operation) {
  if (status == cudaSuccess) {
    return true;
  }
  log::Error(component, operation + " failed: " + cudaGetErrorString(status));
  diagnostics::RecordCudaError();
  return false;
}

bool ShouldRetainDequantizedTensor(const std::string &name) {
  return name == "token_embd.weight" || name == "tok_emb.weight";
}

} // namespace

//==============================================================================
// GGUFTensorData Implementation
//==============================================================================

std::shared_ptr<IQuantizationHandler>
GGUFTensorData::GetQuantizationHandler() const {
  std::string qtype = GetQuantizationType(info.type);
  return CreateQuantizationHandler(qtype);
}

//==============================================================================
// GGUFWeightAccessor Implementation
//==============================================================================

GGUFWeightAccessor::GGUFWeightAccessor(GGUFTensorData *tensor,
                                       bool *dequant_dirty_flag)
    : tensor_(tensor), dequant_dirty_flag_(dequant_dirty_flag) {
  if (!tensor_) {
    log::Error("gguf_weight_accessor", "Null tensor");
  }
}

std::pair<size_t, size_t> GGUFWeightAccessor::GetDimensions() const {
  if (!tensor_ || tensor_->info.shape.size() < 2) {
    return {0, 0};
  }
  return {tensor_->info.shape[0], tensor_->info.shape[1]};
}

std::string GGUFWeightAccessor::GetDataType() const {
  if (!tensor_) {
    return "unknown";
  }
  return TensorTypeToString(tensor_->info.type);
}

bool GGUFWeightAccessor::IsQuantized() const {
  if (!tensor_) {
    return false;
  }
  return tensor_->info.is_quantized();
}

void *GGUFWeightAccessor::GetGpuWeights(cudaStream_t stream) {
  (void)stream;
  if (!tensor_) {
    return nullptr;
  }
  return tensor_->gpu_data;
}

half *GGUFWeightAccessor::GetDequantizedGpuWeights(cudaStream_t stream) {
  if (!tensor_) {
    return nullptr;
  }

  if (!tensor_->info.is_quantized()) {
    if (tensor_->info.type == GGUF::TensorType::F16) {
      return static_cast<half *>(tensor_->gpu_data);
    }
    if (tensor_->info.type == GGUF::TensorType::F32) {
      if (tensor_->dequantized_gpu) {
        return tensor_->dequantized_gpu;
      }
      size_t num_elements = 1;
      for (uint64_t dim : tensor_->info.shape) {
        if (dim == 0 ||
            dim > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            num_elements >
                std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
          log::Error("gguf_weight_accessor",
                     "Invalid F32 tensor shape for conversion: " +
                         tensor_->info.name);
          return nullptr;
        }
        num_elements *= static_cast<size_t>(dim);
      }
      if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        log::Error("gguf_weight_accessor",
                   "F32 tensor too large to convert: " + tensor_->info.name);
        return nullptr;
      }

      half *d_dequantized = nullptr;
      if (!CheckCudaStatus(cudaMalloc(reinterpret_cast<void **>(&d_dequantized),
                                      num_elements * sizeof(half)),
                           "gguf_weight_accessor",
                           "cudaMalloc(f32->f16 cache)")) {
        return nullptr;
      }

      std::vector<float> host_f32(num_elements);
      if (!CheckCudaStatus(
              cudaMemcpy(host_f32.data(), tensor_->gpu_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "gguf_weight_accessor", "cudaMemcpy(f32 device->host)")) {
        cudaFree(d_dequantized);
        return nullptr;
      }
      std::vector<half> host_f16(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        host_f16[i] = __float2half(host_f32[i]);
      }
      if (!CheckCudaStatus(cudaMemcpyAsync(d_dequantized, host_f16.data(),
                                           num_elements * sizeof(half),
                                           cudaMemcpyHostToDevice, stream),
                           "gguf_weight_accessor",
                           "cudaMemcpyAsync(f32->f16 host->device)")) {
        cudaFree(d_dequantized);
        return nullptr;
      }
      if (!CheckCudaStatus(cudaStreamSynchronize(stream),
                           "gguf_weight_accessor",
                           "cudaStreamSynchronize(f32->f16 cache)")) {
        cudaFree(d_dequantized);
        return nullptr;
      }
      tensor_->dequantized_gpu = d_dequantized;
      if (dequant_dirty_flag_)
        *dequant_dirty_flag_ = true;
      return d_dequantized;
    }
    log::Error("gguf_weight_accessor",
               "Unsupported non-quantized GGUF tensor type: " +
                   tensor_->info.type_string());
    return nullptr;
  }

  // Return cached dequantized data if available
  if (tensor_->dequantized_gpu) {
    return tensor_->dequantized_gpu;
  }

  // Perform lazy dequantization
  auto handler = tensor_->GetQuantizationHandler();
  if (!handler) {
    log::Error("gguf_weight_accessor", "No quantization handler");
    return nullptr;
  }

  // Calculate dequantized size
  size_t num_elements = 1;
  for (uint64_t dim : tensor_->info.shape) {
    if (dim == 0 ||
        dim > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        num_elements >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      log::Error("gguf_weight_accessor",
                 "Invalid tensor shape for dequantization: " +
                     tensor_->info.name);
      return nullptr;
    }
    num_elements *= static_cast<size_t>(dim);
  }
  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(half)) {
    log::Error("gguf_weight_accessor",
               "Tensor is too large to allocate dequantized buffer: " +
                   tensor_->info.name);
    return nullptr;
  }

  // Allocate dequantized buffer
  half *d_dequantized = nullptr;
  if (!CheckCudaStatus(cudaMalloc(reinterpret_cast<void **>(&d_dequantized),
                                  num_elements * sizeof(half)),
                       "gguf_weight_accessor",
                       "cudaMalloc(dequantized buffer)")) {
    return nullptr;
  }

  // Dequantize
  handler->DequantizeGpuToGpu(tensor_->gpu_data, d_dequantized, num_elements,
                              stream);
  if (!CheckCudaStatus(cudaGetLastError(), "gguf_weight_accessor",
                       "DequantizeGpuToGpu launch")) {
    cudaFree(d_dequantized);
    return nullptr;
  }

  // Cache result
  tensor_->dequantized_gpu = d_dequantized;
  if (dequant_dirty_flag_)
    *dequant_dirty_flag_ = true;

  log::Debug("gguf_weight_accessor",
             "Dequantized tensor: " + tensor_->info.name + " (" +
                 std::to_string(num_elements) + " elements)");

  return d_dequantized;
}

bool GGUFWeightAccessor::IsDequantizedCached() const {
  return tensor_ && tensor_->IsDequantizedCached();
}

//==============================================================================
// GGUFModelLoader Implementation
//==============================================================================

GGUFModelLoader::GGUFModelLoader() = default;

GGUFModelLoader::~GGUFModelLoader() { FreeGPUMemoryImpl(); }

bool GGUFModelLoader::Load(const std::filesystem::path &model_path) {
  model_path_ = model_path;

  log::Info("gguf_loader", "Loading GGUF model: " + model_path.string());

  FILE *file = fopen(model_path.string().c_str(), "rb");
  if (!file) {
    log::Error("gguf_loader", "Failed to open file: " + model_path.string());
    return false;
  }

  // Parse header
  if (!ParseHeader(file)) {
    fclose(file);
    return false;
  }

  // Parse key-value pairs (metadata)
  if (!ParseKeyValuePairs(file)) {
    fclose(file);
    return false;
  }

  // Parse tensor info
  if (!ParseTensorInfo(file)) {
    fclose(file);
    return false;
  }

  // Load tensor data
  if (!LoadTensorData(file)) {
    fclose(file);
    return false;
  }

  fclose(file);

  // Extract model info
  if (!ExtractModelInfo()) {
    log::Error("gguf_loader", "Failed to extract model info");
    return false;
  }

  // Build tensor name mapping
  BuildTensorNameMapping();

  // Detect quantization
  for (const auto &[name, tensor] : tensors_) {
    if (tensor.info.is_quantized()) {
      quantization_type_ =
          ::inferflux::runtime::cuda::native::GetQuantizationType(
              tensor.info.type);
      if (!quantization_type_.empty()) {
        break;
      }
    }
  }

  log::Info("gguf_loader",
            "Model loaded: " + model_type_ +
                " tensors=" + std::to_string(tensors_.size()) +
                " quantization=" + quantization_type_ +
                " layers=" + std::to_string(model_info_.num_hidden_layers));

  return true;
}

bool GGUFModelLoader::ParseHeader(FILE *file) {
  if (!ReadU32(file, &header_.magic) || !ReadU32(file, &header_.version) ||
      !GGUFReader::ReadInt64(file, &header_.tensor_count) ||
      !GGUFReader::ReadInt64(file, &header_.kv_count)) {
    log::Error("gguf_loader", "Failed to read header");
    return false;
  }

  if (!ValidateGGUFHeader(header_)) {
    return false;
  }
  if (header_.tensor_count < 0 || header_.kv_count < 0) {
    log::Error("gguf_loader", "Negative tensor/kv counts in GGUF header");
    return false;
  }

  log::Debug("gguf_loader",
             "GGUF header: magic=" + std::to_string(header_.magic) +
                 " version=" + std::to_string(header_.version) +
                 " tensors=" + std::to_string(header_.tensor_count) +
                 " kv_count=" + std::to_string(header_.kv_count));

  return true;
}

bool GGUFModelLoader::ParseKeyValuePairs(FILE *file) {
  log::Debug("gguf_loader", "Parsing " + std::to_string(header_.kv_count) +
                                " key-value pairs");

  alignment_ = 32;
  model_info_ = ModelInfo{};
  tokenizer_pieces_.clear();
  tokenizer_merges_.clear();
  tokenizer_pre_.clear();
  tokenizer_chat_template_.clear();
  tokenizer_eos_token_id_ = -1;
  tokenizer_bos_token_id_ = -1;

  for (int64_t i = 0; i < header_.kv_count; ++i) {
    std::string key;
    if (!GGUFReader::ReadString(file, &key)) {
      log::Error("gguf_loader", "Failed to read key");
      return false;
    }

    uint32_t type_val = 0;
    if (!ReadU32(file, &type_val)) {
      log::Error("gguf_loader", "Failed to read value type");
      return false;
    }
    GGUF::ValueType type = static_cast<GGUF::ValueType>(type_val);

    if (key == "general.architecture") {
      if (type != GGUF::ValueType::STRING) {
        log::Error("gguf_loader",
                   "general.architecture has non-string GGUF type");
        return false;
      }
      if (!GGUFReader::ReadString(file, &model_type_)) {
        log::Error("gguf_loader", "Failed to read general.architecture");
        return false;
      }
      model_info_.model_type = model_type_;
      continue;
    }

    if (key == "general.alignment") {
      uint64_t raw_alignment = 0;
      if (type == GGUF::ValueType::UINT32) {
        uint32_t v = 0;
        if (!ReadU32(file, &v)) {
          log::Error("gguf_loader", "Failed to read general.alignment");
          return false;
        }
        raw_alignment = v;
      } else if (type == GGUF::ValueType::UINT64) {
        if (!ReadU64(file, &raw_alignment)) {
          log::Error("gguf_loader", "Failed to read general.alignment");
          return false;
        }
      } else {
        log::Error("gguf_loader",
                   "general.alignment has unsupported GGUF type");
        return false;
      }
      if (raw_alignment == 0 ||
          raw_alignment >
              static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        log::Error("gguf_loader", "general.alignment is out of range");
        return false;
      }
      alignment_ = static_cast<size_t>(raw_alignment);
      if (!IsPowerOfTwo(alignment_)) {
        log::Error("gguf_loader", "general.alignment must be a power of two");
        return false;
      }
      continue;
    }

    const auto read_int_value = [&](int *out) -> bool {
      if (!out) {
        return false;
      }
      int64_t tmp = 0;
      switch (type) {
      case GGUF::ValueType::UINT32: {
        uint32_t v = 0;
        if (!ReadU32(file, &v)) {
          return false;
        }
        tmp = static_cast<int64_t>(v);
        break;
      }
      case GGUF::ValueType::INT32: {
        int32_t v = 0;
        if (fread(&v, sizeof(v), 1, file) != 1) {
          return false;
        }
        tmp = static_cast<int64_t>(v);
        break;
      }
      case GGUF::ValueType::UINT64: {
        uint64_t v = 0;
        if (!ReadU64(file, &v)) {
          return false;
        }
        if (v > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
          return false;
        }
        tmp = static_cast<int64_t>(v);
        break;
      }
      case GGUF::ValueType::INT64: {
        int64_t v = 0;
        if (!GGUFReader::ReadInt64(file, &v)) {
          return false;
        }
        tmp = v;
        break;
      }
      case GGUF::ValueType::BOOL: {
        uint8_t v = 0;
        if (fread(&v, sizeof(v), 1, file) != 1) {
          return false;
        }
        tmp = static_cast<int64_t>(v != 0);
        break;
      }
      default:
        return false;
      }
      if (tmp < 0 || tmp > std::numeric_limits<int>::max()) {
        return false;
      }
      *out = static_cast<int>(tmp);
      return true;
    };
    const auto read_float_value = [&](float *out) -> bool {
      if (!out) {
        return false;
      }
      if (type == GGUF::ValueType::FLOAT32) {
        return fread(out, sizeof(float), 1, file) == 1;
      }
      if (type == GGUF::ValueType::FLOAT64) {
        double v = 0.0;
        if (fread(&v, sizeof(v), 1, file) != 1) {
          return false;
        }
        *out = static_cast<float>(v);
        return true;
      }
      return false;
    };

    // Parse a minimal subset of architecture metadata required by native
    // pipeline initialization (KV-cache shape, heads, vocab).
    if (key.find(".block_count") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.num_hidden_layers = v;
      continue;
    }
    if (key.find(".embedding_length") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.hidden_size = v;
      continue;
    }
    if (key.find(".feed_forward_length") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.intermediate_size = v;
      continue;
    }
    if (key.find(".attention.head_count_kv") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.num_key_value_heads = v;
      continue;
    }
    if (key.find(".attention.head_count") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.num_attention_heads = v;
      continue;
    }
    if (key.find(".context_length") != std::string::npos) {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      model_info_.max_position_embeddings = v;
      continue;
    }
    if (key.find(".rope.freq_base") != std::string::npos) {
      float v = 0.0f;
      if (!read_float_value(&v)) {
        return false;
      }
      model_info_.rope_freq_base = v;
      continue;
    }
    // rope.scaling.factor → inverse gives freq_scale (matches llama.cpp)
    if (key.find(".rope.scaling.factor") != std::string::npos ||
        key.find(".rope.scale_linear") != std::string::npos) {
      float v = 0.0f;
      if (!read_float_value(&v)) {
        return false;
      }
      if (v > 0.0f) {
        model_info_.rope_freq_scale = 1.0f / v;
      }
      continue;
    }
    if (key.find(".attention.layer_norm_rms_epsilon") != std::string::npos) {
      float v = 0.0f;
      if (!read_float_value(&v)) {
        return false;
      }
      model_info_.rms_norm_eps = v;
      continue;
    }
    if (key == "tokenizer.ggml.eos_token_id") {
      int v = -1;
      if (!read_int_value(&v)) {
        return false;
      }
      tokenizer_eos_token_id_ = v;
      continue;
    }
    if (key == "tokenizer.ggml.bos_token_id") {
      int v = -1;
      if (!read_int_value(&v)) {
        return false;
      }
      tokenizer_bos_token_id_ = v;
      continue;
    }
    if (key == "tokenizer.ggml.add_bos_token") {
      int v = 0;
      if (!read_int_value(&v)) {
        return false;
      }
      tokenizer_add_bos_token_ = (v != 0);
      continue;
    }
    if (key == "tokenizer.ggml.pre") {
      if (type != GGUF::ValueType::STRING) {
        return false;
      }
      if (!GGUFReader::ReadString(file, &tokenizer_pre_)) {
        return false;
      }
      continue;
    }
    if (key == "tokenizer.chat_template") {
      if (type != GGUF::ValueType::STRING) {
        return false;
      }
      if (!GGUFReader::ReadString(file, &tokenizer_chat_template_)) {
        return false;
      }
      continue;
    }
    if (key == "tokenizer.ggml.tokens" && type == GGUF::ValueType::ARRAY) {
      uint32_t elem_type_raw = 0;
      uint64_t arr_len = 0;
      if (!ReadU32(file, &elem_type_raw) ||
          !GGUFReader::ReadUint64(file, &arr_len)) {
        return false;
      }
      if (static_cast<GGUF::ValueType>(elem_type_raw) !=
          GGUF::ValueType::STRING) {
        // Unexpected tokenizer type; still skip safely.
        for (uint64_t idx = 0; idx < arr_len; ++idx) {
          if (!GGUFReader::SkipValue(
                  file, static_cast<GGUF::ValueType>(elem_type_raw))) {
            return false;
          }
        }
        continue;
      }
      tokenizer_pieces_.reserve(static_cast<size_t>(std::min<uint64_t>(
          arr_len, static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
      for (uint64_t idx = 0; idx < arr_len; ++idx) {
        std::string piece;
        if (!GGUFReader::ReadString(file, &piece)) {
          return false;
        }
        tokenizer_pieces_.push_back(std::move(piece));
      }
      model_info_.vocab_size = static_cast<int>(std::min<uint64_t>(
          arr_len, static_cast<uint64_t>(std::numeric_limits<int>::max())));
      continue;
    }
    if (key == "tokenizer.ggml.merges" && type == GGUF::ValueType::ARRAY) {
      uint32_t elem_type_raw = 0;
      uint64_t arr_len = 0;
      if (!ReadU32(file, &elem_type_raw) ||
          !GGUFReader::ReadUint64(file, &arr_len)) {
        return false;
      }
      if (static_cast<GGUF::ValueType>(elem_type_raw) !=
          GGUF::ValueType::STRING) {
        for (uint64_t idx = 0; idx < arr_len; ++idx) {
          if (!GGUFReader::SkipValue(
                  file, static_cast<GGUF::ValueType>(elem_type_raw))) {
            return false;
          }
        }
        continue;
      }
      tokenizer_merges_.reserve(static_cast<size_t>(std::min<uint64_t>(
          arr_len, static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
      for (uint64_t idx = 0; idx < arr_len; ++idx) {
        std::string merge;
        if (!GGUFReader::ReadString(file, &merge)) {
          return false;
        }
        tokenizer_merges_.push_back(std::move(merge));
      }
      continue;
    }

    if (!GGUFReader::SkipValue(file, type)) {
      log::Error("gguf_loader", "Failed to skip value for key: " + key);
      return false;
    }
  }

  if (model_info_.num_key_value_heads <= 0 &&
      model_info_.num_attention_heads > 0) {
    model_info_.num_key_value_heads = model_info_.num_attention_heads;
  }
  if (model_info_.head_dim <= 0 && model_info_.hidden_size > 0 &&
      model_info_.num_attention_heads > 0) {
    model_info_.head_dim =
        model_info_.hidden_size / model_info_.num_attention_heads;
    model_info_.rope_dim = model_info_.head_dim;
  }
  if (model_info_.torch_dtype.empty()) {
    model_info_.torch_dtype = "float16";
  }
  if (model_info_.rms_norm_eps <= 0.0f) {
    model_info_.rms_norm_eps = 1e-6f;
  }

  return true;
}

bool GGUFModelLoader::ParseTensorInfo(FILE *file) {
  log::Debug("gguf_loader", "Parsing " + std::to_string(header_.tensor_count) +
                                " tensor info entries");

  for (int64_t i = 0; i < header_.tensor_count; ++i) {
    GGUFTensorData tensor;
    uint64_t offset;

    // Read name
    if (!GGUFReader::ReadString(file, &tensor.info.name)) {
      log::Error("gguf_loader", "Failed to read tensor name");
      return false;
    }

    // Read number of dimensions
    uint32_t n_dims = 0;
    if (!ReadU32(file, &n_dims)) {
      log::Error("gguf_loader", "Failed to read n_dims");
      return false;
    }
    if (n_dims == 0) {
      log::Error("gguf_loader",
                 "Tensor has zero dimensions: " + tensor.info.name);
      return false;
    }
    constexpr uint32_t kMaxDims = 8;
    if (n_dims > kMaxDims) {
      log::Error("gguf_loader", "Tensor has unsupported dimension count: " +
                                    std::to_string(n_dims));
      return false;
    }

    // Read dimensions
    tensor.info.shape.resize(n_dims);
    for (uint32_t j = 0; j < n_dims; ++j) {
      int64_t dim = 0;
      if (!GGUFReader::ReadInt64(file, &dim)) {
        log::Error("gguf_loader", "Failed to read dimension");
        return false;
      }
      if (dim <= 0) {
        log::Error("gguf_loader", "Invalid tensor dimension " +
                                      std::to_string(dim) + " for " +
                                      tensor.info.name);
        return false;
      }
      tensor.info.shape[j] = static_cast<uint64_t>(dim);
    }

    // Read type
    uint32_t type_val = 0;
    if (!ReadU32(file, &type_val)) {
      log::Error("gguf_loader", "Failed to read tensor type");
      return false;
    }
    if (type_val > static_cast<uint32_t>(GGUF::TensorType::Q8_K)) {
      log::Error("gguf_loader",
                 "Unsupported tensor type id: " + std::to_string(type_val));
      return false;
    }
    tensor.info.type = static_cast<GGUF::TensorType>(type_val);

    // Read offset
    if (!ReadU64(file, &offset)) {
      log::Error("gguf_loader", "Failed to read tensor offset");
      return false;
    }
    if (offset > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      log::Error("gguf_loader", "Tensor offset exceeds platform limits");
      return false;
    }
    tensor.info.offset = static_cast<size_t>(offset);

    // Calculate byte size
    tensor.info.byte_size = CalcTensorSize(tensor.info.type, tensor.info.shape);
    if (tensor.info.byte_size == 0) {
      log::Error("gguf_loader", "Invalid tensor size for " + tensor.info.name +
                                    " (type=" + std::to_string(type_val) + ")");
      return false;
    }

    tensors_[tensor.info.name] = std::move(tensor);
  }

  return true;
}

bool GGUFModelLoader::LoadTensorData(FILE *file) {
  log::Info("gguf_loader", "Loading tensor data...");

  const long metadata_end = ftell(file);
  if (metadata_end < 0) {
    log::Error("gguf_loader", "Failed to determine metadata end position");
    return false;
  }
  data_section_offset_ = AlignUp(static_cast<size_t>(metadata_end), alignment_);

  for (auto &[name, tensor] : tensors_) {
    // Seek to tensor data
    if (tensor.info.offset >
        std::numeric_limits<size_t>::max() - data_section_offset_) {
      log::Error("gguf_loader", "Tensor data offset overflow: " + name);
      return false;
    }
    const size_t absolute_offset = data_section_offset_ + tensor.info.offset;
    long seek_offset = 0;
    if (!SizeToLong(absolute_offset, &seek_offset)) {
      log::Error("gguf_loader", "Tensor offset exceeds seek range: " + name);
      return false;
    }
    if (fseek(file, seek_offset, SEEK_SET) != 0) {
      log::Error("gguf_loader", "Failed to seek to tensor: " + name);
      return false;
    }

    // Allocate buffer
    tensor.cpu_data.resize(tensor.info.byte_size);

    // Read data
    if (fread(tensor.cpu_data.data(), 1, tensor.info.byte_size, file) !=
        tensor.info.byte_size) {
      log::Error("gguf_loader", "Failed to read tensor data: " + name);
      return false;
    }
  }

  size_t total_size = 0;
  for (const auto &[name, tensor] : tensors_) {
    total_size += tensor.info.byte_size;
  }

  log::Info("gguf_loader", "Loaded " + std::to_string(tensors_.size()) +
                               " tensors, " +
                               std::to_string(total_size / kMiB) +
                               " MB (data_section_offset=" +
                               std::to_string(data_section_offset_) + ")");

  return true;
}

bool GGUFModelLoader::ExtractModelInfo() {
  if (model_info_.model_type.empty()) {
    model_info_.model_type = model_type_.empty() ? "unknown" : model_type_;
  }
  if (model_info_.torch_dtype.empty()) {
    model_info_.torch_dtype = "float16"; // GGUF typically stores fp16/f32
  }
  if (model_info_.rms_norm_eps <= 0.0f) {
    model_info_.rms_norm_eps = 1e-6f;
  }

  // Count layers from tensor names if metadata omitted it.
  if (model_info_.num_hidden_layers <= 0) {
    for (const auto &[name, tensor] : tensors_) {
      auto parts = GGUFReader::ParseTensorName(name);
      if (parts.layer >= 0 && parts.layer + 1 > model_info_.num_hidden_layers) {
        model_info_.num_hidden_layers = parts.layer + 1;
      }
    }
  }

  // Get vocab size from token embedding
  auto tok_emb_it = tensors_.find("tok_emb.weight");
  if (tok_emb_it == tensors_.end()) {
    tok_emb_it = tensors_.find("token_embd.weight");
  }
  if (tok_emb_it != tensors_.end()) {
    if (tok_emb_it->second.info.shape.size() >= 2) {
      const uint64_t d0 = tok_emb_it->second.info.shape[0];
      const uint64_t d1 = tok_emb_it->second.info.shape[1];
      model_info_.vocab_size =
          SaturatingToInt(static_cast<size_t>(std::min<uint64_t>(
              std::max<uint64_t>(d0, d1),
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
      if (model_info_.hidden_size <= 0) {
        model_info_.hidden_size =
            SaturatingToInt(static_cast<size_t>(std::min<uint64_t>(
                std::min<uint64_t>(d0, d1),
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
      }
    } else if (tok_emb_it->second.info.shape.size() >= 1 &&
               model_info_.vocab_size <= 0) {
      model_info_.vocab_size =
          SaturatingToInt(static_cast<size_t>(std::min<uint64_t>(
              tok_emb_it->second.info.shape[0],
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
    }
  }
  if (model_info_.vocab_size <= 0 && !tokenizer_pieces_.empty()) {
    model_info_.vocab_size = static_cast<int>(
        std::min<size_t>(tokenizer_pieces_.size(),
                         static_cast<size_t>(std::numeric_limits<int>::max())));
  }

  // Get hidden size from first attention layer when still missing.
  if (model_info_.hidden_size <= 0) {
    for (const auto &[name, tensor] : tensors_) {
      if (name.find("attn_q.weight") != std::string::npos ||
          name.find("attn_q.bias") != std::string::npos) {
        if (tensor.info.shape.size() >= 2) {
          model_info_.hidden_size =
              SaturatingToInt(static_cast<size_t>(std::min<uint64_t>(
                  tensor.info.shape[1],
                  static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
          break;
        }
      }
    }
  }

  if (model_info_.num_key_value_heads <= 0 &&
      model_info_.num_attention_heads > 0) {
    model_info_.num_key_value_heads = model_info_.num_attention_heads;
  }
  if (model_info_.head_dim <= 0 && model_info_.hidden_size > 0 &&
      model_info_.num_attention_heads > 0) {
    model_info_.head_dim =
        model_info_.hidden_size / model_info_.num_attention_heads;
  }
  if (model_info_.rope_dim <= 0 && model_info_.head_dim > 0) {
    model_info_.rope_dim = model_info_.head_dim;
  }
  model_info_.rope_type = InferRopeType(model_info_.model_type);

  log::Info("gguf_loader",
            "Model info: type=" + model_info_.model_type +
                " layers=" + std::to_string(model_info_.num_hidden_layers) +
                " hidden=" + std::to_string(model_info_.hidden_size) +
                " vocab=" + std::to_string(model_info_.vocab_size));

  return true;
}

void GGUFModelLoader::BuildTensorNameMapping() {
  for (const auto &[gguf_name, tensor] : tensors_) {
    std::string internal_name =
        GGUFReader::MapTensorName(gguf_name, model_type_);
    gguf_to_internal_name_map_[gguf_name] = internal_name;
    internal_to_gguf_name_map_[internal_name] = gguf_name;
  }
}

bool GGUFModelLoader::UploadToGPU(cudaStream_t stream) {
  log::Info("gguf_loader", "Uploading quantized weights to GPU...");

  // Calculate buffer size
  quantized_buffer_size_ = CalcGPUBufferSize();

  // Allocate unified buffer
  if (!CheckCudaStatus(cudaMalloc(&d_quantized_buffer_, quantized_buffer_size_),
                       "gguf_loader", "cudaMalloc(quantized buffer)")) {
    d_quantized_buffer_ = nullptr;
    return false;
  }

  // Copy quantized weights
  size_t offset = 0;
  uint8_t *buffer_base = static_cast<uint8_t *>(d_quantized_buffer_);
  for (auto &[name, tensor] : tensors_) {
    size_t size = tensor.cpu_data.size();
    if (!CheckCudaStatus(cudaMemcpyAsync(buffer_base + offset,
                                         tensor.cpu_data.data(), size,
                                         cudaMemcpyHostToDevice, stream),
                         "gguf_loader", "cudaMemcpyAsync(" + name + ")")) {
      cudaFree(d_quantized_buffer_);
      d_quantized_buffer_ = nullptr;
      return false;
    }

    tensor.gpu_data = buffer_base + offset;
    tensor.gpu_offset = offset;
    offset += size;
  }

  if (!CheckCudaStatus(cudaStreamSynchronize(stream), "gguf_loader",
                       "cudaStreamSynchronize(upload)")) {
    cudaFree(d_quantized_buffer_);
    d_quantized_buffer_ = nullptr;
    return false;
  }

  log::Info("gguf_loader", "Upload complete: " +
                               std::to_string(quantized_buffer_size_ / kMiB) +
                               " MB");

  // Free CPU memory
  FreeCPUMemory();

  return true;
}

void GGUFModelLoader::FreeCPUMemory() {
  for (auto &[name, tensor] : tensors_) {
    tensor.ClearCPUMemory();
  }
}

void GGUFModelLoader::FreeGPUMemory() { FreeGPUMemoryImpl(); }

void GGUFModelLoader::FreeGPUMemoryImpl() {
  ClearDequantizedCache();
  if (d_quantized_buffer_) {
    CheckCudaStatus(cudaFree(d_quantized_buffer_), "gguf_loader",
                    "cudaFree(quantized buffer)");
    d_quantized_buffer_ = nullptr;
  }
  if (d_dequantized_buffer_) {
    CheckCudaStatus(cudaFree(d_dequantized_buffer_), "gguf_loader",
                    "cudaFree(dequantized buffer)");
    d_dequantized_buffer_ = nullptr;
  }
}

void *GGUFModelLoader::GetGPUBuffer() const { return d_quantized_buffer_; }

size_t GGUFModelLoader::GetGPUSize() const { return quantized_buffer_size_; }

void GGUFModelLoader::SetDequantizedCachePolicy(DequantizedCachePolicy policy) {
  dequantized_cache_policy_ = policy;
}

DequantizedCachePolicy GGUFModelLoader::GetDequantizedCachePolicy() const {
  return dequantized_cache_policy_;
}

void GGUFModelLoader::ClearDequantizedCache() {
  if (!has_dequantized_entries_) {
    return;
  }
  has_dequantized_entries_ = false;
  for (auto &[name, tensor] : tensors_) {
    if (tensor.dequantized_gpu) {
      // Only free request/batch scoped dequantized caches for projection-like
      // quantized tensors. Quantized token embeddings must stay resident even
      // in memory-first modes because every request reuses them and the native
      // path still reads them through the dequantized gather path.
      if (tensor.info.is_quantized() && !ShouldRetainDequantizedTensor(name)) {
        CheckCudaStatus(cudaFree(tensor.dequantized_gpu), "gguf_loader",
                        "cudaFree(tensor.dequantized_gpu:" + name + ")");
        tensor.dequantized_gpu = nullptr;
      } else {
        // Retained entries mean we still have dequantized data
        has_dequantized_entries_ = true;
      }
    }
  }
}

const ModelInfo &GGUFModelLoader::GetModelInfo() const { return model_info_; }

std::string GGUFModelLoader::GetFormat() const { return "gguf"; }

bool GGUFModelLoader::IsQuantized() const {
  return !quantization_type_.empty();
}

std::string GGUFModelLoader::GetQuantizationType() const {
  return quantization_type_;
}

std::shared_ptr<IWeightAccessor>
GGUFModelLoader::GetWeightAccessor(const std::string &tensor_name) {
  // Check cache first
  auto it = weight_accessor_cache_.find(tensor_name);
  if (it != weight_accessor_cache_.end()) {
    return it->second;
  }

  // Map internal name to GGUF name
  auto gguf_it = internal_to_gguf_name_map_.find(tensor_name);
  if (gguf_it == internal_to_gguf_name_map_.end()) {
    log::Warn("gguf_loader", "Tensor not found: " + tensor_name);
    return nullptr;
  }

  // Get tensor data
  auto tensor_it = tensors_.find(gguf_it->second);
  if (tensor_it == tensors_.end()) {
    log::Warn("gguf_loader", "GGUF tensor not found: " + gguf_it->second);
    return nullptr;
  }

  // Create accessor
  auto accessor = std::make_shared<GGUFWeightAccessor>(
      &tensor_it->second, &has_dequantized_entries_);

  weight_accessor_cache_[tensor_name] = accessor;
  return accessor;
}

const GGUFTensorData *
GGUFModelLoader::GetTensorByGGUFName(const std::string &gguf_name) const {
  auto it = tensors_.find(gguf_name);
  if (it != tensors_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<std::string> GGUFModelLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

size_t GGUFModelLoader::CalcGPUBufferSize() const {
  size_t total = 0;
  for (const auto &[name, tensor] : tensors_) {
    total += tensor.cpu_data.size();
  }
  return total;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
