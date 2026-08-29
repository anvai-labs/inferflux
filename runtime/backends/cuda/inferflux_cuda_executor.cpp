#include "runtime/backends/cuda/inferflux_cuda_executor.h"
#include "model/gguf_tokenizer.h"
#include "model/tokenizer_factory.h"
#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/kv_cache_planner.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"
#include "runtime/backends/cuda/native/quantized_weight_map_adapter.h"
#include "runtime/backends/cuda/native/safetensors_parser.h"
#include "runtime/string_utils.h"
#include "server/diagnostics/crash_handler.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/cuda_sync_trace.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/hybrid_kv_cache_gpu.h"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include "runtime/backends/cuda/native/model_forward.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/weight_map.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <thread>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#ifndef F_OK
#define F_OK 0
#endif
#define access _access
#define MAP_FAILED ((void *)-1)
inline void munmap(void *addr, size_t) { UnmapViewOfFile(addr); }
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

namespace {

constexpr std::size_t kKiB = 1024ULL;
constexpr std::size_t kMiB = kKiB * kKiB;
constexpr std::size_t kGiB = kMiB * kKiB;

// BFloat16 → Float16 conversion (CPU-side)
// BF16: 1 sign + 8 exponent + 7 mantissa
// FP16: 1 sign + 5 exponent + 10 mantissa
inline uint16_t bf16_to_fp16(uint16_t bf16) {
  uint16_t sign = (bf16 >> 15) & 0x1;
  int32_t exponent = (bf16 >> 7) & 0xFF; // 8-bit BF16 exponent
  uint16_t mantissa = bf16 & 0x7F;       // 7-bit BF16 mantissa

  // Handle special cases
  if (exponent == 0xFF) {
    // Inf or NaN
    if (mantissa == 0) {
      return (sign << 15) | 0x7C00; // FP16 inf
    }
    return (sign << 15) | 0x7E00; // FP16 NaN
  }

  if (exponent == 0) {
    // Zero or denormal → FP16 zero (BF16 denormals are too small for FP16)
    return (sign << 15);
  }

  // Convert exponent: BF16 bias = 127, FP16 bias = 15
  int32_t fp16_exp = exponent - 127 + 15;

  if (fp16_exp >= 0x1F) {
    // Overflow → FP16 infinity
    return (sign << 15) | 0x7C00;
  }

  if (fp16_exp <= 0) {
    // Underflow → FP16 denormal or zero
    if (fp16_exp < -10) {
      return (sign << 15); // Too small, zero
    }
    // Denormalize: shift mantissa right
    uint32_t full_mantissa =
        (1 << 10) | (mantissa << 3); // implicit 1 + shift 7→10
    int shift = 1 - fp16_exp;
    full_mantissa >>= shift;
    return (sign << 15) | (full_mantissa & 0x3FF);
  }

  // Normal case: shift 7-bit mantissa to 10-bit (pad with zeros)
  uint16_t fp16_mantissa = mantissa << 3;
  return (sign << 15) | (fp16_exp << 10) | fp16_mantissa;
}

void ConvertBF16ToFP16(const void *src, size_t num_elements, void *dst) {
  const uint16_t *in = static_cast<const uint16_t *>(src);
  uint16_t *out = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = bf16_to_fp16(in[i]);
  }
}

bool CheckedMulSize(std::size_t a, std::size_t b, std::size_t *out) {
  if (!out) {
    return false;
  }
  if (a != 0U && b > (std::numeric_limits<std::size_t>::max() / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

// Use inferflux::ToLower from string_utils.h (included via header chain).
// Local wrapper for backward compatibility with call sites.
std::string ToLowerAscii(const std::string &value) {
  return inferflux::ToLower(value);
}

bool DecodeMappingDebugEnabled(const inferflux::NativeExecutionPolicy &policy) {
  return policy.debug_decode_mapping;
}

bool ConsumeDecodeMappingBudget(
    const inferflux::NativeExecutionPolicy &policy) {
  static thread_local int last_limit = -1;
  static thread_local int remaining = 0;
  if (policy.debug_decode_mapping_limit != last_limit) {
    last_limit = policy.debug_decode_mapping_limit;
    remaining = policy.debug_decode_mapping_limit;
  }
  if (remaining <= 0) {
    return false;
  }
  --remaining;
  return true;
}

bool NativeLogitsDebugEnabled(const inferflux::NativeExecutionPolicy &policy) {
  return policy.debug_logits;
}

bool ConsumeNativeLogitsBudget(const inferflux::NativeExecutionPolicy &policy) {
  static thread_local int last_limit = -1;
  static thread_local int remaining = 0;
  if (policy.debug_logits_limit != last_limit) {
    last_limit = policy.debug_logits_limit;
    remaining = policy.debug_logits_limit;
  }
  if (remaining <= 0) {
    return false;
  }
  --remaining;
  return true;
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
void LogNativeTopLogits(std::string_view stage, const float *d_logits_row,
                        int vocab_size, int64_t request_id,
                        std::string_view client_request_id, int sequence_id,
                        uint64_t sequence_generation, int n_past,
                        const inferflux::NativeExecutionPolicy &policy) {
  if (!NativeLogitsDebugEnabled(policy) || !ConsumeNativeLogitsBudget(policy) ||
      !d_logits_row || vocab_size <= 0) {
    return;
  }

  std::vector<float> h_logits(static_cast<std::size_t>(vocab_size));
  if (cudaMemcpy(h_logits.data(), d_logits_row, sizeof(float) * vocab_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return;
  }

  constexpr std::size_t kTopK = 5;
  std::vector<std::pair<float, int>> scored(
      static_cast<std::size_t>(vocab_size));
  for (int i = 0; i < vocab_size; ++i) {
    scored[static_cast<std::size_t>(i)] = {
        h_logits[static_cast<std::size_t>(i)], i};
  }
  const std::size_t top_n = std::min<std::size_t>(kTopK, scored.size());
  std::partial_sort(
      scored.begin(), scored.begin() + top_n, scored.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::string payload;
  for (std::size_t i = 0; i < top_n; ++i) {
    if (!payload.empty()) {
      payload += " ";
    }
    payload += "[" + std::to_string(scored[i].second) +
               "]=" + std::to_string(scored[i].first);
  }

  const std::string client_prefix =
      client_request_id.empty()
          ? std::string()
          : "client_request_id=" + std::string(client_request_id) + ", ";
  inferflux::log::Info(
      "inferflux_cuda_executor",
      "debug_logits[" + std::string(stage) + "]: " + client_prefix +
          "request_id=" + std::to_string(request_id) +
          ", sequence_id=" + std::to_string(sequence_id) +
          ", sequence_generation=" + std::to_string(sequence_generation) +
          ", n_past=" + std::to_string(n_past) + " top-5: " + payload);
}
#endif

void LogSampleMapping(std::string_view stage, int input_idx, int64_t request_id,
                      std::string_view client_request_id, int sequence_id,
                      uint64_t sequence_generation, int n_past, int token_count,
                      int token_id, const inferflux::ITokenizer *tokenizer,
                      const inferflux::NativeExecutionPolicy &policy) {
  if (!DecodeMappingDebugEnabled(policy) ||
      !ConsumeDecodeMappingBudget(policy)) {
    return;
  }
  const std::string piece = (tokenizer && token_id >= 0)
                                ? tokenizer->TokenToString(token_id)
                                : std::string();
  inferflux::log::Info(
      "inferflux_cuda_executor",
      std::string(stage) + ": input_idx=" + std::to_string(input_idx) + ", " +
          (client_request_id.empty()
               ? std::string()
               : "client_request_id=" + std::string(client_request_id) + ", ") +
          "request_id=" + std::to_string(request_id) +
          ", sequence_id=" + std::to_string(sequence_id) +
          ", sequence_generation=" + std::to_string(sequence_generation) +
          ", n_past=" + std::to_string(n_past) +
          ", token_count=" + std::to_string(token_count) +
          ", sampled_token=" + std::to_string(token_id) + ", piece=" + piece);
}

std::string VisibleTokenPiece(const inferflux::ITokenizer *tokenizer,
                              int token_id) {
  if (!tokenizer || token_id < 0) {
    return {};
  }
  if (tokenizer->IsTerminalGeneratedToken(token_id)) {
    return {};
  }
  return tokenizer->TokenToString(token_id);
}

std::string
MatmulModeToString(inferflux::runtime::cuda::native::MatmulExecutionMode mode) {
  switch (mode) {
  case inferflux::runtime::cuda::native::MatmulExecutionMode::
      kFusedDequantTileGemm:
    return "fused_dequant_tile_gemm";
  case inferflux::runtime::cuda::native::MatmulExecutionMode::
      kCompatDequantizeThenGemm:
    return "compat_dequantize_then_gemm";
  }
  return "unknown";
}

bool IsKnownQuantizationType(const std::string &quantization_type) {
  const std::string lowered = ToLowerAscii(quantization_type);
  return lowered == "q2_k" || lowered == "q3_k" || lowered == "q4_0" ||
         lowered == "q4_1" || lowered == "q4_k" || lowered == "q4_k_m" ||
         lowered == "q5_0" || lowered == "q5_1" || lowered == "q5_k" ||
         lowered == "q5_k_m" || lowered == "q6_k" || lowered == "q8_0" ||
         lowered == "q8_1" || lowered == "q8_k";
}

inferflux::SafetensorsLoader::ModelConfig
ConvertModelInfo(const inferflux::runtime::cuda::native::ModelInfo &info) {
  inferflux::SafetensorsLoader::ModelConfig cfg;
  cfg.hidden_size = info.hidden_size;
  cfg.num_hidden_layers = info.num_hidden_layers;
  cfg.num_attention_heads = info.num_attention_heads;
  cfg.num_key_value_heads = info.num_key_value_heads;
  cfg.head_dim = info.head_dim;
  if (cfg.head_dim == 0 && cfg.num_attention_heads > 0 && cfg.hidden_size > 0) {
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
  }
  cfg.intermediate_size = info.intermediate_size;
  cfg.vocab_size = info.vocab_size;
  cfg.max_position_embeddings = info.max_position_embeddings;
  cfg.rope_freq_base = info.rope_freq_base;
  cfg.rope_freq_scale = info.rope_freq_scale;
  cfg.rope_dim = info.rope_dim;
  cfg.model_type = info.model_type;
  cfg.activation = info.activation;
  cfg.torch_dtype = info.torch_dtype;
  cfg.rms_norm_eps = info.rms_norm_eps;
  return cfg;
}

void WarmQuantizedLaneCache(inferflux::QuantizedWeightMap *weight_map) {
  if (!weight_map) {
    return;
  }

  // Warm only small permanent-cache tensors up front so overlap lanes don't
  // race first-touch dequantized cache creation in the shared GGUF loader.
  // Keep lm_head lazy to avoid eager large allocations in memory-first mode.
  (void)weight_map->EmbedTokens();
  (void)weight_map->FinalNorm();
  const int layers = weight_map->NumLayers();
  for (int layer = 0; layer < layers; ++layer) {
    (void)weight_map->LayerInputNorm(layer);
    (void)weight_map->LayerPostAttnNorm(layer);
    (void)weight_map->LayerQProjBias(layer);
    (void)weight_map->LayerKProjBias(layer);
    (void)weight_map->LayerVProjBias(layer);
  }
}

} // namespace

namespace inferflux {

#ifdef INFERFLUX_HAS_CUDA

using json = nlohmann::json;

//==============================================================================
// SafetensorsLoader Implementation
//==============================================================================

SafetensorsLoader::SafetensorsLoader() = default;

bool SafetensorsLoader::LoadModel(const std::string &model_path) {
  model_path_ = model_path;

  log::Info("safetensors_loader", "Loading model from: " + model_path);

  // Step 1: Parse config.json
  std::string config_path = model_path + "/config.json";
  if (!ParseConfig(config_path)) {
    log::Warn("safetensors_loader",
              "Failed to load config.json, using defaults");
  }

  // Step 2: Load and parse safetensors shard files
  std::vector<std::string> shard_files;

  std::string index_path = model_path + "/model.safetensors.index.json";

  if (access(index_path.c_str(), F_OK) == 0) {
    log::Info("safetensors_loader", "Loading from index file");

    try {
      std::ifstream f(index_path);
      json index = json::parse(f);

      std::set<std::string> unique_shards;
      if (index.contains("weight_map")) {
        auto weight_map = index["weight_map"];
        for (auto &[name, shard] : weight_map.items()) {
          unique_shards.insert(shard.get<std::string>());
        }
      }

      for (const auto &shard : unique_shards) {
        shard_files.push_back(model_path + "/" + shard);
      }

    } catch (const std::exception &e) {
      log::Error("safetensors_loader",
                 "Failed to parse index: " + std::string(e.what()));
      return false;
    }
  } else {
    // Try single model.safetensors file
    std::string single_path = model_path + "/model.safetensors";
    if (access(single_path.c_str(), F_OK) == 0) {
      shard_files.push_back(single_path);
    } else {
      log::Error("safetensors_loader",
                 "No safetensors files found in " + model_path);
      return false;
    }
  }

  log::Info("safetensors_loader",
            "Found " + std::to_string(shard_files.size()) + " shard files");

  // Step 3: Parse each shard file
  for (const auto &shard_path : shard_files) {
    log::Info("safetensors_loader", "Parsing shard: " + shard_path);

    auto parser = std::make_unique<SafetensorsParser>(shard_path);
    if (!parser->Parse()) {
      log::Error("safetensors_loader", "Failed to parse shard: " + shard_path);
      return false;
    }

    auto tensor_names = parser->GetTensorNames();
    for (const auto &tensor_name : tensor_names) {
      const auto *tensor_info = parser->GetTensor(tensor_name);
      if (tensor_info && tensors_.find(tensor_name) == tensors_.end()) {
        Tensor tensor;
        tensor.name = tensor_info->name;
        tensor.shape = tensor_info->shape;
        tensor.dtype = tensor_info->dtype;
        tensor.offset = tensor_info->offset;
        tensor.size = tensor_info->byte_size;
        tensor.cpu_data = tensor_info->data_ptr;
        tensor.gpu_data = nullptr;

        tensors_[tensor_name] = std::move(tensor);
      }
    }

    shard_parsers_.push_back(std::move(parser));
  }

  log::Info("safetensors_loader",
            "Model loaded successfully: " + std::to_string(tensors_.size()) +
                " tensors");

  // Log key tensors
  std::vector<std::string> key_tensors = {
      "model.embed_tokens.weight",
      "model.layers.0.input_layernorm.weight",
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.weight",
      "model.layers.0.mlp.gate_proj.weight",
      "model.layers.0.mlp.down_proj.weight",
      "lm_head.weight"};

  for (const auto &tensor : key_tensors) {
    const auto *info = GetTensor(tensor);
    if (info) {
      std::string shape_str = "[";
      for (size_t i = 0; i < info->shape.size(); i++) {
        if (i > 0)
          shape_str += ", ";
        shape_str += std::to_string(info->shape[i]);
      }
      shape_str += "]";
      log::Info("safetensors_loader",
                "  Tensor: " + tensor + " dtype=" + info->dtype +
                    " shape=" + shape_str +
                    " size=" + std::to_string(info->size / kMiB) + " MB");
    }
  }

  return true;
}

bool SafetensorsLoader::LoadIndex(const std::string &index_path) {
  try {
    std::ifstream f(index_path);
    if (!f.is_open()) {
      log::Error("safetensors_loader", "Cannot open index: " + index_path);
      return false;
    }

    json index = json::parse(f);

    if (!index.contains("weight_map")) {
      log::Error("safetensors_loader", "Index missing weight_map");
      return false;
    }

    auto weight_map = index["weight_map"];

    std::unordered_map<std::string, std::vector<std::string>> shard_tensors;
    for (auto &[tensor_name, shard_file] : weight_map.items()) {
      shard_tensors[shard_file.get<std::string>()].push_back(tensor_name);
    }

    log::Info("safetensors_loader",
              "Found " + std::to_string(shard_tensors.size()) + " shard files");

    for (auto &[shard_file, tensor_names] : shard_tensors) {
      std::string full_path = model_path_ + "/" + shard_file;

      struct stat st;
      if (stat(full_path.c_str(), &st) != 0) {
        log::Warn("safetensors_loader", "Cannot stat shard: " + full_path);
        continue;
      }

      for (const auto &tensor_name : tensor_names) {
        Tensor tensor;
        tensor.name = tensor_name;
        tensor.dtype = "";
        tensor.offset = 0;
        tensor.size = 0;
        tensor.cpu_data = nullptr;
        tensor.gpu_data = nullptr;

        tensors_[tensor_name] = std::move(tensor);
      }

      log::Info("safetensors_loader",
                "Shard " + shard_file + ": " +
                    std::to_string(tensor_names.size()) + " tensors, " +
                    std::to_string(st.st_size / static_cast<off_t>(kGiB)) +
                    " GB");
    }

    return true;

  } catch (const std::exception &e) {
    log::Error("safetensors_loader",
               "JSON parse error: " + std::string(e.what()));
    return false;
  }
}

bool SafetensorsLoader::ParseConfig(const std::string &config_path) {
  try {
    std::ifstream f(config_path);
    if (!f.is_open()) {
      return false;
    }

    json config = json::parse(f);

    if (config.contains("hidden_size")) {
      config_.hidden_size = config["hidden_size"];
    }
    if (config.contains("num_hidden_layers")) {
      config_.num_hidden_layers = config["num_hidden_layers"];
    }
    if (config.contains("num_attention_heads")) {
      config_.num_attention_heads = config["num_attention_heads"];
    }
    if (config.contains("num_key_value_heads")) {
      config_.num_key_value_heads = config["num_key_value_heads"];
    }
    if (config.contains("head_dim")) {
      config_.head_dim = config["head_dim"];
    } else {
      config_.head_dim = config_.hidden_size / config_.num_attention_heads;
    }
    if (config.contains("intermediate_size")) {
      config_.intermediate_size = config["intermediate_size"];
    }
    if (config.contains("vocab_size")) {
      config_.vocab_size = config["vocab_size"];
    }
    if (config.contains("max_position_embeddings")) {
      config_.max_position_embeddings = config["max_position_embeddings"];
    }

    // RoPE settings
    if (config.contains("rope_theta")) {
      config_.rope_freq_base = config["rope_theta"];
    }

    // Model type
    if (config.contains("model_type")) {
      config_.model_type = config["model_type"];
    } else if (config.contains("architectures")) {
      auto archs = config["architectures"];
      if (archs.is_array() && archs.size() > 0) {
        config_.model_type = archs[0];
      }
    }

    // Activation
    if (config.contains("hidden_act")) {
      config_.activation = config["hidden_act"];
    }

    // Dtype
    if (config.contains("torch_dtype")) {
      config_.torch_dtype = config["torch_dtype"].get<std::string>();
    }

    // RMS norm epsilon
    if (config.contains("rms_norm_eps")) {
      config_.rms_norm_eps = config["rms_norm_eps"].get<float>();
    }

    log::Info("safetensors_loader",
              "Model config: " + config_.model_type +
                  ", hidden_size=" + std::to_string(config_.hidden_size) +
                  ", num_layers=" + std::to_string(config_.num_hidden_layers) +
                  ", num_heads=" + std::to_string(config_.num_attention_heads) +
                  ", head_dim=" + std::to_string(config_.head_dim));

    return true;

  } catch (const std::exception &e) {
    log::Error("safetensors_loader",
               "Config parse error: " + std::string(e.what()));
    return false;
  }
}

const SafetensorsLoader::Tensor *
SafetensorsLoader::GetTensor(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::string> SafetensorsLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

bool SafetensorsLoader::UploadToGPU(cudaStream_t stream,
                                    bool skip_bf16_conversion) {
  log::Info("safetensors_loader",
            "Uploading " + std::to_string(tensors_.size()) + " tensors to GPU");

  size_t total_size = 0;
  for (const auto &[name, tensor] : tensors_) {
    total_size += tensor.size;
  }

  log::Info("safetensors_loader",
            "Total GPU memory needed: " + std::to_string(total_size / kGiB) +
                " GB");

  cudaError_t err = cudaMalloc(&d_weights_buffer_, total_size);
  if (err != cudaSuccess) {
    log::Error("safetensors_loader",
               "cudaMalloc failed: " + std::string(cudaGetErrorString(err)) +
                   " (size=" + std::to_string(total_size) + " bytes)");
    return false;
  }

  log::Info("safetensors_loader",
            "Allocated GPU buffer at " +
                std::to_string(reinterpret_cast<uintptr_t>(d_weights_buffer_)));

  size_t offset = 0;
  size_t uploaded = 0;
  size_t bf16_converted = 0;
  std::vector<uint16_t> convert_buf; // Reusable buffer for BF16→FP16

  for (auto &[name, tensor] : tensors_) {
    if (!tensor.cpu_data) {
      log::Warn("safetensors_loader",
                "Tensor " + name + " has no CPU data, skipping");
      tensor.gpu_data = nullptr;
      continue;
    }

    const void *upload_data = tensor.cpu_data;
    bool needs_conversion = (tensor.dtype == "BF16") && !skip_bf16_conversion;

    if (needs_conversion) {
      // BF16→FP16 conversion on CPU before upload
      size_t num_elements = tensor.size / sizeof(uint16_t);
      if (convert_buf.size() < num_elements) {
        convert_buf.resize(num_elements);
      }
      ConvertBF16ToFP16(tensor.cpu_data, num_elements, convert_buf.data());
      upload_data = convert_buf.data();
      bf16_converted++;
    }

    err = cudaMemcpyAsync(static_cast<uint8_t *>(d_weights_buffer_) + offset,
                          upload_data, tensor.size, cudaMemcpyHostToDevice,
                          stream);

    if (err != cudaSuccess) {
      log::Error("safetensors_loader",
                 "cudaMemcpyAsync failed for " + name + ": " +
                     std::string(cudaGetErrorString(err)));
      cudaFree(d_weights_buffer_);
      d_weights_buffer_ = nullptr;
      return false;
    }

    tensor.gpu_data = static_cast<uint8_t *>(d_weights_buffer_) + offset;
    tensor.gpu_offset = offset;
    offset += tensor.size;
    uploaded++;

    if (uploaded <= 10 || uploaded % 50 == 0) {
      log::Info("safetensors_loader",
                "  Uploaded " + name + " (" +
                    std::to_string(tensor.size / kMiB) + " MB) -> GPU offset " +
                    std::to_string(tensor.gpu_offset) +
                    (needs_conversion ? " [BF16→FP16]" : ""));
    }
  }

  if (bf16_converted > 0) {
    log::Info("safetensors_loader", "Converted " +
                                        std::to_string(bf16_converted) +
                                        " BF16 tensors to FP16");
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    log::Error("safetensors_loader", "cudaStreamSynchronize failed: " +
                                         std::string(cudaGetErrorString(err)));
    cudaFree(d_weights_buffer_);
    d_weights_buffer_ = nullptr;
    return false;
  }

  log::Info("safetensors_loader",
            "Successfully uploaded " + std::to_string(uploaded) +
                " tensors to GPU (" + std::to_string(offset) + " bytes)");

  total_gpu_size_ = offset;

  // Free CPU memory after GPU upload
  log::Info("safetensors_loader", "Freeing CPU memory (keeping GPU memory)");
  for (auto &[name, tensor] : tensors_) {
    tensor.cpu_data = nullptr;
  }
  shard_parsers_.clear();

  return true;
}

void SafetensorsLoader::FreeCPUMemory() {
  log::Info("safetensors_loader", "Freeing CPU memory for " +
                                      std::to_string(tensors_.size()) +
                                      " tensors");

  for (auto &[name, tensor] : tensors_) {
    if (tensor.cpu_data && tensor.cpu_data != MAP_FAILED) {
      munmap(tensor.cpu_data, tensor.size);
      tensor.cpu_data = nullptr;
    }
  }

  log::Info("safetensors_loader", "CPU memory freed");
}

void SafetensorsLoader::FreeGPUMemory() {
  if (d_weights_buffer_) {
    log::Info("safetensors_loader", "Freeing GPU buffer (" +
                                        std::to_string(total_gpu_size_ / kMiB) +
                                        " MB)");
    cudaFree(d_weights_buffer_);
    d_weights_buffer_ = nullptr;
    total_gpu_size_ = 0;

    for (auto &[name, tensor] : tensors_) {
      tensor.gpu_data = nullptr;
      tensor.gpu_offset = 0;
    }
  }
}

SafetensorsLoader::~SafetensorsLoader() {
  FreeCPUMemory();
  FreeGPUMemory();
}

#else // !INFERFLUX_HAS_CUDA

// CPU-only stubs for SafetensorsLoader
SafetensorsLoader::SafetensorsLoader() = default;
SafetensorsLoader::~SafetensorsLoader() = default;
bool SafetensorsLoader::LoadModel(const std::string &) { return false; }
const SafetensorsLoader::Tensor *
SafetensorsLoader::GetTensor(const std::string &) const {
  return nullptr;
}
std::vector<std::string> SafetensorsLoader::GetTensorNames() const {
  return {};
}
bool SafetensorsLoader::UploadToGPU(cudaStream_t, bool) { return false; }
void SafetensorsLoader::FreeCPUMemory() {}
void SafetensorsLoader::FreeGPUMemory() {}
bool SafetensorsLoader::LoadIndex(const std::string &) { return false; }
bool SafetensorsLoader::LoadShard(const std::string &) { return false; }
bool SafetensorsLoader::ParseConfig(const std::string &) { return false; }

#endif // INFERFLUX_HAS_CUDA

//==============================================================================
// InferfluxCudaExecutor Implementation
//==============================================================================

InferfluxCudaExecutor::InferfluxCudaExecutor() = default;

InferfluxCudaExecutor::~InferfluxCudaExecutor() {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  std::vector<PendingSequenceRelease> pending_sequence_releases;
  {
    std::lock_guard<std::mutex> lock(pending_sequence_releases_mutex_);
    pending_sequence_releases.swap(pending_sequence_releases_);
  }
  for (auto &pending : pending_sequence_releases) {
    if (pending.compute_done) {
      cudaEventDestroy(pending.compute_done);
    }
    if (pending.decode_done) {
      cudaEventDestroy(pending.decode_done);
    }
    if (pending.prefill_done) {
      cudaEventDestroy(pending.prefill_done);
    }
  }
  StopLaneDispatcher();
  DestroyLaneOverlapResources();
  model_forward_.reset();
  sampler_.reset();
  kv_cache_.reset();
  gemm_.reset();
  weight_map_.reset();
  quantized_weight_adapter_.reset();
  quantized_weight_map_.reset();
  if (d_logits_) {
    if (cudaFree(d_logits_) != cudaSuccess) {
      log::Error("inferflux_cuda_executor", "cudaFree(d_logits_) failed");
    }
    d_logits_ = nullptr;
  }
  if (forward_start_)
    cudaEventDestroy(forward_start_);
  if (forward_stop_)
    cudaEventDestroy(forward_stop_);
  if (sampling_start_)
    cudaEventDestroy(sampling_start_);
  if (sampling_stop_)
    cudaEventDestroy(sampling_stop_);
  if (decode_stream_)
    cudaStreamDestroy(decode_stream_);
  if (prefill_stream_)
    cudaStreamDestroy(prefill_stream_);
#endif
  tokenizer_.reset();
  loader_.reset();
  memory_ledger_.Clear();
  active_max_batch_ = 0;
  active_max_seq_ = 0;
#ifdef INFERFLUX_HAS_CUDA
  FreeDeviceMemory();
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
  }
  if (copy_stream_) {
    cudaStreamDestroy(copy_stream_);
  }
#endif
}

#ifdef INFERFLUX_HAS_CUDA

namespace {

bool CheckCudaStatus(cudaError_t status, const std::string &operation) {
  if (status == cudaSuccess) {
    return true;
  }
  log::Error("inferflux_cuda_executor",
             operation + " failed: " + cudaGetErrorString(status));
  return false;
}

bool CheckBF16Support() {
  int device = 0;
  if (!CheckCudaStatus(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp prop;
  if (!CheckCudaStatus(cudaGetDeviceProperties(&prop, device),
                       "cudaGetDeviceProperties")) {
    return false;
  }
  return prop.major >= 8;
}

bool WaitForLaneHandle(
    UnifiedBatchLaneDispatcher *dispatcher, UnifiedBatchHandle handle,
    std::vector<LlamaCppBackend::UnifiedBatchOutput> *outputs,
    bool *decode_lane, std::string *error, double *elapsed_ms = nullptr,
    bool *timed_out = nullptr,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  if (!dispatcher || handle == 0) {
    if (timed_out) {
      *timed_out = false;
    }
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto status =
        dispatcher->TryCollect(handle, outputs, decode_lane, error, elapsed_ms);
    if (status == UnifiedBatchLaneDispatcher::CollectStatus::kSuccess) {
      if (timed_out) {
        *timed_out = false;
      }
      return true;
    }
    if (status == UnifiedBatchLaneDispatcher::CollectStatus::kFailed ||
        status == UnifiedBatchLaneDispatcher::CollectStatus::kMissing) {
      if (timed_out) {
        *timed_out = false;
      }
      return false;
    }
    std::this_thread::yield();
  }
  auto status =
      dispatcher->TryCollect(handle, outputs, decode_lane, error, elapsed_ms);
  if (timed_out) {
    *timed_out =
        (status == UnifiedBatchLaneDispatcher::CollectStatus::kPending);
  }
  return status == UnifiedBatchLaneDispatcher::CollectStatus::kSuccess;
}

} // namespace

bool InferfluxCudaExecutor::InitializeCUDA() {
  cudaError_t err = cudaGetDevice(&device_id_);
  if (err != cudaSuccess) {
    log::Error("inferflux_cuda_executor",
               "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
    return false;
  }

  err = cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    log::Error("inferflux_cuda_executor",
               "cudaStreamCreate failed: " +
                   std::string(cudaGetErrorString(err)));
    return false;
  }

  err = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    log::Error("inferflux_cuda_executor",
               "cudaStreamCreate failed: " +
                   std::string(cudaGetErrorString(err)));
    return false;
  }

  log::Info("inferflux_cuda_executor",
            "CUDA initialized on device " + std::to_string(device_id_));
  return true;
}

void InferfluxCudaExecutor::FreeDeviceMemory() {
  // GPU memory is managed by SafetensorsLoader and native components
}

void InferfluxCudaExecutor::MaybeRefreshMemoryLedger() {
  // Throttle memory ledger refresh to every 64 batches (~1/s at typical
  // decode throughput). The ledger rebuild is CPU-only but involves ~10
  // UpsertItem calls and a full metrics push, adding measurable overhead
  // when called on every single batch.
  if ((++batch_counter_ & 63) != 0) {
    return;
  }
  RefreshMemoryLedger();
}

void InferfluxCudaExecutor::RefreshMemoryLedger() {
  memory_ledger_.Clear();
  if (!loaded_model_path_.empty()) {
    memory_ledger_.SetModelLabel(loaded_model_path_.filename().string());
  }

  const std::size_t weights_bytes = model_loader_
                                        ? model_loader_->GetGPUSize()
                                        : (loader_ ? loader_->GetGPUSize() : 0);
  if (weights_bytes > 0) {
    memory_ledger_.UpsertItem(model_loader_ ? "weights.loader"
                                            : "weights.safetensors",
                              runtime::cuda::native::MemoryDomain::kWeights,
                              runtime::cuda::native::MemoryLifetime::kModel,
                              weights_bytes, weights_bytes);
  }

  auto record_qwm_scratch = [&](const char *label,
                                const QuantizedWeightMap *map) {
    if (!map) {
      return;
    }
    const std::size_t reserved_bytes = map->ScratchReservedBytes();
    const std::size_t in_use_bytes = map->ScratchInUseBytes();
    const std::size_t high_water_bytes = map->ScratchHighWaterBytes();
    const std::size_t evictable_bytes = reserved_bytes;
    if (reserved_bytes == 0 && high_water_bytes == 0) {
      return;
    }
    memory_ledger_.UpsertItem(
        label, runtime::cuda::native::MemoryDomain::kBatchEphemeral,
        runtime::cuda::native::MemoryLifetime::kBatch, reserved_bytes,
        in_use_bytes, high_water_bytes, evictable_bytes);
  };

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (kv_cache_) {
    const std::size_t kv_bytes = kv_cache_->GetMemoryUsage();
    memory_ledger_.UpsertItem(
        "kv_cache.primary", runtime::cuda::native::MemoryDomain::kKvCache,
        runtime::cuda::native::MemoryLifetime::kPool, kv_bytes, kv_bytes);
  }

  auto record_forward = [&](const char *device_label, const char *host_label,
                            const ModelForward *forward) {
    if (!forward) {
      return;
    }
    const std::size_t device_bytes = forward->DeviceWorkspaceBytes();
    if (device_bytes > 0) {
      memory_ledger_.UpsertItem(
          device_label, runtime::cuda::native::MemoryDomain::kWorkspaceDevice,
          runtime::cuda::native::MemoryLifetime::kModel, device_bytes,
          device_bytes);
    }
    const std::size_t host_bytes = forward->HostWorkspaceBytes();
    if (host_bytes > 0) {
      memory_ledger_.UpsertItem(
          host_label, runtime::cuda::native::MemoryDomain::kWorkspaceHostPinned,
          runtime::cuda::native::MemoryLifetime::kModel, host_bytes,
          host_bytes);
    }
  };
  auto record_sampler = [&](const char *label, const GpuSampler *sampler) {
    if (!sampler) {
      return;
    }
    const std::size_t device_bytes = sampler->DeviceWorkspaceBytes();
    if (device_bytes > 0) {
      memory_ledger_.UpsertItem(
          label, runtime::cuda::native::MemoryDomain::kWorkspaceDevice,
          runtime::cuda::native::MemoryLifetime::kModel, device_bytes,
          device_bytes);
    }
  };

  record_forward("workspace.forward.primary", "workspace.forward.primary.host",
                 model_forward_.get());
  record_sampler("workspace.sampler.primary", sampler_.get());
  record_qwm_scratch("batch_ephemeral.quantized_scratch.primary",
                     quantized_weight_map_.get());

  if (d_logits_ && active_max_batch_ > 0 && model_config_.vocab_size > 0) {
    const std::size_t logits_bytes =
        static_cast<std::size_t>(active_max_batch_) *
        static_cast<std::size_t>(model_config_.vocab_size) * sizeof(float);
    memory_ledger_.UpsertItem(
        "workspace.logits.primary",
        runtime::cuda::native::MemoryDomain::kWorkspaceDevice,
        runtime::cuda::native::MemoryLifetime::kModel, logits_bytes,
        logits_bytes);
  }

  if (lane_overlap_ready_ && active_max_batch_ > 0 &&
      model_config_.vocab_size > 0) {
    const std::size_t lane_logits_bytes =
        static_cast<std::size_t>(active_max_batch_) *
        static_cast<std::size_t>(model_config_.vocab_size) * sizeof(float);
    record_forward("workspace.forward.decode", "workspace.forward.decode.host",
                   decode_lane_forward_.get());
    record_forward("workspace.forward.prefill",
                   "workspace.forward.prefill.host",
                   prefill_lane_forward_.get());
    record_sampler("workspace.sampler.decode", decode_lane_sampler_.get());
    record_sampler("workspace.sampler.prefill", prefill_lane_sampler_.get());
    record_qwm_scratch("batch_ephemeral.quantized_scratch.decode",
                       decode_lane_quantized_weight_map_.get());
    record_qwm_scratch("batch_ephemeral.quantized_scratch.prefill",
                       prefill_lane_quantized_weight_map_.get());
    if (d_decode_logits_) {
      memory_ledger_.UpsertItem(
          "workspace.logits.decode",
          runtime::cuda::native::MemoryDomain::kWorkspaceDevice,
          runtime::cuda::native::MemoryLifetime::kModel, lane_logits_bytes,
          lane_logits_bytes);
    }
    if (d_prefill_logits_) {
      memory_ledger_.UpsertItem(
          "workspace.logits.prefill",
          runtime::cuda::native::MemoryDomain::kWorkspaceDevice,
          runtime::cuda::native::MemoryLifetime::kModel, lane_logits_bytes,
          lane_logits_bytes);
    }
  }
#endif

  MetricsRegistry::MemoryUsageMetrics total;
  total.reserved_bytes = memory_ledger_.TotalReservedBytes();
  total.in_use_bytes = memory_ledger_.TotalInUseBytes();
  total.high_water_bytes = memory_ledger_.TotalHighWaterBytes();
  total.evictable_bytes = memory_ledger_.TotalEvictableBytes();
  std::unordered_map<std::string, MetricsRegistry::MemoryUsageMetrics> domains;
  for (const auto &summary : memory_ledger_.AggregateByDomain()) {
    MetricsRegistry::MemoryUsageMetrics usage;
    usage.reserved_bytes = summary.usage.reserved_bytes;
    usage.in_use_bytes = summary.usage.in_use_bytes;
    usage.high_water_bytes = summary.usage.high_water_bytes;
    usage.evictable_bytes = summary.usage.evictable_bytes;
    domains.emplace(runtime::cuda::native::ToString(summary.domain), usage);
  }
  GlobalMetrics().SetInferfluxCudaModelMemorySnapshot(
      memory_ledger_.ModelLabel(), total, std::move(domains));
}

bool InferfluxCudaExecutor::ConfigureDequantizedCachePolicy(
    const std::string &raw_policy) {
  std::string policy = ToLowerAscii(raw_policy);
  if (policy.empty()) {
    policy = "none";
  }

  runtime::cuda::native::DequantizedCachePolicy parsed =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  if (!runtime::cuda::native::ParseDequantizedCachePolicy(policy, &parsed)) {
    log::Warn("inferflux_cuda_executor", "Invalid dequantized cache policy '" +
                                             policy +
                                             "'; falling back to none");
    parsed = runtime::cuda::native::DequantizedCachePolicy::kNone;
    policy = "none";
  }
  dequantized_cache_policy_ = parsed;
  dequantized_cache_policy_hint_ = policy;

  if (model_loader_) {
    model_loader_->SetDequantizedCachePolicy(dequantized_cache_policy_);
  }
  return true;
}

void InferfluxCudaExecutor::ReleaseBatchScopedDequantizedCache() {
  if (dequantized_cache_policy_ ==
      runtime::cuda::native::DequantizedCachePolicy::kModelLifetime) {
    return;
  }
  if (!model_loader_ || model_loader_->GetFormat() != "gguf") {
    return;
  }

  // Acquire lane_overlap_mutex_ to ensure no lane threads are still using
  // quantized weight maps or scratch buffers when we clear/release them.
  std::lock_guard<std::mutex> lane_guard(lane_overlap_mutex_);

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Skip expensive stream synchronization when no dequant scratch is in use.
  // With MMVQ/MMQ as the primary dispatch path, dequant scratch is only
  // allocated for cuBLAS fallback (M>64 prefill) which is rare during decode.
  // The 3 stream syncs (~2-4ms) per batch were the root cause of c=1
  // throughput regression (81.7 → 50 tok/s).
  const bool has_scratch =
      quantized_weight_map_ && quantized_weight_map_->ScratchInUseBytes() > 0;
#else
  const bool has_scratch = true;
#endif

  if (has_scratch) {
    if (compute_stream_) {
      CheckCudaStatus(cudaStreamSynchronize(compute_stream_),
                      "cudaStreamSynchronize(compute_stream_,dequant_cleanup)");
    }
#ifdef INFERFLUX_NATIVE_KERNELS_READY
    if (decode_stream_) {
      CheckCudaStatus(cudaStreamSynchronize(decode_stream_),
                      "cudaStreamSynchronize(decode_stream_,dequant_cleanup)");
    }
    if (prefill_stream_) {
      CheckCudaStatus(cudaStreamSynchronize(prefill_stream_),
                      "cudaStreamSynchronize(prefill_stream_,dequant_cleanup)");
    }
#endif

#ifdef INFERFLUX_NATIVE_KERNELS_READY
    if (quantized_weight_map_) {
      quantized_weight_map_->ClearCache();
      quantized_weight_map_->ReleaseScratchBuffer();
    }
    if (decode_lane_quantized_weight_map_) {
      decode_lane_quantized_weight_map_->ClearCache();
      decode_lane_quantized_weight_map_->ReleaseScratchBuffer();
    }
    if (prefill_lane_quantized_weight_map_) {
      prefill_lane_quantized_weight_map_->ClearCache();
      prefill_lane_quantized_weight_map_->ReleaseScratchBuffer();
    }
#endif
    RefreshMemoryLedger();
  }

  // ClearDequantizedCache() nullifies shared_ptrs to dequantized GPU weights.
  // This can free GPU memory that kernels on other streams may still be
  // reading. Sync all active streams before clearing to prevent use-after-free
  // when concurrent lanes have in-flight GPU work referencing cached weights.
  // The HasDequantizedCache() check avoids the 3-stream sync penalty (~2-4ms)
  // in the common case where no weights were lazily dequantized this batch.
  if (model_loader_->HasDequantizedCache()) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
    if (compute_stream_) {
      cudaStreamSynchronize(compute_stream_);
    }
    if (decode_stream_) {
      cudaStreamSynchronize(decode_stream_);
    }
    if (prefill_stream_) {
      cudaStreamSynchronize(prefill_stream_);
    }
#endif
    model_loader_->ClearDequantizedCache();
  }
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
void InferfluxCudaExecutor::DestroyLaneOverlapResources() {
  std::lock_guard<std::mutex> lock(lane_overlap_mutex_);
  DestroyLaneOverlapResourcesUnlocked();
}

void InferfluxCudaExecutor::DestroyLaneOverlapResourcesUnlocked() {
  lane_overlap_ready_ = false;
  decode_lane_forward_.reset();
  prefill_lane_forward_.reset();
  decode_lane_sampler_.reset();
  prefill_lane_sampler_.reset();
  decode_lane_gemm_.reset();
  prefill_lane_gemm_.reset();
  decode_lane_quantized_weight_adapter_.reset();
  prefill_lane_quantized_weight_adapter_.reset();
  decode_lane_quantized_weight_map_.reset();
  prefill_lane_quantized_weight_map_.reset();
  if (d_decode_logits_) {
    if (cudaFree(d_decode_logits_) != cudaSuccess) {
      log::Error("inferflux_cuda_executor",
                 "cudaFree(d_decode_logits_) failed");
    }
    d_decode_logits_ = nullptr;
  }
  if (d_prefill_logits_) {
    if (cudaFree(d_prefill_logits_) != cudaSuccess) {
      log::Warn("inferflux_cuda_executor",
                "cudaFree(d_prefill_logits_) failed");
    }
    d_prefill_logits_ = nullptr;
  }
}

bool InferfluxCudaExecutor::CanRunLaneOverlap() const {
  if (!lane_overlap_ready_ || !decode_lane_forward_ || !prefill_lane_forward_ ||
      !decode_lane_sampler_ || !prefill_lane_sampler_ || !decode_lane_gemm_ ||
      !prefill_lane_gemm_ || !d_decode_logits_ || !d_prefill_logits_) {
    return false;
  }

  if (model_loader_ && model_loader_->GetFormat() == "gguf") {
    return decode_lane_quantized_weight_map_ &&
           prefill_lane_quantized_weight_map_ &&
           decode_lane_quantized_weight_adapter_ &&
           prefill_lane_quantized_weight_adapter_;
  }
  return true;
}

InferfluxCudaExecutor::LaneExecutionResources
InferfluxCudaExecutor::PrimaryLaneResources() {
  LaneExecutionResources resources;
  resources.forward = model_forward_.get();
  resources.sampler = sampler_.get();
  resources.gemm = gemm_.get();
  resources.quantized_weights = quantized_weight_map_.get();
  resources.logits = d_logits_;
  resources.stream = compute_stream_;
  return resources;
}

InferfluxCudaExecutor::LaneExecutionResources
InferfluxCudaExecutor::GetLaneResources(bool decode_lane) {
  if (CanRunLaneOverlap()) {
    LaneExecutionResources lane_resources;
    lane_resources.forward =
        decode_lane ? decode_lane_forward_.get() : prefill_lane_forward_.get();
    lane_resources.sampler =
        decode_lane ? decode_lane_sampler_.get() : prefill_lane_sampler_.get();
    lane_resources.gemm =
        decode_lane ? decode_lane_gemm_.get() : prefill_lane_gemm_.get();
    if (model_loader_ && model_loader_->GetFormat() == "gguf") {
      lane_resources.quantized_weights =
          decode_lane ? decode_lane_quantized_weight_map_.get()
                      : prefill_lane_quantized_weight_map_.get();
    } else {
      lane_resources.quantized_weights = nullptr;
    }
    lane_resources.logits = decode_lane ? d_decode_logits_ : d_prefill_logits_;
    lane_resources.stream = decode_lane ? decode_stream_ : prefill_stream_;
    return lane_resources;
  }
  return PrimaryLaneResources();
}

bool InferfluxCudaExecutor::InitializeLaneOverlapResources(
    const SafetensorsLoader::ModelConfig &config, bool want_bf16,
    int max_batch) {
  DestroyLaneOverlapResourcesUnlocked();
  if (!overlap_enabled_) {
    return false;
  }
  if (!decode_stream_ || !prefill_stream_) {
    return false;
  }
  const bool is_gguf_path =
      (model_loader_ && model_loader_->GetFormat() == "gguf");
  if (!kv_cache_) {
    return false;
  }
  if (!is_gguf_path && !weight_map_) {
    return false;
  }
  if (is_gguf_path && !model_loader_) {
    return false;
  }

  decode_lane_gemm_ = std::make_unique<CublasGemm>();
  prefill_lane_gemm_ = std::make_unique<CublasGemm>();
  if (!decode_lane_gemm_->Initialize(decode_stream_) ||
      !prefill_lane_gemm_->Initialize(prefill_stream_)) {
    log::Warn("inferflux_cuda_executor",
              "Failed to initialize lane-overlap cuBLAS handles");
    DestroyLaneOverlapResources();
    return false;
  }
  decode_lane_gemm_->PreallocateWorkspace(4 * 1024 * 1024);
  prefill_lane_gemm_->PreallocateWorkspace(4 * 1024 * 1024);

  if (is_gguf_path) {
    // GGUF path currently feeds FP16 into the forward path.
    decode_lane_forward_ = CreateModelForward(config.model_type);
    prefill_lane_forward_ = CreateModelForward(config.model_type);
  } else if (want_bf16) {
    decode_lane_forward_ =
        CreateModelForwardTyped<__nv_bfloat16>(config.model_type);
    prefill_lane_forward_ =
        CreateModelForwardTyped<__nv_bfloat16>(config.model_type);
  } else {
    decode_lane_forward_ = CreateModelForward(config.model_type);
    prefill_lane_forward_ = CreateModelForward(config.model_type);
  }
  if (!decode_lane_forward_ || !prefill_lane_forward_) {
    log::Warn("inferflux_cuda_executor",
              "Failed to create lane-overlap forward replicas");
    DestroyLaneOverlapResources();
    return false;
  }

  if (is_gguf_path) {
    decode_lane_quantized_weight_map_ = std::make_unique<QuantizedWeightMap>();
    prefill_lane_quantized_weight_map_ = std::make_unique<QuantizedWeightMap>();
    if (!decode_lane_quantized_weight_map_->Build(
            model_loader_.get(), model_info_, decode_stream_) ||
        !prefill_lane_quantized_weight_map_->Build(
            model_loader_.get(), model_info_, prefill_stream_)) {
      log::Warn("inferflux_cuda_executor",
                "Failed to build GGUF lane-local quantized maps");
      DestroyLaneOverlapResources();
      return false;
    }
    const bool allow_fused_quantized_matmul =
        quantized_matmul_mode_ ==
        runtime::cuda::native::MatmulExecutionMode::kFusedDequantTileGemm;
    decode_lane_quantized_weight_map_->SetAllowFusedQuantizedMatmul(
        allow_fused_quantized_matmul);
    decode_lane_quantized_weight_map_->SetBatchDequantCacheEnabled(
        execution_policy_.enable_batch_dequant_cache);
    prefill_lane_quantized_weight_map_->SetAllowFusedQuantizedMatmul(
        allow_fused_quantized_matmul);
    prefill_lane_quantized_weight_map_->SetBatchDequantCacheEnabled(
        execution_policy_.enable_batch_dequant_cache);

    decode_lane_quantized_weight_adapter_ =
        std::make_unique<QuantizedWeightMapAdapter>(
            decode_lane_quantized_weight_map_.get());
    prefill_lane_quantized_weight_adapter_ =
        std::make_unique<QuantizedWeightMapAdapter>(
            prefill_lane_quantized_weight_map_.get());
    WarmQuantizedLaneCache(decode_lane_quantized_weight_map_.get());
    WarmQuantizedLaneCache(prefill_lane_quantized_weight_map_.get());

    const auto gguf_config = ConvertModelInfo(model_info_);
    if (!decode_lane_forward_->Initialize(
            gguf_config, *decode_lane_quantized_weight_adapter_,
            kv_cache_.get(), decode_lane_gemm_.get(), decode_stream_) ||
        !prefill_lane_forward_->Initialize(
            gguf_config, *prefill_lane_quantized_weight_adapter_,
            kv_cache_.get(), prefill_lane_gemm_.get(), prefill_stream_)) {
      log::Warn("inferflux_cuda_executor",
                "Failed to initialize GGUF lane-overlap forward replicas");
      DestroyLaneOverlapResources();
      return false;
    }
    decode_lane_forward_->SetExecutionPolicy(execution_policy_);
    prefill_lane_forward_->SetExecutionPolicy(execution_policy_);
  } else {
    if (!decode_lane_forward_->Initialize(config, *weight_map_, kv_cache_.get(),
                                          decode_lane_gemm_.get(),
                                          decode_stream_) ||
        !prefill_lane_forward_->Initialize(
            config, *weight_map_, kv_cache_.get(), prefill_lane_gemm_.get(),
            prefill_stream_)) {
      log::Warn("inferflux_cuda_executor",
                "Failed to initialize lane-overlap forward replicas");
      DestroyLaneOverlapResources();
      return false;
    }
    decode_lane_forward_->SetExecutionPolicy(execution_policy_);
    prefill_lane_forward_->SetExecutionPolicy(execution_policy_);
  }

  // CUDA graphs are unsafe with lane overlap workers: decode and prefill
  // workers run on separate threads with varying batch sizes, causing graph
  // destroy/capture races. Disable graphs on lane-specific forwards only.
  // The primary forward keeps graphs enabled for non-overlapped execution
  // (single-sequence decode or when lane overlap is not triggered).
  // The lane_overlap_mutex_ in ExecuteLaneBatchForAsync and
  // ReleaseBatchScopedDequantizedCache prevents concurrent access to shared
  // resources during lane execution.
  {
    NativeExecutionPolicy lane_policy = execution_policy_;
    lane_policy.disable_cuda_graph = true;
    decode_lane_forward_->SetExecutionPolicy(lane_policy);
    prefill_lane_forward_->SetExecutionPolicy(lane_policy);
    log::Info("inferflux_cuda_executor",
              "CUDA graphs disabled on lane forwards (overlap active); "
              "primary forward retains graph support");
  }

  decode_lane_sampler_ = std::make_unique<GpuSampler>();
  prefill_lane_sampler_ = std::make_unique<GpuSampler>();
  if (!decode_lane_sampler_->Initialize(config.vocab_size, decode_stream_) ||
      !prefill_lane_sampler_->Initialize(config.vocab_size, prefill_stream_)) {
    log::Warn("inferflux_cuda_executor",
              "Failed to initialize lane-overlap samplers");
    DestroyLaneOverlapResources();
    return false;
  }

  std::size_t logits_elements = 0;
  std::size_t logits_bytes = 0;
  if (!CheckedMulSize(static_cast<std::size_t>(max_batch),
                      static_cast<std::size_t>(config.vocab_size),
                      &logits_elements) ||
      !CheckedMulSize(logits_elements, sizeof(float), &logits_bytes)) {
    log::Warn("inferflux_cuda_executor",
              "Lane-overlap logits allocation overflow");
    DestroyLaneOverlapResources();
    return false;
  }
  if (!CheckCudaStatus(cudaMalloc(&d_decode_logits_, logits_bytes),
                       "cudaMalloc(d_decode_logits_)") ||
      !CheckCudaStatus(cudaMalloc(&d_prefill_logits_, logits_bytes),
                       "cudaMalloc(d_prefill_logits_)")) {
    log::Warn("inferflux_cuda_executor",
              "Failed to allocate lane-overlap logits buffers");
    DestroyLaneOverlapResources();
    return false;
  }

  lane_overlap_ready_ = true;
  const std::string overlap_mode =
      is_gguf_path ? " (GGUF lane-local quantized maps)" : "";
  log::Info(
      "inferflux_cuda_executor",
      "Lane-overlap replicas initialized (decode/prefill forward + sampler + "
      "logits)" +
          overlap_mode);
  return true;
}

bool InferfluxCudaExecutor::EnsureLaneOverlapResources() {
  std::lock_guard<std::mutex> lock(lane_overlap_mutex_);
  if (!overlap_enabled_) {
    return false;
  }
  if (lane_overlap_ready_) {
    return true;
  }
  if (lane_overlap_init_attempted_) {
    return false;
  }

  lane_overlap_init_attempted_ = true;
  const bool want_bf16 = (inference_dtype_ == InferenceDtype::kBF16);
  if (!InitializeLaneOverlapResources(model_config_, want_bf16,
                                      active_max_batch_)) {
    log::Warn("inferflux_cuda_executor",
              "Lane-overlap resources unavailable; mixed workloads will use "
              "single-lane execution");
    RefreshMemoryLedger();
    return false;
  }

  RefreshMemoryLedger();
  return true;
}

bool InferfluxCudaExecutor::InitializeNativePipeline() {
  const auto &config = model_config_;

  // Detect inference dtype from model config (overridable via bootstrap env).
  bool want_bf16 = (config.torch_dtype == "bfloat16");
  if (bootstrap_config_.ForceFp16()) {
    want_bf16 = false;
    log::Info("inferflux_cuda_executor",
              "INFERFLUX_CUDA_DTYPE=fp16 override: forcing FP16 pipeline");
  } else if (bootstrap_config_.ForceBf16()) {
    want_bf16 = true;
    log::Info("inferflux_cuda_executor",
              "INFERFLUX_CUDA_DTYPE=bf16 override: forcing BF16 pipeline");
  }
  if (want_bf16 && !CheckBF16Support()) {
    log::Warn("inferflux_cuda_executor",
              "BF16 requested but GPU SM < 80; falling back to FP16");
    want_bf16 = false;
  }
  inference_dtype_ = want_bf16 ? InferenceDtype::kBF16 : InferenceDtype::kFP16;

  const std::string &kv_precision_choice =
      bootstrap_config_.kv_precision_choice;

  if (kv_precision_choice == "auto") {
    kv_precision_ = want_bf16 ? runtime::cuda::native::KvPrecision::kBf16
                              : runtime::cuda::native::KvPrecision::kFp16;
  } else if (!runtime::cuda::native::ParseKvPrecision(kv_precision_choice,
                                                      &kv_precision_)) {
    log::Warn("inferflux_cuda_executor", "Invalid KV precision '" +
                                             kv_precision_choice +
                                             "', falling back to auto");
    kv_precision_ = want_bf16 ? runtime::cuda::native::KvPrecision::kBf16
                              : runtime::cuda::native::KvPrecision::kFp16;
  }

  if (kv_precision_ == runtime::cuda::native::KvPrecision::kBf16 &&
      !CheckBF16Support()) {
    log::Warn("inferflux_cuda_executor",
              "BF16 KV cache requested but GPU SM < 80; using FP16 KV cache");
    kv_precision_ = runtime::cuda::native::KvPrecision::kFp16;
  }
  if (kv_precision_ == runtime::cuda::native::KvPrecision::kInt8 ||
      kv_precision_ == runtime::cuda::native::KvPrecision::kFp8) {
    log::Warn("inferflux_cuda_executor",
              "KV precision '" +
                  runtime::cuda::native::KvPrecisionToString(kv_precision_) +
                  "' is not implemented yet; using FP16 KV cache");
    kv_precision_ = runtime::cuda::native::KvPrecision::kFp16;
  }

  log::Info("inferflux_cuda_executor",
            "Inference dtype: " + std::string(want_bf16 ? "bf16" : "fp16") +
                " (torch_dtype=" + config.torch_dtype + "), kv_dtype=" +
                runtime::cuda::native::KvPrecisionToString(kv_precision_));

  // 1. Initialize cuBLAS wrapper
  gemm_ = std::make_unique<CublasGemm>();
  if (!gemm_->Initialize(compute_stream_)) {
    log::Error("inferflux_cuda_executor", "Failed to initialize cuBLAS");
    return false;
  }
  // Pre-allocate cuBLAS workspace to avoid dynamic allocation during
  // CUDA graph capture. 4 MB is sufficient for typical decode geometries.
  gemm_->PreallocateWorkspace(4 * 1024 * 1024);

  // 2. Allocate KV cache (independent precision policy)
  // Defaults can be overridden via env vars to reduce GPU memory usage.
  // Default max_batch is scheduler-aligned to avoid reserving unused slots.
  int max_batch = bootstrap_config_.kv_max_batch;
  int max_seq = bootstrap_config_.kv_max_seq;
  bool max_seq_overridden = bootstrap_config_.kv_max_seq_overridden;
  if (!bootstrap_config_.invalid_kv_max_batch.empty()) {
    log::Warn("inferflux_cuda_executor",
              "Ignoring invalid INFERFLUX_CUDA_KV_MAX_BATCH='" +
                  bootstrap_config_.invalid_kv_max_batch + "'");
  } else if (max_batch != 16) {
    log::Info("inferflux_cuda_executor",
              "KV cache max_batch overridden to " + std::to_string(max_batch));
  }
  // KV cache batch slots must accommodate the maximum concurrent sequences
  // the scheduler will decode.  The scheduler's SequenceSlotManager handles
  // arbitrary slot counts, so the minimum is just enough for batched decode.
  constexpr int kMinKvBatch = 4;
  if (max_batch < kMinKvBatch) {
    log::Warn("inferflux_cuda_executor",
              "KV max_batch=" + std::to_string(max_batch) + " < minimum (" +
                  std::to_string(kMinKvBatch) + "), clamping to " +
                  std::to_string(kMinKvBatch));
    max_batch = kMinKvBatch;
  }
  if (!bootstrap_config_.invalid_kv_max_seq.empty()) {
    log::Warn("inferflux_cuda_executor",
              "Ignoring invalid INFERFLUX_CUDA_KV_MAX_SEQ='" +
                  bootstrap_config_.invalid_kv_max_seq + "'");
  } else if (max_seq_overridden) {
    log::Info("inferflux_cuda_executor",
              "KV cache max_seq overridden to " + std::to_string(max_seq));
  }
  if (config.max_position_embeddings > 0 &&
      config.max_position_embeddings < max_seq) {
    max_seq = config.max_position_embeddings;
  }

  const bool kv_auto_tune = bootstrap_config_.kv_auto_tune;
  const std::size_t kv_budget_bytes = bootstrap_config_.kv_budget_bytes;
  if (!bootstrap_config_.invalid_kv_budget_mb.empty()) {
    log::Warn("inferflux_cuda_executor",
              "Ignoring invalid INFERFLUX_CUDA_KV_BUDGET_MB='" +
                  bootstrap_config_.invalid_kv_budget_mb + "'");
  }

  const double kv_budget_ratio = bootstrap_config_.kv_budget_ratio;
  if (!bootstrap_config_.invalid_kv_free_mem_ratio.empty()) {
    log::Warn("inferflux_cuda_executor",
              "Ignoring invalid INFERFLUX_CUDA_KV_FREE_MEM_RATIO='" +
                  bootstrap_config_.invalid_kv_free_mem_ratio + "'");
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (kv_auto_tune && kv_budget_bytes == 0) {
    const cudaError_t mem_info_status =
        cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_info_status != cudaSuccess) {
      log::Warn("inferflux_cuda_executor",
                "cudaMemGetInfo failed for KV auto-tuning: " +
                    std::string(cudaGetErrorString(mem_info_status)));
      free_bytes = 0;
      total_bytes = 0;
    }
  }

  const std::size_t kv_element_bytes = sizeof(half);
  runtime::cuda::native::KvCachePlanInput kv_plan_input;
  kv_plan_input.requested_max_batch = max_batch;
  kv_plan_input.requested_max_seq = max_seq;
  kv_plan_input.min_max_batch = kMinKvBatch;
  kv_plan_input.min_max_seq = 128;
  kv_plan_input.model_max_position_embeddings = config.max_position_embeddings;
  kv_plan_input.bytes_per_token_per_sequence =
      runtime::cuda::native::EstimateKvBytesPerTokenPerSequence(
          config.num_hidden_layers, config.num_key_value_heads, config.head_dim,
          kv_element_bytes);
  kv_plan_input.auto_tune_enabled = kv_auto_tune;
  kv_plan_input.max_seq_overridden = max_seq_overridden;
  kv_plan_input.explicit_budget_bytes = kv_budget_bytes;
  kv_plan_input.free_bytes = free_bytes;
  kv_plan_input.budget_ratio = kv_budget_ratio;

  const auto kv_plan = runtime::cuda::native::PlanKvCache(kv_plan_input);
  GlobalMetrics().RecordInferfluxCudaKvAutoTunePlan(
      kv_plan_input.requested_max_seq, kv_plan.max_seq, kv_plan.requested_bytes,
      kv_plan.planned_bytes, kv_plan.budget_bytes);
  max_batch = kv_plan.max_batch;
  max_seq = kv_plan.max_seq;
  active_max_batch_ = max_batch;
  active_max_seq_ = max_seq;

  if (kv_plan.auto_tuned_seq) {
    log::Info(
        "inferflux_cuda_executor",
        "KV auto-tune reduced max_seq to " + std::to_string(max_seq) +
            " (requested_bytes=" +
            std::to_string(kv_plan.requested_bytes / kMiB) +
            " MiB, budget=" + std::to_string(kv_plan.budget_bytes / kMiB) +
            " MiB, planned=" + std::to_string(kv_plan.planned_bytes / kMiB) +
            " MiB" +
            (total_bytes > 0
                 ? ", free/total=" + std::to_string(free_bytes / kMiB) + "/" +
                       std::to_string(total_bytes / kMiB) + " MiB"
                 : "") +
            ")");
  } else {
    const std::string budget_source =
        kv_budget_bytes > 0 ? "explicit"
                            : (free_bytes > 0 ? "free_mem" : "none");
    log::Info("inferflux_cuda_executor",
              "KV allocation plan: max_batch=" + std::to_string(max_batch) +
                  ", max_seq=" + std::to_string(max_seq) + ", estimated=" +
                  std::to_string(kv_plan.planned_bytes / kMiB) +
                  " MiB, auto_tune=" + (kv_auto_tune ? "on" : "off") +
                  ", budget_source=" + budget_source +
                  (free_bytes > 0
                       ? ", free/total=" + std::to_string(free_bytes / kMiB) +
                             "/" + std::to_string(total_bytes / kMiB) + " MiB"
                       : ""));
  }

  const int kv_base_slots = bootstrap_config_.kv_base_slots;
  if (!bootstrap_config_.invalid_kv_base_slots.empty()) {
    log::Warn("inferflux_cuda_executor",
              "Ignoring invalid INFERFLUX_CUDA_KV_BASE_SLOTS='" +
                  bootstrap_config_.invalid_kv_base_slots + "'");
  }

  if (kv_base_slots > 0) {
    // Hybrid KV cache: dense base + per-slot overflow
    if (kv_precision_ == runtime::cuda::native::KvPrecision::kBf16) {
      auto cache = std::make_unique<HybridKvCacheGpuTyped<__nv_bfloat16>>();
      if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                           config.head_dim, max_seq, max_batch,
                           kv_base_slots)) {
        log::Error("inferflux_cuda_executor",
                   "Failed to allocate hybrid BF16 KV cache");
        return false;
      }
      kv_cache_ = std::move(cache);
    } else {
      auto cache = std::make_unique<HybridKvCacheGpuTyped<half>>();
      if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                           config.head_dim, max_seq, max_batch,
                           kv_base_slots)) {
        log::Error("inferflux_cuda_executor",
                   "Failed to allocate hybrid FP16 KV cache");
        return false;
      }
      kv_cache_ = std::move(cache);
    }
    log::Info(
        "inferflux_cuda_executor",
        "Using hybrid KV cache: base_slots=" + std::to_string(kv_base_slots) +
            ", max_batch=" + std::to_string(max_batch));
  } else if (kv_precision_ == runtime::cuda::native::KvPrecision::kBf16) {
    auto cache = std::make_unique<KvCacheGpuTyped<__nv_bfloat16>>();
    if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                         config.head_dim, max_seq, max_batch)) {
      log::Error("inferflux_cuda_executor", "Failed to allocate BF16 KV cache");
      return false;
    }
    kv_cache_ = std::move(cache);
  } else {
    auto cache = std::make_unique<KvCacheGpu>();
    if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                         config.head_dim, max_seq, max_batch)) {
      log::Error("inferflux_cuda_executor", "Failed to allocate FP16 KV cache");
      return false;
    }
    kv_cache_ = std::move(cache);
  }
  GlobalMetrics().SetInferfluxCudaKvCacheOccupancy(/*active_sequences=*/0,
                                                   /*max_sequences=*/max_batch);

  // 2.5. Strategy selection (foundation layer for native quantized runtime).
  runtime::cuda::native::QuantizedRuntimeStrategyRegistry &registry =
      runtime::cuda::native::QuantizedRuntimeStrategyRegistry::Instance();
  registry.RegisterDefaults();

  if (model_loader_ && model_loader_->GetFormat() == "gguf") {
    cudaDeviceProp prop{};
    int sm_major = 0;
    int sm_minor = 0;
    if (CheckCudaStatus(cudaGetDeviceProperties(&prop, device_id_),
                        "cudaGetDeviceProperties(strategy selection)")) {
      sm_major = prop.major;
      sm_minor = prop.minor;
    }

    std::string quantization_type = model_loader_->GetQuantizationType();
    const bool is_quantized_model = model_loader_->IsQuantized();
    if (!quantization_type.empty() &&
        IsKnownQuantizationType(quantization_type)) {
      const auto tensor_type =
          runtime::cuda::native::StringToTensorType(quantization_type);
      const auto selection =
          registry.Select(tensor_type, kv_precision_, sm_major, sm_minor);
      if (selection.weight_layout && selection.matmul && selection.attention) {
        quantized_matmul_mode_ = selection.matmul->Mode();
        quantized_matmul_strategy_id_ = selection.matmul->Id();
        const auto mode = quantized_matmul_mode_;
        log::Info("inferflux_cuda_executor",
                  "Selected GGUF strategies: layout=" +
                      selection.weight_layout->Id() + ", matmul=" +
                      selection.matmul->Id() + " (" + MatmulModeToString(mode) +
                      ")" + ", attention=" + selection.attention->Id() + " (" +
                      selection.reason + ")");
        if (require_fused_quantized_matmul_ && is_quantized_model &&
            mode != runtime::cuda::native::MatmulExecutionMode::
                        kFusedDequantTileGemm) {
          log::Error("inferflux_cuda_executor",
                     "Strict quantized runtime policy rejected startup: "
                     "fused dequant-tile GEMM required but selected matmul '" +
                         selection.matmul->Id() + "' (" +
                         MatmulModeToString(mode) + ")");
          return false;
        }
      } else {
        const std::string reason =
            "Incomplete GGUF strategy selection: " + selection.reason;
        if (require_fused_quantized_matmul_ && is_quantized_model) {
          log::Error("inferflux_cuda_executor",
                     "Strict quantized runtime policy rejected startup: " +
                         reason);
          return false;
        }
        log::Warn("inferflux_cuda_executor", reason);
      }
    } else if (is_quantized_model) {
      const std::string reason =
          "Quantized GGUF model loaded but quantization type '" +
          quantization_type + "' is unknown; strategy selection skipped";
      if (require_fused_quantized_matmul_) {
        log::Error("inferflux_cuda_executor",
                   "Strict quantized runtime policy rejected startup: " +
                       reason);
        return false;
      }
      log::Warn("inferflux_cuda_executor", reason);
    }
  }

  const bool allow_fused_quantized_matmul =
      quantized_matmul_mode_ ==
      runtime::cuda::native::MatmulExecutionMode::kFusedDequantTileGemm;
  log::Info("inferflux_cuda_executor",
            "Quantized matmul execution mode: " +
                MatmulModeToString(quantized_matmul_mode_) + ", strategy_id=" +
                quantized_matmul_strategy_id_ + ", allow_fused=" +
                std::string(allow_fused_quantized_matmul ? "true" : "false"));

  // 3. Create weights + forward implementation
  if (model_loader_) {
    quantized_weight_map_ = std::make_unique<QuantizedWeightMap>();
    if (!quantized_weight_map_->Build(model_loader_.get(), model_info_,
                                      compute_stream_)) {
      log::Error("inferflux_cuda_executor",
                 "Failed to build quantized weight map");
      return false;
    }
    quantized_weight_map_->SetAllowFusedQuantizedMatmul(
        allow_fused_quantized_matmul);
    quantized_weight_map_->SetBatchDequantCacheEnabled(
        execution_policy_.enable_batch_dequant_cache);
    quantized_weight_adapter_ = std::make_unique<QuantizedWeightMapAdapter>(
        quantized_weight_map_.get());
    // GGUF always dequantizes to FP16, so use LlamaForwardTyped<half>
    model_forward_ = CreateModelForward(config.model_type);
    if (!model_forward_) {
      log::Error("inferflux_cuda_executor",
                 "Failed to create forward for model_type: " +
                     config.model_type);
      return false;
    }
    auto gguf_config = ConvertModelInfo(model_info_);
    if (!model_forward_->Initialize(gguf_config, *quantized_weight_adapter_,
                                    kv_cache_.get(), gemm_.get(),
                                    compute_stream_)) {
      log::Error("inferflux_cuda_executor",
                 "Failed to initialize forward pass for GGUF model");
      return false;
    }
    model_forward_->SetExecutionPolicy(execution_policy_);
  } else {
    weight_map_ = std::make_unique<WeightMap>();
    if (!weight_map_->Build(*loader_, config)) {
      log::Error("inferflux_cuda_executor", "Failed to build weight map");
      return false;
    }
    if (want_bf16) {
      model_forward_ =
          CreateModelForwardTyped<__nv_bfloat16>(config.model_type);
    } else {
      model_forward_ = CreateModelForward(config.model_type);
    }
    if (!model_forward_) {
      log::Error("inferflux_cuda_executor",
                 "Unsupported model_type: " + config.model_type);
      return false;
    }
    if (!model_forward_->Initialize(config, *weight_map_, kv_cache_.get(),
                                    gemm_.get(), compute_stream_)) {
      log::Error("inferflux_cuda_executor",
                 "Failed to initialize forward pass");
      return false;
    }
    model_forward_->SetExecutionPolicy(execution_policy_);
  }

  // 3b. Pre-warm weight caches so CUDA graph capture doesn't encounter
  // illegal cudaStreamSynchronize from lazy F32→FP16 dequantization.
  model_forward_->WarmWeightCaches();

  // Dispatch reachability probe: verifies every selectable operator can
  // actually execute and down-ranks divergent ones for the process
  // lifetime (see dispatch_catalog.h / dispatch_operator_health.h).
  // Opt out with INFERFLUX_CUDA_DISPATCH_PROBE=0.
  const char *probe_env = std::getenv("INFERFLUX_CUDA_DISPATCH_PROBE");
  const bool probe_enabled = !probe_env || (std::string(probe_env) != "0" &&
                                            std::string(probe_env) != "false");
  if (probe_enabled) {
    const std::string downgraded = model_forward_->ProbeDispatchPaths();
    if (!downgraded.empty()) {
      GlobalMetrics().RecordInferfluxCudaDispatchDivergence(
          "probe", "probe", "probe", "down_ranked");
    }
  }

  // 4. Initialize GPU sampler
  sampler_ = std::make_unique<GpuSampler>();
  if (!sampler_->Initialize(config.vocab_size, compute_stream_)) {
    log::Error("inferflux_cuda_executor", "Failed to initialize GPU sampler");
    return false;
  }

  // 5. Allocate device logits buffer (sized for batched decode)
  if (config.vocab_size <= 0) {
    log::Error("inferflux_cuda_executor",
               "Invalid vocab_size for logits buffer");
    return false;
  }
  std::size_t logits_elements = 0;
  std::size_t logits_bytes = 0;
  if (!CheckedMulSize(static_cast<std::size_t>(max_batch),
                      static_cast<std::size_t>(config.vocab_size),
                      &logits_elements) ||
      !CheckedMulSize(logits_elements, sizeof(float), &logits_bytes)) {
    log::Error("inferflux_cuda_executor", "Logits buffer size overflow");
    return false;
  }
  cudaError_t err = cudaMalloc(&d_logits_, logits_bytes);
  if (err != cudaSuccess) {
    log::Error("inferflux_cuda_executor", "Failed to allocate logits buffer");
    return false;
  }

  // 6. Load tokenizer. Use llama.cpp's proven tokenizer (LlamaTokenizer) for
  // Encode/Decode — it handles all regex-based pre-tokenization patterns
  // (GPT-2, Qwen2, LLaMA-3, etc.) correctly. The GGUF metadata tokenizer
  // (GGUFTokenizer) is used only for chat template rendering via
  // NativeFormatChat(), where its strategy-based renderer (ChatML/Llama/
  // Mistral/Gemma) is the authoritative path.
  {
    tokenizer_.reset();
    const std::string tokenizer_path =
        model_loader_ ? loaded_model_path_.string() : loader_->GetModelPath();
    const std::string model_format =
        model_loader_ ? model_loader_->GetFormat() : "safetensors";
    tokenizer_ = CreateTokenizer(tokenizer_path, model_format);
    if (!tokenizer_) {
      log::Error("inferflux_cuda_executor",
                 "Failed to initialize tokenizer for model path: " +
                     tokenizer_path);
      return false;
    }
    log::Info("inferflux_cuda_executor",
              "Initialized tokenizer (llama.cpp) for model: " + tokenizer_path);

    // Initialize GGUF chat template renderer for NativeFormatChat(). This is
    // separate from the main tokenizer — it only renders chat messages into
    // the correct template format (ChatML, Llama, Mistral, Gemma).
    if (auto *gguf_loader =
            dynamic_cast<runtime::cuda::native::GGUFModelLoader *>(
                model_loader_.get())) {
      auto gguf_tokenizer = std::make_unique<GGUFTokenizer>();
      if (gguf_tokenizer->InitializeFromMetadata(
              gguf_loader->TokenizerPieces(), gguf_loader->TokenizerMerges(),
              gguf_loader->TokenizerPreTokenizer(),
              gguf_loader->TokenizerBosTokenId(),
              gguf_loader->TokenizerEosTokenId(),
              gguf_loader->TokenizerAddBosToken(),
              gguf_loader->TokenizerChatTemplate(),
              gguf_loader->TokenizerTokenTypes())) {
        chat_template_tokenizer_ = std::move(gguf_tokenizer);
        log::Info("inferflux_cuda_executor",
                  "Initialized GGUF chat template renderer");
      }
    }
  }

  // 9. Read timing sample-rate from NativeExecutionPolicy.
  //    0 → disable event recording entirely (max throughput).
  //    N > 0 → record events every Nth batch.
  timing_sample_rate_ = execution_policy_.timing_sample_rate;
  log::Info("inferflux_cuda_executor",
            "Timing sample rate: " + std::to_string(timing_sample_rate_) +
                (timing_sample_rate_ <= 0 ? " (disabled)" : ""));

  // Create cudaEvent pairs for forward/sampling timing
  if (timing_sample_rate_ > 0) {
    if (!CheckCudaStatus(cudaEventCreate(&forward_start_),
                         "cudaEventCreate(forward_start_)") ||
        !CheckCudaStatus(cudaEventCreate(&forward_stop_),
                         "cudaEventCreate(forward_stop_)") ||
        !CheckCudaStatus(cudaEventCreate(&sampling_start_),
                         "cudaEventCreate(sampling_start_)") ||
        !CheckCudaStatus(cudaEventCreate(&sampling_stop_),
                         "cudaEventCreate(sampling_stop_)")) {
      return false;
    }
  }

  // 10. Create lane-specific streams for async overlap
  if (!CheckCudaStatus(
          cudaStreamCreateWithFlags(&decode_stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(decode_stream_)") ||
      !CheckCudaStatus(
          cudaStreamCreateWithFlags(&prefill_stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(prefill_stream_)")) {
    return false;
  }

  RefreshMemoryLedger();
  log::Info("inferflux_cuda_executor", memory_ledger_.Describe());

  log::Info("inferflux_cuda_executor",
            "Native inference pipeline initialized successfully");
  return true;
}
#else
bool InferfluxCudaExecutor::InitializeNativePipeline() {
  log::Warn("inferflux_cuda_executor",
            "Native kernels not compiled; pipeline unavailable");
  return false;
}
#endif

bool InferfluxCudaExecutor::LoadModel(const std::filesystem::path &model_path,
                                      const LlamaBackendConfig &config) {
  log::Info("inferflux_cuda_executor",
            "Loading InferFlux CUDA model from: " + model_path.string());
  loaded_model_path_ = model_path;
  memory_ledger_.Clear();
  active_max_batch_ = 0;
  active_max_seq_ = 0;
  fallback_mode_ = false;
  fallback_reason_.clear();
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  StopLaneDispatcher();
#endif
  loader_.reset();
  model_loader_.reset();
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  weight_map_.reset();
  quantized_weight_map_.reset();
#endif

  // Initialize phase overlap settings from config
  overlap_enabled_ = config.cuda_phase_overlap_scaffold;
  min_prefill_tokens_ = config.cuda_phase_overlap_min_prefill_tokens;
  lane_overlap_init_attempted_ = false;
  dequantized_cache_policy_hint_ =
      config.inferflux_cuda_dequantized_cache_policy;
  bootstrap_config_ =
      NativeBootstrapConfig::FromEnv(config.inferflux_cuda_kv_cache_dtype);
  require_fused_quantized_matmul_ =
      config.inferflux_cuda_require_fused_quantized_matmul;
  execution_policy_ = NativeExecutionPolicy::FromEnv();
  quantized_matmul_mode_ =
      runtime::cuda::native::MatmulExecutionMode::kFusedDequantTileGemm;
  quantized_matmul_strategy_id_ = "matmul.fused.dequant_tile_gemm.v1";
  if (execution_policy_.require_fused_quantized_matmul_override) {
    require_fused_quantized_matmul_ =
        execution_policy_.require_fused_quantized_matmul;
  }
  if (!execution_policy_.dequantized_cache_policy_override.empty()) {
    dequantized_cache_policy_hint_ =
        execution_policy_.dequantized_cache_policy_override;
  }
  ConfigureDequantizedCachePolicy(dequantized_cache_policy_hint_);
  log::Info(
      "inferflux_cuda_executor",
      "Phase overlap: " +
          std::string(overlap_enabled_ ? "enabled" : "disabled") +
          ", min_prefill_tokens=" + std::to_string(min_prefill_tokens_) +
          ", kv_dtype_hint=" + bootstrap_config_.kv_precision_choice +
          ", dequant_cache_policy=" + dequantized_cache_policy_hint_ +
          ", require_fused_quantized_matmul=" +
          std::string(require_fused_quantized_matmul_ ? "true" : "false"));

  // Initialize CUDA
  if (!InitializeCUDA()) {
    return false;
  }

  auto detected_loader = runtime::cuda::native::CreateModelLoader(model_path);
  if (!detected_loader) {
    log::Error("inferflux_cuda_executor",
               "Failed to detect model loader for path: " +
                   model_path.string());
    return false;
  }

  const std::string detected_format = detected_loader->GetFormat();
  detected_loader->SetDequantizedCachePolicy(dequantized_cache_policy_);
  if (detected_format == "safetensors") {
    // Keep legacy safetensors loader path until the generic loader path is
    // feature-parity validated for BF16/FP16 conversion controls.
    loader_ = std::make_unique<SafetensorsLoader>();
    if (!loader_->LoadModel(model_path.string())) {
      log::Error("inferflux_cuda_executor", "Failed to load safetensors model");
      return false;
    }
    model_config_ = loader_->GetConfig();

    // Decide whether to skip BF16→FP16 conversion
    // If model is BF16 and GPU supports it, keep BF16 for native pipeline
    // Env var INFERFLUX_CUDA_DTYPE=fp16 forces FP16 conversion
    bool skip_bf16 = false;
#ifdef INFERFLUX_NATIVE_KERNELS_READY
    {
      bool force_fp16 = bootstrap_config_.ForceFp16();
      if (model_config_.torch_dtype == "bfloat16" && CheckBF16Support() &&
          !force_fp16) {
        skip_bf16 = true;
        log::Info(
            "inferflux_cuda_executor",
            "BF16 model detected with SM >= 80; uploading weights as BF16");
      } else if (model_config_.torch_dtype == "bfloat16" && force_fp16) {
        log::Info("inferflux_cuda_executor",
                  "BF16 model with FP16 override; converting BF16→FP16 on CPU");
      }
    }
#endif

    // Upload weights to GPU
    log::Info("inferflux_cuda_executor", "Uploading weights to GPU...");
    if (!loader_->UploadToGPU(compute_stream_, skip_bf16)) {
      log::Error("inferflux_cuda_executor", "Failed to upload weights to GPU");
      return false;
    }
  } else {
    model_loader_ = std::move(detected_loader);
    model_loader_->SetDequantizedCachePolicy(dequantized_cache_policy_);
    if (!model_loader_->Load(model_path)) {
      log::Error("inferflux_cuda_executor",
                 "Failed to load model via loader: " + detected_format);
      return false;
    }
    model_info_ = model_loader_->GetModelInfo();
    model_config_ = ConvertModelInfo(model_info_);
    log::Info("inferflux_cuda_executor", "Uploading model weights to GPU via " +
                                             detected_format + " loader...");
    if (!model_loader_->UploadToGPU(compute_stream_)) {
      log::Error("inferflux_cuda_executor",
                 "Failed to upload model weights via " + detected_format +
                     " loader");
      return false;
    }
  }

  // Initialize native inference pipeline
  if (!InitializeNativePipeline()) {
    log::Error("inferflux_cuda_executor",
               "Native pipeline initialization failed");
    return false;
  }

  // Report native backend uses FA2 attention kernel
  GlobalMetrics().SetCudaAttentionKernel("fa2");

  log::Info("inferflux_cuda_executor",
            "InferFlux CUDA model loaded successfully");
  model_loaded_ = true;
  return true;
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY

// ==========================================================================
// Phase Overlap Helper Methods
// ==========================================================================

bool InferfluxCudaExecutor::IsPrefillLikeInput(
    const UnifiedBatchInput &input) const {
  // Prefill requests have multiple tokens or don't request logits
  return input.tokens.size() > 1 || !input.request_logits;
}

bool InferfluxCudaExecutor::HasMixedWorkload(
    const std::vector<UnifiedBatchInput> &inputs) const {
  if (!overlap_enabled_) {
    return false;
  }

  bool has_prefill = false;
  bool has_decode = false;

  for (const auto &input : inputs) {
    if (IsPrefillLikeInput(input)) {
      has_prefill = true;
    } else {
      has_decode = true;
    }
    if (has_prefill && has_decode) {
      return true;
    }
  }

  return false;
}

void InferfluxCudaExecutor::SplitBatchByType(
    const std::vector<UnifiedBatchInput> &inputs,
    std::vector<size_t> &prefill_indices,
    std::vector<size_t> &decode_indices) const {
  prefill_indices.clear();
  decode_indices.clear();

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (IsPrefillLikeInput(inputs[i])) {
      prefill_indices.push_back(i);
    } else {
      decode_indices.push_back(i);
    }
  }
}

InferfluxCudaExecutor::LaneExecutionResult
InferfluxCudaExecutor::ExecuteLaneBatch(
    const std::vector<UnifiedBatchInput> &inputs,
    const LaneExecutionResources &resources) {
  LaneExecutionResult result;
  result.outputs.resize(inputs.size());

  if (inputs.empty()) {
    return result;
  }
  if (!resources.forward || !resources.sampler || !resources.gemm ||
      !resources.logits || !resources.stream) {
    log::Error("inferflux_cuda_executor",
               "ExecuteLaneBatch invoked without valid lane resources");
    for (auto &output : result.outputs) {
      output.ok = false;
      output.token = -1;
    }
    return result;
  }

  const bool uses_primary_pipeline =
      (resources.forward == model_forward_.get());
  resources.forward->SetStream(resources.stream);
  resources.gemm->SetStream(resources.stream);
  if (resources.quantized_weights) {
    resources.quantized_weights->SetStream(resources.stream);
  }

  struct DecodeEntry {
    int input_idx;
    int64_t request_id;
    std::string client_request_id;
    int token_id;
    int n_past;
    int sequence_id;
    uint64_t sequence_generation;
    float temperature;
    int top_k;
    float top_p;
    uint32_t seed;
    SamplingParams sampling;
  };
  std::vector<DecodeEntry> decode_group;
  std::vector<int> prefill_indices;
  decode_group.reserve(inputs.size());
  prefill_indices.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    if (input.tokens.size() == 1 && input.request_logits) {
      decode_group.push_back(
          {static_cast<int>(i), input.request_id, input.client_request_id,
           input.tokens[0], input.n_past, input.sequence_id,
           input.sequence_generation, input.sampling.temperature,
           input.sampling.top_k, input.sampling.top_p, input.sampling.seed,
           input.sampling});
    } else {
      prefill_indices.push_back(static_cast<int>(i));
    }
  }

  double decode_ms_total = 0.0;
  double sample_ms_total = 0.0;
  int decode_tokens_total = 0;
  const int decode_batch_capacity =
      kv_cache_ ? std::max(1, kv_cache_->MaxBatchSize()) : 32;
  if (!decode_group.empty()) {
    for (size_t offset = 0; offset < decode_group.size();
         offset += static_cast<size_t>(decode_batch_capacity)) {
      const int B = static_cast<int>(
          std::min(decode_group.size() - offset,
                   static_cast<size_t>(decode_batch_capacity)));
      std::vector<int> batch_tokens(B);
      std::vector<int> batch_n_past(B);
      std::vector<int> batch_seq_ids(B);
      std::vector<float> batch_temps(B);
      std::vector<int> batch_top_ks(B);
      std::vector<float> batch_top_ps(B);
      std::vector<uint32_t> batch_seeds(B);
      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        batch_tokens[b] = entry.token_id;
        batch_n_past[b] = entry.n_past;
        batch_seq_ids[b] = entry.sequence_id;
        batch_temps[b] = entry.temperature;
        batch_top_ks[b] = entry.top_k;
        batch_top_ps[b] = entry.top_p;
        batch_seeds[b] = entry.seed;
      }

      const auto forward_start = std::chrono::steady_clock::now();
      const bool fwd_ok = resources.forward->BatchForward(
          batch_tokens, batch_n_past, batch_seq_ids, resources.logits, B);
      const auto forward_end = std::chrono::steady_clock::now();
      decode_ms_total +=
          std::chrono::duration<double, std::milli>(forward_end - forward_start)
              .count();

      if (!fwd_ok) {
        log::Error("inferflux_cuda_executor", "Lane BatchForward failed");
        for (int b = 0; b < B; ++b) {
          const auto &entry = decode_group[offset + static_cast<size_t>(b)];
          auto &output = result.outputs[entry.input_idx];
          output.ok = false;
          output.token = -1;
          output.piece.clear();
        }
        continue;
      }

      std::vector<int> sampled_tokens;
      const auto sample_start = std::chrono::steady_clock::now();
      resources.sampler->SampleBatch(resources.logits, B, batch_temps,
                                     batch_top_ks, batch_top_ps, batch_seeds,
                                     &sampled_tokens);
      const auto sample_end = std::chrono::steady_clock::now();
      sample_ms_total +=
          std::chrono::duration<double, std::milli>(sample_end - sample_start)
              .count();

      if (DecodeMappingDebugEnabled(execution_policy_)) {
        log::Info("inferflux_cuda_executor",
                  "decode_mapping[lane]: batch_size=" + std::to_string(B) +
                      ", batch_offset=" + std::to_string(offset));
        for (int b = 0; b < B; ++b) {
          if (!ConsumeDecodeMappingBudget(execution_policy_)) {
            break;
          }
          const auto &entry = decode_group[offset + static_cast<size_t>(b)];
          const int token_id = sampled_tokens[b];
          const std::string piece = (tokenizer_ && token_id >= 0)
                                        ? tokenizer_->TokenToString(token_id)
                                        : std::string();
          log::Info("inferflux_cuda_executor",
                    "decode_mapping[lane]: input_idx=" +
                        std::to_string(entry.input_idx) + ", " +
                        (entry.client_request_id.empty()
                             ? std::string()
                             : "client_request_id=" + entry.client_request_id +
                                   ", ") +
                        "request_id=" + std::to_string(entry.request_id) +
                        ", sequence_id=" + std::to_string(entry.sequence_id) +
                        ", sequence_generation=" +
                        std::to_string(entry.sequence_generation) +
                        ", n_past=" + std::to_string(entry.n_past) +
                        ", sampled_token=" + std::to_string(token_id) +
                        ", piece=" + piece);
        }
      }

      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        int token_id = sampled_tokens[b];
        auto &output = result.outputs[entry.input_idx];
        if (tokenizer_ && tokenizer_->IsTerminalGeneratedToken(token_id)) {
          output.token = -1;
          output.piece.clear();
        } else {
          output.token = token_id;
          output.piece = VisibleTokenPiece(tokenizer_.get(), token_id);
          if (!output.piece.empty()) {
            perf_accum_.generated_tokens.fetch_add(1,
                                                   std::memory_order_relaxed);
          }
        }
        output.ok = true;
      }

      decode_tokens_total += B;
    }
  }

  double prefill_ms_total = 0.0;
  int prompt_tokens_total = 0;
  int sampled_prefill_total = 0;
  bool requires_lane_stream_drain = false;
  for (int idx : prefill_indices) {
    const auto &input = inputs[static_cast<size_t>(idx)];
    auto &output = result.outputs[static_cast<size_t>(idx)];
    output.ok = false;
    output.token = -1;
    output.piece.clear();

    const int token_count = static_cast<int>(input.tokens.size());
    const auto forward_start = std::chrono::steady_clock::now();
    if (!resources.forward->Forward(input.tokens, input.n_past,
                                    input.sequence_id, resources.logits)) {
      log::Error("inferflux_cuda_executor", "Lane Forward failed");
      continue;
    }
    const auto forward_end = std::chrono::steady_clock::now();
    prefill_ms_total +=
        std::chrono::duration<double, std::milli>(forward_end - forward_start)
            .count();
    prompt_tokens_total += token_count;

    if (input.request_logits) {
      const auto sample_start = std::chrono::steady_clock::now();
      resources.sampler->EnqueueSample(
          resources.logits, input.sampling.temperature, input.sampling.top_k,
          input.sampling.top_p, input.sampling.seed);
      const int token_id = resources.sampler->CollectSample();
      const auto sample_end = std::chrono::steady_clock::now();
      sample_ms_total +=
          std::chrono::duration<double, std::milli>(sample_end - sample_start)
              .count();
      ++sampled_prefill_total;
      LogSampleMapping("sample_mapping[lane_prefill]", idx, input.request_id,
                       input.client_request_id, input.sequence_id,
                       input.sequence_generation, input.n_past, token_count,
                       token_id, tokenizer_.get(), execution_policy_);

      if (tokenizer_ && tokenizer_->IsTerminalGeneratedToken(token_id)) {
        output.token = -1;
        output.piece.clear();
      } else {
        output.token = token_id;
        output.piece = VisibleTokenPiece(tokenizer_.get(), token_id);
        if (!output.piece.empty()) {
          perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
        }
      }
    } else {
      requires_lane_stream_drain = true;
    }

    output.ok = true;
  }

  if (requires_lane_stream_drain &&
      !CheckCudaStatus(
          runtime::cuda::native::TracedCudaStreamSynchronize(
              runtime::cuda::native::SyncTraceSite::kLaneExecutionDrain,
              resources.stream),
          "cudaStreamSynchronize(lane_execution)")) {
    for (auto &output : result.outputs) {
      output.ok = false;
      output.token = -1;
      output.piece.clear();
    }
  }

  if (decode_tokens_total > 0) {
    GlobalMetrics().RecordInferfluxCudaForwardPass(
        /*is_decode=*/true, decode_tokens_total, decode_ms_total);
    perf_accum_.decode_ms.store(
        perf_accum_.decode_ms.load(std::memory_order_relaxed) + decode_ms_total,
        std::memory_order_relaxed);
  }
  if (prompt_tokens_total > 0) {
    GlobalMetrics().RecordInferfluxCudaForwardPass(
        /*is_decode=*/false, prompt_tokens_total, prefill_ms_total);
    perf_accum_.prefill_ms.store(
        perf_accum_.prefill_ms.load(std::memory_order_relaxed) +
            prefill_ms_total,
        std::memory_order_relaxed);
    perf_accum_.prompt_tokens.fetch_add(prompt_tokens_total,
                                        std::memory_order_relaxed);
  }
  if (decode_tokens_total + sampled_prefill_total > 0 &&
      sample_ms_total > 0.0) {
    GlobalMetrics().RecordInferfluxCudaSampling(
        decode_tokens_total + sampled_prefill_total, sample_ms_total);
  }

  result.elapsed_ms = decode_ms_total + prefill_ms_total + sample_ms_total;

  if (uses_primary_pipeline) {
    model_forward_->SetStream(compute_stream_);
    gemm_->SetStream(compute_stream_);
    if (quantized_weight_map_) {
      quantized_weight_map_->SetStream(compute_stream_);
    }
  }

  return result;
}

InferfluxCudaExecutor::LaneExecutionResult
InferfluxCudaExecutor::ExecuteLaneBatchForAsync(
    const std::vector<UnifiedBatchInput> &inputs, bool decode_lane) {
  LaneExecutionResult result;
  if (inputs.empty()) {
    return result;
  }

  // Hold lane_overlap_mutex_ for the entire lane execution to prevent
  // concurrent DestroyLaneOverlapResources() or
  // ReleaseBatchScopedDequantizedCache() from invalidating lane resources
  // while this lane is still using them.
  std::lock_guard<std::mutex> lane_guard(lane_overlap_mutex_);

  const LaneExecutionResources lane_resources = GetLaneResources(decode_lane);
  const bool shares_quantized_map =
      quantized_weight_map_ &&
      lane_resources.quantized_weights == quantized_weight_map_.get();
  const bool shared_pipeline = lane_resources.forward == model_forward_.get() ||
                               lane_resources.gemm == gemm_.get() ||
                               shares_quantized_map ||
                               lane_resources.logits == d_logits_;

  GlobalMetrics().RecordCudaLaneExecutionStart(decode_lane);
  struct ScopedLaneStop {
    bool decode_lane;
    ~ScopedLaneStop() {
      GlobalMetrics().RecordCudaLaneExecutionStop(decode_lane);
    }
  } scoped_stop{decode_lane};

  std::unique_lock<std::mutex> pipeline_lock(shared_pipeline_mutex_,
                                             std::defer_lock);
  if (shared_pipeline) {
    pipeline_lock.lock();
  }

  result = ExecuteLaneBatch(inputs, lane_resources);
  // Mixed-workload overlap batches are cleaned up once by the outer sync path
  // after both lane handles complete. Doing that here would let decode/prefill
  // workers race on shared GGUF dequant cache and scratch ownership.
  return result;
}

bool InferfluxCudaExecutor::EnsureLaneDispatcherStarted() {
  std::lock_guard<std::mutex> lock(lane_dispatcher_mutex_);
  if (lane_dispatcher_.IsRunning()) {
    return true;
  }
  const bool started = lane_dispatcher_.Start(
      [this](const std::vector<UnifiedBatchInput> &captured_inputs,
             bool decode_lane) -> UnifiedBatchLaneDispatcher::ExecutionResult {
        UnifiedBatchLaneDispatcher::ExecutionResult dispatch_result;
        auto lane_result =
            ExecuteLaneBatchForAsync(captured_inputs, decode_lane);
        dispatch_result.success = true;
        dispatch_result.outputs = std::move(lane_result.outputs);
        dispatch_result.elapsed_ms = lane_result.elapsed_ms;
        return dispatch_result;
      });
  if (started) {
    GlobalMetrics().RecordCudaLaneWorkerRestart();
  }
  return started;
}

void InferfluxCudaExecutor::StopLaneDispatcher() {
  std::lock_guard<std::mutex> lock(lane_dispatcher_mutex_);
  lane_dispatcher_.Stop();
}

std::vector<LlamaCppBackend::UnifiedBatchOutput>
InferfluxCudaExecutor::ExecuteUnifiedBatchWithOverlap(
    const std::vector<UnifiedBatchInput> &inputs) {
  std::vector<UnifiedBatchOutput> outputs;
  outputs.resize(inputs.size());

  // Split batch into prefill and decode subsets
  std::vector<size_t> prefill_indices;
  std::vector<size_t> decode_indices;
  SplitBatchByType(inputs, prefill_indices, decode_indices);

  // Check if prefill is large enough to warrant overlap
  int total_prefill_tokens = 0;
  for (size_t idx : prefill_indices) {
    total_prefill_tokens += static_cast<int>(inputs[idx].tokens.size());
  }

  if (total_prefill_tokens < min_prefill_tokens_ || prefill_indices.empty() ||
      decode_indices.empty()) {
    // Fall back to standard execution
    log::Info("inferflux_cuda_executor",
              "Skipping overlap: prefill_tokens=" +
                  std::to_string(total_prefill_tokens) +
                  ", min=" + std::to_string(min_prefill_tokens_) +
                  ", prefill_count=" + std::to_string(prefill_indices.size()) +
                  ", decode_count=" + std::to_string(decode_indices.size()));
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }

  if (!EnsureLaneOverlapResources() || !CanRunLaneOverlap()) {
    log::Info("inferflux_cuda_executor",
              "Skipping overlap: lane replicas unavailable");
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }

  log::Info("inferflux_cuda_executor",
            "Using async overlap for mixed batch (prefill=" +
                std::to_string(prefill_indices.size()) +
                ", decode=" + std::to_string(decode_indices.size()) +
                ", prefill_tokens=" + std::to_string(total_prefill_tokens) +
                ")");

  std::vector<UnifiedBatchInput> decode_inputs;
  std::vector<UnifiedBatchInput> prefill_inputs;
  decode_inputs.reserve(decode_indices.size());
  prefill_inputs.reserve(prefill_indices.size());
  for (size_t idx : decode_indices) {
    decode_inputs.push_back(inputs[idx]);
  }
  for (size_t idx : prefill_indices) {
    prefill_inputs.push_back(inputs[idx]);
  }

  if (!EnsureLaneDispatcherStarted()) {
    log::Warn("inferflux_cuda_executor",
              "Lane dispatcher unavailable; falling back to single-lane "
              "execution");
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }

  const UnifiedBatchHandle decode_handle =
      lane_dispatcher_.Submit(decode_inputs, /*decode_lane=*/true);
  if (decode_handle == 0) {
    GlobalMetrics().RecordCudaLaneEnqueueReject(/*decode_lane=*/true);
    GlobalMetrics().SetCudaLaneQueueDepth(
        /*decode_lane=*/true,
        static_cast<int>(lane_dispatcher_.PendingCount(/*decode_lane=*/true)));
    log::Warn("inferflux_cuda_executor",
              "Decode lane submission rejected; falling back to single-lane "
              "execution");
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }
  GlobalMetrics().RecordCudaLaneSubmission(/*decode_lane=*/true);
  GlobalMetrics().SetCudaLaneQueueDepth(
      /*decode_lane=*/true,
      static_cast<int>(lane_dispatcher_.PendingCount(/*decode_lane=*/true)));

  const UnifiedBatchHandle prefill_handle =
      lane_dispatcher_.Submit(prefill_inputs, /*decode_lane=*/false);
  if (prefill_handle == 0) {
    GlobalMetrics().RecordCudaLaneEnqueueReject(/*decode_lane=*/false);
    GlobalMetrics().SetCudaLaneQueueDepth(
        /*decode_lane=*/false,
        static_cast<int>(lane_dispatcher_.PendingCount(/*decode_lane=*/false)));
    std::vector<UnifiedBatchOutput> drained_decode_outputs;
    bool drained_decode_lane = true;
    std::string drained_error;
    double drained_elapsed_ms = 0.0;
    bool drained_timed_out = false;
    (void)WaitForLaneHandle(&lane_dispatcher_, decode_handle,
                            &drained_decode_outputs, &drained_decode_lane,
                            &drained_error, &drained_elapsed_ms,
                            &drained_timed_out);
    GlobalMetrics().RecordCudaLaneCompletion(/*decode_lane=*/true);
    GlobalMetrics().SetCudaLaneQueueDepth(
        /*decode_lane=*/true,
        static_cast<int>(lane_dispatcher_.PendingCount(/*decode_lane=*/true)));
    log::Warn("inferflux_cuda_executor",
              "Prefill lane submission rejected; falling back to single-lane "
              "execution");
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }
  GlobalMetrics().RecordCudaLaneSubmission(/*decode_lane=*/false);
  GlobalMetrics().SetCudaLaneQueueDepth(
      /*decode_lane=*/false,
      static_cast<int>(lane_dispatcher_.PendingCount(/*decode_lane=*/false)));

  std::vector<UnifiedBatchOutput> decode_outputs;
  std::vector<UnifiedBatchOutput> prefill_outputs;
  bool decode_lane = true;
  bool prefill_lane = false;
  std::string decode_error;
  std::string prefill_error;
  double decode_elapsed_ms = 0.0;
  double prefill_elapsed_ms = 0.0;
  bool decode_timed_out = false;
  bool prefill_timed_out = false;
  const bool decode_ok = WaitForLaneHandle(
      &lane_dispatcher_, decode_handle, &decode_outputs, &decode_lane,
      &decode_error, &decode_elapsed_ms, &decode_timed_out);
  GlobalMetrics().RecordCudaLaneCompletion(decode_lane);
  GlobalMetrics().SetCudaLaneQueueDepth(
      decode_lane, static_cast<int>(lane_dispatcher_.PendingCount(
                       /*decode_lane=*/decode_lane)));
  const bool prefill_ok = WaitForLaneHandle(
      &lane_dispatcher_, prefill_handle, &prefill_outputs, &prefill_lane,
      &prefill_error, &prefill_elapsed_ms, &prefill_timed_out);
  GlobalMetrics().RecordCudaLaneCompletion(prefill_lane);
  GlobalMetrics().SetCudaLaneQueueDepth(
      prefill_lane, static_cast<int>(lane_dispatcher_.PendingCount(
                        /*decode_lane=*/prefill_lane)));

  if (!decode_ok || !prefill_ok) {
    if (!decode_ok) {
      if (decode_timed_out) {
        GlobalMetrics().RecordCudaLaneCollectTimeout(decode_lane);
      }
      log::Error("inferflux_cuda_executor",
                 "Decode lane overlap execution failed: " + decode_error);
    }
    if (!prefill_ok) {
      if (prefill_timed_out) {
        GlobalMetrics().RecordCudaLaneCollectTimeout(prefill_lane);
      }
      log::Error("inferflux_cuda_executor",
                 "Prefill lane overlap execution failed: " + prefill_error);
    }
    return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/false);
  }

  for (auto &output : outputs) {
    output.ok = false;
    output.token = -1;
    output.piece.clear();
  }
  for (size_t i = 0; i < decode_indices.size() && i < decode_outputs.size();
       ++i) {
    outputs[decode_indices[i]] = std::move(decode_outputs[i]);
  }
  for (size_t i = 0; i < prefill_indices.size() && i < prefill_outputs.size();
       ++i) {
    outputs[prefill_indices[i]] = std::move(prefill_outputs[i]);
  }

  const double total_sequential_ms = decode_elapsed_ms + prefill_elapsed_ms;
  const double actual_concurrent_ms =
      std::max(decode_elapsed_ms, prefill_elapsed_ms);
  const double overlap_ms = total_sequential_ms - actual_concurrent_ms;
  if (overlap_ms > 0.0) {
    GlobalMetrics().RecordCudaLaneOverlap(overlap_ms);
    const int reduction_pct =
        total_sequential_ms > 0.0
            ? static_cast<int>((100.0 * overlap_ms) / total_sequential_ms)
            : 0;
    log::Info("inferflux_cuda_executor",
              "Phase overlap: " + std::to_string(overlap_ms) + "ms saved (" +
                  std::to_string(reduction_pct) + "% reduction)");
  }

  return outputs;
}

#endif // INFERFLUX_NATIVE_KERNELS_READY

std::vector<LlamaCppBackend::UnifiedBatchOutput>
InferfluxCudaExecutor::ExecuteUnifiedBatch(
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &inputs) {
  return ExecuteUnifiedBatch(inputs, /*allow_overlap=*/true);
}

bool InferfluxCudaExecutor::NativeSupportsUnifiedBatchBurst() const {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_loaded_ || !model_forward_ || !sampler_ || !tokenizer_ ||
      !kv_cache_) {
    return false;
  }
  const auto &policy = execution_policy_;
  // Debug traces walk the per-step path; keep them on it so trace tooling
  // (decode_mapping, debug_logits) keeps working.
  if (policy.debug_decode_mapping || policy.debug_logits) {
    return false;
  }
  return policy.enable_decode_burst && policy.enable_batched_decode;
#else
  return false;
#endif
}

UnifiedBurstResult InferfluxCudaExecutor::NativeExecuteUnifiedBatchBurst(
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &inputs,
    const UnifiedBurstOptions &options, const BurstTokenSink &sink) {
  UnifiedBurstResult result;
  const std::size_t n = inputs.size();
  result.last_tokens.assign(n, -1);
  result.finished.assign(n, false);
  result.ok = false;

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!NativeSupportsUnifiedBatchBurst() || !sink || n == 0) {
    return result;
  }
  for (const auto &in : inputs) {
    if (in.tokens.size() != 1 || !in.request_logits ||
        in.sampling.temperature > 0.0f) {
      return result;
    }
  }
  const int B = static_cast<int>(n);
  const int decode_capacity = std::max(1, kv_cache_->MaxBatchSize());
  const int max_seq_len = std::max(1, kv_cache_->MaxSeqLen());
  if (B > decode_capacity ||
      B > runtime::cuda::native::DecodeBurstController::kMaxBurstBatch) {
    return result;
  }

  NVTX_SCOPE("DecodeBurst");
  diagnostics::ScopedBreadcrumb breadcrumb("ExecuteUnifiedBatchBurst");

  struct ScopedDequantCacheCleanup {
    InferfluxCudaExecutor *executor{nullptr};
    ~ScopedDequantCacheCleanup() {
      if (executor) {
        executor->ReleaseBatchScopedDequantizedCache();
      }
    }
  } scoped_cleanup{this};

  // Always serialize access to the shared GPU pipeline (matches the standard
  // ExecuteUnifiedBatch path).
  std::lock_guard<std::mutex> shared_pipeline_lock(shared_pipeline_mutex_);

  if (!burst_controller_.EnsureResources(compute_stream_)) {
    log::Warn("inferflux_cuda_executor",
              "DecodeBurst: resource allocation failed");
    return result;
  }

  // Step-0 metadata comes from the host (full BatchForward: upload + capture
  // or replay); every later step is fed on device by the token-feed kernel.
  std::vector<int> batch_tokens(B), batch_n_past(B), batch_seq_ids(B);
  for (int b = 0; b < B; ++b) {
    batch_tokens[b] = inputs[static_cast<std::size_t>(b)].tokens[0];
    batch_n_past[b] = inputs[static_cast<std::size_t>(b)].n_past;
    batch_seq_ids[b] = inputs[static_cast<std::size_t>(b)].sequence_id;
  }

  // Per-sequence step budget: policy chunk, caller cap, batch-wide token
  // cap, and KV headroom (kv_lens reaches n_past + steps).
  int k_chunk =
      std::min(options.max_tokens_per_seq, burst_controller_.max_slots());
  k_chunk =
      std::min(k_chunk, std::max(1, execution_policy_.decode_burst_chunk));
  k_chunk = std::min(k_chunk, std::max(1, options.max_batch_tokens / B));
  std::vector<int> step_budget(B, 1);
  for (int b = 0; b < B; ++b) {
    const int headroom = std::max(1, max_seq_len - batch_n_past[b] - 1);
    step_budget[b] = std::max(1, std::min(k_chunk, headroom));
  }

  const int eos_ids[1] = {tokenizer_->EosTokenId()};
  if (!burst_controller_.BeginChunk(step_budget.data(), B, eos_ids, 1)) {
    return result;
  }

  const bool record_timing =
      ShouldRecordTimingSample(timing_sample_rate_, &timing_batch_counter_);
  if (record_timing) {
    cudaEventRecord(forward_start_, compute_stream_);
  }

  // Step 0: host-fed forward (uploads metadata; captures the decode graph on
  // first use for this batch size).
  if (!model_forward_->BatchForward(batch_tokens, batch_n_past, batch_seq_ids,
                                    d_logits_, B)) {
    log::Error("inferflux_cuda_executor", "DecodeBurst: step-0 forward failed");
    return result;
  }
  if (!burst_controller_.EnqueueStepEpilogue(
          /*slot_idx=*/0, model_forward_.get(), sampler_.get(), d_logits_, B)) {
    log::Error("inferflux_cuda_executor",
               "DecodeBurst: step-0 epilogue failed");
    return result;
  }

  // Without a replayable decode graph every step would cost a full kernel
  // launch storm from the host, defeating the point of the burst — degrade
  // to a single step and let the caller drive the next one.
  const int steps =
      model_forward_->DecodeGraphReady(B)
          ? *std::max_element(step_budget.begin(), step_budget.end())
          : 1;
  int enqueued = 1;
  for (int i = 1; i < steps; ++i) {
    if (!burst_controller_.EnqueueStep(i, model_forward_.get(), sampler_.get(),
                                       d_logits_, B)) {
      log::Error("inferflux_cuda_executor", "DecodeBurst: step enqueue failed");
      // Consume whatever was enqueued so far; treat this as a short chunk.
      break;
    }
    ++enqueued;
  }

  if (record_timing) {
    cudaEventRecord(forward_stop_, compute_stream_);
  }

  // Consume slots in order while the GPU runs ahead. Sequences already
  // stopped (EOS / sink stop) are skipped; their replayed steps are wasted
  // compute only.
  std::vector<bool> stopped(static_cast<std::size_t>(B), false);
  for (int i = 0; i < enqueued; ++i) {
    while (!burst_controller_.PollSlot(i)) {
      std::this_thread::yield();
    }
    bool any_active = false;
    for (int b = 0; b < B; ++b) {
      if (stopped[static_cast<std::size_t>(b)]) {
        continue;
      }
      const int token_id = burst_controller_.ReadSlot(i, b);
      if (token_id < 0 || tokenizer_->IsTerminalGeneratedToken(token_id)) {
        // Terminal token: matches the per-step path — emit nothing, finish
        // the request, never forward the terminal token.
        result.last_tokens[static_cast<std::size_t>(b)] = -1;
        result.finished[static_cast<std::size_t>(b)] = true;
        stopped[static_cast<std::size_t>(b)] = true;
        burst_controller_.FreezeSequence(b);
        continue;
      }
      const std::string piece = VisibleTokenPiece(tokenizer_.get(), token_id);
      if (!piece.empty()) {
        perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
      }
      result.last_tokens[static_cast<std::size_t>(b)] = token_id;
      BurstTokenEvent ev;
      ev.input_idx = static_cast<std::size_t>(b);
      ev.token_id = token_id;
      ev.piece = piece;
      if (!sink(ev)) {
        result.finished[static_cast<std::size_t>(b)] = true;
        stopped[static_cast<std::size_t>(b)] = true;
        burst_controller_.FreezeSequence(b);
      }
    }
    for (int b = 0; b < B; ++b) {
      if (!stopped[static_cast<std::size_t>(b)]) {
        any_active = true;
      }
    }
    if (!any_active) {
      break;
    }
  }

  burst_controller_.FinishChunk(enqueued - 1);

  if (record_timing) {
    float chunk_ms = 0.0f;
    if (CheckCudaStatus(
            cudaEventElapsedTime(&chunk_ms, forward_start_, forward_stop_),
            "cudaEventElapsedTime(decode_burst_chunk)")) {
      GlobalMetrics().RecordInferfluxCudaForwardLatency(chunk_ms);
      perf_accum_.decode_ms.store(
          perf_accum_.decode_ms.load(std::memory_order_relaxed) + chunk_ms,
          std::memory_order_relaxed);
    }
  }
  GlobalMetrics().RecordInferfluxCudaForwardShape(/*is_decode=*/true, B);
  MaybeRefreshMemoryLedger();
  result.ok = true;
#endif // INFERFLUX_NATIVE_KERNELS_READY
  return result;
}

std::vector<LlamaCppBackend::UnifiedBatchOutput>
InferfluxCudaExecutor::ExecuteUnifiedBatch(
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &inputs,
    bool allow_overlap) {
  if (!model_loaded_) {
    log::Error("inferflux_cuda_executor", "Model not loaded");
    return {};
  }

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_) {
    log::Warn("inferflux_cuda_executor",
              "InferFlux CUDA pipeline not initialized");
    return {};
  }

  struct ScopedDequantCacheCleanup {
    InferfluxCudaExecutor *executor{nullptr};
    ~ScopedDequantCacheCleanup() {
      if (executor) {
        executor->ReleaseBatchScopedDequantizedCache();
      }
    }
  } scoped_cleanup{this};

  NVTX_SCOPE("NativeExecuteUnifiedBatch");
  diagnostics::ScopedBreadcrumb breadcrumb("ExecuteUnifiedBatch");

  // Check for mixed workload and use overlap path if enabled
  if (allow_overlap && HasMixedWorkload(inputs)) {
    auto outputs = ExecuteUnifiedBatchWithOverlap(inputs);
    MaybeRefreshMemoryLedger();
    return outputs;
  }

  // Always serialize access to the shared GPU pipeline (model_forward_,
  // d_logits_, sampler_) in the standard (non-overlap) execution path.
  // Without this lock, concurrent ExecuteUnifiedBatch() calls from the
  // scheduler race on shared device buffers, causing segfaults at c≥8.
  std::lock_guard<std::mutex> shared_pipeline_lock(shared_pipeline_mutex_);

  // Standard execution path for non-mixed workloads
  std::vector<UnifiedBatchOutput> outputs;
  outputs.resize(inputs.size());

  // Partition inputs into decode (tokens.size()==1, request_logits) and others
  struct DecodeEntry {
    int input_idx;
    int64_t request_id;
    std::string client_request_id;
    int token_id;
    int n_past;
    int sequence_id;
    uint64_t sequence_generation;
    float temperature;
    int top_k;
    float top_p;
    uint32_t seed;
    SamplingParams sampling;
  };
  std::vector<DecodeEntry> decode_group;
  std::vector<int> prefill_indices;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    if (input.tokens.size() == 1 && input.request_logits) {
      decode_group.push_back(
          {static_cast<int>(i), input.request_id, input.client_request_id,
           input.tokens[0], input.n_past, input.sequence_id,
           input.sequence_generation, input.sampling.temperature,
           input.sampling.top_k, input.sampling.top_p, input.sampling.seed,
           input.sampling});
    } else {
      prefill_indices.push_back(static_cast<int>(i));
    }
  }

  // Invalidate device-side relay if any prefills are pending (they change
  // KV cache state and sequence positions, making the relay's on-device
  // metadata stale).
  if (!prefill_indices.empty()) {
    decode_relay_active_ = false;
  }

  // === Batched decode group ===
  const int decode_batch_capacity =
      kv_cache_ ? std::max(1, kv_cache_->MaxBatchSize()) : 32;
  if (!decode_group.empty()) {
    // Pre-allocate batch vectors once at maximum capacity to avoid per-batch
    // heap allocations in the hot loop.
    const int max_B = static_cast<int>(std::min(
        decode_group.size(), static_cast<size_t>(decode_batch_capacity)));
    std::vector<int> batch_tokens(max_B);
    std::vector<int> batch_n_past(max_B);
    std::vector<int> batch_seq_ids(max_B);
    std::vector<float> batch_temps(max_B);
    std::vector<int> batch_top_ks(max_B);
    std::vector<float> batch_top_ps(max_B);
    std::vector<uint32_t> batch_seeds(max_B);
    std::vector<int> sampled_tokens;

    for (size_t offset = 0; offset < decode_group.size();
         offset += static_cast<size_t>(decode_batch_capacity)) {
      NVTX_SCOPE("BatchedDecode");
      const int B = static_cast<int>(
          std::min(decode_group.size() - offset,
                   static_cast<size_t>(decode_batch_capacity)));

      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        batch_tokens[b] = entry.token_id;
        batch_n_past[b] = entry.n_past;
        batch_seq_ids[b] = entry.sequence_id;
        batch_temps[b] = entry.temperature;
        batch_top_ks[b] = entry.top_k;
        batch_top_ps[b] = entry.top_p;
        batch_seeds[b] = entry.seed;
      }

      // Record forward timing based on sample rate.
      // rate=0: never record (pure throughput). rate=N: every Nth batch.
      const bool record_timing =
          ShouldRecordTimingSample(timing_sample_rate_, &timing_batch_counter_);
      if (record_timing) {
        cudaEventRecord(forward_start_, compute_stream_);
      }

      // Device-side token relay: if the relay is active and batch config
      // matches, replay the CUDA graph without H2D metadata upload.
      // DeviceTokenRelay already updated the graph input buffers on device
      // during the previous token's post-sampling step.
      bool fwd_ok = false;
      if (decode_relay_active_ && decode_relay_batch_size_ == B) {
        fwd_ok = model_forward_->BatchForwardReplay(d_logits_, B);
        if (!fwd_ok) {
          // Graph replay failed — fall back to full BatchForward
          decode_relay_active_ = false;
        }
      }
      if (!fwd_ok) {
        fwd_ok = model_forward_->BatchForward(batch_tokens, batch_n_past,
                                              batch_seq_ids, d_logits_, B);
        decode_relay_active_ = false;
      }
      if (record_timing) {
        cudaEventRecord(forward_stop_, compute_stream_);
      }

      if (!fwd_ok) {
        log::Error("inferflux_cuda_executor", "BatchForward failed");
        for (int b = 0; b < B; ++b) {
          const auto &entry = decode_group[offset + static_cast<size_t>(b)];
          outputs[entry.input_idx].ok = false;
          outputs[entry.input_idx].token = -1;
        }
        decode_relay_active_ = false;
        continue;
      }

      // Apply per-sequence repetition penalties before batch sampling.
      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        const auto &sp = entry.sampling;
        float eff_rep = sp.repetition_penalty;
        float eff_freq = sp.frequency_penalty;
        if (sp.temperature == 0.0f && eff_rep == 1.0f && eff_freq == 0.0f) {
          eff_rep = 1.15f;
        }
        if (eff_rep == 1.0f && eff_freq == 0.0f && sp.presence_penalty == 0.0f)
          continue;
        auto it = sequence_recent_tokens_.find(entry.sequence_id);
        if (it == sequence_recent_tokens_.end() || it->second.empty())
          continue;
        std::unordered_map<int, int> freq;
        for (int t : it->second)
          freq[t]++;
        std::vector<int> ids, counts;
        ids.reserve(freq.size());
        counts.reserve(freq.size());
        for (auto &[id, cnt] : freq) {
          ids.push_back(id);
          counts.push_back(cnt);
        }
        float *seq_logits = d_logits_ + b * model_config_.vocab_size;
        sampler_->ApplyPenalties(seq_logits, ids, counts, eff_rep, eff_freq,
                                 sp.presence_penalty);
      }
      sampler_->EnqueueSampleBatch(d_logits_, B, batch_temps, batch_top_ks,
                                   batch_top_ps, batch_seeds);
      sampler_->CollectSampleBatch(&sampled_tokens);
      // Record sampled tokens in per-sequence histories.
      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        auto &history = sequence_recent_tokens_[entry.sequence_id];
        history.push_back(sampled_tokens[b]);
        if (static_cast<int>(history.size()) > kPenaltyWindowSize) {
          history.erase(history.begin());
        }
      }

      // Set up device-side relay for the NEXT decode token: copy sampled
      // token IDs directly to the graph's input buffer on device and
      // increment n_past, avoiding the H2D metadata upload next iteration.
      if (model_forward_->GetBatchMetaDevice() &&
          model_forward_->GetMaxBatchSize() > 0) {
        cuda_kernel::DeviceTokenRelay(
            sampler_->DeviceResultBatch(), model_forward_->GetBatchMetaDevice(),
            B, model_forward_->GetMaxBatchSize(), compute_stream_);
        decode_relay_active_ = true;
        decode_relay_batch_size_ = B;
      }

#ifdef INFERFLUX_NATIVE_KERNELS_READY
      if (NativeLogitsDebugEnabled(execution_policy_)) {
        for (int b = 0; b < B; ++b) {
          const auto &entry = decode_group[offset + static_cast<size_t>(b)];
          LogNativeTopLogits(
              "primary", d_logits_ + b * model_config_.vocab_size,
              model_config_.vocab_size, entry.request_id,
              entry.client_request_id, entry.sequence_id,
              entry.sequence_generation, entry.n_past, execution_policy_);
        }
      }
#endif

      if (DecodeMappingDebugEnabled(execution_policy_)) {
        log::Info("inferflux_cuda_executor",
                  "decode_mapping[primary]: batch_size=" + std::to_string(B) +
                      ", batch_offset=" + std::to_string(offset));
        for (int b = 0; b < B; ++b) {
          if (!ConsumeDecodeMappingBudget(execution_policy_)) {
            break;
          }
          const auto &entry = decode_group[offset + static_cast<size_t>(b)];
          const int token_id = sampled_tokens[b];
          const std::string piece = (tokenizer_ && token_id >= 0)
                                        ? tokenizer_->TokenToString(token_id)
                                        : std::string();
          log::Info("inferflux_cuda_executor",
                    "decode_mapping[primary]: input_idx=" +
                        std::to_string(entry.input_idx) + ", " +
                        (entry.client_request_id.empty()
                             ? std::string()
                             : "client_request_id=" + entry.client_request_id +
                                   ", ") +
                        "request_id=" + std::to_string(entry.request_id) +
                        ", sequence_id=" + std::to_string(entry.sequence_id) +
                        ", sequence_generation=" +
                        std::to_string(entry.sequence_generation) +
                        ", n_past=" + std::to_string(entry.n_past) +
                        ", sampled_token=" + std::to_string(token_id) +
                        ", piece=" + piece);
        }
      }

      GlobalMetrics().RecordInferfluxCudaForwardShape(/*is_decode=*/true, B);
      if (record_timing) {
        float fwd_ms = 0.0f;
        if (CheckCudaStatus(
                cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
                "cudaEventElapsedTime(forward,decode_batch)")) {
          GlobalMetrics().RecordInferfluxCudaForwardLatency(fwd_ms);
          perf_accum_.decode_ms.store(
              perf_accum_.decode_ms.load(std::memory_order_relaxed) + fwd_ms,
              std::memory_order_relaxed);
        }
      }

      for (int b = 0; b < B; ++b) {
        const auto &entry = decode_group[offset + static_cast<size_t>(b)];
        int token_id = sampled_tokens[b];
        if (tokenizer_ && tokenizer_->IsTerminalGeneratedToken(token_id)) {
          outputs[entry.input_idx].token = -1;
          outputs[entry.input_idx].piece = "";
        } else {
          outputs[entry.input_idx].token = token_id;
          outputs[entry.input_idx].piece =
              VisibleTokenPiece(tokenizer_.get(), token_id);
          if (!outputs[entry.input_idx].piece.empty()) {
            perf_accum_.generated_tokens.fetch_add(1,
                                                   std::memory_order_relaxed);
          }
        }
        outputs[entry.input_idx].ok = true;
      }
    }
  }

  // === Prefill group (sequential, variable lengths) ===
  // Clear KV cache slots for new prefill requests (n_past==0) to prevent
  // stale data from a prior sequence occupant from corrupting attention.
  for (int idx : prefill_indices) {
    const auto &pi = inputs[idx];
    if (pi.n_past == 0 && pi.sequence_id >= 0 && kv_cache_) {
      kv_cache_->ClearSequenceAsync(pi.sequence_id, compute_stream_);
    }
  }

  for (int idx : prefill_indices) {
    const auto &input = inputs[idx];
    UnifiedBatchOutput &output = outputs[idx];
    output.ok = false;
    output.token = -1;

    bool is_decode = (input.tokens.size() == 1);
    int batch_tokens = static_cast<int>(input.tokens.size());

    const bool record_prefill_timing =
        ShouldRecordTimingSample(timing_sample_rate_, &timing_batch_counter_);

    if (!input.request_logits) {
      NVTX_SCOPE("ForwardPass");
      if (record_prefill_timing) {
        cudaEventRecord(forward_start_, compute_stream_);
      }
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("inferflux_cuda_executor", "Forward pass failed");
      }
      if (record_prefill_timing) {
        cudaEventRecord(forward_stop_, compute_stream_);
        runtime::cuda::native::TracedCudaEventSynchronize(
            runtime::cuda::native::SyncTraceSite::kTimingForwardReady,
            forward_stop_);
        GlobalMetrics().RecordInferfluxCudaForwardShape(is_decode,
                                                        batch_tokens);
        float fwd_ms = 0.0f;
        if (CheckCudaStatus(
                cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
                "cudaEventElapsedTime(forward,prefill_no_logits)")) {
          GlobalMetrics().RecordInferfluxCudaForwardLatency(fwd_ms);
          perf_accum_.prefill_ms.store(
              perf_accum_.prefill_ms.load(std::memory_order_relaxed) + fwd_ms,
              std::memory_order_relaxed);
        }
      } else {
        // No timing — just sync stream to ensure forward completes
        runtime::cuda::native::TracedCudaStreamSynchronize(
            runtime::cuda::native::SyncTraceSite::kPrefillForwardDrain,
            compute_stream_);
        GlobalMetrics().RecordInferfluxCudaForwardShape(is_decode,
                                                        batch_tokens);
      }
      perf_accum_.prompt_tokens.fetch_add(batch_tokens,
                                          std::memory_order_relaxed);
      output.ok = true;
      continue;
    }

    // Forward pass
    {
      NVTX_SCOPE("ForwardPass");
      if (record_prefill_timing) {
        cudaEventRecord(forward_start_, compute_stream_);
      }
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("inferflux_cuda_executor", "Forward pass failed");
        continue;
      }
      if (record_prefill_timing) {
        cudaEventRecord(forward_stop_, compute_stream_);
      }
    }

    // Sample
    {
      NVTX_SCOPE("Sampling");
      if (record_prefill_timing) {
        cudaEventRecord(sampling_start_, compute_stream_);
      }
      // Apply repetition/frequency/presence penalties before sampling.
      // When greedy decoding (temperature==0) and no explicit penalty is set,
      // apply a minimal default penalty to prevent degenerate repetition
      // loops that occur with some instruct-tuned models.
      {
        auto &history = sequence_recent_tokens_[input.sequence_id];
        float eff_rep_penalty = input.sampling.repetition_penalty;
        float eff_freq_penalty = input.sampling.frequency_penalty;
        if (input.sampling.temperature == 0.0f && eff_rep_penalty == 1.0f &&
            eff_freq_penalty == 0.0f) {
          eff_rep_penalty =
              1.15f; // Default penalty for greedy decode to prevent loops
        }
        if (!history.empty() &&
            (eff_rep_penalty != 1.0f || eff_freq_penalty != 0.0f ||
             input.sampling.presence_penalty != 0.0f)) {
          // Build unique token list with frequency counts
          std::unordered_map<int, int> freq;
          for (int t : history)
            freq[t]++;
          std::vector<int> ids, counts;
          ids.reserve(freq.size());
          counts.reserve(freq.size());
          for (auto &[id, cnt] : freq) {
            ids.push_back(id);
            counts.push_back(cnt);
          }
          sampler_->ApplyPenalties(d_logits_, ids, counts, eff_rep_penalty,
                                   eff_freq_penalty,
                                   input.sampling.presence_penalty);
        }
      }
      sampler_->EnqueueSample(d_logits_, input.sampling.temperature,
                              input.sampling.top_k, input.sampling.top_p,
                              input.sampling.seed);
      int token_id = sampler_->CollectSample();
      // Record token in per-sequence history for future penalty application
      {
        auto &history = sequence_recent_tokens_[input.sequence_id];
        history.push_back(token_id);
        if (static_cast<int>(history.size()) > kPenaltyWindowSize) {
          history.erase(history.begin());
        }
      }
#ifdef INFERFLUX_NATIVE_KERNELS_READY
      LogNativeTopLogits("primary_prefill", d_logits_, model_config_.vocab_size,
                         input.request_id, input.client_request_id,
                         input.sequence_id, input.sequence_generation,
                         input.n_past, execution_policy_);
#endif
      if (record_prefill_timing) {
        cudaEventRecord(sampling_stop_, compute_stream_);
      }

      // Deferred timing: compute elapsed from already-completed events
      GlobalMetrics().RecordInferfluxCudaForwardShape(is_decode, batch_tokens);
      if (record_prefill_timing) {
        float fwd_ms = 0.0f;
        if (CheckCudaStatus(
                cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
                "cudaEventElapsedTime(forward,prefill_logits)")) {
          GlobalMetrics().RecordInferfluxCudaForwardLatency(fwd_ms);
          if (is_decode) {
            perf_accum_.decode_ms.store(
                perf_accum_.decode_ms.load(std::memory_order_relaxed) + fwd_ms,
                std::memory_order_relaxed);
          } else {
            perf_accum_.prefill_ms.store(
                perf_accum_.prefill_ms.load(std::memory_order_relaxed) + fwd_ms,
                std::memory_order_relaxed);
            perf_accum_.prompt_tokens.fetch_add(batch_tokens,
                                                std::memory_order_relaxed);
          }
        }
        float samp_ms = 0.0f;
        if (CheckCudaStatus(
                runtime::cuda::native::TracedCudaEventSynchronize(
                    runtime::cuda::native::SyncTraceSite::kTimingSamplingReady,
                    sampling_stop_),
                "cudaEventSynchronize(sampling_stop_)") &&
            CheckCudaStatus(
                cudaEventElapsedTime(&samp_ms, sampling_start_, sampling_stop_),
                "cudaEventElapsedTime(sampling,prefill_logits)")) {
          GlobalMetrics().RecordInferfluxCudaSampling(1, samp_ms);
        }
      } // end if (record_prefill_timing)

      if (tokenizer_ && tokenizer_->IsTerminalGeneratedToken(token_id)) {
        output.token = -1;
        output.piece = "";
      } else {
        output.token = token_id;
        output.piece = VisibleTokenPiece(tokenizer_.get(), token_id);
        if (!output.piece.empty()) {
          perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
        }
      }
      LogSampleMapping("sample_mapping[primary_prefill]", idx, input.request_id,
                       input.client_request_id, input.sequence_id,
                       input.sequence_generation, input.n_past, batch_tokens,
                       token_id, tokenizer_.get(), execution_policy_);
      output.ok = true;
    }
  }

  MaybeRefreshMemoryLedger();
  return outputs;
#else
  log::Warn("inferflux_cuda_executor",
            "Native inference not compiled (INFERFLUX_NATIVE_KERNELS_READY=0)");
  return {};
#endif
}

bool InferfluxCudaExecutor::SupportsAsyncUnifiedBatch() const {
  // Disable async dispatch to use the synchronous ExecuteUnifiedBatch path
  // which supports true batching (BatchForward + SampleBatch). The async
  // path routes every decode step through the lane dispatcher, adding
  // per-step overhead from queue operations, condition variable signaling,
  // and thread context switches that accumulate to a 2x slowdown.
  // Phase overlap for mixed prefill/decode workloads is handled internally
  // by ExecuteUnifiedBatchWithOverlap when called from the sync path.
  return false;
}

UnifiedBatchHandle InferfluxCudaExecutor::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane lane) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_ || inputs.empty()) {
    return 0;
  }
  if (!EnsureLaneDispatcherStarted()) {
    return 0;
  }

  const bool is_decode =
      (lane == UnifiedBatchLane::kDecode) ||
      (lane == UnifiedBatchLane::kAuto &&
       std::all_of(inputs.begin(), inputs.end(),
                   [](const UnifiedBatchInput &in) {
                     return in.tokens.size() == 1 && in.request_logits;
                   }));

  const UnifiedBatchHandle handle = lane_dispatcher_.Submit(inputs, is_decode);
  if (handle == 0) {
    GlobalMetrics().RecordCudaLaneEnqueueReject(is_decode);
    GlobalMetrics().SetCudaLaneQueueDepth(
        is_decode, static_cast<int>(lane_dispatcher_.PendingCount(is_decode)));
    return 0;
  }

  GlobalMetrics().RecordCudaLaneSubmission(is_decode);
  GlobalMetrics().SetCudaLaneQueueDepth(
      is_decode, static_cast<int>(lane_dispatcher_.PendingCount(is_decode)));
  return handle;
#else
  (void)inputs;
  (void)lane;
  return 0;
#endif
}

bool InferfluxCudaExecutor::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (handle == 0) {
    return false;
  }
  bool decode_lane = false;
  std::string error;
  auto status =
      lane_dispatcher_.TryCollect(handle, outputs, &decode_lane, &error);
  if (status == UnifiedBatchLaneDispatcher::CollectStatus::kPending ||
      status == UnifiedBatchLaneDispatcher::CollectStatus::kMissing) {
    return false;
  }

  GlobalMetrics().RecordCudaLaneCompletion(decode_lane);
  GlobalMetrics().SetCudaLaneQueueDepth(
      decode_lane,
      static_cast<int>(lane_dispatcher_.PendingCount(decode_lane)));
  if (status == UnifiedBatchLaneDispatcher::CollectStatus::kFailed) {
    log::Error("inferflux_cuda_executor",
               "Async lane execution failed: " + error);
    return false;
  }
  return true;
#else
  (void)handle;
  (void)outputs;
  return false;
#endif
}

bool InferfluxCudaExecutor::RunNativeInference(
    const std::vector<UnifiedBatchInput> &inputs,
    std::vector<UnifiedBatchOutput> *outputs) {
  auto results = ExecuteUnifiedBatch(inputs);
  const bool has_results = !results.empty();
  if (outputs) {
    *outputs = std::move(results);
  }
  return has_results;
}

#else // !INFERFLUX_HAS_CUDA

// CPU-only stubs for InferfluxCudaExecutor CUDA-dependent methods.
bool InferfluxCudaExecutor::LoadModel(const std::filesystem::path &,
                                      const LlamaBackendConfig &) {
  log::Error("inferflux_cuda_executor", "CUDA not available");
  return false;
}

std::vector<LlamaCppBackend::UnifiedBatchOutput>
InferfluxCudaExecutor::ExecuteUnifiedBatch(
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &) {
  return {};
}

bool InferfluxCudaExecutor::SupportsAsyncUnifiedBatch() const { return false; }

bool InferfluxCudaExecutor::NativeSupportsUnifiedBatchBurst() const {
  return false;
}

UnifiedBurstResult InferfluxCudaExecutor::NativeExecuteUnifiedBatchBurst(
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &,
    const UnifiedBurstOptions &, const BurstTokenSink &) {
  return UnifiedBurstResult{};
}

UnifiedBatchHandle InferfluxCudaExecutor::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &, UnifiedBatchLane) {
  return 0;
}

bool InferfluxCudaExecutor::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle, std::vector<UnifiedBatchOutput> *) {
  return false;
}

bool InferfluxCudaExecutor::RunNativeInference(
    const std::vector<UnifiedBatchInput> &, std::vector<UnifiedBatchOutput> *) {
  return false;
}

bool InferfluxCudaExecutor::ConfigureDequantizedCachePolicy(
    const std::string &raw_policy) {
  std::string policy = ToLowerAscii(raw_policy);
  if (policy.empty()) {
    policy = "none";
  }

  runtime::cuda::native::DequantizedCachePolicy parsed =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  if (!runtime::cuda::native::ParseDequantizedCachePolicy(policy, &parsed)) {
    parsed = runtime::cuda::native::DequantizedCachePolicy::kNone;
    policy = "none";
  }
  dequantized_cache_policy_ = parsed;
  dequantized_cache_policy_hint_ = policy;

  if (model_loader_) {
    model_loader_->SetDequantizedCachePolicy(dequantized_cache_policy_);
  }
  return true;
}

void InferfluxCudaExecutor::ReleaseBatchScopedDequantizedCache() {
  if (!model_loader_ || model_loader_->GetFormat() != "gguf") {
    return;
  }
  if (dequantized_cache_policy_ ==
      runtime::cuda::native::DequantizedCachePolicy::kModelLifetime) {
    return;
  }
  model_loader_->ClearDequantizedCache();
}

void InferfluxCudaExecutor::MaybeRefreshMemoryLedger() {}

#endif // INFERFLUX_HAS_CUDA

// ==========================================================================
// NativeTakePerf (works on all build paths)
// ==========================================================================

InferfluxCudaRuntime::PerfSnapshot InferfluxCudaExecutor::NativeTakePerf() {
  PerfSnapshot snap;
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  snap.prefill_ms = perf_accum_.prefill_ms.exchange(0.0);
  snap.decode_ms = perf_accum_.decode_ms.exchange(0.0);
  snap.prompt_tokens = perf_accum_.prompt_tokens.exchange(0);
  snap.generated_tokens = perf_accum_.generated_tokens.exchange(0);
#endif
  return snap;
}

bool InferfluxCudaExecutor::ShouldRecordTimingSample(
    int sample_rate, std::atomic<int> *counter) {
  if (sample_rate <= 0 || !counter) {
    return false;
  }
  int val = counter->fetch_add(1, std::memory_order_relaxed) + 1;
  return (val % sample_rate) == 0;
}

bool InferfluxCudaExecutor::ShouldRecordTimingSample(int sample_rate,
                                                     int *counter) {
  if (sample_rate <= 0 || !counter) {
    return false;
  }
  ++(*counter);
  return (*counter % sample_rate) == 0;
}

// ==========================================================================
// Native* method overrides (no CUDA dependency)
// ==========================================================================

std::vector<int>
InferfluxCudaExecutor::NativeTokenize(const std::string &prompt) const {
  if (tokenizer_) {
    return tokenizer_->Tokenize(prompt);
  }
  return {};
}

int InferfluxCudaExecutor::NativeTokenCount(const std::string &text) const {
  if (tokenizer_) {
    return static_cast<int>(tokenizer_->Tokenize(text).size());
  }
  return 0;
}

bool InferfluxCudaExecutor::NativeIsReady() const { return model_loaded_; }

void InferfluxCudaExecutor::NativeFreeSequence(int sequence_id) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!kv_cache_ || sequence_id < 0) {
    return;
  }
  // Sequence slots are reused by the scheduler. Ensure in-flight work on all
  // native streams has completed before zeroing the shared KV rows so the next
  // request never observes stale cache state from a prior occupant.
  if (compute_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(compute_stream_),
                    "cudaStreamSynchronize(compute_stream_,free_sequence)");
  }
  if (decode_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(decode_stream_),
                    "cudaStreamSynchronize(decode_stream_,free_sequence)");
  }
  if (prefill_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(prefill_stream_),
                    "cudaStreamSynchronize(prefill_stream_,free_sequence)");
  }
  kv_cache_->ClearSequence(sequence_id);
  sequence_recent_tokens_.erase(sequence_id);
#endif
}

LlamaCppBackend::SequenceReleaseFence
InferfluxCudaExecutor::NativeBeginFreeSequence(int sequence_id) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!kv_cache_ || sequence_id < 0) {
    return {};
  }

  PendingSequenceRelease pending;
  pending.sequence_id = sequence_id;
  {
    std::lock_guard<std::mutex> lock(pending_sequence_releases_mutex_);
    pending.token = next_sequence_release_token_++;
  }

  auto record_event = [&](cudaStream_t stream, const char *label,
                          cudaEvent_t *event_out) -> bool {
    if (!stream) {
      return true;
    }
    if (!CheckCudaStatus(
            cudaEventCreateWithFlags(event_out, cudaEventDisableTiming),
            std::string("cudaEventCreateWithFlags(") + label + ")")) {
      return false;
    }
    if (!CheckCudaStatus(cudaEventRecord(*event_out, stream),
                         std::string("cudaEventRecord(") + label + ")")) {
      cudaEventDestroy(*event_out);
      *event_out = nullptr;
      return false;
    }
    return true;
  };

  if (!record_event(compute_stream_, "free_sequence.compute",
                    &pending.compute_done) ||
      !record_event(decode_stream_, "free_sequence.decode",
                    &pending.decode_done) ||
      !record_event(prefill_stream_, "free_sequence.prefill",
                    &pending.prefill_done)) {
    if (pending.compute_done) {
      cudaEventDestroy(pending.compute_done);
    }
    if (pending.decode_done) {
      cudaEventDestroy(pending.decode_done);
    }
    if (pending.prefill_done) {
      cudaEventDestroy(pending.prefill_done);
    }
    NativeFreeSequence(sequence_id);
    return {};
  }

  if (!pending.compute_done && !pending.decode_done && !pending.prefill_done) {
    kv_cache_->ClearSequence(sequence_id);
    return {};
  }

  {
    std::lock_guard<std::mutex> lock(pending_sequence_releases_mutex_);
    pending_sequence_releases_.push_back(pending);
  }
  LlamaCppBackend::SequenceReleaseFence fence;
  fence.token = pending.token;
  fence.pending = true;
  return fence;
#else
  (void)sequence_id;
  return {};
#endif
}

bool InferfluxCudaExecutor::NativePollFreeSequence(
    const LlamaCppBackend::SequenceReleaseFence &fence) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!fence.pending || fence.token == 0) {
    return true;
  }

  auto ready = [&](cudaEvent_t event, const char *label) -> bool {
    if (!event) {
      return true;
    }
    cudaError_t err = cudaEventQuery(event);
    if (err == cudaSuccess) {
      return true;
    }
    if (err == cudaErrorNotReady) {
      return false;
    }
    log::Warn("inferflux_cuda_executor",
              "cudaEventQuery failed during sequence release poll: " +
                  std::string(label) +
                  " err=" + std::string(cudaGetErrorString(err)));
    cudaGetLastError();
    return true;
  };

  PendingSequenceRelease pending;
  {
    std::lock_guard<std::mutex> lock(pending_sequence_releases_mutex_);
    auto it = std::find_if(pending_sequence_releases_.begin(),
                           pending_sequence_releases_.end(),
                           [&](const PendingSequenceRelease &candidate) {
                             return candidate.token == fence.token;
                           });
    if (it == pending_sequence_releases_.end()) {
      return true;
    }
    pending = *it;
  }

  if (!ready(pending.compute_done, "compute") ||
      !ready(pending.decode_done, "decode") ||
      !ready(pending.prefill_done, "prefill")) {
    return false;
  }

  bool erased = false;
  {
    std::lock_guard<std::mutex> lock(pending_sequence_releases_mutex_);
    auto it = std::find_if(pending_sequence_releases_.begin(),
                           pending_sequence_releases_.end(),
                           [&](const PendingSequenceRelease &candidate) {
                             return candidate.token == fence.token;
                           });
    if (it != pending_sequence_releases_.end()) {
      pending = *it;
      pending_sequence_releases_.erase(it);
      erased = true;
    }
  }
  if (!erased) {
    return true;
  }

  kv_cache_->ClearSequence(pending.sequence_id);
  if (pending.compute_done) {
    cudaEventDestroy(pending.compute_done);
  }
  if (pending.decode_done) {
    cudaEventDestroy(pending.decode_done);
  }
  if (pending.prefill_done) {
    cudaEventDestroy(pending.prefill_done);
  }
  return true;
#else
  (void)fence;
  return true;
#endif
}

void InferfluxCudaExecutor::NativeCopySequencePrefix(int src_seq, int dst_seq,
                                                     int n_tokens) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!kv_cache_ || n_tokens <= 0 || src_seq < 0 || dst_seq < 0) {
    return;
  }
  if (src_seq == dst_seq) {
    return;
  }
  if (!kv_cache_->CopySequencePrefix(src_seq, dst_seq, n_tokens,
                                     compute_stream_)) {
    log::Warn("inferflux_cuda_executor",
              "NativeCopySequencePrefix failed (src=" +
                  std::to_string(src_seq) + ", dst=" + std::to_string(dst_seq) +
                  ", tokens=" + std::to_string(n_tokens) + ")");
  }
#else
  (void)src_seq;
  (void)dst_seq;
  (void)n_tokens;
#endif
}

std::vector<uint8_t>
InferfluxCudaExecutor::NativeSerializeSequence(int sequence_id) const {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!kv_cache_ || sequence_id < 0) {
    return {};
  }
  std::vector<uint8_t> blob;
  if (!kv_cache_->SerializeSequence(sequence_id, &blob)) {
    return {};
  }
  return blob;
#else
  (void)sequence_id;
  return {};
#endif
}

bool InferfluxCudaExecutor::NativeHydrateSequence(
    int dest_sequence_id, const std::vector<uint8_t> &blob) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!kv_cache_ || dest_sequence_id < 0 || blob.empty()) {
    return false;
  }
  return kv_cache_->HydrateSequence(dest_sequence_id, blob, compute_stream_);
#else
  (void)dest_sequence_id;
  (void)blob;
  return false;
#endif
}

InferfluxCudaRuntime::ChatResult InferfluxCudaExecutor::NativeFormatChat(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) const {
  ChatResult result;
  // Prefer the dedicated chat template renderer (GGUFTokenizer with
  // strategy-based rendering). Fall back to the main tokenizer.
  const ITokenizer *tmpl_tok = chat_template_tokenizer_
                                   ? chat_template_tokenizer_.get()
                                   : tokenizer_.get();
  if (!tmpl_tok) {
    return result;
  }
  auto chat = tmpl_tok->ApplyChatTemplate(messages, add_assistant_prefix);
  result.prompt = std::move(chat.prompt);
  result.valid = chat.valid;
  return result;
}

const ITokenizer *InferfluxCudaExecutor::NativeGetTokenizer() const {
  return tokenizer_.get();
}

int InferfluxCudaExecutor::NativeVocabSize() const {
  return model_config_.vocab_size;
}

int InferfluxCudaExecutor::NativeEmbedDims() const {
  return model_config_.hidden_size;
}

std::vector<float> InferfluxCudaExecutor::NativeEmbed(const std::string &text) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_ || !tokenizer_ || model_config_.hidden_size <= 0) {
    return {};
  }
  std::vector<int> tokens = tokenizer_->Tokenize(text, /*add_bos=*/false);
  if (tokens.empty()) {
    return {};
  }

  const int hidden = model_config_.hidden_size;

  // Device buffer for the output embedding.
  float *d_embed = nullptr;
  cudaError_t err = cudaMalloc(&d_embed, hidden * sizeof(float));
  if (err != cudaSuccess) {
    log::Error("inferflux_cuda_executor",
               "NativeEmbed: cudaMalloc failed for embedding buffer");
    return {};
  }

  // Use a temporary sequence ID for the embedding pass.
  static std::atomic<int> embed_seq_counter{900000};
  const int embed_seq_id = embed_seq_counter.fetch_add(1);

  bool ok = model_forward_->EmbedForward(tokens, embed_seq_id, d_embed);
  if (!ok) {
    cudaFree(d_embed);
    return {};
  }

  std::vector<float> result(hidden);
  cudaMemcpyAsync(result.data(), d_embed, hidden * sizeof(float),
                  cudaMemcpyDeviceToHost, compute_stream_);
  cudaStreamSynchronize(compute_stream_);
  cudaFree(d_embed);
  return result;
#else
  (void)text;
  return {};
#endif
}

AttentionTensorData InferfluxCudaExecutor::CaptureAttentionTensors() {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_) {
    AttentionTensorData result;
    result.ok = false;
    result.error = "No model forward pass available";
    return result;
  }

  // Forward to the model forward implementation
  return model_forward_->CaptureAttentionTensors();
#else
  AttentionTensorData result;
  result.ok = false;
  result.error = "Native kernels not ready";
  return result;
#endif
}

int InferfluxCudaExecutor::CopyLastLogitsToHost(float *host_buf, int buf_size) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!d_logits_ || !sampler_ || model_config_.vocab_size <= 0) {
    return 0;
  }
  int n = model_config_.vocab_size;
  if (buf_size < n) {
    return 0;
  }
  sampler_->CopyLogitsToHost(d_logits_, host_buf);
  return n;
#else
  (void)host_buf;
  (void)buf_size;
  return 0;
#endif
}

int InferfluxCudaExecutor::BurstDecodeGreedy(int sequence_id, int n_past_start,
                                             int first_token_id, int n_tokens,
                                             int eos_token_id,
                                             std::vector<int> *out_tokens) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_ || !sampler_ || n_tokens <= 0) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(shared_pipeline_mutex_);

  out_tokens->clear();
  out_tokens->reserve(n_tokens);

  int *d_meta = model_forward_->GetBatchMetaDevice();
  const int max_B = model_forward_->GetMaxBatchSize();
  if (!d_meta || max_B <= 0) {
    return 0; // Device-side relay not supported
  }

  // Allocate device-side EOS flag if needed
  if (!d_eos_flag_) {
    cudaMalloc(&d_eos_flag_, sizeof(int));
  }

  // Allocate device-side token accumulation buffer
  int *d_token_buf = nullptr;
  cudaMalloc(&d_token_buf, n_tokens * sizeof(int));
  if (!d_token_buf) {
    return 0;
  }

  const int B = 1;
  std::vector<int> batch_tokens = {first_token_id};
  std::vector<int> batch_n_past = {n_past_start};
  std::vector<int> batch_seq_ids = {sequence_id};
  std::vector<float> batch_temps = {0.0f}; // greedy
  std::vector<int> batch_top_ks = {0};
  std::vector<float> batch_top_ps = {1.0f};
  std::vector<uint32_t> batch_seeds = {UINT32_MAX};

  // First token: normal BatchForward (uploads metadata, captures/replays graph)
  bool fwd_ok = model_forward_->BatchForward(batch_tokens, batch_n_past,
                                             batch_seq_ids, d_logits_, B);
  if (!fwd_ok) {
    cudaFree(d_token_buf);
    return 0;
  }

  // Sample first token — lightweight device-only argmax (no D2H, no event)
  sampler_->EnqueueGreedyArgmaxDeviceOnly(d_logits_, B);

  // Accumulate + relay + EOS check — all on device
  cuda_kernel::AppendTokenToBuffer(sampler_->DeviceResultBatch(), d_token_buf,
                                   0, compute_stream_);
  cuda_kernel::DeviceTokenRelay(sampler_->DeviceResultBatch(), d_meta, B, max_B,
                                compute_stream_);
  cuda_kernel::DeviceCheckEos(sampler_->DeviceResultBatch(), B, eos_token_id,
                              d_eos_flag_, compute_stream_);

  int tokens_generated = 1;

  // Burst loop: graph replay + argmax + relay — entire pipeline on device.
  // No host sync, no D2H memcpy, no event records between tokens.
  for (int t = 1; t < n_tokens; ++t) {
    if (!model_forward_->BatchForwardReplay(d_logits_, B)) {
      break;
    }
    sampler_->EnqueueGreedyArgmaxDeviceOnly(d_logits_, B);
    cuda_kernel::AppendTokenToBuffer(sampler_->DeviceResultBatch(), d_token_buf,
                                     t, compute_stream_);
    cuda_kernel::DeviceTokenRelay(sampler_->DeviceResultBatch(), d_meta, B,
                                  max_B, compute_stream_);
    cuda_kernel::DeviceCheckEos(sampler_->DeviceResultBatch(), B, eos_token_id,
                                d_eos_flag_, compute_stream_);
    ++tokens_generated;
  }

  // Single sync: collect ALL tokens from device in one D2H copy
  cudaStreamSynchronize(compute_stream_);

  std::vector<int> h_tokens(tokens_generated);
  cudaMemcpy(h_tokens.data(), d_token_buf, tokens_generated * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Check EOS flag
  int h_eos = 0;
  cudaMemcpy(&h_eos, d_eos_flag_, sizeof(int), cudaMemcpyDeviceToHost);

  // Copy to output, truncating at EOS if hit
  for (int i = 0; i < tokens_generated; ++i) {
    if (h_tokens[i] == eos_token_id) {
      break;
    }
    out_tokens->push_back(h_tokens[i]);
  }

  cudaFree(d_token_buf);
  return static_cast<int>(out_tokens->size());
#else
  (void)sequence_id;
  (void)n_past_start;
  (void)first_token_id;
  (void)n_tokens;
  (void)eos_token_id;
  (void)out_tokens;
  return 0;
#endif
}

} // namespace inferflux
