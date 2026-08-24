#include "runtime/backends/cuda/native/gguf_util.h"
#include "server/logging/logger.h"
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace {

template <typename T> bool ReadScalar(FILE *file, T *value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "ReadScalar expects POD values");
  if (!file || !value) {
    return false;
  }
  return fread(value, sizeof(T), 1, file) == 1;
}

bool SkipBytes(FILE *file, uint64_t count) {
  if (!file) {
    return false;
  }
  if (count == 0) {
    return true;
  }
  std::array<unsigned char, 4096> buffer{};
  uint64_t remaining = count;
  while (remaining > 0) {
    const size_t chunk =
        static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
    if (fread(buffer.data(), 1, chunk, file) != chunk) {
      return false;
    }
    remaining -= chunk;
  }
  return true;
}

} // namespace

//==============================================================================
// GGUF::TensorInfo Implementation
//==============================================================================

size_t GGUF::TensorInfo::block_size() const {
  // Block sizes from ggml-common.h
  switch (type) {
  case GGUF::TensorType::Q4_0:
  case GGUF::TensorType::Q4_1:
  case GGUF::TensorType::Q5_0:
  case GGUF::TensorType::Q5_1:
  case GGUF::TensorType::Q8_0:
  case GGUF::TensorType::Q8_1:
    return 32;
  case GGUF::TensorType::Q2_K:
  case GGUF::TensorType::Q3_K:
  case GGUF::TensorType::Q4_K:
  case GGUF::TensorType::Q5_K:
  case GGUF::TensorType::Q6_K:
  case GGUF::TensorType::Q8_K:
    return 256;
  default:
    return 1; // Non-block quantized
  }
}

std::string GGUF::TensorInfo::type_string() const {
  return TensorTypeToString(type);
}

//==============================================================================
// GGUFReader Implementation
//==============================================================================

bool GGUFReader::ReadUint64(FILE *file, uint64_t *value) {
  return ReadScalar(file, value);
}

bool GGUFReader::ReadInt64(FILE *file, int64_t *value) {
  return ReadScalar(file, value);
}

bool GGUFReader::ReadString(FILE *file, std::string *str) {
  uint64_t len;
  if (!ReadUint64(file, &len)) {
    return false;
  }
  constexpr uint64_t kMaxStringLen = 1ULL << 30; // 1 GiB safety bound
  if (len > kMaxStringLen ||
      len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  str->resize(static_cast<size_t>(len));
  if (len > 0 && fread(str->data(), 1, len, file) != len) {
    return false;
  }

  return true;
}

bool GGUFReader::ReadValue(FILE *file, GGUF::ValueType type,
                           std::vector<uint8_t> *output) {
  if (!output) {
    return false;
  }
  output->clear();

  const auto read_fixed = [&](size_t bytes) -> bool {
    output->resize(bytes);
    return bytes == 0 || fread(output->data(), 1, bytes, file) == bytes;
  };

  switch (type) {
  case GGUF::ValueType::UINT8:
  case GGUF::ValueType::INT8:
  case GGUF::ValueType::BOOL:
    return read_fixed(1);

  case GGUF::ValueType::UINT16:
  case GGUF::ValueType::INT16:
    return read_fixed(2);

  case GGUF::ValueType::UINT32:
  case GGUF::ValueType::INT32:
  case GGUF::ValueType::FLOAT32:
    return read_fixed(4);

  case GGUF::ValueType::STRING: {
    std::string str;
    if (!ReadString(file, &str)) {
      return false;
    }
    output->assign(str.begin(), str.end());
    return true;
  }

  case GGUF::ValueType::ARRAY: {
    uint32_t elem_type_raw = 0;
    if (!ReadScalar(file, &elem_type_raw)) {
      return false;
    }
    GGUF::ValueType elem_type = static_cast<GGUF::ValueType>(elem_type_raw);
    uint64_t len = 0;
    if (!ReadUint64(file, &len)) {
      return false;
    }
    for (uint64_t i = 0; i < len; ++i) {
      std::vector<uint8_t> temp;
      if (!ReadValue(file, elem_type, &temp)) {
        return false;
      }
    }
    return true;
  }

  case GGUF::ValueType::UINT64:
  case GGUF::ValueType::INT64:
  case GGUF::ValueType::FLOAT64:
    return read_fixed(8);

  default:
    log::Error("gguf_reader", "Unknown value type: " +
                                  std::to_string(static_cast<uint32_t>(type)));
    return false;
  }
}

bool GGUFReader::SkipValue(FILE *file, GGUF::ValueType type) {
  switch (type) {
  case GGUF::ValueType::UINT8:
  case GGUF::ValueType::INT8:
  case GGUF::ValueType::BOOL:
    return SkipBytes(file, 1);
  case GGUF::ValueType::UINT16:
  case GGUF::ValueType::INT16:
    return SkipBytes(file, 2);
  case GGUF::ValueType::UINT32:
  case GGUF::ValueType::INT32:
  case GGUF::ValueType::FLOAT32:
    return SkipBytes(file, 4);
  case GGUF::ValueType::UINT64:
  case GGUF::ValueType::INT64:
  case GGUF::ValueType::FLOAT64:
    return SkipBytes(file, 8);
  case GGUF::ValueType::STRING: {
    uint64_t len = 0;
    if (!ReadUint64(file, &len)) {
      return false;
    }
    return SkipBytes(file, len);
  }
  case GGUF::ValueType::ARRAY: {
    uint32_t elem_type_raw = 0;
    if (!ReadScalar(file, &elem_type_raw)) {
      return false;
    }
    GGUF::ValueType elem_type = static_cast<GGUF::ValueType>(elem_type_raw);
    uint64_t len = 0;
    if (!ReadUint64(file, &len)) {
      return false;
    }
    for (uint64_t i = 0; i < len; ++i) {
      if (!SkipValue(file, elem_type)) {
        return false;
      }
    }
    return true;
  }
  default:
    return false;
  }
}

GGUFReader::TensorNameParts
GGUFReader::ParseTensorName(const std::string &name) {
  TensorNameParts parts;

  // Check for special tensors
  if (name == "tok_emb.weight" || name == "token_embd.weight") {
    parts.layer = -1;
    parts.component = "tok_emb";
    parts.type = "weight";
    return parts;
  }

  if (name == "output.weight" || name == "lm_head.weight") {
    parts.layer = -2;
    parts.component = "output";
    parts.type = "weight";
    return parts;
  }

  // Parse layer tensors: "blk.N.component.type"
  size_t dot_pos = name.find('.');
  if (dot_pos == std::string::npos) {
    return parts;
  }

  std::string prefix = name.substr(0, dot_pos);
  if (prefix != "blk") {
    return parts;
  }

  size_t second_dot = name.find('.', dot_pos + 1);
  if (second_dot == std::string::npos) {
    return parts;
  }

  std::string layer_str = name.substr(dot_pos + 1, second_dot - dot_pos - 1);
  parts.layer = std::stoi(layer_str);

  size_t third_dot = name.find('.', second_dot + 1);
  if (third_dot == std::string::npos) {
    parts.component = name.substr(second_dot + 1);
    parts.type = "weight"; // Default
    return parts;
  }

  parts.component = name.substr(second_dot + 1, third_dot - second_dot - 1);
  parts.type = name.substr(third_dot + 1);

  return parts;
}

std::string GGUFReader::MapTensorName(const std::string &gguf_name,
                                      const std::string &model_type) {
  if (gguf_name == "tok_emb.weight" || gguf_name == "token_embd.weight") {
    return "model.embed_tokens.weight";
  }
  if (gguf_name == "output.weight") {
    return "lm_head.weight";
  }
  if (gguf_name == "output_norm.weight" || gguf_name == "norm.weight" ||
      gguf_name == "ln_f.weight") {
    return "model.norm.weight";
  }
  if (gguf_name == "output_norm.bias" || gguf_name == "norm.bias" ||
      gguf_name == "ln_f.bias") {
    return "model.norm.bias";
  }

  const auto parts = ParseTensorName(gguf_name);
  if (parts.layer >= 0) {
    std::string mapped_component;
    if (parts.component == "attn_q") {
      mapped_component = "self_attn.q_proj";
    } else if (parts.component == "attn_k") {
      mapped_component = "self_attn.k_proj";
    } else if (parts.component == "attn_v") {
      mapped_component = "self_attn.v_proj";
    } else if (parts.component == "attn_o" ||
               parts.component == "attn_output") {
      mapped_component = "self_attn.o_proj";
    } else if (parts.component == "attn_norm") {
      mapped_component = "input_layernorm";
    } else if (parts.component == "ffn_norm") {
      mapped_component = "post_attention_layernorm";
    } else if (parts.component == "ffn_gate") {
      mapped_component = "mlp.gate_proj";
    } else if (parts.component == "ffn_up") {
      mapped_component = "mlp.up_proj";
    } else if (parts.component == "ffn_down") {
      mapped_component = "mlp.down_proj";
    }
    if (!mapped_component.empty()) {
      return "model.layers." + std::to_string(parts.layer) + "." +
             mapped_component + "." + parts.type;
    }
  }

  // Fallback static maps for architecture-specific edge cases.
  if (model_type == "qwen2" || model_type == "qwen") {
    const auto &map = GetQwen2TensorMap();
    auto it = map.find(gguf_name);
    if (it != map.end()) {
      return it->second;
    }
  } else if (model_type == "llama") {
    const auto &map = GetLlamaTensorMap();
    auto it = map.find(gguf_name);
    if (it != map.end()) {
      return it->second;
    }
  }

  // Fallback: return original name
  return gguf_name;
}

const std::unordered_map<std::string, std::string> &
GGUFReader::GetQwen2TensorMap() {
  static const std::unordered_map<std::string, std::string> map = {
      // Token embeddings
      {"tok_emb.weight", "model.embed_tokens.weight"},
      {"output.weight", "lm_head.weight"},

      // Layer norms
      {"ln_f.weight", "model.norm.weight"},
      {"ln_f.bias", "model.norm.bias"},

      // Attention layers (blk.N.attn_*)
      {"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
      {"blk.0.attn_q.bias", "model.layers.0.self_attn.q_proj.bias"},
      {"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"},
      {"blk.0.attn_k.bias", "model.layers.0.self_attn.k_proj.bias"},
      {"blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"},
      {"blk.0.attn_v.bias", "model.layers.0.self_attn.v_proj.bias"},
      {"blk.0.attn_o.weight", "model.layers.0.self_attn.o_proj.weight"},

      // FFN layers (blk.N.ffn_*)
      {"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
      {"blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"},
      {"blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"},

      // Layer norms per layer
      {"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
      {"blk.0.ffn_norm.weight",
       "model.layers.0.post_attention_layernorm.weight"},
  };
  return map;
}

const std::unordered_map<std::string, std::string> &
GGUFReader::GetLlamaTensorMap() {
  static const std::unordered_map<std::string, std::string> map = {
      // Token embeddings
      {"token_embd.weight", "model.embed_tokens.weight"},
      {"output.weight", "lm_head.weight"},

      // Layer norms
      {"ln_f.weight", "model.norm.weight"},
      {"ln_f.bias", "model.norm.bias"},

      // Attention layers (blk.N.attn_*)
      {"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
      {"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"},
      {"blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"},
      {"blk.0.attn_o.weight", "model.layers.0.self_attn.o_proj.weight"},

      // FFN layers (blk.N.ffn_*)
      {"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
      {"blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"},
      {"blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"},

      // Layer norms per layer
      {"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
      {"blk.0.ffn_norm.weight",
       "model.layers.0.post_attention_layernorm.weight"},
  };
  return map;
}

//==============================================================================
// Utility Functions
//==============================================================================

bool ValidateGGUFHeader(const GGUF::Header &header) {
  if (header.magic != GGUF::MAGIC) {
    log::Error("gguf", "Invalid magic number: " + std::to_string(header.magic));
    return false;
  }

  if (header.version == 0 || header.version == 1) {
    log::Error("gguf", "Unsupported legacy GGUF version: " +
                           std::to_string(header.version));
    return false;
  }

  if (header.version > GGUF::VERSION) {
    log::Error("gguf",
               "Unsupported version: " + std::to_string(header.version));
    return false;
  }

  return true;
}

std::string ValueTypeToString(GGUF::ValueType type) {
  static const std::unordered_map<GGUF::ValueType, std::string> names = {
      {GGUF::ValueType::UINT8, "uint8"},     {GGUF::ValueType::INT8, "int8"},
      {GGUF::ValueType::UINT16, "uint16"},   {GGUF::ValueType::INT16, "int16"},
      {GGUF::ValueType::UINT32, "uint32"},   {GGUF::ValueType::INT32, "int32"},
      {GGUF::ValueType::FLOAT32, "float32"}, {GGUF::ValueType::BOOL, "bool"},
      {GGUF::ValueType::STRING, "string"},   {GGUF::ValueType::ARRAY, "array"},
      {GGUF::ValueType::UINT64, "uint64"},   {GGUF::ValueType::INT64, "int64"},
      {GGUF::ValueType::FLOAT64, "float64"},
  };

  auto it = names.find(type);
  if (it != names.end()) {
    return it->second;
  }
  return "unknown";
}

std::string TensorTypeToString(GGUF::TensorType type) {
  static const std::unordered_map<GGUF::TensorType, std::string> names = {
      {GGUF::TensorType::F32, "f32"},   {GGUF::TensorType::F16, "f16"},
      {GGUF::TensorType::Q4_0, "q4_0"}, {GGUF::TensorType::Q4_1, "q4_1"},
      {GGUF::TensorType::Q5_0, "q5_0"}, {GGUF::TensorType::Q5_1, "q5_1"},
      {GGUF::TensorType::Q8_0, "q8_0"}, {GGUF::TensorType::Q8_1, "q8_1"},
      {GGUF::TensorType::Q2_K, "q2_k"}, {GGUF::TensorType::Q3_K, "q3_k"},
      {GGUF::TensorType::Q4_K, "q4_k"}, {GGUF::TensorType::Q5_K, "q5_k"},
      {GGUF::TensorType::Q6_K, "q6_k"}, {GGUF::TensorType::Q8_K, "q8_k"},
  };

  auto it = names.find(type);
  if (it != names.end()) {
    return it->second;
  }
  return "unknown";
}

GGUF::TensorType StringToTensorType(const std::string &str) {
  static const std::unordered_map<std::string, GGUF::TensorType> map = {
      {"f32", GGUF::TensorType::F32},   {"f16", GGUF::TensorType::F16},
      {"q4_0", GGUF::TensorType::Q4_0}, {"q4_1", GGUF::TensorType::Q4_1},
      {"q5_0", GGUF::TensorType::Q5_0}, {"q5_1", GGUF::TensorType::Q5_1},
      {"q8_0", GGUF::TensorType::Q8_0}, {"q8_1", GGUF::TensorType::Q8_1},
      {"q2_k", GGUF::TensorType::Q2_K}, {"q3_k", GGUF::TensorType::Q3_K},
      {"q4_k", GGUF::TensorType::Q4_K}, {"q4_k_m", GGUF::TensorType::Q4_K},
      {"q5_k", GGUF::TensorType::Q5_K}, {"q5_k_m", GGUF::TensorType::Q5_K},
      {"q6_k", GGUF::TensorType::Q6_K}, {"q6_k_m", GGUF::TensorType::Q6_K},
      {"q8_k", GGUF::TensorType::Q8_K}, {"q8_k_m", GGUF::TensorType::Q8_K},
  };

  auto it = map.find(str);
  if (it != map.end()) {
    return it->second;
  }
  return GGUF::TensorType::F16; // Default
}

bool IsQuantizedType(GGUF::TensorType type) {
  return type != GGUF::TensorType::F32 && type != GGUF::TensorType::F16;
}

std::string GetQuantizationType(GGUF::TensorType type) {
  switch (type) {
  case GGUF::TensorType::Q4_0:
    return "q4_0";
  case GGUF::TensorType::Q4_1:
    return "q4_1";
  case GGUF::TensorType::Q4_K:
    return "q4_k_m"; // GGUF doesn't distinguish Q4_K and Q4_K_M
  case GGUF::TensorType::Q5_0:
    return "q5_0";
  case GGUF::TensorType::Q5_1:
    return "q5_1";
  case GGUF::TensorType::Q5_K:
    return "q5_k_m";
  case GGUF::TensorType::Q6_K:
    return "q6_k";
  case GGUF::TensorType::Q8_0:
    return "q8_0";
  case GGUF::TensorType::Q8_1:
    return "q8_1";
  case GGUF::TensorType::Q2_K:
    return "q2_k";
  case GGUF::TensorType::Q3_K:
    return "q3_k";
  case GGUF::TensorType::Q8_K:
    return "q8_k";
  default:
    return "";
  }
}

size_t CalcTensorSize(GGUF::TensorType type,
                      const std::vector<uint64_t> &shape) {
  if (shape.empty()) {
    return 0;
  }
  size_t num_elements = 1;
  for (uint64_t dim : shape) {
    if (dim == 0 ||
        dim > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        num_elements >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return 0;
    }
    num_elements *= static_cast<size_t>(dim);
  }

  if (type == GGUF::TensorType::F32) {
    if (num_elements > std::numeric_limits<size_t>::max() / 4) {
      return 0;
    }
    return num_elements * 4;
  } else if (type == GGUF::TensorType::F16) {
    if (num_elements > std::numeric_limits<size_t>::max() / 2) {
      return 0;
    }
    return num_elements * 2;
  }

  // For quantized types, use block-based calculation
  size_t block_size = 256;
  if (type >= GGUF::TensorType::Q4_0 && type <= GGUF::TensorType::Q8_1) {
    block_size = 32;
  }

  size_t num_blocks = (num_elements + block_size - 1) / block_size;

  // Block sizes from ggml-common.h
  static const std::unordered_map<GGUF::TensorType, size_t> block_bytes = {
      {GGUF::TensorType::Q4_0, 18},  // sizeof(half) + 32/2
      {GGUF::TensorType::Q4_1, 20},  // 2*sizeof(half) + 32/2
      {GGUF::TensorType::Q5_0, 22},  // sizeof(half) + 4 + 32/2
      {GGUF::TensorType::Q5_1, 24},  // 2*sizeof(half) + 4 + 32/2
      {GGUF::TensorType::Q8_0, 34},  // sizeof(half) + 32
      {GGUF::TensorType::Q8_1, 36},  // 2*sizeof(half) + 32
      {GGUF::TensorType::Q2_K, 96},  // 2*sizeof(half) + 256/16 + 256/4
      {GGUF::TensorType::Q3_K, 110}, // sizeof(half) + 256/4 + 256/8 + 12
      {GGUF::TensorType::Q4_K, 144}, // 2*sizeof(half) + 12 + 256/2
      {GGUF::TensorType::Q5_K, 176}, // 2*sizeof(half) + 12 + 256/2 + 256/8
      {GGUF::TensorType::Q6_K, 210}, // sizeof(half) + 256/16 + 3*256/4
      {GGUF::TensorType::Q8_K,
       292}, // sizeof(float) + 256 + 256/16*sizeof(int16_t)
  };

  auto it = block_bytes.find(type);
  if (it == block_bytes.end()) {
    return 0;
  }
  if (num_blocks > std::numeric_limits<size_t>::max() / it->second) {
    return 0;
  }
  return num_blocks * it->second;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
