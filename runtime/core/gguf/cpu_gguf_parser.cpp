#include "runtime/core/gguf/igguf_parser.h"
#include "server/logging/logger.h"

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace inferflux {
namespace runtime {
namespace core {
namespace gguf {

namespace {

// GGUF constants
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

// Tensor name mapping tables (from gguf_util.cpp)
const std::unordered_map<std::string, std::string> &GetQwen2TensorMap() {
  static const std::unordered_map<std::string, std::string> kMap = {
      {"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
      {"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"},
      {"blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"},
      // ... (abbreviated for brevity, full map in original)
  };
  return kMap;
}

template <typename T> bool ReadScalar(std::FILE *file, T *value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "ReadScalar expects POD values");
  if (!file || !value) {
    return false;
  }
  return fread(value, sizeof(T), 1, file) == 1;
}

bool SkipBytes(std::FILE *file, uint64_t count) {
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
// CpuGgufParser Implementation
//==============================================================================

class CpuGgufParser : public IGGUFParser {
public:
  bool ParseHeader(const std::filesystem::path &path,
                   GgufHeader *header) override {
    if (!header) {
      return false;
    }

    std::FILE *file = fopen(path.string().c_str(), "rb");
    if (!file) {
      log::Error("cpu_gguf_parser",
                 "Failed to open GGUF file: " + path.string());
      return false;
    }

    bool success = false;
    do {
      if (!ReadScalar(file, &header->magic)) {
        log::Error("cpu_gguf_parser", "Failed to read magic number");
        break;
      }

      if (!ReadScalar(file, &header->version)) {
        log::Error("cpu_gguf_parser", "Failed to read version");
        break;
      }

      if (!ReadScalar(file, &header->tensor_count)) {
        log::Error("cpu_gguf_parser", "Failed to read tensor count");
        break;
      }

      if (!ReadScalar(file, &header->kv_count)) {
        log::Error("cpu_gguf_parser", "Failed to read KV count");
        break;
      }

      success = ValidateHeader(*header);
    } while (false);

    fclose(file);
    return success;
  }

  bool ReadTensorInfo(std::FILE *file, GgufTensorInfo *info) override {
    if (!file || !info) {
      return false;
    }

    uint64_t name_len;
    if (!ReadScalar(file, &name_len)) {
      return false;
    }

    constexpr uint64_t kMaxNameLen = 1024;
    if (name_len > kMaxNameLen) {
      log::Error("cpu_gguf_parser", "Tensor name too long");
      return false;
    }

    info->name.resize(static_cast<size_t>(name_len));
    if (name_len > 0 &&
        fread(info->name.data(), 1, name_len, file) != name_len) {
      return false;
    }

    uint32_t n_dims;
    if (!ReadScalar(file, &n_dims)) {
      return false;
    }

    info->shape.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
      uint64_t dim;
      if (!ReadScalar(file, &dim)) {
        return false;
      }
      info->shape[i] = static_cast<size_t>(dim);
    }

    uint32_t type_raw;
    if (!ReadScalar(file, &type_raw)) {
      return false;
    }
    info->type = static_cast<GgufTensorType>(type_raw);

    uint64_t offset;
    if (!ReadScalar(file, &offset)) {
      return false;
    }
    info->offset = static_cast<size_t>(offset);

    // Calculate byte size
    size_t elems = 1;
    for (auto dim : info->shape) {
      elems *= dim;
    }

    // Type-specific size calculation
    switch (info->type) {
    case GgufTensorType::F32:
      info->byte_size = elems * 4;
      break;
    case GgufTensorType::F16:
      info->byte_size = elems * 2;
      break;
    default:
      // Quantized types - handled differently
      info->byte_size = CalcQuantizedSize(info->type, info->shape);
      break;
    }

    return true;
  }

  bool ReadKeyValue(std::FILE *file, std::string *key,
                    GgufValueType *type) override {
    if (!file || !key || !type) {
      return false;
    }

    // Read key string
    uint64_t key_len;
    if (!ReadScalar(file, &key_len)) {
      return false;
    }

    constexpr uint64_t kMaxKeyLen = 1024;
    if (key_len > kMaxKeyLen) {
      return false;
    }

    key->resize(static_cast<size_t>(key_len));
    if (key_len > 0 && fread(key->data(), 1, key_len, file) != key_len) {
      return false;
    }

    // Read value type
    uint32_t type_raw;
    if (!ReadScalar(file, &type_raw)) {
      return false;
    }
    *type = static_cast<GgufValueType>(type_raw);

    return true;
  }

  bool SkipValue(std::FILE *file, GgufValueType type) override {
    if (!file) {
      return false;
    }

    switch (type) {
    case GgufValueType::UINT8:
    case GgufValueType::INT8:
    case GgufValueType::BOOL:
      // 1-byte types
      return SkipBytes(file, 1);

    case GgufValueType::UINT16:
    case GgufValueType::INT16:
      // 2-byte types
      return SkipBytes(file, 2);

    case GgufValueType::UINT32:
    case GgufValueType::INT32:
    case GgufValueType::FLOAT32:
      // 4-byte types
      return SkipBytes(file, 4);

    case GgufValueType::UINT64:
    case GgufValueType::INT64:
    case GgufValueType::FLOAT64:
      return SkipBytes(file, 8);

    case GgufValueType::STRING: {
      uint64_t len;
      if (!ReadScalar(file, &len)) {
        return false;
      }
      return SkipBytes(file, len);
    }

    case GgufValueType::ARRAY: {
      // GGUF spec order: element type (u32) first, then count (u64).
      uint32_t elem_type;
      uint64_t count;
      if (!ReadScalar(file, &elem_type) || !ReadScalar(file, &count)) {
        return false;
      }
      if (count > (uint64_t{1} << 32)) {
        // Sanity bound: real GGUF arrays never approach 4B elements; refuse
        // to spin on a misaligned cursor.
        log::Warn("cpu_gguf_parser",
                  "Unreasonable array length in SkipValue: " +
                      std::to_string(count));
        return false;
      }
      const auto elem = static_cast<GgufValueType>(elem_type);
      for (uint64_t i = 0; i < count; ++i) {
        if (!SkipValue(file, elem)) {
          return false;
        }
      }
      return true;
    }

    default:
      log::Warn("cpu_gguf_parser",
                "Unknown value type in SkipValue: " +
                    std::to_string(static_cast<uint32_t>(type)));
      return false;
    }
  }

  bool ValidateHeader(const GgufHeader &header) const override {
    if (header.magic != GGUF_MAGIC) {
      log::Error("cpu_gguf_parser",
                 "Invalid GGUF magic: " + std::to_string(header.magic));
      return false;
    }

    if (header.version > GGUF_VERSION) {
      log::Warn("cpu_gguf_parser",
                "GGUF version " + std::to_string(header.version) +
                    " newer than supported " + std::to_string(GGUF_VERSION));
      // Continue anyway - forward compatibility
    }

    if (header.tensor_count < 0 || header.kv_count < 0) {
      log::Error("cpu_gguf_parser", "Invalid counts in header");
      return false;
    }

    return true;
  }

  bool IsAvailable() const override { return true; }

private:
  size_t CalcQuantizedSize(GgufTensorType type,
                           const std::vector<size_t> &shape) const {
    // Calculate size for quantized tensors
    size_t block_size = 256; // Most quantized types
    size_t bytes_per_block = 0;

    switch (type) {
    case GgufTensorType::Q4_0:
      block_size = 32;
      bytes_per_block = 18; // 4-byte scales + 16-byte data per 32 values
      break;
    case GgufTensorType::Q4_1:
      block_size = 32;
      bytes_per_block = 24; // 4-byte scales + 4-byte mins + 16-byte data
      break;
    case GgufTensorType::Q5_0:
      block_size = 32;
      bytes_per_block = 22;
      break;
    case GgufTensorType::Q5_1:
      block_size = 32;
      bytes_per_block = 28;
      break;
    case GgufTensorType::Q8_0:
      block_size = 32;
      bytes_per_block = 34; // 4-byte scales + 32-byte data per 32 values
      break;
    case GgufTensorType::Q2_K:
    case GgufTensorType::Q3_K:
    case GgufTensorType::Q4_K:
    case GgufTensorType::Q5_K:
    case GgufTensorType::Q6_K:
    case GgufTensorType::Q8_K:
      block_size = 256;
      // Complex size calculation for K-quants
      // Simplified here - full implementation would match llama.cpp
      bytes_per_block = (type == GgufTensorType::Q2_K)   ? 82
                        : (type == GgufTensorType::Q3_K) ? 98
                        : (type == GgufTensorType::Q4_K) ? 144
                        : (type == GgufTensorType::Q5_K) ? 180
                        : (type == GgufTensorType::Q6_K) ? 216
                                                         : 300;
      break;
    default:
      return 0;
    }

    // Calculate total size
    size_t num_elements = 1;
    for (auto dim : shape) {
      num_elements *= dim;
    }

    size_t num_blocks = (num_elements + block_size - 1) / block_size;
    return num_blocks * bytes_per_block;
  }
};

//==============================================================================
// GgufTensorInfo Implementation
//==============================================================================

size_t GgufTensorInfo::block_size() const {
  switch (type) {
  case GgufTensorType::Q4_0:
  case GgufTensorType::Q4_1:
  case GgufTensorType::Q5_0:
  case GgufTensorType::Q5_1:
  case GgufTensorType::Q8_0:
  case GgufTensorType::Q8_1:
    return 32;
  case GgufTensorType::Q2_K:
  case GgufTensorType::Q3_K:
  case GgufTensorType::Q4_K:
  case GgufTensorType::Q5_K:
  case GgufTensorType::Q6_K:
  case GgufTensorType::Q8_K:
    return 256;
  default:
    return 1; // Non-block quantized
  }
}

std::string GgufTensorInfo::type_string() const {
  switch (type) {
  case GgufTensorType::F32:
    return "f32";
  case GgufTensorType::F16:
    return "f16";
  case GgufTensorType::Q4_0:
    return "q4_0";
  case GgufTensorType::Q4_1:
    return "q4_1";
  case GgufTensorType::Q5_0:
    return "q5_0";
  case GgufTensorType::Q5_1:
    return "q5_1";
  case GgufTensorType::Q8_0:
    return "q8_0";
  case GgufTensorType::Q8_1:
    return "q8_1";
  case GgufTensorType::Q2_K:
    return "q2_k";
  case GgufTensorType::Q3_K:
    return "q3_k";
  case GgufTensorType::Q4_K:
    return "q4_k";
  case GgufTensorType::Q5_K:
    return "q5_k";
  case GgufTensorType::Q6_K:
    return "q6_k";
  case GgufTensorType::Q8_K:
    return "q8_k";
  default:
    return "unknown";
  }
}

//==============================================================================
// Factory Function
//==============================================================================

std::unique_ptr<IGGUFParser> CreateCpuGgufParser() {
  return std::make_unique<CpuGgufParser>();
}

} // namespace gguf
} // namespace core
} // namespace runtime
} // namespace inferflux
