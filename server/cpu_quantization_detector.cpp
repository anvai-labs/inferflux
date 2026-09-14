#include "runtime/core/gguf/igguf_parser.h"
#include "server/logging/logger.h"
#include "server/quantization_detection.h"
#include "server/startup_advisor.h"

#include <map>
#include <set>

namespace inferflux {
namespace server {

// Bring types into scope
using inferflux::QuantizationType;
namespace gguf = inferflux::runtime::core::gguf;

/**
 * @brief CPU-only quantization detector
 *
 * Uses CpuGgufParser to detect quantization types from GGUF files
 * without any CUDA dependencies.
 */
class CpuQuantizationDetector : public IQuantizationDetector {
public:
  QuantizationType
  DetectFromGgufMetadata(const std::filesystem::path &path) const override {

    auto parser = runtime::core::gguf::CreateCpuGgufParser();

    runtime::core::gguf::GgufHeader header;
    if (!parser->ParseHeader(path, &header)) {
      log::Error("cpu_quant_detector", "Failed to parse GGUF header");
      return QuantizationType::kUnknown;
    }

    if (!parser->ValidateHeader(header)) {
      log::Error("cpu_quant_detector", "Invalid GGUF header");
      return QuantizationType::kUnknown;
    }

    // Open file to read tensor info (RAII-managed handle)
    auto file_deleter = [](FILE *f) {
      if (f)
        fclose(f);
    };
    std::unique_ptr<FILE, decltype(file_deleter)> file(
        fopen(path.string().c_str(), "rb"), file_deleter);
    if (!file) {
      log::Error("cpu_quant_detector", "Failed to open GGUF file");
      return QuantizationType::kUnknown;
    }

    // Skip header (already parsed) and the KV metadata section — tensor
    // infos start only after all key-value pairs (issue: every GGUF with
    // KV metadata previously failed here with "Tensor name too long"
    // because the KV bytes were misparsed as tensor infos).
    static constexpr long kGgufHeaderOffset =
        24; // magic + version + tensor_count + kv_count
    fseek(file.get(), kGgufHeaderOffset, SEEK_SET);

    std::string kv_key;
    runtime::core::gguf::GgufValueType kv_type;
    for (int64_t kv = 0; kv < header.kv_count; ++kv) {
      if (!parser->ReadKeyValue(file.get(), &kv_key, &kv_type) ||
          !parser->SkipValue(file.get(), kv_type)) {
        log::Error("cpu_quant_detector", "Failed to skip GGUF KV metadata");
        return QuantizationType::kUnknown;
      }
    }

    // Count tensor types
    std::map<runtime::core::gguf::GgufTensorType, size_t> type_counts;
    size_t total_tensors = 0;
    size_t f32_count = 0;
    size_t f16_count = 0;

    for (int64_t i = 0; i < header.tensor_count; ++i) {
      runtime::core::gguf::GgufTensorInfo info;
      if (!parser->ReadTensorInfo(file.get(), &info)) {
        log::Error("cpu_quant_detector", "Failed to read tensor info");
        return QuantizationType::kUnknown;
      }

      total_tensors++;
      type_counts[info.type]++;

      if (info.type == runtime::core::gguf::GgufTensorType::F32) {
        f32_count++;
      } else if (info.type == runtime::core::gguf::GgufTensorType::F16) {
        f16_count++;
      }
    }

    if (total_tensors == 0) {
      return QuantizationType::kUnknown;
    }

    // Determine quantization type based on majority
    size_t best_quantized_count = 0;
    runtime::core::gguf::GgufTensorType best_type =
        runtime::core::gguf::GgufTensorType::F32;

    for (const auto &kv : type_counts) {
      const auto &type = kv.first;
      const auto &count = kv.second;

      if (type == runtime::core::gguf::GgufTensorType::F32 ||
          type == runtime::core::gguf::GgufTensorType::F16) {
        continue; // Skip non-quantized types
      }

      if (count > best_quantized_count) {
        best_quantized_count = count;
        best_type = type;
      }
    }

    // Return detected quantization
    if (best_quantized_count == 0) {
      // No quantized tensors found
      return inferflux::QuantizationType::kFp32;
    }

    return GgufTensorTypeToQuantizationType(best_type);
  }

  bool IsAvailable() const override { return true; }

  const char *GetName() const override { return "CPU"; }

private:
  inferflux::QuantizationType GgufTensorTypeToQuantizationType(
      runtime::core::gguf::GgufTensorType type) const {
    using GgufTensorType = runtime::core::gguf::GgufTensorType;

    switch (type) {
    case GgufTensorType::Q4_0:
      return inferflux::QuantizationType::kQ4_0;
    case GgufTensorType::Q4_1:
      return inferflux::QuantizationType::kQ4_1;
    case GgufTensorType::Q5_0:
      return inferflux::QuantizationType::kQ5_0;
    case GgufTensorType::Q5_1:
      return inferflux::QuantizationType::kQ5_1;
    case GgufTensorType::Q8_0:
      return inferflux::QuantizationType::kQ8_0;
    case GgufTensorType::Q2_K:
      return inferflux::QuantizationType::kQ2_K;
    case GgufTensorType::Q3_K:
      return inferflux::QuantizationType::kQ3_K;
    case GgufTensorType::Q4_K:
      return inferflux::QuantizationType::kQ4_K_M;
    case GgufTensorType::Q5_K:
      return inferflux::QuantizationType::kQ5_K_M;
    case GgufTensorType::Q6_K:
      return inferflux::QuantizationType::kQ6_K;
    case GgufTensorType::Q8_K:
      return inferflux::QuantizationType::kQ8_K;
    case GgufTensorType::F32:
      return inferflux::QuantizationType::kFp32;
    case GgufTensorType::F16:
      return inferflux::QuantizationType::kFp16;
    default:
      return inferflux::QuantizationType::kUnknown;
    }
  }
};

//==============================================================================
// Factory Function
//==============================================================================

std::unique_ptr<IQuantizationDetector> CreateQuantizationDetector() {
  return std::make_unique<CpuQuantizationDetector>();
}

} // namespace server
} // namespace inferflux
