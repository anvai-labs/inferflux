#include "server/startup_advisor.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef INFERFLUX_HAS_ROCM
#include <hip/hip_runtime.h>
#endif

namespace inferflux {
namespace {

const char *kComponent = "startup_advisor";
constexpr std::uint64_t kKiB = 1024ULL;
constexpr std::uint64_t kMiB = kKiB * kKiB;
constexpr std::uint64_t kGiB = kMiB * kKiB;
constexpr std::uint32_t kLoadOverheadNumerator = 11U;
constexpr std::uint32_t kLoadOverheadDenominator = 10U;

using GgufTensorType = ::inferflux::runtime::cuda::native::GGUF::TensorType;

bool UsesCudaBackend(const std::string &backend) {
  return ParseLlamaBackendTarget(backend) == LlamaBackendTarget::kCuda;
}

bool UsesCpuBackend(const std::string &backend) {
  return ParseLlamaBackendTarget(backend) == LlamaBackendTarget::kCpu;
}

bool IsQuantizedQuantizationType(QuantizationType q) {
  switch (q) {
  case QuantizationType::kQ8_1:
  case QuantizationType::kQ8_0:
  case QuantizationType::kQ8_K:
  case QuantizationType::kQ6_K:
  case QuantizationType::kQ5_1:
  case QuantizationType::kQ5_0:
  case QuantizationType::kQ5_K_M:
  case QuantizationType::kQ5_K:
  case QuantizationType::kQ4_1:
  case QuantizationType::kQ4_0:
  case QuantizationType::kQ4_K_M:
  case QuantizationType::kQ4_K:
  case QuantizationType::kQ3_K:
  case QuantizationType::kQ3_K_M:
  case QuantizationType::kQ2_K:
    return true;
  default:
    return false;
  }
}

QuantizationType QuantizationTypeFromGgufTensorType(GgufTensorType type) {
  switch (type) {
  case GgufTensorType::F32:
    return QuantizationType::kFp32;
  case GgufTensorType::F16:
    return QuantizationType::kFp16;
  case GgufTensorType::Q8_1:
    return QuantizationType::kQ8_1;
  case GgufTensorType::Q8_0:
    return QuantizationType::kQ8_0;
  case GgufTensorType::Q8_K:
    return QuantizationType::kQ8_K;
  case GgufTensorType::Q6_K:
    return QuantizationType::kQ6_K;
  case GgufTensorType::Q5_1:
    return QuantizationType::kQ5_1;
  case GgufTensorType::Q5_0:
    return QuantizationType::kQ5_0;
  case GgufTensorType::Q5_K:
    return QuantizationType::kQ5_K;
  case GgufTensorType::Q4_1:
    return QuantizationType::kQ4_1;
  case GgufTensorType::Q4_0:
    return QuantizationType::kQ4_0;
  case GgufTensorType::Q4_K:
    return QuantizationType::kQ4_K;
  case GgufTensorType::Q3_K:
    return QuantizationType::kQ3_K;
  case GgufTensorType::Q2_K:
    return QuantizationType::kQ2_K;
  default:
    return QuantizationType::kUnknown;
  }
}

bool ReadU32(FILE *file, uint32_t *out) {
  if (!file || !out) {
    return false;
  }
  return fread(out, sizeof(uint32_t), 1, file) == 1;
}

std::filesystem::path ResolveGgufPath(const std::filesystem::path &path) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(path, ec) &&
      path.extension() == ".gguf") {
    return path;
  }
  if (!std::filesystem::is_directory(path, ec)) {
    return {};
  }

  std::filesystem::path first_gguf;
  for (const auto &entry : std::filesystem::directory_iterator(path, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    if (entry.path().extension() != ".gguf") {
      continue;
    }
    if (first_gguf.empty() || entry.path().filename() < first_gguf.filename()) {
      first_gguf = entry.path();
    }
  }
  return first_gguf;
}

QuantizationType
DetectQuantizationFromGgufMetadata(const std::filesystem::path &path) {
  const std::filesystem::path gguf_path = ResolveGgufPath(path);
  if (gguf_path.empty()) {
    return QuantizationType::kUnknown;
  }

  FILE *file = fopen(gguf_path.string().c_str(), "rb");
  if (!file) {
    return QuantizationType::kUnknown;
  }

  ::inferflux::runtime::cuda::native::GGUF::Header header{};
  if (!ReadU32(file, &header.magic) || !ReadU32(file, &header.version) ||
      !::inferflux::runtime::cuda::native::GGUFReader::ReadInt64(
          file, &header.tensor_count) ||
      !::inferflux::runtime::cuda::native::GGUFReader::ReadInt64(
          file, &header.kv_count)) {
    fclose(file);
    return QuantizationType::kUnknown;
  }

  if (!::inferflux::runtime::cuda::native::ValidateGGUFHeader(header) ||
      header.tensor_count < 0 || header.kv_count < 0) {
    fclose(file);
    return QuantizationType::kUnknown;
  }

  for (int64_t i = 0; i < header.kv_count; ++i) {
    std::string key;
    if (!::inferflux::runtime::cuda::native::GGUFReader::ReadString(file,
                                                                    &key)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    uint32_t type_raw = 0;
    if (!ReadU32(file, &type_raw)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    const auto value_type =
        static_cast<::inferflux::runtime::cuda::native::GGUF::ValueType>(
            type_raw);
    if (!::inferflux::runtime::cuda::native::GGUFReader::SkipValue(
            file, value_type)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
  }

  constexpr std::size_t kTensorTypeCount =
      static_cast<std::size_t>(GgufTensorType::Q8_K) + 1U;
  std::array<std::size_t, kTensorTypeCount> type_counts{};

  for (int64_t i = 0; i < header.tensor_count; ++i) {
    std::string tensor_name;
    if (!::inferflux::runtime::cuda::native::GGUFReader::ReadString(
            file, &tensor_name)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    uint32_t n_dims = 0;
    if (!ReadU32(file, &n_dims)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    for (uint32_t dim_idx = 0; dim_idx < n_dims; ++dim_idx) {
      int64_t dim = 0;
      if (!::inferflux::runtime::cuda::native::GGUFReader::ReadInt64(file,
                                                                     &dim)) {
        fclose(file);
        return QuantizationType::kUnknown;
      }
    }

    uint32_t type_raw = 0;
    if (!ReadU32(file, &type_raw)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    uint64_t offset = 0;
    if (!::inferflux::runtime::cuda::native::GGUFReader::ReadUint64(file,
                                                                    &offset)) {
      fclose(file);
      return QuantizationType::kUnknown;
    }
    (void)offset;

    if (type_raw >= kTensorTypeCount) {
      continue;
    }
    ++type_counts[type_raw];
  }

  fclose(file);

  std::size_t best_quantized_count = 0;
  GgufTensorType best_quantized_type = GgufTensorType::F16;
  for (std::size_t idx = 0; idx < kTensorTypeCount; ++idx) {
    const auto type = static_cast<GgufTensorType>(idx);
    if (!::inferflux::runtime::cuda::native::IsQuantizedType(type)) {
      continue;
    }
    if (type_counts[idx] > best_quantized_count) {
      best_quantized_count = type_counts[idx];
      best_quantized_type = type;
    }
  }

  if (best_quantized_count > 0) {
    return QuantizationTypeFromGgufTensorType(best_quantized_type);
  }

  const std::size_t f16_count =
      type_counts[static_cast<std::size_t>(GgufTensorType::F16)];
  const std::size_t f32_count =
      type_counts[static_cast<std::size_t>(GgufTensorType::F32)];
  if (f16_count == 0 && f32_count == 0) {
    return QuantizationType::kUnknown;
  }
  return f16_count >= f32_count ? QuantizationType::kFp16
                                : QuantizationType::kFp32;
}

std::uint64_t ScaleByPercent(std::uint64_t value, std::uint32_t percent) {
#if defined(__SIZEOF_INT128__)
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(value) * percent;
  return static_cast<std::uint64_t>(scaled / 100U);
#else
  return (value / 100U) * percent + ((value % 100U) * percent) / 100U;
#endif
}

std::uint64_t ScaleByRatio(std::uint64_t value, std::uint32_t numerator,
                           std::uint32_t denominator) {
  if (denominator == 0U) {
    return value;
  }
#if defined(__SIZEOF_INT128__)
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(value) * numerator;
  return static_cast<std::uint64_t>(scaled / denominator);
#else
  return (value / denominator) * numerator +
         ((value % denominator) * numerator) / denominator;
#endif
}

int ClampToInt(std::uint64_t value) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

double PercentUsage(std::uint64_t used, std::uint64_t total) {
  if (total == 0) {
    return 0.0;
  }
#if defined(__SIZEOF_INT128__)
  const unsigned __int128 basis_points =
      (static_cast<unsigned __int128>(used) * 10000U) / total;
  const unsigned __int128 clamped = std::min<unsigned __int128>(
      basis_points, static_cast<unsigned __int128>(100000000U));
  return static_cast<double>(static_cast<std::uint32_t>(clamped)) / 100.0;
#else
  const std::uint64_t basis_points =
      (used / total) * 10000U + ((used % total) * 10000U) / total;
  const std::uint64_t clamped =
      std::min<std::uint64_t>(basis_points, 100000000U);
  return static_cast<double>(clamped) / 100.0;
#endif
}

// ---------------------------------------------------------------------------
// Helper: Format bytes to human-readable string
// ---------------------------------------------------------------------------
std::string FormatBytes(std::uint64_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_index = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024.0 && unit_index < 4) {
    size /= 1024.0;
    unit_index++;
  }

  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
  return std::string(buffer);
}

// ---------------------------------------------------------------------------
// Helper: Detect quantization from GGUF model filename
// ---------------------------------------------------------------------------
QuantizationType DetectQuantizationFromFilename(const std::string &filename) {
  std::string lower = filename;
  std::transform(
      lower.begin(), lower.end(), lower.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  if (lower.find("q8_k") != std::string::npos)
    return QuantizationType::kQ8_K;
  if (lower.find("q8_1") != std::string::npos)
    return QuantizationType::kQ8_1;
  if (lower.find("q8_0") != std::string::npos)
    return QuantizationType::kQ8_0;
  if (lower.find("q6_k") != std::string::npos)
    return QuantizationType::kQ6_K;
  if (lower.find("q5_k_m") != std::string::npos)
    return QuantizationType::kQ5_K_M;
  if (lower.find("q5_k") != std::string::npos)
    return QuantizationType::kQ5_K;
  if (lower.find("q5_1") != std::string::npos)
    return QuantizationType::kQ5_1;
  if (lower.find("q5_0") != std::string::npos)
    return QuantizationType::kQ5_0;
  if (lower.find("q4_k_m") != std::string::npos)
    return QuantizationType::kQ4_K_M;
  if (lower.find("q4_k") != std::string::npos)
    return QuantizationType::kQ4_K;
  if (lower.find("q4_1") != std::string::npos)
    return QuantizationType::kQ4_1;
  if (lower.find("q4_0") != std::string::npos)
    return QuantizationType::kQ4_0;
  if (lower.find("q3_k_m") != std::string::npos)
    return QuantizationType::kQ3_K_M;
  if (lower.find("q3_k") != std::string::npos)
    return QuantizationType::kQ3_K;
  if (lower.find("q2_k") != std::string::npos)
    return QuantizationType::kQ2_K;
  if (lower.find("bf16") != std::string::npos)
    return QuantizationType::kBf16;
  if (lower.find("fp16") != std::string::npos ||
      lower.find("f16") != std::string::npos) {
    return QuantizationType::kFp16;
  }
  if (lower.find("fp32") != std::string::npos ||
      lower.find("f32") != std::string::npos) {
    return QuantizationType::kFp32;
  }

  return QuantizationType::kUnknown;
}

// ---------------------------------------------------------------------------
// Rule: Dynamic slot allocation based on model size and GPU memory
// ---------------------------------------------------------------------------
int CheckDynamicSlotAllocation(const StartupAdvisorContext &ctx,
                               MemoryAllocationRecommendation &rec) {
  if (!ctx.gpu.available || ctx.gpu.total_vram_bytes == 0)
    return 0;
  if (ctx.models.empty())
    return 0;

  rec = CalculateOptimalSlotAllocation(ctx);
  if (!rec.valid)
    return 0;

  // Only recommend if significantly different from current config
  double slot_ratio = static_cast<double>(rec.recommended_max_slots) /
                      std::max(1, ctx.config.max_parallel_sequences);

  if (slot_ratio < 0.5 || slot_ratio > 2.0) {
    log::Info(kComponent,
              "[RECOMMEND] slot_allocation: GPU has " +
                  FormatBytes(ctx.gpu.total_vram_bytes) + " (" +
                  std::to_string(ctx.gpu.total_vram_bytes / kMiB) + " MB)\n" +
                  "  Model: " + ctx.models[0].id + " (" +
                  FormatBytes(rec.model_size_bytes) + " loaded, " +
                  ctx.models[0].quantization_string + ")\n" +
                  "  Current: max_parallel_sequences=" +
                  std::to_string(ctx.config.max_parallel_sequences) +
                  ", n_ctx=" + std::to_string(ctx.config.n_ctx) + "\n" +
                  "  Recommended: max_parallel_sequences=" +
                  std::to_string(rec.recommended_max_slots) +
                  ", n_ctx=" + std::to_string(rec.recommended_n_ctx) + "\n" +
                  "  Memory breakdown:\n" +
                  "    - Model: " + FormatBytes(rec.model_size_bytes) + "\n" +
                  "    - Overhead: " + FormatBytes(rec.overhead_bytes) + "\n" +
                  "    - KV cache: " + FormatBytes(rec.total_kv_bytes) + " (" +
                  std::to_string(rec.recommended_max_slots) + " slots × " +
                  FormatBytes(rec.per_slot_kv_bytes) + " per slot)\n" +
                  "    - Total: " + FormatBytes(rec.total_needed_bytes) + " (" +
                  std::to_string(rec.utilization_percent) + "% of GPU)\n" +
                  "  Config:\n" + rec.config_yaml_snippet);

    // Log warnings if any
    for (const auto &warning : rec.warnings) {
      log::Warn(kComponent, "[WARNING] " + warning);
    }

    return 1;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Rule: Validate quantization format matches model file
// ---------------------------------------------------------------------------
int CheckQuantizationMismatch(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto &m : ctx.models) {
    if (m.format == "gguf" && m.quantization == QuantizationType::kUnknown) {
      log::Info(kComponent,
                "[RECOMMEND] quantization: Model '" + m.id +
                    "' is GGUF but quantization type is unknown — ensure the "
                    "path points to a valid GGUF file with readable tensor "
                    "metadata");
      ++count;
    }
  }
  return count;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API: Calculate optimal slot allocation
// ---------------------------------------------------------------------------
MemoryAllocationRecommendation
CalculateOptimalSlotAllocation(const StartupAdvisorContext &ctx) {

  MemoryAllocationRecommendation rec;
  if (ctx.models.empty() || !ctx.gpu.available) {
    return rec;
  }

  // Use largest model for calculation
  const AdvisorModelInfo &model = *std::max_element(
      ctx.models.begin(), ctx.models.end(), [](const auto &a, const auto &b) {
        return a.file_size_bytes < b.file_size_bytes;
      });

  // Step 1: Estimate loaded model size
  rec.model_size_bytes = EstimateLoadedModelSize(model);
  if (rec.model_size_bytes == 0) {
    // Fallback to file size if estimation fails
    rec.model_size_bytes = model.file_size_bytes;
  }

  // Step 2: Calculate available memory (target % of total VRAM)
  // Configurable via INFERFLUX_GPU_UTILIZATION_PCT (0-100, default: 85)
  constexpr int kDefaultTargetUtilizationPct = 85;
  int target_utilization_pct = kDefaultTargetUtilizationPct;

  if (const char *env_util = std::getenv("INFERFLUX_GPU_UTILIZATION_PCT")) {
    try {
      int util_pct = std::stoi(env_util);
      if (util_pct >= 50 && util_pct <= 98) {
        target_utilization_pct = util_pct;
      }
    } catch (const std::exception &ex) {
      log::Warn(kComponent, "Invalid INFERFLUX_GPU_UTILIZATION_PCT '" +
                                std::string(env_util) + "': " + ex.what());
    }
  }

  std::uint64_t target_vram =
      ScaleByPercent(ctx.gpu.total_vram_bytes,
                     static_cast<std::uint32_t>(target_utilization_pct));

  // Step 3: Estimate overhead with activation tensors
  // Base overhead: CUDA context, fragmentation (configurable via
  // INFERFLUX_OVERHEAD_GB) Activation overhead: 1.5x model size for FP16, 1.2x
  // for quantized, 1.0x for FP32
  constexpr std::uint64_t kDefaultOverheadBytes = kGiB; // 1 GiB base

  // Determine activation overhead percentage based on quantization.
  int activation_overhead_pct = 0; // Default for FP32
  if (model.quantization == QuantizationType::kFp16 ||
      model.quantization == QuantizationType::kBf16) {
    activation_overhead_pct = 50;
  } else if (IsQuantizedQuantizationType(model.quantization)) {
    activation_overhead_pct = 20;
  }

  // Calculate activation memory
  std::uint64_t activation_memory =
      ScaleByPercent(rec.model_size_bytes,
                     static_cast<std::uint32_t>(activation_overhead_pct));

  // Base overhead (configurable)
  std::uint64_t overhead_bytes = kDefaultOverheadBytes;
  if (const char *env_overhead = std::getenv("INFERFLUX_OVERHEAD_GB")) {
    try {
      int overhead_gb = std::stoi(env_overhead);
      if (overhead_gb >= 0 && overhead_gb <= 16) {
        overhead_bytes = static_cast<std::uint64_t>(overhead_gb) * kGiB;
      }
    } catch (const std::exception &ex) {
      log::Warn(kComponent, "Invalid INFERFLUX_OVERHEAD_GB '" +
                                std::string(env_overhead) + "': " + ex.what());
    }
  }

  // Total overhead = base + activation + fragmentation allowance
  constexpr std::uint32_t kFragmentationAllowancePct = 10;
  const std::uint64_t base_overhead = overhead_bytes + activation_memory;
  rec.overhead_bytes =
      base_overhead + ScaleByPercent(base_overhead, kFragmentationAllowancePct);

  // Log memory breakdown if verbose
  if (std::getenv("INFERFLUX_STARTUP_ADVISOR_VERBOSE")) {
    std::string breakdown =
        "Memory calculation:\n" + std::string("  Model size: ") +
        FormatBytes(rec.model_size_bytes) + "\n" +
        "  Activation overhead: " + FormatBytes(activation_memory) + " (" +
        std::to_string(activation_overhead_pct) + "%)\n" +
        "  Base overhead: " + FormatBytes(overhead_bytes) + "\n" +
        "  Fragmentation allowance: 10%\n" +
        "  Total overhead: " + FormatBytes(rec.overhead_bytes);
    log::Info(kComponent, breakdown);
  }

  // Step 4: Calculate memory available for KV cache
  std::uint64_t available_for_kv = 0;
  if (rec.model_size_bytes >= target_vram) {
    rec.warnings.push_back(
        "Model size exceeds 85% of GPU VRAM - will likely fail to load");
    return rec;
  }
  if (rec.overhead_bytes >= target_vram - rec.model_size_bytes) {
    rec.warnings.push_back("Overhead estimate leaves no room for KV cache at "
                           "current target utilization");
    return rec;
  }
  available_for_kv = target_vram - rec.model_size_bytes - rec.overhead_bytes;

  // Step 5: Determine model architecture (defaults for Qwen 2.5 3B)
  int n_layers = model.n_layers > 0 ? model.n_layers : 36;
  int hidden_dim = model.hidden_dim > 0 ? model.hidden_dim : 128;

  // Step 6: Get or estimate context window
  int n_ctx = ctx.config.n_ctx > 0 ? ctx.config.n_ctx : 2048;

  // Step 7: Calculate per-slot KV cache size
  rec.per_slot_kv_bytes = CalculatePerSlotKvSize(n_ctx, hidden_dim, n_layers,
                                                 32, sizeof(float) * 2);
  if (rec.per_slot_kv_bytes == 0) {
    rec.warnings.push_back(
        "Invalid KV cache estimate (per-slot bytes is zero)");
    return rec;
  }

  // Step 8: Calculate max slots based on available KV memory
  int max_slots_by_memory =
      ClampToInt(available_for_kv / rec.per_slot_kv_bytes);

  // Step 9: Apply practical limits
  // Minimum 10 slots for reasonable concurrent operation
  // Maximum 256 slots for large contexts
  // Both can be overridden via environment variables
  int min_slots = 10;
  int max_slots = 256;

  if (const char *env_min = std::getenv("INFERFLUX_MIN_SLOTS")) {
    try {
      min_slots = std::max(4, std::stoi(env_min)); // Allow 4-256 range
    } catch (const std::exception &ex) {
      log::Warn(kComponent, "Invalid INFERFLUX_MIN_SLOTS '" +
                                std::string(env_min) + "': " + ex.what());
    }
  }
  if (const char *env_max = std::getenv("INFERFLUX_MAX_SLOTS")) {
    try {
      max_slots = std::min(512, std::stoi(env_max)); // Allow up to 512
    } catch (const std::exception &ex) {
      log::Warn(kComponent, "Invalid INFERFLUX_MAX_SLOTS '" +
                                std::string(env_max) + "': " + ex.what());
    }
  }

  rec.recommended_max_slots =
      std::max(min_slots, std::min(max_slots, max_slots_by_memory));
  rec.recommended_n_ctx = n_ctx;

  // Step 10: Calculate totals
  rec.total_kv_bytes = rec.per_slot_kv_bytes * rec.recommended_max_slots;
  rec.total_needed_bytes =
      rec.model_size_bytes + rec.overhead_bytes + rec.total_kv_bytes;
  rec.available_bytes = ctx.gpu.total_vram_bytes;
  rec.utilization_percent =
      PercentUsage(rec.total_needed_bytes, ctx.gpu.total_vram_bytes);

  rec.valid = true;

  // Step 11: Generate config YAML snippet
  std::ostringstream config;
  config << "runtime:\n";
  config << "  llama:\n";
  config << "    max_parallel_sequences: " << rec.recommended_max_slots << "\n";
  config << "    n_ctx: " << rec.recommended_n_ctx;
  rec.config_yaml_snippet = config.str();

  return rec;
}

// ---------------------------------------------------------------------------
// Public API: Detect quantization type
// ---------------------------------------------------------------------------
QuantizationType DetectQuantization(const std::string &model_path,
                                    const std::string &format) {
  if (format == "gguf") {
    const auto from_metadata =
        DetectQuantizationFromGgufMetadata(std::filesystem::path(model_path));
    if (from_metadata != QuantizationType::kUnknown) {
      return from_metadata;
    }
    return DetectQuantizationFromFilename(
        std::filesystem::path(model_path).filename().string());
  }
  // TODO: Detect from safetensors metadata
  return QuantizationType::kUnknown;
}

// ---------------------------------------------------------------------------
// Public API: Get quantization string
// ---------------------------------------------------------------------------
std::string GetQuantizationString(QuantizationType q) {
  switch (q) {
  case QuantizationType::kFp32:
    return "FP32";
  case QuantizationType::kFp16:
    return "FP16";
  case QuantizationType::kBf16:
    return "BF16";
  case QuantizationType::kQ8_1:
    return "Q8_1";
  case QuantizationType::kQ8_0:
    return "Q8_0";
  case QuantizationType::kQ8_K:
    return "Q8_K";
  case QuantizationType::kQ6_K:
    return "Q6_K";
  case QuantizationType::kQ5_1:
    return "Q5_1";
  case QuantizationType::kQ5_0:
    return "Q5_0";
  case QuantizationType::kQ5_K_M:
    return "Q5_K_M";
  case QuantizationType::kQ5_K:
    return "Q5_K";
  case QuantizationType::kQ4_1:
    return "Q4_1";
  case QuantizationType::kQ4_0:
    return "Q4_0";
  case QuantizationType::kQ4_K_M:
    return "Q4_K_M";
  case QuantizationType::kQ4_K:
    return "Q4_K";
  case QuantizationType::kQ3_K:
    return "Q3_K";
  case QuantizationType::kQ3_K_M:
    return "Q3_K_M";
  case QuantizationType::kQ2_K:
    return "Q2_K";
  default:
    return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Public API: Estimate loaded model size
// ---------------------------------------------------------------------------
std::uint64_t EstimateLoadedModelSize(const AdvisorModelInfo &model) {
  // Base size from file
  std::uint64_t file_size = model.file_size_bytes;
  if (file_size == 0)
    return 0;

  // Quantization compression ratio represented as numerator/denominator.
  // Effective loaded size for safetensors is approximated as:
  // file_size * denominator / numerator.
  std::uint32_t compression_numerator = 1U;
  std::uint32_t compression_denominator = 1U;

  switch (model.quantization) {
  case QuantizationType::kQ2_K:
    compression_numerator = 8U;
    compression_denominator = 1U;
    break;
  case QuantizationType::kQ3_K:
  case QuantizationType::kQ3_K_M:
    compression_numerator = 5U;
    compression_denominator = 1U;
    break;
  case QuantizationType::kQ4_0:
  case QuantizationType::kQ4_1:
  case QuantizationType::kQ4_K_M:
  case QuantizationType::kQ4_K:
    compression_numerator = 9U;
    compression_denominator = 2U; // ~4.5x compression from FP16
    break;
  case QuantizationType::kQ5_0:
  case QuantizationType::kQ5_1:
  case QuantizationType::kQ5_K_M:
  case QuantizationType::kQ5_K:
    compression_numerator = 7U;
    compression_denominator = 2U;
    break;
  case QuantizationType::kQ6_K:
    compression_numerator = 3U;
    compression_denominator = 1U;
    break;
  case QuantizationType::kQ8_1:
  case QuantizationType::kQ8_0:
  case QuantizationType::kQ8_K:
    compression_numerator = 2U;
    compression_denominator = 1U;
    break;
  case QuantizationType::kFp16:
  case QuantizationType::kBf16:
    compression_numerator = 2U;
    compression_denominator = 1U; // 2x from FP32
    break;
  case QuantizationType::kFp32:
    compression_numerator = 1U;
    compression_denominator = 1U;
    break;
  default:
    if (model.format == "gguf") {
      const QuantizationType detected =
          DetectQuantization(model.path, model.format);
      if (detected == QuantizationType::kUnknown) {
        compression_numerator = 2U;
        compression_denominator = 1U;
      } else {
        AdvisorModelInfo inferred = model;
        inferred.quantization = detected;
        return EstimateLoadedModelSize(inferred);
      }
    }
    break;
  }

  // For GGUF, file size is already compressed
  // For safetensors, need to account for quantization
  if (model.format == "gguf") {
    // GGUF file is the actual size when loaded
    // Add ~10% for overhead (CUDA context, etc.)
    return ScaleByRatio(file_size, kLoadOverheadNumerator,
                        kLoadOverheadDenominator);
  } else {
    // Safetensors: file contains FP32/BF16 weights
    // Calculate based on compression
    const std::uint64_t estimated_loaded =
        ScaleByRatio(file_size, compression_denominator, compression_numerator);
    return ScaleByRatio(estimated_loaded, kLoadOverheadNumerator,
                        kLoadOverheadDenominator);
  }
}

// ---------------------------------------------------------------------------
// Public API: Calculate per-slot KV cache size
// ---------------------------------------------------------------------------
std::uint64_t CalculatePerSlotKvSize(int n_ctx, int hidden_dim, int n_layers,
                                     int n_heads, size_t element_size) {
  (void)n_heads;
  if (n_ctx <= 0 || hidden_dim <= 0 || n_layers <= 0 || element_size == 0U) {
    return 0;
  }

  // KV cache for one sequence:
  // - K cache: n_ctx × hidden_dim × n_layers × element_size
  // - V cache: n_ctx × hidden_dim × n_layers × element_size
  // Total: 2 × n_ctx × hidden_dim × n_layers × element_size

  return 2ULL * static_cast<std::uint64_t>(n_ctx) *
         static_cast<std::uint64_t>(hidden_dim) *
         static_cast<std::uint64_t>(n_layers) *
         static_cast<std::uint64_t>(element_size);
}

// ---------------------------------------------------------------------------
// Existing rules (preserved)
// ---------------------------------------------------------------------------

// Rule 1: Backend mismatch — safetensors on CUDA using llama.cpp provider.
int CheckBackendMismatch(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto &m : ctx.models) {
    if (m.format == "safetensors" && UsesCudaBackend(m.backend) &&
        m.backend_provider == "llama_cpp") {
      log::Info(kComponent, "[RECOMMEND] backend: Model '" + m.id +
                                "' uses safetensors on CUDA — set "
                                "backend to inferflux_cuda");
      ++count;
    }
  }
  return count;
}

// Rule 2: Attention kernel — GPU SM >= 8.0, CUDA on, FA disabled.
int CheckAttentionKernel(const StartupAdvisorContext &ctx) {
  if (!ctx.config.cuda_enabled || !ctx.gpu.available)
    return 0;
  if (ctx.gpu.compute_major < 8)
    return 0;
  if (ctx.config.flash_attention_enabled)
    return 0;

  log::Info(kComponent,
            "[RECOMMEND] attention: GPU '" + ctx.gpu.device_name + "' (SM " +
                std::to_string(ctx.gpu.compute_major) + "." +
                std::to_string(ctx.gpu.compute_minor) +
                ") supports FA2 — set runtime.cuda.flash_attention.enabled: "
                "true and runtime.cuda.attention.kernel: fa2");
  return 1;
}

// Rule 3: Batch size vs VRAM — can fit more concurrent sequences.
int CheckBatchSizeVsVram(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available || ctx.gpu.free_vram_bytes == 0)
    return 0;
  if (ctx.models.empty())
    return 0;

  // Estimate per the largest model.
  std::uint64_t max_model_size = 0;
  for (const auto &m : ctx.models) {
    if (m.file_size_bytes > max_model_size)
      max_model_size = m.file_size_bytes;
  }
  if (max_model_size == 0)
    return 0;

  // Heuristic: each batch slot costs ~15% of model size in KV/activation.
  const std::uint64_t per_slot =
      std::max<std::uint64_t>(ScaleByPercent(max_model_size, 15U), 1ULL);

  int suggested = ClampToInt(ctx.gpu.free_vram_bytes / per_slot);
  if (suggested < 1)
    suggested = 1;

  const std::uint64_t suggested_x2 =
      static_cast<std::uint64_t>(suggested) * 2ULL;
  const std::uint64_t current_x3 =
      static_cast<std::uint64_t>(std::max(ctx.config.max_batch_size, 1)) * 3ULL;
  if (suggested_x2 > current_x3) {
    log::Info(kComponent, "[RECOMMEND] batch_size: " +
                              std::to_string(ctx.gpu.free_vram_bytes / kMiB) +
                              " MB VRAM free — increase "
                              "runtime.scheduler.max_batch_size to " +
                              std::to_string(suggested) + " (current: " +
                              std::to_string(ctx.config.max_batch_size) + ")");
    return 1;
  }
  return 0;
}

// Rule 4: Phase overlap — CUDA on, overlap off, batch_size >= 4.
int CheckPhaseOverlap(const StartupAdvisorContext &ctx) {
  if (!ctx.config.cuda_enabled)
    return 0;
  if (ctx.config.phase_overlap_enabled)
    return 0;
  if (ctx.config.max_batch_size < 4)
    return 0;

  log::Info(kComponent,
            "[RECOMMEND] phase_overlap: CUDA enabled with batch_size >= 4 — "
            "set runtime.cuda.phase_overlap.enabled: true for mixed-batch "
            "decode prioritization");
  return 1;
}

// Rule 5: KV cache pages — large free VRAM, low page count.
int CheckKvCachePages(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available || ctx.gpu.free_vram_bytes == 0)
    return 0;

  // Suggest more pages when free VRAM > 4 GB and pages <= 64.
  constexpr std::uint64_t kFourGiB = 4ULL * kGiB;
  if (ctx.gpu.free_vram_bytes < kFourGiB)
    return 0;
  if (ctx.config.kv_cpu_pages > 64)
    return 0;

  constexpr std::uint64_t kPageGranularityBytes = 64ULL * kMiB;
  std::size_t suggested =
      static_cast<std::size_t>(ctx.gpu.free_vram_bytes / kPageGranularityBytes);
  if (suggested <= ctx.config.kv_cpu_pages)
    return 0;

  log::Info(kComponent,
            "[RECOMMEND] kv_cache: " +
                std::to_string(ctx.gpu.free_vram_bytes / kMiB) +
                " MB VRAM free with only " +
                std::to_string(ctx.config.kv_cpu_pages) +
                " KV pages — increase runtime.paged_kv.cpu_pages to " +
                std::to_string(suggested));
  return 1;
}

// Rule 6: Tensor parallelism — multi-GPU, TP=1, large model.
int CheckTensorParallelism(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available)
    return 0;
  if (ctx.gpu.device_count <= 1)
    return 0;
  if (ctx.config.tp_degree > 1)
    return 0;

  // Check if any model barely fits (file_size > 70% of single-GPU VRAM).
  const std::uint64_t vram_pressure_threshold =
      ScaleByPercent(ctx.gpu.total_vram_bytes, 70U);
  for (const auto &m : ctx.models) {
    if (m.file_size_bytes > vram_pressure_threshold) {
      log::Info(kComponent, "[RECOMMEND] tensor_parallel: " +
                                std::to_string(ctx.gpu.device_count) +
                                " GPUs detected but TP=1 — model '" + m.id +
                                "' uses " +
                                std::to_string(m.file_size_bytes / kMiB) +
                                " MB, set runtime.tensor_parallel: " +
                                std::to_string(ctx.gpu.device_count));
      return 1;
    }
  }
  return 0;
}

// Rule 7: Unknown format — model format not detected.
int CheckUnknownFormat(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto &m : ctx.models) {
    if (m.format == "unknown") {
      log::Info(kComponent,
                "[RECOMMEND] format: Model '" + m.id +
                    "' has unknown format — set models[*].format explicitly "
                    "(gguf, safetensors, or hf)");
      ++count;
    }
  }
  return count;
}

// Rule 8: GPU unused — GPU available but all models on CPU.
int CheckGpuUnused(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available)
    return 0;
  if (ctx.models.empty())
    return 0;

  bool all_cpu = std::all_of(
      ctx.models.begin(), ctx.models.end(),
      [](const AdvisorModelInfo &m) { return UsesCpuBackend(m.backend); });
  if (!all_cpu)
    return 0;

  log::Info(kComponent, "[RECOMMEND] gpu_unused: GPU '" + ctx.gpu.device_name +
                            "' is available but all models use CPU — set "
                            "runtime.cuda.enabled: true");
  return 1;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int RunStartupAdvisor(const StartupAdvisorContext &ctx) {
  if (const char *env = std::getenv("INFERFLUX_DISABLE_STARTUP_ADVISOR")) {
    std::string val = env;
    if (val == "true" || val == "1" || val == "yes")
      return 0;
  }

  if (ctx.models.empty())
    return 0;

  log::Info(kComponent, "=== InferFlux Startup Recommendations ===");

  int total = 0;
  total += CheckBackendMismatch(ctx);
  total += CheckAttentionKernel(ctx);
  total += CheckBatchSizeVsVram(ctx);
  total += CheckPhaseOverlap(ctx);
  total += CheckKvCachePages(ctx);
  total += CheckTensorParallelism(ctx);
  total += CheckUnknownFormat(ctx);
  total += CheckGpuUnused(ctx);

  // NEW: Dynamic slot allocation
  MemoryAllocationRecommendation mem_rec;
  total += CheckDynamicSlotAllocation(ctx, mem_rec);
  total += CheckQuantizationMismatch(ctx);

  if (total == 0) {
    log::Info(kComponent, "=== No recommendations — config looks good! ===");
  } else {
    log::Info(kComponent, "=== End Recommendations (" + std::to_string(total) +
                              " suggestion" + (total != 1 ? "s" : "") +
                              ") ===");
  }
  return total;
}

AdvisorGpuInfo ProbeCudaGpu() {
  AdvisorGpuInfo info;
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return info;
  }
  info.available = true;
  info.device_count = device_count;

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  }

  std::size_t free_mem = 0, total_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
    info.free_vram_bytes = static_cast<std::uint64_t>(free_mem);
  }

  // NEW: Calculate recommended reserve (15%)
  info.recommended_reserve_bytes = ScaleByPercent(info.total_vram_bytes, 15U);
  info.usable_vram_bytes =
      info.total_vram_bytes - info.recommended_reserve_bytes;
#endif
  return info;
}

AdvisorGpuInfo ProbeRocmGpu() {
  AdvisorGpuInfo info;
#ifdef INFERFLUX_HAS_ROCM
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return info;
  }
  info.available = true;
  info.device_count = device_count;

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
    info.device_name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  }

  std::size_t free_mem = 0, total_mem = 0;
  if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
    info.free_vram_bytes = static_cast<std::uint64_t>(free_mem);
  }

  // NEW: Calculate recommended reserve (15%)
  info.recommended_reserve_bytes = ScaleByPercent(info.total_vram_bytes, 15U);
  info.usable_vram_bytes =
      info.total_vram_bytes - info.recommended_reserve_bytes;
#endif
  return info;
}

} // namespace inferflux
