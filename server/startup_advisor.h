#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace inferflux {

// Quantization type detection
enum class QuantizationType {
  kUnknown,
  kFp32,   // 32-bit float
  kFp16,   // 16-bit float
  kBf16,   // 16-bit bfloat
  kQ8_1,   // 8-bit quantization (Q8_1)
  kQ8_0,   // 8-bit quantization
  kQ8_K,   // 8-bit K-block quantization
  kQ6_K,   // 6-bit quantization
  kQ5_1,   // 5-bit quantization (Q5_1)
  kQ5_0,   // 5-bit quantization (Q5_0)
  kQ5_K_M, // 5-bit quantization (medium)
  kQ5_K,   // 5-bit quantization
  kQ4_1,   // 4-bit quantization (Q4_1)
  kQ4_0,   // 4-bit quantization (Q4_0)
  kQ4_K_M, // 4-bit quantization (medium)
  kQ4_K,   // 4-bit quantization
  kQ3_K,   // 3-bit quantization
  kQ3_K_M, // 3-bit quantization (medium)
  kQ2_K,   // 2-bit quantization
};

// Pure-data snapshot of a loaded model for advisor inspection.
struct AdvisorModelInfo {
  std::string id;
  std::string path;
  std::string format{"unknown"};
  std::string backend{"cpu"};
  std::string backend_provider{"llama_cpp"};
  bool backend_fallback{false};
  std::string backend_fallback_reason;
  std::uint64_t file_size_bytes{0};
  bool is_moe{false};
  int n_experts{0};

  // NEW: Memory and quantization info
  QuantizationType quantization{QuantizationType::kUnknown};
  std::string quantization_string; // e.g., "Q4_K_M", "Q5_K_M"
  int n_params{0};                 // Parameter count (billions)
  int n_layers{0};                 // Transformer layers
  int hidden_dim{0};               // Hidden dimension
  int n_ctx_max{0};                // Maximum context window supported
  std::uint64_t estimated_loaded_size_bytes{0}; // Actual GPU memory when loaded
  double compression_ratio{1.0}; // Quantization compression ratio
};

// GPU hardware snapshot (populated by ProbeCudaGpu / ProbeRocmGpu).
struct AdvisorGpuInfo {
  bool available{false};
  int device_count{0};
  std::string device_name;
  int compute_major{0};
  int compute_minor{0};
  int sm_count{0};
  std::uint64_t total_vram_bytes{0};
  std::uint64_t free_vram_bytes{0};

  // NEW: Detailed memory breakdown
  std::uint64_t recommended_reserve_bytes{0}; // Memory to keep free (15%)
  std::uint64_t usable_vram_bytes{0};         // Total - reserve
};

// Effective config snapshot taken at startup.
struct AdvisorConfig {
  bool cuda_enabled{false};
  bool flash_attention_enabled{false};
  std::string cuda_attention_kernel{"auto"};
  bool phase_overlap_enabled{false};
  int max_batch_size{8};
  int max_batch_tokens{2048};
  std::size_t kv_cpu_pages{32};
  bool prefer_inferflux{true};
  bool allow_llama_cpp_fallback{true};
  bool strict_inferflux_request{false};
  std::string backend_priority;
  int tp_degree{1};
  bool speculative_enabled{false};

  // NEW: Sequence slot configuration
  int max_parallel_sequences{128};      // Current configured value
  int n_ctx{2048};                      // Current context window
  std::uint64_t sequence_slot_bytes{0}; // Actual KV per slot
};

// Memory allocation recommendation
struct MemoryAllocationRecommendation {
  bool valid{false};

  // Model memory
  std::uint64_t model_size_bytes{0};
  std::uint64_t overhead_bytes{0};

  // KV cache calculation
  int recommended_max_slots{0};
  int recommended_n_ctx{0};
  std::uint64_t per_slot_kv_bytes{0};
  std::uint64_t total_kv_bytes{0};

  // Memory breakdown
  std::uint64_t total_needed_bytes{0};
  std::uint64_t available_bytes{0};
  double utilization_percent{0.0};

  // Configuration recommendation
  std::string config_yaml_snippet;

  // Warnings
  std::vector<std::string> warnings;
};

// Everything the advisor needs — assembled in main.cpp, no live pointers.
struct StartupAdvisorContext {
  std::vector<AdvisorModelInfo> models;
  AdvisorGpuInfo gpu;
  AdvisorConfig config;
};

// Run all recommendation rules.  Returns the number of recommendations logged.
// Respects INFERFLUX_DISABLE_STARTUP_ADVISOR=true (returns 0 immediately).
int RunStartupAdvisor(const StartupAdvisorContext &ctx);

// Hardware probing helpers — safe to call on any platform.
// On builds without the matching SDK they return a default (unavailable)
// struct.
AdvisorGpuInfo ProbeCudaGpu();
AdvisorGpuInfo ProbeRocmGpu();

// NEW: Model size detection and slot allocation
MemoryAllocationRecommendation
CalculateOptimalSlotAllocation(const StartupAdvisorContext &ctx);

// NEW: Quantization detection from model path/metadata
QuantizationType DetectQuantization(const std::string &model_path,
                                    const std::string &format);
std::string GetQuantizationString(QuantizationType q);

// NEW: Estimate loaded model size from file size and quantization
std::uint64_t EstimateLoadedModelSize(const AdvisorModelInfo &model);

// NEW: Calculate per-slot KV cache size
std::uint64_t
CalculatePerSlotKvSize(int n_ctx, int hidden_dim, int n_layers, int n_heads,
                       size_t element_size = sizeof(float) * 2 // K + V
);

} // namespace inferflux
