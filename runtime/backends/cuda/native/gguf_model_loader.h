#pragma once

#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantization_handler.h"

// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief GGUF tensor data container
 *
 * Stores quantized tensor data with metadata for dequantization.
 */
class GGUFTensorData {
public:
  GGUF::TensorInfo info;
  std::vector<uint8_t> cpu_data;  // Quantized data on CPU
  void *gpu_data{nullptr};        // Quantized data on GPU
  half *dequantized_gpu{nullptr}; // Dequantized FP16 data on GPU (cached)
  size_t gpu_offset{0};           // Offset in unified GPU buffer

  /**
   * @brief Get quantization handler for this tensor
   */
  std::shared_ptr<IQuantizationHandler> GetQuantizationHandler() const;

  /**
   * @brief Check if dequantized cache exists
   */
  bool IsDequantizedCached() const { return dequantized_gpu != nullptr; }

  void ClearCPUMemory() {
    cpu_data.clear();
    cpu_data.shrink_to_fit();
  }
};

/**
 * @brief GGUF model loader
 *
 * Implements IModelLoader interface for GGUF format models.
 * Supports quantized weights (Q4_K_M, Q5_K_M, Q6_K) with lazy dequantization.
 *
 * Key features:
 * - Parses GGUF file format (metadata + tensors)
 * - Detects quantization type from tensor data types
 * - Maps GGUF tensor names to internal naming convention
 * - Lazy dequantization (only dequantize when needed)
 * - GPU caching of dequantized weights
 */
class GGUFModelLoader : public IModelLoader {
public:
  GGUFModelLoader();
  ~GGUFModelLoader() override;

  // IModelLoader interface
  bool Load(const std::filesystem::path &model_path) override;
  const ModelInfo &GetModelInfo() const override;
  std::string GetFormat() const override;
  bool IsQuantized() const override;
  std::string GetQuantizationType() const override;
  bool UploadToGPU(cudaStream_t stream) override;
  void FreeCPUMemory() override;
  void FreeGPUMemory() override;
  void *GetGPUBuffer() const override;
  size_t GetGPUSize() const override;
  void SetDequantizedCachePolicy(DequantizedCachePolicy policy) override;
  DequantizedCachePolicy GetDequantizedCachePolicy() const override;
  void ClearDequantizedCache() override;

  /**
   * @brief Get weight accessor for a specific tensor
   * @param tensor_name Internal tensor name (e.g.,
   * "model.layers.0.self_attn.q_proj.weight")
   * @return Shared pointer to weight accessor, or nullptr if not found
   */
  std::shared_ptr<IWeightAccessor>
  GetWeightAccessor(const std::string &tensor_name) override;

  /**
   * @brief Get tensor by GGUF name (original name)
   */
  const GGUFTensorData *GetTensorByGGUFName(const std::string &gguf_name) const;

  /**
   * @brief Get all tensor names (internal naming)
   */
  std::vector<std::string> GetTensorNames() const;

  const std::vector<std::string> &TokenizerPieces() const {
    return tokenizer_pieces_;
  }
  const std::vector<std::string> &TokenizerMerges() const {
    return tokenizer_merges_;
  }
  const std::string &TokenizerPreTokenizer() const { return tokenizer_pre_; }
  const std::string &TokenizerChatTemplate() const {
    return tokenizer_chat_template_;
  }
  int TokenizerEosTokenId() const { return tokenizer_eos_token_id_; }
  int TokenizerBosTokenId() const { return tokenizer_bos_token_id_; }
  bool TokenizerAddBosToken() const { return tokenizer_add_bos_token_; }

  /**
   * @brief Get GGUF to internal tensor name mapping
   */
  const std::unordered_map<std::string, std::string> &
  GetTensorNameMapping() const {
    return gguf_to_internal_name_map_;
  }

private:
  void FreeGPUMemoryImpl();

  // File handling
  bool ParseHeader(FILE *file);
  bool ParseKeyValuePairs(FILE *file);
  bool ParseTensorInfo(FILE *file);
  bool LoadTensorData(FILE *file);

  // Model info extraction
  bool ExtractModelInfo();

  // Tensor name mapping
  void BuildTensorNameMapping();

  // GPU upload
  bool UploadQuantizedToGPU(cudaStream_t stream);
  bool UploadDequantizedToGPU(cudaStream_t stream);

  // Helper: get tensor size for GPU allocation
  size_t CalcGPUBufferSize() const;

  // Parsed data
  GGUF::Header header_;
  ModelInfo model_info_;
  std::string model_type_;        // "qwen2", "llama", etc.
  std::string quantization_type_; // "q4_k_m", "q5_k_m", etc.
  size_t alignment_{32};
  size_t data_section_offset_{0};
  std::vector<std::string> tokenizer_pieces_;
  std::vector<std::string> tokenizer_merges_;
  std::string tokenizer_pre_;
  std::string tokenizer_chat_template_;
  int tokenizer_eos_token_id_{-1};
  int tokenizer_bos_token_id_{-1};
  bool tokenizer_add_bos_token_{true};

  // Tensors (keyed by GGUF name)
  std::unordered_map<std::string, GGUFTensorData> tensors_;

  // Name mapping
  std::unordered_map<std::string, std::string> gguf_to_internal_name_map_;
  std::unordered_map<std::string, std::string> internal_to_gguf_name_map_;

  // GPU memory
  void *d_quantized_buffer_{nullptr};   // Quantized weights on GPU
  void *d_dequantized_buffer_{nullptr}; // Dequantized FP16 weights (optional)
  size_t quantized_buffer_size_{0};
  size_t dequantized_buffer_size_{0};
  DequantizedCachePolicy dequantized_cache_policy_{
      DequantizedCachePolicy::kNone};
  bool has_dequantized_entries_{
      false}; // Fast bail-out for ClearDequantizedCache

  // File path
  std::filesystem::path model_path_;

  // Weight accessor cache
  mutable std::unordered_map<std::string, std::shared_ptr<IWeightAccessor>>
      weight_accessor_cache_;
};

/**
 * @brief GGUF weight accessor
 *
 * Implements IWeightAccessor for GGUF tensors.
 * Provides lazy dequantization and GPU caching.
 */
class GGUFWeightAccessor : public IWeightAccessor {
public:
  explicit GGUFWeightAccessor(GGUFTensorData *tensor,
                              bool *dequant_dirty_flag = nullptr);

  // IWeightAccessor interface
  std::pair<size_t, size_t> GetDimensions() const override;
  std::string GetDataType() const override;
  bool IsQuantized() const override;
  void *GetGpuWeights(cudaStream_t stream) override;
  half *GetDequantizedGpuWeights(cudaStream_t stream) override;
  bool IsDequantizedCached() const override;

private:
  GGUFTensorData *tensor_;
  bool *dequant_dirty_flag_{nullptr};
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
