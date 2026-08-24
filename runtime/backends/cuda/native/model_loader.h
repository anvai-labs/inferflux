#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
// Forward declarations for non-CUDA builds
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

enum class DequantizedCachePolicy {
  kNone,
  kModelLifetime,
  kBatchLifetime,
};

/**
 * RoPE pairing strategy.
 * - kNorm: consecutive pairs (0,1),(2,3),... — LLaMA, Mistral, Baichuan, etc.
 * - kNeox: split-half pairs (0,d/2),(1,d/2+1),... — Falcon, Qwen, GPT-NeoX,
 * etc.
 */
enum class RopeType {
  kNorm = 0, // GGML_ROPE_TYPE_NORMAL — default for LLaMA family
  kNeox = 2, // GGML_ROPE_TYPE_NEOX   — Falcon, Qwen, GPT-NeoX, etc.
};

/** Infer RoPE type from model architecture name (matches llama.cpp). */
inline RopeType InferRopeType(const std::string &model_type) {
  // NEOX-style: split-half pairing
  if (model_type == "falcon" || model_type == "gptneox" ||
      model_type == "qwen" || model_type == "qwen2" ||
      model_type == "qwen2moe" || model_type == "qwen3" ||
      model_type == "qwen3moe" || model_type == "phi2" ||
      model_type == "phi3" || model_type == "stablelm" ||
      model_type == "starcoder2" || model_type == "gemma" ||
      model_type == "gemma2" || model_type == "gemma3" ||
      model_type == "codeshell" || model_type == "openelm" ||
      model_type == "plamo" || model_type == "bert" ||
      model_type == "nomic-bert") {
    return RopeType::kNeox;
  }
  // NORM-style: consecutive pairing (default for LLaMA and most models)
  return RopeType::kNorm;
}

std::string DequantizedCachePolicyToString(DequantizedCachePolicy policy);
bool ParseDequantizedCachePolicy(const std::string &raw,
                                 DequantizedCachePolicy *out);

// Forward declarations
class IWeightAccessor;

/**
 * @brief Model information structure
 *
 * Contains metadata about the loaded model.
 */
struct ModelInfo {
  // Model architecture
  int hidden_size{0};
  int num_hidden_layers{0};
  int num_attention_heads{0};
  int num_key_value_heads{0}; // For GQA
  int head_dim{0};
  int intermediate_size{0};
  int vocab_size{0};
  int max_position_embeddings{0};

  // RoPE settings
  float rope_freq_base{10000.0f};
  float rope_freq_scale{1.0f};
  int rope_dim{0};
  RopeType rope_type{RopeType::kNorm};

  // Model type
  std::string model_type; // "qwen2", "llama", etc.
  std::string activation; // "silu", "swiglu", etc.

  // Dtype and normalization
  std::string torch_dtype; // "bfloat16", "float16", etc.
  float rms_norm_eps{1e-6f};
};

/**
 * @brief Abstract interface for model loaders
 *
 * Provides a unified interface for loading models from different formats
 * (GGUF, safetensors) while maintaining format-specific optimizations.
 *
 * This interface follows the Interface Segregation Principle by providing
 * focused methods for loading and accessing model data.
 */
class IModelLoader {
public:
  virtual ~IModelLoader() = default;

  /**
   * @brief Load model from file
   * @param model_path Path to model file or directory
   * @return true if successful
   */
  virtual bool Load(const std::filesystem::path &model_path) = 0;

  /**
   * @brief Get model information
   * @return ModelInfo structure with architecture, parameters, etc.
   */
  virtual const ModelInfo &GetModelInfo() const = 0;

  /**
   * @brief Get model format
   * @return Model format (gguf, safetensors, etc.)
   */
  virtual std::string GetFormat() const = 0;

  /**
   * @brief Check if model uses quantization
   * @return true if model is quantized
   */
  virtual bool IsQuantized() const = 0;

  /**
   * @brief Get quantization type
   * @return Quantization type (q4_k_m, q5_k_m, q6_k, etc.) or empty if not
   * quantized
   */
  virtual std::string GetQuantizationType() const = 0;

  /**
   * @brief Upload all weights to GPU memory
   * @param stream CUDA stream for async operations
   * @return true if successful
   */
  virtual bool UploadToGPU(cudaStream_t stream) = 0;

  /**
   * @brief Free CPU memory (after GPU upload)
   */
  virtual void FreeCPUMemory() = 0;

  /**
   * @brief Free GPU memory
   */
  virtual void FreeGPUMemory() = 0;

  /**
   * @brief Get GPU weights buffer
   * @return Pointer to contiguous GPU buffer
   */
  virtual void *GetGPUBuffer() const = 0;

  /**
   * @brief Get total GPU size
   * @return Size of GPU buffer in bytes
   */
  virtual size_t GetGPUSize() const = 0;

  /**
   * @brief Configure lifecycle for dequantized temporary GPU weights
   *
   * none: memory-first, reclaimed at request boundary in runtime cleanup.
   * batch: reclaimed at batch boundary.
   * model: maximize reuse, highest VRAM.
   */
  virtual void SetDequantizedCachePolicy(DequantizedCachePolicy policy) = 0;

  /**
   * @brief Inspect active dequantized cache lifecycle policy
   */
  virtual DequantizedCachePolicy GetDequantizedCachePolicy() const = 0;

  /**
   * @brief Clear cached dequantized GPU weights, if any
   */
  virtual void ClearDequantizedCache() = 0;

  /**
   * @brief Get weight accessor for a specific tensor
   * @param tensor_name Name of tensor
   * @return Shared pointer to weight accessor, or nullptr if not found
   */
  virtual std::shared_ptr<IWeightAccessor>
  GetWeightAccessor(const std::string &tensor_name) = 0;
};

/**
 * @brief Abstract interface for weight access
 *
 * Provides unified access to weights regardless of storage format
 * (FP16, BF16, Q4_K_M, Q5_K_M, Q6_K, etc.)
 *
 * This interface allows lazy dequantization and GPU caching strategies.
 */
class IWeightAccessor {
public:
  virtual ~IWeightAccessor() = default;

  /**
   * @brief Get weight dimensions
   * @return Pair of {rows, cols}
   */
  virtual std::pair<size_t, size_t> GetDimensions() const = 0;

  /**
   * @brief Get weight data type
   * @return Data type (f16, bf16, q4_k_m, etc.)
   */
  virtual std::string GetDataType() const = 0;

  /**
   * @brief Check if weights are quantized
   * @return true if weights use quantization
   */
  virtual bool IsQuantized() const = 0;

  /**
   * @brief Get GPU pointer to weights
   *
   * For non-quantized weights, returns direct GPU pointer.
   * For quantized weights, returns pointer to quantized data.
   *
   * @param stream CUDA stream for async operations
   * @return Pointer to GPU memory
   */
  virtual void *GetGpuWeights(cudaStream_t stream) = 0;

  /**
   * @brief Get dequantized weights on GPU
   *
   * For non-quantized weights, returns direct GPU pointer.
   * For quantized weights, performs lazy dequantization and caches result.
   *
   * @param stream CUDA stream for async operations
   * @return Pointer to GPU memory with FP16 weights
   */
  virtual half *GetDequantizedGpuWeights(cudaStream_t stream) = 0;

  /**
   * @brief Check if dequantized weights are cached on GPU
   * @return true if dequantized weights are already cached
   */
  virtual bool IsDequantizedCached() const = 0;
};

/**
 * @brief Abstract interface for quantization handlers
 *
 * Strategy pattern for handling different quantization types.
 * Each handler knows how to dequantize its specific format.
 *
 * This follows the Open/Closed Principle: new quantization types
 * can be added without modifying existing code.
 */
class IQuantizationHandler {
public:
  virtual ~IQuantizationHandler() = default;

  /**
   * @brief Dequantize weights from GPU to GPU
   * @param quantized Input quantized weights on GPU
   * @param dequantized Output dequantized weights on GPU (FP16)
   * @param num_elements Number of elements to dequantize
   * @param stream CUDA stream for async operations
   */
  virtual void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                                  size_t num_elements, cudaStream_t stream) = 0;

  /**
   * @brief Get quantization type
   * @return Quantization type identifier
   */
  virtual std::string GetType() const = 0;

  /**
   * @brief Get dequantized size
   * @param quantized_size Size of quantized data in bytes
   * @return Size of dequantized data in bytes (FP16)
   */
  virtual size_t GetDequantizedSize(size_t quantized_size) const = 0;

  /**
   * @brief Calculate bits per value
   * @return Average bits per value for this quantization
   */
  virtual double GetBitsPerValue() const = 0;
};

/**
 * @brief Factory function for creating model loaders
 *
 * Detects model format from file path/extension and creates
 * the appropriate loader implementation.
 *
 * @param model_path Path to model file or directory
 * @return Unique pointer to loader, or nullptr if format not detected
 */
std::unique_ptr<IModelLoader>
CreateModelLoader(const std::filesystem::path &model_path);

/**
 * @brief Factory function for creating quantization handlers
 *
 * @param quantization_type Quantization type (e.g., "q4_k_m", "q5_k_m")
 * @return Shared pointer to handler, or nullptr if type not supported
 */
std::shared_ptr<IQuantizationHandler>
CreateQuantizationHandler(const std::string &quantization_type);

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
