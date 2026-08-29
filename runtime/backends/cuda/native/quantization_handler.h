#pragma once

#include "runtime/backends/cuda/native/model_loader.h"

#ifdef INFERFLUX_HAS_CUDA
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
#endif

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief Registry for quantization handlers
 *
 * Allows registration and creation of quantization handlers by type.
 * Implements the Factory pattern with extensible registration.
 */
class QuantizationHandlerRegistry {
public:
  using FactoryFunc = std::function<std::shared_ptr<IQuantizationHandler>()>;

  /**
   * @brief Get singleton instance
   */
  static QuantizationHandlerRegistry &Instance();

  /**
   * @brief Register a quantization handler
   * @param type Quantization type (e.g., "q4_k_m")
   * @param factory Factory function to create handler
   */
  void Register(const std::string &type, FactoryFunc factory);

  /**
   * @brief Create a quantization handler
   * @param type Quantization type
   * @return Handler instance, or nullptr if type not registered
   */
  std::shared_ptr<IQuantizationHandler> Create(const std::string &type) const;

  /**
   * @brief Check if a type is registered
   */
  bool IsRegistered(const std::string &type) const;

  /**
   * @brief Get all registered types
   */
  std::vector<std::string> GetRegisteredTypes() const;

private:
  QuantizationHandlerRegistry() = default;

  std::unordered_map<std::string, FactoryFunc> factories_;
};

/**
 * @brief Helper class for automatic registration
 *
 * Usage in handler implementation files:
 *   static QuantizationHandlerRegistrar<Q4_K_M_Handler> registrar("q4_k_m");
 */
template <typename T> class QuantizationHandlerRegistrar {
public:
  QuantizationHandlerRegistrar(const std::string &type) {
    QuantizationHandlerRegistry::Instance().Register(
        type, []() { return std::make_shared<T>(); });
  }
};

/**
 * @brief Base quantization handler with common functionality
 *
 * Provides utility methods for subclasses.
 */
class BaseQuantizationHandler : public IQuantizationHandler {
public:
  /**
   * @brief Calculate block size for quantization type
   * @param type Quantization type
   * @return Block size (number of values per block)
   */
  static size_t GetBlockSize(const std::string &type);

  /**
   * @brief Calculate quantized size from element count
   * @param num_elements Number of elements
   * @param type Quantization type
   * @return Size in bytes
   */
  static size_t GetQuantizedSize(size_t num_elements, const std::string &type);

protected:
  /**
   * @brief Validate input parameters
   */
  bool ValidateInputs(const void *quantized, half *dequantized,
                      size_t num_elements) const;
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
