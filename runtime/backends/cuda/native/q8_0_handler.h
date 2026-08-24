#pragma once

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
#include <memory>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief Q8_0 quantization handler
 *
 * Q8_0 format:
 * - Block size: 32 values
 * - Block structure: 34 bytes (scale + 32 int8 values)
 * - Bits per value: 8.0
 * - Compression ratio: 2x (vs FP16)
 *
 * Block layout (from ggml-common.h):
 *   struct block_q8_0 {
 *     half d;        // scale (delta), 2 bytes
 *     int8_t qs[32]; // quantized values, 32 bytes
 *   };              // Total: 34 bytes
 *
 * This is the simplest and most common 8-bit quantization format.
 */
class Q8_0_Handler : public IQuantizationHandler {
public:
  Q8_0_Handler() = default;
  ~Q8_0_Handler() override = default;

  /**
   * @brief Dequantize Q8_0 weights from GPU to GPU
   *
   * Converts Q8_0 quantized weights to FP16 on GPU.
   *
   * Formula: dequantized[i] = qs[i] * d
   * where d is the scale (delta)
   */
  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements, cudaStream_t stream) override;

  /**
   * @brief Get quantization type identifier
   */
  std::string GetType() const override { return "q8_0"; }

  /**
   * @brief Get dequantized size from quantized size
   *
   * Q8_0: 34 bytes → 32 values (64 bytes FP16)
   */
  size_t GetDequantizedSize(size_t quantized_size) const override {
    // 34 bytes = 32 values * 1 byte + 2 bytes scale
    size_t num_blocks = quantized_size / 34;
    return num_blocks * 32 * sizeof(half);
  }

  /**
   * @brief Calculate bits per value
   *
   * Q8_0: 32 values in 34 bytes = 272 bits = 8.5 bits/value
   * Actually: 8 bits for values + small overhead for scale
   */
  double GetBitsPerValue() const override { return 8.5; }
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
