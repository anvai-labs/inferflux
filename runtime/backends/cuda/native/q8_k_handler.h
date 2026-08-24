#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
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
#include <string>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief Q8_K quantization handler
 *
 * Q8_K format:
 * - Block size: 256 values (K-series super-block)
 * - Block structure: 292 bytes (float scale + 256 int8 quants + 16 int16 sums)
 * - Bits per value: ~9.125 (292 * 8 / 256)
 * - Used as intermediate quantization in K-series models
 *
 * Block layout (from ggml-common.h):
 *   struct block_q8_K {
 *     float   d;              // delta (scale), 4 bytes
 *     int8_t  qs[256];        // quantized values, 256 bytes
 *     int16_t bsums[16];      // sum of quants in groups of 16, 32 bytes
 *   };                        // Total: 292 bytes
 */
class Q8_K_Handler : public IQuantizationHandler {
public:
  Q8_K_Handler() = default;
  ~Q8_K_Handler() override = default;

  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements, cudaStream_t stream) override;

  std::string GetType() const override { return "q8_k"; }

  size_t GetDequantizedSize(size_t quantized_size) const override {
    size_t num_blocks = quantized_size / sizeof(block_q8_k);
    size_t num_elements = num_blocks * QK_K;
    return num_elements * sizeof(half);
  }

  // 292 bytes per 256 values = 9.125 bits/value
  double GetBitsPerValue() const override { return 9.125; }
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
