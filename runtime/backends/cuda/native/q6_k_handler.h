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
 * @brief Q6_K quantization handler
 *
 * Handles dequantization of Q6_K format (6.5625 bits per weight).
 *
 * Block structure:
 * - 256 values per block
 * - 16 blocks of 16 values each
 * - Each block has 8-bit scale
 * - 6-bit quantized values (4 low bits + 2 high bits)
 * - Super-block scale (FP16)
 */
class Q6_K_Handler : public IQuantizationHandler {
public:
  Q6_K_Handler() = default;
  ~Q6_K_Handler() override = default;

  // IQuantizationHandler interface
  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements, cudaStream_t stream) override;

  std::string GetType() const override { return "q6_k"; }

  size_t GetDequantizedSize(size_t quantized_size) const override;

  double GetBitsPerValue() const override { return 6.5625; }
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
