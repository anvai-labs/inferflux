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
 * @brief Q4_K_M quantization handler
 *
 * Handles dequantization of Q4_K_M format (4.5 bits per weight).
 * This is the most common quantization format for GGUF models.
 *
 * Block structure (from ggml-common.h):
 * - 256 values per block
 * - 8 blocks of 32 values each
 * - Each block has 6-bit scale and 6-bit min
 * - 4-bit quantized values
 * - Super-block scale and min (FP16)
 */
class Q4_K_M_Handler : public IQuantizationHandler {
public:
  Q4_K_M_Handler() = default;
  ~Q4_K_M_Handler() override = default;

  // IQuantizationHandler interface
  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements, cudaStream_t stream) override;

  std::string GetType() const override { return "q4_k_m"; }

  size_t GetDequantizedSize(size_t quantized_size) const override;

  double GetBitsPerValue() const override { return 4.5; }
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
