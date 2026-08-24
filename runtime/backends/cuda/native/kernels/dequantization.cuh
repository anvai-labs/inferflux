#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// GGUF block structures (from ggml-common.h)
#define QK_K 256
#define K_SCALE_SIZE 12

// Q4_K block: 4.5 bits per weight
typedef struct {
  unsigned short d;    // super-block scale for quantized scales (as uint16)
  unsigned short dmin; // super-block scale for quantized mins (as uint16)
  unsigned char scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  unsigned char qs[QK_K / 2];         // 4-bit quants
} block_q4_k;
static_assert(sizeof(block_q4_k) ==
                  2 * sizeof(unsigned short) + K_SCALE_SIZE + QK_K / 2,
              "block_q4_k size mismatch with GGML");

// Q5_K block: 5.5 bits per weight
typedef struct {
  unsigned short d;    // super-block scale for quantized scales (as uint16)
  unsigned short dmin; // super-block scale for quantized mins (as uint16)
  unsigned char scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  unsigned char qh[QK_K / 8];         // quants, high bit
  unsigned char qs[QK_K / 2];         // quants, low 4 bits
} block_q5_k;
static_assert(sizeof(block_q5_k) == 2 * sizeof(unsigned short) + K_SCALE_SIZE +
                                        QK_K / 2 + QK_K / 8,
              "block_q5_k size mismatch with GGML");

// Q6_K block: 6.5625 bits per weight
// Field order matches llama.cpp block_q6_K: ql, qh, scales, d (d is LAST)
typedef struct {
  unsigned char ql[QK_K / 2]; // quants, lower 4 bits (128 bytes)
  unsigned char qh[QK_K / 4]; // quants, upper 2 bits (64 bytes)
  char scales[QK_K / 16];     // scales, quantized with 8 bits (16 bytes)
  unsigned short d;           // super-block scale (as uint16, 2 bytes)
} block_q6_k;
static_assert(sizeof(block_q6_k) ==
                  sizeof(unsigned short) + QK_K / 16 + 3 * QK_K / 4,
              "block_q6_k size mismatch with GGML");

//==============================================================================
// Q4_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q4_K dequantization
 *
 * Dequantizes Q4_K_M format (4.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q4_k_m
 *
 * @param quantized Input quantized weights
 * @param dequantized Output FP16 weights
 * @param num_elements Number of elements to dequantize
 */
__global__ void dequantize_q4_k_kernel(const block_q4_k *quantized,
                                       half *dequantized, size_t num_elements);

/**
 * @brief Host wrapper for Q4_K dequantization
 */
cudaError_t dequantize_q4_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream = 0);

//==============================================================================
// Q5_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q5_K dequantization
 *
 * Dequantizes Q5_K_M format (5.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q5_k_m
 */
__global__ void dequantize_q5_k_kernel(const block_q5_k *quantized,
                                       half *dequantized, size_t num_elements);

/**
 * @brief Host wrapper for Q5_K dequantization
 */
cudaError_t dequantize_q5_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream = 0);

//==============================================================================
// Q6_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q6_K dequantization
 *
 * Dequantizes Q6_K format (6.5625 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q6_k
 */
__global__ void dequantize_q6_k_kernel(const block_q6_k *quantized,
                                       half *dequantized, size_t num_elements);

/**
 * @brief Host wrapper for Q6_K dequantization
 */
cudaError_t dequantize_q6_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream = 0);

//==============================================================================
// Q8_0 Dequantization
//==============================================================================

// Q8_0 block constants
#define QK8_0 32

// Q8_0 block: 8 bits per weight
typedef struct {
  unsigned short d;      // scale (delta), 2 bytes
  signed char qs[QK8_0]; // 8-bit quants, 32 bytes
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(unsigned short) + QK8_0,
              "block_q8_0 size mismatch with GGML");

// Q8_1 block: activation quantization format (matches llama.cpp block_q8_1)
// Per-32-element block with packed scale+sum for efficient dot products.
// ds.x = d (scale = max|x| / 127), ds.y = s (d * sum(qs))
// The pre-computed sum field eliminates dp4a(0x01010101, x, 0) for Q4_K dmin
// correction terms.
#define QK8_1 32
typedef struct {
  half2 ds;            // {d=scale, s=d*sum(qs)} packed as half2
  int8_t qs[QK8_1];   // 32 quantized int8 activation values
} block_q8_1;
static_assert(sizeof(block_q8_1) == sizeof(half2) + QK8_1,
              "block_q8_1 size mismatch with llama.cpp");

/**
 * @brief CUDA kernel for Q8_0 dequantization
 *
 * Dequantizes Q8_0 format (8.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q8_0
 *
 * Formula: dequantized[i] = qs[i] * d
 *
 * @param quantized Input quantized weights
 * @param dequantized Output FP16 weights
 * @param num_elements Number of elements to dequantize
 */
__global__ void dequantize_q8_0_kernel(const block_q8_0 *quantized,
                                       half *dequantized, size_t num_elements);

/**
 * @brief Host wrapper for Q8_0 dequantization
 */
cudaError_t dequantize_q8_0(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream = 0);

//==============================================================================
// Q8_K Dequantization
//==============================================================================

// Q8_K block: 8 bits per weight, 256-element super-blocks
// Used as intermediate quantization in K-series models
typedef struct {
  float d;                // delta (scale), 4 bytes
  signed char qs[QK_K];   // 8-bit quants, 256 bytes
  short bsums[QK_K / 16]; // sum of quants in groups of 16, 32 bytes
} block_q8_k;
static_assert(sizeof(block_q8_k) ==
                  sizeof(float) + QK_K + QK_K / 16 * sizeof(short),
              "block_q8_k size mismatch with GGML");
// Total: 4 + 256 + 32 = 292 bytes

/**
 * @brief CUDA kernel for Q8_K dequantization
 *
 * Dequantizes Q8_K format (8 bits per weight, 256-element blocks) to FP16.
 *
 * Formula: dequantized[i] = qs[i] * d
 *
 * @param quantized Input quantized weights
 * @param dequantized Output FP16 weights
 * @param num_elements Number of elements to dequantize
 */
__global__ void dequantize_q8_k_kernel(const block_q8_k *quantized,
                                       half *dequantized, size_t num_elements);

/**
 * @brief Host wrapper for Q8_K dequantization
 */
cudaError_t dequantize_q8_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream = 0);

//==============================================================================
// Utility Functions
//==============================================================================

/**
 * @brief Calculate grid dimensions for dequantization
 */
inline dim3 calc_dequant_grid(size_t num_elements) {
  int num_blocks = (num_elements + QK_K - 1) / QK_K;
  return dim3((num_blocks + 255) / 256);
}

/**
 * @brief Calculate block dimensions for dequantization
 */
inline dim3 calc_dequant_block() { return dim3(256); }

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
