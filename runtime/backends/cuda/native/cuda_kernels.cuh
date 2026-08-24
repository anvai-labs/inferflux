#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// Templated kernel declarations (BF16 + FP16)
// ============================================================================

template <typename T>
cudaError_t RmsNorm(const T *input, const T *weight, T *output, int count,
                    int hidden_size, float eps, cudaStream_t stream);

// rope_type: 0 = kNorm (consecutive pairs), 2 = kNeox (split-half pairs)
template <typename T>
cudaError_t RoPE(T *q, T *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type = 0);

template <typename T>
cudaError_t SiluMul(const T *gate, const T *up, T *output, int count,
                    cudaStream_t stream);

template <typename T>
cudaError_t ResidualAdd(T *residual, const T *input, int count,
                        cudaStream_t stream);

/**
 * FusedResidualAddRmsNorm: residual += input; output = RmsNorm(residual)
 * Combines two kernels into one, saving a kernel launch per fusion site.
 * count = number of rows, hidden_size = row width.
 */
template <typename T>
cudaError_t ResidualAddRmsNorm(T *residual, const T *input, const T *weight,
                               T *output, int count, int hidden_size, float eps,
                               cudaStream_t stream);

template <typename T>
cudaError_t EmbeddingLookup(const T *table, const int *token_ids, T *output,
                            int seq_len, int hidden_size, cudaStream_t stream);

template <typename T>
cudaError_t HalfToFloat(const T *input, float *output, int count,
                        cudaStream_t stream);

template <typename T>
cudaError_t QuantizeRowsSymmetric(const T *input, int8_t *output,
                                  float *row_scales, int rows, int cols,
                                  cudaStream_t stream);

template <typename T>
cudaError_t SiluMulQuantizeRowsSymmetric(const T *gate, const T *up,
                                         int8_t *output, float *row_scales,
                                         int rows, int cols,
                                         cudaStream_t stream);

/**
 * BiasAdd: output[i] += bias[i % bias_dim] for rows * bias_dim elements.
 * Used to add projection biases after GEMM (e.g., Q/K/V biases in Qwen2).
 */
template <typename T>
cudaError_t BiasAdd(T *output, const T *bias, int rows, int bias_dim,
                    cudaStream_t stream);

// ============================================================================
// Non-templated FP16 overloads (backward compatibility)
// ============================================================================

cudaError_t RmsNorm(const half *input, const half *weight, half *output,
                    int count, int hidden_size, float eps, cudaStream_t stream);

cudaError_t RoPE(half *q, half *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base,
                 cudaStream_t stream);

cudaError_t SiluMul(const half *gate, const half *up, half *output, int count,
                    cudaStream_t stream);

cudaError_t ResidualAdd(half *residual, const half *input, int count,
                        cudaStream_t stream);

cudaError_t ResidualAddRmsNorm(half *residual, const half *input,
                               const half *weight, half *output, int count,
                               int hidden_size, float eps, cudaStream_t stream);

cudaError_t EmbeddingLookup(const half *table, const int *token_ids,
                            half *output, int seq_len, int hidden_size,
                            cudaStream_t stream);

cudaError_t HalfToFloat(const half *input, float *output, int count,
                        cudaStream_t stream);

cudaError_t QuantizeRowsSymmetric(const half *input, int8_t *output,
                                  float *row_scales, int rows, int cols,
                                  cudaStream_t stream);

cudaError_t SiluMulQuantizeRowsSymmetric(const half *gate, const half *up,
                                         int8_t *output, float *row_scales,
                                         int rows, int cols,
                                         cudaStream_t stream);

// ============================================================================
// Batched kernels for multi-sequence decode
// ============================================================================

/**
 * BatchedRoPE: Apply RoPE to B sequences with different n_past values.
 * q layout: [B, num_heads * head_dim], k layout: [B, num_kv_heads * head_dim]
 * d_n_past: [B] per-sequence positions on device.
 * Optional q_bias/k_bias ([heads * head_dim]) are added to the loaded
 * values before rotation — the fused equivalent of BiasAdd-then-RoPE.
 */
template <typename T>
cudaError_t BatchedRoPE(T *q, T *k, int batch_size, int num_heads,
                        int num_kv_heads, int head_dim, const int *d_n_past,
                        float freq_base, cudaStream_t stream, int rope_type = 0,
                        const T *q_bias = nullptr, const T *k_bias = nullptr);

/**
 * BatchedKvAppend: Scatter-copy K/V for B sequences into KV cache.
 * k_new/v_new layout: [B, kv_dim]
 * d_k_dst/d_v_dst: [B] device pointers to each sequence's K/V row.
 */
template <typename T>
cudaError_t BatchedKvAppend(const T *k_new, const T *v_new, T **d_k_dst,
                            T **d_v_dst, int batch_size, int kv_dim,
                            cudaStream_t stream);

/**
 * BatchedKvAppendStrided: Scatter-copy K/V for B sequences into a regular KV
 * cache layout derived on device.
 * k_new/v_new layout: [B, kv_dim]
 * kv_buffer layout: [max_batch][num_layers][2][max_seq_len][kv_dim]
 * d_seq_ids/d_n_past: [B] sequence IDs and append positions on device.
 */
template <typename T>
cudaError_t BatchedKvAppendStrided(const T *k_new, const T *v_new, T *kv_buffer,
                                   const int *d_seq_ids, const int *d_n_past,
                                   int layer, int batch_size, int kv_dim,
                                   size_t slot_stride, size_t layer_stride,
                                   size_t kv_stride, cudaStream_t stream,
                                   const T *v_bias = nullptr);

// ============================================================================
// Embedding extraction kernels
// ============================================================================

/**
 * MeanPool: Average hidden states across token positions.
 * input layout: [seq_len, hidden_size] (T, e.g. half or nv_bfloat16)
 * output layout: [hidden_size] (float)
 * Each output[i] = (1/seq_len) * sum_{t=0..seq_len-1} input[t*hidden_size + i]
 */
template <typename T>
cudaError_t MeanPool(const T *input, float *output, int seq_len,
                     int hidden_size, cudaStream_t stream);

} // namespace cuda_kernel
} // namespace inferflux
