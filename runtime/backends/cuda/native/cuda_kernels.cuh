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

/**
 * BiasAddTriple: fused bias addition for three output tensors (Q, K, V).
 * Replaces three separate BiasAdd launches with a single kernel launch.
 * Each output tensor can have a different bias dimension.
 */
template <typename T>
cudaError_t BiasAddTriple(T *q, T *k, T *v, const T *q_bias, const T *k_bias,
                          const T *v_bias, int rows, int q_dim, int k_dim,
                          int v_dim, cudaStream_t stream);

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
// Mixed-precision kernels (FP32 residual + FP16 weights/activations)
// ============================================================================

/**
 * RmsNormMixed: reads FP32 residual, normalizes in FP32, writes T output.
 * Used when the residual stream is kept in FP32 for numerical precision.
 */
template <typename T>
cudaError_t RmsNormMixed(const float *input, const T *weight, T *output,
                         int count, int hidden_size, float eps,
                         cudaStream_t stream);

/**
 * ResidualAddMixed: residual(FP32) += input(T).
 * Adds a T-precision input (e.g., O-projection output) into the FP32 residual
 * without truncating to FP16.
 */
template <typename T>
cudaError_t ResidualAddMixed(float *residual, const T *input, int count,
                             cudaStream_t stream);

/**
 * ResidualAddRmsNormMixed: residual(FP32) += input(T); output(T) =
 * RmsNorm(residual). Fused version that keeps the residual in FP32 while
 * outputting the normalized result in T for downstream projections.
 */
template <typename T>
cudaError_t ResidualAddRmsNormMixed(float *residual, const T *input,
                                    const T *weight, T *output, int count,
                                    int hidden_size, float eps,
                                    cudaStream_t stream);

/**
 * EmbeddingLookupF32: look up embeddings from a T table, write FP32 output.
 * Initializes the FP32 residual stream from the embedding table.
 */
template <typename T>
cudaError_t EmbeddingLookupF32(const T *table, const int *token_ids,
                               float *output, int seq_len, int hidden_size,
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

/**
 * BatchedKvAppendIndirect: Same as Strided but uses slot base pointer table.
 * Supports hybrid KV cache where slots may not be contiguous.
 */
template <typename T>
cudaError_t BatchedKvAppendIndirect(const T *k_new, const T *v_new,
                                    T *const *slot_base_ptrs,
                                    const int *d_seq_ids, const int *d_n_past,
                                    int layer, int batch_size, int kv_dim,
                                    size_t layer_stride, size_t kv_stride,
                                    cudaStream_t stream);

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

/**
 * DeviceTokenRelay: Copy sampled token IDs from sampler output to the
 * BatchForward metadata buffer, and increment n_past — entirely on device.
 *
 * This eliminates the per-token D2H → host processing → H2D round-trip
 * that otherwise adds ~10ms of WDDM scheduling latency on Windows.
 *
 * batch_meta layout: [token_ids(B)][n_past(B)][seq_ids(B)][kv_lens(B)]
 * sampled_tokens: output from BatchedArgmaxKernel (B ints)
 *
 * After this kernel, the graph can be replayed immediately without host sync.
 */
cudaError_t DeviceTokenRelay(const int *sampled_tokens, int *batch_meta,
                              int batch_size, int max_batch_size,
                              cudaStream_t stream);

/**
 * AppendTokenToBuffer: Copy one sampled token to a position in an
 * accumulation buffer. Used by BurstDecodeGreedy to collect all N tokens
 * on device without per-token D2H sync.
 */
cudaError_t AppendTokenToBuffer(const int *sampled_token, int *token_buffer,
                                 int position, cudaStream_t stream);

/**
 * DeviceCheckEos: Check if any sampled token matches the EOS token ID.
 * Writes 1 to d_has_eos if any token matches, 0 otherwise.
 * The host can poll this flag periodically instead of syncing per token.
 */
cudaError_t DeviceCheckEos(const int *sampled_tokens, int batch_size,
                            int eos_token_id, int *d_has_eos,
                            cudaStream_t stream);

/**
 * ConvertHalfToFloat: Convert half/bfloat16 tensor to float on device.
 * Used by attention tensor profiling to capture intermediate tensors.
 */
template <typename T>
cudaError_t ConvertHalfToFloat(const T *d_input, float *d_output,
                                size_t num_elements, cudaStream_t stream);

} // namespace cuda_kernel
} // namespace inferflux
