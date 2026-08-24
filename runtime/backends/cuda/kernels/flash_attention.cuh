#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// Tiled FlashAttention-2 with causal mask and GQA support.
template <typename T>
cudaError_t FlashAttention2Typed(const T *Q, const T *K, const T *V, T *O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream = 0);

/**
 * FlashDecodeMultiSeq: Batched decode attention for B sequences with
 * different KV cache lengths. Each query has query_len=1.
 *
 * d_k_ptrs/d_v_ptrs: [B] device pointers to each sequence's K/V cache
 *   for the current layer (layout: [kv_len, num_kv_heads * head_dim]).
 * d_kv_lens: [B] per-sequence KV lengths on device.
 * Q layout: [B, num_heads * head_dim]
 * O layout: [B, num_heads * head_dim]
 */
template <typename T>
cudaError_t FlashDecodeMultiSeq(const T *Q, const T *const *d_k_ptrs,
                                const T *const *d_v_ptrs, T *O,
                                const int *d_kv_lens, int batch_size,
                                int num_heads, int num_kv_heads, int head_dim,
                                float scale, cudaStream_t stream = 0);

/**
 * FlashDecodeMultiSeqStrided: Batched decode attention for B sequences with
 * different KV lengths, deriving K/V bases on device from a regular KV cache
 * layout.
 *
 * kv_buffer layout: [max_batch][num_layers][2][max_seq_len][kv_dim]
 * d_seq_ids/d_kv_lens: [B] sequence IDs and KV lengths on device.
 */
template <typename T>
cudaError_t FlashDecodeMultiSeqStrided(
    const T *Q, const T *kv_buffer, T *O, const int *d_seq_ids,
    const int *d_kv_lens, int layer, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, size_t slot_stride, size_t layer_stride,
    size_t kv_stride, float scale, cudaStream_t stream = 0);

/**
 * FlashDecodeMultiSeqStridedSplit: split-parallel decode attention.
 *
 * Parallelizes over the Q-heads sharing each KV head (qsplit — targets the
 * short-context case where the monolithic GQA kernel leaves most SMs idle)
 * and over fixed KV chunks of `chunk` positions (ksplits — targets long
 * contexts where serial tile iteration dominates). When ksplits > 1, CTAs
 * write unnormalized online-softmax partials to `d_partials`
 * ([batch][num_heads][ksplits][2 + head_dim] floats) and a combine pass
 * merges them; otherwise output is written directly.
 */
template <typename T>
cudaError_t FlashDecodeMultiSeqStridedSplit(
    const T *Q, const T *kv_buffer, T *O, float *d_partials,
    const int *d_seq_ids, const int *d_kv_lens, int layer, int batch_size,
    int num_heads, int num_kv_heads, int head_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, float scale, int qsplit, int chunk,
    int ksplits, cudaStream_t stream = 0);

} // namespace cuda_kernel
} // namespace inferflux
