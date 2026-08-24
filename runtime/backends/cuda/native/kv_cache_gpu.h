#pragma once

#include <cstddef>
#include <cstdint>
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
#include <vector>

namespace inferflux {

/**
 * IKvCacheGpu: non-templated base interface for sequence management.
 * Used by InferfluxCudaExecutor to call ClearSequence without knowing the dtype.
 */
class IKvCacheGpu {
public:
  virtual ~IKvCacheGpu() = default;
  virtual void ClearSequence(int seq_id) = 0;
  virtual void ClearSequenceAsync(int seq_id, cudaStream_t stream) = 0;
  virtual bool CopySequencePrefix(int src_seq, int dst_seq, int n_tokens,
                                  cudaStream_t stream) = 0;
  virtual bool SerializeSequence(int seq_id,
                                 std::vector<uint8_t> *out) const = 0;
  virtual bool HydrateSequence(int seq_id, const std::vector<uint8_t> &blob,
                               cudaStream_t stream) = 0;
  virtual size_t GetMemoryUsage() const = 0;
  virtual int MaxSeqLen() const = 0;
  virtual int MaxBatchSize() const = 0;
};

/**
 * KvCacheGpuTyped<T>: pre-allocated contiguous GPU memory for KV cache.
 *
 * Layout: [max_batch][num_layers][2 (K/V)][max_seq_len][kv_dim]
 * where kv_dim = num_kv_heads * head_dim.
 */
template <typename T> class KvCacheGpuTyped : public IKvCacheGpu {
public:
  KvCacheGpuTyped() = default;
  ~KvCacheGpuTyped() override;

  KvCacheGpuTyped(const KvCacheGpuTyped &) = delete;
  KvCacheGpuTyped &operator=(const KvCacheGpuTyped &) = delete;

  bool Allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                int max_batch_size);

  T *GetK(int layer, int seq_id) const;
  T *GetV(int layer, int seq_id) const;

  cudaError_t Append(int layer, int seq_id, int start_pos, int seq_len,
                     const T *k_new, const T *v_new, cudaStream_t stream);

  /**
   * Compute K/V destination pointers for batched append.
   * Fills h_k_ptrs and h_v_ptrs with pointers to the start of each
   * sequence's KV row at the given layer and position.
   */
  void GetBatchAppendPtrs(int layer, const int *seq_ids, const int *n_past,
                          int batch_size, T **h_k_ptrs, T **h_v_ptrs) const;

  /**
   * Compute K/V cache base pointers for batched FlashDecode.
   * Returns pointers to each sequence's K and V cache for the given layer.
   */
  void GetBatchKVPtrs(int layer, const int *seq_ids, int batch_size,
                      const T **h_k_ptrs, const T **h_v_ptrs) const;

  T *Buffer() const { return buffer_; }
  size_t SlotStride() const { return slot_stride_; }
  size_t LayerStride() const { return layer_stride_; }
  size_t KvStride() const { return kv_stride_; }
  int KvDim() const { return kv_dim_; }

  void ClearSequence(int seq_id) override;
  void ClearSequenceAsync(int seq_id, cudaStream_t stream) override;
  bool CopySequencePrefix(int src_seq, int dst_seq, int n_tokens,
                          cudaStream_t stream) override;
  bool SerializeSequence(int seq_id, std::vector<uint8_t> *out) const override;
  bool HydrateSequence(int seq_id, const std::vector<uint8_t> &blob,
                       cudaStream_t stream) override;

  size_t GetMemoryUsage() const override { return total_bytes_; }
  int MaxSeqLen() const override { return max_seq_len_; }
  int MaxBatchSize() const override { return max_batch_size_; }

private:
  T *buffer_{nullptr};
  size_t total_bytes_{0};

  int num_layers_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int kv_dim_{0};
  int max_seq_len_{0};
  int max_batch_size_{0};

  size_t slot_stride_{0};
  size_t layer_stride_{0};
  size_t kv_stride_{0};
};

// Backward-compatible alias
using KvCacheGpu = KvCacheGpuTyped<half>;

} // namespace inferflux
