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
 * Used by InferfluxCudaExecutor to call ClearSequence without knowing the
 * dtype.
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

  /// Device pointer table: slot_base_ptrs[seq_id] → base of that slot's KV
  /// memory. Used by indirect kernels for hybrid KV cache support.
  /// Returns nullptr if not available.
  virtual void *SlotBasePtrsDevice() const { return nullptr; }
  /// True if the KV cache is a single contiguous buffer (stride-based kernels
  /// can be used). False for hybrid/paged caches requiring indirect kernels.
  virtual bool IsContiguous() const { return false; }
  /// Contiguous buffer base, or nullptr for non-contiguous caches.
  virtual void *ContiguousBuffer() const { return nullptr; }
  virtual size_t SlotStride() const = 0;
  virtual size_t LayerStride() const = 0;
  virtual size_t KvStride() const = 0;
  virtual int KvDim() const = 0;

  /// Type-erased K/V access for use from templated forward pass code.
  /// Returns device pointer to K cache for (layer, seq_id).
  virtual void *GetKVoid(int layer, int seq_id) const = 0;
  /// Returns device pointer to V cache for (layer, seq_id).
  virtual void *GetVVoid(int layer, int seq_id) const = 0;
  /// Type-erased KV append.
  virtual cudaError_t AppendVoid(int layer, int seq_id, int start_pos,
                                 int seq_len, const void *k_new,
                                 const void *v_new, cudaStream_t stream) = 0;
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
  bool IsContiguous() const override { return true; }
  void *ContiguousBuffer() const override { return buffer_; }
  size_t SlotStride() const override { return slot_stride_; }
  size_t LayerStride() const override { return layer_stride_; }
  size_t KvStride() const override { return kv_stride_; }
  int KvDim() const override { return kv_dim_; }
  void *SlotBasePtrsDevice() const override { return d_slot_base_ptrs_; }

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

  void *GetKVoid(int layer, int seq_id) const override {
    return GetK(layer, seq_id);
  }
  void *GetVVoid(int layer, int seq_id) const override {
    return GetV(layer, seq_id);
  }
  cudaError_t AppendVoid(int layer, int seq_id, int start_pos, int seq_len,
                         const void *k_new, const void *v_new,
                         cudaStream_t stream) override {
    return Append(layer, seq_id, start_pos, seq_len,
                  static_cast<const T *>(k_new), static_cast<const T *>(v_new),
                  stream);
  }

private:
  T *buffer_{nullptr};
  T **d_slot_base_ptrs_{nullptr};
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
