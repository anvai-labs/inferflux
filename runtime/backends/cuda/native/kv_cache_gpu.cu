#include "runtime/backends/cuda/native/kv_cache_gpu.h"

#include "server/logging/logger.h"
#include <algorithm>
#include <cstring>
#include <cuda_bf16.h>

namespace inferflux {

template <typename T> KvCacheGpuTyped<T>::~KvCacheGpuTyped() {
  if (buffer_) {
    cudaFree(buffer_);
    buffer_ = nullptr;
  }
}

template <typename T>
bool KvCacheGpuTyped<T>::Allocate(int num_layers, int num_kv_heads,
                                  int head_dim, int max_seq_len,
                                  int max_batch_size) {
  num_layers_ = num_layers;
  num_kv_heads_ = num_kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = num_kv_heads * head_dim;
  max_seq_len_ = max_seq_len;
  max_batch_size_ = max_batch_size;

  // Layout: [batch][layer][2][seq_len][kv_dim]
  kv_stride_ = static_cast<size_t>(max_seq_len) * kv_dim_;
  layer_stride_ = 2 * kv_stride_;
  slot_stride_ = static_cast<size_t>(num_layers) * layer_stride_;

  size_t total_elements = static_cast<size_t>(max_batch_size) * slot_stride_;
  total_bytes_ = total_elements * sizeof(T);

  cudaError_t err = cudaMalloc(&buffer_, total_bytes_);
  if (err != cudaSuccess) {
    log::Error("kv_cache_gpu",
               "cudaMalloc failed for KV cache: " +
                   std::string(cudaGetErrorString(err)) + " (size=" +
                   std::to_string(total_bytes_ / (1024 * 1024)) + " MB)");
    return false;
  }

  cudaMemset(buffer_, 0, total_bytes_);

  log::Info(
      "kv_cache_gpu",
      "Allocated KV cache: " + std::to_string(total_bytes_ / (1024 * 1024)) +
          " MB (layers=" + std::to_string(num_layers) +
          ", kv_heads=" + std::to_string(num_kv_heads) +
          ", head_dim=" + std::to_string(head_dim) +
          ", max_seq=" + std::to_string(max_seq_len) +
          ", max_batch=" + std::to_string(max_batch_size) + ")");
  return true;
}

template <typename T> T *KvCacheGpuTyped<T>::GetK(int layer, int seq_id) const {
  return buffer_ + seq_id * slot_stride_ + layer * layer_stride_;
}

template <typename T> T *KvCacheGpuTyped<T>::GetV(int layer, int seq_id) const {
  return buffer_ + seq_id * slot_stride_ + layer * layer_stride_ + kv_stride_;
}

template <typename T>
cudaError_t KvCacheGpuTyped<T>::Append(int layer, int seq_id, int start_pos,
                                       int seq_len, const T *k_new,
                                       const T *v_new, cudaStream_t stream) {
  size_t copy_bytes = static_cast<size_t>(seq_len) * kv_dim_ * sizeof(T);

  T *k_dst = GetK(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;
  T *v_dst = GetV(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;

  cudaError_t err = cudaMemcpyAsync(k_dst, k_new, copy_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess)
    return err;

  err = cudaMemcpyAsync(v_dst, v_new, copy_bytes, cudaMemcpyDeviceToDevice,
                        stream);
  return err;
}

template <typename T> void KvCacheGpuTyped<T>::ClearSequence(int seq_id) {
  if (!buffer_ || seq_id >= max_batch_size_)
    return;
  size_t bytes = slot_stride_ * sizeof(T);
  cudaMemset(buffer_ + seq_id * slot_stride_, 0, bytes);
}

template <typename T>
void KvCacheGpuTyped<T>::ClearSequenceAsync(int seq_id, cudaStream_t stream) {
  if (!buffer_ || seq_id >= max_batch_size_)
    return;
  size_t bytes = slot_stride_ * sizeof(T);
  cudaMemsetAsync(buffer_ + seq_id * slot_stride_, 0, bytes, stream);
}

template <typename T>
bool KvCacheGpuTyped<T>::CopySequencePrefix(int src_seq, int dst_seq,
                                            int n_tokens,
                                            cudaStream_t stream) {
  if (!buffer_ || src_seq < 0 || dst_seq < 0 || src_seq >= max_batch_size_ ||
      dst_seq >= max_batch_size_) {
    return false;
  }
  if (n_tokens <= 0 || src_seq == dst_seq) {
    return true;
  }
  const int copy_tokens = std::min(n_tokens, max_seq_len_);
  const size_t copy_bytes =
      static_cast<size_t>(copy_tokens) * kv_dim_ * sizeof(T);
  for (int layer = 0; layer < num_layers_; ++layer) {
    T *k_src = GetK(layer, src_seq);
    T *k_dst = GetK(layer, dst_seq);
    cudaError_t err = cudaMemcpyAsync(k_dst, k_src, copy_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      return false;
    }
    T *v_src = GetV(layer, src_seq);
    T *v_dst = GetV(layer, dst_seq);
    err = cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyDeviceToDevice,
                          stream);
    if (err != cudaSuccess) {
      return false;
    }
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

template <typename T>
bool KvCacheGpuTyped<T>::SerializeSequence(int seq_id,
                                           std::vector<uint8_t> *out) const {
  if (!out || !buffer_ || seq_id < 0 || seq_id >= max_batch_size_) {
    return false;
  }
  const size_t bytes = slot_stride_ * sizeof(T);
  out->resize(bytes);
  const T *src = buffer_ + static_cast<size_t>(seq_id) * slot_stride_;
  return cudaMemcpy(out->data(), src, bytes, cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

template <typename T>
bool KvCacheGpuTyped<T>::HydrateSequence(int seq_id,
                                         const std::vector<uint8_t> &blob,
                                         cudaStream_t stream) {
  if (!buffer_ || seq_id < 0 || seq_id >= max_batch_size_) {
    return false;
  }
  const size_t bytes = slot_stride_ * sizeof(T);
  if (blob.size() != bytes) {
    return false;
  }
  T *dst = buffer_ + static_cast<size_t>(seq_id) * slot_stride_;
  if (cudaMemcpyAsync(dst, blob.data(), bytes, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

template <typename T>
void KvCacheGpuTyped<T>::GetBatchAppendPtrs(int layer, const int *seq_ids,
                                            const int *n_past, int batch_size,
                                            T **h_k_ptrs,
                                            T **h_v_ptrs) const {
  for (int b = 0; b < batch_size; ++b) {
    h_k_ptrs[b] =
        GetK(layer, seq_ids[b]) + static_cast<size_t>(n_past[b]) * kv_dim_;
    h_v_ptrs[b] =
        GetV(layer, seq_ids[b]) + static_cast<size_t>(n_past[b]) * kv_dim_;
  }
}

template <typename T>
void KvCacheGpuTyped<T>::GetBatchKVPtrs(int layer, const int *seq_ids,
                                        int batch_size, const T **h_k_ptrs,
                                        const T **h_v_ptrs) const {
  for (int b = 0; b < batch_size; ++b) {
    h_k_ptrs[b] = GetK(layer, seq_ids[b]);
    h_v_ptrs[b] = GetV(layer, seq_ids[b]);
  }
}

// Explicit template instantiations
template class KvCacheGpuTyped<half>;
template class KvCacheGpuTyped<__nv_bfloat16>;

} // namespace inferflux
