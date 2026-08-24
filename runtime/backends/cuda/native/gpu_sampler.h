#pragma once

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
#include <curand.h>
#include <vector>

namespace inferflux {

/**
 * GPU-side sampling: temperature, top-k, top-p, multinomial.
 *
 * For greedy (temperature=0): argmax via parallel reduction.
 * For stochastic: softmax -> top-k -> top-p -> multinomial sample.
 */
class GpuSampler {
public:
  GpuSampler() = default;
  ~GpuSampler();

  GpuSampler(const GpuSampler &) = delete;
  GpuSampler &operator=(const GpuSampler &) = delete;

  /**
   * Initialize with vocab size and CUDA stream.
   */
  bool Initialize(int vocab_size, cudaStream_t stream);

  /**
   * Sample a token from logits.
   *
   * @param d_logits    [vocab_size] FP32 logits on device
   * @param temperature Temperature (0 = greedy)
   * @param top_k       Top-k filtering (0 = disabled)
   * @param top_p       Top-p / nucleus filtering (1.0 = disabled)
   * @param seed        Random seed (UINT32_MAX = auto)
   * @return Sampled token ID
   */
  int Sample(const float *d_logits, float temperature, int top_k, float top_p,
             uint32_t seed = UINT32_MAX);
  void EnqueueSample(const float *d_logits, float temperature, int top_k,
                     float top_p, uint32_t seed = UINT32_MAX);
  int CollectSample();

  /**
   * Copy last-position logits from device to a host buffer.
   * The caller must provide a buffer of at least vocab_size floats.
   * This must be called after Sample() / SampleBatch() and before the next
   * forward pass overwrites the logits buffer.
   *
   * @param d_logits  [vocab_size] FP32 logits on device (same pointer passed
   *                  to Sample)
   * @param host_buf  [vocab_size] FP32 host destination
   */
  void CopyLogitsToHost(const float *d_logits, float *host_buf);

  /**
   * Sample tokens from batched logits.
   *
   * @param d_logits      [batch_size * vocab_size] FP32 logits on device
   * @param batch_size    Number of sequences in batch
   * @param temperatures  Per-sequence temperature
   * @param top_ks        Per-sequence top-k
   * @param top_ps        Per-sequence top-p
   * @param out_tokens    Output: sampled token IDs
   */
  void SampleBatch(const float *d_logits, int batch_size,
                   const std::vector<float> &temperatures,
                   const std::vector<int> &top_ks,
                   const std::vector<float> &top_ps,
                   const std::vector<uint32_t> &seeds,
                   std::vector<int> *out_tokens);
  void EnqueueSampleBatch(const float *d_logits, int batch_size,
                          const std::vector<float> &temperatures,
                          const std::vector<int> &top_ks,
                          const std::vector<float> &top_ps,
                          const std::vector<uint32_t> &seeds);
  void CollectSampleBatch(std::vector<int> *out_tokens);

  /**
   * Graph-safe batched greedy argmax: launches ONLY the BatchedArgmaxKernel
   * writing the device result buffer. No memcpy, no event, no host state
   * changes — used by the decode-burst pipeline, which copies results into
   * its own ring slots. Returns false on launch failure.
   */
  bool EnqueueBatchedArgmax(const float *d_logits, int batch_size);

  /// Device pointer to the batched argmax results ([kMaxBatchSize] ints).
  /// Valid after Initialize(); written by EnqueueBatchedArgmax.
  const int *batch_result_device() const { return d_result_batch_; }

  std::size_t DeviceWorkspaceBytes() const;
  std::size_t HostWorkspaceBytes() const;

private:
  int GreedyArgmax(const float *d_logits);
  int StochasticSample(const float *d_logits, float temperature, int top_k,
                       float top_p);

  /**
   * Batched greedy argmax: one kernel launch + one sync for B sequences.
   * Reduces B cudaStreamSynchronize calls to 1.
   */
  void GreedyArgmaxBatch(const float *d_logits, int batch_size,
                         std::vector<int> *out_tokens);

  cudaStream_t stream_{nullptr};
  int vocab_size_{0};

  // Scratch buffers
  float *d_probs_{nullptr};       // [vocab_size] probabilities
  int *d_indices_{nullptr};       // [vocab_size] sorted indices
  float *d_temp_{nullptr};        // [vocab_size] temp storage
  int *d_result_{nullptr};        // [1] sampled token ID (device)
  int *h_result_pinned_{nullptr}; // [1] sampled token ID (host pinned)

  // For argmax
  float *d_max_val_{nullptr}; // [1] max value
  int *d_max_idx_{nullptr};   // [1] max index

  // Batch buffers (allocated once, sized for max batch)
  static constexpr int kMaxBatchSize = 64;
  int *d_result_batch_{nullptr};        // [kMaxBatchSize] on device
  int *h_result_batch_pinned_{nullptr}; // [kMaxBatchSize] host pinned
  float *h_logits_pinned_{nullptr};     // [vocab_size] host pinned

  // cuRAND
  curandGenerator_t rng_{nullptr};
  float *d_uniform_{nullptr}; // [1] random uniform
  bool rng_initialized_{false};
  cudaEvent_t completion_event_{nullptr};
  bool completion_pending_{false};
  int pending_batch_size_{0};
};

} // namespace inferflux
