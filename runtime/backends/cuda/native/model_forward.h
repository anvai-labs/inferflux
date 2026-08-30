#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/cuda/inferflux_cuda_executor.h"
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "runtime/backends/cuda/native/weight_map.h"

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
#include <memory>
#include <string>
#include <vector>

namespace inferflux {

/**
 * Abstract interface for model-type-specific forward passes.
 *
 * Implementations handle the full transformer forward pass from
 * token IDs through logits computation.
 */
class ModelForward {
public:
  virtual ~ModelForward() = default;

  /**
   * Initialize scratch buffers and bind to model weights.
   */
  virtual bool Initialize(const SafetensorsLoader::ModelConfig &config,
                          const WeightMap &weights, IKvCacheGpu *kv_cache,
                          CublasGemm *gemm, cudaStream_t stream) = 0;

  /**
   * Run forward pass for a single sequence.
   *
   * @param token_ids   Token IDs to process
   * @param n_past      Number of past KV cache entries
   * @param sequence_id Sequence slot in KV cache
   * @param d_logits    Output: [vocab_size] FP32 logits on device
   * @return true on success
   */
  virtual bool Forward(const std::vector<int> &token_ids, int n_past,
                       int sequence_id, float *d_logits) = 0;

  /**
   * Run batched forward pass for multiple decode sequences (1 token each).
   *
   * Batches compute-dominant GEMMs while keeping attention per-sequence.
   * Default implementation falls back to sequential Forward() calls.
   *
   * @param token_ids    One token per sequence [batch_size]
   * @param n_past       KV cache positions per sequence [batch_size]
   * @param sequence_ids KV cache slot per sequence [batch_size]
   * @param d_logits     Output: [batch_size * vocab_size] FP32 logits
   * @param batch_size   Number of sequences
   * @return true on success
   */
  /**
   * Replay the CUDA graph without uploading batch metadata from host.
   *
   * Used after DeviceTokenRelay has updated the graph's input buffers
   * directly on device. Skips the H2D metadata copy, eliminating the
   * per-token WDDM scheduling round-trip (~10ms on Windows).
   *
   * @param d_logits     Output: [batch_size * vocab_size] FP32 logits
   * @param batch_size   Number of sequences (must match captured graph)
   * @return true if graph replayed; false if graph not available
   */
  virtual bool BatchForwardReplay(float *d_logits, int batch_size) {
    return false; // Default: not supported (requires captured graph)
  }

  /**
   * Get the device-side batch metadata pointer for DeviceTokenRelay.
   * Returns nullptr if device-side relay is not supported.
   */
  virtual int *GetBatchMetaDevice() { return nullptr; }

  /**
   * Get the maximum batch size for the pre-allocated metadata buffers.
   */
  virtual int GetMaxBatchSize() const { return 0; }

  virtual bool BatchForward(const std::vector<int> &token_ids,
                            const std::vector<int> &n_past,
                            const std::vector<int> &sequence_ids,
                            float *d_logits, int batch_size) {
    // Default: sequential Forward() calls (backward compat)
    for (int i = 0; i < batch_size; ++i) {
      std::vector<int> single_token = {token_ids[i]};
      if (!Forward(single_token, n_past[i], sequence_ids[i],
                   d_logits + i * VocabSize())) {
        return false;
      }
    }
    return true;
  }

  /// Device-resident batch decode metadata used by the burst token feed.
  struct BatchMetaDevice {
    int *token_ids{nullptr};
    int *n_past{nullptr};
    int *seq_ids{nullptr};
    int *kv_lens{nullptr};
  };

  /// Access the device batch metadata buffers (fixed addresses). Only
  /// meaningful for forwarders that support device-fed decode.
  virtual BatchMetaDevice BatchMetaDevicePointers() const { return {}; }

  /// True when a replayable decode graph exists for this batch size — i.e.
  /// subsequent steps can be enqueued as a single graph launch.
  virtual bool DecodeGraphReady(int /*batch_size*/) const { return false; }

  /**
   * Batched forward pass WITHOUT the host metadata upload: the device-side
   * token/n_past/kv_len buffers are assumed current (advanced on device by
   * the burst token-feed kernel). Used for burst-pipelined decode steps
   * after the first. Default: unsupported.
   */
  virtual bool BatchForwardDevice(int /*batch_size*/, float * /*d_logits*/) {
    return false;
  }

  /**
   * Pre-warm lazily-initialized weight caches (F32→FP16 norm weights,
   * attention biases, embeddings).  Must be called once after Initialize()
   * so that subsequent CUDA graph capture does not encounter illegal
   * cudaStreamSynchronize calls inside the capture region.
   */
  virtual void WarmWeightCaches() {}

  /**
   * Load-time dispatch reachability probe. Runs selectable FFN and down-proj
   * operators on layer-0 weights through the production executor stages and
   * marks divergent operators unhealthy (self-heal: the dispatch rules
   * then skip them). Returns "" when not implemented. Must be called only
   * at load time, after WarmWeightCaches().
   */
  virtual std::string ProbeDispatchPaths() { return ""; }

  /**
   * Set the CUDA stream for forward passes.
   * Subclasses should propagate to cuBLAS handle and sampler.
   */
  virtual void SetStream(cudaStream_t /*stream*/) {}

  virtual void SetExecutionPolicy(const NativeExecutionPolicy & /*policy*/) {}

  /**
   * Return the vocab size for offset calculations in batched forward.
   */
  virtual int VocabSize() const = 0;

  /**
   * Run a forward pass for embedding extraction (mean-pooled hidden states).
   *
   * Runs all transformer layers, applies final RmsNorm, then mean-pools
   * across token positions. Returns FP32 embeddings on device.
   *
   * @param token_ids    Input token IDs
   * @param sequence_id  KV cache sequence slot
   * @param d_output     Output: [hidden_size] FP32 embeddings on device
   * @return true on success
   */
  virtual bool EmbedForward(const std::vector<int> &token_ids, int sequence_id,
                            float *d_output) {
    (void)token_ids;
    (void)sequence_id;
    (void)d_output;
    return false; // Default: not supported
  }

  /**
   * Return the hidden size for embedding dimension calculations.
   */
  virtual int HiddenSize() const = 0;

  /**
   * Free scratch buffers.
   */
  virtual void FreeScratchBuffers() = 0;

  /**
   * Startup-accounted device workspace owned by this forwarder.
   */
  virtual std::size_t DeviceWorkspaceBytes() const { return 0; }

  /**
   * Startup-accounted pinned host workspace owned by this forwarder.
   */
  virtual std::size_t HostWorkspaceBytes() const { return 0; }

  /**
   * Model type name for logging.
   */
  virtual std::string ModelType() const = 0;

  /**
   * Capture intermediate attention tensors for debugging/profiling.
   * Default implementation returns empty data with "not implemented" message.
   */
  virtual AttentionTensorData CaptureAttentionTensors() {
    return {{}, false, "Not implemented for this model type"};
  }
};

} // namespace inferflux
