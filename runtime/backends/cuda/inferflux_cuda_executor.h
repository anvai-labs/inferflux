#pragma once

#include "runtime/backends/cuda/decode_relay_fingerprint.h"
#include "runtime/backends/cuda/inferflux_cuda_runtime.h"
#include "runtime/backends/cuda/native/decode_burst.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/model_memory_ledger.h"
#include "runtime/backends/cuda/native/native_bootstrap_config.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "runtime/backends/cuda/native/strategy_registry.h"
#include "runtime/execution/unified_batch_lane_dispatcher.h"

#ifdef INFERFLUX_HAS_CUDA
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
#endif

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations for native kernel components
namespace inferflux {
class ModelForward;
class IKvCacheGpu;
template <typename T> class KvCacheGpuTyped;
using KvCacheGpu = KvCacheGpuTyped<half>;
class CublasGemm;
class GpuSampler;
template <typename T> class WeightMapTyped;
using WeightMap = WeightMapTyped<half>;
class ITokenizer;
class QuantizedWeightMap;
class QuantizedWeightMapAdapter;
} // namespace inferflux

namespace inferflux {

// Forward declaration
class SafetensorsParser;

/**
 * Safetensors model loader
 *
 * Loads HuggingFace safetensors format models directly.
 * No llama.cpp dependency.
 */
class SafetensorsLoader {
public:
  struct Tensor {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;    // "f16", "f32", "bf16", etc.
    size_t offset;        // File offset
    size_t size;          // Byte size
    void *cpu_data;       // Mapped CPU data
    void *gpu_data;       // GPU memory (after UploadToGPU)
    size_t gpu_offset{0}; // Offset within GPU buffer
  };

  struct ModelConfig {
    // Model architecture
    int hidden_size{0};
    int num_hidden_layers{0};
    int num_attention_heads{0};
    int num_key_value_heads{0}; // For GQA
    int head_dim{0};
    int intermediate_size{0};
    int vocab_size{0};
    int max_position_embeddings{0};

    // RoPE settings
    float rope_freq_base{10000.0f};
    float rope_freq_scale{1.0f};
    int rope_dim{0};

    // Model type
    std::string model_type; // "qwen2", "llama", etc.
    std::string activation; // "silu", "swiglu", etc.

    // Dtype and normalization
    std::string torch_dtype; // "bfloat16", "float16", etc.
    float rms_norm_eps{1e-6f};
  };

  SafetensorsLoader();
  ~SafetensorsLoader(); // Automatically frees CPU and GPU memory

  /**
   * Load model from safetensors directory
   * @param model_path Path containing model.safetensors.index.json
   */
  bool LoadModel(const std::string &model_path);

  /**
   * Get tensor by name
   */
  const Tensor *GetTensor(const std::string &name) const;

  /**
   * Get all tensor names
   */
  std::vector<std::string> GetTensorNames() const;

  /**
   * Get model configuration
   */
  const ModelConfig &GetConfig() const { return config_; }

  /**
   * Upload all tensors to GPU memory.
   * @param skip_bf16_conversion If true, upload BF16 weights as-is
   *        (for native BF16 inference pipeline).
   */
  bool UploadToGPU(cudaStream_t stream, bool skip_bf16_conversion = false);

  /**
   * Free CPU memory (after GPU upload)
   */
  void FreeCPUMemory();

  /**
   * Free GPU memory
   */
  void FreeGPUMemory();

  /**
   * Get model directory path
   */
  const std::string &GetModelPath() const { return model_path_; }

  /**
   * Get GPU weights buffer
   */
  void *GetGPUBuffer() const { return d_weights_buffer_; }

  /**
   * Get total GPU size
   */
  size_t GetGPUSize() const { return total_gpu_size_; }

private:
  bool LoadIndex(const std::string &index_path);
  bool LoadShard(const std::string &shard_path);
  bool ParseConfig(const std::string &config_path);

  std::unordered_map<std::string, Tensor> tensors_;
  ModelConfig config_;
  std::string model_path_;
  size_t total_size_bytes_{0};

  // Keep shard parsers alive to maintain valid CPU data mappings
  std::vector<std::unique_ptr<class SafetensorsParser>> shard_parsers_;

  // GPU memory
  void *d_weights_buffer_{nullptr}; // Contiguous GPU buffer for all weights
  size_t total_gpu_size_{0};        // Total size of GPU buffer
};

/**
 * InferFlux CUDA Executor
 *
 * Implements InferFlux CUDA inference using:
 * - Safetensors format (direct loading)
 * - Custom FlashAttention/PagedAttention kernels
 * - CUTLASS/cuBLAS for GEMM
 * - NO llama.cpp dependency
 */
class InferfluxCudaExecutor final : public InferfluxCudaRuntime {
#ifdef INFERFLUX_TESTING
  friend class ExecutorTestAccess;
#endif
public:
  InferfluxCudaExecutor();
  ~InferfluxCudaExecutor() override;

  // InferfluxCudaRuntime interface
  std::string Name() const override { return "inferflux_cuda"; }
  bool IsFallback() const override { return fallback_mode_; }
  const std::string &FallbackReason() const override {
    return fallback_reason_;
  }

  /**
   * Load model from safetensors
   */
  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override;

  /**
   * Execute batch (to be implemented with InferFlux CUDA kernels)
   */
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  bool NativeSupportsUnifiedBatchBurst() const override;
  UnifiedBurstResult
  NativeExecuteUnifiedBatchBurst(const std::vector<UnifiedBatchInput> &inputs,
                                 const UnifiedBurstOptions &options,
                                 const BurstTokenSink &sink) override;

  bool SupportsAsyncUnifiedBatch() const override;

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override;

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;

  std::shared_ptr<BackendInterface> BackendHandle() const override {
    return nullptr; // No llama backend!
  }

  // Native perf
  PerfSnapshot NativeTakePerf() override;

  // Timing helper: 0 disables event timing, N>0 records every Nth native work
  // item (decode batch or prefill request). Exposed for contract tests.
  static bool ShouldRecordTimingSample(int sample_rate,
                                       std::atomic<int> *counter);
  // Non-atomic overload for tests and single-threaded callers.
  static bool ShouldRecordTimingSample(int sample_rate, int *counter);

  // Native* overrides — provide tokenization/readiness without llama.cpp
  std::vector<int> NativeTokenize(const std::string &prompt) const override;
  int NativeTokenCount(const std::string &text) const override;
  bool NativeIsReady() const override;
  void NativeFreeSequence(int sequence_id) override;
  LlamaCppBackend::SequenceReleaseFence
  NativeBeginFreeSequence(int sequence_id) override;
  bool NativePollFreeSequence(
      const LlamaCppBackend::SequenceReleaseFence &fence) override;
  void NativeCopySequencePrefix(int src_seq, int dst_seq,
                                int n_tokens) override;
  std::vector<uint8_t> NativeSerializeSequence(int sequence_id) const override;
  bool NativeHydrateSequence(int dest_sequence_id,
                             const std::vector<uint8_t> &blob) override;
  ChatResult NativeFormatChat(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const override;
  const ITokenizer *NativeGetTokenizer() const override;
  int CopyLastLogitsToHost(float *host_buf, int buf_size) override;
  int NativeVocabSize() const override;
  std::vector<float> NativeEmbed(const std::string &text) override;
  int NativeEmbedDims() const override;
  AttentionTensorData CaptureAttentionTensors() override;
  const runtime::cuda::native::ModelMemoryLedger &MemoryLedger() const {
    return memory_ledger_;
  }

  // Native-specific functionality
  bool HasFlashAttention2() const { return has_flash_attention_2_; }
  int GetDeviceId() const { return device_id_; }

  enum class InferenceDtype { kFP16, kBF16 };
  InferenceDtype GetInferenceDtype() const { return inference_dtype_; }

private:
  // Model loading
  std::unique_ptr<SafetensorsLoader> loader_;
  std::unique_ptr<runtime::cuda::native::IModelLoader> model_loader_;
  runtime::cuda::native::ModelInfo model_info_;
  std::filesystem::path loaded_model_path_;
  bool model_loaded_{false};
  bool fallback_mode_{false};
  std::string fallback_reason_;

  // CUDA state
  int device_id_{0};
  cudaStream_t compute_stream_{nullptr};
  cudaStream_t copy_stream_{nullptr};

  // Model parameters (from safetensors)
  SafetensorsLoader::ModelConfig model_config_;

  // GPU memory pointers
  void *d_weights_{nullptr}; // All weights on GPU

  // Kernel capabilities
  bool has_flash_attention_2_{true};
  bool use_flash_attention_{true};
  InferenceDtype inference_dtype_{InferenceDtype::kFP16};
  runtime::cuda::native::KvPrecision kv_precision_{
      runtime::cuda::native::KvPrecision::kFp16};
  NativeBootstrapConfig bootstrap_config_{};
  runtime::cuda::native::DequantizedCachePolicy dequantized_cache_policy_{
      runtime::cuda::native::DequantizedCachePolicy::kNone};
  std::string dequantized_cache_policy_hint_{"none"};
  bool require_fused_quantized_matmul_{false};
  runtime::cuda::native::MatmulExecutionMode quantized_matmul_mode_{
      runtime::cuda::native::MatmulExecutionMode::kFusedDequantTileGemm};
  std::string quantized_matmul_strategy_id_{
      "matmul.fused.dequant_tile_gemm.v1"};
  runtime::cuda::native::ModelMemoryLedger memory_ledger_;
  int active_max_batch_{0};
  int active_max_seq_{0};

  // Native kernel pipeline components (only available with CUDA)
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  std::unique_ptr<ModelForward> model_forward_;
  std::unique_ptr<IKvCacheGpu> kv_cache_; // Typed KV cache (FP16 or BF16)
  std::unique_ptr<CublasGemm> gemm_;
  std::unique_ptr<GpuSampler> sampler_;
  std::unique_ptr<WeightMap> weight_map_;
  std::unique_ptr<QuantizedWeightMap> quantized_weight_map_;
  std::unique_ptr<QuantizedWeightMapAdapter> quantized_weight_adapter_;
  runtime::cuda::native::DecodeBurstController burst_controller_;
#endif
  std::unique_ptr<ITokenizer> tokenizer_;
  // GGUF chat template renderer (separate from tokenizer_). Used only for
  // NativeFormatChat() to render ChatML/Llama/Mistral/Gemma templates.
  std::unique_ptr<ITokenizer> chat_template_tokenizer_;

  // Per-sequence recent token history for repetition penalty.
  // Maps sequence_id → circular buffer of last N generated tokens.
  static constexpr int kPenaltyWindowSize = 64;
  // Mutated from the decode path (under shared_pipeline_mutex_) AND from the
  // sequence-release paths, which run on scheduler/eviction threads and hold
  // no pipeline lock. History MUST be erased wherever the KV slot is cleared:
  // it is keyed by sequence_id (a KV slot), so a stale entry makes the next
  // request on that slot sample under the previous request's tokens — with
  // the implicit greedy 1.15 penalty this visibly corrupts the very first
  // generated token.
  void ClearSequenceRecentTokens(int sequence_id);

  mutable std::mutex sequence_recent_tokens_mutex_;
  // Per-slot repetition-penalty history. The generation tag invalidates the
  // history when a slot is recycled for a new request: the scheduler bumps
  // sequence_generation on reuse, and keying by slot id alone would bias the
  // new request's first tokens with the previous request's tokens (probe op
  // X reproduces this deterministically).
  struct SequenceRecentTokens {
    uint64_t generation{0};
    std::vector<int> tokens;
  };
  std::unordered_map<int, SequenceRecentTokens> sequence_recent_tokens_;

  // Device logits buffer
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  float *d_logits_{nullptr};
#endif

  // Async lane support
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  cudaStream_t decode_stream_{nullptr};
  cudaStream_t prefill_stream_{nullptr};

  struct LaneExecutionResources {
    ModelForward *forward{nullptr};
    GpuSampler *sampler{nullptr};
    CublasGemm *gemm{nullptr};
    QuantizedWeightMap *quantized_weights{nullptr};
    float *logits{nullptr};
    cudaStream_t stream{nullptr};
  };

  struct LaneExecutionResult {
    std::vector<UnifiedBatchOutput> outputs;
    double elapsed_ms{0.0};
  };

  bool lane_overlap_ready_{false};
  bool lane_overlap_init_attempted_{false};
  std::mutex lane_overlap_mutex_;
  std::mutex shared_pipeline_mutex_;
  uint64_t batch_counter_{0}; // Throttle for periodic operations
  std::unique_ptr<ModelForward> decode_lane_forward_;
  std::unique_ptr<ModelForward> prefill_lane_forward_;
  std::unique_ptr<GpuSampler> decode_lane_sampler_;
  std::unique_ptr<GpuSampler> prefill_lane_sampler_;
  std::unique_ptr<CublasGemm> decode_lane_gemm_;
  std::unique_ptr<CublasGemm> prefill_lane_gemm_;
  // GGUF overlap lanes need independent quantized maps/adapters because
  // QuantizedWeightMap uses mutable dequant scratch/cache state.
  std::unique_ptr<QuantizedWeightMap> decode_lane_quantized_weight_map_;
  std::unique_ptr<QuantizedWeightMap> prefill_lane_quantized_weight_map_;
  std::unique_ptr<QuantizedWeightMapAdapter>
      decode_lane_quantized_weight_adapter_;
  std::unique_ptr<QuantizedWeightMapAdapter>
      prefill_lane_quantized_weight_adapter_;
  float *d_decode_logits_{nullptr};
  float *d_prefill_logits_{nullptr};
  UnifiedBatchLaneDispatcher lane_dispatcher_;
  std::mutex lane_dispatcher_mutex_;

  struct PendingSequenceRelease {
    uint64_t token{0};
    int sequence_id{-1};
    cudaEvent_t compute_done{nullptr};
    cudaEvent_t decode_done{nullptr};
    cudaEvent_t prefill_done{nullptr};
  };
  std::vector<PendingSequenceRelease> pending_sequence_releases_;
  uint64_t next_sequence_release_token_{1};
  std::mutex pending_sequence_releases_mutex_;
#endif

  // Performance timing via cudaEvents
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  cudaEvent_t forward_start_{nullptr};
  cudaEvent_t forward_stop_{nullptr};
  cudaEvent_t sampling_start_{nullptr};
  cudaEvent_t sampling_stop_{nullptr};

  // Timing sample rate: 0 = disabled, N>0 = record every Nth batch.
  // Loaded through NativeExecutionPolicy during model initialization.
  int timing_sample_rate_{0};
  std::atomic<int> timing_batch_counter_{0};
  NativeExecutionPolicy execution_policy_{};

  struct NativePerfAccumulator {
    std::atomic<double> prefill_ms{0.0};
    std::atomic<double> decode_ms{0.0};
    std::atomic<int32_t> prompt_tokens{0};
    std::atomic<int32_t> generated_tokens{0};
  };
  NativePerfAccumulator perf_accum_;

  // Device-side token relay: avoids H2D metadata upload between decode tokens.
  // When active, DeviceTokenRelay updates graph input buffers on device,
  // and BatchForwardReplay replays the graph without host round-trip.
  // Atomic to prevent TOCTOU races with concurrent scheduler threads.
  std::atomic<bool> decode_relay_active_{false};
  std::atomic<int> decode_relay_batch_size_{0};
  // Identity of the batch the relay is armed for (see
  // decode_relay_fingerprint.h). Batch size alone is not a valid guard: a
  // different batch of the same size would inherit the previous batch's device
  // metadata and decode on the departed sequences' KV slots. Guarded by
  // shared_pipeline_mutex_ (the standard decode path holds it across
  // guard-check and re-arm).
  cuda::DecodeRelayFingerprint decode_relay_armed_;
  int *d_eos_flag_{nullptr}; // [1] device flag for EOS detection

  /**
   * Multi-token burst decode: runs N greedy forward+sample steps entirely
   * on GPU using graph replay + DeviceTokenRelay, syncing to host only once
   * at the end. Eliminates N-1 WDDM round-trips.
   *
   * Returns actual tokens generated (may be < n_tokens if EOS hit).
   * Tokens written to out_tokens[0..return_value-1].
   */
  int BurstDecodeGreedy(int sequence_id, int n_past_start, int first_token_id,
                        int n_tokens, int eos_token_id,
                        std::vector<int> *out_tokens);

  // Phase overlap configuration
  bool overlap_enabled_{true};
  int min_prefill_tokens_{256};
#endif

  // Internal helpers
  bool InitializeCUDA();
  bool InitializeNativePipeline();
  void RefreshMemoryLedger();
  void MaybeRefreshMemoryLedger();
  void FreeDeviceMemory();
  bool RunNativeInference(const std::vector<UnifiedBatchInput> &inputs,
                          std::vector<UnifiedBatchOutput> *outputs);
  bool ConfigureDequantizedCachePolicy(const std::string &raw_policy);
  void ReleaseBatchScopedDequantizedCache();
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs,
                      bool allow_overlap);
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  bool
  InitializeLaneOverlapResources(const SafetensorsLoader::ModelConfig &config,
                                 bool want_bf16, int max_batch);
  bool EnsureLaneOverlapResources();
  void DestroyLaneOverlapResources();
  void DestroyLaneOverlapResourcesUnlocked();
  bool CanRunLaneOverlap() const;
  LaneExecutionResources PrimaryLaneResources();
  LaneExecutionResources GetLaneResources(bool decode_lane);
  LaneExecutionResult
  ExecuteLaneBatch(const std::vector<UnifiedBatchInput> &inputs,
                   const LaneExecutionResources &resources);
  LaneExecutionResult
  ExecuteLaneBatchForAsync(const std::vector<UnifiedBatchInput> &inputs,
                           bool decode_lane);
  bool EnsureLaneDispatcherStarted();
  void StopLaneDispatcher();
#endif

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Phase overlap helpers
  bool HasMixedWorkload(const std::vector<UnifiedBatchInput> &inputs) const;
  void SplitBatchByType(const std::vector<UnifiedBatchInput> &inputs,
                        std::vector<size_t> &prefill_indices,
                        std::vector<size_t> &decode_indices) const;
  bool IsPrefillLikeInput(const UnifiedBatchInput &input) const;
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatchWithOverlap(const std::vector<UnifiedBatchInput> &inputs);
#endif
};

} // namespace inferflux
