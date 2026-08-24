#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"
#include "runtime/backends/native/native_inference_runtime.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace inferflux {

// NativeGpuBackend is the shared base for all InferFlux-native GPU backends.
// It owns a NativeInferenceRuntime, handles parity delegation for structured
// output, and implements the decode/generate token loops, logprob collection,
// embedding passthrough, and sampling state management.
//
// Subclasses provide:
//   - Constructor with device-specific GpuDeviceStrategy
//   - CreateNativeRuntime() factory for the hardware-specific runtime
//   - Name(), NativeKernelsReady() as needed
class NativeGpuBackend : public GpuAcceleratedBackend {
public:
  explicit NativeGpuBackend(std::unique_ptr<GpuDeviceStrategy> strategy);
  ~NativeGpuBackend() override;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

  // Subclass must create the device-specific runtime after device init.
  virtual std::unique_ptr<NativeInferenceRuntime> CreateNativeRuntime() = 0;

  // --- LlamaCppBackend overrides (delegating to runtime) ---
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;
  bool SupportsUnifiedBatchBurst() const override;
  UnifiedBurstResult
  ExecuteUnifiedBatchBurst(const std::vector<UnifiedBatchInput> &inputs,
                           const UnifiedBurstOptions &options,
                           const BurstTokenSink &sink) override;
  bool SupportsAsyncUnifiedBatch() const override;
  bool SupportsSplitPrefillDecodeHandoff() const override;
  bool SupportsProcessLocalSequenceTransfer() const override;
  UnifiedBatchHandle SubmitUnifiedBatchAsync(
      const std::vector<UnifiedBatchInput> &inputs,
      UnifiedBatchLane lane = UnifiedBatchLane::kAuto) override;
  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;
  int UnifiedBatchTokenCapacity() const override;
  PrefillResult Prefill(const std::string &prompt, int sequence_id) override;
  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override;
  void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) override;
  void FreeSequence(int sequence_id) override;
  SequenceReleaseFence BeginFreeSequence(int sequence_id) override;
  bool PollFreeSequence(const SequenceReleaseFence &fence) override;
  std::vector<uint8_t> SerializeSequence(int sequence_id) override;
  bool HydrateSequence(int dest_sequence_id,
                       const std::vector<uint8_t> &blob) override;
  std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1,
         const std::vector<std::string> &stop_seqs = {}) override;
  std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {}) override;
  void SetupSampler(const std::string &grammar, const std::string &root,
                    const SamplingParams &sp) override;
  void TeardownSampler() override;
  PerfSnapshot TakePerf() override;
  ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) override;
  std::vector<TopLogitEntry> TopLogitsForParity(int top_n) override;
  int TokenCount(const std::string &text) const override;
  std::vector<int> TokenizeForCache(const std::string &prompt) const override;
  bool IsReady() const override;

  // --- Embedding (overrides BackendInterface via LlamaCppBackend) ---
  // Embed uses native embeddings when available, falls back to parity backend.
  std::vector<float> Embed(const std::string &text) override;
  int EmbedDims() const override;
  // Legacy entry points — kept for backward compat, delegate to Embed/EmbedDims
  std::vector<float> EmbedForParity(const std::string &text);
  int EmbedDimsForParity() const;

  // --- Native capability contracts ---
  virtual bool SupportsLogprobsContract() const;
  virtual bool SupportsStructuredOutputContract() const;
  virtual bool SupportsEmbeddingsContract() const;
  virtual bool SupportsSpeculativeDecodingContract() const;

  // --- Capability reporting ---
  BackendCapabilities ReportCapabilities() const override;
  bool IsFallback() const override { return fallback_mode_; }
  const std::string &FallbackReason() const override {
    return fallback_reason_;
  }

  // --- Runtime introspection ---
  std::string ExecutorKind() const { return runtime_kind_; }
  bool IsFallbackExecutor() const { return fallback_mode_; }

protected:
  // Log tag for subclass-specific messages.
  virtual const char *LogTag() const { return "native_gpu_backend"; }

  // Access the runtime (under lock in the caller).
  NativeInferenceRuntime *Runtime() const { return runtime_.get(); }

  // Parity delegate target (e.g. kCuda) — used for TuneLlamaBackendConfig.
  virtual LlamaBackendTarget ParityTarget() const {
    return strategy_ ? strategy_->Target() : LlamaBackendTarget::kCpu;
  }

private:
  bool IsParityDelegateAvailable() const;
  std::shared_ptr<LlamaCppBackend> EnsureParityBackend() const;
  std::shared_ptr<BackendInterface> DelegateBackend() const;
  bool UsesStructuredConstraintSampler() const;
  SamplingParams SnapshotSamplingParams() const;
  TokenLogprob CollectNativeLogprob(int token_id, const std::string &piece,
                                    int top_n);

  std::unique_ptr<NativeInferenceRuntime> runtime_;
  std::filesystem::path loaded_model_path_;
  LlamaBackendConfig loaded_config_{};
  std::filesystem::path parity_load_path_;
  bool parity_delegate_enabled_{true};
  mutable bool parity_delegate_available_{false};
  mutable bool parity_delegate_init_attempted_{false};
  mutable std::shared_ptr<LlamaCppBackend> parity_backend_;
  std::string runtime_kind_{"native"};
  bool fallback_mode_{true};
  std::string fallback_reason_;
  mutable std::recursive_mutex runtime_mutex_;
  mutable std::mutex parity_backend_mutex_;
  mutable std::mutex sampling_mutex_;
  SamplingParams active_sampling_{};
  bool sampling_active_{false};
  bool structured_constraint_sampler_active_{false};
  std::vector<float> host_logits_buf_;
};

} // namespace inferflux
