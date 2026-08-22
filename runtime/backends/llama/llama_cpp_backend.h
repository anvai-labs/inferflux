#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/ep_dispatch.h"
#include "runtime/backends/gpu/backend_config_extensions.h"
#include "runtime/logprob.h"
#include "runtime/multimodal/image_preprocessor.h"
#include "scheduler/request_batch.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <llama.h>

#ifdef INFERFLUX_HAS_MTMD
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/mtmd.h"
#endif

namespace inferflux {

class LlamaCppBackend : public BackendInterface {
public:
  LlamaCppBackend();
  virtual ~LlamaCppBackend();

  // Type aliases for backward compatibility with code that references
  // LlamaCppBackend::UnifiedBatchInput, etc.
  using UnifiedBatchInput = ::inferflux::UnifiedBatchInput;
  using UnifiedBatchOutput = ::inferflux::UnifiedBatchOutput;
  using UnifiedBatchLane = ::inferflux::UnifiedBatchLane;
  using UnifiedBatchHandle = ::inferflux::UnifiedBatchHandle;
  using PrefillResult = ::inferflux::PrefillResult;
  using SequenceReleaseFence = ::inferflux::SequenceReleaseFence;
  using PerfSnapshot = ::inferflux::PerfSnapshot;
  using ChatTemplateResult = ::inferflux::ChatTemplateResult;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

  // Load a multimodal projector (mmproj) GGUF alongside the text model.
  // Must be called after LoadModel(). Returns false if ENABLE_MTMD is off.
  bool LoadMmproj(const std::filesystem::path &mmproj_path);

  // True when a mmproj has been loaded and supports vision input.
  bool SupportsVision() const override { return vision_ready_; }

  // Input/output for one step of multi-sequence batch decode.
  struct BatchDecodeInput {
    int sequence_id;
    int n_past;
    int feed_token;
  };
  struct BatchDecodeOutput {
    int token{-1};
    std::string piece;
  };

  // Execute a mixed batch of prefill and decode sequences.
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  // Burst-pipelined decode: backends that keep sampled tokens on device may
  // run several decode steps per call and stream tokens through the sink
  // while the GPU runs ahead. Default: not supported (callers fall back to
  // per-token ExecuteUnifiedBatch).
  virtual bool SupportsUnifiedBatchBurst() const { return false; }
  virtual UnifiedBurstResult
  ExecuteUnifiedBatchBurst(const std::vector<UnifiedBatchInput> &inputs,
                           const UnifiedBurstOptions &options,
                           const BurstTokenSink &sink) {
    (void)inputs;
    (void)options;
    (void)sink;
    return UnifiedBurstResult{};
  }

  bool SupportsAsyncUnifiedBatch() const override;
  UnifiedBatchHandle SubmitUnifiedBatchAsync(
      const std::vector<UnifiedBatchInput> &inputs,
      UnifiedBatchLane lane = UnifiedBatchLane::kAuto) override;
  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;
  int UnifiedBatchTokenCapacity() const override;
  bool SupportsSplitPrefillDecodeHandoff() const override { return false; }
  bool SupportsProcessLocalSequenceTransfer() const override { return false; }
  PrefillResult Prefill(const std::string &prompt, int sequence_id) override;
  std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1,
         const std::vector<std::string> &stop_seqs = {}) override;
  void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) override;
  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override;
  void FreeSequence(int sequence_id) override;
  SequenceReleaseFence BeginFreeSequence(int sequence_id) override;
  bool PollFreeSequence(const SequenceReleaseFence &fence) override;
  std::vector<uint8_t> SerializeSequence(int sequence_id) override;
  bool HydrateSequence(int dest_sequence_id,
                       const std::vector<uint8_t> &blob) override;
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
  int TokenCount(const std::string &text) const override;
  std::vector<int> TokenizeForCache(const std::string &prompt) const override;
  std::vector<TopLogitEntry> TopLogitsForParity(int top_n) override;
  std::vector<float> Embed(const std::string &text) override;
  int EmbedDims() const override;

  // Execute one shared decode step for N sequences simultaneously.
  std::vector<BatchDecodeOutput>
  BatchDecodeStep(std::vector<BatchDecodeInput> &inputs);

  bool IsMoE() const override;
  int ExpertCount() const override;
  int ActiveExperts() const override;

  // Vision-aware generation.
  std::string GenerateWithImages(
      const std::string &prompt, const std::vector<DecodedImage> &images,
      int max_tokens,
      const std::function<bool(const std::string &, const TokenLogprob *)>
          &on_chunk = {},
      const std::function<bool()> &should_stop = {},
      const std::vector<std::string> &stop_seqs = {});

  // Backward-compat wrappers (grammar-only, default SamplingParams).
  void EnableGrammarConstraint(const std::string &grammar,
                               const std::string &root);
  void DisableGrammarConstraint();

  bool IsReady() const override { return context_ != nullptr || test_ready_; }
  int ContextSize() const override {
    return context_ ? static_cast<int>(llama_n_ctx(context_)) : 0;
  }
  bool FlashAttentionEnabled() const override {
    return config_.use_flash_attention;
  }

  PerfSnapshot TakePerf() override;

  ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) override;

  bool IsTerminalGeneratedToken(int token) const;
  void ForceReadyForTests() { test_ready_ = true; }

  // Rank within the Tensor Parallel group.
  int TPRank() const { return tp_rank_; }

  // BackendInterface implementation
  std::string Name() const override;
  bool IsFallback() const override { return false; }
  const std::string &FallbackReason() const override {
    static const std::string empty_reason;
    return empty_reason;
  }
  BackendCapabilities ReportCapabilities() const override {
    return BackendCapabilities{};
  }

protected:
  explicit LlamaCppBackend(bool acquire_backend);
  struct llama_sampler *active_sampler_{nullptr};
  std::shared_ptr<EPDispatch> ep_dispatch_;
  int tp_rank_{0};
  mutable std::recursive_mutex backend_state_mutex_;

private:
  void TeardownSamplerImpl();
  std::vector<int> Tokenize(const std::string &prompt, bool add_bos) const;
  std::string TokenToString(int token) const;

  TokenLogprob CollectLogprob(int token_id, const std::string &token_str,
                              int top_n) const;

  llama_model *model_{nullptr};
  llama_context *context_{nullptr};
  const struct llama_vocab *vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
  bool test_ready_{false};
  bool llama_backend_acquired_{false};
  PerfSnapshot last_perf_{};
  llama_context *embed_ctx_{nullptr};
  bool EnsureEmbedCtx();
  mutable std::mutex async_results_mutex_;
  UnifiedBatchHandle next_async_handle_{1};
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      async_results_;

#ifdef INFERFLUX_HAS_MTMD
  mtmd_context *mtmd_ctx_{nullptr};
#endif
  bool vision_ready_{false};
};

} // namespace inferflux
