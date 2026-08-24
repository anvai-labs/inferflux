#pragma once

#include "backend_config.h"
#include "backend_types.h"
#include "runtime/backends/backend_capabilities.h"
#include "runtime/logprob.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace inferflux {

struct SamplingParams;

/// Sequence release fence for async KV cache cleanup.
struct SequenceReleaseFence {
  uint64_t token{0};
  bool pending{false};
};

/// Performance snapshot from a backend.
struct PerfSnapshot {
  double prefill_ms{0};
  double decode_ms{0};
  int32_t prompt_tokens{0};
  int32_t generated_tokens{0};
};

/// Result of applying a chat template.
struct ChatTemplateResult {
  bool valid{false};
  std::string prompt;
};

/// Abstract interface for all inference backends.
///
/// This interface defines the full contract that the scheduler, router, and
/// HTTP server use to interact with backends. All methods have default
/// implementations so that a standalone backend only needs to implement what
/// it supports.
class BackendInterface {
public:
  virtual ~BackendInterface() = default;

  // ========================================================================
  // Model Loading
  // ========================================================================

  virtual bool LoadModel(const std::filesystem::path &model_path,
                         const LlamaBackendConfig &config) = 0;

  // ========================================================================
  // Unified Batch Execution (Core API)
  // ========================================================================

  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) = 0;

  virtual int UnifiedBatchTokenCapacity() const {
    return 2048; // Default safe limit
  }

  // ========================================================================
  // Async Execution
  // ========================================================================

  virtual bool SupportsAsyncUnifiedBatch() const { return false; }

  virtual UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane = UnifiedBatchLane::kAuto) {
    (void)inputs;
    (void)lane;
    return 0;
  }

  virtual bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs) {
    (void)handle;
    (void)outputs;
    return false;
  }

  // ========================================================================
  // Phased Inference
  // ========================================================================

  virtual bool SupportsSplitPrefillDecodeHandoff() const { return false; }
  virtual bool SupportsProcessLocalSequenceTransfer() const { return false; }

  virtual PrefillResult Prefill(const std::string &prompt, int sequence_id) {
    (void)prompt;
    (void)sequence_id;
    return {};
  }

  virtual PrefillResult PrefillPartial(const std::string &prompt,
                                       int sequence_id, int n_past_start) {
    (void)prompt;
    (void)sequence_id;
    (void)n_past_start;
    return {};
  }

  virtual std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1, const std::vector<std::string> &stop_seqs = {}) {
    (void)n_past;
    (void)sequence_id;
    (void)max_tokens;
    (void)on_chunk;
    (void)should_stop;
    (void)logprob_top_n;
    (void)out_logprobs;
    (void)first_token;
    (void)stop_seqs;
    return {};
  }

  virtual std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {}) {
    (void)prompt;
    (void)max_tokens;
    (void)on_chunk;
    (void)should_stop;
    (void)logprob_top_n;
    (void)out_logprobs;
    (void)stop_seqs;
    return {};
  }

  // ========================================================================
  // Sequence Lifecycle
  // ========================================================================

  virtual void FreeSequence(int sequence_id) { (void)sequence_id; }

  virtual SequenceReleaseFence BeginFreeSequence(int sequence_id) {
    FreeSequence(sequence_id);
    return {};
  }

  virtual bool PollFreeSequence(const SequenceReleaseFence &fence) {
    (void)fence;
    return true;
  }

  virtual void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) {
    (void)src_seq;
    (void)dst_seq;
    (void)n_tokens;
  }

  virtual std::vector<uint8_t> SerializeSequence(int sequence_id) {
    (void)sequence_id;
    return {};
  }

  virtual bool HydrateSequence(int dest_sequence_id,
                               const std::vector<uint8_t> &blob) {
    (void)dest_sequence_id;
    (void)blob;
    return false;
  }

  // ========================================================================
  // Tokenization & Sampling
  // ========================================================================

  virtual int TokenCount(const std::string &text) const {
    (void)text;
    return 0;
  }

  virtual std::vector<int> TokenizeForCache(const std::string &prompt) const {
    (void)prompt;
    return {};
  }

  virtual void SetupSampler(const std::string &grammar, const std::string &root,
                            const SamplingParams &sp) {
    (void)grammar;
    (void)root;
    (void)sp;
  }

  virtual void TeardownSampler() {}

  virtual ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) {
    (void)messages;
    (void)add_assistant_prefix;
    return {};
  }

  // ========================================================================
  // Embeddings
  // ========================================================================

  virtual std::vector<float> Embed(const std::string &text) {
    (void)text;
    return {};
  }

  virtual int EmbedDims() const { return 0; }

  // ========================================================================
  // Capabilities & Diagnostics
  // ========================================================================

  virtual std::string Name() const = 0;

  virtual bool IsFallback() const { return false; }

  virtual const std::string &FallbackReason() const {
    static const std::string empty_reason;
    return empty_reason;
  }

  virtual bool IsReady() const { return false; }

  virtual bool SupportsVision() const { return false; }

  virtual bool IsMoE() const { return false; }
  virtual int ExpertCount() const { return 0; }
  virtual int ActiveExperts() const { return 0; }

  virtual int ContextSize() const { return 0; }
  virtual bool FlashAttentionEnabled() const { return false; }

  virtual PerfSnapshot TakePerf() { return {}; }

  virtual std::vector<TopLogitEntry> TopLogitsForParity(int top_n) {
    (void)top_n;
    return {};
  }

  /// Report the full capability set of this backend.
  /// Default returns all-true (matches LlamaCppBackend defaults).
  virtual BackendCapabilities ReportCapabilities() const {
    return BackendCapabilities{};
  }
};

} // namespace inferflux
