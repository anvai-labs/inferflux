#pragma once
#include "runtime/backends/common/device_placement.h"

#include "backend_config.h"
#include "backend_types.h"
#include "model/chat_template_renderer.h"
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

/// Snapshot of intermediate attention tensors for debugging/profiling.
struct AttentionTensorSnapshot {
  int layer_idx{-1};
  std::string operation;   // "qkv_projection", "rope", "attention_scores", etc.
  std::vector<float> data; // Flattened tensor data (on host)
  std::vector<int> shape;  // Tensor shape [batch, seq, heads, dim]
};

/// Container for attention tensor snapshots across all layers.
struct AttentionTensorData {
  std::vector<AttentionTensorSnapshot> snapshots;
  bool ok{false};
  std::string error;
};

/// Result of applying a chat template.
struct ChatTemplateResult {
  bool valid{false};
  std::string prompt;
  // Which family the template was detected as — lets the caller select a
  // matching response parser (e.g. ReasoningSplitter for ChatML-family
  // <think> tags vs. HarmonySplitter for gpt-oss's channel markers).
  ChatTemplateFamily family{ChatTemplateFamily::kChatML};
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
  // Addressable sequence ids are [0, capacity); zero means unspecified.
  virtual int SequenceCapacity() const { return 0; }
  // Configured tokens per sequence; zero means unknown, not unlimited.
  virtual int SequenceContextCapacity() const { return 0; }
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

  // Ensure the sequence's KV holds exactly positions [0, keep_from).
  // Returns true when that state is guaranteed. Returns false AFTER fully
  // clearing the sequence (hybrid/recurrent memory cannot trim a suffix that
  // includes its final cell) — the caller must then prefill from position 0.
  // The default (unknown capability) reports failure so callers fall back to
  // a full prefill, which is always safe.
  virtual bool TruncateSequence(int sequence_id, int keep_from) {
    (void)sequence_id;
    (void)keep_from;
    return false;
  }

  // Copy the first n_tokens positions of src_seq into dst_seq (dst is fully
  // cleared first). Returns false when the copy could not be completed (e.g.
  // the post-copy trim failed on hybrid memory); dst is fully cleared on
  // failure, so the caller must fall back to a full prefill.
  virtual bool CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) {
    (void)src_seq;
    (void)dst_seq;
    (void)n_tokens;
    return false;
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

  virtual DevicePlacement Placement() const { return {}; }
  virtual std::string LoadError() const { return ""; }
  virtual bool IsReady() const { return false; }

  virtual bool SupportsVision() const { return false; }

  virtual bool IsMoE() const { return false; }
  virtual int ExpertCount() const { return 0; }
  virtual int ActiveExperts() const { return 0; }

  virtual int ContextSize() const { return 0; }
  virtual bool FlashAttentionEnabled() const { return false; }

  // GGUF model metadata for /v1/models API (Ollama-style model details).
  struct ModelMetadata {
    std::string architecture;   // "qwen2", "llama", "gemma", etc.
    std::string quantization;   // "Q4_K_M", "Q6_K", "F16", etc.
    int64_t parameter_count{0}; // Total parameters (approx).
    int context_length{0};      // max_position_embeddings.
    int embedding_length{0};    // hidden_size.
    int num_layers{0};
    int num_heads{0};
    int num_kv_heads{0};       // GQA heads.
    std::string chat_template; // Jinja2 template from GGUF metadata.
  };
  virtual ModelMetadata GetModelMetadata() const { return {}; }

  virtual PerfSnapshot TakePerf() { return {}; }

  virtual std::vector<TopLogitEntry> TopLogitsForParity(int top_n) {
    (void)top_n;
    return {};
  }

  /// Capture intermediate attention tensors for debugging/profiling.
  /// Only implemented when INFERFLUX_DEBUG_ATTENTION_TENSORS=1.
  virtual AttentionTensorData CaptureAttentionTensors() {
    return {{}, false, "Not implemented for this backend"};
  }

  /// Report the full capability set of this backend.
  /// Default returns all-true (matches LlamaCppBackend defaults).
  virtual BackendCapabilities ReportCapabilities() const {
    return BackendCapabilities{};
  }
};

} // namespace inferflux
