#include "runtime/backends/native/native_gpu_backend.h"

#include "model/model_format.h"
#include "model/tokenizer.h"
#include "runtime/backends/backend_utils.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "runtime/string_utils.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#if defined(INFERFLUX_HAS_CUDA)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace inferflux {

namespace {

// Free-VRAM probe for the parity-delegate pre-check. CUDA builds query the
// driver; other device stacks report "unknown" and the pre-check is skipped
// (the lazy load then fails through its normal error path).
bool QueryFreeDeviceMemory(size_t *free_bytes) {
#if defined(INFERFLUX_HAS_CUDA)
  size_t free_v = 0;
  size_t total_v = 0;
  if (cudaMemGetInfo(&free_v, &total_v) != cudaSuccess) {
    return false;
  }
  *free_bytes = free_v;
  return true;
#else
  (void)free_bytes;
  return false;
#endif
}
bool IsVisibleNativePiece(std::string_view piece) { return !piece.empty(); }

int ClampBurstChunkTokens(int value) {
  constexpr int kMaxBurstChunkTokens = 64;
  return std::max(0, std::min(value, kMaxBurstChunkTokens));
}

std::string NormalizeNativeOutputPiece(const ITokenizer *tokenizer,
                                       int token_id,
                                       std::string_view decoded_piece) {
  if (token_id < 0) {
    return {};
  }
  if (tokenizer && tokenizer->IsTerminalGeneratedToken(token_id)) {
    return {};
  }
  if (!decoded_piece.empty()) {
    return std::string(decoded_piece);
  }
  if (!tokenizer) {
    return {};
  }
  return tokenizer->TokenToString(token_id);
}

} // namespace

NativeGpuBackend::NativeGpuBackend(std::unique_ptr<GpuDeviceStrategy> strategy)
    : GpuAcceleratedBackend(std::move(strategy)) {}

NativeGpuBackend::~NativeGpuBackend() = default;

bool NativeGpuBackend::LoadModel(const std::filesystem::path &model_path,
                                 const LlamaBackendConfig &config) {
  const bool strict_native_execution =
      ParseBoolEnv("INFERFLUX_CUDA_STRICT", false);
  loaded_model_path_ = model_path;
  loaded_config_ = config;
  parity_delegate_enabled_ =
      !ParseBoolEnv("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", true);
  parity_delegate_available_ = false;
  parity_delegate_init_attempted_ = false;
  parity_backend_.reset();
  parity_load_path_.clear();

  if (parity_delegate_enabled_) {
    // Full load: device init + llama.cpp scaffold (needed for grammar support)
    if (!GpuAcceleratedBackend::LoadModel(model_path, config)) {
      log::Error(LogTag(),
                 "GpuAcceleratedBackend device init / model load failed");
      return false;
    }
  } else {
    // Lightweight: device init only, skip ~5.7 GB llama.cpp scaffold
    LlamaBackendConfig tuned;
    if (!GpuAcceleratedBackend::InitializeDevice(config, &tuned)) {
      log::Error(LogTag(), "Device initialization failed");
      return false;
    }
    log::Info(LogTag(),
              "Scaffold model load skipped (parity delegate disabled)");
  }

  // Create the device-specific native runtime via the subclass.
  runtime_ = CreateNativeRuntime();
  if (!runtime_) {
    log::Error(LogTag(), "failed to create native runtime");
    return false;
  }

  if (!runtime_->LoadModel(model_path, config)) {
    log::Error(LogTag(),
               "failed to load model using runtime '" + runtime_->Name() + "'");
    return false;
  }

  fallback_mode_ = runtime_->IsFallback();
  fallback_reason_ = runtime_->FallbackReason();
  runtime_kind_ = runtime_->Name();
  if (strict_native_execution && fallback_mode_) {
    std::string strict_reason =
        fallback_reason_.empty()
            ? "strict mode rejected runtime '" + runtime_kind_ + "'"
            : fallback_reason_;
    log::Error(LogTag(), "strict mode rejected model load: " + strict_reason);
    return false;
  }
  if (fallback_mode_ && !fallback_reason_.empty()) {
    log::Warn(LogTag(), fallback_reason_ + " (runtime=" + runtime_kind_ + ")");
  }
  if (parity_delegate_enabled_) {
    const std::string resolved_format =
        ResolveModelFormat(model_path.string(), /*requested_format=*/"auto");
    const std::string parity_path =
        ResolveLlamaLoadPath(model_path.string(), resolved_format);
    if (!parity_path.empty()) {
      parity_load_path_ = parity_path;
      parity_delegate_available_ = true;
    } else {
      log::Info(LogTag(),
                "Native parity delegate unavailable for model path '" +
                    model_path.string() +
                    "' (no GGUF-compatible artifact detected)");
    }
  }
  return true;
}

// --- Runtime delegation ---

std::vector<LlamaCppBackend::UnifiedBatchOutput>
NativeGpuBackend::ExecuteUnifiedBatch(
    const std::vector<UnifiedBatchInput> &inputs) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return {};
  }
  return runtime_->ExecuteUnifiedBatch(inputs);
}

bool NativeGpuBackend::SupportsUnifiedBatchBurst() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return false;
  }
  return runtime_->NativeSupportsUnifiedBatchBurst();
}

UnifiedBurstResult NativeGpuBackend::ExecuteUnifiedBatchBurst(
    const std::vector<UnifiedBatchInput> &inputs,
    const UnifiedBurstOptions &options, const BurstTokenSink &sink) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return UnifiedBurstResult{};
  }
  return runtime_->NativeExecuteUnifiedBatchBurst(inputs, options, sink);
}

bool NativeGpuBackend::SupportsAsyncUnifiedBatch() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return false;
  }
  return runtime_->SupportsAsyncUnifiedBatch();
}

bool NativeGpuBackend::TryGreedyBurstDecodeTokens(
    int sequence_id, int n_past_start, int first_token_id,
    const SamplingParams &sampling, int max_tokens,
    std::vector<BurstDecodeOutput> *outputs, std::string *reason) {
  if (outputs) {
    outputs->clear();
  }

  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    if (reason) {
      *reason = "missing_runtime";
    }
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible("decode",
                                                             "missing_runtime");
    return false;
  }

  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
  if (!tokenizer) {
    if (reason) {
      *reason = "missing_tokenizer";
    }
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible(
        "decode", "missing_tokenizer");
    return false;
  }

  const int burst_chunk_tokens = NativeBurstChunkTokens();
  if (burst_chunk_tokens <= 0) {
    if (reason) {
      *reason = "disabled";
    }
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible("decode",
                                                             "disabled");
    return false;
  }

  if (sampling.temperature > 0.0f) {
    if (reason) {
      *reason = "temperature";
    }
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible("decode",
                                                             "temperature");
    return false;
  }

  const int burst_n = std::min(std::max(1, max_tokens), burst_chunk_tokens);
  std::vector<int> token_ids;
  const int eos_id = tokenizer->EosTokenId();
  const int generated = runtime_->BurstDecodeGreedy(
      sequence_id, n_past_start, first_token_id, burst_n, eos_id, &token_ids);
  if (generated <= 0 || token_ids.empty()) {
    if (reason) {
      *reason = "fallback";
    }
    GlobalMetrics().RecordInferfluxCudaBurstDecodeFallback("decode");
    return false;
  }

  if (outputs) {
    outputs->reserve(token_ids.size());
    for (int token_id : token_ids) {
      BurstDecodeOutput out;
      out.token = token_id;
      out.terminal = tokenizer->IsTerminalGeneratedToken(token_id);
      if (!out.terminal) {
        out.piece =
            NormalizeNativeOutputPiece(tokenizer, token_id, std::string_view{});
      }
      outputs->push_back(std::move(out));
    }
  }
  if (reason) {
    reason->clear();
  }
  GlobalMetrics().RecordInferfluxCudaBurstDecodeChunk("decode", burst_n,
                                                      generated);
  return true;
}

bool NativeGpuBackend::SupportsSplitPrefillDecodeHandoff() const {
  return runtime_ != nullptr && !fallback_mode_;
}

bool NativeGpuBackend::SupportsProcessLocalSequenceTransfer() const {
  return runtime_ != nullptr && !fallback_mode_;
}

LlamaCppBackend::UnifiedBatchHandle NativeGpuBackend::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane lane) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return 0;
  }
  return runtime_->SubmitUnifiedBatchAsync(inputs, lane);
}

bool NativeGpuBackend::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return false;
  }
  return runtime_->TryCollectUnifiedBatchAsync(handle, outputs);
}

int NativeGpuBackend::UnifiedBatchTokenCapacity() const {
  auto backend = DelegateBackend();
  if (!backend) {
    return LlamaCppBackend::UnifiedBatchTokenCapacity();
  }
  return backend->UnifiedBatchTokenCapacity();
}

PrefillResult NativeGpuBackend::Prefill(const std::string &prompt,
                                        int sequence_id) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->Prefill(prompt, sequence_id);
}

PrefillResult NativeGpuBackend::PrefillPartial(const std::string &prompt,
                                               int sequence_id,
                                               int n_past_start) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->PrefillPartial(prompt, sequence_id, n_past_start);
}

void NativeGpuBackend::CopySequencePrefix(int src_seq, int dst_seq,
                                          int n_tokens) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    runtime_->NativeCopySequencePrefix(src_seq, dst_seq, n_tokens);
  }
}

void NativeGpuBackend::FreeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    runtime_->NativeFreeSequence(sequence_id);
  }
}

SequenceReleaseFence NativeGpuBackend::BeginFreeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeBeginFreeSequence(sequence_id);
  }
  auto backend = DelegateBackend();
  return backend ? backend->BeginFreeSequence(sequence_id)
                 : SequenceReleaseFence{};
}

bool NativeGpuBackend::PollFreeSequence(const SequenceReleaseFence &fence) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativePollFreeSequence(fence);
  }
  auto backend = DelegateBackend();
  return backend ? backend->PollFreeSequence(fence) : true;
}

std::vector<uint8_t> NativeGpuBackend::SerializeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    auto blob = runtime_->NativeSerializeSequence(sequence_id);
    if (!blob.empty()) {
      return blob;
    }
  }
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->SerializeSequence(sequence_id);
}

bool NativeGpuBackend::HydrateSequence(int dest_sequence_id,
                                       const std::vector<uint8_t> &blob) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_ && runtime_->NativeHydrateSequence(dest_sequence_id, blob)) {
    return true;
  }
  auto backend = DelegateBackend();
  if (!backend) {
    return false;
  }
  return backend->HydrateSequence(dest_sequence_id, blob);
}

// --- Decode / Generate token loops ---

std::string NativeGpuBackend::Decode(
    int n_past, int sequence_id, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs, int first_token,
    const std::vector<std::string> &stop_seqs) {
  if (UsesStructuredConstraintSampler()) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Decode(n_past, sequence_id, max_tokens, on_chunk,
                                    should_stop, logprob_top_n, out_logprobs,
                                    first_token, stop_seqs);
    }
  }

  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);

  if (!runtime_ || sequence_id < 0 || n_past < 0 || max_tokens <= 0) {
    return {};
  }

  const bool collect_lp = (logprob_top_n > 0) || (out_logprobs != nullptr);
  const SamplingParams sampling = SnapshotSamplingParams();
  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
  if (first_token < 0 || tokenizer == nullptr) {
    log::Warn(LogTag(), "native decode requires first_token and tokenizer");
    return {};
  }

  std::string output;
  int current_token = first_token;
  int visible_tokens_generated = 0;
  int non_emitting_steps = 0;
  const int max_non_emitting_steps = std::max(max_tokens * 8, 32);

  if (max_tokens > 0) {
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      return output;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   std::string_view{});
    if (IsVisibleNativePiece(piece)) {
      output += piece;
      ++visible_tokens_generated;

      std::string emit_piece;
      const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
      if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
        return output;
      }
      if (stop_hit || (should_stop && should_stop())) {
        return output;
      }
    } else {
      ++non_emitting_steps;
    }
  }

  const int burst_chunk_tokens = NativeBurstChunkTokens();
  const char *burst_ineligible_reason =
      burst_chunk_tokens > 0
          ? NativeBurstDecodeIneligibleReason(sampling, collect_lp,
                                              static_cast<bool>(on_chunk),
                                              stop_seqs, tokenizer)
          : "disabled";
  const bool can_burst =
      burst_chunk_tokens > 0 && burst_ineligible_reason == nullptr;
  if (burst_ineligible_reason != nullptr) {
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible(
        "decode", burst_ineligible_reason);
  }
  const int eos_id = tokenizer ? tokenizer->EosTokenId() : -1;

  while (visible_tokens_generated < max_tokens) {
    if (should_stop && should_stop()) {
      break;
    }

    if (can_burst) {
      const int remaining = max_tokens - visible_tokens_generated;
      const int burst_n = std::min(remaining, burst_chunk_tokens);
      std::vector<int> burst_tokens;
      const int generated = runtime_->BurstDecodeGreedy(
          sequence_id, n_past, current_token, burst_n, eos_id, &burst_tokens);
      if (generated > 0) {
        GlobalMetrics().RecordInferfluxCudaBurstDecodeChunk("decode", burst_n,
                                                            generated);
        bool stop = false;
        for (int token_id : burst_tokens) {
          ++n_past;
          current_token = token_id;
          if (tokenizer->IsTerminalGeneratedToken(current_token)) {
            stop = true;
            break;
          }
          std::string piece = NormalizeNativeOutputPiece(
              tokenizer, current_token, std::string_view{});
          if (!IsVisibleNativePiece(piece)) {
            if (++non_emitting_steps >= max_non_emitting_steps) {
              log::Warn(LogTag(),
                        "native decode aborted after too many non-emitting "
                        "tokens");
              stop = true;
              break;
            }
            continue;
          }

          non_emitting_steps = 0;
          output += piece;
          ++visible_tokens_generated;
        }
        if (stop || burst_tokens.empty()) {
          break;
        }
        continue;
      }
      GlobalMetrics().RecordInferfluxCudaBurstDecodeFallback("decode");
    }

    UnifiedBatchInput step_input;
    step_input.sequence_id = sequence_id;
    step_input.n_past = n_past;
    step_input.tokens = {current_token};
    step_input.request_logits = true;
    step_input.sampling = sampling;

    auto step = runtime_->ExecuteUnifiedBatch({step_input});
    if (step.empty() || !step.front().ok || step.front().token < 0) {
      break;
    }

    ++n_past;
    current_token = step.front().token;
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      break;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   step.front().piece);
    if (!IsVisibleNativePiece(piece)) {
      if (++non_emitting_steps >= max_non_emitting_steps) {
        log::Warn(LogTag(),
                  "native decode aborted after too many non-emitting tokens");
        break;
      }
      continue;
    }

    non_emitting_steps = 0;
    output += piece;
    ++visible_tokens_generated;

    TokenLogprob tlp;
    const TokenLogprob *tlp_ptr = nullptr;
    if (collect_lp) {
      tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
      tlp_ptr = &tlp;
      if (out_logprobs) {
        out_logprobs->push_back(tlp);
      }
    }

    std::string emit_piece;
    const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
      break;
    }
    if (stop_hit) {
      break;
    }
  }

  return output;
}

std::string NativeGpuBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs,
    const std::vector<std::string> &stop_seqs) {
  if (UsesStructuredConstraintSampler()) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Generate(prompt, max_tokens, on_chunk, should_stop,
                                      logprob_top_n, out_logprobs, stop_seqs);
    }
  }

  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);

  if (!runtime_ || max_tokens <= 0) {
    return {};
  }

  const bool collect_lp = (logprob_top_n > 0) || (out_logprobs != nullptr);
  const std::vector<int> prompt_tokens = runtime_->NativeTokenize(prompt);
  if (prompt_tokens.empty()) {
    return {};
  }

  constexpr int kDirectGenerateSequenceId = 0;
  const int sequence_id = kDirectGenerateSequenceId;
  const SamplingParams sampling = SnapshotSamplingParams();
  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
  std::string output;

  struct SequenceReleaser final {
    NativeInferenceRuntime *runtime{nullptr};
    int sequence_id{-1};
    ~SequenceReleaser() {
      if (runtime && sequence_id >= 0) {
        runtime->NativeFreeSequence(sequence_id);
      }
    }
  } release{runtime_.get(), sequence_id};

  runtime_->NativeFreeSequence(sequence_id);

  UnifiedBatchInput prefill_input;
  prefill_input.sequence_id = sequence_id;
  prefill_input.n_past = 0;
  prefill_input.tokens = prompt_tokens;
  prefill_input.request_logits = true;
  prefill_input.sampling = sampling;

  auto first_step = runtime_->ExecuteUnifiedBatch({prefill_input});
  if (first_step.empty() || !first_step.front().ok ||
      first_step.front().token < 0) {
    return {};
  }

  int current_token = first_step.front().token;
  int n_past = static_cast<int>(prompt_tokens.size());
  int visible_tokens_generated = 0;
  int non_emitting_steps = 0;
  const int max_non_emitting_steps = std::max(max_tokens * 8, 32);

  if (max_tokens > 0) {
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      return output;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   first_step.front().piece);
    if (IsVisibleNativePiece(piece)) {
      output += piece;
      ++visible_tokens_generated;

      TokenLogprob tlp;
      const TokenLogprob *tlp_ptr = nullptr;
      if (collect_lp) {
        tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
        tlp_ptr = &tlp;
        if (out_logprobs) {
          out_logprobs->push_back(tlp);
        }
      }

      std::string emit_piece;
      const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
      if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
        return output;
      }
      if (stop_hit || (should_stop && should_stop())) {
        return output;
      }
    } else {
      ++non_emitting_steps;
    }
  }

  // Burst decode fast path: when greedy (temperature <= 0), no logprobs,
  // and no streaming callback, run chunks of tokens on device without
  // per-token host sync. This eliminates WDDM scheduling latency on Windows
  // while preserving the standard path for logprobs/streaming/stop control.
  const int burst_chunk_tokens = NativeBurstChunkTokens();
  const char *burst_ineligible_reason =
      burst_chunk_tokens > 0
          ? NativeBurstDecodeIneligibleReason(sampling, collect_lp,
                                              static_cast<bool>(on_chunk),
                                              stop_seqs, tokenizer)
          : "disabled";
  const bool can_burst =
      burst_chunk_tokens > 0 && burst_ineligible_reason == nullptr;
  if (burst_ineligible_reason != nullptr) {
    GlobalMetrics().RecordInferfluxCudaBurstDecodeIneligible(
        "generate", burst_ineligible_reason);
  }
  const int eos_id = tokenizer ? tokenizer->EosTokenId() : -1;

  while (visible_tokens_generated < max_tokens) {
    if (should_stop && should_stop()) {
      break;
    }

    // Try burst decode path
    if (can_burst) {
      const int remaining = max_tokens - visible_tokens_generated;
      const int burst_n = std::min(remaining, burst_chunk_tokens);
      std::vector<int> burst_tokens;

      int generated = runtime_->BurstDecodeGreedy(
          sequence_id, n_past, current_token, burst_n, eos_id, &burst_tokens);

      if (generated > 0) {
        GlobalMetrics().RecordInferfluxCudaBurstDecodeChunk("generate", burst_n,
                                                            generated);
        // Process burst results on host
        bool stop = false;
        for (int i = 0; i < static_cast<int>(burst_tokens.size()); ++i) {
          ++n_past;
          current_token = burst_tokens[i];
          if (tokenizer->IsTerminalGeneratedToken(current_token)) {
            stop = true;
            break;
          }
          std::string piece = NormalizeNativeOutputPiece(
              tokenizer, current_token, std::string_view{});
          if (!IsVisibleNativePiece(piece)) {
            if (++non_emitting_steps >= max_non_emitting_steps) {
              log::Warn(LogTag(),
                        "native generate aborted after too many non-emitting "
                        "tokens");
              stop = true;
              break;
            }
            continue;
          }
          non_emitting_steps = 0;
          output += piece;
          ++visible_tokens_generated;
        }
        if (stop || burst_tokens.empty()) {
          break;
        }
        continue; // next burst
      }
      GlobalMetrics().RecordInferfluxCudaBurstDecodeFallback("generate");
      // BurstDecodeGreedy returned 0 — fall through to standard path
    }

    // Standard per-token path (with logprobs, streaming, or stochastic)
    UnifiedBatchInput step_input;
    step_input.sequence_id = sequence_id;
    step_input.n_past = n_past;
    step_input.tokens = {current_token};
    step_input.request_logits = true;
    step_input.sampling = sampling;

    auto step = runtime_->ExecuteUnifiedBatch({step_input});
    if (step.empty() || !step.front().ok || step.front().token < 0) {
      break;
    }

    ++n_past;
    current_token = step.front().token;
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      break;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   step.front().piece);
    if (!IsVisibleNativePiece(piece)) {
      if (++non_emitting_steps >= max_non_emitting_steps) {
        log::Warn(LogTag(),
                  "native generate aborted after too many non-emitting tokens");
        break;
      }
      continue;
    }

    non_emitting_steps = 0;
    output += piece;
    ++visible_tokens_generated;

    TokenLogprob tlp;
    const TokenLogprob *tlp_ptr = nullptr;
    if (collect_lp) {
      tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
      tlp_ptr = &tlp;
      if (out_logprobs) {
        out_logprobs->push_back(tlp);
      }
    }

    std::string emit_piece;
    const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
      break;
    }
    if (stop_hit) {
      break;
    }
  }

  return output;
}

// --- Sampler ---

void NativeGpuBackend::SetupSampler(const std::string &grammar,
                                    const std::string &root,
                                    const SamplingParams &sp) {
  if (auto backend = DelegateBackend()) {
    backend->SetupSampler(grammar, root, sp);
  }
  if (!grammar.empty()) {
    if (auto parity_backend = EnsureParityBackend()) {
      parity_backend->SetupSampler(grammar, root, sp);
    }
  }
  if (!grammar.empty()) {
    if (!IsParityDelegateAvailable()) {
      log::Warn(LogTag(),
                "native sampler grammar constraints are unavailable (no parity "
                "delegate backend)");
    }
  }
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  active_sampling_ = sp;
  sampling_active_ = true;
  structured_constraint_sampler_active_ = !grammar.empty();
}

void NativeGpuBackend::TeardownSampler() {
  if (auto backend = DelegateBackend()) {
    backend->TeardownSampler();
  }
  {
    std::lock_guard<std::mutex> lock(parity_backend_mutex_);
    if (parity_backend_) {
      parity_backend_->TeardownSampler();
    }
  }
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  active_sampling_ = SamplingParams{};
  sampling_active_ = false;
  structured_constraint_sampler_active_ = false;
}

// --- Perf / Chat / Token / IsReady ---

PerfSnapshot NativeGpuBackend::TakePerf() {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  auto backend = DelegateBackend();
  if (backend) {
    return backend->TakePerf();
  }
  if (runtime_) {
    auto native = runtime_->NativeTakePerf();
    PerfSnapshot snap;
    snap.prefill_ms = native.prefill_ms;
    snap.decode_ms = native.decode_ms;
    snap.prompt_tokens = native.prompt_tokens;
    snap.generated_tokens = native.generated_tokens;
    return snap;
  }
  return {};
}

ChatTemplateResult NativeGpuBackend::FormatChatMessages(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    auto native = runtime_->NativeFormatChat(messages, add_assistant_prefix);
    if (native.valid) {
      ChatTemplateResult result;
      result.prompt = std::move(native.prompt);
      result.valid = true;
      return result;
    }
  }
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->FormatChatMessages(messages, add_assistant_prefix);
}

int NativeGpuBackend::TokenCount(const std::string &text) const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeTokenCount(text);
  }
  return 0;
}

std::vector<int>
NativeGpuBackend::TokenizeForCache(const std::string &prompt) const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeTokenize(prompt);
  }
  return {};
}

bool NativeGpuBackend::IsReady() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeIsReady();
  }
  return false;
}

// --- Logprobs / Embeddings ---

std::vector<TopLogitEntry> NativeGpuBackend::TopLogitsForParity(int top_n) {
  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
  if (!runtime_ || top_n <= 0) {
    return {};
  }
  const int vocab = runtime_->NativeVocabSize();
  if (vocab <= 0) {
    return {};
  }
  if (static_cast<int>(host_logits_buf_.size()) < vocab) {
    host_logits_buf_.resize(vocab);
  }
  const int copied =
      runtime_->CopyLastLogitsToHost(host_logits_buf_.data(), vocab);
  if (copied <= 0) {
    return {};
  }
  const ITokenizer *tok = runtime_->NativeGetTokenizer();
  return ComputeTopLogits(host_logits_buf_.data(), vocab, top_n,
                          [tok](int32_t id) -> std::string {
                            return tok ? tok->TokenToString(id) : "";
                          });
}

std::vector<float> NativeGpuBackend::EmbedForParity(const std::string &text) {
  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
  if (runtime_ && runtime_->NativeEmbedDims() > 0) {
    auto result = runtime_->NativeEmbed(text);
    if (!result.empty()) {
      return result;
    }
  }
  auto backend = EnsureParityBackend();
  if (!backend || !backend->IsReady()) {
    return {};
  }
  return backend->Embed(text);
}

int NativeGpuBackend::EmbedDimsForParity() const {
  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
  if (runtime_) {
    int dims = runtime_->NativeEmbedDims();
    if (dims > 0) {
      return dims;
    }
  }
  auto backend = EnsureParityBackend();
  if (!backend || !backend->IsReady()) {
    return 0;
  }
  return backend->EmbedDims();
}

// --- Embedding overrides ---

std::vector<float> NativeGpuBackend::Embed(const std::string &text) {
  return EmbedForParity(text);
}

int NativeGpuBackend::EmbedDims() const { return EmbedDimsForParity(); }

// --- Capability reporting ---

BackendCapabilities NativeGpuBackend::ReportCapabilities() const {
  BackendCapabilities caps;
  caps.supports_logprobs = SupportsLogprobsContract();
  caps.supports_structured_output = SupportsStructuredOutputContract();
  caps.supports_embeddings = SupportsEmbeddingsContract();
  caps.supports_speculative_decoding = SupportsSpeculativeDecodingContract();
  caps.supports_vision = SupportsVision();
  caps.supports_kv_prefix_transfer = true;
  return caps;
}

// --- Capability contracts ---

bool NativeGpuBackend::SupportsSpeculativeDecodingContract() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  return runtime_ && runtime_->NativeIsReady() &&
         runtime_->NativeGetTokenizer() != nullptr;
}

bool NativeGpuBackend::SupportsLogprobsContract() const { return true; }

bool NativeGpuBackend::SupportsStructuredOutputContract() const {
  return IsParityDelegateAvailable();
}

bool NativeGpuBackend::SupportsEmbeddingsContract() const { return true; }

// --- Private helpers ---

bool NativeGpuBackend::IsParityDelegateAvailable() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  return runtime_ && runtime_->NativeIsReady() && parity_delegate_enabled_ &&
         parity_delegate_available_;
}

std::shared_ptr<LlamaCppBackend> NativeGpuBackend::EnsureParityBackend() const {
  if (!IsParityDelegateAvailable()) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(parity_backend_mutex_);
  if (parity_backend_) {
    return parity_backend_;
  }
  if (parity_delegate_init_attempted_) {
    return nullptr;
  }
  parity_delegate_init_attempted_ = true;

  // Pre-check free VRAM so a doomed lazy load fails fast with a clear
  // message instead of an OOM inside llama.cpp. The delegate re-reads the
  // GGUF through llama.cpp, so budget at least the model file's size.
  {
    std::error_code ec;
    const auto model_bytes = std::filesystem::file_size(parity_load_path_, ec);
    size_t free_bytes = 0;
    if (!ec && QueryFreeDeviceMemory(&free_bytes) &&
        free_bytes < static_cast<size_t>(model_bytes)) {
      log::Warn(
          LogTag(),
          "Skipping parity delegate load: only " +
              std::to_string(free_bytes / (1024 * 1024)) +
              " MiB free but model needs at least " +
              std::to_string(static_cast<size_t>(model_bytes) / (1024 * 1024)) +
              " MiB; structured-output remains unavailable on native "
              "provider");
      parity_delegate_available_ = false;
      return nullptr;
    }
  }

  auto backend = std::make_shared<LlamaCppBackend>();
  const auto tuned_cfg = TuneLlamaBackendConfig(ParityTarget(), loaded_config_);
  if (!backend->LoadModel(parity_load_path_, tuned_cfg)) {
    log::Warn(LogTag(), "Failed to initialize parity delegate backend from '" +
                            parity_load_path_.string() +
                            "'; structured-output/embeddings remain "
                            "unavailable on native provider");
    parity_delegate_available_ = false;
    return nullptr;
  }

  parity_backend_ = std::move(backend);
  log::Info(LogTag(), "Initialized native parity delegate backend from '" +
                          parity_load_path_.string() + "'");
  return parity_backend_;
}

std::shared_ptr<BackendInterface> NativeGpuBackend::DelegateBackend() const {
  if (!runtime_) {
    return nullptr;
  }
  return runtime_->BackendHandle();
}

bool NativeGpuBackend::UsesStructuredConstraintSampler() const {
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  return structured_constraint_sampler_active_;
}

SamplingParams NativeGpuBackend::SnapshotSamplingParams() const {
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  if (!sampling_active_) {
    return {};
  }
  return active_sampling_;
}

int NativeGpuBackend::NativeBurstChunkTokens() const {
  const char *raw = std::getenv("INFERFLUX_NATIVE_BURST_CHUNK_TOKENS");
  if (!raw || *raw == '\0') {
#ifdef _WIN32
    return 8;
#else
    return 0;
#endif
  }
  try {
    return ClampBurstChunkTokens(std::stoi(raw));
  } catch (const std::exception &) {
    return 0;
  }
}

const char *NativeGpuBackend::NativeBurstDecodeIneligibleReason(
    const SamplingParams &sampling, bool collect_logprobs,
    bool has_chunk_callback, const std::vector<std::string> &stop_seqs,
    const ITokenizer *tokenizer) const {
  if (!runtime_) {
    return "missing_runtime";
  }
  if (!tokenizer) {
    return "missing_tokenizer";
  }
  if (collect_logprobs) {
    return "logprobs";
  }
  if (has_chunk_callback) {
    return "streaming_callback";
  }
  if (sampling.temperature > 0.0f) {
    return "temperature";
  }
  if (!stop_seqs.empty()) {
    return "stop_sequences";
  }
  return nullptr;
}

TokenLogprob NativeGpuBackend::CollectNativeLogprob(int token_id,
                                                    const std::string &piece,
                                                    int top_n) {
  const int vocab = runtime_->NativeVocabSize();
  if (vocab <= 0) {
    return {};
  }
  if (static_cast<int>(host_logits_buf_.size()) < vocab) {
    host_logits_buf_.resize(vocab);
  }
  int copied = runtime_->CopyLastLogitsToHost(host_logits_buf_.data(), vocab);
  if (copied <= 0) {
    return {};
  }
  const ITokenizer *tok = runtime_->NativeGetTokenizer();
  return ComputeLogprob(host_logits_buf_.data(), vocab, token_id, piece, top_n,
                        [tok](int32_t id) -> std::string {
                          return tok ? tok->TokenToString(id) : "";
                        });
}

} // namespace inferflux
