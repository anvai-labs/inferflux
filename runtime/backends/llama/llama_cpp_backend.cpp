#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/backends/backend_utils.h"
#include "runtime/execution/parallel_context.h"
#include "server/logging/logger.h"

#include <llama.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void BatchClear(llama_batch &batch) { batch.n_tokens = 0; }

struct BatchTokenInput {
  llama_token id;
  llama_pos pos;
  bool logits{false};
};

void BatchAdd(llama_batch &batch, const BatchTokenInput &input) {
  if (!batch.seq_id[batch.n_tokens]) {
    throw std::runtime_error("llama_batch capacity exceeded");
  }
  batch.token[batch.n_tokens] = input.id;
  batch.pos[batch.n_tokens] = input.pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = 0;
  batch.logits[batch.n_tokens] = input.logits ? 1 : 0;
  batch.n_tokens++;
}

// Sequence-aware variant for phased prefill/decode (§2.5 Option A).
struct BatchSeqTokenInput {
  llama_token id;
  llama_pos pos;
  llama_seq_id seq_id;
  bool logits{false};
};

void BatchAddSeq(llama_batch &batch, const BatchSeqTokenInput &input) {
  if (!batch.seq_id[batch.n_tokens]) {
    throw std::runtime_error("llama_batch capacity exceeded");
  }
  batch.token[batch.n_tokens] = input.id;
  batch.pos[batch.n_tokens] = input.pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = input.seq_id;
  batch.logits[batch.n_tokens] = input.logits ? 1 : 0;
  batch.n_tokens++;
}

int EffectiveBatchTokenCap(const llama_context *ctx, int config_batch_size) {
  int cap = std::max(1, config_batch_size);
  if (ctx) {
    int per_sequence_cap = static_cast<int>(llama_n_ctx_seq(ctx));
    if (per_sequence_cap > 1) {
      // Keep one KV position free to avoid edge-case slot exhaustion on
      // full-capacity submissions.
      cap = std::min(cap, per_sequence_cap - 1);
    } else if (per_sequence_cap > 0) {
      cap = std::min(cap, per_sequence_cap);
    }
  }
  return std::max(1, cap);
}

} // namespace

namespace inferflux {

namespace {
std::mutex g_llama_init_mutex;
int g_llama_init_refcount = 0;
using BackendStateLock = std::lock_guard<std::recursive_mutex>;

bool SizeToInt32(std::size_t value, int32_t *out) {
  if (!out) {
    return false;
  }
  if (value > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  *out = static_cast<int32_t>(value);
  return true;
}

bool TokenTraceDebugEnabled() {
  static const bool enabled =
      std::getenv("INFERFLUX_DEBUG_TOKEN_TRACE") != nullptr;
  return enabled;
}

bool ConsumeTokenTraceBudget() {
  static const int initial_budget = [] {
    int parsed = 128;
    if (const char *env = std::getenv("INFERFLUX_DEBUG_TOKEN_TRACE_LIMIT")) {
      char *end = nullptr;
      const long value = std::strtol(env, &end, 10);
      if (end != env && *end == '\0' && value > 0 &&
          value <= static_cast<long>(std::numeric_limits<int>::max())) {
        parsed = static_cast<int>(value);
      }
    }
    return parsed;
  }();
  static std::atomic<int> remaining(initial_budget);
  int current = remaining.load(std::memory_order_relaxed);
  while (current > 0) {
    if (remaining.compare_exchange_weak(current, current - 1,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool LogitsDebugEnabled() {
  static const bool enabled = std::getenv("INFERFLUX_DEBUG_LOGITS") != nullptr;
  return enabled;
}

bool ConsumeLogitsDebugBudget() {
  static const int initial_budget = [] {
    int parsed = 64;
    if (const char *env = std::getenv("INFERFLUX_DEBUG_LOGITS_LIMIT")) {
      char *end = nullptr;
      const long value = std::strtol(env, &end, 10);
      if (end != env && *end == '\0' && value > 0 &&
          value <= static_cast<long>(std::numeric_limits<int>::max())) {
        parsed = static_cast<int>(value);
      }
    }
    return parsed;
  }();
  static std::atomic<int> remaining(initial_budget);
  int current = remaining.load(std::memory_order_relaxed);
  while (current > 0) {
    if (remaining.compare_exchange_weak(current, current - 1,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void LogTokenTrace(std::string_view stage, int64_t request_id, int sequence_id,
                   std::string_view client_request_id,
                   uint64_t sequence_generation, int n_past, int token_id,
                   const std::string &piece) {
  if (!TokenTraceDebugEnabled() || !ConsumeTokenTraceBudget()) {
    return;
  }
  const std::string client_prefix =
      client_request_id.empty()
          ? std::string()
          : "client_request_id=" + std::string(client_request_id) + ", ";
  log::Info("llama_backend",
            "token_trace[" + std::string(stage) + "]: " + client_prefix +
                "request_id=" + std::to_string(request_id) +
                ", sequence_id=" + std::to_string(sequence_id) +
                ", sequence_generation=" + std::to_string(sequence_generation) +
                ", n_past=" + std::to_string(n_past) + ", sampled_token=" +
                std::to_string(token_id) + ", piece=" + piece);
}

void LogTopLogits(std::string_view stage, const float *logits, int vocab_size,
                  int64_t request_id, std::string_view client_request_id,
                  int sequence_id, uint64_t sequence_generation, int n_past) {
  if (!LogitsDebugEnabled() || !ConsumeLogitsDebugBudget() || !logits ||
      vocab_size <= 0) {
    return;
  }
  constexpr std::size_t kTopK = 5;
  std::array<std::pair<float, int>, kTopK> top{};
  for (auto &entry : top) {
    entry = {std::numeric_limits<float>::lowest(), -1};
  }
  for (int i = 0; i < vocab_size; ++i) {
    const float value = logits[i];
    for (std::size_t slot = 0; slot < kTopK; ++slot) {
      if (value > top[slot].first) {
        for (std::size_t shift = kTopK - 1; shift > slot; --shift) {
          top[shift] = top[shift - 1];
        }
        top[slot] = {value, i};
        break;
      }
    }
  }
  std::string payload;
  for (std::size_t i = 0; i < kTopK; ++i) {
    if (top[i].second < 0) {
      continue;
    }
    if (!payload.empty()) {
      payload += " ";
    }
    payload += "[" + std::to_string(top[i].second) +
               "]=" + std::to_string(top[i].first);
  }
  const std::string client_prefix =
      client_request_id.empty()
          ? std::string()
          : "client_request_id=" + std::string(client_request_id) + ", ";
  log::Info("llama_backend",
            "debug_logits[" + std::string(stage) + "]: " + client_prefix +
                "request_id=" + std::to_string(request_id) +
                ", sequence_id=" + std::to_string(sequence_id) +
                ", sequence_generation=" + std::to_string(sequence_generation) +
                ", n_past=" + std::to_string(n_past) + " top-5: " + payload);
}

void LlamaBackendAcquire() {
  std::lock_guard<std::mutex> lock(g_llama_init_mutex);
  if (g_llama_init_refcount++ == 0) {
    llama_backend_init();
  }
}

void LlamaBackendRelease() {
  std::lock_guard<std::mutex> lock(g_llama_init_mutex);
  if (--g_llama_init_refcount == 0) {
    llama_backend_free();
  }
}
} // namespace

LlamaCppBackend::LlamaCppBackend() : LlamaCppBackend(true) {}

LlamaCppBackend::LlamaCppBackend(bool acquire_backend)
    : llama_backend_acquired_(acquire_backend) {
  if (llama_backend_acquired_) {
    LlamaBackendAcquire();
  }
}

LlamaCppBackend::~LlamaCppBackend() {
  TeardownSamplerImpl();
#ifdef INFERFLUX_HAS_MTMD
  if (mtmd_ctx_) {
    mtmd_free(mtmd_ctx_);
    mtmd_ctx_ = nullptr;
  }
#endif
  if (embed_ctx_ != nullptr) {
    llama_free(embed_ctx_);
    embed_ctx_ = nullptr;
  }
  if (context_ != nullptr) {
    llama_free(context_);
    context_ = nullptr;
  }
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  if (llama_backend_acquired_) {
    LlamaBackendRelease();
  }
}

bool LlamaCppBackend::LoadModel(const std::filesystem::path &model_path,
                                const LlamaBackendConfig &config) {
  BackendStateLock lock(backend_state_mutex_);
  test_ready_ = false;
  if (!std::filesystem::exists(model_path)) {
    log::Error("llama_backend",
               "model path does not exist: " + model_path.string());
    return false;
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config.gpu_layers;

  model_ =
      llama_model_load_from_file(model_path.string().c_str(), model_params);
  if (!model_) {
    log::Error("llama_backend",
               "failed to load model from " + model_path.string());
    return false;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = config.ctx_size;
  ctx_params.n_batch = config.batch_size;
  // Pre-allocate logit output rows for all concurrent sequences so that
  // BatchDecodeStep() can call llama_get_logits_ith(ctx, i) correctly.
  ctx_params.n_seq_max =
      static_cast<uint32_t>(std::max(config.max_parallel_sequences, 1));

  // Wire Flash Attention (§2.7): set llama.cpp context parameter directly.
  // LLAMA_FLASH_ATTN_TYPE_ENABLED lets llama.cpp choose FA on any supported
  // backend (Metal, CUDA); the tile parameter is stored for future FA3 CUDA
  // integration.
  ctx_params.flash_attn_type = config.use_flash_attention
                                   ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                   : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  if (config.use_flash_attention) {
    log::Info("llama_backend", "FlashAttention enabled (tile=" +
                                   std::to_string(config.flash_attention_tile) +
                                   ")");
  }

  context_ = llama_init_from_model(model_, ctx_params);
  if (!context_) {
    log::Error("llama_backend", "failed to create context");
    return false;
  }
  vocab_ = llama_model_get_vocab(model_);
  if (!vocab_) {
    log::Error("llama_backend", "failed to obtain vocabulary");
    return false;
  }
  n_vocab_ = llama_vocab_n_tokens(vocab_);
  config_ = config;

  // §P1f: Initialize Expert Parallel dispatcher.
  if (IsMoE()) {
    int n_experts = ExpertCount();
    auto &pc = ParallelContext::Get();
    if (pc.IsInitialized() && pc.WorldSize() > 1) {
      ep_dispatch_ = std::make_shared<DistributedEPDispatch>(
          pc.Rank(), pc.WorldSize(), n_experts);
    } else {
      ep_dispatch_ = std::make_shared<LocalEPDispatch>(n_experts);
    }
  }

  return true;
}

int LlamaCppBackend::TokenCount(const std::string &text) const {
  auto tokens = Tokenize(text, /*add_bos=*/false);
  return static_cast<int>(tokens.size());
}

std::vector<int>
LlamaCppBackend::TokenizeForCache(const std::string &prompt) const {
  // Returns the BPE token sequence (with BOS) for use in the KV prefix store.
  // Returns empty when no model is loaded so the caller can skip prefix reuse.
  return Tokenize(prompt, /*add_bos=*/true);
}

std::vector<int> LlamaCppBackend::Tokenize(const std::string &prompt,
                                           bool add_bos) const {
  if (!model_) {
    return {};
  }
  if (!vocab_) {
    return {};
  }
  std::vector<llama_token> tokens;
  tokens.resize(prompt.size() + 8);
  int32_t prompt_len = 0;
  int32_t token_capacity = 0;
  if (!SizeToInt32(prompt.size(), &prompt_len) ||
      !SizeToInt32(tokens.size(), &token_capacity)) {
    log::Error("llama_backend", "Prompt/token capacity exceeds int32 range");
    return {};
  }
  int n = llama_tokenize(vocab_, prompt.c_str(), prompt_len, tokens.data(),
                         token_capacity, add_bos, true);
  if (n < 0) {
    tokens.resize(static_cast<std::size_t>(-n));
    if (!SizeToInt32(tokens.size(), &token_capacity)) {
      log::Error("llama_backend", "Token capacity exceeds int32 range");
      return {};
    }
    n = llama_tokenize(vocab_, prompt.c_str(), prompt_len, tokens.data(),
                       token_capacity, add_bos, true);
  }
  tokens.resize(n);
  return {tokens.begin(), tokens.end()};
}

void LlamaCppBackend::SetupSampler(const std::string &grammar,
                                   const std::string &root,
                                   const SamplingParams &sp) {
  BackendStateLock lock(backend_state_mutex_);
  TeardownSampler();
  if (!vocab_) {
    return;
  }
  auto params = llama_sampler_chain_default_params();
  auto *chain = llama_sampler_chain_init(params);

  // Grammar constraint (if provided).
  if (!grammar.empty()) {
    llama_sampler_chain_add(chain, llama_sampler_init_grammar(
                                       vocab_, grammar.c_str(), root.c_str()));
  }

  // Repetition / frequency / presence penalties.
  bool has_penalties =
      (sp.frequency_penalty != 0.0f || sp.presence_penalty != 0.0f ||
       sp.repetition_penalty != 1.0f);
  if (has_penalties) {
    llama_sampler_chain_add(chain,
                            llama_sampler_init_penalties(
                                sp.penalty_last_n, sp.repetition_penalty,
                                sp.frequency_penalty, sp.presence_penalty));
  }

  // Logit bias: bias specific tokens.
  if (!sp.logit_bias.empty()) {
    std::vector<llama_logit_bias> llama_logit_biases;
    llama_logit_biases.reserve(sp.logit_bias.size());
    for (const auto &[token_id, bias] : sp.logit_bias) {
      llama_logit_biases.push_back({static_cast<llama_token>(token_id), bias});
    }
    llama_sampler_chain_add(
        chain, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab_),
                                             llama_logit_biases.size(),
                                             llama_logit_biases.data()));
  }

  // Top-K filtering.
  if (sp.top_k > 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
  }

  // Min-P filtering.
  if (sp.min_p > 0.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
  }

  // Top-P (nucleus) filtering.
  if (sp.top_p < 1.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
  }

  // Terminal: greedy when temperature <= 0, stochastic otherwise.
  if (sp.temperature <= 0.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
  }

  active_sampler_ = chain;
}

void LlamaCppBackend::TeardownSampler() {
  BackendStateLock lock(backend_state_mutex_);
  TeardownSamplerImpl();
}

void LlamaCppBackend::TeardownSamplerImpl() {
  if (active_sampler_) {
    llama_sampler_free(active_sampler_);
    active_sampler_ = nullptr;
  }
}

std::string LlamaCppBackend::TokenToString(int token) const {
  if (!vocab_) {
    return {};
  }
  std::string buf;
  buf.resize(16);
  int32_t buf_capacity = 0;
  if (!SizeToInt32(buf.size(), &buf_capacity)) {
    return {};
  }
  int written =
      llama_token_to_piece(vocab_, token, buf.data(), buf_capacity, 0, false);
  if (written < 0) {
    buf.resize(static_cast<std::size_t>(-written));
    if (!SizeToInt32(buf.size(), &buf_capacity)) {
      return {};
    }
    if (llama_token_to_piece(vocab_, token, buf.data(), buf_capacity, 0,
                             false) < 0) {
      return {};
    }
  } else {
    buf.resize(static_cast<std::size_t>(written));
  }
  return buf;
}

bool LlamaCppBackend::IsTerminalGeneratedToken(int token) const {
  if (token < 0 || !vocab_) {
    return token < 0;
  }
  return llama_vocab_is_eog(vocab_, static_cast<llama_token>(token));
}

std::string LlamaCppBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs,
    const std::vector<std::string> &stop_seqs) {
  BackendStateLock lock(backend_state_mutex_);
  log::Debug("llama_backend",
             "Generate() called: prompt_len=" + std::to_string(prompt.size()) +
                 ", max_tokens=" + std::to_string(max_tokens));

  if (!IsReady()) {
    log::Error("llama_backend", "Generate() called but backend not ready");
    return {};
  }

  auto tokenization_start = std::chrono::high_resolution_clock::now();
  auto prompt_tokens = Tokenize(prompt, true);
  auto tokenization_end = std::chrono::high_resolution_clock::now();
  auto tokenization_ms = std::chrono::duration<float, std::milli>(
                             tokenization_end - tokenization_start)
                             .count();

  log::Debug("llama_backend",
             "Tokenization complete: " + std::to_string(prompt_tokens.size()) +
                 " tokens in " + std::to_string(tokenization_ms) + "ms");

  if (prompt_tokens.empty()) {
    log::Warn("llama_backend", "Tokenization returned empty tokens");
    return {};
  }

  // Generate() always uses sequence 0; clear it so each request starts with a
  // consistent KV position baseline.
  llama_memory_seq_rm(llama_get_memory(context_), 0, -1, -1);
  log::Debug("llama_backend", "KV cache cleared for sequence 0");

  const int32_t token_cap = static_cast<int32_t>(
      EffectiveBatchTokenCap(context_, config_.batch_size));
  log::Debug("llama_backend", "Batch token cap: " + std::to_string(token_cap));

  llama_batch batch = llama_batch_init(token_cap, 0, 1);
  llama_pos position = 0;

  // Process prompt tokens in batches
  auto prefill_start = std::chrono::high_resolution_clock::now();
  int batch_count = 0;
  for (std::size_t start = 0; start < prompt_tokens.size();
       start += static_cast<std::size_t>(token_cap)) {
    std::size_t end = std::min(prompt_tokens.size(),
                               start + static_cast<std::size_t>(token_cap));
    log::Debug("llama_backend", "Processing prompt batch " +
                                    std::to_string(batch_count) + ": tokens [" +
                                    std::to_string(start) + "-" +
                                    std::to_string(end) + "]");

    for (std::size_t i = start; i < end; ++i) {
      BatchAdd(batch,
               {prompt_tokens[i], position++, i == prompt_tokens.size() - 1});
    }

    log::Debug("llama_backend", "Calling llama_decode for prompt batch " +
                                    std::to_string(batch_count) + " (" +
                                    std::to_string(end - start) + " tokens)");
    auto decode_start = std::chrono::high_resolution_clock::now();

    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend", "llama_decode failed for prompt");
      llama_batch_free(batch);
      return {};
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_ms =
        std::chrono::duration<float, std::milli>(decode_end - decode_start)
            .count();
    log::Debug("llama_backend", "llama_decode completed for batch " +
                                    std::to_string(batch_count) + " in " +
                                    std::to_string(decode_ms) + "ms");

    BatchClear(batch);
    batch_count++;
  }

  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto prefill_ms =
      std::chrono::duration<float, std::milli>(prefill_end - prefill_start)
          .count();
  const float prefill_per_token_ms =
      prefill_ms / static_cast<float>(prompt_tokens.size());
  log::Info("llama_backend",
            "Prefill complete: " + std::to_string(prompt_tokens.size()) +
                " tokens in " + std::to_string(prefill_ms) + "ms (" +
                std::to_string(prefill_per_token_ms) + "ms/token)");

  std::string output;
  int tokens_remaining = std::max(max_tokens, 1);

  log::Debug("llama_backend", "Starting generation loop: max_tokens=" +
                                  std::to_string(tokens_remaining));

  auto generation_start = std::chrono::high_resolution_clock::now();
  int generated_count = 0;

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop()) {
      log::Debug("llama_backend", "Generation stopped by should_stop callback");
      break;
    }

    LogTopLogits("generate", llama_get_logits(context_), n_vocab_, -1, "", 0, 0,
                 static_cast<int>(position));
    auto sample_start = std::chrono::high_resolution_clock::now();
    int token = llama_sampler_sample(active_sampler_, context_, -1);
    auto sample_end = std::chrono::high_resolution_clock::now();
    auto sample_ms =
        std::chrono::duration<float, std::milli>(sample_end - sample_start)
            .count();

    if (IsTerminalGeneratedToken(token)) {
      log::Debug("llama_backend", "EOS token received at position " +
                                      std::to_string(generated_count));
      break;
    }

    std::string piece = TokenToString(token);
    LogTokenTrace("generate", -1, 0, "", 0, static_cast<int>(position), token,
                  piece);
    // Collect logprob before the next llama_decode invalidates the logits ptr.
    if (out_logprobs) {
      out_logprobs->push_back(CollectLogprob(token, piece, logprob_top_n));
    }
    output += piece;
    const int emitted_index = generated_count;
    generated_count++;

    if (emitted_index == 0 || emitted_index % 10 == 0) {
      log::Debug("llama_backend",
                 "Generated token " + std::to_string(emitted_index) +
                     ": sample_ms=" + std::to_string(sample_ms) +
                     ", output_len=" + std::to_string(output.size()));
    }

    // Check stop sequences: trim output and compute the streaming-safe portion.
    std::string emit_piece;
    bool stop_triggered = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty()) {
      const TokenLogprob *lp_ptr = (out_logprobs && !out_logprobs->empty())
                                       ? &out_logprobs->back()
                                       : nullptr;
      if (!on_chunk(emit_piece, lp_ptr)) {
        log::Debug("llama_backend", "Generation stopped by on_chunk callback");
        break;
      }
    }
    if (stop_triggered) {
      log::Debug("llama_backend", "Stop sequence triggered at token " +
                                      std::to_string(generated_count));
      break;
    }
    if (should_stop && should_stop()) {
      log::Debug("llama_backend",
                 "Generation stopped by should_stop callback (post-chunk)");
      break;
    }
    if (tokens_remaining <= 0) {
      break;
    }
    // Context-window management: sliding-window KV eviction when the next
    // token would exceed n_ctx.  We discard the oldest half of the KV cache
    // and shift the remaining positions so generation can continue.
    {
      llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
      if (position >= n_ctx - 1 && n_ctx > 1) {
        llama_pos keep = n_ctx / 2;
        llama_pos discard = position - keep + 1;
        if (discard > 0) {
          log::Debug("llama_backend",
                     "Sliding window KV eviction: discarding " +
                         std::to_string(discard) + " positions, keeping " +
                         std::to_string(keep));
          llama_memory_seq_rm(llama_get_memory(context_), 0, 0, discard);
          llama_memory_seq_add(llama_get_memory(context_), 0, discard,
                               static_cast<llama_pos>(-1), -discard);
          position -= discard;
        }
      }
    }
    BatchAdd(batch, {token, position++, true});

    auto decode_start = std::chrono::high_resolution_clock::now();
    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend",
                 "llama_decode failed while generating at token " +
                     std::to_string(generated_count));
      break;
    }
    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_ms =
        std::chrono::duration<float, std::milli>(decode_end - decode_start)
            .count();

    if (generated_count == 0 || generated_count % 10 == 0) {
      log::Debug("llama_backend", "llama_decode for generated token: " +
                                      std::to_string(decode_ms) + "ms");
    }

    BatchClear(batch);
  }

  auto generation_end = std::chrono::high_resolution_clock::now();
  auto generation_ms = std::chrono::duration<float, std::milli>(
                           generation_end - generation_start)
                           .count();
  const float generation_token_count =
      static_cast<float>(std::max(1, generated_count));
  log::Info("llama_backend",
            "Generation complete: " + std::to_string(generated_count) +
                " tokens in " + std::to_string(generation_ms) + "ms (" +
                std::to_string(generation_ms / generation_token_count) +
                "ms/token)");

  llama_batch_free(batch);

  // Capture GGML-native timing (t_p_eval_ms = prefill, t_eval_ms = decode).
  if (context_) {
    auto raw = llama_perf_context(context_);
    last_perf_ = {raw.t_p_eval_ms, raw.t_eval_ms, raw.n_p_eval, raw.n_eval};
    llama_perf_context_reset(context_);
  }

  return output;
}

bool LlamaCppBackend::LoadMmproj(const std::filesystem::path &mmproj_path) {
  BackendStateLock lock(backend_state_mutex_);
#ifdef INFERFLUX_HAS_MTMD
  if (!model_) {
    log::Error("llama_backend", "LoadMmproj: text model must be loaded first");
    return false;
  }
  if (mtmd_ctx_) {
    mtmd_free(mtmd_ctx_);
    mtmd_ctx_ = nullptr;
    vision_ready_ = false;
  }
  auto params = mtmd_context_params_default();
  params.use_gpu = (config_.gpu_layers > 0);
  params.n_threads = 4;
  params.print_timings = false;
  params.warmup = false;
  mtmd_ctx_ = mtmd_init_from_file(mmproj_path.string().c_str(), model_, params);
  if (!mtmd_ctx_) {
    log::Error("llama_backend",
               "failed to load mmproj from " + mmproj_path.string());
    return false;
  }
  vision_ready_ = mtmd_support_vision(mtmd_ctx_);
  log::Info("llama_backend", "mmproj loaded from " + mmproj_path.string() +
                                 " (vision=" + std::to_string(vision_ready_) +
                                 ")");
  return vision_ready_;
#else
  (void)mmproj_path;
  log::Error("llama_backend", "ENABLE_MTMD=OFF; vision support unavailable");
  return false;
#endif
}

std::string LlamaCppBackend::GenerateWithImages(
    const std::string &prompt, const std::vector<DecodedImage> &images,
    int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop,
    const std::vector<std::string> &stop_seqs) {
  BackendStateLock lock(backend_state_mutex_);
#ifdef INFERFLUX_HAS_MTMD
  if (!IsReady() || !vision_ready_ || !mtmd_ctx_ || images.empty()) {
    return Generate(prompt, max_tokens, on_chunk, should_stop, 0, nullptr,
                    stop_seqs);
  }

  // Build bitmap list from raw image bytes.
  std::vector<mtmd_bitmap *> bitmaps;
  bitmaps.reserve(images.size());
  for (const auto &img : images) {
    if (img.raw_bytes.empty())
      continue;
    auto *bmp = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx_, img.raw_bytes.data(), img.raw_bytes.size());
    if (!bmp) {
      log::Error("llama_backend",
                 "failed to decode image bitmap; falling back to text-only");
      for (auto *b : bitmaps)
        mtmd_bitmap_free(b);
      return Generate(prompt, max_tokens, on_chunk, should_stop, 0, nullptr,
                      stop_seqs);
    }
    if (!img.image_id.empty()) {
      mtmd_bitmap_set_id(bmp, img.image_id.c_str());
    }
    bitmaps.push_back(bmp);
  }

  if (bitmaps.empty()) {
    return Generate(prompt, max_tokens, on_chunk, should_stop, 0, nullptr,
                    stop_seqs);
  }

  // Tokenize prompt + image bitmaps into interleaved chunks.
  auto *chunks = mtmd_input_chunks_init();
  mtmd_input_text input_text;
  input_text.text = prompt.c_str();
  input_text.add_special = true;
  input_text.parse_special = true;
  const mtmd_bitmap **bitmap_ptr =
      const_cast<const mtmd_bitmap **>(bitmaps.data());
  int32_t rc =
      mtmd_tokenize(mtmd_ctx_, chunks, &input_text, bitmap_ptr, bitmaps.size());
  for (auto *b : bitmaps)
    mtmd_bitmap_free(b);
  if (rc != 0) {
    log::Error("llama_backend", "mtmd_tokenize failed (rc=" +
                                    std::to_string(rc) + "); falling back");
    mtmd_input_chunks_free(chunks);
    return Generate(prompt, max_tokens, on_chunk, should_stop, 0, nullptr,
                    stop_seqs);
  }

  // Evaluate all chunks (text decode + image encode) through the model.
  llama_pos n_past = 0;
  int32_t n_batch = config_.batch_size;
  rc = mtmd_helper_eval_chunks(mtmd_ctx_, context_, chunks, n_past, 0, n_batch,
                               /*logits_last=*/true, &n_past);
  mtmd_input_chunks_free(chunks);
  if (rc != 0) {
    log::Error("llama_backend", "mtmd_helper_eval_chunks failed (rc=" +
                                    std::to_string(rc) + ")");
    return {};
  }

  // Decode loop: generate tokens starting from the updated n_past position.
  int32_t batch_cap = std::max<int32_t>(config_.batch_size, max_tokens + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);

  std::string output;
  int tokens_remaining = std::max(max_tokens, 1);

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop())
      break;
    int token = llama_sampler_sample(active_sampler_, context_, -1);
    if (IsTerminalGeneratedToken(token))
      break;
    std::string piece = TokenToString(token);
    output += piece;
    std::string emit_piece;
    bool stop_triggered = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr))
      break;
    if (stop_triggered)
      break;
    if (should_stop && should_stop())
      break;
    BatchAdd(batch, {token, n_past++, true});
    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend", "llama_decode failed (vision decode loop)");
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);
  return output;
#else
  (void)images;
  return Generate(prompt, max_tokens, on_chunk, should_stop, 0, nullptr,
                  stop_seqs);
#endif
}

LlamaCppBackend::PrefillResult
LlamaCppBackend::Prefill(const std::string &prompt, int sequence_id) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_ || !vocab_) {
    return {};
  }
  auto prompt_tokens = Tokenize(prompt, /*add_bos=*/true);
  if (prompt_tokens.empty()) {
    return {};
  }
  // Clear any previous KV state for this sequence slot before filling it.
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(sequence_id), -1, -1);

  const int32_t token_cap = static_cast<int32_t>(
      EffectiveBatchTokenCap(context_, config_.batch_size));
  llama_batch batch = llama_batch_init(token_cap, 0, 1);
  for (std::size_t start = 0; start < prompt_tokens.size();
       start += static_cast<std::size_t>(token_cap)) {
    std::size_t end = std::min(prompt_tokens.size(),
                               start + static_cast<std::size_t>(token_cap));
    for (std::size_t i = start; i < end; ++i) {
      BatchAddSeq(batch, {prompt_tokens[i], static_cast<llama_pos>(i),
                          static_cast<llama_seq_id>(sequence_id),
                          /*logits=*/i == prompt_tokens.size() - 1});
    }
    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend", "Prefill: llama_decode failed for seq " +
                                      std::to_string(sequence_id));
      llama_batch_free(batch);
      return {};
    }
    BatchClear(batch);
  }
  // Sample the first output token while the logit buffer is fresh.
  // This must happen before any subsequent Prefill() call, which would
  // overwrite the buffer and cause a logit-buffer race in multi-seq batches.
  // Greedy sampling is used here (not per-request params) because
  // active_sampler_ is set up by SamplerScope in batch_executor only for the
  // Decode() phase.
  PrefillResult result;
  result.n_past = static_cast<int>(prompt_tokens.size());
  result.ok = true;
  {
    const float *logits = llama_get_logits(context_);
    int first_tok = -1;
    if (logits) {
      LogTopLogits("prefill", logits, n_vocab_, -1, "", sequence_id, 0,
                   static_cast<int>(prompt_tokens.size()) - 1);
      auto max_it = std::max_element(logits, logits + n_vocab_);
      first_tok = static_cast<int>(std::distance(logits, max_it));
    }
    if (!IsTerminalGeneratedToken(first_tok)) {
      result.first_token = first_tok;
      result.first_piece = TokenToString(first_tok);
      LogTokenTrace("prefill", -1, sequence_id, "", 0, result.n_past, first_tok,
                    result.first_piece);
    }
  }
  llama_batch_free(batch);
  return result;
}

void LlamaCppBackend::CopySequencePrefix(int src_seq, int dst_seq,
                                         int n_tokens) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_)
    return;
  // Clear dst slot first so no stale KV cells survive from a previous request.
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(dst_seq), -1, -1);
  if (n_tokens <= 0) {
    return;
  }
  // Partial seq_cp can assert on cross-stream KV layouts; copy full source KV
  // state first, then trim the destination to [0, n_tokens) via seq_rm.
  llama_memory_seq_cp(llama_get_memory(context_),
                      static_cast<llama_seq_id>(src_seq),
                      static_cast<llama_seq_id>(dst_seq),
                      static_cast<llama_pos>(0), static_cast<llama_pos>(-1));
  llama_memory_seq_rm(
      llama_get_memory(context_), static_cast<llama_seq_id>(dst_seq),
      static_cast<llama_pos>(n_tokens), static_cast<llama_pos>(-1));
}

// Legacy API shape retained for backend compatibility.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
LlamaCppBackend::PrefillResult
LlamaCppBackend::PrefillPartial(const std::string &prompt, int sequence_id,
                                int n_past_start) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_ || !vocab_)
    return {};
  auto prompt_tokens = Tokenize(prompt, /*add_bos=*/true);
  if (prompt_tokens.empty())
    return {};
  int n_total = static_cast<int>(prompt_tokens.size());
  if (n_past_start >= n_total) {
    // Prefix covers the entire prompt; no suffix tokens to evaluate and no
    // fresh logits from which to sample first_token.
    PrefillResult r;
    r.ok = true;
    r.n_past = n_total;
    return r;
  }
  const int32_t token_cap = static_cast<int32_t>(
      EffectiveBatchTokenCap(context_, config_.batch_size));
  llama_batch batch = llama_batch_init(token_cap, 0, 1);
  for (int start = n_past_start; start < n_total; start += token_cap) {
    int end = std::min(n_total, start + token_cap);
    for (int i = start; i < end; ++i) {
      BatchAddSeq(batch, {prompt_tokens[static_cast<std::size_t>(i)],
                          static_cast<llama_pos>(i),
                          static_cast<llama_seq_id>(sequence_id),
                          /*logits=*/i == n_total - 1});
    }
    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend", "PrefillPartial: llama_decode failed seq " +
                                      std::to_string(sequence_id));
      llama_batch_free(batch);
      return {};
    }
    BatchClear(batch);
  }
  PrefillResult result;
  result.n_past = n_total;
  result.ok = true;
  {
    const float *logits = llama_get_logits(context_);
    int first_tok = -1;
    if (logits) {
      LogTopLogits("prefill_partial", logits, n_vocab_, -1, "", sequence_id, 0,
                   n_total - 1);
      auto max_it = std::max_element(logits, logits + n_vocab_);
      first_tok = static_cast<int>(std::distance(logits, max_it));
    }
    if (!IsTerminalGeneratedToken(first_tok)) {
      result.first_token = first_tok;
      result.first_piece = TokenToString(first_tok);
      LogTokenTrace("prefill_partial", -1, sequence_id, "", 0, result.n_past,
                    first_tok, result.first_piece);
    }
  }
  llama_batch_free(batch);
  return result;
}

std::string LlamaCppBackend::Decode(
    int n_past, int sequence_id, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs, int first_token,
    const std::vector<std::string> &stop_seqs) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_ || !vocab_ || n_past < 0) {
    return {};
  }
  int32_t batch_cap = std::max<int32_t>(config_.batch_size, max_tokens + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  llama_pos position = static_cast<llama_pos>(n_past);

  std::string output;
  int tokens_remaining = std::max(max_tokens, 1);

  // If a first token was pre-sampled by Prefill() (to avoid the logit-buffer
  // race in multi-sequence prefill), emit and feed it before the loop.
  if (!IsTerminalGeneratedToken(first_token) && tokens_remaining > 0) {
    std::string piece = TokenToString(first_token);
    LogTokenTrace("decode_seed", -1, sequence_id, "", 0,
                  static_cast<int>(position), first_token, piece);
    if (out_logprobs) {
      out_logprobs->push_back(
          CollectLogprob(first_token, piece, logprob_top_n));
    }
    output += piece;
    tokens_remaining--;
    std::string emit_ft;
    bool ft_stop = ApplyStop(piece, output, stop_seqs, &emit_ft);
    if (on_chunk && !emit_ft.empty()) {
      const TokenLogprob *ft_lp = (out_logprobs && !out_logprobs->empty())
                                      ? &out_logprobs->back()
                                      : nullptr;
      if (!on_chunk(emit_ft, ft_lp)) {
        llama_batch_free(batch);
        return output;
      }
    }
    if (ft_stop) {
      llama_batch_free(batch);
      return output;
    }
    if (should_stop && should_stop()) {
      llama_batch_free(batch);
      return output;
    }
    BatchAddSeq(batch, {first_token, position++,
                        static_cast<llama_seq_id>(sequence_id), true});
    if (llama_decode(context_, batch) != 0) {
      llama_batch_free(batch);
      return output;
    }
    BatchClear(batch);
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop())
      break;
    LogTopLogits("decode", llama_get_logits(context_), n_vocab_, -1, "",
                 sequence_id, 0, static_cast<int>(position) - 1);
    int token = llama_sampler_sample(active_sampler_, context_, -1);
    if (IsTerminalGeneratedToken(token))
      break;
    std::string piece = TokenToString(token);
    LogTokenTrace("decode", -1, sequence_id, "", 0, static_cast<int>(position),
                  token, piece);
    if (out_logprobs) {
      out_logprobs->push_back(CollectLogprob(token, piece, logprob_top_n));
    }
    output += piece;
    std::string emit_piece;
    bool stop_triggered = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty()) {
      const TokenLogprob *lp_ptr = (out_logprobs && !out_logprobs->empty())
                                       ? &out_logprobs->back()
                                       : nullptr;
      if (!on_chunk(emit_piece, lp_ptr))
        break;
    }
    if (stop_triggered)
      break;
    if (should_stop && should_stop())
      break;
    // Context-window management: sliding-window KV eviction (same as
    // Generate(), but scoped to the sequence slot for phased decode).
    {
      llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
      if (position >= n_ctx - 1 && n_ctx > 1) {
        llama_pos keep = n_ctx / 2;
        llama_pos discard = position - keep + 1;
        if (discard > 0) {
          llama_memory_seq_rm(llama_get_memory(context_),
                              static_cast<llama_seq_id>(sequence_id), 0,
                              discard);
          llama_memory_seq_add(llama_get_memory(context_),
                               static_cast<llama_seq_id>(sequence_id), discard,
                               static_cast<llama_pos>(-1), -discard);
          position -= discard;
        }
      }
    }
    BatchAddSeq(batch, {token, position++,
                        static_cast<llama_seq_id>(sequence_id), true});
    if (llama_decode(context_, batch) != 0) {
      log::Error("llama_backend", "Decode: llama_decode failed for seq " +
                                      std::to_string(sequence_id));
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);

  // Capture GGML-native timing for Decode phase.
  if (context_) {
    auto raw = llama_perf_context(context_);
    last_perf_ = {raw.t_p_eval_ms, raw.t_eval_ms, raw.n_p_eval, raw.n_eval};
    llama_perf_context_reset(context_);
  }

  return output;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

std::vector<LlamaCppBackend::BatchDecodeOutput>
LlamaCppBackend::BatchDecodeStep(std::vector<BatchDecodeInput> &inputs) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_ || !vocab_ || inputs.empty()) {
    return {};
  }
  int n = static_cast<int>(inputs.size());
  llama_batch batch = llama_batch_init(n, 0, 1);

  for (int i = 0; i < n; ++i) {
    auto &inp = inputs[i];
    // Per-sequence context-window eviction (mirrors the logic in Decode()).
    llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
    if (inp.n_past >= static_cast<int>(n_ctx) - 1 && n_ctx > 1) {
      llama_pos keep = n_ctx / 2;
      llama_pos discard = static_cast<llama_pos>(inp.n_past) - keep + 1;
      if (discard > 0) {
        llama_memory_seq_rm(llama_get_memory(context_),
                            static_cast<llama_seq_id>(inp.sequence_id), 0,
                            discard);
        llama_memory_seq_add(llama_get_memory(context_),
                             static_cast<llama_seq_id>(inp.sequence_id),
                             discard, static_cast<llama_pos>(-1), -discard);
        inp.n_past -= static_cast<int>(discard);
      }
    }
    BatchAddSeq(batch, {inp.feed_token, static_cast<llama_pos>(inp.n_past),
                        static_cast<llama_seq_id>(inp.sequence_id),
                        /*logits=*/true});
  }

  if (llama_decode(context_, batch) != 0) {
    log::Error("llama_backend", "BatchDecodeStep: llama_decode failed");
    llama_batch_free(batch);
    return {};
  }
  llama_batch_free(batch);

  std::vector<BatchDecodeOutput> results(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    inputs[i].n_past++; // advance position for the next step
    const float *logits = llama_get_logits_ith(context_, i);
    if (!logits) {
      results[static_cast<std::size_t>(i)].token = -1;
      continue;
    }
    LogTopLogits("batch_decode", logits, n_vocab_, -1, "",
                 inputs[i].sequence_id, 0, inputs[i].n_past - 1);
    auto max_it = std::max_element(logits, logits + n_vocab_);
    int tok = static_cast<int>(std::distance(logits, max_it));
    if (IsTerminalGeneratedToken(tok)) {
      results[static_cast<std::size_t>(i)].token = -1;
    } else {
      results[static_cast<std::size_t>(i)].token = tok;
      results[static_cast<std::size_t>(i)].piece = TokenToString(tok);
      LogTokenTrace("batch_decode", -1, inputs[i].sequence_id, "", 0,
                    inputs[i].n_past, tok,
                    results[static_cast<std::size_t>(i)].piece);
    }
  }
  return results;
}

bool LlamaCppBackend::SupportsAsyncUnifiedBatch() const { return false; }

int LlamaCppBackend::UnifiedBatchTokenCapacity() const {
  return EffectiveBatchTokenCap(context_, config_.batch_size);
}

LlamaCppBackend::UnifiedBatchHandle LlamaCppBackend::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane /*lane*/) {
  auto outputs = ExecuteUnifiedBatch(inputs);
  std::lock_guard<std::mutex> lock(async_results_mutex_);
  const UnifiedBatchHandle handle = next_async_handle_++;
  async_results_[handle] = std::move(outputs);
  return handle;
}

bool LlamaCppBackend::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
  if (!outputs || handle == 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(async_results_mutex_);
  auto it = async_results_.find(handle);
  if (it == async_results_.end()) {
    return false;
  }
  *outputs = std::move(it->second);
  async_results_.erase(it);
  return true;
}

std::vector<LlamaCppBackend::UnifiedBatchOutput>
LlamaCppBackend::ExecuteUnifiedBatch(
    const std::vector<UnifiedBatchInput> &inputs) {
  BackendStateLock lock(backend_state_mutex_);
  log::Debug("llama_backend", "ExecuteUnifiedBatch called with " +
                                  std::to_string(inputs.size()) + " inputs");

  if (!context_ || !vocab_ || inputs.empty()) {
    log::Warn("llama_backend",
              "ExecuteUnifiedBatch: context/vocab not ready or inputs empty");
    return {};
  }
  std::vector<UnifiedBatchOutput> results(inputs.size());

  // Count total tokens and requests for logits.
  int total_tokens = 0;
  int requests_with_logits = 0;
  for (const auto &inp : inputs) {
    total_tokens += static_cast<int>(inp.tokens.size());
    if (inp.request_logits) {
      requests_with_logits++;
    }
  }

  log::Debug(
      "llama_backend",
      "ExecuteUnifiedBatch: total_tokens=" + std::to_string(total_tokens) +
          ", requests_with_logits=" + std::to_string(requests_with_logits));

  if (total_tokens == 0) {
    log::Debug("llama_backend", "ExecuteUnifiedBatch: no tokens to process");
    return std::vector<UnifiedBatchOutput>(inputs.size());
  }

  const int token_cap = EffectiveBatchTokenCap(context_, config_.batch_size);
  if (total_tokens > token_cap) {
    log::Error("llama_backend", "ExecuteUnifiedBatch: token batch exceeds cap "
                                "(" +
                                    std::to_string(total_tokens) + " > " +
                                    std::to_string(token_cap) + ")");
    return results;
  }

  llama_batch batch = llama_batch_init(total_tokens, 0, 1);
  std::vector<int> logit_indices(inputs.size(), -1);
  int token_offset = 0;

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &inp = inputs[i];
    if (inp.tokens.empty())
      continue;

    log::Debug("llama_backend",
               "Input " + std::to_string(i) +
                   ": seq_id=" + std::to_string(inp.sequence_id) +
                   ", n_past=" + std::to_string(inp.n_past) +
                   ", tokens=" + std::to_string(inp.tokens.size()));

    for (std::size_t j = 0; j < inp.tokens.size(); ++j) {
      bool is_last = (j == inp.tokens.size() - 1);
      bool should_request = is_last && inp.request_logits;

      BatchAddSeq(batch,
                  {inp.tokens[j], static_cast<llama_pos>(inp.n_past + j),
                   static_cast<llama_seq_id>(inp.sequence_id), should_request});

      if (should_request) {
        // llama_get_logits_ith() / llama_sampler_sample(..., ith) expects the
        // token index within the submitted batch, not the ordinal index among
        // logits-enabled requests.
        logit_indices[i] = token_offset;
      }
      token_offset++;
    }
  }

  log::Debug("llama_backend", "Calling llama_decode with " +
                                  std::to_string(total_tokens) + " tokens...");
  auto decode_start = std::chrono::high_resolution_clock::now();

  if (llama_decode(context_, batch) != 0) {
    log::Error("llama_backend", "ExecuteUnifiedBatch: llama_decode failed");
    llama_batch_free(batch);
    return results;
  }

  auto decode_end = std::chrono::high_resolution_clock::now();
  auto decode_ms =
      std::chrono::duration<float, std::milli>(decode_end - decode_start)
          .count();
  const float token_count = static_cast<float>(std::max(1, total_tokens));
  log::Info("llama_backend",
            "llama_decode completed: " + std::to_string(decode_ms) + "ms for " +
                std::to_string(total_tokens) + " tokens (" +
                std::to_string(decode_ms / token_count) + "ms/token)");

  llama_batch_free(batch);

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (logit_indices[i] >= 0) {
      const auto &inp = inputs[i];

      // Correctness Fix (§P1b): Use per-request sampling parameters.
      // We leverage SetupSampler to prepare the backend's active_sampler_
      // for this specific sequence's logit set.
      SetupSampler("", "", inp.sampling);

      if (active_sampler_) {
        const float *logits = llama_get_logits_ith(context_, logit_indices[i]);
        LogTopLogits("unified_batch", logits, n_vocab_, inp.request_id,
                     inp.client_request_id, inp.sequence_id,
                     inp.sequence_generation,
                     inp.n_past + static_cast<int>(inp.tokens.size()) - 1);
        int tok =
            llama_sampler_sample(active_sampler_, context_, logit_indices[i]);
        llama_sampler_accept(active_sampler_, tok);

        results[i].ok = true;
        if (IsTerminalGeneratedToken(tok)) {
          results[i].token = -1;
        } else {
          results[i].token = tok;
          results[i].piece = TokenToString(tok);
          LogTokenTrace("unified_batch", inp.request_id, inp.sequence_id,
                        inp.client_request_id, inp.sequence_generation,
                        inp.n_past + static_cast<int>(inp.tokens.size()), tok,
                        results[i].piece);
        }
      } else {
        // Fallback to greedy if SetupSampler failed.
        const float *logits = llama_get_logits_ith(context_, logit_indices[i]);
        if (logits) {
          LogTopLogits("unified_batch_fallback", logits, n_vocab_,
                       inp.request_id, inp.client_request_id, inp.sequence_id,
                       inp.sequence_generation,
                       inp.n_past + static_cast<int>(inp.tokens.size()) - 1);
          auto max_it = std::max_element(logits, logits + n_vocab_);
          int tok = static_cast<int>(std::distance(logits, max_it));
          results[i].ok = true;
          if (IsTerminalGeneratedToken(tok)) {
            results[i].token = -1;
          } else {
            results[i].token = tok;
            results[i].piece = TokenToString(tok);
            LogTokenTrace("unified_batch_fallback", inp.request_id,
                          inp.sequence_id, inp.client_request_id,
                          inp.sequence_generation,
                          inp.n_past + static_cast<int>(inp.tokens.size()), tok,
                          results[i].piece);
          }
        }
      }
      TeardownSampler();
    } else if (inputs[i].request_logits) {
      // Requested logits but no tokens were provided in this batch step.
      results[i].ok = false;
    } else {
      results[i].ok = true; // No logits requested, step succeeded.
    }
  }

  return results;
}

void LlamaCppBackend::FreeSequence(int sequence_id) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_)
    return;
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(sequence_id), -1, -1);
}

LlamaCppBackend::SequenceReleaseFence
LlamaCppBackend::BeginFreeSequence(int sequence_id) {
  FreeSequence(sequence_id);
  (void)sequence_id;
  return {};
}

bool LlamaCppBackend::PollFreeSequence(const SequenceReleaseFence &fence) {
  (void)fence;
  return true;
}

std::vector<uint8_t> LlamaCppBackend::SerializeSequence(int sequence_id) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_)
    return {};
  std::size_t size = llama_state_seq_get_size(
      context_, static_cast<llama_seq_id>(sequence_id));
  if (size == 0)
    return {};
  std::vector<uint8_t> buf(size);
  std::size_t written = llama_state_seq_get_data(
      context_, buf.data(), buf.size(), static_cast<llama_seq_id>(sequence_id));
  if (written == 0)
    return {};
  buf.resize(written);
  return buf;
}

bool LlamaCppBackend::HydrateSequence(int dest_sequence_id,
                                      const std::vector<uint8_t> &blob) {
  BackendStateLock lock(backend_state_mutex_);
  if (!context_ || blob.empty())
    return false;
  std::size_t read =
      llama_state_seq_set_data(context_, blob.data(), blob.size(),
                               static_cast<llama_seq_id>(dest_sequence_id));
  return read > 0;
}

bool LlamaCppBackend::IsMoE() const { return ExpertCount() > 0; }

int LlamaCppBackend::ExpertCount() const {
  if (!model_)
    return 0;
  char buf[32] = {};
  int32_t len =
      llama_model_meta_val_str(model_, "llm.expert_count", buf, sizeof(buf));
  if (len <= 0)
    return 0;
  return std::atoi(buf);
}

int LlamaCppBackend::ActiveExperts() const {
  if (!model_)
    return 0;
  char buf[32] = {};
  int32_t len = llama_model_meta_val_str(model_, "llm.expert_used_count", buf,
                                         sizeof(buf));
  if (len <= 0)
    return 0;
  return std::atoi(buf);
}

LlamaCppBackend::PerfSnapshot LlamaCppBackend::TakePerf() {
  BackendStateLock lock(backend_state_mutex_);
  auto snap = last_perf_;
  last_perf_ = {};
  return snap;
}

void LlamaCppBackend::EnableGrammarConstraint(const std::string &grammar,
                                              const std::string &root) {
  SetupSampler(grammar, root, {});
}

void LlamaCppBackend::DisableGrammarConstraint() { TeardownSampler(); }

TokenLogprob LlamaCppBackend::CollectLogprob(int token_id,
                                             const std::string &token_str,
                                             int top_n) const {
  const float *logits = llama_get_logits(context_);
  return ComputeLogprob(logits, n_vocab_, token_id, token_str, top_n,
                        [this](int32_t id) { return TokenToString(id); });
}

std::vector<TopLogitEntry> LlamaCppBackend::TopLogitsForParity(int top_n) {
  BackendStateLock lock(backend_state_mutex_);
  const float *logits = context_ ? llama_get_logits(context_) : nullptr;
  return ComputeTopLogits(logits, n_vocab_, top_n,
                          [this](int32_t id) { return TokenToString(id); });
}

// §2.3 — model-native chat template formatting.
// Uses llama_chat_apply_template() from llama.h (built into the llama lib,
// no extra compilation units required).  The function reads the model's
// built-in chat template from GGUF metadata (NULL tmpl argument selects the
// model's own template).  Supported models include Llama 3.x, Mistral/Mixtral,
// Qwen 2/2.5, Hermes, DeepSeek, Phi-3, ChatML, and others in the predefined
// list; for unsupported models it returns valid=false and the caller falls
// back to the plain text preamble.
LlamaCppBackend::ChatTemplateResult LlamaCppBackend::FormatChatMessages(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  ChatTemplateResult result;
  if (!model_ || messages.empty()) {
    return result;
  }

  // Keep content strings alive for the duration of the C-struct array.
  std::vector<std::string> contents;
  contents.reserve(messages.size());
  std::vector<llama_chat_message> chat;
  chat.reserve(messages.size());
  for (const auto &[role, content] : messages) {
    contents.push_back(content);
    chat.push_back({role.c_str(), contents.back().c_str()});
  }

  // Initial buffer: 2× total content length + headroom for role tokens.
  int total_chars = 0;
  for (const auto &[r, c] : messages) {
    total_chars += static_cast<int>(r.size() + c.size());
  }
  int buf_size = std::max(4096, total_chars * 2 + 512);
  std::vector<char> buf(buf_size);

  // NULL template → use the model's built-in template.
  int32_t n =
      llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                add_assistant_prefix, buf.data(), buf_size);
  if (n < 0) {
    // Template not in the predefined list; caller falls back to preamble.
    return result;
  }
  if (n > buf_size) {
    // Buffer was too small; retry with exact size.
    buf.resize(static_cast<std::size_t>(n) + 1);
    n = llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                  add_assistant_prefix, buf.data(), n);
    if (n < 0) {
      return result;
    }
  }

  result.prompt = std::string(buf.data(), static_cast<std::size_t>(n));
  result.valid = true;
  return result;
}

bool LlamaCppBackend::EnsureEmbedCtx() {
  if (embed_ctx_)
    return true;
  if (!model_)
    return false;
  auto ep = llama_context_default_params();
  ep.embeddings = true;
  ep.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  ep.n_ctx = 512;
  embed_ctx_ = llama_init_from_model(model_, ep);
  return embed_ctx_ != nullptr;
}

int LlamaCppBackend::EmbedDims() const {
  if (!model_)
    return 0;
  return llama_model_n_embd(model_);
}

std::vector<float> LlamaCppBackend::Embed(const std::string &text) {
  BackendStateLock lock(backend_state_mutex_);
  if (!EnsureEmbedCtx())
    return {};
  if (!vocab_)
    return {};

  // Tokenize without BOS — standard for embedding models.
  auto tokens = Tokenize(text, /*add_bos=*/false);
  if (tokens.empty())
    return {};

  int32_t batch_cap = std::max<int32_t>(
      config_.batch_size, static_cast<int32_t>(tokens.size()) + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    BatchAddSeq(batch, {tokens[i], static_cast<llama_pos>(i),
                        /*seq_id=*/0,
                        /*logits=*/i == tokens.size() - 1});
  }

  if (llama_decode(embed_ctx_, batch) != 0) {
    log::Error("llama_backend", "Embed: llama_decode failed");
    llama_batch_free(batch);
    return {};
  }
  llama_batch_free(batch);

  float *emb = llama_get_embeddings_seq(embed_ctx_, 0);
  if (!emb) {
    log::Error("llama_backend",
               "Embed: llama_get_embeddings_seq returned null");
    // Reset context state for this sequence.
    llama_memory_seq_rm(llama_get_memory(embed_ctx_), 0, -1, -1);
    return {};
  }

  int n_embd = llama_model_n_embd(model_);
  std::vector<float> result(emb, emb + n_embd);

  // Clear KV state for the sequence so the context can be reused.
  llama_memory_seq_rm(llama_get_memory(embed_ctx_), 0, -1, -1);
  return result;
}

std::string LlamaCppBackend::Name() const { return "llama_cpu"; }

} // namespace inferflux
