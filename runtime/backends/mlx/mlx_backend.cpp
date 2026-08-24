#include "runtime/backends/mlx/mlx_backend.h"
#include "runtime/backends/backend_utils.h"
#include "server/logging/logger.h"

#include <chrono>
#include <filesystem>
#include <string>

namespace inferflux {

MlxBackend::MlxBackend() = default;

bool MlxBackend::LoadModel(const std::filesystem::path &model_path,
                           const LlamaBackendConfig &config) {
#if !INFERFLUX_HAS_MLX
  log::Error("mlx_backend",
             "backend requested but binary was built without ENABLE_MLX "
             "support. Rebuild with -DENABLE_MLX=ON.");
  return false;
#else
  // MLX-native path: a model directory containing config.json + *.safetensors.
  if (std::filesystem::is_directory(model_path)) {
    // Stage 1: parse config.json and catalogue all tensors from shard headers.
    descriptor_ = loader_.LoadDirectory(model_path);
    if (!descriptor_.valid) {
      log::Error("mlx_backend",
                 "Directory load failed: " + model_path.string());
      return false;
    }

    // Stage 2: materialise all weight tensors into MLX arrays on Metal.
    weight_store_ = loader_.LoadWeights(descriptor_);
    if (!weight_store_.ok) {
      log::Error("mlx_backend",
                 "Weight materialisation failed for " + model_path.string());
      return false;
    }

    // Initialise the execution engine and hand it the loaded weight store.
    if (!engine_.Initialize()) {
      log::Error("mlx_backend", "Engine initialisation failed");
      return false;
    }
    if (!engine_.LoadWeights(weight_store_, descriptor_.config)) {
      log::Error("mlx_backend", "Engine weight loading failed");
      return false;
    }
    // Stage 3: load the HuggingFace tokenizer from the same directory.
    if (!tokenizer_.Load(model_path)) {
      log::Error("mlx_backend", "Tokenizer load failed for " +
                                    model_path.string() +
                                    " (continuing without tokenizer)");
      // Not fatal — tokenizer is optional until Stage 4 inference.
    }

    engine_ready_ = true;

    log::Info("mlx_backend",
              "Model ready: " + descriptor_.config.model_type + " (" +
                  std::to_string(descriptor_.config.num_hidden_layers) +
                  " layers, " + std::to_string(weight_store_.count) +
                  " tensors). Stage 3 (kernel execution) is the next slice.");
    return true;
  }

  // GGUF file path: delegate to the llama.cpp backend.
  return LlamaCppBackend::LoadModel(model_path, config);
#endif
}

// ---------------------------------------------------------------------------
// LlamaCppBackend virtual overrides
// ---------------------------------------------------------------------------

bool MlxBackend::IsReady() const {
  return engine_ready_ || LlamaCppBackend::IsReady();
}

int MlxBackend::TokenCount(const std::string &text) const {
  if (tokenizer_.Loaded()) {
    auto enc = tokenizer_.Encode(text, /*add_bos=*/false);
    return enc.ok ? static_cast<int>(enc.ids.size()) : 0;
  }
  return LlamaCppBackend::TokenCount(text);
}

void MlxBackend::SetupSampler(const std::string &grammar,
                              const std::string &root,
                              const SamplingParams &sp) {
  if (engine_ready_) {
    mlx_sp_ = sp;
    mlx_grammar_ = grammar;
    return;
  }
  LlamaCppBackend::SetupSampler(grammar, root, sp);
}

void MlxBackend::TeardownSampler() {
  if (engine_ready_) {
    mlx_sp_ = {};
    mlx_grammar_.clear();
    return;
  }
  LlamaCppBackend::TeardownSampler();
}

LlamaCppBackend::PerfSnapshot MlxBackend::TakePerf() {
  if (engine_ready_) {
    auto snap = mlx_perf_;
    mlx_perf_ = {};
    return snap;
  }
  return LlamaCppBackend::TakePerf();
}

std::vector<int> MlxBackend::TokenizeForCache(const std::string &prompt) const {
  if (tokenizer_.Loaded()) {
    auto enc = tokenizer_.Encode(prompt, /*add_bos=*/true);
    if (enc.ok) {
      return std::vector<int>(enc.ids.begin(), enc.ids.end());
    }
  }
  return LlamaCppBackend::TokenizeForCache(prompt);
}

LlamaCppBackend::ChatTemplateResult MlxBackend::FormatChatMessages(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  // When the HF tokenizer has a chat template, apply it via llama.cpp's
  // Jinja2-lite engine (llama_chat_apply_template with an explicit tmpl
  // string and nullptr model works without a loaded llama model).
  if (engine_ready_ && tokenizer_.HasChatTemplate() && !messages.empty()) {
    const std::string &tmpl = tokenizer_.ChatTemplate();

    // Build llama_chat_message[] keeping content strings alive.
    std::vector<std::string> contents;
    contents.reserve(messages.size());
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto &[role, content] : messages) {
      contents.push_back(content);
      chat.push_back({role.c_str(), contents.back().c_str()});
    }

    int total_chars = 0;
    for (const auto &[r, c] : messages)
      total_chars += static_cast<int>(r.size() + c.size());

    std::string buf(static_cast<size_t>(total_chars * 2 + 512), '\0');
    int written =
        llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(),
                                  add_assistant_prefix, buf.data(), buf.size());
    if (written < 0) {
      // Buffer too small: retry with exact size.
      buf.resize(static_cast<size_t>(-written) + 1, '\0');
      written = llama_chat_apply_template(tmpl.c_str(), chat.data(),
                                          chat.size(), add_assistant_prefix,
                                          buf.data(), buf.size());
    }
    if (written > 0) {
      buf.resize(static_cast<size_t>(written));
      return {true, std::move(buf)};
    }
  }
  // Fall back to base class (requires a loaded llama model; returns
  // valid=false when no model is loaded).
  return LlamaCppBackend::FormatChatMessages(messages, add_assistant_prefix);
}

std::string MlxBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs,
    const std::vector<std::string> &stop_seqs) {
  // Not engine-ready: delegate to the llama.cpp base (GGUF path).
  if (!engine_ready_) {
    return LlamaCppBackend::Generate(prompt, max_tokens, on_chunk, should_stop,
                                     logprob_top_n, out_logprobs, stop_seqs);
  }
  if (!tokenizer_.Loaded()) {
    log::Error("mlx_backend", "Generate: tokenizer not loaded");
    return "";
  }

  auto enc = tokenizer_.Encode(prompt, /*add_bos=*/true);
  if (!enc.ok || enc.ids.empty())
    return "";

  engine_.Reset();
  const int32_t eos_id = tokenizer_.EosId();

  using Clock = std::chrono::steady_clock;
  auto t_prefill_start = Clock::now();

  // Prefill: evaluate all prompt tokens in one forward pass.
  int32_t next_tok = engine_.Step(enc.ids, mlx_sp_);

  auto t_prefill_end = Clock::now();

  if (next_tok < 0)
    return "";

  std::string output;
  auto t_decode_start = Clock::now();
  int decode_tokens = 0;

  for (int t = 0; t < max_tokens; ++t) {
    if (next_tok == eos_id)
      break;
    if (should_stop && should_stop())
      break;

    std::string piece = tokenizer_.Decode({next_tok}, /*skip_special=*/true);
    output += piece;

    // Stop sequence check using shared utility.
    std::string emit_piece;
    bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);

    // Decode step: generate next token (logprobs collected here for the
    // current next_tok, so they're available to pass to the chunk callback).
    int32_t upcoming_tok =
        engine_.Step({next_tok}, mlx_sp_, logprob_top_n, out_logprobs);

    // Emit streaming chunk (after stop-seq check so we never stream the stop).
    if (on_chunk && !emit_piece.empty()) {
      const TokenLogprob *lp_ptr = (out_logprobs && !out_logprobs->empty())
                                       ? &out_logprobs->back()
                                       : nullptr;
      if (!on_chunk(emit_piece, lp_ptr))
        break;
    }
    if (stop_hit)
      break;

    next_tok = upcoming_tok;
    if (next_tok < 0)
      break;
    ++decode_tokens;
  }

  auto t_decode_end = Clock::now();

  // Populate perf snapshot (consumed by TakePerf() → Prometheus).
  {
    auto prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end -
                                                                t_prefill_start)
                          .count();
    auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start)
            .count();
    mlx_perf_ = {prefill_ms, decode_ms, static_cast<int32_t>(enc.ids.size()),
                 static_cast<int32_t>(decode_tokens)};
  }

  return output;
}

// ---------------------------------------------------------------------------
// Phased prefill/decode overrides (INF-8)
// ---------------------------------------------------------------------------

LlamaCppBackend::PrefillResult MlxBackend::Prefill(const std::string &prompt,
                                                   int sequence_id) {
  if (!engine_ready_)
    return LlamaCppBackend::Prefill(prompt, sequence_id);
  if (!tokenizer_.Loaded())
    return {0, false, -1, ""};

  auto enc = tokenizer_.Encode(prompt, /*add_bos=*/true);
  if (!enc.ok || enc.ids.empty())
    return {0, false, -1, ""};

  // Allocate slot and evaluate all prompt tokens in one forward pass.
  engine_.AllocSlot(sequence_id);
  int32_t first_tok = engine_.StepSeq(sequence_id, enc.ids, mlx_sp_);
  int n_past = engine_.SeqNPast(sequence_id);

  if (first_tok < 0 || first_tok == tokenizer_.EosId())
    return {n_past, true, -1, ""};

  std::string first_piece =
      tokenizer_.Decode({first_tok}, /*skip_special=*/true);
  return {n_past, true, first_tok, first_piece};
}

LlamaCppBackend::PrefillResult
MlxBackend::PrefillPartial(const std::string &prompt, int sequence_id,
                           int n_past_start) {
  if (!engine_ready_)
    return LlamaCppBackend::PrefillPartial(prompt, sequence_id, n_past_start);
  if (!tokenizer_.Loaded())
    return {0, false, -1, ""};

  auto enc = tokenizer_.Encode(prompt, /*add_bos=*/true);
  if (!enc.ok || enc.ids.empty())
    return {0, false, -1, ""};

  const int total = static_cast<int>(enc.ids.size());
  // If the prefix already covers the full prompt, no suffix to evaluate.
  if (n_past_start >= total)
    return {total, true, -1, ""};

  // The slot should already have the prefix KV from CopySlotPrefix.
  // Just evaluate the suffix tokens starting at n_past_start.
  std::vector<int32_t> suffix(enc.ids.begin() + n_past_start, enc.ids.end());
  int32_t first_tok = engine_.StepSeq(sequence_id, suffix, mlx_sp_);
  int n_past = engine_.SeqNPast(sequence_id);

  if (first_tok < 0 || first_tok == tokenizer_.EosId())
    return {n_past, true, -1, ""};

  std::string first_piece =
      tokenizer_.Decode({first_tok}, /*skip_special=*/true);
  return {n_past, true, first_tok, first_piece};
}

std::string MlxBackend::Decode(
    int /*n_past*/, int sequence_id, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs, int first_token,
    const std::vector<std::string> &stop_seqs) {
  if (!engine_ready_)
    return LlamaCppBackend::Decode(0, sequence_id, max_tokens, on_chunk,
                                   should_stop, logprob_top_n, out_logprobs,
                                   first_token, stop_seqs);
  if (!tokenizer_.Loaded())
    return "";

  const int32_t eos_id = tokenizer_.EosId();
  std::string output;
  int32_t current_tok = first_token;

  // The first_token was pre-sampled by Prefill while logits were fresh.
  if (current_tok < 0 || current_tok == eos_id)
    return output;

  for (int step = 0; step < max_tokens; ++step) {
    if (should_stop && should_stop())
      break;

    std::string piece = tokenizer_.Decode({current_tok}, /*skip_special=*/true);

    std::string emit_piece;
    bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);

    // Feed current token to get next; logprobs collected here so the pointer
    // is valid when passed to on_chunk below.
    int32_t next_tok = engine_.StepSeq(sequence_id, {current_tok}, mlx_sp_,
                                       logprob_top_n, out_logprobs);

    if (on_chunk && !emit_piece.empty()) {
      const TokenLogprob *lp_ptr = (out_logprobs && !out_logprobs->empty())
                                       ? &out_logprobs->back()
                                       : nullptr;
      if (!on_chunk(emit_piece, lp_ptr))
        break;
    }
    if (stop_hit)
      break;

    current_tok = next_tok;
    if (current_tok < 0 || current_tok == eos_id)
      break;
  }

  return output;
}

void MlxBackend::FreeSequence(int sequence_id) {
  if (engine_ready_) {
    engine_.FreeSlot(sequence_id);
    return;
  }
  LlamaCppBackend::FreeSequence(sequence_id);
}

void MlxBackend::CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) {
  if (engine_ready_) {
    engine_.CopySlotPrefix(src_seq, dst_seq, n_tokens);
    return;
  }
  LlamaCppBackend::CopySequencePrefix(src_seq, dst_seq, n_tokens);
}

std::vector<LlamaCppBackend::UnifiedBatchOutput>
MlxBackend::ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) {
  if (!engine_ready_)
    return LlamaCppBackend::ExecuteUnifiedBatch(inputs);
  if (!tokenizer_.Loaded() || inputs.empty())
    return std::vector<UnifiedBatchOutput>(inputs.size());

  const int32_t eos_id = tokenizer_.EosId();
  std::vector<UnifiedBatchOutput> results(inputs.size());

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &inp = inputs[i];
    if (inp.tokens.empty()) {
      results[i].ok = true;
      continue;
    }

    // Run a forward pass for this sequence's slot.
    int32_t tok = engine_.StepSeq(inp.sequence_id, inp.tokens, mlx_sp_);

    if (!inp.request_logits) {
      results[i].ok = true;
      continue;
    }

    results[i].ok = true;
    if (tok < 0 || tok == eos_id) {
      results[i].token = -1;
    } else {
      results[i].token = tok;
      results[i].piece = tokenizer_.Decode({tok}, /*skip_special=*/true);
    }
  }
  return results;
}

MlxBackend::~MlxBackend() {
  if (engine_ready_) {
    engine_.Shutdown();
  }
}

} // namespace inferflux
