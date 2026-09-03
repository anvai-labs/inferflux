#include "runtime/execution/batch_executor.h"
#include "runtime/backends/backend_utils.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using json = nlohmann::json;

namespace inferflux {

namespace {

// Stamps the first-token timestamp only if unset, so later blanket stamps
// (slice/step finalizers, Generate() return) never overwrite the genuine
// first emission recorded by the per-chunk callback or first decode step.
void MarkFirstTokenIfUnset(InferenceRequest &req) {
  if (req.first_token_time == std::chrono::steady_clock::time_point{}) {
    req.first_token_time = std::chrono::steady_clock::now();
  }
}

bool IsNonAccumulatingCompletion(std::string_view text) {
  return text == "[batch state error]" || IsBackendEmptyResponse(text);
}

bool IsVisibleGeneratedPiece(std::string_view piece) { return !piece.empty(); }

bool AppendGeneratedPiece(InferenceRequest *req, std::string_view raw_piece,
                          std::string *completion, MetricsRegistry *metrics) {
  if (!req || !completion || !metrics) {
    return false;
  }
  auto assembled = req->output_utf8.Append(raw_piece);
  metrics->RecordOutputUtf8Replacements(assembled.replacements);
  if (assembled.text.empty()) {
    return false;
  }

  completion->append(assembled.text);
  // First visible piece = the perceived first token; stamp before the
  // on_token emission so TTFT reflects when the client could first see it.
  MarkFirstTokenIfUnset(*req);
  std::string emit_piece;
  const bool stop_hit =
      ApplyStop(assembled.text, *completion, req->stop, &emit_piece);
  if (req->on_token && !emit_piece.empty()) {
    metrics->RecordStreamTokens(1);
    req->on_token(emit_piece, nullptr);
  }
  return stop_hit;
}

void FinishGeneratedText(InferenceRequest *req, std::string *completion,
                         MetricsRegistry *metrics) {
  if (!req || !completion || !metrics) {
    return;
  }
  auto assembled = req->output_utf8.Finish();
  metrics->RecordOutputUtf8Replacements(assembled.replacements);
  if (assembled.text.empty()) {
    return;
  }
  completion->append(assembled.text);
  MarkFirstTokenIfUnset(*req);
  if (req->on_token) {
    metrics->RecordStreamTokens(1);
    req->on_token(assembled.text, nullptr);
  }
}

int MaxNonEmittingSteps(int max_tokens) { return std::max(max_tokens * 8, 32); }

bool HasValidUnifiedPhasedState(const InferenceRequest &req) {
  if (req.n_past < 0 || req.sequence_id < 0) {
    return false;
  }
  if (req.first_token >= 0) {
    return true;
  }
  if (req.n_past == 0 && !req.bpe_prompt_tokens.empty()) {
    return true;
  }
  return req.n_past > 0 && !req.bpe_prompt_tokens.empty() &&
         req.prefill_offset > 0 &&
         req.prefill_offset < static_cast<int>(req.bpe_prompt_tokens.size());
}

bool UnifiedAssemblyDebugEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("INFERFLUX_DEBUG_UNIFIED_ASSEMBLY");
    return value && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "FALSE";
  }();
  return enabled;
}

int UnifiedAssemblyDebugLimit() {
  static const int limit = []() {
    const char *value = std::getenv("INFERFLUX_DEBUG_UNIFIED_ASSEMBLY_LIMIT");
    if (!value || *value == '\0') {
      return 200;
    }
    try {
      return std::max(1, std::stoi(value));
    } catch (const std::exception &) {
      return 200;
    }
  }();
  return limit;
}

bool ConsumeUnifiedAssemblyBudget() {
  static std::atomic<int> budget{UnifiedAssemblyDebugLimit()};
  int current = budget.load(std::memory_order_relaxed);
  while (current > 0) {
    if (budget.compare_exchange_weak(current, current - 1,
                                     std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

std::string DebugSnippet(std::string_view text, std::size_t max_len = 96) {
  if (text.size() <= max_len) {
    return std::string(text);
  }
  return std::string(text.substr(0, max_len)) + "...";
}

void LogUnifiedAssemblyState(std::string_view stage,
                             const InferenceRequest &req,
                             std::string_view piece,
                             std::string_view completion, int token,
                             int tokens_generated, int n_past, bool stop_hit,
                             bool active) {
  if (!UnifiedAssemblyDebugEnabled() || !ConsumeUnifiedAssemblyBudget()) {
    return;
  }
  log::Info("batch_executor",
            "unified_assembly[" + std::string(stage) +
                "]: request_id=" + std::to_string(req.id) +
                ", sequence_id=" + std::to_string(req.sequence_id) +
                ", n_past=" + std::to_string(n_past) +
                ", token=" + std::to_string(token) +
                ", generated=" + std::to_string(tokens_generated) +
                ", stop_hit=" + std::string(stop_hit ? "true" : "false") +
                ", active=" + std::string(active ? "true" : "false") +
                ", piece=" + json(DebugSnippet(piece)).dump() +
                ", completion=" + json(DebugSnippet(completion)).dump());
}

LlamaCppBackend::UnifiedBatchInput
MakeUnifiedBatchInput(const InferenceRequest &req, int n_past,
                      std::vector<int> tokens, bool request_logits) {
  LlamaCppBackend::UnifiedBatchInput input;
  input.sequence_id = req.sequence_id;
  input.n_past = n_past;
  input.tokens = std::move(tokens);
  input.request_logits = request_logits;
  input.sampling = req.sampling;
  input.request_id = static_cast<int64_t>(req.id);
  input.client_request_id = req.client_request_id;
  input.sequence_generation = req.sequence_generation;
  return input;
}

// Burst decode is eligible for a step when every request is past prefill and
// every input is a single-token greedy decode — the exact set the direct
// step path already routes here (grammar/logprobs/response-format requests
// are excluded upstream by the scheduler).
bool BurstStepEligible(
    const std::vector<InferenceRequest *> &active_reqs,
    const std::vector<LlamaCppBackend::UnifiedBatchInput> &batch_inputs) {
  if (batch_inputs.empty() || active_reqs.size() != batch_inputs.size()) {
    return false;
  }
  for (auto *req : active_reqs) {
    if (!req || req->execution.in_prefill) {
      return false;
    }
  }
  for (const auto &input : batch_inputs) {
    if (input.tokens.size() != 1 || !input.request_logits ||
        input.sampling.temperature > 0.0f) {
      return false;
    }
  }
  return true;
}

// RAII guard that calls SetupSampler before execution and TeardownSampler on
// scope exit.  Always sets up a sampler chain (grammar optional), so sampling
// params (temperature, top_p, …) are always in effect.
class SamplerScope {
public:
  SamplerScope(const InferenceRequest &req,
               const std::shared_ptr<LlamaCppBackend> &backend)
      : backend_(backend ? backend.get() : nullptr) {
    if (!backend_)
      return;
    const auto &c = req.response_format.constraint;
    backend_->SetupSampler(c.has_grammar ? c.grammar : "",
                           c.has_grammar ? c.root : "", req.sampling);
    active_ = true;
  }

  // Alternate constructor for unified batch phased execution where grammar and
  // sampling params are provided directly (not via InferenceRequest).
  SamplerScope(LlamaCppBackend *backend, const std::string &grammar,
               const std::string &root, const SamplingParams &sp)
      : backend_(backend) {
    if (!backend_)
      return;
    backend_->SetupSampler(grammar, root, sp);
    active_ = true;
  }

  SamplerScope(const SamplerScope &) = delete;
  SamplerScope &operator=(const SamplerScope &) = delete;

  ~SamplerScope() {
    if (active_ && backend_) {
      backend_->TeardownSampler();
    }
  }

private:
  LlamaCppBackend *backend_{nullptr};
  bool active_{false};
};

bool WaitForUnifiedBatchAsync(
    LlamaCppBackend *backend, LlamaCppBackend::UnifiedBatchHandle handle,
    std::vector<LlamaCppBackend::UnifiedBatchOutput> *outputs,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  if (!backend || !outputs || handle == 0) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (backend->TryCollectUnifiedBatchAsync(handle, outputs)) {
      return true;
    }
    std::this_thread::yield();
  }
  return backend->TryCollectUnifiedBatchAsync(handle, outputs);
}

BatchExecutor::UnifiedBatchTuning
NormalizeUnifiedBatchTuning(BatchExecutor::UnifiedBatchTuning raw) {
  raw.decode_burst_tokens = std::max(0, raw.decode_burst_tokens);
  raw.chunked_prefill_tokens = std::max(1, raw.chunked_prefill_tokens);
  if (!std::isfinite(raw.mixed_prefill_budget_ratio)) {
    raw.mixed_prefill_budget_ratio = 1.0;
  }
  raw.mixed_prefill_budget_ratio =
      std::clamp(raw.mixed_prefill_budget_ratio, 0.0, 1.0);
  return raw;
}

int ComputePrefillChunkCap(int backend_step_token_cap, int active_decode_reqs,
                           int active_prefill_reqs,
                           const BatchExecutor::UnifiedBatchTuning &tuning) {
  const int bounded_chunk_limit = std::max(
      1, std::min(tuning.chunked_prefill_tokens, backend_step_token_cap));
  const int scaled_prefill_budget = std::max(
      1,
      static_cast<int>(std::floor(static_cast<double>(backend_step_token_cap) *
                                  tuning.mixed_prefill_budget_ratio)));
  const int prefill_token_budget =
      std::max(1, std::min(bounded_chunk_limit, scaled_prefill_budget) -
                      active_decode_reqs);
  if (active_prefill_reqs <= 0) {
    return bounded_chunk_limit;
  }
  return std::max(1, std::min(bounded_chunk_limit,
                              prefill_token_budget / active_prefill_reqs));
}

} // namespace

BatchExecutor::BatchExecutor(
    SimpleTokenizer *tokenizer, std::shared_ptr<CPUDeviceContext> device,
    std::shared_ptr<PagedKVCache> cache, std::shared_ptr<ModelRouter> router,
    std::shared_ptr<SpeculativeDecoder> speculative_decoder,
    MetricsRegistry *metrics)
    : BatchExecutor(tokenizer, std::move(device), std::move(cache),
                    std::move(router), std::move(speculative_decoder),
                    UnifiedBatchTuning{}, metrics) {}

BatchExecutor::BatchExecutor(
    SimpleTokenizer *tokenizer, std::shared_ptr<CPUDeviceContext> device,
    std::shared_ptr<PagedKVCache> cache, std::shared_ptr<ModelRouter> router,
    std::shared_ptr<SpeculativeDecoder> speculative_decoder,
    UnifiedBatchTuning tuning, MetricsRegistry *metrics)
    : tokenizer_(tokenizer), device_(std::move(device)),
      cache_(std::move(cache)), router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      tuning_(NormalizeUnifiedBatchTuning(tuning)),
      metrics_(metrics ? *metrics : GlobalMetrics()) {}

std::vector<InferenceResult> BatchExecutor::ExecuteBatch(
    const RequestBatch &batch,
    const std::vector<std::shared_ptr<LlamaCppBackend>> &backend_overrides) {
  std::size_t n = batch.requests.size();
  std::vector<InferenceResult> results(n);
  std::vector<bool> handled(n, false);
  double total_prefill_ms = 0.0;
  double total_decode_ms = 0.0;

  // Identify requests eligible for multi-sequence batch decode and group them
  // by backend instance so mixed-model/multi-hardware batches still benefit
  // from unified phased execution per backend.
  // Eligibility: phased decode (n_past >= 0, seq_id >= 0), no grammar
  // constraints, no logprob collection, no response format.
  // Grammar uses per-backend grammar_sampler_ — not safe to interleave across
  // backends.
  struct EligibleGroup {
    std::shared_ptr<LlamaCppBackend> backend;
    std::vector<std::size_t> indices;
  };
  std::vector<EligibleGroup> eligible_groups;
  std::unordered_map<LlamaCppBackend *, std::size_t> group_index_by_backend;

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = batch.requests[i];
    auto be = (i < backend_overrides.size()) ? backend_overrides[i] : nullptr;
    if (!be) {
      be = ResolveBackend(req->model, nullptr);
    }
    if (HasValidUnifiedPhasedState(*req) &&
        !req->response_format.constraint.has_grammar &&
        !req->collect_logprobs && !req->response_format.has_format && be &&
        be->IsReady()) {
      auto *key = be.get();
      auto it = group_index_by_backend.find(key);
      if (it == group_index_by_backend.end()) {
        group_index_by_backend.emplace(key, eligible_groups.size());
        eligible_groups.push_back({be, {}});
        it = group_index_by_backend.find(key);
      }
      eligible_groups[it->second].indices.push_back(i);
    }
  }

  for (const auto &group : eligible_groups) {
    std::vector<InferenceRequest *> eligible_reqs;
    eligible_reqs.reserve(group.indices.size());
    for (auto idx : group.indices) {
      eligible_reqs.push_back(batch.requests[idx]);
    }
    // Use the unified execution path (§P1b) to interleave prefill and decode
    // tokens.
    auto outcomes = ExecuteUnifiedBatchPhased(eligible_reqs, group.backend);
    for (std::size_t j = 0; j < group.indices.size(); ++j) {
      results[group.indices[j]] = std::move(outcomes[j].result);
      total_decode_ms += outcomes[j].decode_ms;
      handled[group.indices[j]] = true;
    }
  }

  // Process remaining requests individually (non-eligible).
  for (std::size_t i = 0; i < n; ++i) {
    if (handled[i])
      continue;
    auto *request = batch.requests[i];
    std::shared_ptr<LlamaCppBackend> backend_override;
    if (i < backend_overrides.size()) {
      backend_override = backend_overrides[i];
    }
    auto outcome = ExecuteRequest(*request, backend_override);
    results[i] = std::move(outcome.result);
    total_prefill_ms += outcome.prefill_ms;
    total_decode_ms += outcome.decode_ms;
  }

  if (total_prefill_ms > 0) {
    metrics_.RecordPrefillDuration(total_prefill_ms);
  }
  if (total_decode_ms > 0) {
    metrics_.RecordDecodeDuration(total_decode_ms);
  }
  return results;
}

BatchExecutor::ExecutionOutcome BatchExecutor::ExecuteRequest(
    InferenceRequest &inference,
    std::shared_ptr<LlamaCppBackend> backend_override) {
  ExecutionOutcome outcome;
  auto &response = outcome.result;
  response.prompt_tokens = static_cast<int>(inference.prompt_tokens.size());
  inference.fairness.yielded = false;

  if (inference.cancellation_flag && inference.cancellation_flag->load()) {
    inference.phase = RequestPhase::kAborted;
    MarkFirstTokenIfUnset(inference);
    response.no_backend = true;
    response.completion = "[cancelled]";
    return outcome;
  }

  std::string resolved_model = inference.resolved_model;
  auto backend = std::move(backend_override);
  if (!backend) {
    backend = ResolveBackend(inference.model, &resolved_model);
    inference.resolved_model = resolved_model;
  }
  if (resolved_model.empty()) {
    resolved_model = inference.model;
  }
  response.model_id = resolved_model;

  bool backend_ready = backend && backend->IsReady();
  bool backend_empty_generation = false;
  int slice_limit = inference.fairness.timeslice_tokens;
  inference.fairness.last_timeslice_tokens = slice_limit;
  inference.fairness.timeslice_tokens =
      0; // consumed; refreshed on next fairness pass.
  int decode_limit = inference.max_tokens;
  if (inference.fairness.remaining_decode_tokens >= 0) {
    decode_limit =
        std::min(decode_limit, inference.fairness.remaining_decode_tokens);
  }
  if (slice_limit > 0) {
    decode_limit = std::min(decode_limit, slice_limit);
  }
  if (tuning_.decode_burst_tokens > 0) {
    decode_limit = std::min(decode_limit, tuning_.decode_burst_tokens);
    if (slice_limit <= 0) {
      // Reuse fairness-yield semantics for bounded decode bursts.
      slice_limit = tuning_.decode_burst_tokens;
    }
  }
  if (decode_limit <= 0) {
    decode_limit = 1;
  }

  // Prefill phase: prompt token counting (mirrors what the backend will
  // evaluate).
  {
    ExecutionTimer prefill_timer;
    if (backend_ready) {
      response.prompt_tokens = backend->TokenCount(inference.prompt);
    }
    if (inference.fairness.total_completion_tokens == 0) {
      inference.fairness.reported_prompt_tokens = response.prompt_tokens;
      if (inference.fairness.service_tokens == 0) {
        inference.fairness.service_tokens = response.prompt_tokens;
      }
    }
    outcome.prefill_ms = prefill_timer.ElapsedMs();
  }

  int kv_page = -1;
  if (cache_) {
    try {
      kv_page = cache_->ReservePage();
      if (kv_page >= 0) {
        inference.block_table = {kv_page};
      }
    } catch (const std::exception &ex) {
      log::Error("batch_executor",
                 std::string("KV cache reserve failed: ") + ex.what());
    }
  }

  // Decode phase: token generation (speculative or direct).
  inference.phase = RequestPhase::kDecode;
  ExecutionTimer decode_timer;

  SamplerScope sampler_scope(inference, backend);

  if (speculative_decoder_ && speculative_decoder_->Enabled() &&
      !inference.response_format.has_format &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    auto draft =
        speculative_decoder_->Draft(inference.prompt_tokens, decode_limit);
    auto validated = speculative_decoder_->Validate(
        inference.prompt_tokens, draft, decode_limit, backend);
    response.speculative.total_chunks =
        static_cast<int>(validated.metrics.total_chunks);
    response.speculative.accepted_chunks =
        static_cast<int>(validated.metrics.accepted_chunks);
    response.speculative.reused_tokens =
        static_cast<int>(validated.metrics.reused_tokens);
    response.completion = tokenizer_->Decode(validated.completion_tokens);
    inference.output_tokens = validated.completion_tokens;
  }

  if (response.completion.empty() &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    if (backend && backend->IsReady()) {
      std::function<bool(const std::string &, const TokenLogprob *)> chunk_cb;
      if (inference.on_token) {
        auto on_token = inference.on_token;
        auto cancel_flag = inference.cancellation_flag;
        auto *metrics_ptr = &metrics_;
        auto *req_ptr = &inference;
        chunk_cb = [on_token, cancel_flag, metrics_ptr, req_ptr](
                       const std::string &token_chunk, const TokenLogprob *lp) {
          // Stamp the true first-token time on the first emitted chunk. The
          // legacy Generate() path emits every chunk before returning, so a
          // return-time stamp would report TTFT ≈ total duration.
          MarkFirstTokenIfUnset(*req_ptr);
          metrics_ptr->RecordStreamTokens(1);
          on_token(token_chunk, lp);
          if (cancel_flag && cancel_flag->load()) {
            return false;
          }
          return true;
        };
      }
      std::function<bool()> should_stop;
      if (inference.cancellation_flag) {
        auto flag = inference.cancellation_flag;
        should_stop = [flag]() { return flag->load(); };
      }
      std::string text;
      const int logprob_top_n = inference.logprob_top_n;
      // Use collect_logprobs (not logprob_top_n > 0) so that logprobs=true
      // with top_logprobs=0 still records the selected token's log-probability
      // without the O(V log V) partial-sort for alternatives.
      std::vector<TokenLogprob> *lp_out =
          inference.collect_logprobs ? &response.logprobs : nullptr;
      if (inference.n_past >= 0 && inference.sequence_id >= 0) {
        // Phased path: prompt was already prefilled by the scheduler; run
        // decode only.  Pass first_token so Decode() starts from the
        // pre-sampled token rather than re-sampling from a potentially stale
        // logit buffer.
        text =
            backend->Decode(inference.n_past, inference.sequence_id,
                            decode_limit, chunk_cb, should_stop, logprob_top_n,
                            lp_out, inference.first_token, inference.stop);
      } else if (inference.has_images && backend->SupportsVision()) {
        text = backend->GenerateWithImages(inference.prompt, inference.images,
                                           decode_limit, chunk_cb, should_stop,
                                           inference.stop);
      } else {
        text = backend->Generate(inference.prompt, decode_limit, chunk_cb,
                                 should_stop, logprob_top_n, lp_out,
                                 inference.stop);
      }
      // Record GGML-native perf counters (P1b).
      {
        auto perf = backend->TakePerf();
        if (perf.generated_tokens > 0) {
          metrics_.RecordLlamaPerf(perf.prefill_ms, perf.decode_ms,
                                   perf.prompt_tokens, perf.generated_tokens);
        }
      }
      if (text.empty()) {
        backend_empty_generation = true;
        response.completion = std::string(kBackendEmptyResponseText);
        response.completion_tokens = 0;
        inference.output_tokens.clear();
        metrics_.RecordEmptyGeneration();
      } else {
        response.completion = text;
      }
    } else {
      response.no_backend = true;
      response.completion =
          "No model backend is loaded. Set INFERFLUX_MODEL_PATH or configure "
          "model.path in server.yaml.";
    }
  }

  outcome.decode_ms = decode_timer.ElapsedMs();

  if (backend_ready && !backend_empty_generation) {
    response.completion_tokens = backend->TokenCount(response.completion);
  } else {
    if (IsBackendEmptyResponse(response.completion)) {
      response.completion_tokens = 0;
      inference.output_tokens.clear();
    } else {
      auto completion_tokens = tokenizer_->Encode(response.completion);
      response.completion_tokens = static_cast<int>(completion_tokens.size());
      inference.output_tokens = std::move(completion_tokens);
    }
  }
  // finish_reason="length" when the full max_tokens budget is exhausted.
  // inference.fairness.total_completion_tokens holds prior-slice counts; adding
  // this slice's count gives the running total.  Fairness mid-slice yields will
  // not trigger this because their partial total stays below max_tokens.
  if (!response.no_backend && inference.max_tokens > 0 &&
      inference.fairness.total_completion_tokens + response.completion_tokens >=
          inference.max_tokens) {
    response.finish_reason_length = true;
  }
  // Advance n_past so the next fairness slice continues from the right KV
  // position.
  if (inference.n_past >= 0 && response.completion_tokens > 0) {
    inference.n_past += response.completion_tokens;
  }
  bool first_slice = (inference.fairness.total_completion_tokens == 0);
  int fairness_delta = response.completion_tokens;
  if (first_slice) {
    int prompt_component = response.prompt_tokens;
    if (prompt_component <= 0 &&
        inference.fairness.reported_prompt_tokens >= 0) {
      prompt_component = inference.fairness.reported_prompt_tokens;
    }
    fairness_delta += prompt_component;
  }
  if (fairness_delta > 0) {
    metrics_.RecordFairnessTokens(inference.fairness.priority_level,
                                  fairness_delta);
  }
  inference.fairness.service_tokens += response.completion_tokens;
  inference.fairness.total_completion_tokens += response.completion_tokens;
  bool fairness_active = slice_limit > 0;
  bool exhausted_slice =
      fairness_active && response.completion_tokens >= decode_limit;
  const int remaining_after_slice =
      (inference.fairness.remaining_decode_tokens >= 0)
          ? std::max(0, inference.fairness.remaining_decode_tokens -
                            response.completion_tokens)
          : -1;
  if (fairness_active && exhausted_slice && remaining_after_slice > 0) {
    inference.fairness.yielded = true;
    metrics_.RecordFairnessYield(inference.fairness.priority_level,
                                 response.completion_tokens,
                                 remaining_after_slice);
    inference.phase = RequestPhase::kPending;
  } else {
    inference.phase = RequestPhase::kFinished;
  }

  bool needs_json_validation =
      inference.json_mode ||
      inference.response_format.constraint.require_json_object;
  if (needs_json_validation) {
    try {
      auto parsed = json::parse(response.completion);
      if (inference.response_format.constraint.require_json_object &&
          !parsed.is_object()) {
        throw std::runtime_error("model output was not a JSON object");
      }
      response.completion = parsed.dump();
    } catch (const std::exception &) {
      if (inference.response_format.constraint.require_json_object &&
          !response.no_backend) {
        response.no_backend = true;
        response.completion =
            "Model output violated response_format constraints";
        response.completion_tokens = 0;
        inference.output_tokens.clear();
        inference.phase = RequestPhase::kFinished;
        inference.fairness.yielded = false;
        if (cache_ && kv_page >= 0) {
          cache_->ReleaseBlocks(inference.block_table);
          kv_page = -1;
          inference.block_table.clear();
        }
        MarkFirstTokenIfUnset(inference);
        return outcome;
      }
      response.completion = json({{"output", response.completion}}).dump();
    }
    auto retokens = tokenizer_->Encode(response.completion);
    response.completion_tokens = static_cast<int>(retokens.size());
    inference.output_tokens = std::move(retokens);
  }

  // Accumulate after json_mode so ProcessBatch sees the final post-processed
  // text.
  if (!response.completion.empty() &&
      !IsNonAccumulatingCompletion(response.completion)) {
    inference.accumulated_output.append(response.completion);
    if (inference.fairness.remaining_decode_tokens >= 0) {
      inference.fairness.remaining_decode_tokens = remaining_after_slice;
    }
  }

  if (inference.output_tokens.empty() &&
      !IsNonAccumulatingCompletion(response.completion)) {
    inference.output_tokens = tokenizer_->Encode(response.completion);
  }
  if (cache_ && kv_page >= 0) {
    cache_->ReleaseBlocks(inference.block_table);
    inference.block_table.clear();
  }

  MarkFirstTokenIfUnset(inference);
  inference.fairness.service_tokens += response.completion_tokens;

  // KV memory release and slot bookkeeping (FreeSequence + FreeSeqSlot) are
  // intentionally handled by the scheduler (ProcessBatch / DecodeWorkerLoop)
  // so they always happen together.  The executor only manages the compute
  // path; leaving sequence_id set here allows the scheduler to act on it.

  return outcome;
}

std::vector<BatchExecutor::ExecutionOutcome>
BatchExecutor::ExecuteBatchDecodePhased(
    std::vector<InferenceRequest *> &eligible,
    std::shared_ptr<LlamaCppBackend> backend) {
  std::size_t n = eligible.size();
  std::vector<ExecutionOutcome> outcomes(n);

  // Per-request runtime state for the decode loop.
  struct ReqState {
    bool active{true};
    int tokens_generated{0};
    int non_emitting_steps{0};
    int max_non_emitting_steps{32};
    int decode_limit{0};
    int current_token{-1}; // token to feed in the next BatchDecodeStep
    bool slice_active{false};
  };
  std::vector<ReqState> states(n);

  ExecutionTimer decode_timer;
  std::size_t decode_step_loops = 0;

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    req->fairness.yielded = false;
    out.model_id =
        req->resolved_model.empty() ? req->model : req->resolved_model;
    out.prompt_tokens = static_cast<int>(req->prompt_tokens.size());

    // Compute per-request decode limit respecting fairness timeslice.
    int limit = req->max_tokens;
    if (req->fairness.remaining_decode_tokens >= 0) {
      limit = std::min(limit, req->fairness.remaining_decode_tokens);
    }
    int slice = req->fairness.timeslice_tokens;
    req->fairness.last_timeslice_tokens = slice;
    req->fairness.timeslice_tokens = 0;
    if (slice > 0) {
      limit = std::min(limit, slice);
    }
    if (tuning_.decode_burst_tokens > 0) {
      limit = std::min(limit, tuning_.decode_burst_tokens);
    }
    if (limit <= 0)
      limit = 1;
    states[i].decode_limit = limit;
    states[i].max_non_emitting_steps = MaxNonEmittingSteps(req->max_tokens);
    states[i].slice_active = (slice > 0) || (tuning_.decode_burst_tokens > 0);

    // Emit the first token (pre-sampled by Prefill while its logits were
    // fresh).
    if (req->first_token >= 0) {
      states[i].current_token = req->first_token;

      std::string piece = req->first_piece;
      if (IsVisibleGeneratedPiece(piece)) {
        states[i].tokens_generated = 1;
        const bool stop_hit =
            AppendGeneratedPiece(req, piece, &out.completion, &metrics_);

        if (stop_hit ||
            (req->cancellation_flag && req->cancellation_flag->load())) {
          states[i].active = false;
        }
      } else {
        states[i].non_emitting_steps++;
      }
    } else {
      // EOS at prefill time — nothing to generate.
      states[i].active = false;
      out.completion = std::string(kBackendEmptyResponseText);
      metrics_.RecordEmptyGeneration();
    }
  }

  // Multi-sequence decode loop: one llama_decode() per token step.
  while (true) {
    std::vector<LlamaCppBackend::BatchDecodeInput> batch_inputs;
    std::vector<std::size_t> active_idx;

    for (std::size_t i = 0; i < n; ++i) {
      if (!states[i].active)
        continue;
      auto *req = eligible[i];
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        states[i].active = false;
        continue;
      }
      if (states[i].tokens_generated >= states[i].decode_limit) {
        states[i].active = false;
        continue;
      }
      batch_inputs.push_back(
          {req->sequence_id, req->n_past, states[i].current_token});
      active_idx.push_back(i);
    }

    if (active_idx.empty())
      break;
    ++decode_step_loops;

    auto step = backend->BatchDecodeStep(batch_inputs);
    if (step.empty())
      break; // llama_decode failed

    for (std::size_t j = 0; j < active_idx.size(); ++j) {
      std::size_t i = active_idx[j];
      auto *req = eligible[i];
      req->n_past =
          batch_inputs[j].n_past; // updated in-place by BatchDecodeStep

      const auto &sr = step[j];
      if (sr.token < 0) {
        states[i].active = false; // EOS
        continue;
      }
      states[i].current_token = sr.token;
      if (IsVisibleGeneratedPiece(sr.piece)) {
        states[i].tokens_generated++;
        const bool stop_hit = AppendGeneratedPiece(
            req, sr.piece, &outcomes[i].result.completion, &metrics_);
        if (stop_hit ||
            (req->cancellation_flag && req->cancellation_flag->load())) {
          states[i].active = false;
        }
      } else if (++states[i].non_emitting_steps >=
                 states[i].max_non_emitting_steps) {
        states[i].active = false;
        log::Warn("batch_executor",
                  "phased decode stopped after too many non-emitting tokens");
      }
    }
  }

  double decode_ms = decode_timer.ElapsedMs();

  // Finalise per-request results.
  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    int gen = states[i].tokens_generated;
    out.completion_tokens = gen;
    outcomes[i].decode_ms = decode_ms / static_cast<double>(n);

    // Fairness token accounting.
    bool first_slice = (req->fairness.total_completion_tokens == 0);
    int fairness_delta = gen;
    if (first_slice) {
      int prompt_comp = out.prompt_tokens;
      if (prompt_comp <= 0 && req->fairness.reported_prompt_tokens >= 0) {
        prompt_comp = req->fairness.reported_prompt_tokens;
      }
      fairness_delta += prompt_comp;
    }
    if (fairness_delta > 0) {
      metrics_.RecordFairnessTokens(req->fairness.priority_level,
                                    fairness_delta);
    }
    req->fairness.service_tokens += gen;
    req->fairness.total_completion_tokens += gen;

    // Fairness yield decision.
    bool exhausted = states[i].slice_active && gen >= states[i].decode_limit;
    const int remaining_after_slice =
        (req->fairness.remaining_decode_tokens >= 0)
            ? std::max(0, req->fairness.remaining_decode_tokens - gen)
            : -1;
    if (states[i].slice_active && exhausted && remaining_after_slice > 0) {
      req->fairness.yielded = true;
      metrics_.RecordFairnessYield(req->fairness.priority_level, gen,
                                   remaining_after_slice);
      req->phase = RequestPhase::kPending;
      req->first_token = states[i].current_token;
    } else {
      req->phase = RequestPhase::kFinished;
      req->first_token = -1;
      FinishGeneratedText(req, &out.completion, &metrics_);
    }
    req->first_piece.clear();

    // Accumulate and track remaining decode budget.
    if (!out.completion.empty() &&
        !IsNonAccumulatingCompletion(out.completion)) {
      req->accumulated_output.append(out.completion);
    }
    if (req->fairness.remaining_decode_tokens >= 0) {
      req->fairness.remaining_decode_tokens = remaining_after_slice;
    }

    MarkFirstTokenIfUnset(*req);
    // sequence_id left intact — scheduler handles FreeSequence + FreeSeqSlot.
  }

  metrics_.RecordDecodeDuration(decode_ms);
  metrics_.RecordDecodeStepLoops("decode_phased", decode_step_loops);
  return outcomes;
}

std::vector<BatchExecutor::ExecutionOutcome>
BatchExecutor::ExecuteUnifiedBatchPhased(
    std::vector<InferenceRequest *> &eligible,
    std::shared_ptr<LlamaCppBackend> backend) {
  std::size_t n = eligible.size();
  std::vector<ExecutionOutcome> outcomes(n);

  struct ReqState {
    bool active{true};
    int tokens_generated{0};
    int non_emitting_steps{0};
    int max_non_emitting_steps{32};
    int decode_limit{0};
    int current_token{-1};
    bool slice_active{false};
    bool in_prefill{false};
  };
  std::vector<ReqState> states(n);

  ExecutionTimer exec_timer;
  std::size_t decode_step_loops = 0;

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    req->fairness.yielded = false;
    out.model_id =
        req->resolved_model.empty() ? req->model : req->resolved_model;
    out.prompt_tokens = static_cast<int>(req->prompt_tokens.size());

    int limit = req->max_tokens;
    if (req->fairness.remaining_decode_tokens >= 0) {
      limit = std::min(limit, req->fairness.remaining_decode_tokens);
    }
    int slice = req->fairness.timeslice_tokens;
    req->fairness.last_timeslice_tokens = slice;
    req->fairness.timeslice_tokens = 0;
    if (slice > 0) {
      limit = std::min(limit, slice);
    }
    if (tuning_.decode_burst_tokens > 0) {
      limit = std::min(limit, tuning_.decode_burst_tokens);
    }
    if (limit <= 0)
      limit = 1;
    states[i].decode_limit = limit;
    states[i].max_non_emitting_steps = MaxNonEmittingSteps(req->max_tokens);
    states[i].slice_active = (slice > 0) || (tuning_.decode_burst_tokens > 0);

    // If n_past is 0 and bpe_prompt_tokens is not empty, this is a fresh
    // prefill. (Note: In the current scheduler, Prefill() is called BEFORE
    // ExecuteBatch, so req->n_past > 0 and req->first_token is already set. We
    // handle both cases for future flexibility where Prefill might be
    // interleaved).
    if (req->n_past >= 0 && req->first_token >= 0) {
      states[i].current_token = req->first_token;

      std::string piece = req->first_piece;
      bool stop_hit = false;
      if (IsVisibleGeneratedPiece(piece)) {
        states[i].tokens_generated = 1;
        stop_hit = AppendGeneratedPiece(req, piece, &out.completion, &metrics_);

        if (stop_hit ||
            (req->cancellation_flag && req->cancellation_flag->load())) {
          states[i].active = false;
        }
      } else {
        states[i].non_emitting_steps++;
      }
      LogUnifiedAssemblyState("seeded_decode", *req, piece, out.completion,
                              req->first_token, states[i].tokens_generated,
                              req->n_past, stop_hit, states[i].active);
    } else if (req->n_past == 0 && !req->bpe_prompt_tokens.empty()) {
      states[i].in_prefill = true;
    } else if (req->n_past > 0 && !req->bpe_prompt_tokens.empty() &&
               req->prefill_offset > 0 &&
               req->prefill_offset <
                   static_cast<int>(req->bpe_prompt_tokens.size())) {
      // Deferred scheduler prefill path: prefill_offset may already skip a
      // copied KV prefix. Continue unified prefill from that offset.
      req->n_past = req->prefill_offset;
      states[i].in_prefill = true;
    } else {
      states[i].active = false;
      out.completion = "[batch state error]";
    }
  }

  int loop_count = 0;

  // Pre-allocate batch input vectors outside the decode loop to avoid
  // per-token heap allocations (~1ms savings per token on tight loops).
  std::vector<LlamaCppBackend::UnifiedBatchInput> batch_inputs;
  std::vector<std::size_t> active_idx;
  std::vector<LlamaCppBackend::UnifiedBatchOutput> step;
  std::vector<LlamaCppBackend::UnifiedBatchInput> decode_inputs;
  std::vector<LlamaCppBackend::UnifiedBatchInput> prefill_inputs;
  std::vector<std::size_t> decode_positions;
  std::vector<std::size_t> prefill_positions;
  batch_inputs.reserve(n);
  active_idx.reserve(n);

  while (true) {
    if (++loop_count > 10000) {
      log::Error("batch_executor", "DEADLOCK GUARD: ExecuteUnifiedBatchPhased "
                                   "exceeded 10000 iterations");
      break;
    }

    batch_inputs.clear();
    active_idx.clear();

    int active_prefill_reqs = 0;
    int active_decode_reqs = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!states[i].active) {
        continue;
      }
      auto *req = eligible[i];
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        continue;
      }
      if (states[i].in_prefill) {
        active_prefill_reqs++;
      } else if (states[i].tokens_generated < states[i].decode_limit) {
        active_decode_reqs++;
      }
    }

    const int backend_step_token_cap =
        std::max(1, backend->UnifiedBatchTokenCapacity());
    const int prefill_chunk_cap =
        ComputePrefillChunkCap(backend_step_token_cap, active_decode_reqs,
                               active_prefill_reqs, tuning_);

    for (std::size_t i = 0; i < n; ++i) {
      if (!states[i].active)
        continue;
      auto *req = eligible[i];
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        states[i].active = false;
        continue;
      }

      if (states[i].in_prefill) {
        // Slice the next chunk of the prompt (§P1d).
        int remaining = static_cast<int>(req->bpe_prompt_tokens.size()) -
                        req->prefill_offset;
        int chunk_size = std::min(remaining, prefill_chunk_cap);
        if (remaining > chunk_size) {
          metrics_.RecordPrefillChunkTruncation(
              "unified_phased",
              static_cast<std::size_t>(remaining - chunk_size));
        }
        if (chunk_size <= 0) {
          states[i].in_prefill = false; // Safety
          continue;
        }
        bool is_last_chunk = (req->prefill_offset + chunk_size >=
                              static_cast<int>(req->bpe_prompt_tokens.size()));

        std::vector<int> chunk(
            req->bpe_prompt_tokens.begin() + req->prefill_offset,
            req->bpe_prompt_tokens.begin() + req->prefill_offset + chunk_size);

        // Only request logits on the final chunk of the prefill.
        batch_inputs.push_back(MakeUnifiedBatchInput(
            *req, req->prefill_offset, std::move(chunk), is_last_chunk));
        active_idx.push_back(i);
      } else if (states[i].tokens_generated < states[i].decode_limit) {
        batch_inputs.push_back(MakeUnifiedBatchInput(
            *req, req->n_past, {states[i].current_token}, true));
        active_idx.push_back(i);
      } else {
        states[i].active = false;
      }
    }

    if (active_idx.empty())
      break;
    ++decode_step_loops;

    step.clear();
    bool step_ok = true;
    if (backend->SupportsAsyncUnifiedBatch()) {
      decode_inputs.clear();
      prefill_inputs.clear();
      decode_positions.clear();
      prefill_positions.clear();

      for (std::size_t j = 0; j < batch_inputs.size(); ++j) {
        std::size_t req_idx = active_idx[j];
        if (states[req_idx].in_prefill) {
          prefill_positions.push_back(j);
          prefill_inputs.push_back(batch_inputs[j]);
        } else {
          decode_positions.push_back(j);
          decode_inputs.push_back(batch_inputs[j]);
        }
      }

      step.resize(batch_inputs.size());
      LlamaCppBackend::UnifiedBatchHandle decode_handle = 0;
      LlamaCppBackend::UnifiedBatchHandle prefill_handle = 0;

      if (!decode_inputs.empty()) {
        decode_handle = backend->SubmitUnifiedBatchAsync(
            decode_inputs, LlamaCppBackend::UnifiedBatchLane::kDecode);
        if (decode_handle == 0) {
          step_ok = false;
        }
      }
      if (step_ok && !prefill_inputs.empty()) {
        // Submit both lanes before waiting so async backends can overlap
        // decode/pre-fill execution in the same scheduler step.
        prefill_handle = backend->SubmitUnifiedBatchAsync(
            prefill_inputs, LlamaCppBackend::UnifiedBatchLane::kPrefill);
        if (prefill_handle == 0) {
          step_ok = false;
        }
      }

      if (decode_handle != 0) {
        std::vector<LlamaCppBackend::UnifiedBatchOutput> decode_outputs;
        if (!WaitForUnifiedBatchAsync(backend.get(), decode_handle,
                                      &decode_outputs) ||
            decode_outputs.size() != decode_positions.size()) {
          step_ok = false;
        } else {
          for (std::size_t i = 0; i < decode_positions.size(); ++i) {
            step[decode_positions[i]] = std::move(decode_outputs[i]);
          }
        }
      }

      if (prefill_handle != 0) {
        std::vector<LlamaCppBackend::UnifiedBatchOutput> prefill_outputs;
        if (!WaitForUnifiedBatchAsync(backend.get(), prefill_handle,
                                      &prefill_outputs) ||
            prefill_outputs.size() != prefill_positions.size()) {
          step_ok = false;
        } else {
          for (std::size_t i = 0; i < prefill_positions.size(); ++i) {
            step[prefill_positions[i]] = std::move(prefill_outputs[i]);
          }
        }
      }
    } else {
      step = backend->ExecuteUnifiedBatch(batch_inputs);
      step_ok = !step.empty();
    }

    if (!step_ok || step.empty()) {
      // Backend returned nothing for active inputs — fail them to avoid loop.
      for (auto idx : active_idx)
        states[idx].active = false;
      break;
    }

    for (std::size_t j = 0; j < active_idx.size(); ++j) {
      std::size_t i = active_idx[j];
      auto *req = eligible[i];
      const auto &res = step[j];

      if (states[i].in_prefill) {
        int chunk_processed = static_cast<int>(batch_inputs[j].tokens.size());
        // A failed INTERMEDIATE no-logits chunk leaves a KV hole — advancing
        // would decode over missing context with no error signal. Deactivate
        // instead (the final-chunk branch below handles the last chunk).
        if (!res.ok && req->prefill_offset + chunk_processed <
                           static_cast<int>(req->bpe_prompt_tokens.size())) {
          states[i].active = false;
          states[i].in_prefill = false;
          continue;
        }
        req->prefill_offset += chunk_processed;
        req->n_past = req->prefill_offset;

        if (req->prefill_offset >=
            static_cast<int>(req->bpe_prompt_tokens.size())) {
          states[i].in_prefill = false;
          // After final chunk, the 'res' contains the first generated token.
          if (!res.ok || res.token < 0) {
            states[i].active = false;
            continue;
          }
          states[i].current_token = res.token;
          std::string piece = res.piece;
          bool stop_hit = false;
          if (IsVisibleGeneratedPiece(piece)) {
            states[i].tokens_generated++;
            stop_hit = AppendGeneratedPiece(
                req, piece, &outcomes[i].result.completion, &metrics_);

            if (stop_hit) {
              states[i].active = false;
            }
          } else if (++states[i].non_emitting_steps >=
                     states[i].max_non_emitting_steps) {
            states[i].active = false;
            log::Warn(
                "batch_executor",
                "unified prefill stopped after too many non-emitting tokens");
          }
          LogUnifiedAssemblyState("prefill_emit", *req, piece,
                                  outcomes[i].result.completion, res.token,
                                  states[i].tokens_generated, req->n_past,
                                  stop_hit, states[i].active);
        }
      } else {
        // Decode step processing.
        req->n_past += 1;
        if (!res.ok || res.token < 0) {
          states[i].active = false;
          continue;
        }

        states[i].current_token = res.token;
        std::string piece = res.piece;
        bool stop_hit = false;
        if (IsVisibleGeneratedPiece(piece)) {
          states[i].tokens_generated++;
          stop_hit = AppendGeneratedPiece(
              req, piece, &outcomes[i].result.completion, &metrics_);
          if (req->cancellation_flag && req->cancellation_flag->load()) {
            states[i].active = false;
          }

          if (stop_hit) {
            states[i].active = false;
          }
        } else if (++states[i].non_emitting_steps >=
                   states[i].max_non_emitting_steps) {
          states[i].active = false;
          log::Warn(
              "batch_executor",
              "unified decode stopped after too many non-emitting tokens");
        }
        LogUnifiedAssemblyState("decode_emit", *req, piece,
                                outcomes[i].result.completion, res.token,
                                states[i].tokens_generated, req->n_past,
                                stop_hit, states[i].active);
      }
    }
  }

  double total_ms = exec_timer.ElapsedMs();

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    int gen = states[i].tokens_generated;
    if (gen == 0 && out.completion.empty()) {
      out.completion = std::string(kBackendEmptyResponseText);
      metrics_.RecordEmptyGeneration();
    }
    out.completion_tokens = gen;
    outcomes[i].decode_ms = total_ms / static_cast<double>(n);

    bool first_slice = (req->fairness.total_completion_tokens == 0);
    int fairness_delta = gen;
    if (first_slice) {
      int prompt_comp = out.prompt_tokens;
      if (prompt_comp <= 0 && req->fairness.reported_prompt_tokens >= 0) {
        prompt_comp = req->fairness.reported_prompt_tokens;
      }
      fairness_delta += prompt_comp;
    }
    if (fairness_delta > 0) {
      metrics_.RecordFairnessTokens(req->fairness.priority_level,
                                    fairness_delta);
    }
    req->fairness.service_tokens += gen;
    req->fairness.total_completion_tokens += gen;

    bool exhausted = states[i].slice_active && gen >= states[i].decode_limit;
    const int remaining_after_slice =
        (req->fairness.remaining_decode_tokens >= 0)
            ? std::max(0, req->fairness.remaining_decode_tokens - gen)
            : -1;
    if (states[i].slice_active && exhausted && remaining_after_slice > 0) {
      req->fairness.yielded = true;
      metrics_.RecordFairnessYield(req->fairness.priority_level, gen,
                                   remaining_after_slice);
      req->phase = RequestPhase::kPending;
      req->first_token = states[i].current_token;
    } else {
      req->phase = RequestPhase::kFinished;
      req->first_token = -1;
      FinishGeneratedText(req, &out.completion, &metrics_);
    }
    req->first_piece.clear();

    if (!out.completion.empty() &&
        !IsNonAccumulatingCompletion(out.completion)) {
      req->accumulated_output.append(out.completion);
    }
    if (req->fairness.remaining_decode_tokens >= 0) {
      req->fairness.remaining_decode_tokens = remaining_after_slice;
    }
    LogUnifiedAssemblyState("finalize", *req, std::string_view{},
                            out.completion, -1, gen, req->n_past, false,
                            states[i].active);
    MarkFirstTokenIfUnset(*req);
  }

  metrics_.RecordDecodeStepLoops("unified_phased", decode_step_loops);
  return outcomes;
}

std::shared_ptr<LlamaCppBackend>
BatchExecutor::ResolveBackend(const std::string &requested_model,
                              std::string *resolved_id) {
  if (!router_) {
    return nullptr;
  }
  auto *info = router_->Resolve(requested_model);
  if (!info) {
    return nullptr;
  }
  if (resolved_id) {
    *resolved_id = info->id;
  }
  return std::static_pointer_cast<LlamaCppBackend>(
      router_->GetBackend(info->id));
}

void BatchExecutor::ExecuteUnifiedBatchStep(
    RequestBatch &batch, std::shared_ptr<LlamaCppBackend> backend) {
  if (!backend || batch.empty())
    return;

  int active_prefill_reqs = 0;
  int active_decode_reqs = 0;
  for (auto *req : batch.requests) {
    if (!req->execution.active || req->phase == RequestPhase::kFinished ||
        req->phase == RequestPhase::kAborted) {
      continue;
    }
    if (req->cancellation_flag && req->cancellation_flag->load()) {
      continue;
    }
    if (req->execution.in_prefill) {
      active_prefill_reqs++;
    } else if (req->execution.tokens_generated < req->execution.decode_limit) {
      active_decode_reqs++;
    }
  }
  const int backend_step_token_cap =
      std::max(1, backend->UnifiedBatchTokenCapacity());
  const int prefill_chunk_cap = ComputePrefillChunkCap(
      backend_step_token_cap, active_decode_reqs, active_prefill_reqs, tuning_);

  std::vector<LlamaCppBackend::UnifiedBatchInput> batch_inputs;
  std::vector<InferenceRequest *> active_reqs;

  for (auto *req : batch.requests) {
    if (!req->execution.active || req->phase == RequestPhase::kFinished ||
        req->phase == RequestPhase::kAborted) {
      continue;
    }

    if (req->cancellation_flag && req->cancellation_flag->load()) {
      req->execution.active = false;
      req->phase = RequestPhase::kAborted;
      continue;
    }

    if (req->execution.in_prefill) {
      int remaining =
          static_cast<int>(req->bpe_prompt_tokens.size()) - req->prefill_offset;
      int chunk_size = std::min(remaining, prefill_chunk_cap);
      if (remaining > chunk_size) {
        metrics_.RecordPrefillChunkTruncation(
            "unified_step", static_cast<std::size_t>(remaining - chunk_size));
      }
      if (chunk_size <= 0) {
        req->execution.in_prefill = false;
        continue;
      }
      bool is_last_chunk = (req->prefill_offset + chunk_size >=
                            static_cast<int>(req->bpe_prompt_tokens.size()));
      std::vector<int> chunk(
          req->bpe_prompt_tokens.begin() + req->prefill_offset,
          req->bpe_prompt_tokens.begin() + req->prefill_offset + chunk_size);
      batch_inputs.push_back(MakeUnifiedBatchInput(
          *req, req->prefill_offset, std::move(chunk), is_last_chunk));
      active_reqs.push_back(req);
    } else if (req->execution.tokens_generated < req->execution.decode_limit) {
      batch_inputs.push_back(MakeUnifiedBatchInput(
          *req, req->n_past, {req->execution.current_token}, true));
      active_reqs.push_back(req);
    } else {
      req->execution.active = false;
      req->phase = RequestPhase::kFinished;
    }
  }

  if (active_reqs.empty()) {
    return;
  }
  metrics_.RecordDecodeStepLoops("unified_step", 1);
  const bool native_step_backend = backend->Name() == "inferflux_cuda";

  if (active_prefill_reqs == 0 && active_decode_reqs == 1) {
    InferenceRequest *burst_req = nullptr;
    for (auto *req : active_reqs) {
      if (!req->execution.in_prefill &&
          req->execution.tokens_generated < req->execution.decode_limit) {
        burst_req = req;
        break;
      }
    }
    if (burst_req) {
      std::vector<LlamaCppBackend::BurstDecodeOutput> burst_outputs;
      std::string burst_reason;
      const int remaining_tokens =
          std::max(1, burst_req->execution.decode_limit -
                          burst_req->execution.tokens_generated);
      if (backend->TryGreedyBurstDecodeTokens(
              burst_req->sequence_id, burst_req->n_past,
              burst_req->execution.current_token, burst_req->sampling,
              remaining_tokens, &burst_outputs, &burst_reason)) {
        metrics_.RecordDecodeWorkerExecutionPath(
            "direct_stepwise_native_burst");
        for (const auto &burst_out : burst_outputs) {
          burst_req->n_past += 1;
          burst_req->execution.current_token = burst_out.token;

          if (burst_out.terminal || burst_out.token < 0) {
            burst_req->execution.active = false;
            break;
          }

          bool stop_hit = false;
          if (IsVisibleGeneratedPiece(burst_out.piece)) {
            burst_req->execution.tokens_generated++;
            stop_hit = AppendGeneratedPiece(
                burst_req, burst_out.piece,
                &burst_req->execution.result.completion, &metrics_);
            if (burst_req->cancellation_flag &&
                burst_req->cancellation_flag->load()) {
              burst_req->execution.active = false;
            }
          } else if (++burst_req->execution.non_emitting_steps >=
                     burst_req->execution.max_non_emitting_steps) {
            burst_req->execution.active = false;
            log::Warn(
                "batch_executor",
                "unified burst stopped after too many non-emitting tokens");
          }
          if (stop_hit) {
            burst_req->execution.active = false;
          }
          if (burst_req->execution.tokens_generated >=
              burst_req->execution.decode_limit) {
            burst_req->execution.active = false;
          }
          if (!burst_req->execution.active) {
            break;
          }
        }
        return;
      }
      (void)burst_reason;
    }
  }

  if (native_step_backend) {
    for (auto *req : batch.requests) {
      if (!req || !req->execution.active || req->execution.in_prefill ||
          req->phase == RequestPhase::kFinished ||
          req->phase == RequestPhase::kAborted) {
        continue;
      }
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        continue;
      }
      if (req->execution.tokens_generated < req->execution.decode_limit) {
        metrics_.RecordInferfluxCudaBurstDecodeIneligible("decode",
                                                          "scheduler_stepwise");
      }
    }
  }

  std::vector<LlamaCppBackend::UnifiedBatchOutput> step_outputs;
  if (backend->SupportsAsyncUnifiedBatch()) {
    std::vector<LlamaCppBackend::UnifiedBatchInput> decode_inputs;
    std::vector<LlamaCppBackend::UnifiedBatchInput> prefill_inputs;
    std::vector<std::size_t> decode_positions;
    std::vector<std::size_t> prefill_positions;
    decode_inputs.reserve(batch_inputs.size());
    prefill_inputs.reserve(batch_inputs.size());
    decode_positions.reserve(batch_inputs.size());
    prefill_positions.reserve(batch_inputs.size());

    for (std::size_t j = 0; j < batch_inputs.size(); ++j) {
      if (active_reqs[j]->execution.in_prefill) {
        prefill_positions.push_back(j);
        prefill_inputs.push_back(batch_inputs[j]);
      } else {
        decode_positions.push_back(j);
        decode_inputs.push_back(batch_inputs[j]);
      }
    }

    step_outputs.resize(batch_inputs.size());
    bool step_ok = true;
    LlamaCppBackend::UnifiedBatchHandle decode_handle = 0;
    LlamaCppBackend::UnifiedBatchHandle prefill_handle = 0;

    if (!decode_inputs.empty()) {
      decode_handle = backend->SubmitUnifiedBatchAsync(
          decode_inputs, LlamaCppBackend::UnifiedBatchLane::kDecode);
      if (decode_handle == 0) {
        step_ok = false;
      }
    }
    if (step_ok && !prefill_inputs.empty()) {
      prefill_handle = backend->SubmitUnifiedBatchAsync(
          prefill_inputs, LlamaCppBackend::UnifiedBatchLane::kPrefill);
      if (prefill_handle == 0) {
        step_ok = false;
      }
    }

    if (decode_handle != 0) {
      std::vector<LlamaCppBackend::UnifiedBatchOutput> decode_outputs;
      if (!WaitForUnifiedBatchAsync(backend.get(), decode_handle,
                                    &decode_outputs) ||
          decode_outputs.size() != decode_positions.size()) {
        step_ok = false;
      } else {
        for (std::size_t i = 0; i < decode_positions.size(); ++i) {
          step_outputs[decode_positions[i]] = std::move(decode_outputs[i]);
        }
      }
    }

    if (prefill_handle != 0) {
      std::vector<LlamaCppBackend::UnifiedBatchOutput> prefill_outputs;
      if (!WaitForUnifiedBatchAsync(backend.get(), prefill_handle,
                                    &prefill_outputs) ||
          prefill_outputs.size() != prefill_positions.size()) {
        step_ok = false;
      } else {
        for (std::size_t i = 0; i < prefill_positions.size(); ++i) {
          step_outputs[prefill_positions[i]] = std::move(prefill_outputs[i]);
        }
      }
    }

    if (!step_ok) {
      step_outputs.clear();
    }
  } else if (backend->SupportsUnifiedBatchBurst() &&
             BurstStepEligible(active_reqs, batch_inputs)) {
    // Burst-pipelined decode: the backend runs several tokens per call and
    // streams them through the sink below (emission logic mirrors the
    // per-step loop). On burst failure fall through to the normal path.
    UnifiedBurstOptions burst_options;
    const BurstTokenSink sink = [&](const BurstTokenEvent &ev) -> bool {
      if (ev.input_idx >= active_reqs.size()) {
        return false;
      }
      auto *req = active_reqs[ev.input_idx];
      req->n_past += 1;
      req->execution.current_token = ev.token_id;
      const std::string piece(ev.piece);
      bool stop_hit = false;
      if (IsVisibleGeneratedPiece(piece)) {
        req->execution.tokens_generated++;
        stop_hit = AppendGeneratedPiece(
            req, piece, &req->execution.result.completion, &metrics_);
        if (req->cancellation_flag && req->cancellation_flag->load()) {
          req->execution.active = false;
          return false;
        }
      } else if (++req->execution.non_emitting_steps >=
                 req->execution.max_non_emitting_steps) {
        req->execution.active = false;
        log::Warn("batch_executor",
                  "unified batch burst stopped after too many non-emitting "
                  "tokens");
        return false;
      }
      if (stop_hit) {
        req->execution.active = false;
        return false;
      }
      if (req->execution.tokens_generated >= req->execution.decode_limit) {
        req->execution.active = false;
        return false;
      }
      LogUnifiedAssemblyState("burst_decode_emit", *req, piece,
                              req->execution.result.completion, ev.token_id,
                              req->execution.tokens_generated, req->n_past,
                              stop_hit, req->execution.active);
      return true;
    };
    const UnifiedBurstResult burst =
        backend->ExecuteUnifiedBatchBurst(batch_inputs, burst_options, sink);
    if (burst.ok) {
      // Post-burst bookkeeping: sequences the executor marked finished
      // (device EOS or sink stop) deactivate here; the sink already did
      // per-token emission, n_past, and execution.current_token.
      for (std::size_t j = 0; j < active_reqs.size(); ++j) {
        if (j < burst.finished.size() && burst.finished[j]) {
          active_reqs[j]->execution.active = false;
        }
      }
      return;
    }
    log::Warn("batch_executor",
              "Unified batch burst failed; falling back to per-step decode");
    step_outputs = backend->ExecuteUnifiedBatch(batch_inputs);
  } else {
    step_outputs = backend->ExecuteUnifiedBatch(batch_inputs);
  }
  if (step_outputs.empty()) {
    for (auto *req : active_reqs)
      req->execution.active = false;
    return;
  }

  for (std::size_t j = 0; j < active_reqs.size(); ++j) {
    auto *req = active_reqs[j];
    const auto &res = step_outputs[j];

    if (req->execution.in_prefill) {
      int chunk_processed = static_cast<int>(batch_inputs[j].tokens.size());
      if (!res.ok && req->prefill_offset + chunk_processed <
                         static_cast<int>(req->bpe_prompt_tokens.size())) {
        // Failed intermediate chunk: deactivate rather than decode over the
        // KV hole (see the phased-executor note above).
        req->execution.active = false;
        req->execution.in_prefill = false;
        continue;
      }
      req->prefill_offset += chunk_processed;
      if (req->prefill_offset >=
          static_cast<int>(req->bpe_prompt_tokens.size())) {
        req->execution.in_prefill = false;
        if (!res.ok || res.token < 0) {
          req->execution.active = false;
          continue;
        }
        req->execution.current_token = res.token;
        req->n_past = req->prefill_offset;

        std::string piece = res.piece;
        bool stop_hit = false;
        if (IsVisibleGeneratedPiece(piece)) {
          req->execution.tokens_generated++;
          stop_hit = AppendGeneratedPiece(
              req, piece, &req->execution.result.completion, &metrics_);
          if (stop_hit)
            req->execution.active = false;
        } else if (++req->execution.non_emitting_steps >=
                   req->execution.max_non_emitting_steps) {
          req->execution.active = false;
          log::Warn("batch_executor", "unified prefill step stopped after too "
                                      "many non-emitting tokens");
        }
        if (req->execution.tokens_generated >= req->execution.decode_limit) {
          req->execution.active = false;
        }
        LogUnifiedAssemblyState("step_prefill_emit", *req, piece,
                                req->execution.result.completion, res.token,
                                req->execution.tokens_generated, req->n_past,
                                stop_hit, req->execution.active);
      }
    } else {
      req->n_past += 1;
      if (!res.ok || res.token < 0) {
        req->execution.active = false;
        continue;
      }
      req->execution.current_token = res.token;

      std::string piece = res.piece;
      bool stop_hit = false;
      if (IsVisibleGeneratedPiece(piece)) {
        req->execution.tokens_generated++;
        stop_hit = AppendGeneratedPiece(
            req, piece, &req->execution.result.completion, &metrics_);
        if (req->cancellation_flag && req->cancellation_flag->load()) {
          req->execution.active = false;
        }
      } else if (++req->execution.non_emitting_steps >=
                 req->execution.max_non_emitting_steps) {
        req->execution.active = false;
        log::Warn(
            "batch_executor",
            "unified decode step stopped after too many non-emitting tokens");
      }
      if (stop_hit)
        req->execution.active = false;
      if (req->execution.tokens_generated >= req->execution.decode_limit) {
        req->execution.active = false;
      }
      LogUnifiedAssemblyState("step_decode_emit", *req, piece,
                              req->execution.result.completion, res.token,
                              req->execution.tokens_generated, req->n_past,
                              stop_hit, req->execution.active);
    }
  }
}

} // namespace inferflux
