#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/device_context.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/model_router.h"
#include "scheduler/request_batch.h"
#include "scheduler/scheduler.h"

#include <memory>
#include <string>

namespace inferflux {

class BatchExecutor {
public:
  struct UnifiedBatchTuning {
    // 0 disables scheduler-level decode slicing in executor loops.
    int decode_burst_tokens{0};
    // Upper bound on per-request prefill chunk tokens in mixed execution.
    int chunked_prefill_tokens{512};
    // Fraction of backend token capacity reserved for prefill work in
    // mixed decode/prefill steps (0.0-1.0).
    double mixed_prefill_budget_ratio{1.0};
  };

  BatchExecutor(SimpleTokenizer *tokenizer,
                std::shared_ptr<CPUDeviceContext> device,
                std::shared_ptr<PagedKVCache> cache,
                std::shared_ptr<ModelRouter> router,
                std::shared_ptr<SpeculativeDecoder> speculative_decoder);
  BatchExecutor(SimpleTokenizer *tokenizer,
                std::shared_ptr<CPUDeviceContext> device,
                std::shared_ptr<PagedKVCache> cache,
                std::shared_ptr<ModelRouter> router,
                std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                UnifiedBatchTuning tuning);

  std::vector<InferenceResult> ExecuteBatch(
      const RequestBatch &batch,
      const std::vector<std::shared_ptr<LlamaCppBackend>> &backend_overrides);

  // Perform exactly one inference iteration (step) for the given batch.
  // This is the foundational block for step-wise batch execution.
  // Unlike ExecuteBatch, this does not loop internally; it processes exactly
  // one token per request and returns.  The caller (Scheduler) must loop and
  // can insert new requests between calls.
  void ExecuteUnifiedBatchStep(RequestBatch &batch,
                               std::shared_ptr<LlamaCppBackend> backend);

private:
  struct ExecutionOutcome {
    InferenceResult result;
    double prefill_ms{0};
    double decode_ms{0};
  };

  ExecutionOutcome
  ExecuteRequest(InferenceRequest &inference,
                 std::shared_ptr<LlamaCppBackend> backend_override);

  // Multi-sequence batch decode for eligible phased-decode requests that share
  // the same backend.  Runs one llama_decode() per token step covering all N
  // sequences simultaneously instead of N sequential single-sequence decodes.
  // Eligibility: n_past >= 0, sequence_id >= 0, no grammar, no logprobs.
  // Returns one ExecutionOutcome per request in the same order as `eligible`.
  std::vector<ExecutionOutcome>
  ExecuteBatchDecodePhased(std::vector<InferenceRequest *> &eligible,
                           std::shared_ptr<LlamaCppBackend> backend);

  // Unified batch execution (§P1b): interleaves prefill tokens and decode
  // tokens in a single ExecuteUnifiedBatch() call to maximize compute density.
  // Eligibility: same as ExecuteBatchDecodePhased.
  std::vector<ExecutionOutcome>
  ExecuteUnifiedBatchPhased(std::vector<InferenceRequest *> &eligible,
                            std::shared_ptr<LlamaCppBackend> backend);

  std::shared_ptr<LlamaCppBackend>
  ResolveBackend(const std::string &requested_model, std::string *resolved_id);

  SimpleTokenizer *tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<ModelRouter> router_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  UnifiedBatchTuning tuning_;
};

} // namespace inferflux
