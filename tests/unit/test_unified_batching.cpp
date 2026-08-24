#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/execution/batch_executor.h"
#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"
#include <catch2/catch_amalgamated.hpp>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace inferflux;

// Mock backend to intercept ExecuteUnifiedBatch calls.
class MockUnifiedBackend : public LlamaCppBackend {
public:
  struct CallInfo {
    std::vector<UnifiedBatchInput> inputs;
  };
  std::vector<CallInfo> calls;
  int token_capacity{512};

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    calls.push_back({inputs});
    std::vector<UnifiedBatchOutput> outputs;
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42; // Always return token 42
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    return outputs;
  }

  bool IsReady() const override { return true; }
  int TokenCount(const std::string &) const override { return 5; }
  int UnifiedBatchTokenCapacity() const override { return token_capacity; }
};

class MockAsyncUnifiedBackend : public LlamaCppBackend {
public:
  enum class EventType {
    kSubmitDecode,
    kSubmitPrefill,
    kSubmitAuto,
    kCollect,
  };

  struct SubmitCall {
    UnifiedBatchLane lane{UnifiedBatchLane::kAuto};
    std::vector<UnifiedBatchInput> inputs;
  };

  bool supports_async{true};
  int token_capacity{512};
  std::vector<SubmitCall> submit_calls;
  std::vector<EventType> events;
  UnifiedBatchHandle next_handle{1};
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      pending_outputs;

  bool SupportsAsyncUnifiedBatch() const override { return supports_async; }
  int UnifiedBatchTokenCapacity() const override { return token_capacity; }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    submit_calls.push_back({lane, inputs});
    if (lane == UnifiedBatchLane::kDecode) {
      events.push_back(EventType::kSubmitDecode);
    } else if (lane == UnifiedBatchLane::kPrefill) {
      events.push_back(EventType::kSubmitPrefill);
    } else {
      events.push_back(EventType::kSubmitAuto);
    }
    std::vector<UnifiedBatchOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42;
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    const auto handle = next_handle++;
    pending_outputs.emplace(handle, std::move(outputs));
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    events.push_back(EventType::kCollect);
    if (!outputs) {
      return false;
    }
    auto it = pending_outputs.find(handle);
    if (it == pending_outputs.end()) {
      return false;
    }
    *outputs = std::move(it->second);
    pending_outputs.erase(it);
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    std::vector<UnifiedBatchOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42;
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    return outputs;
  }

  bool IsReady() const override { return true; }
  int TokenCount(const std::string &) const override { return 5; }
};

class EmptyGenerateBackend : public LlamaCppBackend {
public:
  bool IsReady() const override { return true; }

  std::string
  Generate(const std::string &, int,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &,
           const std::function<bool()> &, int,
           std::vector<TokenLogprob> *,
           const std::vector<std::string> &) override {
    return {};
  }

  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }
};

class SequencedAsyncUnifiedBackend : public LlamaCppBackend {
public:
  struct SequencePlan {
    std::deque<UnifiedBatchOutput> outputs;
  };

  bool SupportsAsyncUnifiedBatch() const override { return true; }
  bool IsReady() const override { return true; }
  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }

  void SetSequencePlan(int sequence_id,
                       std::initializer_list<UnifiedBatchOutput> outputs) {
    plans_[sequence_id].outputs = std::deque<UnifiedBatchOutput>(outputs);
  }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane) override {
    std::vector<UnifiedBatchOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
      auto it = plans_.find(input.sequence_id);
      REQUIRE(it != plans_.end());
      REQUIRE_FALSE(it->second.outputs.empty());
      outputs.push_back(it->second.outputs.front());
      it->second.outputs.pop_front();
    }
    const auto handle = next_handle_++;
    pending_[handle] = std::move(outputs);
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    auto it = pending_.find(handle);
    if (it == pending_.end() || !outputs) {
      return false;
    }
    *outputs = std::move(it->second);
    pending_.erase(it);
    return true;
  }

private:
  std::unordered_map<int, SequencePlan> plans_;
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      pending_;
  UnifiedBatchHandle next_handle_{1};
};

int64_t ReadEmptyGenerationsTotal() {
  const std::string output = GlobalMetrics().RenderPrometheus();
  const std::string key =
      "inferflux_empty_generations_total{backend=\"cpu\"} ";
  auto pos = output.find(key);
  if (pos == std::string::npos) {
    return 0;
  }
  pos += key.size();
  auto line_end = output.find('\n', pos);
  const std::string value = output.substr(
      pos, line_end == std::string::npos ? std::string::npos : line_end - pos);
  try {
    return std::stoll(value);
  } catch (const std::exception &) {
    return 0;
  }
}

TEST_CASE("BatchExecutor: Unified Batching & Chunked Prefill",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      cache, [](int) {}, RadixPrefixCacheLimits{1024, 12});
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);

  auto mock_backend = std::make_shared<MockUnifiedBackend>();

  SECTION("Chunked Prefill for a large prompt") {
    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kPrefill;
    req.n_past = 0;      // Required for eligibility
    req.sequence_id = 0; // Required for eligibility
    // Large prompt: 1200 tokens (will take 3 chunks of 512, 512, 176)
    req.bpe_prompt_tokens.assign(1200, 1);
    req.max_tokens = 5;
    req.block_table = {1, 2, 3};

    RequestBatch batch;
    batch.requests = {&req};

    auto results = executor->ExecuteBatch(batch, {mock_backend});

    // kMaxPrefillChunkSize is 512.
    // 1200 tokens -> 512 (chunk 0), 512 (chunk 1), 176 (chunk 2 + first token
    // sampled). Then 4 more tokens to reach max_tokens=5. Total steps in while
    // loop: 3 (prefill) + 4 (remaining decode) = 7 steps.
    REQUIRE(mock_backend->calls.size() == 7);

    // First call: 512 tokens, request_logits=false
    REQUIRE(mock_backend->calls[0].inputs[0].tokens.size() == 512);
    REQUIRE_FALSE(mock_backend->calls[0].inputs[0].request_logits);

    // Third call: 176 tokens, request_logits=true (final chunk)
    REQUIRE(mock_backend->calls[2].inputs[0].tokens.size() == 176);
    REQUIRE(mock_backend->calls[2].inputs[0].request_logits);

    REQUIRE(req.phase == RequestPhase::kFinished);
    REQUIRE(req.total_completion_tokens == 5);
  }

  SECTION("Interleaved Prefill and Decode") {
    InferenceRequest req_prefill;
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 1;
    req_prefill.bpe_prompt_tokens = {1, 2, 3};
    req_prefill.max_tokens = 5;
    req_prefill.block_table = {4};

    InferenceRequest req_decode;
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 5;
    req_decode.sequence_id = 2;
    req_decode.block_table = {5};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {mock_backend, mock_backend});

    // Step 1: prefill (chunk 0), decode (step 0) -> 2 inputs
    // Step 2: prefill-now-decode (step 1), decode (step 1) -> 2 inputs
    REQUIRE(mock_backend->calls.size() >= 2);
    REQUIRE(mock_backend->calls[0].inputs.size() == 2);
    REQUIRE(mock_backend->calls[1].inputs.size() == 2);
  }

  SECTION("Adaptive prefill chunking respects backend token capacity") {
    auto bounded_backend = std::make_shared<MockUnifiedBackend>();
    bounded_backend->token_capacity = 8;

    InferenceRequest req_a;
    req_a.model = "mock";
    req_a.phase = RequestPhase::kPrefill;
    req_a.n_past = 0;
    req_a.sequence_id = 21;
    req_a.bpe_prompt_tokens.assign(10, 1);
    req_a.max_tokens = 1;
    req_a.block_table = {21};

    InferenceRequest req_b;
    req_b.model = "mock";
    req_b.phase = RequestPhase::kPrefill;
    req_b.n_past = 0;
    req_b.sequence_id = 22;
    req_b.bpe_prompt_tokens.assign(10, 2);
    req_b.max_tokens = 1;
    req_b.block_table = {22};

    RequestBatch batch;
    batch.requests = {&req_a, &req_b};

    executor->ExecuteBatch(batch, {bounded_backend, bounded_backend});

    REQUIRE_FALSE(bounded_backend->calls.empty());
    REQUIRE(bounded_backend->calls.front().inputs.size() == 2);
    REQUIRE(bounded_backend->calls.front().inputs[0].tokens.size() == 4);
    REQUIRE(bounded_backend->calls.front().inputs[1].tokens.size() == 4);
    for (const auto &call : bounded_backend->calls) {
      std::size_t total_tokens = 0;
      for (const auto &input : call.inputs) {
        total_tokens += input.tokens.size();
      }
      REQUIRE(total_tokens <=
              static_cast<std::size_t>(bounded_backend->token_capacity));
    }
    REQUIRE(req_a.phase == RequestPhase::kFinished);
    REQUIRE(req_b.phase == RequestPhase::kFinished);
  }

  SECTION("Executor tuning caps prefill chunk size") {
    auto bounded_backend = std::make_shared<MockUnifiedBackend>();
    bounded_backend->token_capacity = 128;

    BatchExecutor::UnifiedBatchTuning tuning;
    tuning.chunked_prefill_tokens = 16;
    auto tuned_executor = std::make_unique<BatchExecutor>(
        &tokenizer, device, cache, router, nullptr, tuning);

    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kPrefill;
    req.n_past = 0;
    req.sequence_id = 41;
    req.bpe_prompt_tokens.assign(40, 1);
    req.max_tokens = 1;
    req.remaining_decode_tokens = 1;
    req.block_table = {41};

    RequestBatch batch;
    batch.requests = {&req};

    auto results = tuned_executor->ExecuteBatch(batch, {bounded_backend});

    REQUIRE(results.size() == 1);
    REQUIRE(bounded_backend->calls.size() >= 3);
    REQUIRE(bounded_backend->calls[0].inputs[0].tokens.size() == 16);
    REQUIRE(bounded_backend->calls[1].inputs[0].tokens.size() == 16);
    REQUIRE(bounded_backend->calls[2].inputs[0].tokens.size() == 8);
  }

  SECTION("Executor tuning applies mixed prefill budget ratio") {
    auto bounded_backend = std::make_shared<MockUnifiedBackend>();
    bounded_backend->token_capacity = 10;

    BatchExecutor::UnifiedBatchTuning tuning;
    tuning.chunked_prefill_tokens = 16;
    tuning.mixed_prefill_budget_ratio = 0.5;
    auto tuned_executor = std::make_unique<BatchExecutor>(
        &tokenizer, device, cache, router, nullptr, tuning);

    InferenceRequest req_prefill_a;
    req_prefill_a.model = "mock";
    req_prefill_a.phase = RequestPhase::kPrefill;
    req_prefill_a.n_past = 0;
    req_prefill_a.sequence_id = 51;
    req_prefill_a.bpe_prompt_tokens.assign(10, 1);
    req_prefill_a.max_tokens = 1;
    req_prefill_a.remaining_decode_tokens = 1;
    req_prefill_a.block_table = {51};

    InferenceRequest req_prefill_b;
    req_prefill_b.model = "mock";
    req_prefill_b.phase = RequestPhase::kPrefill;
    req_prefill_b.n_past = 0;
    req_prefill_b.sequence_id = 52;
    req_prefill_b.bpe_prompt_tokens.assign(10, 2);
    req_prefill_b.max_tokens = 1;
    req_prefill_b.remaining_decode_tokens = 1;
    req_prefill_b.block_table = {52};

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 7;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 2;
    req_decode.remaining_decode_tokens = 2;
    req_decode.sequence_id = 53;
    req_decode.block_table = {53};

    RequestBatch batch;
    batch.requests = {&req_prefill_a, &req_prefill_b, &req_decode};

    auto results = tuned_executor->ExecuteBatch(
        batch, {bounded_backend, bounded_backend, bounded_backend});

    REQUIRE(results.size() == 3);
    REQUIRE_FALSE(bounded_backend->calls.empty());
    REQUIRE(bounded_backend->calls.front().inputs.size() == 3);
    REQUIRE(bounded_backend->calls.front().inputs[0].tokens.size() == 2);
    REQUIRE(bounded_backend->calls.front().inputs[1].tokens.size() == 2);
    REQUIRE(bounded_backend->calls.front().inputs[2].tokens.size() == 1);
  }

  SECTION("Executor tuning enforces continuous decode-step slices") {
    auto bounded_backend = std::make_shared<MockUnifiedBackend>();

    BatchExecutor::UnifiedBatchTuning tuning;
    tuning.decode_burst_tokens = 2;
    auto tuned_executor = std::make_unique<BatchExecutor>(
        &tokenizer, device, cache, router, nullptr, tuning);

    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kPrefill;
    req.n_past = 0;
    req.sequence_id = 61;
    req.bpe_prompt_tokens = {1, 2, 3, 4};
    req.max_tokens = 5;
    req.remaining_decode_tokens = 5;
    req.block_table = {61};

    RequestBatch batch;
    batch.requests = {&req};

    auto results = tuned_executor->ExecuteBatch(batch, {bounded_backend});

    REQUIRE(results.size() == 1);
    REQUIRE(req.fairness_yielded);
    REQUIRE(req.phase == RequestPhase::kPending);
    REQUIRE(req.total_completion_tokens == 2);
    REQUIRE(req.remaining_decode_tokens == 3);
    REQUIRE(results[0].completion_tokens == 2);
  }

  SECTION("Deferred prefill resumes from non-zero prefill_offset") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 4;
    req_prefill.prefill_offset = 4;
    req_prefill.sequence_id = 6;
    req_prefill.bpe_prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {16};

    RequestBatch batch;
    batch.requests = {&req_prefill};

    auto results = executor->ExecuteBatch(batch, {async_backend});

    REQUIRE_FALSE(async_backend->submit_calls.empty());
    REQUIRE(async_backend->submit_calls.front().lane ==
            LlamaCppBackend::UnifiedBatchLane::kPrefill);
    REQUIRE(async_backend->submit_calls.front().inputs.size() == 1);
    REQUIRE(async_backend->submit_calls.front().inputs[0].n_past == 4);
    REQUIRE(async_backend->submit_calls.front().inputs[0].tokens.size() == 4);
    REQUIRE(req_prefill.phase == RequestPhase::kFinished);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].completion != "[batch state error]");
  }

  SECTION("Groups phased unified execution by backend instance") {
    auto mock_backend_a = std::make_shared<MockUnifiedBackend>();
    auto mock_backend_b = std::make_shared<MockUnifiedBackend>();

    InferenceRequest req_a;
    req_a.model = "model-a";
    req_a.phase = RequestPhase::kPrefill;
    req_a.n_past = 0;
    req_a.sequence_id = 3;
    req_a.bpe_prompt_tokens = {1, 2, 3};
    req_a.max_tokens = 2;
    req_a.block_table = {10};

    InferenceRequest req_b;
    req_b.model = "model-b";
    req_b.phase = RequestPhase::kPrefill;
    req_b.n_past = 0;
    req_b.sequence_id = 4;
    req_b.bpe_prompt_tokens = {4, 5, 6};
    req_b.max_tokens = 2;
    req_b.block_table = {11};

    RequestBatch batch;
    batch.requests = {&req_a, &req_b};

    executor->ExecuteBatch(batch, {mock_backend_a, mock_backend_b});

    REQUIRE_FALSE(mock_backend_a->calls.empty());
    REQUIRE_FALSE(mock_backend_b->calls.empty());
    REQUIRE(mock_backend_a->calls.front().inputs.size() == 1);
    REQUIRE(mock_backend_b->calls.front().inputs.size() == 1);
    REQUIRE(req_a.phase == RequestPhase::kFinished);
    REQUIRE(req_b.phase == RequestPhase::kFinished);
  }

  SECTION("Async lane split prefers decode lane in mixed-phase steps") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 7;
    req_prefill.bpe_prompt_tokens.assign(640, 1); // > one chunk and prefill
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {12};

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.sequence_id = 8;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 2;
    req_decode.block_table = {13};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {async_backend, async_backend});

    bool saw_decode_lane = false;
    bool saw_prefill_lane = false;
    for (const auto &call : async_backend->submit_calls) {
      if (call.lane == LlamaCppBackend::UnifiedBatchLane::kDecode) {
        saw_decode_lane = true;
      }
      if (call.lane == LlamaCppBackend::UnifiedBatchLane::kPrefill) {
        saw_prefill_lane = true;
      }
    }
    REQUIRE(saw_decode_lane);
    REQUIRE(saw_prefill_lane);
    REQUIRE(req_prefill.phase == RequestPhase::kFinished);
    REQUIRE(req_decode.phase == RequestPhase::kFinished);
  }

  SECTION("Async mixed-phase step submits both lanes before first collection") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 9;
    req_prefill.bpe_prompt_tokens.assign(640, 1);
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {14};

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.sequence_id = 10;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 2;
    req_decode.block_table = {15};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {async_backend, async_backend});

    REQUIRE(async_backend->events.size() >= 3);
    REQUIRE(async_backend->events[0] ==
            MockAsyncUnifiedBackend::EventType::kSubmitDecode);
    REQUIRE(async_backend->events[1] ==
            MockAsyncUnifiedBackend::EventType::kSubmitPrefill);
    REQUIRE(async_backend->events[2] ==
            MockAsyncUnifiedBackend::EventType::kCollect);
  }

  SECTION("ExecuteUnifiedBatchStep splits async submissions by lane") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.sequence_id = 11;
    req_prefill.n_past = 0;
    req_prefill.exec_active = true;
    req_prefill.exec_in_prefill = true;
    req_prefill.prefill_offset = 0;
    req_prefill.exec_decode_limit = 2;
    req_prefill.exec_tokens_generated = 0;
    req_prefill.bpe_prompt_tokens.assign(48, 1);

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.sequence_id = 12;
    req_decode.n_past = 8;
    req_decode.exec_active = true;
    req_decode.exec_in_prefill = false;
    req_decode.exec_decode_limit = 2;
    req_decode.exec_tokens_generated = 0;
    req_decode.exec_current_token = 42;

    RequestBatch step_batch;
    step_batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteUnifiedBatchStep(step_batch, async_backend);

    bool saw_decode_lane = false;
    bool saw_prefill_lane = false;
    bool saw_auto_lane = false;
    for (const auto &call : async_backend->submit_calls) {
      if (call.lane == LlamaCppBackend::UnifiedBatchLane::kDecode) {
        saw_decode_lane = true;
      } else if (call.lane == LlamaCppBackend::UnifiedBatchLane::kPrefill) {
        saw_prefill_lane = true;
      } else {
        saw_auto_lane = true;
      }
    }
    REQUIRE(saw_decode_lane);
    REQUIRE(saw_prefill_lane);
    REQUIRE_FALSE(saw_auto_lane);
  }

  SECTION(
      "ExecuteUnifiedBatchStep respects backend token capacity for prefill") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();
    async_backend->token_capacity = 6;

    InferenceRequest req_prefill_a;
    req_prefill_a.model = "mock";
    req_prefill_a.sequence_id = 30;
    req_prefill_a.n_past = 0;
    req_prefill_a.exec_active = true;
    req_prefill_a.exec_in_prefill = true;
    req_prefill_a.prefill_offset = 0;
    req_prefill_a.exec_decode_limit = 1;
    req_prefill_a.exec_tokens_generated = 0;
    req_prefill_a.bpe_prompt_tokens.assign(12, 1);

    InferenceRequest req_prefill_b;
    req_prefill_b.model = "mock";
    req_prefill_b.sequence_id = 31;
    req_prefill_b.n_past = 0;
    req_prefill_b.exec_active = true;
    req_prefill_b.exec_in_prefill = true;
    req_prefill_b.prefill_offset = 0;
    req_prefill_b.exec_decode_limit = 1;
    req_prefill_b.exec_tokens_generated = 0;
    req_prefill_b.bpe_prompt_tokens.assign(12, 2);

    RequestBatch step_batch;
    step_batch.requests = {&req_prefill_a, &req_prefill_b};

    executor->ExecuteUnifiedBatchStep(step_batch, async_backend);

    REQUIRE_FALSE(async_backend->submit_calls.empty());
    REQUIRE(async_backend->submit_calls.front().lane ==
            LlamaCppBackend::UnifiedBatchLane::kPrefill);
    REQUIRE(async_backend->submit_calls.front().inputs.size() == 2);
    REQUIRE(async_backend->submit_calls.front().inputs[0].tokens.size() == 3);
    REQUIRE(async_backend->submit_calls.front().inputs[1].tokens.size() == 3);
  }
}

TEST_CASE("BatchExecutor keeps async phased completions isolated per request",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);
  auto backend = std::make_shared<SequencedAsyncUnifiedBackend>();

  LlamaCppBackend::UnifiedBatchOutput a0;
  a0.token = 501;
  a0.piece = "A0";
  a0.ok = true;
  LlamaCppBackend::UnifiedBatchOutput a1;
  a1.token = 502;
  a1.piece = "A1";
  a1.ok = true;
  LlamaCppBackend::UnifiedBatchOutput b1;
  b1.token = 601;
  b1.piece = "B1";
  b1.ok = true;
  backend->SetSequencePlan(101, {a0, a1});
  backend->SetSequencePlan(202, {b1});

  InferenceRequest req_prefill;
  req_prefill.id = 1;
  req_prefill.model = "mock";
  req_prefill.phase = RequestPhase::kPrefill;
  req_prefill.n_past = 0;
  req_prefill.sequence_id = 101;
  req_prefill.bpe_prompt_tokens = {1, 2};
  req_prefill.max_tokens = 2;
  req_prefill.block_table = {101};

  InferenceRequest req_decode;
  req_decode.id = 2;
  req_decode.model = "mock";
  req_decode.phase = RequestPhase::kDecode;
  req_decode.n_past = 10;
  req_decode.sequence_id = 202;
  req_decode.first_token = 600;
  req_decode.first_piece = "B0";
  req_decode.max_tokens = 2;
  req_decode.block_table = {202};

  RequestBatch batch;
  batch.requests = {&req_prefill, &req_decode};

  auto results = executor->ExecuteBatch(batch, {backend, backend});

  REQUIRE(results.size() == 2);
  REQUIRE(results[0].completion == "A0A1");
  REQUIRE(results[1].completion == "B0B1");
  REQUIRE(req_prefill.accumulated_output == "A0A1");
  REQUIRE(req_decode.accumulated_output == "B0B1");
  REQUIRE(req_prefill.phase == RequestPhase::kFinished);
  REQUIRE(req_decode.phase == RequestPhase::kFinished);
}

TEST_CASE("BatchExecutor resumes fairness-sliced seeded decode without re-emitting"
          " first_piece",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  BatchExecutor::UnifiedBatchTuning tuning;
  tuning.decode_burst_tokens = 1;
  auto executor = std::make_unique<BatchExecutor>(
      &tokenizer, device, cache, router, nullptr, tuning);
  auto backend = std::make_shared<SequencedAsyncUnifiedBackend>();

  LlamaCppBackend::UnifiedBatchOutput b1;
  b1.token = 601;
  b1.piece = "B1";
  b1.ok = true;
  LlamaCppBackend::UnifiedBatchOutput b2;
  b2.token = 602;
  b2.piece = "B2";
  b2.ok = true;
  backend->SetSequencePlan(404, {b1, b2});

  InferenceRequest req;
  req.id = 4;
  req.model = "mock";
  req.phase = RequestPhase::kDecode;
  req.n_past = 10;
  req.sequence_id = 404;
  req.first_token = 600;
  req.first_piece = "B0";
  req.max_tokens = 3;
  req.remaining_decode_tokens = 3;
  req.block_table = {404};

  RequestBatch batch;
  batch.requests = {&req};

  auto first = executor->ExecuteBatch(batch, {backend});
  REQUIRE(first.size() == 1);
  REQUIRE(first[0].completion == "B0");
  REQUIRE(req.accumulated_output == "B0");
  REQUIRE(req.phase == RequestPhase::kPending);
  REQUIRE(req.remaining_decode_tokens == 2);

  auto second = executor->ExecuteBatch(batch, {backend});
  REQUIRE(second.size() == 1);
  REQUIRE(second[0].completion == "B1");
  REQUIRE(req.accumulated_output == "B0B1");
  REQUIRE(req.phase == RequestPhase::kPending);
  REQUIRE(req.remaining_decode_tokens == 1);

  auto third = executor->ExecuteBatch(batch, {backend});
  REQUIRE(third.size() == 1);
  REQUIRE(third[0].completion == "B2");
  REQUIRE(req.accumulated_output == "B0B1B2");
  REQUIRE(req.phase == RequestPhase::kFinished);
  REQUIRE(req.total_completion_tokens == 3);
}

TEST_CASE("BatchExecutor resumes fairness-sliced unified prefill decode with"
          " current token carryover",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  BatchExecutor::UnifiedBatchTuning tuning;
  tuning.decode_burst_tokens = 1;
  auto executor = std::make_unique<BatchExecutor>(
      &tokenizer, device, cache, router, nullptr, tuning);
  auto backend = std::make_shared<SequencedAsyncUnifiedBackend>();

  LlamaCppBackend::UnifiedBatchOutput a0;
  a0.token = 700;
  a0.piece = "A0";
  a0.ok = true;
  LlamaCppBackend::UnifiedBatchOutput a1;
  a1.token = 701;
  a1.piece = "A1";
  a1.ok = true;
  backend->SetSequencePlan(505, {a0, a1});

  InferenceRequest req;
  req.id = 5;
  req.model = "mock";
  req.phase = RequestPhase::kPrefill;
  req.n_past = 0;
  req.sequence_id = 505;
  req.bpe_prompt_tokens = {1, 2};
  req.max_tokens = 2;
  req.remaining_decode_tokens = 2;
  req.block_table = {505};

  RequestBatch batch;
  batch.requests = {&req};

  auto first = executor->ExecuteBatch(batch, {backend});
  REQUIRE(first.size() == 1);
  REQUIRE(first[0].completion == "A0");
  REQUIRE(req.accumulated_output == "A0");
  REQUIRE(req.phase == RequestPhase::kPending);
  REQUIRE(req.remaining_decode_tokens == 1);

  auto second = executor->ExecuteBatch(batch, {backend});
  REQUIRE(second.size() == 1);
  REQUIRE(second[0].completion == "A1");
  REQUIRE(req.accumulated_output == "A0A1");
  REQUIRE(req.phase == RequestPhase::kFinished);
  REQUIRE(req.total_completion_tokens == 2);
}

TEST_CASE("BatchExecutor keeps interleaved fairness-sliced phased requests"
          " isolated across resumes",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  BatchExecutor::UnifiedBatchTuning tuning;
  tuning.decode_burst_tokens = 1;
  auto executor = std::make_unique<BatchExecutor>(
      &tokenizer, device, cache, router, nullptr, tuning);
  auto backend = std::make_shared<SequencedAsyncUnifiedBackend>();

  LlamaCppBackend::UnifiedBatchOutput a0;
  a0.token = 800;
  a0.piece = "A0";
  a0.ok = true;
  LlamaCppBackend::UnifiedBatchOutput a1;
  a1.token = 801;
  a1.piece = "A1";
  a1.ok = true;
  LlamaCppBackend::UnifiedBatchOutput b1;
  b1.token = 901;
  b1.piece = "B1";
  b1.ok = true;
  LlamaCppBackend::UnifiedBatchOutput b2;
  b2.token = 902;
  b2.piece = "B2";
  b2.ok = true;
  backend->SetSequencePlan(606, {a0, a1});
  backend->SetSequencePlan(707, {b1, b2});

  InferenceRequest req_prefill;
  req_prefill.id = 6;
  req_prefill.model = "mock";
  req_prefill.phase = RequestPhase::kPrefill;
  req_prefill.n_past = 0;
  req_prefill.sequence_id = 606;
  req_prefill.bpe_prompt_tokens = {1, 2};
  req_prefill.max_tokens = 2;
  req_prefill.remaining_decode_tokens = 2;
  req_prefill.block_table = {606};

  InferenceRequest req_decode;
  req_decode.id = 7;
  req_decode.model = "mock";
  req_decode.phase = RequestPhase::kDecode;
  req_decode.n_past = 12;
  req_decode.sequence_id = 707;
  req_decode.first_token = 900;
  req_decode.first_piece = "B0";
  req_decode.max_tokens = 3;
  req_decode.remaining_decode_tokens = 3;
  req_decode.block_table = {707};

  RequestBatch batch;
  batch.requests = {&req_prefill, &req_decode};

  auto first = executor->ExecuteBatch(batch, {backend, backend});
  REQUIRE(first.size() == 2);
  REQUIRE(first[0].completion == "A0");
  REQUIRE(first[1].completion == "B0");
  REQUIRE(req_prefill.accumulated_output == "A0");
  REQUIRE(req_decode.accumulated_output == "B0");
  REQUIRE(req_prefill.phase == RequestPhase::kPending);
  REQUIRE(req_decode.phase == RequestPhase::kPending);

  auto second = executor->ExecuteBatch(batch, {backend, backend});
  REQUIRE(second.size() == 2);
  REQUIRE(second[0].completion == "A1");
  REQUIRE(second[1].completion == "B1");
  REQUIRE(req_prefill.accumulated_output == "A0A1");
  REQUIRE(req_decode.accumulated_output == "B0B1");
  REQUIRE(req_prefill.phase == RequestPhase::kFinished);
  REQUIRE(req_decode.phase == RequestPhase::kPending);

  auto third = executor->ExecuteBatch(batch, {backend, backend});
  REQUIRE(third.size() == 2);
  REQUIRE(third[1].completion == "B2");
  REQUIRE(req_prefill.accumulated_output == "A0A1");
  REQUIRE(req_decode.accumulated_output == "B0B1B2");
  REQUIRE(req_prefill.phase == RequestPhase::kFinished);
  REQUIRE(req_decode.phase == RequestPhase::kFinished);
}

TEST_CASE("BatchExecutor threads request and lease identity into unified batch"
          " inputs",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  BatchExecutor::UnifiedBatchTuning tuning;
  tuning.decode_burst_tokens = 1;
  auto executor = std::make_unique<BatchExecutor>(
      &tokenizer, device, cache, router, nullptr, tuning);
  auto backend = std::make_shared<MockAsyncUnifiedBackend>();

  InferenceRequest req;
  req.id = 77;
  req.model = "mock";
  req.phase = RequestPhase::kDecode;
  req.n_past = 5;
  req.sequence_id = 909;
  req.sequence_generation = 12;
  req.first_token = 808;
  req.max_tokens = 1;
  req.remaining_decode_tokens = 1;
  req.block_table = {909};

  RequestBatch batch;
  batch.requests = {&req};

  auto results = executor->ExecuteBatch(batch, {backend});
  REQUIRE(results.size() == 1);
  REQUIRE_FALSE(backend->submit_calls.empty());
  REQUIRE(backend->submit_calls.front().inputs.size() == 1);
  const auto &input = backend->submit_calls.front().inputs.front();
  REQUIRE(input.request_id == 77);
  REQUIRE(input.sequence_generation == 12);
  REQUIRE(input.sequence_id == 909);
}

TEST_CASE(
    "BatchExecutor ignores invisible control-token steps in async phased "
    "generation",
    "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);
  auto backend = std::make_shared<SequencedAsyncUnifiedBackend>();

  LlamaCppBackend::UnifiedBatchOutput a;
  a.token = 700;
  a.piece = "A";
  a.ok = true;
  LlamaCppBackend::UnifiedBatchOutput control;
  control.token = 151643;
  control.piece = "";
  control.ok = true;
  LlamaCppBackend::UnifiedBatchOutput b;
  b.token = 701;
  b.piece = "B";
  b.ok = true;
  backend->SetSequencePlan(303, {a, control, b});

  std::vector<std::string> streamed;
  InferenceRequest req;
  req.id = 3;
  req.model = "mock";
  req.phase = RequestPhase::kPrefill;
  req.n_past = 0;
  req.sequence_id = 303;
  req.bpe_prompt_tokens = {1, 2};
  req.max_tokens = 2;
  req.block_table = {303};
  req.on_token = [&](const std::string &piece, const TokenLogprob *) {
    streamed.push_back(piece);
  };

  RequestBatch batch;
  batch.requests = {&req};

  auto results = executor->ExecuteBatch(batch, {backend});

  REQUIRE(results.size() == 1);
  REQUIRE(results[0].completion == "AB");
  REQUIRE(results[0].completion_tokens == 2);
  REQUIRE(req.accumulated_output == "AB");
  REQUIRE(req.total_completion_tokens == 2);
  REQUIRE(req.phase == RequestPhase::kFinished);
  REQUIRE(streamed == std::vector<std::string>{"A", "B"});
}

TEST_CASE("BatchExecutor treats empty backend generation as zero-token sentinel",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);
  auto backend = std::make_shared<EmptyGenerateBackend>();

  GlobalMetrics().SetBackend("cpu");
  const int64_t empty_before = ReadEmptyGenerationsTotal();

  InferenceRequest req;
  req.model = "mock";
  req.prompt = "empty generation";
  req.max_tokens = 4;

  RequestBatch batch;
  batch.requests = {&req};

  auto results = executor->ExecuteBatch(batch, {backend});
  REQUIRE(results.size() == 1);
  REQUIRE_FALSE(results[0].no_backend);
  REQUIRE(results[0].completion == kBackendEmptyResponseText);
  REQUIRE(results[0].completion_tokens == 0);
  REQUIRE(req.accumulated_output.empty());
  REQUIRE(ReadEmptyGenerationsTotal() - empty_before == 1);
}
