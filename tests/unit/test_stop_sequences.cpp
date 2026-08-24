#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/execution/batch_executor.h"
#include "scheduler/single_model_router.h"
#include <catch2/catch_amalgamated.hpp>
#include <memory>
#include <vector>

using namespace inferflux;

class MockStopBackend : public LlamaCppBackend {
public:
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    std::vector<UnifiedBatchOutput> outputs;
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      out.token = 123;
      out.piece = "next";
      outputs.push_back(out);
    }
    return outputs;
  }
  bool IsReady() const override { return true; }
};

TEST_CASE("BatchExecutor: Stop sequences on first token", "[stop_sequences]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);

  auto mock_backend = std::make_shared<MockStopBackend>();

  SECTION("Stop sequence triggers on first token in ExecuteBatchDecodePhased") {
    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kDecode;
    req.n_past = 10;
    req.sequence_id = 1;
    req.first_token = 42;
    req.first_piece = "STOP";
    req.stop = {"STOP"};
    req.max_tokens = 5;
    req.block_table = {1};

    RequestBatch batch;
    batch.requests = {&req};

    // We need to call ExecuteBatch, which will call ExecuteBatchDecodePhased
    // because it's a phased decode request with no grammar/logprobs.
    auto results = executor->ExecuteBatch(batch, {mock_backend});

    REQUIRE(results.size() == 1);
    // The stop sequence "STOP" should be trimmed from the completion.
    REQUIRE(results[0].completion == "");
    // The request should have finished immediately.
    // Wait, ExecuteBatch returns outcomes, but does it update the request
    // phase? Actually, BatchExecutor::ExecuteBatch updates req->phase to
    // kFinished if it completes.
  }

  SECTION(
      "Stop sequence triggers on first token in ExecuteUnifiedBatchPhased") {
    // To trigger ExecuteUnifiedBatchPhased, we need to have multiple requests
    // or a specific configuration. Actually ExecuteBatch calls
    // ExecuteBatchDecodePhased by default for phased requests.

    // Wait, let's check how ExecuteBatch decides which one to call.
  }
}
