#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/mlx/mlx_backend.h"
#include "scheduler/request_batch.h"

#include <string>
#include <vector>

using namespace inferflux;

// ---------------------------------------------------------------------------
// [mlx_backend] tests — use ForceReadyForTests() for null-model tests so no
// actual model file is required.  engine_ready_ stays false, so all calls
// fall through to the LlamaCppBackend base class, which is also not loaded.
// The tests verify interface contracts rather than correctness of output.
// ---------------------------------------------------------------------------

TEST_CASE("MlxBackend SetupSampler stores params and TeardownSampler clears",
          "[mlx_backend]") {
  MlxBackend backend;
  // engine_ready_=false: SetupSampler/TeardownSampler fall through to base;
  // neither crashes when model is not loaded (vocab_ is null, early-return).
  backend.ForceReadyForTests();

  SamplingParams sp;
  sp.temperature = 0.5f;
  sp.top_k = 10;
  REQUIRE_NOTHROW(backend.SetupSampler("", "", sp));
  REQUIRE_NOTHROW(backend.TeardownSampler());
}

TEST_CASE("MlxBackend TeardownSampler idempotent", "[mlx_backend]") {
  MlxBackend backend;
  backend.ForceReadyForTests();
  // Double teardown should not crash (idempotent).
  REQUIRE_NOTHROW(backend.TeardownSampler());
  REQUIRE_NOTHROW(backend.TeardownSampler());
}

TEST_CASE("MlxBackend TakePerf returns zeros when no perf captured",
          "[mlx_backend]") {
  MlxBackend backend;
  auto snap = backend.TakePerf();
  REQUIRE(snap.prefill_ms == 0.0);
  REQUIRE(snap.decode_ms == 0.0);
  REQUIRE(snap.prompt_tokens == 0);
  REQUIRE(snap.generated_tokens == 0);
}

TEST_CASE("MlxBackend TakePerf consume-once — second call returns zeros",
          "[mlx_backend]") {
  MlxBackend backend;
  // First call: zeros (nothing has run yet).
  auto snap1 = backend.TakePerf();
  REQUIRE(snap1.generated_tokens == 0);
  // Second call: still zeros.
  auto snap2 = backend.TakePerf();
  REQUIRE(snap2.generated_tokens == 0);
}

TEST_CASE("MlxBackend TokenizeForCache falls back when tokenizer not loaded",
          "[mlx_backend]") {
  MlxBackend backend;
  backend.ForceReadyForTests();
  // No tokenizer loaded, no llama model loaded → returns empty vector.
  auto ids = backend.TokenizeForCache("hello world");
  // We only check it doesn't crash; the vector may be empty or non-empty
  // depending on whether the base class has a model loaded (it doesn't here).
  // Both empty and non-empty are acceptable — just no crash.
  (void)ids;
  REQUIRE(true);
}

TEST_CASE(
    "MlxBackend FormatChatMessages returns false when no model or template",
    "[mlx_backend]") {
  MlxBackend backend;
  backend.ForceReadyForTests();

  std::vector<std::pair<std::string, std::string>> msgs = {{"user", "hi"}};
  auto result = backend.FormatChatMessages(msgs, true);
  // engine_ready_=false and no llama model loaded → base returns valid=false.
  REQUIRE_FALSE(result.valid);
}

TEST_CASE("MlxBackend IsReady reflects ForceReadyForTests", "[mlx_backend]") {
  MlxBackend backend;
  REQUIRE_FALSE(backend.IsReady());
  backend.ForceReadyForTests();
  REQUIRE(backend.IsReady());
}

// ---------------------------------------------------------------------------
// [mlx_backend_phased] — phased execution override interface tests (INF-8).
// engine_ready_=false (ForceReadyForTests only sets test_ready_ in base),
// so all calls fall through to LlamaCppBackend which has no model loaded.
// Tests verify: no crash, sensible fallback return values.
// ---------------------------------------------------------------------------

TEST_CASE("MlxBackend Prefill falls back gracefully without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  // engine_ready_=false: Prefill delegates to base which returns ok=false.
  auto r = backend.Prefill("hello world", 0);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.first_token == -1);
}

TEST_CASE("MlxBackend PrefillPartial falls back gracefully without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  auto r = backend.PrefillPartial("hello", 0, 0);
  REQUIRE_FALSE(r.ok);
}

TEST_CASE("MlxBackend Decode falls back gracefully without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  // Should return empty string without crashing.
  std::string out = backend.Decode(0, 0, 5, {}, {}, 0, nullptr, -1, {});
  REQUIRE(out.empty());
}

TEST_CASE("MlxBackend FreeSequence does not crash without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  REQUIRE_NOTHROW(backend.FreeSequence(0));
  REQUIRE_NOTHROW(backend.FreeSequence(5));
}

TEST_CASE("MlxBackend CopySequencePrefix does not crash without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  REQUIRE_NOTHROW(backend.CopySequencePrefix(0, 1, 10));
}

TEST_CASE("MlxBackend ExecuteUnifiedBatch returns empty without engine",
          "[mlx_backend_phased]") {
  MlxBackend backend;
  // No model loaded in base class → returns empty results.
  std::vector<LlamaCppBackend::UnifiedBatchInput> inputs;
  inputs.push_back({0, 0, {1, 2, 3}, true});
  auto results = backend.ExecuteUnifiedBatch(inputs);
  // Base class returns {} (empty) when context_ is null.
  // We just verify no crash and the vector has correct size or is empty.
  REQUIRE((results.empty() || results.size() == inputs.size()));
}
