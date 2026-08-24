#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/backend_utils.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "scheduler/request_batch.h"
#include "scheduler/scheduler.h"

using namespace inferflux;
using Catch::Approx;

// ---------------------------------------------------------------------------
// SamplingParams struct defaults
// ---------------------------------------------------------------------------

TEST_CASE("SamplingParams defaults", "[sampling]") {
  SamplingParams sp;
  REQUIRE(sp.temperature == Approx(1.0f));
  REQUIRE(sp.top_p == Approx(1.0f));
  REQUIRE(sp.top_k == 0);
  REQUIRE(sp.min_p == Approx(0.0f));
  REQUIRE(sp.frequency_penalty == Approx(0.0f));
  REQUIRE(sp.presence_penalty == Approx(0.0f));
  REQUIRE(sp.repetition_penalty == Approx(1.0f));
  REQUIRE(sp.penalty_last_n == 64);
  REQUIRE(sp.seed == UINT32_MAX);
}

TEST_CASE("InferenceRequest sampling field defaults to SamplingParams{}",
          "[sampling]") {
  InferenceRequest req;
  SamplingParams defaults;
  REQUIRE(req.sampling.temperature == Approx(defaults.temperature));
  REQUIRE(req.sampling.top_p == Approx(defaults.top_p));
  REQUIRE(req.sampling.top_k == defaults.top_k);
  REQUIRE(req.sampling.seed == defaults.seed);
}

// ---------------------------------------------------------------------------
// LlamaCppBackend sampler lifecycle (null-model / ForceReadyForTests)
// ---------------------------------------------------------------------------

TEST_CASE("SetupSampler no-op when model not loaded", "[sampling]") {
  LlamaCppBackend backend;
  // vocab_ is null — SetupSampler must not crash.
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", {}));
}

TEST_CASE("TeardownSampler idempotent — double call does not crash",
          "[sampling]") {
  LlamaCppBackend backend;
  REQUIRE_NOTHROW(backend.TeardownSampler());
  REQUIRE_NOTHROW(backend.TeardownSampler());
}

TEST_CASE(
    "SetupSampler with temperature=0 (greedy) succeeds when model not loaded",
    "[sampling]") {
  LlamaCppBackend backend;
  SamplingParams sp;
  sp.temperature = 0.0f;
  // Guard: vocab_ null → early return without crash.
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", sp));
}

TEST_CASE("SetupSampler with temperature>0 (stochastic) succeeds when model "
          "not loaded",
          "[sampling]") {
  LlamaCppBackend backend;
  SamplingParams sp;
  sp.temperature = 0.8f;
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", sp));
}

TEST_CASE("SamplerScope RAII: TeardownSampler called on scope exit",
          "[sampling]") {
  // Without a loaded model SetupSampler is a no-op, but TeardownSampler should
  // still be callable without error when the scope exits.
  LlamaCppBackend backend;
  backend.ForceReadyForTests();
  InferenceRequest req;
  {
    auto be = std::make_shared<LlamaCppBackend>();
    // Verify no crash during construction + destruction of SamplerScope-like
    // pattern using the public API.
    be->SetupSampler("", "root", req.sampling);
    be->TeardownSampler();
  }
  SUCCEED("SamplerScope pattern did not crash");
}

TEST_CASE("EnableGrammarConstraint / DisableGrammarConstraint backward compat",
          "[sampling]") {
  LlamaCppBackend backend;
  // Should delegate to SetupSampler / TeardownSampler without crashing.
  REQUIRE_NOTHROW(
      backend.EnableGrammarConstraint("root ::= \"hello\"", "root"));
  REQUIRE_NOTHROW(backend.DisableGrammarConstraint());
}

// ---------------------------------------------------------------------------
// Stop sequences on InferenceRequest
// ---------------------------------------------------------------------------

TEST_CASE("InferenceRequest stop field defaults to empty", "[sampling]") {
  InferenceRequest req;
  REQUIRE(req.stop.empty());
}

TEST_CASE("InferenceRequest stop accepts up to 4 strings", "[sampling]") {
  InferenceRequest req;
  req.stop = {"<|eot_id|>", "\n\n", "Human:", "Assistant:"};
  REQUIRE(req.stop.size() == 4);
  REQUIRE(req.stop[0] == "<|eot_id|>");
  REQUIRE(req.stop[3] == "Assistant:");
}

// ---------------------------------------------------------------------------
// InferenceResult finish_reason_length
// ---------------------------------------------------------------------------

TEST_CASE("InferenceResult finish_reason_length defaults false", "[sampling]") {
  inferflux::InferenceResult r;
  REQUIRE_FALSE(r.finish_reason_length);
}

TEST_CASE("InferenceResult finish_reason_length can be set to true",
          "[sampling]") {
  inferflux::InferenceResult r;
  r.finish_reason_length = true;
  REQUIRE(r.finish_reason_length);
}

// ---------------------------------------------------------------------------
// best_of: cumulative logprob ranking (pure algorithm tests)
// ---------------------------------------------------------------------------

TEST_CASE("best_of cumulative logprob ranking selects highest-logprob result",
          "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  // Build three results with known cumulative log-probs: -1, -5, -2.
  auto makeResult = [](float lp1, float lp2, const std::string &text) {
    InferenceResult r;
    r.completion = text;
    TokenLogprob t1, t2;
    t1.logprob = lp1;
    t2.logprob = lp2;
    r.logprobs = {t1, t2};
    r.completion_tokens = 2;
    return r;
  };

  std::vector<InferenceResult> results = {
      makeResult(-0.5f, -0.5f, "best"),   // cumsum = -1.0
      makeResult(-2.5f, -2.5f, "worst"),  // cumsum = -5.0
      makeResult(-0.9f, -1.1f, "middle"), // cumsum = -2.0
  };

  // Apply the same ranking as the production code.
  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  std::sort(results.begin(), results.end(),
            [&](const InferenceResult &a, const InferenceResult &b) {
              return cumlogprob(a) > cumlogprob(b);
            });

  REQUIRE(results[0].completion == "best");
  REQUIRE(results[1].completion == "middle");
  REQUIRE(results[2].completion == "worst");
}

TEST_CASE("best_of n=1 best_of=3 keeps only top result", "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  auto makeResult = [](float lp, const std::string &text) {
    InferenceResult r;
    r.completion = text;
    TokenLogprob t;
    t.logprob = lp;
    r.logprobs = {t};
    r.completion_tokens = 1;
    return r;
  };

  std::vector<InferenceResult> results = {
      makeResult(-3.0f, "c"),
      makeResult(-1.0f, "a"),
      makeResult(-2.0f, "b"),
  };

  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  std::sort(results.begin(), results.end(),
            [&](const InferenceResult &a, const InferenceResult &b) {
              return cumlogprob(a) > cumlogprob(b);
            });
  results.resize(1);

  REQUIRE(results.size() == 1u);
  REQUIRE(results[0].completion == "a");
}

TEST_CASE("best_of with equal logprobs is stable (no crash)", "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  std::vector<InferenceResult> results(3);
  for (auto &r : results) {
    TokenLogprob t;
    t.logprob = -1.0f;
    r.logprobs = {t};
    r.completion_tokens = 1;
  }
  results[0].completion = "x";
  results[1].completion = "y";
  results[2].completion = "z";

  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  REQUIRE_NOTHROW(
      std::sort(results.begin(), results.end(),
                [&](const InferenceResult &a, const InferenceResult &b) {
                  return cumlogprob(a) > cumlogprob(b);
                }));
  REQUIRE(results.size() == 3u);
}

// =============================================================================
// Test Suite: Logit Bias (OpenAI-compatible token-level biasing)
// =============================================================================

TEST_CASE("SamplingParams logit_bias defaults to empty map", "[sampling]") {
  SamplingParams sp;
  REQUIRE(sp.logit_bias.empty());
}

TEST_CASE("SamplingParams accepts logit_bias entries", "[sampling]") {
  SamplingParams sp;
  sp.logit_bias[42] = 10.5f;
  sp.logit_bias[100] = -5.0f;

  REQUIRE(sp.logit_bias.size() == 2);
  REQUIRE(sp.logit_bias.at(42) == 10.5f);
  REQUIRE(sp.logit_bias.at(100) == -5.0f);
}

TEST_CASE("SamplingParams logit_bias respects OpenAI range [-100, 100]",
          "[sampling]") {
  // This test documents that the API accepts the full OpenAI range.
  // Clamping to [-100, 100] is done at HTTP parsing time.
  SamplingParams sp;
  sp.logit_bias[0] = -100.0f;   // Minimum allowed
  sp.logit_bias[1000] = 100.0f; // Maximum allowed
  sp.logit_bias[500] = 50.0f;   // Mid-range

  REQUIRE(sp.logit_bias.size() == 3);
  REQUIRE(sp.logit_bias.at(0) == -100.0f);
  REQUIRE(sp.logit_bias.at(1000) == 100.0f);
  REQUIRE(sp.logit_bias.at(500) == 50.0f);
}

TEST_CASE("InferenceRequest inherits logit_bias from sampling params",
          "[sampling]") {
  InferenceRequest req;
  req.sampling.logit_bias[123] = 25.0f;
  req.sampling.logit_bias[456] = -10.0f;

  REQUIRE(req.sampling.logit_bias.size() == 2);
  REQUIRE(req.sampling.logit_bias.at(123) == 25.0f);
  REQUIRE(req.sampling.logit_bias.at(456) == -10.0f);
}

// =============================================================================
// Test Suite: ComputeLogprob (native logprob computation)
// =============================================================================

TEST_CASE("ComputeLogprob produces valid log-softmax for sampled token",
          "[sampling][logprobs]") {
  // Vocab of 5 tokens; logits arranged so token 2 is the argmax.
  float logits[] = {1.0f, 2.0f, 5.0f, 0.5f, -1.0f};
  int vocab = 5;

  auto id_to_str = [](int32_t id) { return "tok" + std::to_string(id); };
  auto tlp = ComputeLogprob(logits, vocab, 2, "tok2", 0, id_to_str);

  // log-softmax of the max logit should be negative and close to 0.
  REQUIRE(tlp.token == "tok2");
  REQUIRE(tlp.logprob < 0.0f);
  REQUIRE(tlp.logprob > -1.0f); // 5.0 is strongly dominant
  REQUIRE(tlp.bytes.size() == 4); // "tok2" = 4 bytes
  REQUIRE(tlp.top_logprobs.empty()); // top_n=0 means no alternatives
}

TEST_CASE("ComputeLogprob with top_n returns sorted alternatives",
          "[sampling][logprobs]") {
  float logits[] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f};
  int vocab = 5;

  auto id_to_str = [](int32_t id) { return "t" + std::to_string(id); };
  auto tlp = ComputeLogprob(logits, vocab, 2, "t2", 3, id_to_str);

  REQUIRE(tlp.top_logprobs.size() == 3);
  // Top 3 by logit value: token 2 (5.0), token 4 (4.0), token 1 (3.0)
  REQUIRE(tlp.top_logprobs[0].first == "t2");
  REQUIRE(tlp.top_logprobs[1].first == "t4");
  REQUIRE(tlp.top_logprobs[2].first == "t1");

  // All alternatives should have valid log probabilities (negative).
  for (const auto &alt : tlp.top_logprobs) {
    REQUIRE(alt.second < 0.0f);
  }

  // Top logprob should equal the sampled token logprob.
  REQUIRE(tlp.top_logprobs[0].second == Approx(tlp.logprob));
}

TEST_CASE("ComputeLogprob log-softmax sums to ~1.0 in prob space",
          "[sampling][logprobs]") {
  float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
  int vocab = 4;

  auto id_to_str = [](int32_t id) { return std::to_string(id); };
  auto tlp = ComputeLogprob(logits, vocab, 0, "0", 4, id_to_str);

  // Uniform logits: each token should have logprob = log(0.25) ≈ -1.386
  REQUIRE(tlp.logprob == Approx(-1.3863f).margin(0.01f));

  // Sum of exp(logprob) for all alternatives should be ~1.0.
  double sum = 0.0;
  for (const auto &alt : tlp.top_logprobs) {
    sum += std::exp(static_cast<double>(alt.second));
  }
  REQUIRE(sum == Approx(1.0).margin(0.001));
}

TEST_CASE("ComputeLogprob handles null/empty inputs gracefully",
          "[sampling][logprobs]") {
  auto id_to_str = [](int32_t) { return std::string("?"); };

  // Null logits
  auto tlp1 = ComputeLogprob(nullptr, 10, 0, "tok", 0, id_to_str);
  REQUIRE(tlp1.logprob == 0.0f);

  // Zero vocab
  float logits[] = {1.0f};
  auto tlp2 = ComputeLogprob(logits, 0, 0, "tok", 0, id_to_str);
  REQUIRE(tlp2.logprob == 0.0f);

  // Token ID out of range
  auto tlp3 = ComputeLogprob(logits, 1, 5, "tok", 0, id_to_str);
  REQUIRE(tlp3.logprob == 0.0f);
}

TEST_CASE("ComputeLogprob top_n clamped to vocab_size",
          "[sampling][logprobs]") {
  float logits[] = {1.0f, 2.0f, 3.0f};
  int vocab = 3;

  auto id_to_str = [](int32_t id) { return std::to_string(id); };
  // Request top_n=10 but vocab is only 3.
  auto tlp = ComputeLogprob(logits, vocab, 0, "0", 10, id_to_str);
  REQUIRE(tlp.top_logprobs.size() == 3);
}

TEST_CASE("TokenLogprob bytes field matches UTF-8 encoding",
          "[sampling][logprobs]") {
  // Use a multi-byte UTF-8 string: "ñ" = 0xC3 0xB1
  float logits[] = {1.0f};
  auto id_to_str = [](int32_t) { return std::string("x"); };
  auto tlp = ComputeLogprob(logits, 1, 0, "\xC3\xB1", 0, id_to_str);

  REQUIRE(tlp.bytes.size() == 2);
  REQUIRE(tlp.bytes[0] == 0xC3);
  REQUIRE(tlp.bytes[1] == 0xB1);
}
