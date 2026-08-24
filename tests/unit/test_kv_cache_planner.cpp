#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/kv_cache_planner.h"

namespace inferflux::runtime::cuda::native {
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

TEST_CASE("KV planner estimates bytes per token and cache bytes",
          "[kv_planner]") {
  const std::size_t per_token = EstimateKvBytesPerTokenPerSequence(
      /*num_layers=*/2, /*num_kv_heads=*/4, /*head_dim=*/8,
      /*kv_element_bytes=*/2);
  REQUIRE(per_token == 256);

  const std::size_t cache_bytes =
      EstimateKvCacheBytes(/*max_batch=*/16, /*max_seq=*/1024, per_token);
  REQUIRE(cache_bytes == static_cast<std::size_t>(16) * 1024 * 256);
}

TEST_CASE("KV planner clamps batch to scheduler minimum and seq to model max",
          "[kv_planner]") {
  KvCachePlanInput input;
  input.requested_max_batch = 4;
  input.requested_max_seq = 8192;
  input.min_max_batch = 16;
  input.min_max_seq = 128;
  input.model_max_position_embeddings = 4096;
  input.bytes_per_token_per_sequence = 65536;
  input.auto_tune_enabled = false;

  const KvCachePlanOutput out = PlanKvCache(input);
  REQUIRE(out.max_batch == 16);
  REQUIRE(out.max_seq == 4096);
  REQUIRE(out.auto_tuned_seq == false);
}

TEST_CASE("KV planner auto-tunes seq down to budget when not overridden",
          "[kv_planner]") {
  KvCachePlanInput input;
  input.requested_max_batch = 32;
  input.requested_max_seq = 4096;
  input.min_max_batch = 16;
  input.min_max_seq = 128;
  input.model_max_position_embeddings = 0;
  input.bytes_per_token_per_sequence = 65536; // 64 KiB/token/seq
  input.auto_tune_enabled = true;
  input.max_seq_overridden = false;
  input.explicit_budget_bytes = 2 * kGiB;

  const KvCachePlanOutput out = PlanKvCache(input);
  REQUIRE(out.auto_tuned_seq == true);
  REQUIRE(out.max_batch == 32);
  REQUIRE(out.max_seq == 1024);
  REQUIRE(out.planned_bytes <= out.budget_bytes);
}

TEST_CASE("KV planner skips auto-tune when max_seq is explicitly overridden",
          "[kv_planner]") {
  KvCachePlanInput input;
  input.requested_max_batch = 32;
  input.requested_max_seq = 4096;
  input.min_max_batch = 16;
  input.min_max_seq = 128;
  input.bytes_per_token_per_sequence = 65536;
  input.auto_tune_enabled = true;
  input.max_seq_overridden = true;
  input.explicit_budget_bytes = 2 * kGiB;

  const KvCachePlanOutput out = PlanKvCache(input);
  REQUIRE(out.auto_tuned_seq == false);
  REQUIRE(out.max_seq == 4096);
}

TEST_CASE("KV planner derives budget from free memory ratio when explicit "
          "budget is absent",
          "[kv_planner]") {
  KvCachePlanInput input;
  input.requested_max_batch = 16;
  input.requested_max_seq = 4096;
  input.min_max_batch = 16;
  input.min_max_seq = 128;
  input.bytes_per_token_per_sequence = 65536;
  input.auto_tune_enabled = true;
  input.max_seq_overridden = false;
  input.explicit_budget_bytes = 0;
  input.free_bytes = 8 * kGiB;
  input.budget_ratio = 0.25;

  const KvCachePlanOutput out = PlanKvCache(input);
  REQUIRE(out.budget_bytes == 2 * kGiB);
  REQUIRE(out.auto_tuned_seq == true);
  REQUIRE(out.max_seq == 2048);
  REQUIRE(out.planned_bytes == 2 * kGiB);
}

TEST_CASE("KV planner keeps plan unchanged when budget cannot be computed",
          "[kv_planner]") {
  KvCachePlanInput input;
  input.requested_max_batch = 16;
  input.requested_max_seq = 4096;
  input.min_max_batch = 16;
  input.min_max_seq = 128;
  input.bytes_per_token_per_sequence = 65536;
  input.auto_tune_enabled = true;
  input.max_seq_overridden = false;
  input.explicit_budget_bytes = 0;
  input.free_bytes = 0;
  input.budget_ratio = 0.30;

  const KvCachePlanOutput out = PlanKvCache(input);
  REQUIRE(out.budget_bytes == 0);
  REQUIRE(out.auto_tuned_seq == false);
  REQUIRE(out.max_seq == 4096);
  REQUIRE(out.planned_bytes == out.requested_bytes);
  REQUIRE(out.requested_bytes == 4 * kGiB);
}

} // namespace
} // namespace inferflux::runtime::cuda::native
