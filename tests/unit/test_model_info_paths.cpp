#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "scheduler/model_router.h"
#include "scheduler/single_model_router.h"
#include "server/http/model_json.h"

#include <memory>

using namespace inferflux;

namespace {

std::shared_ptr<LlamaCppBackend> ReadyBackend() {
  auto backend = std::make_shared<LlamaCppBackend>();
  backend->ForceReadyForTests();
  return backend;
}

} // namespace

TEST_CASE("Model runtime diagnostics report backend capacity and mode policy",
          "[model_identity][cache_usage]") {
  class CapacityBackend : public LlamaCppBackend {
  public:
    bool IsReady() const override { return true; }
    int SequenceCapacity() const override { return 2; }
    int SequenceContextCapacity() const override { return 32768; }
  } backend;
  for (const auto *label : {"llama_cpp_rocm", "cpu", "inferflux_cuda"}) {
    ModelInfo info;
    info.backend = label;
    info.capabilities.supports_kv_prefix_transfer = true;
    info.gguf.context_length = 262144;
    auto runtime = BuildModelRuntimeJson(info, &backend, true, false);
    REQUIRE(runtime["sequence_capacity"] == 2);
    REQUIRE(runtime["context_tokens_per_sequence"] == 32768);
    REQUIRE(runtime["cache_reuse"]["phased"]["prefix"] == true);
    REQUIRE(runtime["cache_reuse"]["phased"]["session"] == false);
    REQUIRE(runtime["cache_reuse"]["full_generate"]["prefix"] == false);
    REQUIRE(runtime["cache_reuse"]["full_generate"]["session"] == false);
    runtime = BuildModelRuntimeJson(info, &backend, false, true);
    REQUIRE(runtime["cache_reuse"]["phased"]["prefix"] == false);
    REQUIRE(runtime["cache_reuse"]["phased"]["session"] == true);
    info.capabilities.supports_kv_prefix_transfer = false;
    runtime = BuildModelRuntimeJson(info, &backend, true, true);
    REQUIRE(runtime["cache_reuse"]["phased"]["prefix"] == false);
    REQUIRE(runtime["cache_reuse"]["phased"]["session"] == false);
  }
}

TEST_CASE("Model runtime diagnostics distinguish unknown from zero capacity",
          "[model_identity][cache_usage]") {
  ModelInfo info;
  info.capabilities.supports_kv_prefix_transfer = true;
  auto runtime = BuildModelRuntimeJson(info, nullptr, true, true);
  REQUIRE(runtime["sequence_capacity"].is_null());
  REQUIRE(runtime["context_tokens_per_sequence"].is_null());
  REQUIRE(runtime["cache_reuse"]["phased"]["prefix"] == false);
  REQUIRE(runtime["cache_reuse"]["phased"]["session"] == false);
  runtime = BuildModelRuntimeJson(info, ReadyBackend().get(), true, false);
  REQUIRE(runtime["sequence_capacity"].is_null());
  REQUIRE(runtime["context_tokens_per_sequence"].is_null());
}

TEST_CASE("SingleModelRouter preserves source and effective load paths",
          "[model_paths]") {
  SingleModelRouter router;

  ModelInfo info;
  info.id = "model-a";
  info.path = "hf://org/repo";
  info.effective_load_path = "/tmp/cache/org/repo/model.Q4_K_M.gguf";
  info.backend = "cpu";
  REQUIRE(router.RegisterModel(info, ReadyBackend()));

  auto models = router.ListModels();
  REQUIRE(models.size() == 1u);
  REQUIRE(models[0].path == "hf://org/repo");
  REQUIRE(models[0].source_path == "hf://org/repo");
  REQUIRE(models[0].effective_load_path ==
          "/tmp/cache/org/repo/model.Q4_K_M.gguf");
}

TEST_CASE("SingleModelRouter defaults effective path to source path",
          "[model_paths]") {
  SingleModelRouter router;

  ModelInfo info;
  info.id = "model-b";
  info.path = "/models/model.gguf";
  info.backend = "cpu";
  REQUIRE(router.RegisterModel(info, ReadyBackend()));

  auto models = router.ListModels();
  REQUIRE(models.size() == 1u);
  REQUIRE(models[0].path == "/models/model.gguf");
  REQUIRE(models[0].source_path == "/models/model.gguf");
  REQUIRE(models[0].effective_load_path == "/models/model.gguf");
}

TEST_CASE("SingleModelRouter ResolveExact enforces explicit identity",
          "[model_identity]") {
  SingleModelRouter router;

  ModelInfo default_info;
  default_info.id = "default-model";
  default_info.path = "/models/default.gguf";
  default_info.backend = "cpu";
  REQUIRE(router.RegisterModel(default_info, ReadyBackend()));

  ModelInfo other_info;
  other_info.id = "other-model";
  other_info.path = "/models/other.gguf";
  other_info.backend = "cpu";
  REQUIRE(router.RegisterModel(other_info, ReadyBackend()));

  auto *fallback = router.Resolve("missing-model");
  REQUIRE(fallback != nullptr);
  REQUIRE(fallback->id == "default-model");

  REQUIRE(router.ResolveExact("missing-model") == nullptr);
  auto *resolved_other = router.ResolveExact("other-model");
  REQUIRE(resolved_other != nullptr);
  REQUIRE(resolved_other->id == "other-model");
}

TEST_CASE("SingleModelRouter admin identity operations reject unknown ids",
          "[model_identity]") {
  SingleModelRouter router;

  ModelInfo first;
  first.id = "model-a";
  first.path = "/models/a.gguf";
  first.backend = "cpu";
  REQUIRE(router.RegisterModel(first, ReadyBackend()));

  ModelInfo second;
  second.id = "model-b";
  second.path = "/models/b.gguf";
  second.backend = "cpu";
  REQUIRE(router.RegisterModel(second, ReadyBackend()));

  REQUIRE_FALSE(router.SetDefaultModel("missing-model"));
  REQUIRE_FALSE(router.UnloadModel("missing-model"));

  REQUIRE(router.SetDefaultModel("model-b"));
  REQUIRE(router.DefaultModelId() == "model-b");
  REQUIRE(router.UnloadModel("model-b"));
  REQUIRE(router.ResolveExact("model-b") == nullptr);
  REQUIRE(router.DefaultModelId() == "model-a");
}

TEST_CASE("Device discovery alone is not reported as verified placement",
          "[model_identity][placement]") {
  ModelInfo info;
  info.backend = "llama_cpp_cuda";
  info.ready = true;
  auto model = BuildModelIdentityJson(info);
  REQUIRE(model["placement"]["state"] == "unverified");
  REQUIRE(model["placement"]["gpu_weight_bytes"] == 0);
  REQUIRE(model["placement"]["stable_id"].is_null());
  REQUIRE_FALSE(model["placement"].contains("embedding_batch_limits"));
  info.placement.requested = "cuda:1";
  info.placement.effective = "cuda:1";
  info.placement.vendor = "cuda";
  info.placement.state = "verified_weights";
  info.placement.gpu_weight_bytes = 4096;
  info.placement.cpu_weight_bytes = 1024;
  info.placement.stable_id = "0000:01:00.0";
  info.placement.embedding_batch = EmbeddingBatchGeometry::Resolve(2);
  model = BuildModelIdentityJson(info);
  REQUIRE(model["placement"]["gpu_weight_bytes"] == 4096);
  REQUIRE(model["placement"]["cpu_weight_bytes"] == 1024);
  REQUIRE(model["placement"]["stable_id"] == "0000:01:00.0");
  const auto limits = model["placement"]["embedding_batch_limits"];
  REQUIRE(limits["max_sequences"] == 2);
  REQUIRE(limits["tokens_per_sequence"] == 512);
  REQUIRE(limits["max_batch_tokens"] == 1024);
}
