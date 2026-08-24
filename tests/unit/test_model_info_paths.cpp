#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "scheduler/model_router.h"
#include "scheduler/single_model_router.h"

#include <memory>

using namespace inferflux;

namespace {

std::shared_ptr<LlamaCppBackend> ReadyBackend() {
  auto backend = std::make_shared<LlamaCppBackend>();
  backend->ForceReadyForTests();
  return backend;
}

} // namespace

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
