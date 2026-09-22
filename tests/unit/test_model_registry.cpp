#include <catch2/catch_amalgamated.hpp>

#include "scheduler/model_registry.h"
#include "scheduler/model_router.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace inferflux;

// ---------------------------------------------------------------------------
// Minimal ModelRouter stub that records LoadModel / UnloadModel calls.
// ---------------------------------------------------------------------------

class StubRouter : public ModelRouter {
public:
  struct LoadCall {
    std::string path;
    std::string backend;
    std::string id;
    std::string format;
  };

  std::vector<ModelLoadSpec> specifications;
  std::vector<LoadCall> load_calls;
  std::vector<std::string> unload_calls;
  int next_id_suffix{0};
  bool fail_load{false};
  bool fail_unload{false};

  std::vector<ModelInfo> ListModels() const override { return {}; }

  std::string LoadModel(const std::string &path, const std::string &backend,
                        const std::string &id,
                        const std::string &format) override {
    if (fail_load)
      return "";
    load_calls.push_back({path, backend, id, format});
    return id.empty() ? ("auto-id-" + std::to_string(next_id_suffix++)) : id;
  }

  std::string LoadModel(const ModelLoadSpec &spec) override {
    specifications.push_back(spec);
    return LoadModel(spec.path, spec.backend, spec.id, spec.format);
  }

  bool UnloadModel(const std::string &id) override {
    unload_calls.push_back(id);
    return !fail_unload;
  }

  ModelInfo *Resolve(const std::string &) override { return nullptr; }
  ModelInfo *ResolveExact(const std::string &) override { return nullptr; }
  std::shared_ptr<BackendInterface> GetBackend(const std::string &) override {
    return nullptr;
  }
  bool SetDefaultModel(const std::string &) override { return false; }
  std::string DefaultModelId() const override { return ""; }
  std::string Name() const override { return "stub"; }
};

// ---------------------------------------------------------------------------
// Helper: write a registry YAML to a temp file.
// ---------------------------------------------------------------------------
static fs::path WriteTempRegistry(const std::string &content) {
  auto tmp = fs::temp_directory_path() /
             ("ifx_reg_" +
              std::to_string(
                  std::hash<std::thread::id>{}(std::this_thread::get_id())) +
              ".yaml");
  std::ofstream f(tmp);
  f << content;
  return tmp;
}

// ---------------------------------------------------------------------------
// [model_registry] tests
// ---------------------------------------------------------------------------

TEST_CASE("ModelRegistry empty file loads zero models", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry("models: []\n");
  int n = reg.LoadAndWatch(path, /*poll_ms=*/99999);
  reg.Stop();
  fs::remove(path);

  REQUIRE(n == 0);
  REQUIRE(router->load_calls.empty());
}

TEST_CASE("ModelRegistry loads models on startup", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry(R"(
models:
  - id: m1
    path: /models/a.gguf
    format: gguf
    backend: cpu
  - id: m2
    path: /models/b.gguf
)");

  int n = reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);

  REQUIRE(n == 2);
  REQUIRE(router->load_calls.size() == 2u);
  REQUIRE(router->load_calls[0].id == "m1");
  REQUIRE(router->load_calls[0].path == "/models/a.gguf");
  REQUIRE(router->load_calls[0].backend == "cpu");
  REQUIRE(router->load_calls[0].format == "gguf");
  REQUIRE(router->load_calls[1].id == "m2");
  REQUIRE(router->load_calls[1].format == "auto");
}

TEST_CASE("ModelRegistry normalizes format values and defaults invalid ones",
          "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry(R"(
models:
  - id: sf
    path: /models/sf
    format: safe_tensors
  - id: bad
    path: /models/bad
    format: unknown-format
)");

  int n = reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);

  REQUIRE(n == 2);
  REQUIRE(router->load_calls.size() == 2u);
  REQUIRE(router->load_calls[1].id == "sf");
  REQUIRE(router->load_calls[1].format == "safetensors");
  REQUIRE(router->load_calls[0].id == "bad");
  REQUIRE(router->load_calls[0].format == "auto");
}

TEST_CASE("ModelRegistry skips entry with no path", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry(R"(
models:
  - id: no-path-entry
  - id: ok
    path: /models/c.gguf
)");

  int n = reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);

  // Only the entry with a path should be loaded.
  REQUIRE(n == 1);
  REQUIRE(router->load_calls.size() == 1u);
  REQUIRE(router->load_calls[0].id == "ok");
}

TEST_CASE("ModelRegistry handles load failure gracefully", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  router->fail_load = true;
  ModelRegistry reg(router);

  auto path = WriteTempRegistry(R"(
models:
  - id: m1
    path: /models/a.gguf
)");

  int n = reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);

  REQUIRE(n == 0);
}

TEST_CASE("ModelRegistry Reload removes models no longer in file",
          "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  // Initial: two models
  auto path = WriteTempRegistry(R"(
models:
  - id: m1
    path: /models/a.gguf
  - id: m2
    path: /models/b.gguf
)");

  reg.LoadAndWatch(path, 99999);
  REQUIRE(router->load_calls.size() == 2u);

  // Overwrite with one model removed.
  {
    std::ofstream f(path);
    f << "models:\n  - id: m1\n    path: /models/a.gguf\n";
  }
  reg.Reload();
  reg.Stop();
  fs::remove(path);

  // m2 should have been unloaded.
  REQUIRE(router->unload_calls.size() == 1u);
  REQUIRE(router->unload_calls[0] == "m2");
}

TEST_CASE("ModelRegistry Reload adds new models", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry("models: []\n");
  reg.LoadAndWatch(path, 99999);
  REQUIRE(router->load_calls.empty());

  // Add a model.
  {
    std::ofstream f(path);
    f << "models:\n  - id: new\n    path: /models/new.gguf\n";
  }
  reg.Reload();
  reg.Stop();
  fs::remove(path);

  REQUIRE(router->load_calls.size() == 1u);
  REQUIRE(router->load_calls[0].id == "new");
}

TEST_CASE("ModelRegistry ManagedIds reflects loaded models",
          "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry(R"(
models:
  - id: alpha
    path: /a.gguf
  - id: beta
    path: /b.gguf
)");
  reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);

  auto ids = reg.ManagedIds();
  REQUIRE(ids.count("alpha") == 1u);
  REQUIRE(ids.count("beta") == 1u);
}

TEST_CASE("ModelRegistry Stop is idempotent", "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);
  REQUIRE_NOTHROW(reg.Stop());
  REQUIRE_NOTHROW(reg.Stop());
}

TEST_CASE("ModelRegistry LoadAndWatch is idempotent (second call is no-op)",
          "[model_registry]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);

  auto path = WriteTempRegistry("models:\n  - id: x\n    path: /x.gguf\n");
  reg.LoadAndWatch(path, 99999);
  int n2 = reg.LoadAndWatch(path, 99999); // second call → 0
  reg.Stop();
  fs::remove(path);

  REQUIRE(n2 == 0);
  REQUIRE(router->load_calls.size() == 1u); // loaded only once
}

TEST_CASE("ModelRegistry rejects duplicate identities before loading",
          "[model_registry][placement]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);
  const bool same_path = GENERATE(true, false);
  auto path = WriteTempRegistry(
      "models:\n  - id: amd\n    path: /a.gguf\n  - id: " +
      std::string(same_path ? "nvidia" : "amd") +
      "\n    path: " + std::string(same_path ? "/a.gguf" : "/b.gguf") + "\n");
  const int loaded = reg.LoadAndWatch(path, 99999);
  reg.Stop();
  fs::remove(path);
  REQUIRE(loaded == 0);
  REQUIRE(router->load_calls.empty());
}

TEST_CASE("ModelRegistry retains ownership when removal is refused",
          "[model_registry][placement]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);
  auto path = WriteTempRegistry("models:\n  - id: busy\n    path: /a.gguf\n");
  REQUIRE(reg.LoadAndWatch(path, 99999) == 1);
  reg.Stop();
  router->fail_unload = true;
  WriteTempRegistry("models: []\n");
  REQUIRE(reg.Reload() == 0);
  REQUIRE(reg.ManagedIds().count("busy") == 1);
  router->fail_unload = false;
  REQUIRE(reg.Reload() == -1);
  REQUIRE(reg.ManagedIds().empty());
  fs::remove(path);
}

TEST_CASE("ModelRegistry carries resources through the typed router contract",
          "[model_registry][placement]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);
  auto path =
      WriteTempRegistry("models:\n  - id: amd\n    path: /a.gguf\n    device: "
                        "rocm:0\n    context_size: 8192\n    gpu_layers: 8\n   "
                        " max_parallel_sequences: 2\n    kv_cache_type: f16\n");
  REQUIRE(reg.LoadAndWatch(path, 99999) == 1);
  reg.Stop();
  fs::remove(path);
  REQUIRE(router->specifications.size() == 1);
  const auto &spec = router->specifications.front();
  REQUIRE(spec.device == "rocm:0");
  REQUIRE(spec.context_size == 8192);
  REQUIRE(spec.gpu_layers == 8);
  REQUIRE(spec.max_parallel_sequences == 2);
  REQUIRE(spec.kv_cache_type == "f16");
}

TEST_CASE("ModelRegistry refuses live specification changes atomically",
          "[model_registry][placement]") {
  auto router = std::make_shared<StubRouter>();
  ModelRegistry reg(router);
  auto path = WriteTempRegistry(
      "models:\n  - id: a\n    path: /a.gguf\n  - id: b\n    path: /b.gguf\n");
  REQUIRE(reg.LoadAndWatch(path, 99999) == 2);
  reg.Stop();
  const auto change = GENERATE(
      "backend: rocm", "device: cuda:0", "context_size: 4096", "gpu_layers: 8",
      "max_parallel_sequences: 2", "kv_cache_type: q8_0");
  WriteTempRegistry(std::string("models:\n  - id: a\n    path: /a.gguf\n    ") +
                    change + "\n");
  REQUIRE(reg.Reload() == 0);
  REQUIRE(router->unload_calls.empty());
  REQUIRE(router->load_calls.size() == 2);
  REQUIRE(reg.ManagedIds().size() == 2);
  fs::remove(path);
}
