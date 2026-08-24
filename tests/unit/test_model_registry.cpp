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

  std::vector<LoadCall> load_calls;
  std::vector<std::string> unload_calls;
  int next_id_suffix{0};
  bool fail_load{false};

  std::vector<ModelInfo> ListModels() const override { return {}; }

  std::string LoadModel(const std::string &path, const std::string &backend,
                        const std::string &id,
                        const std::string &format) override {
    if (fail_load)
      return "";
    load_calls.push_back({path, backend, id, format});
    return id.empty() ? ("auto-id-" + std::to_string(next_id_suffix++)) : id;
  }

  bool UnloadModel(const std::string &id) override {
    unload_calls.push_back(id);
    return true;
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
