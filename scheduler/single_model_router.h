#pragma once

#include "runtime/backends/common/backend_config.h"
#include "runtime/backends/common/backend_interface.h"
#include "scheduler/model_router.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

// SingleModelRouter is the default ModelRouter implementation.
// It wraps a single LlamaCppBackend and presents it through the ModelRouter
// interface. Resolve() always returns the primary model regardless of the
// requested model ID, mirroring the original single-model-per-server behavior.
//
// Future multi-model routers can replace this at construction time in main.cpp
// without touching Scheduler or HttpServer.
class SingleModelRouter : public ModelRouter {
public:
  // Construct with a pre-loaded backend. The model info (id, path, backend
  // label) describes it for the /v1/models list endpoint.
  explicit SingleModelRouter(std::shared_ptr<BackendInterface> backend,
                             ModelInfo info);

  // Construct empty (no model loaded yet).
  SingleModelRouter();
  explicit SingleModelRouter(
      const LlamaBackendConfig &default_backend_config,
      const std::string &default_backend_hint = "cpu",
      const std::vector<std::string> &backend_priority = {});

  // Registers an already-loaded backend (used by server startup/tests).
  bool RegisterModel(const ModelInfo &info,
                     std::shared_ptr<BackendInterface> backend);

  // ModelRouter interface.
  std::vector<ModelInfo> ListModels() const override;
  std::string LoadModel(const std::string &path,
                        const std::string &backend_hint = "",
                        const std::string &requested_id = "",
                        const std::string &model_format = "auto") override;
  bool UnloadModel(const std::string &id) override;
  ModelInfo *Resolve(const std::string &requested_model) override;
  ModelInfo *ResolveExact(const std::string &model_id) override;
  std::shared_ptr<BackendInterface>
  GetBackend(const std::string &model_id) override;
  bool SetDefaultModel(const std::string &model_id) override;
  std::string DefaultModelId() const override;
  std::string LastLoadError() const override;
  std::string Name() const override { return "single"; }

  // Returns the underlying backend (nullptr if no model is loaded).
  std::shared_ptr<BackendInterface> Backend() const;

private:
  struct Entry {
    ModelInfo info;
    std::shared_ptr<BackendInterface> backend;
    std::chrono::steady_clock::time_point load_time;
  };

  std::string GenerateModelIdLocked(const std::string &path) const;
  std::string EnsureUniqueIdLocked(const std::string &preferred) const;
  void RecordModelReadyLocked(const Entry &entry, bool ready) const;
  void UpdateDefaultModelLocked();
  std::vector<std::string>
  BuildBackendCandidates(const std::string &backend_hint) const;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> models_;
  std::string default_model_id_;
  std::string last_load_error_;
  LlamaBackendConfig default_backend_config_{};
  std::string default_backend_hint_{"cpu"};
  std::vector<std::string> backend_priority_{"cpu"};
};

} // namespace inferflux
