#include "runtime/backends/backend_manager.h"
#include "runtime/backends/backend_factory.h"
#include "server/logging/logger.h"

namespace inferflux {

std::shared_ptr<BackendInterface>
BackendManager::LoadBackend(const std::string &name, const std::string &path,
                            const LlamaBackendConfig &config,
                            bool prefer_cuda) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = backends_.find(name);
  if (it != backends_.end() && it->second && it->second->IsReady()) {
    return it->second;
  }

  auto selection = BackendFactory::Create(prefer_cuda ? "cuda" : "cpu");
  if (!selection.backend) {
    log::Error("backend_manager",
               "failed to create backend strategy for " + name);
    return nullptr;
  }

  auto merged_config = MergeBackendConfig(config, selection);
  if (!selection.backend->LoadModel(path, merged_config)) {
    log::Error("backend_manager",
               "failed to load model " + path + " for " + name);
    return nullptr;
  }

  backends_[name] = selection.backend;
  return selection.backend;
}

std::shared_ptr<BackendInterface>
BackendManager::GetBackend(const std::string &name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = backends_.find(name);
  if (it == backends_.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace inferflux
