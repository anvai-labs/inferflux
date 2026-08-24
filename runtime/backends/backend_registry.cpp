#include "runtime/backends/backend_registry.h"
#include "runtime/backends/backend_factory.h"
#include "server/logging/logger.h"

namespace inferflux {

BackendRegistry &BackendRegistry::Instance() {
  static BackendRegistry instance;
  return instance;
}

void BackendRegistry::Register(LlamaBackendTarget target,
                               BackendProvider provider, CreatorFn fn) {
  std::lock_guard<std::mutex> lock(mutex_);
  Key key{target, provider};
  creators_[key] = std::move(fn);
}

std::shared_ptr<BackendInterface>
BackendRegistry::Create(LlamaBackendTarget target,
                        BackendProvider provider) const {
  std::lock_guard<std::mutex> lock(mutex_);
  Key key{target, provider};
  auto it = creators_.find(key);
  if (it == creators_.end()) {
    return nullptr;
  }
  return it->second();
}

bool BackendRegistry::Has(LlamaBackendTarget target,
                          BackendProvider provider) const {
  std::lock_guard<std::mutex> lock(mutex_);
  Key key{target, provider};
  return creators_.find(key) != creators_.end();
}

std::vector<LlamaBackendTarget> BackendRegistry::AvailableTargets() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<LlamaBackendTarget> targets;
  for (const auto &entry : creators_) {
    bool found = false;
    for (auto t : targets) {
      if (t == entry.first.target) {
        found = true;
        break;
      }
    }
    if (!found) {
      targets.push_back(entry.first.target);
    }
  }
  return targets;
}

} // namespace inferflux
