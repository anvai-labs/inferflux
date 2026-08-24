#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/llama/llama_backend_traits.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

enum class BackendProvider;

class BackendRegistry {
public:
  using CreatorFn = std::function<std::shared_ptr<BackendInterface>()>;

  static BackendRegistry &Instance();

  void Register(LlamaBackendTarget target, BackendProvider provider,
                CreatorFn fn);
  std::shared_ptr<BackendInterface> Create(LlamaBackendTarget target,
                                           BackendProvider provider) const;
  bool Has(LlamaBackendTarget target, BackendProvider provider) const;
  std::vector<LlamaBackendTarget> AvailableTargets() const;

private:
  BackendRegistry() = default;

  struct Key {
    LlamaBackendTarget target;
    BackendProvider provider;
    bool operator==(const Key &other) const {
      return target == other.target && provider == other.provider;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key &k) const {
      return std::hash<int>()(static_cast<int>(k.target)) ^
             (std::hash<int>()(static_cast<int>(k.provider)) << 16);
    }
  };

  mutable std::mutex mutex_;
  std::unordered_map<Key, CreatorFn, KeyHash> creators_;
};

} // namespace inferflux
