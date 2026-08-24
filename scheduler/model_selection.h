#pragma once

#include "runtime/backends/backend_capabilities.h"
#include "scheduler/model_router.h"

#include <memory>
#include <string>

namespace inferflux {

enum class ModelSelectionStatus {
  kSelected,
  kNotFound,
  kUnsupported,
  kBackendUnavailable,
};

enum class CapabilityFallbackScope {
  kAnyCompatible,
  kSamePathOnly,
};

struct ModelSelectionOptions {
  bool allow_capability_fallback_for_default{true};
  bool require_ready_backend{false};
  CapabilityFallbackScope capability_fallback_scope{
      CapabilityFallbackScope::kAnyCompatible};
};

struct ModelSelectionResult {
  ModelSelectionStatus status{ModelSelectionStatus::kNotFound};
  ModelInfo info{};
  std::shared_ptr<BackendInterface> backend;
  bool used_fallback{false};
  std::string fallback_from_backend;
  std::string fallback_feature;
  std::string missing_feature;
  std::string reason;
};

std::string CapabilityFallbackScopeToString(CapabilityFallbackScope scope);
bool IsCapabilityFallbackScopeValue(const std::string &value);
CapabilityFallbackScope
ParseCapabilityFallbackScope(const std::string &value,
                             CapabilityFallbackScope default_scope =
                                 CapabilityFallbackScope::kAnyCompatible);

ModelSelectionResult
SelectModelForRequest(ModelRouter *router, const std::string &requested_model,
                      const BackendFeatureRequirements &requirements,
                      const ModelSelectionOptions &options = {});

} // namespace inferflux
