#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/llama/llama_backend_traits.h"

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

enum class BackendProvider {
  kLlamaCpp,
  kNative,
};

struct BackendExposurePolicy {
  bool prefer_inferflux{true};
  bool allow_llama_cpp_fallback{true};
  // When true, explicit InferFlux-engine hints (e.g. inferflux_cuda) fail
  // fast unless native kernels are ready; no scaffold/delegate path is
  // accepted.
  bool strict_inferflux_request{false};
};

struct BackendFactoryResult {
  std::shared_ptr<BackendInterface> backend;
  std::string backend_label{"cpu"};
  LlamaBackendTarget target{LlamaBackendTarget::kCpu};
  LlamaBackendTraits traits{};
  BackendCapabilities capabilities{};
  BackendProvider provider{BackendProvider::kLlamaCpp};
  bool used_fallback{false};
  std::string fallback_reason;
  bool require_strict_inferflux_execution{false};
  LlamaBackendConfig config{};
};

class BackendFactory {
public:
  static BackendFactoryResult Create(const std::string &backend_hint);
  static std::string NormalizeHint(const std::string &backend_hint);
  static std::string CanonicalBackendId(BackendProvider provider,
                                        LlamaBackendTarget target);
  static std::string ProviderLabel(BackendProvider provider);
  static BackendProvider ParseProviderLabel(const std::string &provider_label);
  static std::vector<std::string>
  NormalizeHintList(const std::vector<std::string> &backend_hints,
                    const std::string &default_hint = "cpu");
  static void SetExposurePolicy(const BackendExposurePolicy &policy);
  static BackendExposurePolicy ExposurePolicy();
};

LlamaBackendConfig MergeBackendConfig(const LlamaBackendConfig &defaults,
                                      const BackendFactoryResult &selection);

} // namespace inferflux
