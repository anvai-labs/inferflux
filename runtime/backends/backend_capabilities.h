#pragma once

#include <string>

namespace inferflux {

// BackendCapabilities captures the feature surface exposed by a loaded model.
// Keep this small and backend-agnostic so HTTP/scheduler code can gate
// requests without depending on backend implementation details.
struct BackendCapabilities {
  bool supports_generation{true};
  bool supports_streaming{true};
  bool supports_logprobs{true};
  bool supports_structured_output{true};
  bool supports_embeddings{true};
  bool supports_vision{false};
  bool supports_speculative_decoding{true};
  bool supports_fairness_preemption{true};
  bool supports_kv_prefix_transfer{true};
};

// Feature requirements extracted from a request.
struct BackendFeatureRequirements {
  bool needs_generation{false};
  bool needs_streaming{false};
  bool needs_logprobs{false};
  bool needs_structured_output{false};
  bool needs_embeddings{false};
  bool needs_vision{false};
  bool needs_speculative_decoding{false};
  bool needs_fairness_preemption{false};
};

struct CapabilityCheckResult {
  bool supported{true};
  std::string missing_feature;
  std::string reason;
};

// Shared constructors for request capability requirements.
// These helpers keep endpoint-level routing policy in sync across scheduler,
// HTTP handlers, and future admission paths.
BackendFeatureRequirements BuildGenerationFeatureRequirements(
    bool needs_streaming, bool needs_logprobs, bool needs_structured_output,
    bool needs_vision, bool needs_speculative_decoding = false,
    bool needs_fairness_preemption = false);
BackendFeatureRequirements BuildEmbeddingFeatureRequirements();

CapabilityCheckResult
CheckBackendCapabilities(const BackendCapabilities &capabilities,
                         const BackendFeatureRequirements &requirements);

} // namespace inferflux
