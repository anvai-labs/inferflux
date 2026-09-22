#include "runtime/backends/backend_capabilities.h"

namespace inferflux {

namespace {

CapabilityCheckResult Missing(const std::string &feature,
                              const std::string &reason) {
  CapabilityCheckResult out;
  out.supported = false;
  out.missing_feature = feature;
  out.reason = reason;
  return out;
}

} // namespace

BackendFeatureRequirements BuildGenerationFeatureRequirements(
    bool needs_streaming, bool needs_logprobs, bool needs_structured_output,
    bool needs_vision, bool needs_speculative_decoding,
    bool needs_fairness_preemption) {
  BackendFeatureRequirements requirements;
  requirements.needs_generation = true;
  requirements.needs_streaming = needs_streaming;
  requirements.needs_logprobs = needs_logprobs;
  requirements.needs_structured_output = needs_structured_output;
  requirements.needs_vision = needs_vision;
  requirements.needs_speculative_decoding = needs_speculative_decoding;
  requirements.needs_fairness_preemption = needs_fairness_preemption;
  return requirements;
}

BackendFeatureRequirements BuildEmbeddingFeatureRequirements() {
  BackendFeatureRequirements requirements;
  requirements.needs_embeddings = true;
  return requirements;
}

CapabilityCheckResult
CheckBackendCapabilities(const BackendCapabilities &capabilities,
                         const BackendFeatureRequirements &requirements) {
  if (requirements.needs_generation && !capabilities.supports_generation) {
    return Missing("generation",
                   "Selected model does not support text generation");
  }
  if (requirements.needs_structured_output &&
      !capabilities.supports_structured_output) {
    return Missing(
        "structured_output",
        "Selected model/backend does not support response_format constraints");
  }
  if (requirements.needs_logprobs && !capabilities.supports_logprobs) {
    return Missing("logprobs",
                   "Selected model/backend does not support logprobs");
  }
  if (requirements.needs_streaming && !capabilities.supports_streaming) {
    return Missing("streaming",
                   "Selected model/backend does not support streaming");
  }
  if (requirements.needs_embeddings && !capabilities.supports_embeddings) {
    return Missing("embeddings",
                   "Selected model/backend does not support embeddings");
  }
  if (requirements.needs_vision && !capabilities.supports_vision) {
    return Missing("vision",
                   "Selected model/backend does not support image inputs");
  }
  if (requirements.needs_speculative_decoding &&
      !capabilities.supports_speculative_decoding) {
    return Missing(
        "speculative_decoding",
        "Selected model/backend does not support speculative decoding");
  }
  if (requirements.needs_fairness_preemption &&
      !capabilities.supports_fairness_preemption) {
    return Missing(
        "fairness_preemption",
        "Selected model/backend does not support fairness preemption");
  }
  return {};
}

} // namespace inferflux
