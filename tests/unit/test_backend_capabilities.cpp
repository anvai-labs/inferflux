#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/backend_capabilities.h"

using namespace inferflux;

TEST_CASE("CheckBackendCapabilities succeeds when requirements are empty",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  BackendFeatureRequirements requirements;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE(result.supported);
  REQUIRE(result.missing_feature.empty());
  REQUIRE(result.reason.empty());
}

TEST_CASE("BuildGenerationFeatureRequirements maps request features",
          "[backend_capabilities]") {
  auto requirements =
      BuildGenerationFeatureRequirements(true, true, false, true, true, true);

  REQUIRE(requirements.needs_streaming);
  REQUIRE(requirements.needs_logprobs);
  REQUIRE_FALSE(requirements.needs_structured_output);
  REQUIRE(requirements.needs_vision);
  REQUIRE(requirements.needs_speculative_decoding);
  REQUIRE(requirements.needs_fairness_preemption);
  REQUIRE_FALSE(requirements.needs_embeddings);
}

TEST_CASE("BuildEmbeddingFeatureRequirements sets embeddings-only requirement",
          "[backend_capabilities]") {
  auto requirements = BuildEmbeddingFeatureRequirements();
  REQUIRE(requirements.needs_embeddings);
  REQUIRE_FALSE(requirements.needs_streaming);
  REQUIRE_FALSE(requirements.needs_logprobs);
  REQUIRE_FALSE(requirements.needs_structured_output);
  REQUIRE_FALSE(requirements.needs_vision);
  REQUIRE_FALSE(requirements.needs_speculative_decoding);
  REQUIRE_FALSE(requirements.needs_fairness_preemption);
}

TEST_CASE("CheckBackendCapabilities rejects unsupported logprobs",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  capabilities.supports_logprobs = false;

  BackendFeatureRequirements requirements;
  requirements.needs_logprobs = true;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE_FALSE(result.supported);
  REQUIRE(result.missing_feature == "logprobs");
  REQUIRE(result.reason.find("logprobs") != std::string::npos);
}

TEST_CASE("CheckBackendCapabilities rejects unsupported vision",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  capabilities.supports_vision = false;

  BackendFeatureRequirements requirements;
  requirements.needs_vision = true;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE_FALSE(result.supported);
  REQUIRE(result.missing_feature == "vision");
  REQUIRE(result.reason.find("image inputs") != std::string::npos);
}

TEST_CASE("Embedding-only models reject plain generation before execution",
          "[backend_capabilities][embeddings_admission]") {
  BackendCapabilities capabilities;
  capabilities.supports_generation = false;
  const auto generation = CheckBackendCapabilities(
      capabilities,
      BuildGenerationFeatureRequirements(false, false, false, false));
  REQUIRE_FALSE(generation.supported);
  REQUIRE(generation.missing_feature == "generation");
  REQUIRE(CheckBackendCapabilities(capabilities,
                                   BuildEmbeddingFeatureRequirements())
              .supported);
}
