#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/backends/cuda/inferflux_cuda_backend.h"
#include "scheduler/single_model_router.h"

namespace inferflux {

namespace {

class NativeContractBackendStub final : public InferfluxCudaBackend {
public:
  NativeContractBackendStub(bool supports_logprobs,
                            bool supports_structured_output,
                            bool supports_embeddings,
                            bool supports_speculative_decoding)
      : supports_logprobs_(supports_logprobs),
        supports_structured_output_(supports_structured_output),
        supports_embeddings_(supports_embeddings),
        supports_speculative_decoding_(supports_speculative_decoding) {}

  bool SupportsLogprobsContract() const override { return supports_logprobs_; }
  bool SupportsStructuredOutputContract() const override {
    return supports_structured_output_;
  }
  bool SupportsEmbeddingsContract() const override {
    return supports_embeddings_;
  }
  bool SupportsSpeculativeDecodingContract() const override {
    return supports_speculative_decoding_;
  }
  bool IsReady() const override { return true; }

private:
  bool supports_logprobs_{false};
  bool supports_structured_output_{false};
  bool supports_embeddings_{false};
  bool supports_speculative_decoding_{false};
};

} // namespace

TEST_CASE("SingleModelRouter applies native capability contract",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<LlamaCppBackend>();
  backend->ForceReadyForTests();

  ModelInfo info;
  info.id = "native-contract";
  info.path = "models/native.gguf";
  info.backend = "cuda";
  info.backend_provider = "inferflux";

  REQUIRE(router->RegisterModel(info, backend));
  auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE_FALSE(caps.supports_logprobs);
  REQUIRE_FALSE(caps.supports_structured_output);
  REQUIRE_FALSE(caps.supports_embeddings);
  REQUIRE_FALSE(caps.supports_speculative_decoding);
  REQUIRE(caps.supports_kv_prefix_transfer);
}

TEST_CASE("SingleModelRouter maps endpoint contracts from native backend",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<NativeContractBackendStub>(
      /*supports_logprobs=*/false, /*supports_structured_output=*/false,
      /*supports_embeddings=*/false, /*supports_speculative_decoding=*/true);

  ModelInfo info;
  info.id = "native-explicit-contract";
  info.path = "models/native.safetensors";
  info.backend = "cuda";
  info.backend_provider = "inferflux";

  REQUIRE(router->RegisterModel(info, backend));
  const auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE_FALSE(caps.supports_logprobs);
  REQUIRE_FALSE(caps.supports_structured_output);
  REQUIRE_FALSE(caps.supports_embeddings);
  REQUIRE(caps.supports_speculative_decoding);
  REQUIRE(caps.supports_kv_prefix_transfer);
}

TEST_CASE("SingleModelRouter preserves enabled native endpoint contracts",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<NativeContractBackendStub>(
      /*supports_logprobs=*/true, /*supports_structured_output=*/true,
      /*supports_embeddings=*/true, /*supports_speculative_decoding=*/true);

  ModelInfo info;
  info.id = "native-enabled-contract";
  info.path = "models/native.gguf";
  info.backend = "cuda";
  info.backend_provider = "inferflux";

  REQUIRE(router->RegisterModel(info, backend));
  const auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE(caps.supports_logprobs);
  REQUIRE(caps.supports_structured_output);
  REQUIRE(caps.supports_embeddings);
  REQUIRE(caps.supports_speculative_decoding);
  REQUIRE(caps.supports_kv_prefix_transfer);
}

TEST_CASE("SingleModelRouter keeps llama.cpp capability defaults",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<LlamaCppBackend>();
  backend->ForceReadyForTests();

  ModelInfo info;
  info.id = "llama-contract";
  info.path = "models/llama.gguf";
  info.backend = "cuda";
  info.backend_provider = "llama_cpp";

  REQUIRE(router->RegisterModel(info, backend));
  auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE(caps.supports_logprobs);
  REQUIRE(caps.supports_structured_output);
  REQUIRE(caps.supports_embeddings);
}

} // namespace inferflux
