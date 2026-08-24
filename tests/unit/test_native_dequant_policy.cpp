#include <catch2/catch_amalgamated.hpp>

#define private public
#include "runtime/backends/cuda/inferflux_cuda_executor.h"
#undef private

namespace inferflux {
namespace {

class MockDequantPolicyLoader : public runtime::cuda::native::IModelLoader {
public:
  explicit MockDequantPolicyLoader(std::string format)
      : format_(std::move(format)) {}

  bool Load(const std::filesystem::path &) override { return true; }
  const runtime::cuda::native::ModelInfo &GetModelInfo() const override {
    return model_info_;
  }
  std::string GetFormat() const override { return format_; }
  bool IsQuantized() const override { return true; }
  std::string GetQuantizationType() const override { return "q4_k_m"; }
  bool UploadToGPU(cudaStream_t) override { return true; }
  void FreeCPUMemory() override {}
  void FreeGPUMemory() override {}
  void *GetGPUBuffer() const override { return nullptr; }
  size_t GetGPUSize() const override { return 0; }
  void SetDequantizedCachePolicy(
      runtime::cuda::native::DequantizedCachePolicy policy) override {
    policy_ = policy;
    ++set_policy_calls_;
  }
  runtime::cuda::native::DequantizedCachePolicy
  GetDequantizedCachePolicy() const override {
    return policy_;
  }
  void ClearDequantizedCache() override { ++clear_calls_; }
  std::shared_ptr<runtime::cuda::native::IWeightAccessor>
  GetWeightAccessor(const std::string &) override {
    return nullptr;
  }

  int clear_calls() const { return clear_calls_; }
  int set_policy_calls() const { return set_policy_calls_; }

private:
  std::string format_;
  runtime::cuda::native::ModelInfo model_info_{};
  runtime::cuda::native::DequantizedCachePolicy policy_{
      runtime::cuda::native::DequantizedCachePolicy::kNone};
  int clear_calls_{0};
  int set_policy_calls_{0};
};

} // namespace

TEST_CASE("InferfluxCudaExecutor dequant policy parser uses none as default",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;

  REQUIRE(executor.ConfigureDequantizedCachePolicy(""));
  CHECK(executor.dequantized_cache_policy_ ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);
  CHECK(executor.dequantized_cache_policy_hint_ == "none");

  REQUIRE(executor.ConfigureDequantizedCachePolicy("invalid_value"));
  CHECK(executor.dequantized_cache_policy_ ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);
  CHECK(executor.dequantized_cache_policy_hint_ == "none");
}

TEST_CASE("InferfluxCudaExecutor dequant policy parser accepts none batch model",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;

  REQUIRE(executor.ConfigureDequantizedCachePolicy("none"));
  CHECK(executor.dequantized_cache_policy_ ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);

  REQUIRE(executor.ConfigureDequantizedCachePolicy("batch"));
  CHECK(executor.dequantized_cache_policy_ ==
        runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime);

  REQUIRE(executor.ConfigureDequantizedCachePolicy("model"));
  CHECK(executor.dequantized_cache_policy_ ==
        runtime::cuda::native::DequantizedCachePolicy::kModelLifetime);
}

TEST_CASE("InferfluxCudaExecutor releases dequant cache for non-model policies "
          "on GGUF loaders",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;

  // kNone: request-boundary cleanup clears dequantized cache
  auto *none_loader = new MockDequantPolicyLoader("gguf");
  executor.model_loader_.reset(none_loader);
  executor.dequantized_cache_policy_ =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  executor.ReleaseBatchScopedDequantizedCache();
  CHECK(none_loader->clear_calls() == 1);

  // kBatchLifetime: clear between batches
  auto *batch_loader = new MockDequantPolicyLoader("gguf");
  executor.model_loader_.reset(batch_loader);
  executor.dequantized_cache_policy_ =
      runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
  executor.ReleaseBatchScopedDequantizedCache();
  CHECK(batch_loader->clear_calls() == 1);
}

TEST_CASE("InferfluxCudaExecutor retains dequant cache for model policy and "
          "non-GGUF loaders",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;

  auto *model_loader = new MockDequantPolicyLoader("gguf");
  executor.model_loader_.reset(model_loader);
  executor.dequantized_cache_policy_ =
      runtime::cuda::native::DequantizedCachePolicy::kModelLifetime;
  executor.ReleaseBatchScopedDequantizedCache();
  CHECK(model_loader->clear_calls() == 0);

  auto *safetensors_loader = new MockDequantPolicyLoader("safetensors");
  executor.model_loader_.reset(safetensors_loader);
  executor.dequantized_cache_policy_ =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  executor.ReleaseBatchScopedDequantizedCache();
  CHECK(safetensors_loader->clear_calls() == 0);
}

} // namespace inferflux
