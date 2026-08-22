#include <catch2/catch_amalgamated.hpp>

#include "support/executor_test_access.h"

namespace inferflux {
namespace {

class MockDequantPolicyLoader : public runtime::cuda::native::IModelLoader {
public:
  explicit MockDequantPolicyLoader(std::string format, bool has_cache = true)
      : format_(std::move(format)), has_cache_(has_cache) {}

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
  void ClearDequantizedCache() override {
    ++clear_calls_;
    has_cache_ = false;
  }
  bool HasDequantizedCache() const override { return has_cache_; }
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
  bool has_cache_{true};
  int clear_calls_{0};
  int set_policy_calls_{0};
};

} // namespace

TEST_CASE("InferfluxCudaExecutor dequant policy parser uses none as default",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;
  ExecutorTestAccess acc(executor);

  REQUIRE(acc.ConfigureDequantizedCachePolicy(""));
  CHECK(acc.dequantized_cache_policy() ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);
  CHECK(acc.dequantized_cache_policy_hint() == "none");

  REQUIRE(acc.ConfigureDequantizedCachePolicy("invalid_value"));
  CHECK(acc.dequantized_cache_policy() ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);
  CHECK(acc.dequantized_cache_policy_hint() == "none");
}

TEST_CASE(
    "InferfluxCudaExecutor dequant policy parser accepts none batch model",
    "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;
  ExecutorTestAccess acc(executor);

  REQUIRE(acc.ConfigureDequantizedCachePolicy("none"));
  CHECK(acc.dequantized_cache_policy() ==
        runtime::cuda::native::DequantizedCachePolicy::kNone);

  REQUIRE(acc.ConfigureDequantizedCachePolicy("batch"));
  CHECK(acc.dequantized_cache_policy() ==
        runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime);

  REQUIRE(acc.ConfigureDequantizedCachePolicy("model"));
  CHECK(acc.dequantized_cache_policy() ==
        runtime::cuda::native::DequantizedCachePolicy::kModelLifetime);
}

TEST_CASE("InferfluxCudaExecutor releases dequant cache for non-model policies "
          "on GGUF loaders",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;
  ExecutorTestAccess acc(executor);

  // kNone: request-boundary cleanup clears dequantized cache
  auto *none_loader = new MockDequantPolicyLoader("gguf");
  acc.model_loader().reset(none_loader);
  acc.dequantized_cache_policy() =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  acc.ReleaseBatchScopedDequantizedCache();
  CHECK(none_loader->clear_calls() == 1);

  // kBatchLifetime: clear between batches
  auto *batch_loader = new MockDequantPolicyLoader("gguf");
  acc.model_loader().reset(batch_loader);
  acc.dequantized_cache_policy() =
      runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
  acc.ReleaseBatchScopedDequantizedCache();
  CHECK(batch_loader->clear_calls() == 1);
}

TEST_CASE("InferfluxCudaExecutor retains dequant cache for model policy and "
          "non-GGUF loaders",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;
  ExecutorTestAccess acc(executor);

  auto *model_loader = new MockDequantPolicyLoader("gguf");
  acc.model_loader().reset(model_loader);
  acc.dequantized_cache_policy() =
      runtime::cuda::native::DequantizedCachePolicy::kModelLifetime;
  acc.ReleaseBatchScopedDequantizedCache();
  CHECK(model_loader->clear_calls() == 0);

  auto *safetensors_loader = new MockDequantPolicyLoader("safetensors");
  acc.model_loader().reset(safetensors_loader);
  acc.dequantized_cache_policy() =
      runtime::cuda::native::DequantizedCachePolicy::kNone;
  acc.ReleaseBatchScopedDequantizedCache();
  CHECK(safetensors_loader->clear_calls() == 0);
}

TEST_CASE("InferfluxCudaExecutor avoids cache cleanup when no dequantized "
          "weights exist",
          "[native_forward][dequant_policy]") {
  InferfluxCudaExecutor executor;
  ExecutorTestAccess acc(executor);

  auto *loader = new MockDequantPolicyLoader("gguf", false);
  acc.model_loader().reset(loader);
  acc.dequantized_cache_policy() =
      runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
  acc.ReleaseBatchScopedDequantizedCache();
  CHECK(loader->clear_calls() == 0);
}

} // namespace inferflux
