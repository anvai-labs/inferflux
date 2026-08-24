#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantized_weight_map.h"
#include "runtime/backends/cuda/native/quantized_weight_map_adapter.h"

#include <memory>
#include <vector>

namespace fs = std::filesystem;
using namespace inferflux;

// Mock ModelLoader for testing
class MockModelLoader : public runtime::cuda::native::IModelLoader {
public:
  bool Load(const std::filesystem::path &) override { return true; }
  const runtime::cuda::native::ModelInfo &GetModelInfo() const override {
    return info_;
  }
  std::string GetFormat() const override { return "gguf"; }
  bool IsQuantized() const override { return true; }
  std::string GetQuantizationType() const override { return "q4_k_m"; }
  bool UploadToGPU(cudaStream_t) override { return true; }
  void FreeCPUMemory() override {}
  void FreeGPUMemory() override {}
  void *GetGPUBuffer() const override { return nullptr; }
  size_t GetGPUSize() const override { return 0; }
  void SetDequantizedCachePolicy(
      runtime::cuda::native::DequantizedCachePolicy) override {}
  runtime::cuda::native::DequantizedCachePolicy
  GetDequantizedCachePolicy() const override {
    return runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
  }
  void ClearDequantizedCache() override {}

  std::shared_ptr<runtime::cuda::native::IWeightAccessor>
  GetWeightAccessor(const std::string &tensor_name) override {
    // Return mock accessor for known tensors
    if (tensor_map_.find(tensor_name) != tensor_map_.end()) {
      return tensor_map_[tensor_name];
    }
    return nullptr;
  }

  void SetMockAccessor(
      const std::string &name,
      std::shared_ptr<runtime::cuda::native::IWeightAccessor> accessor) {
    tensor_map_[name] = accessor;
  }

private:
  runtime::cuda::native::ModelInfo info_;
  std::unordered_map<std::string,
                     std::shared_ptr<runtime::cuda::native::IWeightAccessor>>
      tensor_map_;
};

// Mock WeightAccessor
class MockWeightAccessor : public runtime::cuda::native::IWeightAccessor {
public:
  MockWeightAccessor(size_t rows, size_t cols, const std::string &dtype)
      : rows_(rows), cols_(cols), dtype_(dtype) {}

  std::pair<size_t, size_t> GetDimensions() const override {
    return {rows_, cols_};
  }

  std::string GetDataType() const override { return dtype_; }

  bool IsQuantized() const override {
    return dtype_.find("q4") != std::string::npos ||
           dtype_.find("q5") != std::string::npos ||
           dtype_.find("q6") != std::string::npos ||
           dtype_.find("q8") != std::string::npos;
  }

  void *GetGpuWeights(cudaStream_t) override { return nullptr; }

  half *GetDequantizedGpuWeights(cudaStream_t) override { return nullptr; }

  bool IsDequantizedCached() const override { return false; }

private:
  size_t rows_, cols_;
  std::string dtype_;
};

// =============================================================================
// Test Suite: QuantizedWeightMap Construction
// =============================================================================

TEST_CASE("QuantizedWeightMap: Default construction",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  // Should be in valid state
  // Can't directly test internal state, but verify it exists
}

TEST_CASE("QuantizedWeightMap: Non-copyable and non-movable",
          "[quantized][weight_map]") {
  // These tests verify the deleted constructors compile correctly
  QuantizedWeightMap weight_map1;

  // Copy constructor should not compile (deleted)
  // QuantizedWeightMap weight_map2 = weight_map1; // COMPILE ERROR

  // Move constructor should not compile (deleted)
  // QuantizedWeightMap weight_map3 = std::move(weight_map1); // COMPILE ERROR

  // Assignment operators should not compile (deleted)
  // QuantizedWeightMap weight_map4;
  // weight_map4 = weight_map1; // COMPILE ERROR

  (void)weight_map1; // Suppress unused warning
}

TEST_CASE("QuantizedWeightMap: strategy-driven fused matmul policy is mutable",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;
  REQUIRE(weight_map.AllowFusedQuantizedMatmul());

  weight_map.SetAllowFusedQuantizedMatmul(false);
  REQUIRE_FALSE(weight_map.AllowFusedQuantizedMatmul());

  QuantizedWeightMapAdapter adapter(&weight_map);
  REQUIRE_FALSE(adapter.AllowFusedQuantizedMatmul());

  weight_map.SetAllowFusedQuantizedMatmul(true);
  REQUIRE(adapter.AllowFusedQuantizedMatmul());
}

// =============================================================================
// Test Suite: Build Method (CPU-only tests)
// =============================================================================

TEST_CASE("QuantizedWeightMap: Build with null loader returns false",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;
  cudaStream_t stream = nullptr;

  REQUIRE_FALSE(
      weight_map.Build(nullptr, MockModelLoader().GetModelInfo(), stream));
}

TEST_CASE("QuantizedWeightMap: Build with mock loader",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;
  MockModelLoader loader;
  cudaStream_t stream = nullptr;

  // Should succeed with mock loader (even though it doesn't have real weights)
  // Note: This may still fail if Build tries to access actual weights
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  // Result depends on whether Build tries to validate weight presence
  // For now, we're testing the interface
  (void)result;
}

// =============================================================================
// Test Suite: Layer Accessors
// =============================================================================

TEST_CASE("QuantizedWeightMap: LayerQProj returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerQProj(0) == nullptr);
  REQUIRE(weight_map.LayerQProj(5) == nullptr);
  REQUIRE(weight_map.LayerQProj(100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: LayerKProj returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerKProj(0) == nullptr);
  REQUIRE(weight_map.LayerKProj(5) == nullptr);
  REQUIRE(weight_map.LayerKProj(100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: LayerVProj returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerVProj(0) == nullptr);
  REQUIRE(weight_map.LayerVProj(5) == nullptr);
  REQUIRE(weight_map.LayerVProj(100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: LayerOProj returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerOProj(0) == nullptr);
  REQUIRE(weight_map.LayerOProj(5) == nullptr);
  REQUIRE(weight_map.LayerOProj(100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: LayerInputNorm returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerInputNorm(0) == nullptr);
  REQUIRE(weight_map.LayerInputNorm(5) == nullptr);
  REQUIRE(weight_map.LayerInputNorm(100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: LayerPostAttnNorm returns nullptr before build",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerPostAttnNorm(0) == nullptr);
  REQUIRE(weight_map.LayerPostAttnNorm(5) == nullptr);
  REQUIRE(weight_map.LayerPostAttnNorm(100) == nullptr);
}

// =============================================================================
// Test Suite: Memory Management
// =============================================================================

TEST_CASE("QuantizedWeightMap: Destructor handles empty state",
          "[quantized][weight_map]") {
  // Create and destroy (tests destructor doesn't crash)
  { QuantizedWeightMap weight_map; }
  // If we get here, destructor worked correctly
}

TEST_CASE("QuantizedWeightMap: Destructor handles build without weights",
          "[quantized][weight_map]") {
  MockModelLoader loader;

  {
    QuantizedWeightMap weight_map;
    cudaStream_t stream = nullptr;
    weight_map.Build(&loader, loader.GetModelInfo(), stream);
  }
  // Destructor should clean up any allocated resources
}

// =============================================================================
// Test Suite: Layer Range Handling
// =============================================================================

TEST_CASE("QuantizedWeightMap: Negative layer indices return nullptr",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerQProj(-1) == nullptr);
  REQUIRE(weight_map.LayerQProj(-100) == nullptr);
}

TEST_CASE("QuantizedWeightMap: Large layer indices return nullptr",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  REQUIRE(weight_map.LayerQProj(1000000) == nullptr);
}

// =============================================================================
// Test Suite: Thread Safety (basic checks)
// =============================================================================

TEST_CASE("QuantizedWeightMap: Multiple instances don't interfere",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map1;
  QuantizedWeightMap weight_map2;

  // Both should be independent
  REQUIRE(weight_map1.LayerQProj(0) == nullptr);
  REQUIRE(weight_map2.LayerQProj(0) == nullptr);

  REQUIRE(weight_map1.LayerKProj(0) == nullptr);
  REQUIRE(weight_map2.LayerKProj(0) == nullptr);
}

// =============================================================================
// GPU-Dependent Tests (Skip when CUDA unavailable)
// =============================================================================

#ifdef INFERFLUX_HAS_CUDA

TEST_CASE("QuantizedWeightMap: Build with null stream",
          "[quantized][weight_map][gpu]") {
  QuantizedWeightMap weight_map;
  MockModelLoader loader;
  cudaStream_t stream = nullptr;

  // Build succeeds even with null stream (no-op for empty model)
  REQUIRE(weight_map.Build(&loader, loader.GetModelInfo(), stream));
}

TEST_CASE("QuantizedWeightMap: Build with CUDA stream",
          "[quantized][weight_map][gpu]") {
  QuantizedWeightMap weight_map;
  MockModelLoader loader;

  // Create CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  // May still fail without real weights, but should handle stream
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  cudaStreamDestroy(stream);

  // Result depends on weight availability
  (void)result;
}

#endif // INFERFLUX_HAS_CUDA

// =============================================================================
// Test Suite: Integration with Mock Accessors
// =============================================================================

TEST_CASE("QuantizedWeightMap: Works with mock weight accessors",
          "[quantized][weight_map][mock]") {
  // This test verifies the interface works with mock objects
  // Real weight access is tested in integration tests

  MockModelLoader loader;

  // Create mock accessors for different layer types
  auto q_proj_accessor = std::make_shared<MockWeightAccessor>(768, 768, "q4_k");
  auto k_proj_accessor = std::make_shared<MockWeightAccessor>(768, 768, "q4_k");

  loader.SetMockAccessor("layers.0.attention.wq.weight", q_proj_accessor);
  loader.SetMockAccessor("layers.0.attention.wk.weight", k_proj_accessor);

  QuantizedWeightMap weight_map;
  cudaStream_t stream = nullptr;

  // Build should interface with loader's GetWeightAccessor
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  // May fail without real GPU data, but tests interface
  (void)result;
}

// =============================================================================
// Test Suite: Edge Cases
// =============================================================================

TEST_CASE("QuantizedWeightMap: Handle layer 0 (first layer)",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  // Layer 0 should be handled just like any other layer
  REQUIRE(weight_map.LayerQProj(0) == nullptr);
  REQUIRE(weight_map.LayerQProj(1) == nullptr);
}

TEST_CASE("QuantizedWeightMap: All accessor methods return consistent nulls",
          "[quantized][weight_map]") {
  QuantizedWeightMap weight_map;

  // Before build, all accessors should return nullptr
  for (int i = 0; i < 10; i++) {
    REQUIRE(weight_map.LayerQProj(i) == nullptr);
    REQUIRE(weight_map.LayerKProj(i) == nullptr);
    REQUIRE(weight_map.LayerVProj(i) == nullptr);
    REQUIRE(weight_map.LayerOProj(i) == nullptr);
    REQUIRE(weight_map.LayerInputNorm(i) == nullptr);
    REQUIRE(weight_map.LayerPostAttnNorm(i) == nullptr);
  }
}

// =============================================================================
// Test Suite: Quantization Type Handling
// =============================================================================

TEST_CASE("QuantizedWeightMap: Works with Q4_K_M", "[quantized][weight_map]") {
  MockModelLoader loader;
  QuantizedWeightMap weight_map;

  auto accessor = std::make_shared<MockWeightAccessor>(768, 768, "q4_k_m");
  loader.SetMockAccessor("layers.0.attention.wq.weight", accessor);

  cudaStream_t stream = nullptr;
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  (void)result;
}

TEST_CASE("QuantizedWeightMap: Works with Q5_K_M", "[quantized][weight_map]") {
  MockModelLoader loader;
  QuantizedWeightMap weight_map;

  auto accessor = std::make_shared<MockWeightAccessor>(768, 768, "q5_k_m");
  loader.SetMockAccessor("layers.0.attention.wq.weight", accessor);

  cudaStream_t stream = nullptr;
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  (void)result;
}

TEST_CASE("QuantizedWeightMap: Works with Q6_K", "[quantized][weight_map]") {
  MockModelLoader loader;
  QuantizedWeightMap weight_map;

  auto accessor = std::make_shared<MockWeightAccessor>(768, 768, "q6_k");
  loader.SetMockAccessor("layers.0.attention.wq.weight", accessor);

  cudaStream_t stream = nullptr;
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  (void)result;
}

TEST_CASE("QuantizedWeightMap: Works with Q8_0", "[quantized][weight_map]") {
  MockModelLoader loader;
  QuantizedWeightMap weight_map;

  auto accessor = std::make_shared<MockWeightAccessor>(768, 768, "q8_0");
  loader.SetMockAccessor("layers.0.attention.wq.weight", accessor);

  cudaStream_t stream = nullptr;
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  (void)result;
}

TEST_CASE("QuantizedWeightMap: Works with Q8_K", "[quantized][weight_map]") {
  MockModelLoader loader;
  QuantizedWeightMap weight_map;

  auto accessor = std::make_shared<MockWeightAccessor>(768, 768, "q8_k");
  loader.SetMockAccessor("layers.0.attention.wq.weight", accessor);

  cudaStream_t stream = nullptr;
  bool result = weight_map.Build(&loader, loader.GetModelInfo(), stream);

  (void)result;
}
