#include <catch2/catch_amalgamated.hpp>
#include <memory>

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/llama/llama_cpp_backend.h" // For LlamaBackendConfig

using namespace inferflux;

// Mock BackendInterface implementation for testing
class MockBackend : public BackendInterface {
public:
  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override {
    loaded_ = true;
    model_path_ = model_path;
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    std::vector<UnifiedBatchOutput> outputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      outputs[i].ok = true;
      outputs[i].token = 42;
      outputs[i].piece = "mock";
    }
    return outputs;
  }

  int UnifiedBatchTokenCapacity() const override { return capacity_; }

  bool SupportsAsyncUnifiedBatch() const override { return supports_async_; }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    if (!supports_async_) {
      return 0; // Not supported
    }
    return next_handle_++;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    return false; // Not implemented in mock
  }

  std::string Name() const override { return "mock_backend"; }

  bool IsFallback() const override { return is_fallback_; }

  const std::string &FallbackReason() const override {
    return fallback_reason_;
  }

  // Mock setters
  void SetCapacity(int capacity) { capacity_ = capacity; }
  void SetSupportsAsync(bool supports) { supports_async_ = supports; }
  void SetFallback(bool fallback, const std::string &reason = "") {
    is_fallback_ = fallback;
    fallback_reason_ = reason;
  }

private:
  bool loaded_{false};
  std::filesystem::path model_path_;
  int capacity_{2048};
  bool supports_async_{false};
  bool is_fallback_{false};
  std::string fallback_reason_;
  UnifiedBatchHandle next_handle_{1};
};

TEST_CASE("BackendInterface can be mocked", "[backend_interface]") {
  MockBackend backend;

  REQUIRE(backend.Name() == "mock_backend");
  REQUIRE(backend.IsFallback() == false);
  REQUIRE(backend.FallbackReason().empty());
}

TEST_CASE("BackendInterface LoadModel implementation", "[backend_interface]") {
  MockBackend backend;
  LlamaBackendConfig config;
  config.ctx_size = 2048;

  bool result = backend.LoadModel("/path/to/model.gguf", config);

  REQUIRE(result == true);
}

TEST_CASE("BackendInterface ExecuteUnifiedBatch implementation",
          "[backend_interface]") {
  MockBackend backend;

  std::vector<UnifiedBatchInput> inputs(2);
  inputs[0].tokens = {1, 2};
  inputs[1].tokens = {3};

  auto outputs = backend.ExecuteUnifiedBatch(inputs);

  REQUIRE(outputs.size() == 2);
  REQUIRE(outputs[0].ok == true);
  REQUIRE(outputs[0].token == 42);
  REQUIRE(outputs[0].piece == "mock");
  REQUIRE(outputs[1].ok == true);
  REQUIRE(outputs[1].token == 42);
  REQUIRE(outputs[1].piece == "mock");
}

TEST_CASE("BackendInterface handles empty batch", "[backend_interface]") {
  MockBackend backend;

  std::vector<UnifiedBatchInput> inputs;
  auto outputs = backend.ExecuteUnifiedBatch(inputs);

  REQUIRE(outputs.size() == 0);
}

TEST_CASE("BackendInterface UnifiedBatchTokenCapacity default",
          "[backend_interface]") {
  MockBackend backend;

  REQUIRE(backend.UnifiedBatchTokenCapacity() == 2048);
}

TEST_CASE("BackendInterface UnifiedBatchTokenCapacity custom",
          "[backend_interface]") {
  MockBackend backend;
  backend.SetCapacity(4096);

  REQUIRE(backend.UnifiedBatchTokenCapacity() == 4096);
}

TEST_CASE("BackendInterface async methods default implementation",
          "[backend_interface]") {
  MockBackend backend;

  REQUIRE(backend.SupportsAsyncUnifiedBatch() == false);

  std::vector<UnifiedBatchInput> inputs;
  auto handle =
      backend.SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kAuto);
  REQUIRE(handle == 0); // Default returns 0 (not supported)
}

TEST_CASE("BackendInterface async methods with support",
          "[backend_interface]") {
  MockBackend backend;
  backend.SetSupportsAsync(true);

  REQUIRE(backend.SupportsAsyncUnifiedBatch() == true);

  std::vector<UnifiedBatchInput> inputs(1);
  inputs[0].tokens = {1, 2};
  auto handle =
      backend.SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kPrefill);
  REQUIRE(handle == 1); // Should return non-zero handle
}

TEST_CASE("BackendInterface fallback metadata", "[backend_interface]") {
  MockBackend backend;

  REQUIRE(backend.IsFallback() == false);
  REQUIRE(backend.FallbackReason().empty());

  backend.SetFallback(true, "test reason");

  REQUIRE(backend.IsFallback() == true);
  REQUIRE(backend.FallbackReason() == "test reason");
}

TEST_CASE("BackendInterface is polymorphic", "[backend_interface]") {
  MockBackend backend;
  BackendInterface *interface = &backend;

  // Test that we can call methods through interface pointer
  REQUIRE(interface->Name() == "mock_backend");
  REQUIRE(interface->IsFallback() == false);

  std::vector<UnifiedBatchInput> inputs;
  auto outputs = interface->ExecuteUnifiedBatch(inputs);
  REQUIRE(outputs.size() == 0);
}

TEST_CASE("UnifiedBatchLane enum works through interface",
          "[backend_interface]") {
  MockBackend backend;

  // Test all lane types compile and work
  std::vector<UnifiedBatchInput> inputs;
  backend.SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kAuto);
  backend.SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kPrefill);
  backend.SubmitUnifiedBatchAsync(inputs, UnifiedBatchLane::kDecode);

  // If this compiles and runs, the enum works correctly
  REQUIRE(true);
}
