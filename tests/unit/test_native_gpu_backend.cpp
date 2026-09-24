#include <catch2/catch_amalgamated.hpp>

#include "model/tokenizer.h"
#include "runtime/backends/gpu/gpu_device_strategy.h"
#include "runtime/backends/native/native_gpu_backend.h"
#include "tests/support/scoped_env.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace inferflux {
namespace {

class FakeTokenizer final : public ITokenizer {
public:
  bool Load(const std::string &model_path) override {
    loaded_ = !model_path.empty();
    return loaded_;
  }

  std::vector<int> Tokenize(const std::string &text,
                            bool add_bos = true) const override {
    (void)add_bos;
    std::vector<int> tokens;
    for (char ch : text) {
      tokens.push_back(static_cast<int>(ch));
    }
    return tokens;
  }

  std::string Detokenize(const std::vector<int> &tokens) const override {
    std::string out;
    for (int token : tokens) {
      out += TokenToString(token);
    }
    return out;
  }

  std::string TokenToString(int token_id) const override {
    if (token_id == 10) {
      return "A";
    }
    if (token_id == 11) {
      return "B";
    }
    if (token_id == 12) {
      return "C";
    }
    if (token_id == 13) {
      return "D";
    }
    return {};
  }

  ChatResult ApplyChatTemplate(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const override {
    (void)messages;
    (void)add_assistant_prefix;
    return {};
  }

  int BosTokenId() const override { return 1; }
  int EosTokenId() const override { return 2; }
  int VocabSize() const override { return 32; }
  bool IsLoaded() const override { return loaded_; }

private:
  bool loaded_{true};
};

class FakeGpuDeviceStrategy final : public GpuDeviceStrategy {
public:
  bool Initialize() override { return true; }
  bool IsAvailable() const override { return true; }
  GpuDeviceInfo GetDeviceInfo() const override {
    return {"fake-gpu", "sm_fake", 1024, 0, true, "fa2"};
  }
  LlamaBackendTarget Target() const override {
    return LlamaBackendTarget::kCuda;
  }
  void RecordMetrics(const LlamaBackendConfig &config) override {
    (void)config;
  }
};

class FakeNativeRuntime final : public NativeInferenceRuntime {
public:
  explicit FakeNativeRuntime(FakeTokenizer *tokenizer)
      : tokenizer_(tokenizer) {}

  std::string Name() const override { return "inferflux_cuda"; }
  bool IsFallback() const override { return false; }
  int NativeSequenceCapacity() const override { return 2; }
  int NativeSequenceContextCapacity() const override { return 32768; }
  const std::string &FallbackReason() const override {
    return fallback_reason_;
  }

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override {
    (void)model_path;
    (void)config;
    ready_ = true;
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    ++execute_calls_;
    std::vector<UnifiedBatchOutput> outputs(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      outputs[i].ok = true;
      if (standard_index_ < static_cast<int>(standard_tokens_.size())) {
        outputs[i].token =
            standard_tokens_[static_cast<std::size_t>(standard_index_++)];
        outputs[i].piece =
            tokenizer_ ? tokenizer_->TokenToString(outputs[i].token) : "";
      } else {
        outputs[i].token = -1;
      }
    }
    return outputs;
  }

  bool SupportsAsyncUnifiedBatch() const override { return false; }
  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    (void)inputs;
    (void)lane;
    return 0;
  }
  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    (void)handle;
    (void)outputs;
    return false;
  }

  std::shared_ptr<BackendInterface> BackendHandle() const override {
    return nullptr;
  }

  bool NativeIsReady() const override { return ready_; }
  const ITokenizer *NativeGetTokenizer() const override { return tokenizer_; }
  int NativeVocabSize() const override {
    return tokenizer_ ? tokenizer_->VocabSize() : 0;
  }

  int BurstDecodeGreedy(int sequence_id, int n_past_start, int first_token_id,
                        int n_tokens, int eos_token_id,
                        std::vector<int> *out_tokens) override {
    (void)sequence_id;
    (void)n_past_start;
    (void)first_token_id;
    (void)eos_token_id;
    ++burst_calls_;
    out_tokens->clear();
    if (!burst_enabled_) {
      return 0;
    }
    const int remaining = static_cast<int>(burst_tokens_.size()) - burst_index_;
    const int take = std::max(0, std::min(n_tokens, remaining));
    for (int i = 0; i < take; ++i) {
      out_tokens->push_back(
          burst_tokens_[static_cast<std::size_t>(burst_index_++)]);
    }
    return take;
  }

  void SetStandardTokens(std::vector<int> tokens) {
    standard_tokens_ = std::move(tokens);
    standard_index_ = 0;
  }

  void SetBurstTokens(std::vector<int> tokens, bool enabled = true) {
    burst_tokens_ = std::move(tokens);
    burst_index_ = 0;
    burst_enabled_ = enabled;
  }

  int execute_calls() const { return execute_calls_; }
  int burst_calls() const { return burst_calls_; }

private:
  FakeTokenizer *tokenizer_{nullptr};
  std::string fallback_reason_;
  bool ready_{false};
  bool burst_enabled_{true};
  std::vector<int> standard_tokens_;
  std::vector<int> burst_tokens_;
  int standard_index_{0};
  int burst_index_{0};
  int execute_calls_{0};
  int burst_calls_{0};
};

class TestNativeGpuBackend final : public NativeGpuBackend {
public:
  explicit TestNativeGpuBackend(std::unique_ptr<FakeNativeRuntime> runtime)
      : NativeGpuBackend(std::make_unique<FakeGpuDeviceStrategy>()),
        pending_runtime_(std::move(runtime)),
        runtime_raw_(pending_runtime_.get()) {}

  std::unique_ptr<NativeInferenceRuntime> CreateNativeRuntime() override {
    return std::move(pending_runtime_);
  }

  FakeNativeRuntime *runtime() const { return runtime_raw_; }

private:
  std::unique_ptr<FakeNativeRuntime> pending_runtime_;
  FakeNativeRuntime *runtime_raw_{nullptr};
};

TEST_CASE(
    "Native loaders reject explicit embedding geometry before initialization",
    "[native_gpu_backend][embedding_geometry]") {
  FakeTokenizer tokenizer;
  auto runtime = std::make_unique<FakeNativeRuntime>(&tokenizer);
  auto *raw = runtime.get();
  TestNativeGpuBackend backend(std::move(runtime));
  LlamaBackendConfig config;
  config.embedding_batch_size = 1;
  REQUIRE_FALSE(backend.LoadModel("fake.gguf", config));
  REQUIRE_FALSE(raw->IsReady());
}

TEST_CASE("Invalid embedding geometry does not initialize a GPU strategy",
          "[native_gpu_backend][embedding_geometry]") {
  class CountingStrategy : public GpuDeviceStrategy {
  public:
    int calls{0};
    bool Initialize() override {
      ++calls;
      return true;
    }
    bool IsAvailable() const override { return true; }
    GpuDeviceInfo GetDeviceInfo() const override { return {}; }
    LlamaBackendTarget Target() const override {
      return LlamaBackendTarget::kCuda;
    }
    void RecordMetrics(const LlamaBackendConfig &) override {}
  };
  auto strategy = std::make_unique<CountingStrategy>();
  auto *raw = strategy.get();
  GpuAcceleratedBackend backend(std::move(strategy));
  LlamaBackendConfig config;
  config.embedding_batch_size = 0;
  REQUIRE_FALSE(backend.LoadModel("/must-not-load.gguf", config));
  REQUIRE(raw->calls == 0);
}

TEST_CASE("NativeGpuBackend Decode uses burst path for eligible greedy decode",
          "[native_gpu_backend]") {
  test::ScopedEnvVar disable_parity("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE",
                                    "1");
  test::ScopedEnvVar burst_tokens("INFERFLUX_NATIVE_BURST_CHUNK_TOKENS", "4");

  FakeTokenizer tokenizer;
  auto runtime = std::make_unique<FakeNativeRuntime>(&tokenizer);
  runtime->SetBurstTokens({11, 12});
  auto *runtime_raw = runtime.get();
  TestNativeGpuBackend backend(std::move(runtime));

  REQUIRE(backend.LoadModel("fake.gguf", {}));
  REQUIRE(backend.SequenceCapacity() == 2);
  REQUIRE(backend.SequenceContextCapacity() == 32768);

  SamplingParams greedy;
  greedy.temperature = 0.0f;
  backend.SetupSampler("", "root", greedy);

  const std::string output =
      backend.Decode(/*n_past=*/1, /*sequence_id=*/7, /*max_tokens=*/3, {}, {},
                     /*logprob_top_n=*/0, nullptr,
                     /*first_token=*/10, {});

  REQUIRE(output == "ABC");
  REQUIRE(runtime_raw->burst_calls() == 1);
  REQUIRE(runtime_raw->execute_calls() == 0);
}

TEST_CASE(
    "NativeGpuBackend Decode keeps streaming requests on standard token path",
    "[native_gpu_backend]") {
  test::ScopedEnvVar disable_parity("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE",
                                    "1");
  test::ScopedEnvVar burst_tokens("INFERFLUX_NATIVE_BURST_CHUNK_TOKENS", "4");

  FakeTokenizer tokenizer;
  auto runtime = std::make_unique<FakeNativeRuntime>(&tokenizer);
  runtime->SetStandardTokens({11, 12});
  runtime->SetBurstTokens({11, 12});
  auto *runtime_raw = runtime.get();
  TestNativeGpuBackend backend(std::move(runtime));

  REQUIRE(backend.LoadModel("fake.gguf", {}));

  SamplingParams greedy;
  greedy.temperature = 0.0f;
  backend.SetupSampler("", "root", greedy);

  std::string streamed;
  const std::string output = backend.Decode(
      /*n_past=*/1, /*sequence_id=*/7, /*max_tokens=*/3,
      [&streamed](const std::string &chunk, const TokenLogprob *logprob) {
        (void)logprob;
        streamed += chunk;
        return true;
      },
      {}, /*logprob_top_n=*/0, nullptr, /*first_token=*/10, {});

  REQUIRE(output == "ABC");
  REQUIRE(streamed == "ABC");
  REQUIRE(runtime_raw->burst_calls() == 0);
  REQUIRE(runtime_raw->execute_calls() == 2);
}

} // namespace
} // namespace inferflux
