#include <catch2/catch_amalgamated.hpp>

#define private public
#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/quantized_weight_map.h"
#undef private

#include <unordered_map>

namespace inferflux {
namespace {

class ScratchMockAccessor : public runtime::cuda::native::IWeightAccessor {
public:
  ScratchMockAccessor(size_t rows, size_t cols, bool quantized)
      : rows_(rows), cols_(cols), quantized_(quantized) {}

  std::pair<size_t, size_t> GetDimensions() const override {
    return {rows_, cols_};
  }

  std::string GetDataType() const override {
    return quantized_ ? "q4_k_m" : "f16";
  }

  bool IsQuantized() const override { return quantized_; }

  void *GetGpuWeights(cudaStream_t) override { return nullptr; }

  half *GetDequantizedGpuWeights(cudaStream_t) override { return nullptr; }

  bool IsDequantizedCached() const override { return false; }

private:
  size_t rows_{0};
  size_t cols_{0};
  bool quantized_{false};
};

class ScratchMockLoader : public runtime::cuda::native::IModelLoader {
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
    return runtime::cuda::native::DequantizedCachePolicy::kNone;
  }

  void ClearDequantizedCache() override {}

  std::shared_ptr<runtime::cuda::native::IWeightAccessor>
  GetWeightAccessor(const std::string &tensor_name) override {
    auto it = accessors_.find(tensor_name);
    if (it == accessors_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void SetModelInfo(const runtime::cuda::native::ModelInfo &info) {
    info_ = info;
  }

  void SetAccessor(
      const std::string &name,
      const std::shared_ptr<runtime::cuda::native::IWeightAccessor> &accessor) {
    accessors_[name] = accessor;
  }

private:
  runtime::cuda::native::ModelInfo info_{};
  std::unordered_map<std::string,
                     std::shared_ptr<runtime::cuda::native::IWeightAccessor>>
      accessors_;
};

TEST_CASE("QuantizedWeightMap scratch sizing ignores quantized lm_head",
          "[quantized][weight_map][scratch]") {
  ScratchMockLoader loader;
  runtime::cuda::native::ModelInfo info{};
  info.num_hidden_layers = 1;
  loader.SetModelInfo(info);

  constexpr size_t kSmallRows = 16;
  constexpr size_t kSmallCols = 16;
  constexpr size_t kLargeRows = 32000;
  constexpr size_t kLargeCols = 4096;

  loader.SetAccessor("model.layers.0.self_attn.q_proj.weight",
                     std::make_shared<ScratchMockAccessor>(kSmallRows,
                                                           kSmallCols,
                                                           /*quantized=*/true));
  loader.SetAccessor("lm_head.weight",
                     std::make_shared<ScratchMockAccessor>(
                         kLargeRows, kLargeCols, /*quantized=*/true));

  QuantizedWeightMap map;
  REQUIRE(map.Build(&loader, info, nullptr));

#ifdef INFERFLUX_HAS_CUDA
  REQUIRE(map.scratch_buffer_elements_ == (kSmallRows * kSmallCols));
#else
  REQUIRE(map.scratch_buffer_elements_ == 0);
#endif
}

TEST_CASE(
    "QuantizedWeightMap scratch stays zero when only lm_head is quantized",
    "[quantized][weight_map][scratch]") {
  ScratchMockLoader loader;
  runtime::cuda::native::ModelInfo info{};
  info.num_hidden_layers = 1;
  loader.SetModelInfo(info);

  loader.SetAccessor("lm_head.weight", std::make_shared<ScratchMockAccessor>(
                                           32000, 4096, /*quantized=*/true));

  QuantizedWeightMap map;
  REQUIRE(map.Build(&loader, info, nullptr));

#ifdef INFERFLUX_HAS_CUDA
  REQUIRE(map.scratch_buffer_elements_ == 0);
#else
  REQUIRE(map.scratch_buffer_elements_ == 0);
#endif
}

TEST_CASE("QuantizedWeightMap ClearCache preserves embedding and final norm "
          "globals",
          "[quantized][weight_map][scratch]") {
  ScratchMockLoader loader;
  runtime::cuda::native::ModelInfo info{};
  info.num_hidden_layers = 0;
  loader.SetModelInfo(info);

  QuantizedWeightMap map;
  REQUIRE(map.Build(&loader, info, nullptr));

  const half *embed = reinterpret_cast<const half *>(0x1110);
  const half *norm = reinterpret_cast<const half *>(0x2220);
  const half *lm_head = reinterpret_cast<const half *>(0x3330);
  map.embed_tokens_ = embed;
  map.final_norm_ = norm;
  map.lm_head_ = lm_head;
  map.embed_tokens_accessor =
      std::make_shared<ScratchMockAccessor>(4, 4, /*quantized=*/true);
  map.final_norm_accessor =
      std::make_shared<ScratchMockAccessor>(1, 4, /*quantized=*/false);
  map.lm_head_accessor =
      std::make_shared<ScratchMockAccessor>(4, 4, /*quantized=*/true);

  map.ClearCache();

  REQUIRE(map.embed_tokens_ == embed);
  REQUIRE(map.final_norm_ == norm);
  REQUIRE(map.lm_head_ == nullptr);
}

TEST_CASE("GGUFModelLoader ClearDequantizedCache retains quantized token "
          "embeddings",
          "[quantized][weight_map][scratch]") {
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping GGUF cache-retention test.");
    return;
  }

  runtime::cuda::native::GGUFModelLoader loader;
  auto &embed = loader.tensors_["token_embd.weight"];
  embed.info.name = "token_embd.weight";
  embed.info.type = runtime::cuda::native::GGUF::TensorType::Q4_K;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&embed.dequantized_gpu),
                     16 * sizeof(half)) == cudaSuccess);

  auto &proj = loader.tensors_["blk.0.attn_q.weight"];
  proj.info.name = "blk.0.attn_q.weight";
  proj.info.type = runtime::cuda::native::GGUF::TensorType::Q4_K;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&proj.dequantized_gpu),
                     16 * sizeof(half)) == cudaSuccess);

  // Dirty flag must be set since we bypassed GGUFWeightAccessor.
  loader.has_dequantized_entries_ = true;
  loader.ClearDequantizedCache();

  REQUIRE(embed.dequantized_gpu != nullptr);
  REQUIRE(proj.dequantized_gpu == nullptr);

  REQUIRE(cudaFree(embed.dequantized_gpu) == cudaSuccess);
  embed.dequantized_gpu = nullptr;
#else
  SUCCEED("CUDA not enabled; skipping GGUF cache-retention test.");
#endif
}

} // namespace
} // namespace inferflux
