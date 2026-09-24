#include "external/llama.cpp/src/llama-model.h"
#include "model/model_load_spec.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "runtime/backends/llama/llama_device_placement.h"
#include <catch2/catch_amalgamated.hpp>
#include <cstdlib>
#include <memory>
#include <yaml-cpp/yaml.h>
using namespace inferflux;

TEST_CASE("Model load defaults remain inherited",
          "[model_load_spec][placement]") {
  const auto spec = ParseModelLoadSpec(YAML::Load("path: /a.gguf"));
  REQUIRE_FALSE(spec.HasOverrides());
  REQUIRE_FALSE(spec.Apply({}).embedding_batch_size);
  REQUIRE(EmbeddingBatchGeometry::Resolve({})->max_sequences == 32);
  REQUIRE(EmbeddingBatchGeometry::Resolve({})->max_batch_tokens == 16384);
  LlamaBackendConfig defaults;
  defaults.ctx_size = 8192;
  defaults.gpu_layers = 8;
  defaults.max_parallel_sequences = 2;
  const auto config = spec.Apply(defaults);
  REQUIRE(config.ctx_size == 8192);
  REQUIRE(config.gpu_layers == 8);
  REQUIRE(config.max_parallel_sequences == 2);
}
TEST_CASE("Shared model parser preserves explicit resource overrides",
          "[model_load_spec][placement]") {
  const auto input = GENERATE(
      "{\"path\":\"/"
      "a.gguf\",\"device\":\"rocm:1\",\"context_size\":4096,\"gpu_layers\":8,"
      "\"max_parallel_sequences\":2,\"kv_cache_type\":\"q8_0\"}",
      "path: /a.gguf\ndevice: rocm:1\ncontext_size: 4096\ngpu_layers: "
      "8\nmax_parallel_sequences: 2\nkv_cache_type: q8_0");
  const auto spec = ParseModelLoadSpec(YAML::Load(input));
  const auto config = spec.Apply({});
  REQUIRE(config.device == "rocm:1");
  REQUIRE(config.ctx_size == 4096);
  REQUIRE(config.gpu_layers == 8);
  REQUIRE(config.max_parallel_sequences == 2);
  REQUIRE(config.llama_kv_cache_type == "q8_0");
}
TEST_CASE("Model selectors reject ambiguous and malformed devices",
          "[model_load_spec][placement]") {
  const auto value = GENERATE("0", "cuda:-1", "cuda:", "rocm:2147483648",
                              "hip:0", "cuda:0junk", "rocm: 0", "cuda:+1");
  REQUIRE_FALSE(ParseDeviceSelector(value));
}
TEST_CASE("Model resource validation fails closed",
          "[model_load_spec][placement]") {
  const auto value = GENERATE(
      "context_size: 0", "gpu_layers: -2", "max_parallel_sequences: 257",
      "max_parallel_sequences: 0", "kv_cache_type: bogus", "device_id: 0",
      "device: cpu:0", "device: cuda:0\ngpu_layers: 0",
      "embedding_batch_size: 0", "embedding_batch_size: 33",
      "embedding_batch_size: -1", "embedding_batch_size: 1.5");
  REQUIRE_THROWS(ParseModelLoadSpec(YAML::Load(value)));
}

TEST_CASE("Explicit zero offload and KV overrides survive backend tuning",
          "[model_load_spec][placement]") {
  ModelLoadSpec spec;
  spec.gpu_layers = 0;
  spec.kv_cache_type = "q8_0";
  const auto config =
      TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, spec.Apply({}));
  REQUIRE(config.gpu_layers == 0);
  REQUIRE(config.llama_kv_cache_type == "q8_0");
  // Loading rejects quantized KV without FA instead of silently changing dtype.
  REQUIRE_FALSE(config.use_flash_attention);
}
TEST_CASE("Unavailable vendor-qualified devices never resolve to CPU",
          "[model_load_spec][placement]") {
  const auto value = GENERATE("cuda:2147483647", "rocm:2147483647", "cpu:0");
  std::string error;
  REQUIRE(ResolveLlamaDevice(value, &error) == nullptr);
  REQUIRE(error.find("placement_") == 0);
}

TEST_CASE("Pinned BERT decode API is not a text generation capability",
          "[placement][embeddings_admission]") {
  std::unique_ptr<llama_model> bert(
      llama_model_create(LLM_ARCH_BERT, llama_model_default_params()));
  REQUIRE(bert);
  bert->arch = LLM_ARCH_BERT;
  bert->hparams.causal_attn = false;
  REQUIRE(llama_model_has_decoder(bert.get()));
  REQUIRE_FALSE(LlamaModelSupportsGeneration(bert.get()));
  std::unique_ptr<llama_model> qwen(
      llama_model_create(LLM_ARCH_QWEN2, llama_model_default_params()));
  REQUIRE(qwen);
  qwen->arch = LLM_ARCH_QWEN2;
  qwen->hparams.causal_attn = true;
  REQUIRE(LlamaModelSupportsGeneration(qwen.get()));
  REQUIRE_FALSE(LlamaModelSupportsGeneration(nullptr));
}

TEST_CASE("Embedding batch geometry is explicit immutable and independent",
          "[model_load_spec][embedding_geometry]") {
  const int size = GENERATE(1, 2, 32);
  const bool json = GENERATE(false, true);
  const auto input =
      json ? "{\"embedding_batch_size\":" + std::to_string(size) + "}"
           : "embedding_batch_size: " + std::to_string(size);
  const auto spec = ParseModelLoadSpec(YAML::Load(input));
  REQUIRE(spec.HasOverrides());
  const auto config =
      TuneLlamaBackendConfig(LlamaBackendTarget::kRocm, spec.Apply({}));
  REQUIRE(config.embedding_batch_size == size);
  REQUIRE(config.ctx_size == LlamaBackendConfig{}.ctx_size);
  REQUIRE(config.max_parallel_sequences ==
          LlamaBackendConfig{}.max_parallel_sequences);
  const auto geometry =
      EmbeddingBatchGeometry::Resolve(config.embedding_batch_size);
  REQUIRE(geometry);
  REQUIRE(geometry->max_sequences == size);
  REQUIRE(geometry->max_batch_tokens == size * 512);
  auto changed = spec;
  changed.embedding_batch_size = size == 1 ? 2 : 1;
  REQUIRE(spec != changed);
  REQUIRE(ParseModelLoadSpec(YAML::Load("path: /a.gguf"))
              .Apply(config)
              .embedding_batch_size == size);
}

TEST_CASE("Invalid direct embedding geometry fails before model or device load",
          "[model_load_spec][embedding_geometry]") {
  const int size = GENERATE(-1, 0, 33);
  REQUIRE_FALSE(EmbeddingBatchGeometry::Resolve(size));
  LlamaBackendConfig config;
  config.embedding_batch_size = size;
  LlamaCppBackend backend;
  REQUIRE_FALSE(backend.LoadModel("/must-not-load.gguf", config));
  REQUIRE(backend.LoadError() ==
          "embedding_batch_size must be between 1 and 32");
}

TEST_CASE("Explicit embedding geometry survives rejected backend reloads",
          "[model_load_spec][embedding_native]") {
  const char *model = std::getenv("INFERFLUX_TEST_EMBEDDING_MODEL");
  if (!model || !*model)
    SKIP("Optional real encoder asset required; run on CPU before acceptance");
  LlamaCppBackend backend;
  LlamaBackendConfig config;
  config.ctx_size = 512;
  config.max_parallel_sequences = 1;
  config.gpu_layers = 0;
  config.embedding_batch_size = 1;
  REQUIRE(backend.LoadModel(model, config));
  const std::vector<std::string> inputs{"A red apple.", "A blue bicycle.",
                                        "A green tree."};
  const auto original = backend.EmbedBatch(inputs);
  REQUIRE(original.size() == inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    REQUIRE(original[i].size() == 384);
    const auto single = backend.EmbedBatch({inputs[i]});
    REQUIRE(single[0] == original[i]);
  }
  const bool absent = GENERATE(false, true);
  auto changed = config;
  changed.embedding_batch_size =
      absent ? std::optional<int>{} : std::optional<int>{32};
  REQUIRE_FALSE(backend.LoadModel("/must-not-load.gguf", changed));
  REQUIRE(backend.LoadError() ==
          "embedding geometry reload requires a fresh backend");
  REQUIRE(backend.IsReady());
  REQUIRE(backend.Placement().embedding_batch->max_sequences == 1);
  REQUIRE(backend.EmbedBatch(inputs) == original);
}
