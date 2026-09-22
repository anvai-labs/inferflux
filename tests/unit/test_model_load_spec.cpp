#include "model/model_load_spec.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "runtime/backends/llama/llama_device_placement.h"
#include <catch2/catch_amalgamated.hpp>
#include <yaml-cpp/yaml.h>
using namespace inferflux;

TEST_CASE("Model load defaults remain inherited",
          "[model_load_spec][placement]") {
  const auto spec = ParseModelLoadSpec(YAML::Load("path: /a.gguf"));
  REQUIRE_FALSE(spec.HasOverrides());
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
      "device: cpu:0", "device: cuda:0\ngpu_layers: 0");
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
