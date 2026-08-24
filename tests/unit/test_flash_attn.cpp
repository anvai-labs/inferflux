#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "server/metrics/metrics.h"

using namespace inferflux;

// ---------------------------------------------------------------------------
// LlamaBackendConfig defaults
// ---------------------------------------------------------------------------

TEST_CASE("LlamaBackendConfig: use_flash_attention defaults to false",
          "[flash_attn]") {
  LlamaBackendConfig cfg;
  REQUIRE_FALSE(cfg.use_flash_attention);
}

TEST_CASE("LlamaBackendConfig: flash_attention_tile defaults to 128",
          "[flash_attn]") {
  LlamaBackendConfig cfg;
  REQUIRE(cfg.flash_attention_tile == 128);
}

TEST_CASE("LlamaBackendConfig: use_flash_attention can be set to true",
          "[flash_attn]") {
  LlamaBackendConfig cfg;
  cfg.use_flash_attention = true;
  REQUIRE(cfg.use_flash_attention);
}

TEST_CASE("LlamaBackendConfig: flash_attention_tile can be customised",
          "[flash_attn]") {
  LlamaBackendConfig cfg;
  cfg.flash_attention_tile = 256;
  REQUIRE(cfg.flash_attention_tile == 256);
}

// ---------------------------------------------------------------------------
// LlamaCppBackend::FlashAttentionEnabled()
// ---------------------------------------------------------------------------

TEST_CASE(
    "FlashAttentionEnabled() returns false on a freshly constructed backend",
    "[flash_attn]") {
  // config_ is default-initialised (use_flash_attention=false) before LoadModel
  // is called.
  LlamaCppBackend backend;
  REQUIRE_FALSE(backend.FlashAttentionEnabled());
}

// ---------------------------------------------------------------------------
// MetricsRegistry::SetFlashAttentionEnabled gauge
// ---------------------------------------------------------------------------

TEST_CASE("SetFlashAttentionEnabled(true) sets gauge to 1 in Prometheus output",
          "[flash_attn]") {
  MetricsRegistry reg;
  reg.SetFlashAttentionEnabled(true);
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("inferflux_flash_attention_enabled 1") !=
          std::string::npos);
}

TEST_CASE(
    "SetFlashAttentionEnabled(false) sets gauge to 0 in Prometheus output",
    "[flash_attn]") {
  MetricsRegistry reg;
  reg.SetFlashAttentionEnabled(false);
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("inferflux_flash_attention_enabled 0") !=
          std::string::npos);
}

TEST_CASE(
    "SetFlashAttentionEnabled toggle: false -> true -> false is idempotent",
    "[flash_attn]") {
  MetricsRegistry reg;
  reg.SetFlashAttentionEnabled(false);
  reg.SetFlashAttentionEnabled(true);
  reg.SetFlashAttentionEnabled(false);
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("inferflux_flash_attention_enabled 0") !=
          std::string::npos);
}

TEST_CASE("Prometheus output includes HELP and TYPE lines for "
          "flash_attention_enabled",
          "[flash_attn]") {
  MetricsRegistry reg;
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("# HELP inferflux_flash_attention_enabled") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_flash_attention_enabled gauge") !=
          std::string::npos);
}
