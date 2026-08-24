#include <catch2/catch_amalgamated.hpp>

#include "server/startup_advisor.h"

#include <cstdlib>

#ifdef _WIN32
inline void portable_setenv(const char *name, const char *value) {
  _putenv_s(name, value);
}
inline void portable_unsetenv(const char *name) {
  _putenv_s(name, "");
}
#else
inline void portable_setenv(const char *name, const char *value) {
  setenv(name, value, 1);
}
inline void portable_unsetenv(const char *name) {
  unsetenv(name);
}
#endif

// Helper to build a well-tuned CUDA context (zero recommendations expected).
static inferflux::StartupAdvisorContext WellTunedCudaContext() {
  inferflux::StartupAdvisorContext ctx;
  ctx.gpu.available = true;
  ctx.gpu.device_count = 1;
  ctx.gpu.device_name = "NVIDIA RTX 4090";
  ctx.gpu.compute_major = 8;
  ctx.gpu.compute_minor = 9;
  ctx.gpu.sm_count = 128;
  ctx.gpu.total_vram_bytes = 24ULL * 1024 * 1024 * 1024;
  ctx.gpu.free_vram_bytes = 2ULL * 1024 * 1024 * 1024; // tight VRAM

  ctx.config.cuda_enabled = true;
  ctx.config.flash_attention_enabled = true;
  ctx.config.cuda_attention_kernel = "fa2";
  ctx.config.phase_overlap_enabled = true;
  ctx.config.max_batch_size = 16;
  ctx.config.max_batch_tokens = 4096;
  ctx.config.kv_cpu_pages = 128;
  ctx.config.prefer_inferflux = true;
  ctx.config.tp_degree = 1;

  inferflux::AdvisorModelInfo m;
  m.id = "llama3-8b";
  m.path = "/models/llama3-8b-q4_k_m.gguf"; // Include quantization to avoid
                                            // recommendation
  m.format = "gguf";
  m.backend = "cuda";
  m.backend_provider = "inferflux";
  m.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
  m.quantization =
      inferflux::DetectQuantization(m.path, m.format); // Detect from filename
  ctx.models.push_back(m);

  return ctx;
}

TEST_CASE("Well-tuned CUDA config produces 0 recommendations",
          "[startup_advisor]") {
  auto ctx = WellTunedCudaContext();
  int count = inferflux::RunStartupAdvisor(ctx);
  REQUIRE(count == 0);
}

TEST_CASE("FA disabled on Ampere GPU triggers attention recommendation",
          "[startup_advisor]") {
  auto ctx = WellTunedCudaContext();
  ctx.config.flash_attention_enabled = false;
  ctx.config.cuda_attention_kernel = "auto";

  int count = inferflux::RunStartupAdvisor(ctx);
  REQUIRE(count >= 1);
}

TEST_CASE("GPU available with all models on CPU triggers gpu_unused",
          "[startup_advisor]") {
  inferflux::StartupAdvisorContext ctx;
  ctx.gpu.available = true;
  ctx.gpu.device_count = 1;
  ctx.gpu.device_name = "NVIDIA A100";
  ctx.gpu.compute_major = 8;
  ctx.gpu.compute_minor = 0;
  ctx.gpu.total_vram_bytes = 40ULL * 1024 * 1024 * 1024;
  ctx.gpu.free_vram_bytes = 38ULL * 1024 * 1024 * 1024;

  ctx.config.cuda_enabled = false;

  inferflux::AdvisorModelInfo m;
  m.id = "llama3-8b";
  m.format = "gguf";
  m.backend = "cpu";
  m.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
  ctx.models.push_back(m);

  int count = inferflux::RunStartupAdvisor(ctx);
  // Should at least get the gpu_unused recommendation.
  REQUIRE(count >= 1);
}

TEST_CASE("Multi-GPU with TP=1 and large model triggers tensor_parallel",
          "[startup_advisor]") {
  inferflux::StartupAdvisorContext ctx;
  ctx.gpu.available = true;
  ctx.gpu.device_count = 4;
  ctx.gpu.device_name = "NVIDIA A100";
  ctx.gpu.compute_major = 8;
  ctx.gpu.compute_minor = 0;
  ctx.gpu.total_vram_bytes = 40ULL * 1024 * 1024 * 1024;
  ctx.gpu.free_vram_bytes = 10ULL * 1024 * 1024 * 1024;

  ctx.config.cuda_enabled = true;
  ctx.config.flash_attention_enabled = true;
  ctx.config.phase_overlap_enabled = true;
  ctx.config.max_batch_size = 8;
  ctx.config.kv_cpu_pages = 128;
  ctx.config.tp_degree = 1;

  inferflux::AdvisorModelInfo m;
  m.id = "llama-70b";
  m.format = "gguf";
  m.backend = "cuda";
  m.backend_provider = "inferflux";
  // Model uses > 70% of single-GPU VRAM.
  m.file_size_bytes = 35ULL * 1024 * 1024 * 1024;
  ctx.models.push_back(m);

  int count = inferflux::RunStartupAdvisor(ctx);
  REQUIRE(count >= 1);
}

TEST_CASE("No models loaded produces 0 recommendations", "[startup_advisor]") {
  inferflux::StartupAdvisorContext ctx;
  ctx.gpu.available = true;
  ctx.gpu.device_count = 1;
  ctx.gpu.device_name = "NVIDIA RTX 4090";
  ctx.gpu.compute_major = 8;
  ctx.gpu.compute_minor = 9;
  ctx.gpu.total_vram_bytes = 24ULL * 1024 * 1024 * 1024;
  ctx.gpu.free_vram_bytes = 20ULL * 1024 * 1024 * 1024;
  ctx.config.cuda_enabled = true;
  // No models.
  int count = inferflux::RunStartupAdvisor(ctx);
  REQUIRE(count == 0);
}

TEST_CASE("Suppression env var disables all recommendations",
          "[startup_advisor]") {
  auto ctx = WellTunedCudaContext();
  // Make a suboptimal config that would normally produce recommendations.
  ctx.config.flash_attention_enabled = false;
  ctx.config.phase_overlap_enabled = false;

  // Set suppression env var.
  portable_setenv("INFERFLUX_DISABLE_STARTUP_ADVISOR", "true");
  int count = inferflux::RunStartupAdvisor(ctx);
  REQUIRE(count == 0);
  // Clean up.
  portable_unsetenv("INFERFLUX_DISABLE_STARTUP_ADVISOR");
}
