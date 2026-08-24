#include <catch2/catch_amalgamated.hpp>

#include "server/metrics/metrics.h"

// Test NVTX macros compile to no-ops when CUDA is disabled
#ifndef INFERFLUX_HAS_CUDA
#include "runtime/backends/cuda/native/nvtx_scoped.h"
#endif

TEST_CASE("Native metrics record forward passes", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordInferfluxCudaForwardPass(/*is_decode=*/true, /*batch_size=*/4,
                                   /*forward_ms=*/12.5);
  registry.RecordInferfluxCudaForwardPass(/*is_decode=*/false, /*batch_size=*/128,
                                   /*forward_ms=*/45.0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_forward_passes_total{phase=\"decode\"} "
                      "1") != std::string::npos);
  REQUIRE(output.find(
              "inferflux_cuda_forward_passes_total{phase=\"prefill\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_forward_batch_tokens_total 132") !=
          std::string::npos);
}

TEST_CASE("Native metrics record forward shape without timing", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordInferfluxCudaForwardShape(/*is_decode=*/true, /*batch_size=*/2);
  registry.RecordInferfluxCudaForwardShape(/*is_decode=*/false, /*batch_size=*/12);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_forward_passes_total{phase=\"decode\"} "
                      "1") != std::string::npos);
  REQUIRE(output.find(
              "inferflux_cuda_forward_passes_total{phase=\"prefill\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_forward_batch_tokens_total 14") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_forward_batch_size_total{phase="
                      "\"decode\",bucket=\"2\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_forward_batch_size_total{phase="
                      "\"prefill\",bucket=\"9_16\"} 1") !=
          std::string::npos);
}

TEST_CASE("Native metrics record sampling", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordInferfluxCudaSampling(/*batch_size=*/4, /*sampling_ms=*/0.5);
  registry.RecordInferfluxCudaSampling(/*batch_size=*/1, /*sampling_ms=*/0.3);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_sampling_duration_ms_count 2") !=
          std::string::npos);
}

TEST_CASE("Native KV cache occupancy gauges", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.SetInferfluxCudaKvCacheOccupancy(/*active=*/5, /*max=*/32);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_kv_active_sequences 5") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_max_sequences 32") !=
          std::string::npos);
}

TEST_CASE("Native KV auto-tune metrics record reduced plan", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordInferfluxCudaKvAutoTunePlan(/*requested_max_seq=*/4096,
                                      /*planned_max_seq=*/1024,
                                      /*requested_bytes=*/4294967296ULL,
                                      /*planned_bytes=*/1073741824ULL,
                                      /*budget_bytes=*/1073741824ULL);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_kv_autotune_events_total 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_requested_max_seq 4096") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_planned_max_seq 1024") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_requested_bytes 4294967296") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_planned_bytes 1073741824") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_budget_bytes 1073741824") !=
          std::string::npos);
}

TEST_CASE("Native KV auto-tune metrics keep event counter stable when plan is "
          "unchanged",
          "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordInferfluxCudaKvAutoTunePlan(/*requested_max_seq=*/2048,
                                      /*planned_max_seq=*/2048,
                                      /*requested_bytes=*/536870912ULL,
                                      /*planned_bytes=*/536870912ULL,
                                      /*budget_bytes=*/1073741824ULL);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_kv_autotune_events_total 0") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_requested_max_seq 2048") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_kv_planned_max_seq 2048") !=
          std::string::npos);
}

TEST_CASE("Native forward pass histogram buckets", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  // Record a 5ms forward (should go in 10ms bucket)
  registry.RecordInferfluxCudaForwardPass(true, 1, 5.0);

  auto output = registry.RenderPrometheus();
  // Should be in the 10ms bucket
  REQUIRE(output.find("inferflux_cuda_forward_duration_ms_bucket{le=\"10\"} "
                      "1") != std::string::npos);
  // And in +Inf
  REQUIRE(output.find(
              "inferflux_cuda_forward_duration_ms_bucket{le=\"+Inf\"} 1") !=
          std::string::npos);
}

#ifndef INFERFLUX_HAS_CUDA
TEST_CASE("NVTX macros are no-ops without CUDA", "[native_batch]") {
  // Just verify compilation succeeds
  NVTX_SCOPE("TestScope");
  REQUIRE(true);
}
#endif
