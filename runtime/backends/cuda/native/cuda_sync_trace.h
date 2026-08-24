#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif

namespace inferflux::runtime::cuda::native {

enum class SyncTraceSite : size_t {
  kSamplerResultReady = 0,
  kSamplerBatchResultReady,
  kSamplerLogitsReady,
  kLaneExecutionDrain,
  kPrefillForwardDrain,
  kTimingForwardReady,
  kTimingSamplingReady,
  kBurstChunkReady,
  kCount,
};

struct SyncTraceEntry {
  const char *label;
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> api_ns{0};
};

inline std::array<SyncTraceEntry, static_cast<size_t>(SyncTraceSite::kCount)>
    kSyncTraceEntries = {{
        {"sampler.result_ready"},
        {"sampler.batch_result_ready"},
        {"sampler.logits_ready"},
        {"executor.lane_execution_drain"},
        {"executor.prefill_forward_drain"},
        {"executor.timing_forward_ready"},
        {"executor.timing_sampling_ready"},
        {"burst.chunk_ready"},
    }};

inline bool SyncTraceEnabled() {
  static const bool enabled = []() {
    const char *raw = std::getenv("INFERFLUX_CUDA_SYNC_TRACE");
    return raw && std::string(raw) != "0" && std::string(raw) != "false";
  }();
  return enabled;
}

inline void LogSyncTraceSummaryAtExit() {
  if (!SyncTraceEnabled()) {
    return;
  }
  std::fprintf(stderr, "[native_sync_trace] summary begin\n");
  for (const auto &entry : kSyncTraceEntries) {
    const uint64_t calls = entry.calls.load(std::memory_order_relaxed);
    if (calls == 0) {
      continue;
    }
    const uint64_t api_ns = entry.api_ns.load(std::memory_order_relaxed);
    const double avg_us = calls > 0 ? static_cast<double>(api_ns) /
                                          static_cast<double>(calls) / 1000.0
                                    : 0.0;
    std::fprintf(
        stderr,
        "[native_sync_trace] site=%s calls=%llu api_ns=%llu avg_us=%.3f\n",
        entry.label, static_cast<unsigned long long>(calls),
        static_cast<unsigned long long>(api_ns), avg_us);
  }
  std::fprintf(stderr, "[native_sync_trace] summary end\n");
}

inline void EnsureSyncTraceRegistered() {
  static std::once_flag once;
  if (!SyncTraceEnabled()) {
    return;
  }
  std::call_once(once, []() { std::atexit(LogSyncTraceSummaryAtExit); });
}

inline cudaError_t TracedCudaEventSynchronize(SyncTraceSite site,
                                              cudaEvent_t event) {
  if (!SyncTraceEnabled()) {
    return cudaEventSynchronize(event);
  }
  EnsureSyncTraceRegistered();
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t err = cudaEventSynchronize(event);
  const auto end = std::chrono::steady_clock::now();
  auto &entry = kSyncTraceEntries[static_cast<size_t>(site)];
  entry.calls.fetch_add(1, std::memory_order_relaxed);
  entry.api_ns.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()),
      std::memory_order_relaxed);
  return err;
}

inline cudaError_t TracedCudaStreamSynchronize(SyncTraceSite site,
                                               cudaStream_t stream) {
  if (!SyncTraceEnabled()) {
    return cudaStreamSynchronize(stream);
  }
  EnsureSyncTraceRegistered();
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t err = cudaStreamSynchronize(stream);
  const auto end = std::chrono::steady_clock::now();
  auto &entry = kSyncTraceEntries[static_cast<size_t>(site)];
  entry.calls.fetch_add(1, std::memory_order_relaxed);
  entry.api_ns.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()),
      std::memory_order_relaxed);
  return err;
}

} // namespace inferflux::runtime::cuda::native
