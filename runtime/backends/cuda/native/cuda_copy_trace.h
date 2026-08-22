#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

namespace inferflux::runtime::cuda::native {

enum class CopyTraceSite : size_t {
  kForwardTokenIdsH2D = 0,
  kForwardResidualD2D,
  kBatchMetaH2D,
  kBatchResidualD2D,
  kSamplerGreedyResultD2H,
  kSamplerBatchResultD2H,
  kSamplerLogitsToProbsD2D,
  kSamplerTempToProbsD2D,
  kSamplerCopyLogitsD2H,
  kBurstSlotD2H,
  kCount,
};

struct CopyTraceEntry {
  const char *label;
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> api_ns{0};
};

inline std::array<CopyTraceEntry, static_cast<size_t>(CopyTraceSite::kCount)>
    kCopyTraceEntries = {{
        {"forward.token_ids_h2d"},
        {"forward.residual_d2d"},
        {"batch.meta_h2d"},
        {"batch.residual_d2d"},
        {"sampler.greedy_result_d2h"},
        {"sampler.batch_result_d2h"},
        {"sampler.logits_to_probs_d2d"},
        {"sampler.temp_to_probs_d2d"},
        {"sampler.copy_logits_d2h"},
        {"burst.slot_d2h"},
    }};

inline bool CopyTraceEnabled() {
  static const bool enabled = []() {
    const char *raw = std::getenv("INFERFLUX_CUDA_COPY_TRACE");
    return raw && std::string(raw) != "0" && std::string(raw) != "false";
  }();
  return enabled;
}

inline void LogCopyTraceSummaryAtExit() {
  if (!CopyTraceEnabled()) {
    return;
  }
  std::fprintf(stderr, "[native_copy_trace] summary begin\n");
  for (const auto &entry : kCopyTraceEntries) {
    const uint64_t calls = entry.calls.load(std::memory_order_relaxed);
    if (calls == 0) {
      continue;
    }
    const uint64_t bytes = entry.bytes.load(std::memory_order_relaxed);
    const uint64_t api_ns = entry.api_ns.load(std::memory_order_relaxed);
    const double avg_us = calls > 0 ? static_cast<double>(api_ns) /
                                          static_cast<double>(calls) / 1000.0
                                    : 0.0;
    std::fprintf(
        stderr,
        "[native_copy_trace] site=%s calls=%llu bytes=%llu api_ns=%llu "
        "avg_us=%.3f\n",
        entry.label, static_cast<unsigned long long>(calls),
        static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(api_ns), avg_us);
  }
  std::fprintf(stderr, "[native_copy_trace] summary end\n");
}

inline void EnsureCopyTraceRegistered() {
  static std::once_flag once;
  if (!CopyTraceEnabled()) {
    return;
  }
  std::call_once(once, []() { std::atexit(LogCopyTraceSummaryAtExit); });
}

template <typename DstPtr, typename SrcPtr>
inline cudaError_t
TracedCudaMemcpyAsync(CopyTraceSite site, DstPtr dst, SrcPtr src, size_t count,
                      enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (!CopyTraceEnabled()) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
  }
  EnsureCopyTraceRegistered();
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t err = cudaMemcpyAsync(dst, src, count, kind, stream);
  const auto end = std::chrono::steady_clock::now();
  auto &entry = kCopyTraceEntries[static_cast<size_t>(site)];
  entry.calls.fetch_add(1, std::memory_order_relaxed);
  entry.bytes.fetch_add(static_cast<uint64_t>(count),
                        std::memory_order_relaxed);
  entry.api_ns.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()),
      std::memory_order_relaxed);
  return err;
}

} // namespace inferflux::runtime::cuda::native
