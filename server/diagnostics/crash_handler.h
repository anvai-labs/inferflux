#pragma once

#include <atomic>
#include <cstdint>

namespace inferflux {
namespace diagnostics {

/// Install signal handlers for SIGSEGV, SIGABRT, SIGBUS, SIGFPE.
/// On crash, writes a diagnostic dump to stderr using only
/// async-signal-safe functions, then re-raises the signal for
/// the default handler (core dump).
///
/// Call once from main() after logging is initialized.
void InstallCrashHandler();

/// Thread-local breadcrumb: set before entering a critical section
/// so the crash handler can report what the thread was doing.
/// Uses a fixed-size buffer — no allocations.
///
/// Example:
///   SetCrashBreadcrumb("ExecuteUnifiedBatch: prefill M=32 N=2048");
///   ...
///   ClearCrashBreadcrumb();
void SetCrashBreadcrumb(const char *message);
void ClearCrashBreadcrumb();

/// RAII guard for crash breadcrumbs.
struct ScopedBreadcrumb {
  explicit ScopedBreadcrumb(const char *message) {
    SetCrashBreadcrumb(message);
  }
  ~ScopedBreadcrumb() { ClearCrashBreadcrumb(); }
  ScopedBreadcrumb(const ScopedBreadcrumb &) = delete;
  ScopedBreadcrumb &operator=(const ScopedBreadcrumb &) = delete;
};

/// Atomic counters for CUDA errors — safe to increment from any thread.
/// Read via GetCudaErrorCount() for metrics export.
void RecordCudaError();
uint64_t GetCudaErrorCount();

/// Record active request count for crash diagnostics.
void SetActiveRequests(int count);

} // namespace diagnostics
} // namespace inferflux
