#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace inferflux {

/// Name the calling thread (visible in `top -H`, /proc/<pid>/task/*/comm,
/// gdb and profilers). Every first-party thread should call this on entry:
/// the ROCr spin tamer (server/rocr_spin_tamer.h) treats threads that still
/// carry the process name as third-party runtime threads, so an unnamed
/// first-party thread could be misclassified while busy.
///
/// Linux limits names to 15 characters plus NUL; longer names are rejected
/// by the kernel, so keep names short (convention: "if-<role>").
/// Best-effort: failures are ignored. No-op on unsupported platforms.
inline void SetCurrentThreadName(const char *name) {
#if defined(__linux__)
  (void)pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
  (void)pthread_setname_np(name);
#else
  (void)name;
#endif
}

} // namespace inferflux
