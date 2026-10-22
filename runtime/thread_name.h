#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace inferflux {

/// Name the calling thread (visible in `top -H`, /proc/<pid>/task/*/comm,
/// gdb and profilers). Long-lived first-party threads call this on entry
/// for observability; skip it for short-lived per-request threads (each call
/// is a syscall). Correctness never depends on names: Linux threads inherit
/// their creator's name, so a name proves nothing about who owns a thread.
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
