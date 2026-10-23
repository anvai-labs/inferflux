#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace inferflux {

#if defined(__linux__)
/// Kernel thread id of the calling thread (the TID in /proc/self/task).
/// Async-signal-safe: a single raw syscall.
inline pid_t CurrentThreadId() {
  return static_cast<pid_t>(syscall(SYS_gettid));
}
#endif

/// Names the calling thread (visible in `top -H`, /proc/<pid>/task/*/comm,
/// gdb and profilers). Long-lived first-party threads call this on entry;
/// skip it for short-lived per-request threads (each call is a syscall).
/// Names are for humans only: Linux threads inherit their creator's name,
/// so a name proves nothing about who owns a thread.
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
