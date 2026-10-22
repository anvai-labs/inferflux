#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <mutex>
#include <unordered_set>
#endif

namespace inferflux {

#if defined(__linux__)
/// Kernel thread id of the calling thread (the TID in /proc/self/task).
/// Async-signal-safe: a single raw syscall.
inline pid_t CurrentThreadId() {
  return static_cast<pid_t>(syscall(SYS_gettid));
}

namespace detail {
// Leaked on purpose: thread_local guards of threads that exit during or
// after static destruction still unregister safely.
inline std::mutex &FirstPartyThreadsMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}
inline std::unordered_set<pid_t> &FirstPartyThreads() {
  static auto *threads = new std::unordered_set<pid_t>();
  return *threads;
}

// Registers the owning thread for its lifetime; the destructor runs at
// thread exit, so a TID later reused by another thread is never registered.
struct FirstPartyThreadGuard {
  pid_t tid = CurrentThreadId();
  FirstPartyThreadGuard() {
    std::lock_guard<std::mutex> lock(FirstPartyThreadsMutex());
    FirstPartyThreads().insert(tid);
  }
  ~FirstPartyThreadGuard() {
    std::lock_guard<std::mutex> lock(FirstPartyThreadsMutex());
    FirstPartyThreads().erase(tid);
  }
  FirstPartyThreadGuard(const FirstPartyThreadGuard &) = delete;
  FirstPartyThreadGuard &operator=(const FirstPartyThreadGuard &) = delete;
};
} // namespace detail
#endif

/// Marks the calling thread as an InferFlux (first-party) thread until it
/// exits. Runtime hygiene that acts on other threads (the ROCr spin tamer)
/// never touches a marked thread: an inference thread busy-waiting inside a
/// GPU runtime library must not be mistaken for that runtime's own poller.
/// Every thread InferFlux creates must call this (directly, or through
/// SetCurrentThreadName) on entry. Idempotent; no-op off Linux.
inline void MarkFirstPartyThread() {
#if defined(__linux__)
  thread_local detail::FirstPartyThreadGuard guard;
  (void)guard;
#endif
}

#if defined(__linux__)
/// True while `tid` is a live thread that called MarkFirstPartyThread().
inline bool IsFirstPartyThread(pid_t tid) {
  std::lock_guard<std::mutex> lock(detail::FirstPartyThreadsMutex());
  return detail::FirstPartyThreads().count(tid) != 0;
}
#endif

/// Marks the calling thread first-party (see MarkFirstPartyThread) and names
/// it (visible in `top -H`, /proc/<pid>/task/*/comm, gdb and profilers).
/// Long-lived first-party threads call this on entry; short-lived
/// per-request threads call MarkFirstPartyThread() alone (naming is an extra
/// syscall). Names are for humans only: Linux threads inherit their
/// creator's name, so a name proves nothing about who owns a thread.
///
/// Linux limits names to 15 characters plus NUL; longer names are rejected
/// by the kernel, so keep names short (convention: "if-<role>").
/// Best-effort: naming failures are ignored. No-op on unsupported platforms.
inline void SetCurrentThreadName(const char *name) {
  MarkFirstPartyThread();
#if defined(__linux__)
  (void)pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
  (void)pthread_setname_np(name);
#else
  (void)name;
#endif
}

} // namespace inferflux
