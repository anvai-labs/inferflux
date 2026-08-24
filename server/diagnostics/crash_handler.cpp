#include "server/diagnostics/crash_handler.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif
#else
#include <unistd.h>
#endif

namespace inferflux {
namespace diagnostics {

// --- Thread-local breadcrumb ---

static constexpr int kBreadcrumbMaxLen = 256;

// Thread-local breadcrumb buffer. Fixed-size, no heap allocation,
// safe to read from a signal handler on the crashing thread.
static thread_local char tl_breadcrumb[kBreadcrumbMaxLen] = {};
static thread_local int tl_breadcrumb_len = 0;

void SetCrashBreadcrumb(const char *message) {
  if (!message) {
    tl_breadcrumb[0] = '\0';
    tl_breadcrumb_len = 0;
    return;
  }
  int i = 0;
  while (i < kBreadcrumbMaxLen - 1 && message[i] != '\0') {
    tl_breadcrumb[i] = message[i];
    ++i;
  }
  tl_breadcrumb[i] = '\0';
  tl_breadcrumb_len = i;
}

void ClearCrashBreadcrumb() {
  tl_breadcrumb[0] = '\0';
  tl_breadcrumb_len = 0;
}

// --- CUDA error counter ---

static std::atomic<uint64_t> g_cuda_error_count{0};

void RecordCudaError() {
  g_cuda_error_count.fetch_add(1, std::memory_order_relaxed);
}

uint64_t GetCudaErrorCount() {
  return g_cuda_error_count.load(std::memory_order_relaxed);
}

// --- Active request gauge ---

static std::atomic<int> g_active_requests{0};

void SetActiveRequests(int count) {
  g_active_requests.store(count, std::memory_order_relaxed);
}

// --- Signal handler (async-signal-safe only) ---

namespace {

// Suppress unused-result warnings for write() in signal handler context.
// We can't meaningfully handle write failures during a crash dump.
#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

// Async-signal-safe write wrapper.
void SafeWrite(int fd, const char *buf, size_t len) {
#ifdef _WIN32
  _write(fd, buf, static_cast<unsigned int>(len));
#else
  ::write(fd, buf, len);
#endif
}

// Async-signal-safe integer-to-string for small positive integers.
void WriteInt(int fd, int64_t value) {
  if (value < 0) {
    SafeWrite(fd, "-", 1);
    value = -value;
  }
  char buf[20];
  int pos = sizeof(buf);
  if (value == 0) {
    buf[--pos] = '0';
  } else {
    while (value > 0) {
      buf[--pos] = '0' + static_cast<char>(value % 10);
      value /= 10;
    }
  }
  SafeWrite(fd, buf + pos, sizeof(buf) - pos);
}

const char *SignalName(int sig) {
  switch (sig) {
  case SIGSEGV:
    return "SIGSEGV (Segmentation fault)";
  case SIGABRT:
    return "SIGABRT (Aborted)";
#ifndef _WIN32
  case SIGBUS:
    return "SIGBUS (Bus error)";
#endif
  case SIGFPE:
    return "SIGFPE (Floating point exception)";
  default:
    return "Unknown signal";
  }
}

void CrashSignalHandler(int sig) {
  // All output uses write() which is async-signal-safe.
  const int fd = STDERR_FILENO;

  SafeWrite(fd, "\n", 1);
  SafeWrite(fd, "=== INFERFLUX CRASH DIAGNOSTIC ===\n", 35);

  // Signal info
  SafeWrite(fd, "Signal: ", 8);
  const char *name = SignalName(sig);
  SafeWrite(fd, name, strlen(name));
  SafeWrite(fd, " (", 2);
  WriteInt(fd, sig);
  SafeWrite(fd, ")\n", 2);

  // Thread breadcrumb (only valid for the crashing thread)
  if (tl_breadcrumb_len > 0) {
    SafeWrite(fd, "Operation: ", 11);
    SafeWrite(fd, tl_breadcrumb, tl_breadcrumb_len);
    SafeWrite(fd, "\n", 1);
  } else {
    SafeWrite(fd, "Operation: (no breadcrumb set)\n", 30);
  }

  // Active requests
  int active = g_active_requests.load(std::memory_order_relaxed);
  SafeWrite(fd, "Active requests: ", 17);
  WriteInt(fd, active);
  SafeWrite(fd, "\n", 1);

  // CUDA errors so far
  uint64_t cuda_errors = g_cuda_error_count.load(std::memory_order_relaxed);
  if (cuda_errors > 0) {
    SafeWrite(fd, "CUDA errors recorded: ", 22);
    WriteInt(fd, static_cast<int64_t>(cuda_errors));
    SafeWrite(fd, "\n", 1);
  }

  SafeWrite(fd, "=== END CRASH DIAGNOSTIC ===\n", 29);

  // Re-raise with default handler so we get a core dump.
#ifdef _WIN32
  signal(sig, SIG_DFL);
#else
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = SIG_DFL;
  sigaction(sig, &sa, nullptr);
#endif
  raise(sig);
}

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

} // namespace

void InstallCrashHandler() {
#ifdef _WIN32
  signal(SIGSEGV, CrashSignalHandler);
  signal(SIGABRT, CrashSignalHandler);
  signal(SIGFPE, CrashSignalHandler);
#else
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = CrashSignalHandler;
  // Don't block other signals during handler (keep it simple).
  sigemptyset(&sa.sa_mask);
  // SA_RESETHAND: restore default after first invocation (prevents loops).
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
#endif
}

} // namespace diagnostics
} // namespace inferflux
