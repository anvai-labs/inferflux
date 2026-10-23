#include "runtime/rocr_spin_tamer.h"

#include "runtime/string_utils.h"
#include "runtime/thread_name.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#endif

namespace inferflux {
namespace {

constexpr const char *kRuntimeBasenamePrefix = "libhsa-runtime64";

std::atomic<unsigned long long> g_tamed{0};
std::atomic<void *> g_interposer{nullptr};

#if defined(__linux__)
const char *kComponent = "rocr_tamer";
constexpr int kUnresolved = -2;
// All constant-initialized: the interposer can call in from threads created
// by library constructors, before this file's dynamic initializers run.
std::atomic<int> g_enabled{-1}; // -1 unknown, else 0/1 (env read once).
std::atomic<int> g_core{kUnresolved};
std::atomic<bool> g_allotment_set{false};
cpu_set_t g_allotment; // Written by static init / the test seam only.
std::atomic<int> g_next_name{0};

void CaptureAllotment() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    g_allotment = mask;
    g_allotment_set.store(true, std::memory_order_release);
  }
}

// During static initialization the calling thread is the main thread, so
// this is the operator's taskset/cgroup allotment before any library binding
// can narrow it.
[[maybe_unused]] const bool g_allotment_captured_at_load = [] {
  CaptureAllotment();
  return true;
}();

std::vector<int> AllottedCpus() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (g_allotment_set.load(std::memory_order_acquire)) {
    mask = g_allotment;
  } else if (sched_getaffinity(getpid(), sizeof(mask), &mask) != 0) {
    CPU_ZERO(&mask); // getpid() as a tid names the main thread.
  }
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask)) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

bool Enabled() {
  int enabled = g_enabled.load(std::memory_order_acquire);
  if (enabled < 0) {
    enabled = ParseBoolEnv("INFERFLUX_ROCR_TAME", false) ? 1 : 0;
    g_enabled.store(enabled, std::memory_order_release);
  }
  return enabled == 1;
}

int ResolveCore() {
  int core = g_core.load(std::memory_order_acquire);
  if (core != kUnresolved) {
    return core;
  }
  core = ChooseRocrTameCore(AllottedCpus(),
                            std::getenv("INFERFLUX_ROCR_TAME_CORE"));
  int expected = kUnresolved; // Deterministic; a racing resolver agrees.
  g_core.compare_exchange_strong(expected, core);
  return g_core.load(std::memory_order_acquire);
}

bool HsaRuntimeLoaded() {
  for (const char *soname : {"libhsa-runtime64.so.1", "libhsa-runtime64.so"}) {
    if (void *handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD)) {
      dlclose(handle);
      return true;
    }
  }
  return false;
}
#endif // defined(__linux__)

} // namespace

int ChooseRocrTameCore(const std::vector<int> &allotted, const char *core_env,
                       bool *core_env_rejected) {
  if (core_env_rejected) {
    *core_env_rejected = false;
  }
  if (allotted.size() < 2) {
    return -1; // Nothing to bound, or no allotment: never pin.
  }
  int highest = allotted.front();
  for (int cpu : allotted) {
    highest = cpu > highest ? cpu : highest;
  }
  if (core_env == nullptr || *core_env == '\0') {
    return highest;
  }
  char *end = nullptr;
  errno = 0;
  const long requested = std::strtol(core_env, &end, 10);
  const bool parsed = errno == 0 && end != core_env && *end == '\0' &&
                      requested >= 0 && requested <= INT_MAX;
  if (parsed) {
    for (int cpu : allotted) {
      if (cpu == requested) {
        return cpu;
      }
    }
  }
  if (core_env_rejected) {
    *core_env_rejected = true;
  }
  return highest;
}

bool IsRocrRuntimeObjectPath(const char *object_path) {
  if (object_path == nullptr || *object_path == '\0') {
    return false;
  }
  const char *slash = std::strrchr(object_path, '/');
  const char *base = slash ? slash + 1 : object_path;
  return std::strncmp(base, kRuntimeBasenamePrefix,
                      std::strlen(kRuntimeBasenamePrefix)) == 0;
}

#if defined(__linux__)
bool IsRocrRuntimeAddress(const void *addr) noexcept {
  Dl_info info{};
  return addr != nullptr && dladdr(const_cast<void *>(addr), &info) != 0 &&
         IsRocrRuntimeObjectPath(info.dli_fname);
}

int RocrTameCoreForNewThread(void *(*start_routine)(void *)) noexcept {
  try {
    if (!Enabled() ||
        !IsRocrRuntimeAddress(reinterpret_cast<const void *>(start_routine))) {
      return -1;
    }
    return ResolveCore();
  } catch (...) {
    return -1; // Creator-side frame: never fail thread creation over this.
  }
}

void TameCurrentRocrThread(int core) noexcept {
  int old_cancel_state = 0;
  (void)pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancel_state);
  const int saved_errno = errno;
  cpu_set_t own;
  CPU_ZERO(&own);
  // A core outside the thread's own mask is left alone (pinning elsewhere
  // would break the one-core bound).
  if (core >= 0 && core < CPU_SETSIZE &&
      sched_getaffinity(0, sizeof(own), &own) == 0 && CPU_ISSET(core, &own)) {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(core, &one);
    if (sched_setaffinity(0, sizeof(one), &one) == 0) {
      // Linux nice is per-thread; name the thread explicitly.
      (void)setpriority(PRIO_PROCESS, static_cast<id_t>(CurrentThreadId()), 19);
      char name[16];
      std::snprintf(name, sizeof(name), "rocr-hsa-%d",
                    g_next_name.fetch_add(1) % 100000);
      (void)pthread_setname_np(pthread_self(), name);
      g_tamed.fetch_add(1, std::memory_order_relaxed);
      GlobalMetrics().RecordRuntimeSpinThreadsConfined(1);
    }
  }
  errno = saved_errno;
  (void)pthread_setcancelstate(old_cancel_state, nullptr);
}
#endif // defined(__linux__)

void NoteRocrPthreadInterposerInstalled(void *interposer_fn) noexcept {
  g_interposer.store(interposer_fn, std::memory_order_release);
}

unsigned long long RocrTamedThreadCount() noexcept {
  return g_tamed.load(std::memory_order_relaxed);
}

void ReportRocrTamerStatus() {
#if defined(__linux__)
  if (!Enabled()) {
    return;
  }
  bool core_rejected = false;
  const std::vector<int> allotted = AllottedCpus();
  const int core = ChooseRocrTameCore(
      allotted, std::getenv("INFERFLUX_ROCR_TAME_CORE"), &core_rejected);
  if (core_rejected) {
    log::Warn(kComponent,
              std::string("INFERFLUX_ROCR_TAME_CORE=") +
                  std::getenv("INFERFLUX_ROCR_TAME_CORE") +
                  " is not a CPU in the server's affinity mask; ignoring it");
  }
  void *const interposer = g_interposer.load(std::memory_order_acquire);
  const bool bound = interposer != nullptr &&
                     dlsym(RTLD_DEFAULT, "pthread_create") == interposer;
  if (!bound) {
    log::Warn(kComponent, "enabled, but the pthread_create interposer is not "
                          "active in this binary (e.g. a sanitizer build); "
                          "ROCr threads will not be tamed");
    return;
  }
  if (core < 0) {
    log::Warn(kComponent, "the server is allotted fewer than two CPUs; "
                          "leaving ROCr runtime threads unpinned");
    return;
  }
  log::Info(
      kComponent,
      "enabled: ROCr runtime threads are pinned to CPU " +
          std::to_string(core) + " (of " + std::to_string(allotted.size()) +
          " allotted) at creation; tamed so far: " +
          std::to_string(RocrTamedThreadCount()) +
          (HsaRuntimeLoaded() ? "" : " (libhsa-runtime64 not loaded yet)"));
#endif
}

void ResetRocrTamerForTesting() {
#if defined(__linux__)
  g_enabled.store(-1, std::memory_order_release);
  g_core.store(kUnresolved, std::memory_order_release);
  CaptureAllotment();
#endif
}

} // namespace inferflux
