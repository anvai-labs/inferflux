#include "runtime/rocr_spin_tamer.h"

#include "runtime/string_utils.h"
#include "runtime/thread_name.h"
#include "server/logging/logger.h"

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdio>
#endif

namespace inferflux {
namespace {

constexpr const char *kRuntimeBasenamePrefix = "libhsa-runtime64";

// All constant-initialized: the interposer can call in from threads created
// by library constructors, before any dynamic initializer here has run.
std::atomic<unsigned long long> g_tamed{0};
std::atomic<unsigned long long> g_skipped{0};
std::atomic<void *> g_interposer{nullptr};

#if defined(__linux__)
const char *kComponent = "rocr_tamer";
constexpr int kUnresolved = -2;
std::atomic<int> g_enabled{-1}; // -1 unknown, else 0/1 (env read once).
std::atomic<int> g_core{kUnresolved};
std::atomic<bool> g_allotment_set{false};
cpu_set_t g_allotment; // Written by static init / the test seam only.
std::atomic<int> g_next_name{0};
// Load range of the identified ROCr runtime object; once known, start
// routines are classified by address compare, without the loader lock.
std::atomic<std::uintptr_t> g_runtime_lo{0};
std::atomic<std::uintptr_t> g_runtime_hi{0};
thread_local int t_tamed_cpu = -1;

void CaptureAllotment() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    g_allotment = mask;
    g_allotment_set.store(true, std::memory_order_release);
  }
}

// Runs in this binary's static initialization on the main thread: the
// operator's taskset/cgroup allotment (see the residuals in the header).
[[maybe_unused]] const bool g_allotment_captured_at_load = [] {
  CaptureAllotment();
  return true;
}();

bool AllotmentMask(cpu_set_t *mask) {
  CPU_ZERO(mask);
  if (g_allotment_set.load(std::memory_order_acquire)) {
    *mask = g_allotment;
    return true;
  }
  // getpid() as a tid names the main thread.
  return sched_getaffinity(getpid(), sizeof(*mask), mask) == 0;
}

std::vector<int> AllottedCpus() {
  cpu_set_t mask;
  std::vector<int> cpus;
  if (AllotmentMask(&mask)) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &mask)) {
        cpus.push_back(cpu);
      }
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

// Records the load range of the object containing `base` (dl_iterate_phdr,
// once), so later classifications avoid dladdr.
int CacheRangeCallback(dl_phdr_info *info, size_t, void *data) {
  const auto base = *static_cast<std::uintptr_t *>(data);
  if (static_cast<std::uintptr_t>(info->dlpi_addr) != base) {
    return 0;
  }
  std::uintptr_t lo = UINTPTR_MAX;
  std::uintptr_t hi = 0;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) &ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_LOAD) {
      continue;
    }
    const std::uintptr_t start = info->dlpi_addr + ph.p_vaddr;
    lo = start < lo ? start : lo;
    hi = start + ph.p_memsz > hi ? start + ph.p_memsz : hi;
  }
  if (hi > lo) {
    g_runtime_lo.store(lo, std::memory_order_relaxed);
    g_runtime_hi.store(hi, std::memory_order_release);
  }
  return 1;
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
  if (addr == nullptr) {
    return false;
  }
  const auto value = reinterpret_cast<std::uintptr_t>(addr);
  const std::uintptr_t hi = g_runtime_hi.load(std::memory_order_acquire);
  if (hi != 0) {
    return value >= g_runtime_lo.load(std::memory_order_relaxed) && value < hi;
  }
  Dl_info info{};
  if (dladdr(const_cast<void *>(addr), &info) == 0 ||
      !IsRocrRuntimeObjectPath(info.dli_fname)) {
    return false;
  }
  auto base = reinterpret_cast<std::uintptr_t>(info.dli_fbase);
  dl_iterate_phdr(&CacheRangeCallback, &base);
  return true;
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
  int target = -1;
  if (core >= 0 && core < CPU_SETSIZE &&
      sched_getaffinity(0, sizeof(own), &own) == 0) {
    if (CPU_ISSET(core, &own)) {
      target = core;
    } else {
      // The runtime confined this thread itself (e.g. to the GPU's NUMA
      // node): use the highest CPU of its mask that is also allotted, so
      // threads sharing that mask still share one core.
      cpu_set_t allotted;
      if (AllotmentMask(&allotted)) {
        for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu) {
          if (CPU_ISSET(cpu, &own) && CPU_ISSET(cpu, &allotted)) {
            target = cpu;
            break;
          }
        }
      }
    }
  }
  if (target >= 0) {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(target, &one);
    // Pin only, no nice/SCHED_IDLE: a thread inherits its creator's nice
    // value and an unprivileged thread cannot lower it again, so anything a
    // tamed thread created would stay deprioritized. Affinity can be
    // restored (RestoreInheritedScheduling), so the pin cannot leak.
    if (sched_setaffinity(0, sizeof(one), &one) == 0) {
      char name[16];
      std::snprintf(name, sizeof(name), "rocr-hsa-%d",
                    g_next_name.fetch_add(1) % 100000);
      (void)pthread_setname_np(pthread_self(), name);
      t_tamed_cpu = target;
      g_tamed.fetch_add(1, std::memory_order_relaxed);
    } else {
      g_skipped.fetch_add(1, std::memory_order_relaxed);
    }
  } else if (core >= 0) {
    g_skipped.fetch_add(1, std::memory_order_relaxed);
  }
  errno = saved_errno;
  (void)pthread_setcancelstate(old_cancel_state, nullptr);
}

bool CurrentThreadIsTamedRocrThread() noexcept { return t_tamed_cpu >= 0; }

int CurrentTamedRocrThreadCpu() noexcept { return t_tamed_cpu; }

void RestoreInheritedScheduling(int creator_cpu) noexcept {
  const int saved_errno = errno;
  cpu_set_t own;
  CPU_ZERO(&own);
  // glibc reports "no cpuset" as all CPUs, so the creator cannot tell an
  // explicit attribute affinity from none; the child can: an inherited mask
  // is exactly the creator's single pinned CPU.
  if (creator_cpu >= 0 && sched_getaffinity(0, sizeof(own), &own) == 0 &&
      CPU_COUNT(&own) == 1 && CPU_ISSET(creator_cpu, &own)) {
    cpu_set_t allotted;
    if (AllotmentMask(&allotted)) {
      (void)sched_setaffinity(0, sizeof(allotted), &allotted);
    }
  }
  errno = saved_errno;
}
#endif // defined(__linux__)

void NoteRocrPthreadInterposerInstalled(void *interposer_fn) noexcept {
  g_interposer.store(interposer_fn, std::memory_order_release);
}

unsigned long long RocrTamedThreadCount() noexcept {
  return g_tamed.load(std::memory_order_relaxed);
}

unsigned long long RocrSkippedThreadCount() noexcept {
  return g_skipped.load(std::memory_order_relaxed);
}

void ReportRocrTamerStatus() {
#if defined(__linux__)
  if (!Enabled()) {
    if (HsaRuntimeLoaded()) {
      log::Info(kComponent, "ROCr runtime loaded; set INFERFLUX_ROCR_TAME=1 "
                            "to confine its busy-spin poller threads (see "
                            "docs/ROCM_INSTALLATION_GUIDE_WSL.md)");
    }
    return;
  }
  bool core_rejected = false;
  const std::vector<int> allotted = AllottedCpus();
  (void)ChooseRocrTameCore(allotted, std::getenv("INFERFLUX_ROCR_TAME_CORE"),
                           &core_rejected);
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
  const int core = ResolveCore(); // The value creations actually use.
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
  if (RocrSkippedThreadCount() > 0) {
    log::Warn(kComponent, std::to_string(RocrSkippedThreadCount()) +
                              " ROCr runtime thread(s) could not be pinned "
                              "(their own CPU mask has no allotted CPU)");
  }
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
