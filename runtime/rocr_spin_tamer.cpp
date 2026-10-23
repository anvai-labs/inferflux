#include "runtime/rocr_spin_tamer.h"

#include "runtime/string_utils.h"
#include "runtime/thread_name.h"
#include "server/logging/logger.h"

#include <algorithm>
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

// The core rule, allocation-free (it also runs inside pthread_create):
// fewer than two allotted CPUs -> -1; a digits-only `core_env` naming an
// allotted CPU wins; otherwise the highest allotted CPU.
template <typename Contains>
int ChooseCore(std::size_t count, int highest, Contains contains,
               const char *core_env, bool *core_env_rejected) {
  if (core_env_rejected) {
    *core_env_rejected = false;
  }
  if (count < 2) {
    return -1; // Nothing to bound, or no allotment: never pin.
  }
  if (core_env == nullptr || *core_env == '\0') {
    return highest;
  }
  char *end = nullptr;
  errno = 0;
  // Digits only: strtol alone would also accept leading spaces and a sign.
  const bool digits = *core_env >= '0' && *core_env <= '9';
  const long requested = digits ? std::strtol(core_env, &end, 10) : -1;
  if (digits && errno == 0 && *end == '\0' && requested <= INT_MAX &&
      contains(requested)) {
    return static_cast<int>(requested);
  }
  if (core_env_rejected) {
    *core_env_rejected = true;
  }
  return highest;
}

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
// Double-buffered: a writer (static init, or the test seam) fills the
// inactive buffer, then publishes its index, so a reader racing one write
// never copies a half-written mask. Writes are not expected to overlap
// readers in production (static init runs once).
cpu_set_t g_allotment_buf[2];
std::atomic<int> g_allotment_idx{-1}; // -1: not captured yet.
std::atomic<int> g_next_name{0};
thread_local int t_tamed_cpu = -1;
// A tamed thread's mask before the pin: what its children would have
// inherited without the tamer.
thread_local cpu_set_t t_pre_pin_mask;
// ...and its name before the rename, which its children inherit.
thread_local char t_pre_name[16];

void CaptureAllotment() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    const int next =
        g_allotment_idx.load(std::memory_order_relaxed) == 0 ? 1 : 0;
    g_allotment_buf[next] = mask;
    g_allotment_idx.store(next, std::memory_order_release);
  }
}

// Runs in this binary's static initialization on the main thread: the
// operator's taskset/cgroup allotment (see the residuals in the header).
[[maybe_unused]] const bool g_allotment_captured_at_load = [] {
  CaptureAllotment();
  return true;
}();

#if defined(DLFO_STRUCT_HAS_EH_DBASE)
using FindObjectFn = int (*)(void *, dl_find_object *);
std::atomic<FindObjectFn> g_find_object{nullptr};
// Once, at static init (dlsym takes the loader lock, which is acceptable
// here, off the thread-creation path). Creations before this runs, e.g.
// from library constructors, use the dladdr fallback.
[[maybe_unused]] const bool g_find_object_resolved = [] {
  g_find_object.store(
      reinterpret_cast<FindObjectFn>(dlsym(RTLD_DEFAULT, "_dl_find_object")),
      std::memory_order_release);
  return true;
}();
#endif

bool AllotmentMask(cpu_set_t *mask) {
  CPU_ZERO(mask);
  const int idx = g_allotment_idx.load(std::memory_order_acquire);
  if (idx >= 0) {
    *mask = g_allotment_buf[idx];
    return true;
  }
  // getpid() as a tid names the main thread.
  return sched_getaffinity(getpid(), sizeof(*mask), mask) == 0;
}

// Same accepted values as ParseBoolEnv, without allocating: the first call
// can come from pthread_create in a library constructor or while an
// allocator holds its own locks.
bool ParseTameEnv() noexcept {
  return IsTruthyCStr(std::getenv("INFERFLUX_ROCR_TAME"));
}

bool Enabled() noexcept {
  int enabled = g_enabled.load(std::memory_order_acquire);
  if (enabled < 0) {
    enabled = ParseTameEnv() ? 1 : 0;
    g_enabled.store(enabled, std::memory_order_release);
  }
  return enabled == 1;
}

// True when `own` is exactly the pin inherited from a tamed creator (not an
// affinity the new thread's attribute asked for).
bool OnlyInheritedPin(const InheritedState &inherited, const cpu_set_t &own) {
  return inherited.creator_cpu >= 0 && !inherited.explicit_affinity &&
         CPU_COUNT(&own) == 1 && CPU_ISSET(inherited.creator_cpu, &own) &&
         CPU_COUNT(&inherited.mask) > 0;
}

int ChooseMaskCore(const cpu_set_t &mask, const char *core_env,
                   bool *core_env_rejected) {
  int highest = -1;
  for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu) {
    if (CPU_ISSET(cpu, &mask)) {
      highest = cpu;
      break;
    }
  }
  return ChooseCore(
      static_cast<std::size_t>(CPU_COUNT(&mask)), highest,
      [&](long cpu) { return cpu < CPU_SETSIZE && CPU_ISSET(cpu, &mask); },
      core_env, core_env_rejected);
}

int ResolveCore() {
  int core = g_core.load(std::memory_order_acquire);
  if (core != kUnresolved) {
    return core;
  }
  // No allocation here: this runs inside pthread_create.
  cpu_set_t mask;
  if (!AllotmentMask(&mask)) {
    CPU_ZERO(&mask);
  }
  core = ChooseMaskCore(mask, std::getenv("INFERFLUX_ROCR_TAME_CORE"), nullptr);
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
  int highest = -1;
  for (int cpu : allotted) {
    highest = cpu > highest ? cpu : highest;
  }
  return ChooseCore(
      allotted.size(), highest,
      [&](long cpu) {
        return std::find(allotted.begin(), allotted.end(), cpu) !=
               allotted.end();
      },
      core_env, core_env_rejected);
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
#if defined(DLFO_STRUCT_HAS_EH_DBASE)
  // glibc >= 2.35: lock-free and async-signal-safe, so thread creation never
  // waits on the dynamic loader's lock (a concurrent dlopen). Resolved at
  // run time, not linked, so the binary still starts on older glibc.
  if (const FindObjectFn find = g_find_object.load(std::memory_order_acquire)) {
    dl_find_object found{};
    if (find(const_cast<void *>(addr), &found) != 0 ||
        found.dlfo_link_map == nullptr) {
      return false;
    }
    return IsRocrRuntimeObjectPath(found.dlfo_link_map->l_name);
  }
#endif
  // Older glibc, or before static init resolved it: dladdr takes the loader
  // lock (see the residuals).
  Dl_info info{};
  return dladdr(const_cast<void *>(addr), &info) != 0 &&
         IsRocrRuntimeObjectPath(info.dli_fname);
}

bool RocrTamerEnabled() noexcept { return Enabled(); }

int RocrTameCoreForNewThread(void *(*start_routine)(void *)) noexcept {
  if (!Enabled() ||
      !IsRocrRuntimeAddress(reinterpret_cast<const void *>(start_routine))) {
    return -1;
  }
  return ResolveCore(); // Allocation-free; nothing here throws.
}

void TameCurrentRocrThread(int core, const InheritedState &inherited) noexcept {
  int old_cancel_state = 0;
  (void)pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancel_state);
  const int saved_errno = errno;
  cpu_set_t own;
  CPU_ZERO(&own);
  int target = -1;
  if (core >= 0 && core < CPU_SETSIZE &&
      sched_getaffinity(0, sizeof(own), &own) == 0) {
    // Created by a tamed thread without its own affinity: its mask is the
    // creator's pin; treat the creator's pre-pin mask as its own so that
    // neither the target nor what its children get back is the pin.
    if (OnlyInheritedPin(inherited, own)) {
      own = inherited.mask;
    }
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
      if (inherited.creator_cpu >= 0 && inherited.name[0] != '\0') {
        std::memcpy(t_pre_name, inherited.name, sizeof(t_pre_name));
      } else if (pthread_getname_np(pthread_self(), t_pre_name,
                                    sizeof(t_pre_name)) != 0) {
        t_pre_name[0] = '\0';
      }
      SetCurrentThreadName(name);
      t_pre_pin_mask = own;
      t_tamed_cpu = target;
      g_tamed.fetch_add(1, std::memory_order_relaxed);
    } else {
      g_skipped.fetch_add(1, std::memory_order_relaxed);
    }
  } else if (core >= 0) {
    g_skipped.fetch_add(1, std::memory_order_relaxed);
  }
  if (t_tamed_cpu < 0) {
    // Not pinned: undo anything inherited from a tamed creator, exactly as
    // for a non-ROCr child, so it is not left on the creator's core.
    RestoreInheritedScheduling(inherited);
  }
  errno = saved_errno;
  (void)pthread_setcancelstate(old_cancel_state, nullptr);
}

void CaptureInheritedState(const pthread_attr_t *attr,
                           InheritedState *inherited) noexcept {
  const int creator_cpu = t_tamed_cpu;
  inherited->creator_cpu = creator_cpu;
  if (creator_cpu < 0) {
    return;
  }
  inherited->mask = t_pre_pin_mask;
  std::memcpy(inherited->name, t_pre_name, sizeof(inherited->name));
  inherited->explicit_affinity = false;
  if (attr != nullptr) {
    // glibc reports "no cpuset in the attribute" as all CPUs, so a
    // single-CPU answer equal to the creator's pin is an explicit request.
    cpu_set_t requested;
    CPU_ZERO(&requested);
    inherited->explicit_affinity =
        pthread_attr_getaffinity_np(attr, sizeof(requested), &requested) == 0 &&
        CPU_COUNT(&requested) == 1 && CPU_ISSET(creator_cpu, &requested);
  }
}

void RestoreInheritedScheduling(const InheritedState &inherited) noexcept {
  const int saved_errno = errno;
  // The creator's rocr-hsa-N name was inherited; give back the name it had.
  if (inherited.creator_cpu >= 0 && inherited.name[0] != '\0') {
    SetCurrentThreadName(inherited.name);
  }
  // Still exactly the creator's pin and not asked for: give the thread the
  // mask it would have inherited without the tamer.
  cpu_set_t own;
  CPU_ZERO(&own);
  if (sched_getaffinity(0, sizeof(own), &own) == 0 &&
      OnlyInheritedPin(inherited, own)) {
    (void)sched_setaffinity(0, sizeof(inherited.mask), &inherited.mask);
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
      log::Info(kComponent,
                "ROCr runtime loaded; if its poller threads busy-spin at idle "
                "(WSL2, ROCm without rocm-systems#7898), set "
                "INFERFLUX_ROCR_TAME=1 or use the patched runtime (see "
                "docs/ROCM_INSTALLATION_GUIDE_WSL.md)");
    }
    return;
  }
  // Same rule (ChooseMaskCore) as the creations use, for the rejected flag.
  bool core_rejected = false;
  cpu_set_t allotted;
  if (!AllotmentMask(&allotted)) {
    CPU_ZERO(&allotted);
  }
  (void)ChooseMaskCore(allotted, std::getenv("INFERFLUX_ROCR_TAME_CORE"),
                       &core_rejected);
  if (core_rejected) {
    log::Warn(
        kComponent,
        std::string("INFERFLUX_ROCR_TAME_CORE=") +
            std::getenv("INFERFLUX_ROCR_TAME_CORE") +
            " is not a CPU number in the server's affinity mask; ignoring it");
  }
  void *const interposer = g_interposer.load(std::memory_order_acquire);
  const bool bound = interposer != nullptr &&
                     dlsym(RTLD_DEFAULT, "pthread_create") == interposer;
  if (!bound) {
    log::Warn(kComponent, "enabled, but the pthread_create interposer is not "
                          "in this binary (it is linked only into ROCm-enabled "
                          "x86_64/aarch64 builds without sanitizers); ROCr "
                          "threads will not be tamed");
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
          std::to_string(core) + " (of " +
          std::to_string(CPU_COUNT(&allotted)) +
          " allotted) at creation (a thread ROCr bound elsewhere: the highest "
          "allotted CPU of its own mask); tamed so far: " +
          std::to_string(RocrTamedThreadCount()) +
          (HsaRuntimeLoaded() ? "" : " (libhsa-runtime64 not loaded yet)"));
  if (RocrSkippedThreadCount() > 0) {
    log::Warn(kComponent, std::to_string(RocrSkippedThreadCount()) +
                              " ROCr runtime thread(s) could not be pinned "
                              "(no allotted CPU in their own mask, or the "
                              "kernel refused the pin, e.g. a cpuset change)");
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
