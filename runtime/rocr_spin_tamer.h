#pragma once

#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace inferflux {

// ROCr thread tamer (WSL2/ROCm AsyncEventsLoop workaround).
//
// ROCm 7.x libhsa-runtime64 starts AsyncEventsLoop poller threads at
// hipInit() that busy-spin at 100% CPU each for the process lifetime on
// WSL2/gfx1201 (upstream fix rocm-systems#7898 unreleased; no HSA_* knob
// disables the loop). The tamer bounds that cost to one core.
//
// Identification is exact and happens at thread creation: inferfluxd defines
// pthread_create (server/rocr_pthread_interposer.cpp, linked only into
// ROCm-enabled server builds), and every thread whose start routine lies in
// libhsa-runtime64 is started through a trampoline that, in the new thread,
// pins it to one shared core and names it "rocr-hsa-N". No
// signals, no /proc scans, no background thread. Every libhsa-created thread
// is tamed (they all start through one ROCr-internal routine, so pollers
// cannot be told apart from ROCr's other, mostly blocked, threads).
//
// Why a pin applied in the new thread: ROCr passes an explicit all-CPU
// affinity through pthread_attr, so a pinned creator's mask is not
// inherited, and disabling ROCr's override (HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0)
// under a pinned creator aborts. libhsa never changes a thread's affinity
// after creation (its only affinity import is pthread_attr_setaffinity_np).
// Why a pin only (no nice 19, no SCHED_IDLE): a thread inherits its
// creator's nice/policy and an unprivileged thread cannot lower its nice
// again, so a deprioritized ROCr thread would leak that to any thread it
// creates; SCHED_IDLE can also starve the HSA async signal handlers these
// threads run on native Linux. The pin is the bound, and it is reversible.
//
// Core choice: the server's affinity allotment is the main thread's mask,
// captured during static initialization (the operator's taskset/cgroup).
// Fewer than two allotted CPUs -> never pin (two spinners must not be piled
// onto the server's only CPU). Otherwise a valid INFERFLUX_ROCR_TAME_CORE
// inside the allotment, else the highest allotted CPU.
//
// A ROCr thread whose own (attribute-supplied) mask excludes the chosen core
// -- e.g. ROCr binding it to the GPU's NUMA node -- is pinned to the highest
// CPU in its mask that is also allotted, so each such mask still costs one
// core; with no overlap it is left alone and counted as skipped. Threads that
// a tamed thread creates (and that are not ROCr threads themselves) get the
// mask they would have inherited without the tamer (the creator's mask
// before its pin) unless their attribute sets an affinity explicitly, so
// the pin never leaks to them.
//
// Residuals: the allotment is read in this binary's static initialization,
// after shared-library constructors (libgomp with OMP_PROC_BIND can narrow
// the main thread first; that fails safe: no pin, or a pin inside it); a
// cgroup cpuset later shrunk to exclude the core moves the threads off it;
// on glibc < 2.35 (no lock-free _dl_find_object), and for creations during
// static initialization, each classification's dladdr takes the dynamic
// loader's lock; a future ROCr starting threads via
// std::thread (start routine in libstdc++), or loaded with RTLD_DEEPBIND,
// would not be recognized (fails safe: nothing tamed); sanitizer builds and
// non-x86_64/aarch64 targets compile the interposer out.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable with 1 (default off)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU (default: highest allotted CPU)

// Pure. CPU to pin ROCr runtime threads on, or -1 to never pin. Fewer than
// two allotted CPUs -> -1. A `core_env` naming an allotted CPU wins;
// otherwise the highest allotted CPU. Sets *core_env_rejected when
// `core_env` is non-empty but unparseable or not allotted.
int ChooseRocrTameCore(const std::vector<int> &allotted, const char *core_env,
                       bool *core_env_rejected = nullptr);

// Pure. True when the basename of `object_path` starts with
// "libhsa-runtime64" (a directory of that name does not count).
bool IsRocrRuntimeObjectPath(const char *object_path);

#if defined(__linux__)
// True when `addr` lies in a loaded object accepted by
// IsRocrRuntimeObjectPath (_dl_find_object; dladdr on glibc < 2.35).
bool IsRocrRuntimeAddress(const void *addr) noexcept;

// Creator side, called by the interposer for every pthread_create: the core
// to pin the new thread on, or -1 to create it untouched. No logging.
int RocrTameCoreForNewThread(void *(*start_routine)(void *)) noexcept;

// What a thread created by a tamed thread inherited from it, and what it
// would have inherited without the tamer.
struct InheritedState {
  int creator_cpu = -1; // The creator's pin; -1: not from a tamed thread.
  bool explicit_affinity = false; // The attribute asks for exactly that pin.
  cpu_set_t mask{};               // The creator's mask before its pin.
  char name[16] = {};             // The creator's name before its rename.
};

// Runs in the new thread before its start routine: pin, rename, count. No
// cancellation points, no logging, no allocation. `inherited` describes a
// tamed creator (creator_cpu -1 if none).
void TameCurrentRocrThread(int core, const InheritedState &inherited) noexcept;

// Creator side, for every thread being created: fills *inherited. Its
// creator_cpu is the caller's pin, or -1 when the caller is not a tamed
// thread; otherwise also its pre-pin mask and name, and whether `attr`
// explicitly requests exactly that pin.
void CaptureInheritedState(const pthread_attr_t *attr,
                           InheritedState *inherited) noexcept;

// INFERFLUX_ROCR_TAME (read once). When false the interposer passes every
// creation straight through.
bool RocrTamerEnabled() noexcept;

// Runs in a non-ROCr thread created by a tamed thread: restores the
// creator's original name and, unless the attribute asked for the pin, the
// creator's mask before it was pinned.
void RestoreInheritedScheduling(const InheritedState &inherited) noexcept;
#endif

// Called by the interposer's static initializer with its own address, so
// ReportRocrTamerStatus can check the symbol is actually bound.
void NoteRocrPthreadInterposerInstalled(void *interposer_fn) noexcept;

// Called once from main: logs enabled/core/allotment/interposer state and the
// counts so far; warns on a rejected INFERFLUX_ROCR_TAME_CORE, a single-CPU
// allotment, an inactive interposer, or skipped threads. When disabled it
// only notes (once) that ROCr is loaded and the tamer is available.
void ReportRocrTamerStatus();

// Number of threads tamed so far (rendered as
// inferflux_runtime_spin_threads_confined_total).
unsigned long long RocrTamedThreadCount() noexcept;

// ROCr threads left untamed because their own mask had no allotted CPU.
unsigned long long RocrSkippedThreadCount() noexcept;

// Test seam: re-read the env and re-capture the allotment from the calling
// thread.
void ResetRocrTamerForTesting();

} // namespace inferflux
