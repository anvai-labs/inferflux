#pragma once

#include <vector>

namespace inferflux {

// ROCr thread tamer (WSL2/ROCm AsyncEventsLoop workaround).
//
// ROCm 7.x libhsa-runtime64 starts AsyncEventsLoop poller threads at
// hipInit() that busy-spin at 100% CPU each for the process lifetime on
// WSL2/gfx1201 (upstream fix rocm-systems#7898 unreleased; no HSA_* knob
// disables the loop). The tamer bounds that cost to one core.
//
// Identification is exact and happens at thread creation: inferfluxd defines
// pthread_create (server/rocr_pthread_interposer.cpp, linked only into the
// server binary), and every thread whose start routine lies in
// libhsa-runtime64 is started through a trampoline that, in the new thread,
// pins it to one shared core, sets nice 19 and names it "rocr-hsa-N". No
// signals, no /proc scans, no background thread. Every libhsa-created thread
// is tamed (they all start through one ROCr-internal routine, so pollers
// cannot be told apart from ROCr's other, mostly blocked, threads).
//
// Why a pin applied in the new thread: ROCr passes an explicit all-CPU
// affinity through pthread_attr, so a pinned creator's mask is not
// inherited, and disabling ROCr's override (HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0)
// under a pinned creator aborts. libhsa never changes a thread's affinity
// after creation (its only affinity import is pthread_attr_setaffinity_np).
// Why nice 19 and not SCHED_IDLE: the same thread runs HSA async signal
// handlers, and native Linux enforces SCHED_IDLE strictly enough to starve
// it; nice 19 still gets a share of every CFS period. On WSL2 neither is
// enforced for service-cgroup threads; the pin is the load-bearing bound.
//
// Core choice: the server's affinity allotment is the main thread's mask,
// captured during static initialization (the operator's taskset/cgroup).
// Fewer than two allotted CPUs -> never pin (two spinners must not be piled
// onto the server's only CPU). Otherwise a valid INFERFLUX_ROCR_TAME_CORE
// inside the allotment, else the highest allotted CPU.
//
// Residuals: a cgroup cpuset later shrunk to exclude the core moves the
// threads off it; a future ROCr starting threads via std::thread (start
// routine in libstdc++) would not be recognized (fails safe: nothing tamed);
// sanitizer builds compile the interposer out.
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
// IsRocrRuntimeObjectPath (dladdr).
bool IsRocrRuntimeAddress(const void *addr) noexcept;

// Creator side, called by the interposer for every pthread_create: the core
// to pin the new thread on, or -1 to create it untouched. No logging.
int RocrTameCoreForNewThread(void *(*start_routine)(void *)) noexcept;

// Runs in the new thread before its start routine: pin, nice 19, rename,
// count. No cancellation points, no logging, no allocation.
void TameCurrentRocrThread(int core) noexcept;
#endif

// Called by the interposer's static initializer with its own address, so
// ReportRocrTamerStatus can check the symbol is actually bound.
void NoteRocrPthreadInterposerInstalled(void *interposer_fn) noexcept;

// Called once from main: logs enabled/core/allotment/interposer state and the
// count so far; warns on a rejected INFERFLUX_ROCR_TAME_CORE, a single-CPU
// allotment, or an inactive interposer. No-op when disabled.
void ReportRocrTamerStatus();

// Number of threads tamed so far (also counted in GlobalMetrics()).
unsigned long long RocrTamedThreadCount() noexcept;

// Test seam: re-read the env and re-capture the allotment from the calling
// thread.
void ResetRocrTamerForTesting();

} // namespace inferflux
