#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace inferflux {

// Confine ROCm/ROCr HSA-runtime busy-spin poller threads (AsyncEventsLoop).
//
// ROCm 7.x libhsa-runtime64 spawns two poller threads at hipInit() that
// busy-spin at 100% CPU each for the process lifetime (upstream fix
// rocm-systems#7898 unreleased; no env knob disables the loop). The tamer
// demotes them (nice 19 + SCHED_IDLE) and co-pins them onto one core,
// bounding the cost from 2 cores to 1. On WSL2 SCHED_IDLE/nice are not
// enforced, so the affinity pin is the load-bearing mitigation there.
//
// Identification is positive, not heuristic, because confinement is
// irreversible:
//  1. Prefilter (selects threads to probe): over a ~1.2 s window the thread
//     never blocks (runnable for >= 80% of the window by schedstat, which is
//     immune to host load; user time when schedstat is absent or inert),
//     spends ~no time in the kernel, and makes zero voluntary context
//     switches. Thread names are ignored: threads inherit their creator's.
//  2. Proof (decides confinement): the thread is interrupted with a probe
//     signal up to 3 times and confined only if at least 2 sampled
//     instruction pointers resolve (dladdr) into the ROCr runtime library.
//     A thread is signalled only while it is running/runnable and its
//     signal mask neither blocks nor ignores the probe, so a thread parked
//     in a syscall never sees EINTR from it.
// A thread not proven (other code, or no reply) backs off for 1, 2, 4, ...
// up to 12 scans before it is probed again, so a busy third-party worker
// (ggml/OpenMP/BLAS) receives at most 2 SA_RESTART probe signals per
// back-off period and is never confined, while a real poller that was
// unlucky once is still caught later. Confined threads are remembered by
// (tid, start time); a TID reused before confinement is detected by
// re-checking the start time. At most 8 live threads are confined at once
// (slots free up when confined threads exit). If the probe signal is owned by
// another handler or the architecture is unsupported, nothing is confined.
//
// Linux-only; every entry point is a no-op elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable/disable (default: enabled)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the process affinity mask; invalid values warn
//                             and fall back; pinning skipped when <= 1 CPU
//                             is allowed)

// The PC probe uses SIGRTMIN + this offset (exposed for tests).
constexpr int kRocrTamerProbeSignalOffset = 5;

struct RocrSpinTamerOptions {
  // Fast path: skip the scan unless the ROCr runtime is loaded. Tests
  // disable it to exercise the scanner without a ROCm runtime.
  bool require_hsa_runtime = true;
  // A sampled instruction pointer counts as the owner's when it lies in a
  // loaded object whose path contains this substring.
  std::string code_owner = "libhsa-runtime64";
  // Test seam: when set, decides PC ownership instead of `code_owner`.
  std::function<bool(std::uintptr_t pc)> pc_owner;
};

// Synchronous scan: samples for ~1.2 s, confines proven ROCr spinners, and
// returns the number confined (0 when disabled, unsupported, or none found).
int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options = {});

// Starts the background tamer once (idempotent; never blocks or throws).
// Call after startup has initialized backends. The worker scans
// immediately, again after ~10 s and ~60 s, then every ~5 min for the
// process lifetime, so pollers created lazily or by a model loaded at
// runtime are caught within one period. Starts nothing when
// INFERFLUX_ROCR_TAME is disabled or libhsa-runtime64 is not loaded (it is
// linked into ROCm builds, so CPU-only and CUDA-only builds get no thread). The
// worker blocks on a condition variable between scans and is never destroyed:
// an atexit hook stops and joins it before the logger and metrics registry are
// torn down.
void StartRuntimeSpinTamer() noexcept;

} // namespace inferflux
