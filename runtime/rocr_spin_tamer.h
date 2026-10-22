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
//     immune to host load), spends ~no time in the kernel, and makes zero
//     voluntary context switches. Thread names are ignored: Linux threads
//     inherit their creator's name.
//  2. Proof (decides confinement): the thread is interrupted with a probe
//     signal up to 3 times and confined only if at least 2 sampled
//     instruction pointers resolve (dladdr) into the ROCr runtime library.
//     Threads that block or ignore the probe signal are never signalled.
//     A thread proven not to be ROCr is remembered and never probed again,
//     so a busy third-party worker (ggml/OpenMP/BLAS) receives at most two
//     SA_RESTART probe signals per lifetime and is never confined.
// Confined threads are remembered by (tid, start time); a TID reused before
// confinement is detected by re-checking the start time. If the probe
// signal is owned by another handler or the architecture is unsupported,
// nothing is confined.
//
// Linux-only; every entry point is a no-op elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable/disable (default: enabled)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the process affinity mask; invalid values warn
//                             and fall back; pinning skipped when <= 1 CPU
//                             is allowed)
struct RocrSpinTamerOptions {
  // Fast path: skip the scan unless the ROCr runtime is loaded. Tests
  // disable it to exercise the scanner without a ROCm runtime.
  bool require_hsa_runtime = true;
  // A sampled instruction pointer counts as the owner's when it lies in a
  // loaded object whose path contains this substring.
  std::string code_owner = "libhsa-runtime64";
  // Test seam: when set, decides PC ownership instead of `code_owner`, and
  // proven-not-owner verdicts are not remembered (they are owner-specific).
  std::function<bool(std::uintptr_t pc)> pc_owner;
};

// Synchronous scan: samples for ~1.2 s, confines proven ROCr spinners, and
// returns the number confined (0 when disabled, unsupported, or none found).
int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options = {});

// Requests a background scan and returns immediately; never blocks or
// throws, from any thread. Called at the end of every model load attempt
// (success or failure), since any load can be the first to initialize HIP.
// After a requested scan the worker rescans once ~10 s later and then every
// ~5 min while the ROCr runtime stays loaded, catching pollers created
// lazily after a load. Requests arriving mid-scan coalesce into one
// follow-up scan. Between scans the worker blocks on a condition variable.
// It is never destroyed: an atexit hook stops and joins it, and later
// requests are ignored — safe even when a load races process exit.
void TameRuntimeSpinThreadsAsync();

} // namespace inferflux
