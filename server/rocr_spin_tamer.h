#pragma once

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
//     immune to host load), spends ~no time in the kernel, makes zero
//     voluntary context switches, and still carries the process name.
//  2. Proof (decides confinement): the thread is interrupted with a probe
//     signal several times; every sampled instruction pointer must resolve
//     (dladdr) into the ROCr runtime library. A busy third-party worker
//     (ggml/OpenMP/BLAS) or first-party thread executes elsewhere and can
//     never be confined, however it is named or loaded.
// If the probe signal is unavailable (foreign handler installed) or the
// architecture is unsupported, nothing is confined.
//
// Linux-only; every entry point is a no-op elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable/disable (default: enabled)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the process affinity mask; pinning skipped
//                             when <= 1 CPU is allowed)
struct RocrSpinTamerOptions {
  // Fast path: skip the scan unless the ROCr runtime is loaded. Tests
  // disable it to exercise the scanner without a ROCm runtime.
  bool require_hsa_runtime = true;
  // A thread is confined only when every sampled instruction pointer lies
  // in a loaded object whose path contains this substring. Tests point it at
  // their own binary.
  std::string code_owner = "libhsa-runtime64";
};

// Synchronous scan: samples for ~1.2 s, confines proven ROCr spinners, and
// returns the number confined (0 when disabled, unsupported, or none found).
int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options = {});

// Requests a background scan and returns immediately; never blocks or
// throws. Called after startup and at the end of every model load attempt
// (success or failure), since any load can be the first to initialize HIP.
// Requests arriving while a scan runs coalesce into exactly one follow-up
// scan. Confined threads are renamed and never re-matched. The worker blocks
// on a condition variable between scans and is joined at static destruction.
void TameRuntimeSpinThreadsAsync();

} // namespace inferflux
