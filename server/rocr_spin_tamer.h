#pragma once

namespace inferflux {

// Confine ROCm/ROCr HSA-runtime busy-spin poller threads (AsyncEventsLoop).
//
// ROCm 7.x libhsa-runtime64 spawns ~2 unnamed poller threads at hipInit()
// that busy-spin at 100% CPU each for the process lifetime (upstream fix
// rocm-systems#7898 unreleased; no env knob disables the loop). The tamer
// detects them by signature (unnamed comm, ~100% utime, ~zero syscalls,
// zero voluntary context switches over a measurement window) and demotes
// them (nice 19 + SCHED_IDLE) and co-pins them onto a single core, bounding
// the cost from N cores to 1. On WSL2 SCHED_IDLE/nice are not enforced, so
// the affinity pin is the load-bearing mitigation there.
//
// Soundness guards (a misclassified thread is demoted for its lifetime):
//  - Runs only when libhsa-runtime64 is mapped into the process; CPU-only
//    and CUDA-only processes are never touched.
//  - Only threads still carrying the process name are candidates. Every
//    first-party thread names itself on entry (runtime/thread_name.h), so
//    only third-party runtime threads remain eligible.
//  - If more than kMaxConfined threads match (e.g. a busy CPU threadpool
//    rather than the two pollers), nothing is confined and a warning is
//    logged.
//
// Linux-only; every entry point is a no-op elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable/disable (default: enabled)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the process affinity mask; pinning skipped
//                             when <= 1 CPU is allowed)
struct RocrSpinTamerOptions {
  // Skip the scan unless libhsa-runtime64 is mapped. Tests disable this to
  // exercise the classifier without a ROCm runtime.
  bool require_hsa_runtime = true;
};

// Synchronous scan: samples for ~1.2 s, confines matching threads, and
// returns the number confined (0 when disabled, unsupported, no ROCr
// runtime is loaded, or the burst guard tripped).
int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options = {});

// Requests a background scan and returns immediately; never blocks or
// throws. Called after startup and after every successful model load, since
// a model loaded at runtime can be the first to initialize HIP. Requests
// arriving while a scan runs coalesce into exactly one follow-up scan, so
// pollers spawned mid-window are still caught. Threads already confined are
// renamed and therefore never re-matched. The worker blocks on a condition
// variable between scans and is joined at static destruction.
void TameRuntimeSpinThreadsAsync();

} // namespace inferflux
