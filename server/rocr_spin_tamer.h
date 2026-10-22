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
// Linux-only; both entry points are no-ops elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable/disable (default: enabled)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the process affinity mask; pinning skipped
//                             when <= 1 CPU is allowed)
//
// Returns the number of threads confined (0 when disabled/unsupported).
int TameRuntimeSpinThreads();

// Fire-and-forget wrapper: runs the tamer off the startup path (the
// classifier needs a ~1.2 s measurement window). Never blocks or throws.
// The bounded worker is joined via static-destruction ordering, so a
// fast-failing startup cannot race static teardown.
void TameRuntimeSpinThreadsAsync();

} // namespace inferflux
