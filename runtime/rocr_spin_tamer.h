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
// co-pins them onto one core (bounding the cost from 2 cores to 1) and sets
// nice 19. Not SCHED_IDLE: the same thread runs HSA async signal handlers,
// and native Linux enforces SCHED_IDLE strictly enough to starve it; a
// nice-19 thread still gets a share of every CFS period. (WSL2 enforces
// neither for service-cgroup threads; the pin is the load-bearing part.)
// Pinning the caller around hipInit() instead is not viable: ROCr widens its
// threads' affinity to every CPU, and disabling that override
// (HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0) under a pinned caller aborts.
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
//     Before each signal the thread must still be the same thread (start
//     time), running/runnable, and neither block nor ignore the probe.
//     Residual risk, minimized not eliminated: the state check and the
//     signal are not atomic, and a probe that times out stays pending, so a
//     non-ROCr spinner that enters a syscall in that window can see one
//     EINTR (SA_RESTART covers most calls; epoll/poll/timed waits do not).
// A thread not proven (other code, or no reply) backs off for 1, 2, 4, ...
// up to 12 scans before it is probed again, so a busy third-party worker
// (ggml/OpenMP/BLAS) receives at most 2 SA_RESTART probe signals per
// back-off period and is never confined, while a real poller that was
// unlucky once is still caught later. Confined threads are remembered by
// (tid, start time); a TID reused before confinement is detected by
// re-checking the start time. At most 8 live threads are confined at once
// (slots free up when confined threads exit). All confined threads share
// one pin core, chosen once from the server's affinity mask (captured before
// main runs, so it is the operator's taskset/cgroup allotment; ROCr widens
// its own threads to every CPU, so their masks say nothing about it). With
// no usable allotment, or an allotment of a single CPU, nothing is pinned
// (two 100% spinners must not be piled onto the server's only CPU). A
// thread whose mask excludes the chosen core is left unconfined rather than
// pinned elsewhere; a
// thread counts as confined only once pinned (the pin is the bound; nice is
// not enforced on WSL2). A proven poller that cannot be confined backs off
// like an unproven thread instead of being re-probed every scan. If the
// probe signal is owned by another handler (or set to SIG_IGN) or the
// architecture is unsupported, nothing is confined.
//
// Known limit (why the tamer is opt-in): the proof shows a thread executes
// ROCr code, not which ROCr role it plays. InferFlux's own threads are
// excluded (runtime/thread_name.h), but another runtime-internal thread that
// busy-waits inside libhsa-runtime64 for a whole window would be confined
// too. Validated on WSL2/gfx1201 under real inference: only the two pollers
// were ever confined.
//
// Linux-only; every entry point is a no-op elsewhere.
//
// Knobs (env-only, see docs/CONFIG_REFERENCE.md section 10):
//   INFERFLUX_ROCR_TAME       enable (default: disabled; opt-in with 1)
//   INFERFLUX_ROCR_TAME_CORE  pin target CPU index (default: highest CPU in
//                             the server's affinity mask; values outside it
//                             warn and fall back; pinning skipped when a
//                             thread may run on only one CPU)

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

// Test seam: forgets the process-wide pin core and captured server mask, so
// a test can exercise core selection independently of earlier tests.
void ResetRuntimeSpinTamerPinForTesting();

// Synchronous scan: samples for ~1.2 s, confines proven ROCr spinners, and
// returns the number confined (0 when disabled, unsupported, or none found).
int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options = {});

// Starts the background tamer once (idempotent; never blocks or throws).
// Call after startup has initialized backends. The worker scans
// immediately, again after ~10 s and ~60 s, then every ~5 min for the
// process lifetime, so pollers created lazily or by a model loaded at
// runtime are usually caught within one period (a thread whose probe was
// unproven or timed out backs off, so it can take up to ~1 h). Each scan
// also restores the pin of any confined thread whose affinity was widened
// again. Starts nothing unless
// INFERFLUX_ROCR_TAME=1, or when libhsa-runtime64 is not loaded (it is
// linked into ROCm builds, so CPU-only and CUDA-only builds get no thread). The
// worker blocks on a condition variable between scans and is never destroyed:
// an atexit hook stops and joins it before the logger and metrics registry are
// torn down.
void StartRuntimeSpinTamer() noexcept;

} // namespace inferflux
