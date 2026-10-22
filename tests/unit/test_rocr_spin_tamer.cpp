#include <catch2/catch_amalgamated.hpp>

#include "server/rocr_spin_tamer.h"
#include "support/scoped_env.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>
#endif

using namespace inferflux;

// Runs on every platform so the [rocr_spin_tamer] tag always matches at
// least one test case (an empty match makes the Catch2 binary exit
// non-zero, which would break the macOS CI ctest entry).
TEST_CASE("rocr spin tamer disabled via env returns 0", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar disable("INFERFLUX_ROCR_TAME", "0");
  REQUIRE(inferflux::TameRuntimeSpinThreads() == 0);
}

#if defined(__linux__)
namespace {

pid_t CurrentTid() { return static_cast<pid_t>(syscall(SYS_gettid)); }

int HighestAllowedCpu() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return -1;
  }
  int highest = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      highest = cpu;
    }
  }
  return highest;
}

int CountAllowedCpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return 0;
  }
  return CPU_COUNT(&set);
}

// Best-effort hygiene: some kernels restrict unprivileged SCHED_IDLE ->
// SCHED_OTHER transitions, so restoration results are not asserted.
void RestoreThread(pid_t tid, const cpu_set_t &original) {
  sched_param other{};
  (void)sched_setscheduler(tid, SCHED_OTHER, &other);
  (void)setpriority(PRIO_PROCESS, tid, 0);
  (void)sched_setaffinity(tid, sizeof(original), &original);
}

} // namespace

TEST_CASE("rocr spin tamer confines a pure busy-spin thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  std::atomic<bool> stop{false};
  std::atomic<pid_t> spinner_tid{0};
  cpu_set_t original_mask;
  CPU_ZERO(&original_mask);

  std::thread spinner([&] {
    // Capture own affinity first, then publish the tid: the seq_cst
    // store/load pair makes the mask visible to the main thread.
    cpu_set_t self_mask;
    CPU_ZERO(&self_mask);
    if (sched_getaffinity(0, sizeof(self_mask), &self_mask) == 0) {
      original_mask = self_mask;
    }
    spinner_tid.store(CurrentTid());
    while (!stop.load(std::memory_order_relaxed)) {
      // Zero syscalls, zero voluntary context switches.
    }
  });
  while (spinner_tid.load() == 0) {
    std::this_thread::yield();
  }
  const pid_t tid = spinner_tid.load();

  const int ncpu = CountAllowedCpus();
  const int expected_core = HighestAllowedCpu();

  // Retry once: on a loaded few-core CI runner the spinner may miss the
  // 80%-of-window utime threshold on the first measurement.
  int confined = 0;
  for (int attempt = 0; attempt < 2 && confined == 0; ++attempt) {
    confined = inferflux::TameRuntimeSpinThreads();
  }
  REQUIRE(confined == 1);

  if (ncpu > 1) {
    cpu_set_t now;
    CPU_ZERO(&now);
    REQUIRE(sched_getaffinity(tid, sizeof(now), &now) == 0);
    REQUIRE(CPU_COUNT(&now) == 1);
    REQUIRE(CPU_ISSET(expected_core, &now));
  }

  RestoreThread(tid, original_mask);
  stop.store(true);
  spinner.join();
}

TEST_CASE("rocr spin tamer ignores sleeping threads", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  std::atomic<bool> stop{false};
  std::thread sleeper([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  REQUIRE(inferflux::TameRuntimeSpinThreads() == 0);

  stop.store(true);
  sleeper.join();
}
#endif // defined(__linux__)
