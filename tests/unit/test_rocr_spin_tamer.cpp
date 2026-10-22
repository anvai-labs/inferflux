#include <catch2/catch_amalgamated.hpp>

#include "server/rocr_spin_tamer.h"
#include "support/scoped_env.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#endif

using namespace inferflux;

// Runs on every platform so the [rocr_spin_tamer] tag always matches at
// least one test case (an empty match makes the Catch2 binary exit
// non-zero, which would break the macOS CI ctest entry).
TEST_CASE("rocr spin tamer disabled via env returns 0", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar disable("INFERFLUX_ROCR_TAME", "0");
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  REQUIRE(inferflux::TameRuntimeSpinThreads(options) == 0);
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

// Unnamed pure-userspace spinners (zero syscalls, zero voluntary context
// switches) that stop and join on scope exit — including when a REQUIRE
// throws, so a failed assertion can never destroy a joinable std::thread
// (which would std::terminate the whole test binary).
class SpinnerGroup {
public:
  explicit SpinnerGroup(int count) {
    tids_ = std::make_unique<std::atomic<pid_t>[]>(count);
    for (int i = 0; i < count; ++i) {
      tids_[i].store(0);
      threads_.emplace_back([this, i] {
        tids_[i].store(CurrentTid());
        while (!stop_.load(std::memory_order_relaxed)) {
        }
      });
    }
    for (int i = 0; i < count; ++i) {
      while (tids_[i].load() == 0) {
        std::this_thread::yield();
      }
    }
  }

  ~SpinnerGroup() {
    stop_.store(true);
    for (auto &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  SpinnerGroup(const SpinnerGroup &) = delete;
  SpinnerGroup &operator=(const SpinnerGroup &) = delete;

  pid_t tid(int index) const { return tids_[index].load(); }

private:
  std::atomic<bool> stop_{false};
  std::unique_ptr<std::atomic<pid_t>[]> tids_;
  std::vector<std::thread> threads_;
};

RocrSpinTamerOptions WithoutHsaGate() {
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  return options;
}

} // namespace

TEST_CASE("rocr spin tamer confines a pure busy-spin thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  SpinnerGroup spinners(1);
  const int ncpu = CountAllowedCpus();
  const int expected_core = HighestAllowedCpu();

  // Retry once: on a loaded few-core CI runner the spinner may miss the
  // 80%-of-window utime threshold on the first measurement.
  int confined = 0;
  for (int attempt = 0; attempt < 2 && confined == 0; ++attempt) {
    confined = inferflux::TameRuntimeSpinThreads(WithoutHsaGate());
  }
  REQUIRE(confined == 1);

  if (ncpu > 1) {
    cpu_set_t now;
    CPU_ZERO(&now);
    REQUIRE(sched_getaffinity(spinners.tid(0), sizeof(now), &now) == 0);
    REQUIRE(CPU_COUNT(&now) == 1);
    REQUIRE(CPU_ISSET(expected_core, &now));
  }
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
  // Join before asserting so a failure cannot leave a joinable thread.
  const int confined = inferflux::TameRuntimeSpinThreads(WithoutHsaGate());
  stop.store(true);
  sleeper.join();

  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer skips processes without the ROCr runtime",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  // The test binary never loads libhsa-runtime64, so with the default gate
  // even a real spinner must be left alone.
  SpinnerGroup spinners(1);
  REQUIRE(inferflux::TameRuntimeSpinThreads() == 0);
}

TEST_CASE("rocr spin tamer confines nothing when too many threads match",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  // Five spinners need five uncontended CPUs to each clear the busy
  // threshold; on smaller runners the burst cannot be produced reliably.
  if (CountAllowedCpus() < 8) {
    SKIP("needs >= 8 allowed CPUs to produce a deterministic burst");
  }
  SpinnerGroup spinners(5);
  REQUIRE(inferflux::TameRuntimeSpinThreads(WithoutHsaGate()) == 0);
}
#endif // defined(__linux__)
