#include <catch2/catch_amalgamated.hpp>

#include "server/rocr_spin_tamer.h"
#include "support/scoped_env.h"

#if defined(__linux__)
#include <dlfcn.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
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

// Path of the loaded object holding this test's code (the test binary), as
// dladdr reports it — the same resolution the tamer applies to sampled PCs.
std::string ThisTestBinary() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(&ThisTestBinary), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return info.dli_fname;
}

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
// switches; the loop's code lives in the test binary) that stop and join on
// scope exit — including when a REQUIRE throws, so a failed assertion can
// never destroy a joinable std::thread (which would std::terminate the whole
// test binary).
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

// Scan without the ROCr-runtime fast path, treating `owner` as the code
// owner. Pointing it at the test binary also makes any real ROCr pollers in
// a ROCm-build test process ineligible, keeping these cases hermetic.
RocrSpinTamerOptions OwnedBy(std::string owner) {
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  options.code_owner = std::move(owner);
  return options;
}

} // namespace

TEST_CASE("rocr spin tamer confines a spinner proven to run owner code",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  const std::string owner = ThisTestBinary();
  REQUIRE_FALSE(owner.empty());

  SpinnerGroup spinners(1);
  const int ncpu = CountAllowedCpus();
  const int expected_core = HighestAllowedCpu();

  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedBy(owner)) == 1);

  if (ncpu > 1) {
    cpu_set_t now;
    CPU_ZERO(&now);
    REQUIRE(sched_getaffinity(spinners.tid(0), sizeof(now), &now) == 0);
    REQUIRE(CPU_COUNT(&now) == 1);
    REQUIRE(CPU_ISSET(expected_core, &now));
  }
}

TEST_CASE("rocr spin tamer refuses spinners running other code",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  // Several spinners pass the never-blocks prefilter, but none executes in
  // the code owner, so the PC proof must reject all of them — the case of a
  // busy third-party CPU threadpool in a ROCm process.
  SpinnerGroup spinners(3);
  REQUIRE(inferflux::TameRuntimeSpinThreads(
              OwnedBy("no-such-object-owns-this-code")) == 0);
}

TEST_CASE("rocr spin tamer ignores sleeping threads", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  const std::string owner = ThisTestBinary();
  REQUIRE_FALSE(owner.empty());

  std::atomic<bool> stop{false};
  std::thread sleeper([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  // Join before asserting so a failure cannot leave a joinable thread.
  const int confined = inferflux::TameRuntimeSpinThreads(OwnedBy(owner));
  stop.store(true);
  sleeper.join();

  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer does not re-confine an already confined thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  const std::string owner = ThisTestBinary();
  REQUIRE_FALSE(owner.empty());

  SpinnerGroup spinners(1);
  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedBy(owner)) == 1);
  // The first scan renamed it rocr-spin-N; repeat scans (one per model load)
  // must leave it alone rather than re-confine and double count.
  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedBy(owner)) == 0);
}
#endif // defined(__linux__)
