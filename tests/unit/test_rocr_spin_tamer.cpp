#include <catch2/catch_amalgamated.hpp>

#include "runtime/thread_name.h"
#include "server/rocr_spin_tamer.h"
#include "support/scoped_env.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
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
} // namespace

// The spin loop lives alone in its own linker section; the linker-provided
// __start_/__stop_ symbols bound exactly its code, giving the tests a precise
// "code owner" that no other thread in the process can ever execute in.
extern "C" {
extern const char __start_rocr_tamer_test_spin[];
extern const char __stop_rocr_tamer_test_spin[];
}

__attribute__((noinline, section("rocr_tamer_test_spin"))) void
RocrTamerTestSpinUntil(const std::atomic<bool> *stop) {
  while (!stop->load(std::memory_order_relaxed)) {
  }
}

namespace {

bool InTestSpinLoop(std::uintptr_t pc) {
  return pc >= reinterpret_cast<std::uintptr_t>(__start_rocr_tamer_test_spin) &&
         pc < reinterpret_cast<std::uintptr_t>(__stop_rocr_tamer_test_spin);
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

// Pure-userspace spinners (zero syscalls, zero voluntary context switches,
// executing only RocrTamerTestSpinUntil) that stop and join on scope exit —
// including when a REQUIRE throws, so a failed assertion can never destroy a
// joinable std::thread (which would std::terminate the whole test binary).
class SpinnerGroup {
public:
  explicit SpinnerGroup(int count, const char *name = nullptr) {
    tids_ = std::make_unique<std::atomic<pid_t>[]>(count);
    for (int i = 0; i < count; ++i) {
      tids_[i].store(0);
      threads_.emplace_back([this, i, name] {
        if (name) {
          inferflux::SetCurrentThreadName(name);
        }
        tids_[i].store(CurrentTid());
        RocrTamerTestSpinUntil(&stop_);
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

// Scan without the ROCr-runtime fast path, with PC ownership decided by
// `owner`. Owning only the test spin loop also makes any real ROCr pollers
// in a ROCm-build test process ineligible, keeping these cases hermetic.
RocrSpinTamerOptions OwnedBy(std::function<bool(std::uintptr_t)> owner) {
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  options.pc_owner = std::move(owner);
  return options;
}

// Retries absorb a loaded few-core CI runner where the spinner misses the
// window threshold once; the total must still be exactly `expected`.
int ScanUntil(int expected, const RocrSpinTamerOptions &options) {
  int total = 0;
  for (int attempt = 0; attempt < 3 && total < expected; ++attempt) {
    total += inferflux::TameRuntimeSpinThreads(options);
  }
  return total;
}

} // namespace

TEST_CASE("rocr spin tamer confines a spinner proven to run owner code",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  SpinnerGroup spinners(1);
  const int ncpu = CountAllowedCpus();
  const int expected_core = HighestAllowedCpu();

  REQUIRE(ScanUntil(1, OwnedBy(InTestSpinLoop)) == 1);

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
              OwnedBy([](std::uintptr_t) { return false; })) == 0);
}

TEST_CASE("rocr spin tamer ignores sleeping threads", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);

  std::atomic<bool> stop{false};
  std::thread sleeper([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  // Join before asserting so a failure cannot leave a joinable thread.
  const int confined =
      inferflux::TameRuntimeSpinThreads(OwnedBy(InTestSpinLoop));
  stop.store(true);
  sleeper.join();

  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer does not re-confine an already confined thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  SpinnerGroup spinners(1);
  REQUIRE(ScanUntil(1, OwnedBy(InTestSpinLoop)) == 1);
  // Repeat scans (one per model load) must leave it alone rather than
  // re-confine and double count; it is remembered by (tid, start time).
  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedBy(InTestSpinLoop)) == 0);
}
TEST_CASE("rocr spin tamer confines a poller that inherited a thread name",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", nullptr);
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  // Linux threads inherit their creator's name: ROCr pollers spawned while a
  // registry reload or admin load runs on "if-registry"/"if-http" carry that
  // name. Names must not exclude them; only the PC proof decides.
  SpinnerGroup spinners(1, "if-registry");
  REQUIRE(ScanUntil(1, OwnedBy(InTestSpinLoop)) == 1);
}
#endif // defined(__linux__)
