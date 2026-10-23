#include <catch2/catch_amalgamated.hpp>

#include "runtime/rocr_spin_tamer.h"
#include "runtime/thread_name.h"
#include "support/scoped_env.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#endif

using namespace inferflux;

// Runs on every platform so the [rocr_spin_tamer] tag always matches at
// least one test case (an empty match makes the Catch2 binary exit
// non-zero, which would break the macOS CI ctest entry).
TEST_CASE("rocr spin tamer is off unless opted in", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar unset("INFERFLUX_ROCR_TAME", nullptr);
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  REQUIRE(inferflux::TameRuntimeSpinThreads(options) == 0);
}

TEST_CASE("rocr spin tamer disabled via env returns 0", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar disable("INFERFLUX_ROCR_TAME", "0");
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  REQUIRE(inferflux::TameRuntimeSpinThreads(options) == 0);
}

#if defined(__linux__)

// The spin loop lives alone in its own linker section; the linker-provided
// __start_/__stop_ symbols bound exactly its code, giving the tests a precise
// "code owner" that no other thread in the process can ever execute in.
extern "C" {
extern const char __start_rocr_tamer_test_spin[];
extern const char __stop_rocr_tamer_test_spin[];
}

namespace {

__attribute__((noinline, section("rocr_tamer_test_spin"))) void
RocrTamerTestSpinUntil(const bool *stop) {
  // A GCC atomic builtin, never an out-of-line call: with std::atomic at
  // -O0 (the Coverage CI job) load() is a real function outside this
  // section, so the sampled PCs would not land in the owned range.
  while (!__atomic_load_n(stop, __ATOMIC_RELAXED)) {
  }
}

bool InTestSpinLoop(std::uintptr_t pc) {
  return pc >= reinterpret_cast<std::uintptr_t>(__start_rocr_tamer_test_spin) &&
         pc < reinterpret_cast<std::uintptr_t>(__stop_rocr_tamer_test_spin);
}

int ProbeSignal() { return SIGRTMIN + kRocrTamerProbeSignalOffset; }

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

int LowestAllowedCpu() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return -1;
  }
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      return cpu;
    }
  }
  return -1;
}

int CountAllowedCpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return 0;
  }
  return CPU_COUNT(&set);
}

// True when `tid` has the probe signal pending (queued but undelivered).
bool ProbePending(pid_t tid) {
  std::ifstream file("/proc/self/task/" + std::to_string(tid) + "/status");
  std::string line;
  while (std::getline(file, line)) {
    if (line.rfind("SigPnd:", 0) == 0) {
      const unsigned long long pending =
          std::strtoull(line.c_str() + 7, nullptr, 16);
      return (pending & (1ULL << (ProbeSignal() - 1))) != 0;
    }
  }
  return false;
}

// Pure-userspace spinners (zero syscalls, zero voluntary context switches,
// executing only RocrTamerTestSpinUntil) that stop and join on scope exit —
// including when a REQUIRE throws, so a failed assertion can never destroy a
// joinable std::thread (which would std::terminate the whole test binary).
class SpinnerGroup {
public:
  struct Options {
    // Thread name set with raw pthread_setname_np (simulates a name inherited
    // from a creator, without registering the thread as first-party).
    const char *inherited_name = nullptr;
    bool first_party = false; // MarkFirstPartyThread() before spinning.
    bool block_probe = false; // Block the probe signal before spinning.
  };

  explicit SpinnerGroup(int count) : SpinnerGroup(count, Options{}) {}

  SpinnerGroup(int count, Options options) {
    tids_ = std::make_unique<std::atomic<pid_t>[]>(count);
    for (int i = 0; i < count; ++i) {
      tids_[i].store(0);
      threads_.emplace_back([this, i, options] {
        if (options.inherited_name) {
          pthread_setname_np(pthread_self(), options.inherited_name);
        }
        if (options.first_party) {
          inferflux::MarkFirstPartyThread();
        }
        if (options.block_probe) {
          sigset_t set;
          sigemptyset(&set);
          sigaddset(&set, ProbeSignal());
          pthread_sigmask(SIG_BLOCK, &set, nullptr);
        }
        tids_[i].store(inferflux::CurrentThreadId());
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
    __atomic_store_n(&stop_, true, __ATOMIC_RELAXED);
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
  bool stop_ = false; // Accessed only through __atomic builtins.
  std::unique_ptr<std::atomic<pid_t>[]> tids_;
  std::vector<std::thread> threads_;
};

// Ownership predicate that counts replied samples from the test spin loop,
// so tests can prove the probe really ran. PCs from any other thread in the
// test process are answered "not owned" without being counted, so stray
// threads cannot disturb the counts. Answers "owned" for the first `owned`
// counted calls and "not owned" after that.
struct CountingOwner {
  std::shared_ptr<std::atomic<int>> calls =
      std::make_shared<std::atomic<int>>(0);
  int owned = 0;

  RocrSpinTamerOptions Options() const {
    RocrSpinTamerOptions options;
    options.require_hsa_runtime = false;
    auto counter = calls;
    const int limit = owned;
    options.pc_owner = [counter, limit](std::uintptr_t pc) {
      if (!InTestSpinLoop(pc)) {
        return false;
      }
      return counter->fetch_add(1) < limit;
    };
    return options;
  }
};

RocrSpinTamerOptions OwnedByTestLoop() {
  RocrSpinTamerOptions options;
  options.require_hsa_runtime = false;
  options.pc_owner = InTestSpinLoop;
  return options;
}

// Retries absorb a loaded few-core CI runner where the spinner misses the
// window threshold; the total must still be exactly `expected`.
int ScanUntil(int expected, const RocrSpinTamerOptions &options,
              int max_attempts = 4) {
  int total = 0;
  for (int attempt = 0; attempt < max_attempts && total < expected; ++attempt) {
    total += inferflux::TameRuntimeSpinThreads(options);
  }
  return total;
}

} // namespace

TEST_CASE("rocr spin tamer confines a spinner proven to run owner code",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);

  inferflux::ResetRuntimeSpinTamerPinForTesting();
  SpinnerGroup spinners(1);
  const int ncpu = CountAllowedCpus();
  const int expected_core = HighestAllowedCpu();

  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);

  // Renamed exactly "rocr-spin-N": the kernel stores written bytes verbatim,
  // so a stray newline would end up inside the name.
  std::ifstream comm_file("/proc/self/task/" + std::to_string(spinners.tid(0)) +
                          "/comm");
  const std::string comm((std::istreambuf_iterator<char>(comm_file)),
                         std::istreambuf_iterator<char>());
  REQUIRE(comm.rfind("rocr-spin-", 0) == 0);
  REQUIRE(comm.back() == '\n');
  REQUIRE(comm.find('\n') == comm.size() - 1);

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
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  // Spinners pass the never-blocks prefilter and are probed (the predicate
  // is consulted), but none is proven to run owner code — the case of a busy
  // third-party CPU threadpool in a ROCm process.
  SpinnerGroup spinners(3);
  CountingOwner never_owned; // owned = 0
  int confined = 0;
  for (int attempt = 0; attempt < 4 && never_owned.calls->load() == 0;
       ++attempt) {
    confined += inferflux::TameRuntimeSpinThreads(never_owned.Options());
  }
  REQUIRE(never_owned.calls->load() > 0);
  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer needs a majority of samples in the owner",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  {
    // Only 1 of 3 samples owned: probed, rejected.
    SpinnerGroup spinners(1);
    CountingOwner one_owned;
    one_owned.owned = 1;
    int confined = 0;
    for (int attempt = 0; attempt < 4 && one_owned.calls->load() == 0;
         ++attempt) {
      confined += inferflux::TameRuntimeSpinThreads(one_owned.Options());
    }
    REQUIRE(one_owned.calls->load() >= 1);
    REQUIRE(confined == 0);
  }
  {
    // 2 of 3 owned: confined once 2 replied samples are owned (a sample
    // that times out never reaches the predicate). Each attempt gets a fresh
    // predicate, and there is headroom for a back-off skip if one attempt
    // ends unproven on a loaded runner.
    SpinnerGroup spinners(1);
    int confined = 0;
    int calls_on_success = 0;
    for (int attempt = 0; attempt < 6 && confined == 0; ++attempt) {
      CountingOwner two_owned;
      two_owned.owned = 2;
      confined = inferflux::TameRuntimeSpinThreads(two_owned.Options());
      calls_on_success = two_owned.calls->load();
    }
    REQUIRE(confined == 1);
    REQUIRE(calls_on_success == 2);
  }
}

TEST_CASE("rocr spin tamer ignores sleeping threads", "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  std::atomic<bool> stop{false};
  std::thread sleeper([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  // Join before asserting so a failure cannot leave a joinable thread.
  CountingOwner any_owner;
  any_owner.owned = 1000;
  const int confined = inferflux::TameRuntimeSpinThreads(any_owner.Options());
  stop.store(true);
  sleeper.join();

  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer does not re-confine an already confined thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  SpinnerGroup spinners(1);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
  // Later scans must leave it alone rather than re-confine and double count;
  // it is remembered by (tid, start time).
  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedByTestLoop()) == 0);
}

TEST_CASE("rocr spin tamer confines a poller that inherited a thread name",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  // Linux threads inherit their creator's name: ROCr pollers spawned while a
  // registry reload or admin load runs on "if-registry"/"if-http" carry that
  // name. Names must not exclude them; only the PC proof decides.
  SpinnerGroup::Options named;
  named.inherited_name = "if-registry";
  SpinnerGroup spinners(1, named);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
}

TEST_CASE("rocr spin tamer never confines a first-party thread",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  // An InferFlux inference thread can busy-wait inside libhsa-runtime64
  // (hipStreamSynchronize on WSL2 polls). Executing owner code must not make
  // it a confinement target: registered threads are never probed.
  SpinnerGroup::Options ours;
  ours.first_party = true;
  SpinnerGroup spinners(1, ours);
  CountingOwner owner;
  owner.owned = 1000;
  int confined = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    confined += inferflux::TameRuntimeSpinThreads(owner.Options());
  }
  REQUIRE(owner.calls->load() == 0);
  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer never signals a thread that blocks the probe",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  // RT signals queue instead of coalescing; probing a thread that blocks the
  // signal would pile them up against RLIMIT_SIGPENDING.
  SpinnerGroup::Options blocking;
  blocking.block_probe = true;
  SpinnerGroup spinners(1, blocking);
  CountingOwner owner;
  owner.owned = 1000;
  int confined = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    confined += inferflux::TameRuntimeSpinThreads(owner.Options());
  }
  REQUIRE_FALSE(ProbePending(spinners.tid(0)));
  REQUIRE(owner.calls->load() == 0);
  REQUIRE(confined == 0);
}

TEST_CASE("rocr spin tamer backs off a rejected thread, then re-probes it",
          "[rocr_spin_tamer]") {
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  SpinnerGroup spinners(1);

  // First verdict: not proven (owner rejects every sample).
  CountingOwner rejecting;
  for (int attempt = 0; attempt < 4 && rejecting.calls->load() == 0;
       ++attempt) {
    REQUIRE(inferflux::TameRuntimeSpinThreads(rejecting.Options()) == 0);
  }
  REQUIRE(rejecting.calls->load() > 0);

  // The very next scan skips it (back-off): it is not even probed.
  CountingOwner accepting;
  accepting.owned = 1000;
  REQUIRE(inferflux::TameRuntimeSpinThreads(accepting.Options()) == 0);
  REQUIRE(accepting.calls->load() == 0);

  // Back-off is not a permanent blacklist: a later scan re-probes and, now
  // that the evidence says owner code, confines it. On a loaded runner
  // timed-out probes can add strikes (skip 1, 2, 4, 8 scans), so allow
  // enough attempts to outlast them; the usual case needs only 1-2.
  REQUIRE(ScanUntil(1, accepting.Options(), 16) == 1);
}
TEST_CASE("rocr spin tamer honours INFERFLUX_ROCR_TAME_CORE",
          "[rocr_spin_tamer]") {
  if (CountAllowedCpus() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin choice");
  }
  // Pick a CPU that differs from the default (highest allowed) target.
  const int requested = LowestAllowedCpu();
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE",
                                     std::to_string(requested));
  inferflux::ResetRuntimeSpinTamerPinForTesting();
  SpinnerGroup spinners(1);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
  cpu_set_t now;
  CPU_ZERO(&now);
  REQUIRE(sched_getaffinity(spinners.tid(0), sizeof(now), &now) == 0);
  REQUIRE(CPU_COUNT(&now) == 1);
  REQUIRE(CPU_ISSET(requested, &now));
}
TEST_CASE("rocr spin tamer pins every confined thread to the same core",
          "[rocr_spin_tamer]") {
  if (CountAllowedCpus() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin choice");
  }
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  inferflux::ResetRuntimeSpinTamerPinForTesting();
  SpinnerGroup first(1);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
  SpinnerGroup second(1);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
  cpu_set_t a, b;
  CPU_ZERO(&a);
  CPU_ZERO(&b);
  REQUIRE(sched_getaffinity(first.tid(0), sizeof(a), &a) == 0);
  REQUIRE(sched_getaffinity(second.tid(0), sizeof(b), &b) == 0);
  REQUIRE(CPU_COUNT(&a) == 1);
  REQUIRE(CPU_EQUAL(&a, &b));
}
TEST_CASE("rocr spin tamer never pins onto a single-CPU allotment",
          "[rocr_spin_tamer]") {
  if (CountAllowedCpus() < 2) {
    SKIP("needs >= 2 allowed CPUs to narrow the server mask");
  }
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  // The spinner keeps the full mask (as ROCr's widened pollers do); the
  // scanning thread - whose mask the tamer adopts as the server allotment
  // after a reset - is narrowed to one CPU.
  SpinnerGroup spinners(1);
  cpu_set_t original;
  CPU_ZERO(&original);
  REQUIRE(sched_getaffinity(0, sizeof(original), &original) == 0);
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(LowestAllowedCpu(), &one);
  REQUIRE(sched_setaffinity(0, sizeof(one), &one) == 0);
  inferflux::ResetRuntimeSpinTamerPinForTesting();
  CountingOwner owner;
  owner.owned = 1000;
  int confined = 0;
  for (int attempt = 0; attempt < 4 && owner.calls->load() == 0; ++attempt) {
    confined += inferflux::TameRuntimeSpinThreads(owner.Options());
  }
  // Proven but not confinable: the next scan backs off instead of probing
  // it again.
  const int calls_after_proof = owner.calls->load();
  confined += inferflux::TameRuntimeSpinThreads(owner.Options());
  const int calls_after_next_scan = owner.calls->load();
  cpu_set_t spinner_mask;
  CPU_ZERO(&spinner_mask);
  const bool read = sched_getaffinity(spinners.tid(0), sizeof(spinner_mask),
                                      &spinner_mask) == 0;
  // Restore before asserting so a failure cannot leave the runner narrowed.
  sched_setaffinity(0, sizeof(original), &original);
  inferflux::ResetRuntimeSpinTamerPinForTesting();
  REQUIRE(confined == 0);
  REQUIRE(calls_after_proof > 0);
  REQUIRE(calls_after_next_scan == calls_after_proof);
  REQUIRE(read);
  REQUIRE(CPU_COUNT(&spinner_mask) == CPU_COUNT(&original));
}

TEST_CASE("rocr spin tamer restores a confined thread's widened pin",
          "[rocr_spin_tamer]") {
  if (CountAllowedCpus() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin");
  }
  inferflux::test::ScopedEnvVar enable("INFERFLUX_ROCR_TAME", "1");
  inferflux::test::ScopedEnvVar core("INFERFLUX_ROCR_TAME_CORE", nullptr);
  inferflux::ResetRuntimeSpinTamerPinForTesting();
  SpinnerGroup spinners(1);
  REQUIRE(ScanUntil(1, OwnedByTestLoop()) == 1);
  cpu_set_t pinned;
  CPU_ZERO(&pinned);
  REQUIRE(sched_getaffinity(spinners.tid(0), sizeof(pinned), &pinned) == 0);
  REQUIRE(CPU_COUNT(&pinned) == 1);
  // Simulate the runtime widening the thread's affinity again.
  cpu_set_t all;
  CPU_ZERO(&all);
  REQUIRE(sched_getaffinity(0, sizeof(all), &all) == 0);
  REQUIRE(sched_setaffinity(spinners.tid(0), sizeof(all), &all) == 0);
  // The next scan re-pins it (and does not count it again).
  REQUIRE(inferflux::TameRuntimeSpinThreads(OwnedByTestLoop()) == 0);
  cpu_set_t now;
  CPU_ZERO(&now);
  REQUIRE(sched_getaffinity(spinners.tid(0), sizeof(now), &now) == 0);
  REQUIRE(CPU_EQUAL(&now, &pinned));
}
#endif // defined(__linux__)
