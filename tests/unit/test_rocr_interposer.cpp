// Tests the pthread_create interposer (server/rocr_pthread_interposer.cpp)
// in its own executable: threads started from a stand-in for
// libhsa-runtime64 (tests/support/fake_hsa_runtime.cpp, dlopen'd
// RTLD_LOCAL like ggml-hip loads the real runtime) must be tamed at
// creation; every other thread must be left alone.

#include <catch2/catch_amalgamated.hpp>

#include "runtime/rocr_spin_tamer.h"
#include "server/metrics/metrics.h"
#include "support/scoped_env.h"

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
struct FakeHsaReport {
  int cpu_count;
  int only_cpu;
  int nice;
  char name[16];
  int tls_dtor_ran;
  int done;
};
}

namespace {

using SpawnFn = int (*)(pthread_t *, FakeHsaReport *, int, int, int);
using SpawnWithAttrFn = int (*)(pthread_t *, const pthread_attr_t *,
                                FakeHsaReport *);
using SymbolFn = void *(*)();

enum Mode { kReport = 0, kPthreadExit = 1, kCancellable = 2, kThreadLocal = 3 };

struct FakeRuntime {
  SpawnFn spawn = nullptr;
  SpawnWithAttrFn spawn_with_attr = nullptr;
  SymbolFn symbol = nullptr;
};

const FakeRuntime &Fake() {
  static const FakeRuntime fake = [] {
    FakeRuntime f;
    // RTLD_LOCAL: the worst case (ggml-hip loads the runtime this way); the
    // executable's pthread_create must still win.
    void *handle =
        dlopen(INFERFLUX_FAKE_HSA_RUNTIME_PATH, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
      f.spawn = reinterpret_cast<SpawnFn>(dlsym(handle, "fake_hsa_spawn"));
      f.spawn_with_attr = reinterpret_cast<SpawnWithAttrFn>(
          dlsym(handle, "fake_hsa_spawn_with_attr"));
      f.symbol = reinterpret_cast<SymbolFn>(dlsym(handle, "fake_hsa_symbol"));
    }
    return f;
  }();
  return fake;
}

std::vector<int> AllowedCpus() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  std::vector<int> cpus;
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &mask)) {
        cpus.push_back(cpu);
      }
    }
  }
  return cpus;
}

int OwnNice() {
  return getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
}

bool IsTamedName(const char *name) {
  return std::strncmp(name, "rocr-hsa-", 9) == 0;
}

// Sets the tamer env for a scope. Members are destroyed in reverse order:
// the env is restored first, then ResetOnExit makes the tamer re-read it.
struct ResetOnExit {
  ~ResetOnExit() { inferflux::ResetRocrTamerForTesting(); }
};
struct TamerEnv {
  ResetOnExit reset_on_exit;
  inferflux::test::ScopedEnvVar tame;
  inferflux::test::ScopedEnvVar core;
  TamerEnv(const char *tame_value, const char *core_value)
      : tame("INFERFLUX_ROCR_TAME", tame_value),
        core("INFERFLUX_ROCR_TAME_CORE", core_value) {
    inferflux::ResetRocrTamerForTesting();
  }
};

FakeHsaReport SpawnAndJoin(int mode = kReport) {
  FakeHsaReport report{};
  pthread_t thread{};
  REQUIRE(Fake().spawn(&thread, &report, mode, /*widen_attr=*/1,
                       /*detached=*/0) == 0);
  void *result = nullptr;
  REQUIRE(pthread_join(thread, &result) == 0);
  REQUIRE(result == &report);
  return report;
}

} // namespace

TEST_CASE("interposer: libraries bind pthread_create to the executable",
          "[rocr_interposer]") {
  REQUIRE(Fake().spawn != nullptr);
  REQUIRE(dlsym(RTLD_DEFAULT, "pthread_create") ==
          reinterpret_cast<void *>(&pthread_create));
  REQUIRE(inferflux::IsRocrRuntimeAddress(Fake().symbol()));
}

TEST_CASE("interposer: tamer off leaves runtime threads untouched",
          "[rocr_interposer]") {
  for (const char *off : {static_cast<const char *>(nullptr), "0"}) {
    TamerEnv env(off, nullptr);
    const unsigned long long before = inferflux::RocrTamedThreadCount();
    const FakeHsaReport report = SpawnAndJoin();
    CHECK_FALSE(IsTamedName(report.name));
    CHECK(report.nice == OwnNice());
    if (AllowedCpus().size() > 1) {
      CHECK(report.cpu_count > 1);
    }
    CHECK(inferflux::RocrTamedThreadCount() == before);
  }
}

TEST_CASE("interposer: tames runtime threads at creation",
          "[rocr_interposer]") {
  const std::vector<int> cpus = AllowedCpus();
  if (cpus.size() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin");
  }
  TamerEnv env("1", nullptr);
  const unsigned long long before = inferflux::RocrTamedThreadCount();
  const FakeHsaReport report = SpawnAndJoin();
  CHECK(report.cpu_count == 1);
  CHECK(report.only_cpu == cpus.back());
  CHECK(report.nice == 19);
  CHECK(IsTamedName(report.name));
  CHECK(inferflux::RocrTamedThreadCount() == before + 1);
  CHECK(inferflux::GlobalMetrics().RenderPrometheus().find(
            "inferflux_runtime_spin_threads_confined_total") !=
        std::string::npos);
}

TEST_CASE("interposer: INFERFLUX_ROCR_TAME_CORE override and fallback",
          "[rocr_interposer]") {
  const std::vector<int> cpus = AllowedCpus();
  if (cpus.size() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin");
  }
  {
    TamerEnv env("1", std::to_string(cpus.front()).c_str());
    CHECK(SpawnAndJoin().only_cpu == cpus.front());
  }
  for (const char *bad : {"99999", "abc", "-1"}) {
    TamerEnv env("1", bad);
    const FakeHsaReport report = SpawnAndJoin();
    CHECK(report.cpu_count == 1);
    CHECK(report.only_cpu == cpus.back());
  }
}

TEST_CASE("interposer: threads the executable starts are never tamed",
          "[rocr_interposer]") {
  TamerEnv env("1", nullptr);
  const unsigned long long before = inferflux::RocrTamedThreadCount();
  int cpu_count = 0;
  int nice = -100;
  std::thread([&] {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);
    cpu_count = CPU_COUNT(&mask);
    nice = OwnNice();
  }).join();
  CHECK(nice == OwnNice());
  if (AllowedCpus().size() > 1) {
    CHECK(cpu_count > 1);
  }
  CHECK(inferflux::RocrTamedThreadCount() == before);
}

TEST_CASE("interposer: a single-CPU allotment is never pinned",
          "[rocr_interposer]") {
  const std::vector<int> cpus = AllowedCpus();
  if (cpus.size() < 2) {
    SKIP("needs >= 2 allowed CPUs to narrow the allotment");
  }
  cpu_set_t original;
  CPU_ZERO(&original);
  REQUIRE(sched_getaffinity(0, sizeof(original), &original) == 0);
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(cpus.front(), &one);
  REQUIRE(sched_setaffinity(0, sizeof(one), &one) == 0);
  FakeHsaReport report{};
  unsigned long long before = 0;
  {
    TamerEnv env("1", nullptr); // Captures the narrowed allotment.
    before = inferflux::RocrTamedThreadCount();
    pthread_t thread{};
    const int rc = Fake().spawn(&thread, &report, kReport, /*widen_attr=*/0,
                                /*detached=*/0);
    if (rc == 0) {
      pthread_join(thread, nullptr);
    }
    // Restore before the env scope ends so the reset recaptures it.
    sched_setaffinity(0, sizeof(original), &original);
    REQUIRE(rc == 0);
  }
  CHECK_FALSE(IsTamedName(report.name));
  CHECK(report.nice == OwnNice());
  CHECK(inferflux::RocrTamedThreadCount() == before);
}

TEST_CASE("interposer: pthread_exit, cancellation and thread_local work "
          "through the trampoline",
          "[rocr_interposer]") {
  TamerEnv env("1", nullptr);
  SECTION("pthread_exit") {
    const FakeHsaReport report = SpawnAndJoin(kPthreadExit);
    CHECK(report.cpu_count >= 1);
  }
  SECTION("cancellation") {
    FakeHsaReport report{};
    pthread_t thread{};
    REQUIRE(Fake().spawn(&thread, &report, kCancellable, 1, 0) == 0);
    REQUIRE(pthread_cancel(thread) == 0);
    void *result = nullptr;
    REQUIRE(pthread_join(thread, &result) == 0);
    CHECK(result == PTHREAD_CANCELED);
  }
  SECTION("thread_local destructor") {
    const FakeHsaReport report = SpawnAndJoin(kThreadLocal);
    CHECK(report.tls_dtor_ran == 1);
  }
  SECTION("detached") {
    FakeHsaReport report{};
    pthread_t thread{};
    REQUIRE(Fake().spawn(&thread, &report, kReport, 1, /*detached=*/1) == 0);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (__atomic_load_n(&report.done, __ATOMIC_ACQUIRE) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(__atomic_load_n(&report.done, __ATOMIC_ACQUIRE) == 1);
  }
}

TEST_CASE("interposer: creation failures pass through unchanged",
          "[rocr_interposer]") {
  TamerEnv env("1", nullptr);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  // A stack far larger than any address space can map: creation must fail.
  REQUIRE(pthread_attr_setstacksize(&attr, 1ULL << 46) == 0);
  const unsigned long long before = inferflux::RocrTamedThreadCount();
  FakeHsaReport report{};
  pthread_t thread{};
  const int rc = Fake().spawn_with_attr(&thread, &attr, &report);
  // The C library's own answer for the same attribute.
  using CreateFn =
      int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
  const auto real =
      reinterpret_cast<CreateFn>(dlsym(RTLD_NEXT, "pthread_create"));
  REQUIRE(real != nullptr);
  pthread_t probe{};
  const int real_rc =
      real(&probe, &attr, [](void *) -> void * { return nullptr; }, nullptr);
  if (real_rc == 0) {
    pthread_join(probe, nullptr);
  }
  if (rc == 0) {
    pthread_join(thread, nullptr);
  }
  pthread_attr_destroy(&attr);
  CHECK(rc == real_rc);
  CHECK(rc != 0);
  CHECK(inferflux::RocrTamedThreadCount() == before);
}

TEST_CASE("interposer: concurrent creations are each tamed exactly once",
          "[rocr_interposer]") {
  if (AllowedCpus().size() < 2) {
    SKIP("needs >= 2 allowed CPUs to observe a pin");
  }
  TamerEnv env("1", nullptr);
  const unsigned long long before = inferflux::RocrTamedThreadCount();
  constexpr int kCreators = 8;
  constexpr int kPerCreator = 50;
  std::atomic<int> failures{0};
  std::vector<std::thread> creators;
  for (int c = 0; c < kCreators; ++c) {
    creators.emplace_back([&] {
      for (int i = 0; i < kPerCreator; ++i) {
        FakeHsaReport report{};
        pthread_t thread{};
        if (Fake().spawn(&thread, &report, kReport, 1, 0) != 0 ||
            pthread_join(thread, nullptr) != 0 || report.cpu_count != 1) {
          failures.fetch_add(1);
        }
        std::thread([] {}).join(); // Untouched creations interleaved.
      }
    });
  }
  for (auto &creator : creators) {
    creator.join();
  }
  CHECK(failures.load() == 0);
  CHECK(inferflux::RocrTamedThreadCount() == before + kCreators * kPerCreator);
}
