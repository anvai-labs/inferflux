#include <catch2/catch_amalgamated.hpp>

#include "runtime/rocr_spin_tamer.h"
#include "support/scoped_env.h"

#include <vector>

using namespace inferflux;

// Pure decision logic; runs on every platform (keeps the [rocr_spin_tamer]
// tag non-empty on macOS, where an empty tag match fails the ctest entry).
// The pthread_create interposer itself is exercised in its own executable
// (tests/unit/test_rocr_interposer.cpp).

TEST_CASE("rocr tamer core: fewer than two allotted CPUs never pin",
          "[rocr_spin_tamer]") {
  CHECK(ChooseRocrTameCore({}, nullptr) == -1);
  CHECK(ChooseRocrTameCore({3}, nullptr) == -1);
  CHECK(ChooseRocrTameCore({3}, "3") == -1);
}

TEST_CASE("rocr tamer core: highest allotted CPU by default",
          "[rocr_spin_tamer]") {
  bool rejected = true;
  CHECK(ChooseRocrTameCore({0, 5, 2}, nullptr, &rejected) == 5);
  CHECK_FALSE(rejected);
  CHECK(ChooseRocrTameCore({0, 5, 2}, "", &rejected) == 5);
  CHECK_FALSE(rejected); // An empty value means "not set".
}

TEST_CASE("rocr tamer core: a valid allotted override wins",
          "[rocr_spin_tamer]") {
  bool rejected = true;
  CHECK(ChooseRocrTameCore({0, 5, 2}, "2", &rejected) == 2);
  CHECK_FALSE(rejected);
  CHECK(ChooseRocrTameCore({1, 2}, "1", &rejected) == 1);
  CHECK_FALSE(rejected);
}

TEST_CASE("rocr tamer core: invalid overrides fall back and are reported",
          "[rocr_spin_tamer]") {
  for (const char *bad : {"4", "abc", "2x", "-1", "99999999999"}) {
    bool rejected = false;
    CHECK(ChooseRocrTameCore({0, 5, 2}, bad, &rejected) == 5);
    CHECK(rejected);
  }
}

TEST_CASE("rocr tamer: recognizes the runtime by object basename",
          "[rocr_spin_tamer]") {
  CHECK(IsRocrRuntimeObjectPath("/opt/rocm/lib/libhsa-runtime64.so.1"));
  CHECK(IsRocrRuntimeObjectPath("libhsa-runtime64.so"));
  CHECK(IsRocrRuntimeObjectPath("/b/tests/libhsa-runtime64-inferflux-test.so"));
  CHECK_FALSE(IsRocrRuntimeObjectPath("/opt/libhsa-runtime64/libfoo.so"));
  CHECK_FALSE(IsRocrRuntimeObjectPath("/opt/rocm/lib/libamdhip64.so.7"));
  CHECK_FALSE(IsRocrRuntimeObjectPath(""));
  CHECK_FALSE(IsRocrRuntimeObjectPath(nullptr));
}

#if defined(__linux__)
namespace {
void *LocalStartRoutine(void *arg) { return arg; }

struct ResetOnExit {
  ~ResetOnExit() { ResetRocrTamerForTesting(); }
};
} // namespace

TEST_CASE("rocr tamer: threads started from this binary are never tamed",
          "[rocr_spin_tamer]") {
  ResetOnExit reset_on_exit; // Destroyed after the env is restored.
  for (const char *tame : {"0", "1"}) {
    inferflux::test::ScopedEnvVar env("INFERFLUX_ROCR_TAME", tame);
    ResetRocrTamerForTesting();
    CHECK_FALSE(IsRocrRuntimeAddress(
        reinterpret_cast<const void *>(&LocalStartRoutine)));
    CHECK(RocrTameCoreForNewThread(&LocalStartRoutine) == -1);
  }
}

TEST_CASE("rocr tamer: status report is safe without an interposer",
          "[rocr_spin_tamer]") {
  ResetOnExit reset_on_exit;
  inferflux::test::ScopedEnvVar env("INFERFLUX_ROCR_TAME", "1");
  ResetRocrTamerForTesting();
  // inferflux_tests does not link the interposer: this must warn, not fail.
  CHECK_NOTHROW(ReportRocrTamerStatus());
}
#endif
