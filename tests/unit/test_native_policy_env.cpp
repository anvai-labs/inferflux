// CPU-runnable env-var parsing coverage for the native execution policy.
// Parsing lives in a CUDA-free translation unit, so these cases run in the
// CPU-only CI lane alongside the other base-list tests.
#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "support/scoped_env.h"

namespace inferflux {
using test::ScopedEnvVar;

TEST_CASE("NativeExecutionPolicy parses the shared MMQ layout kill switch",
          "[native_policy_env]") {
  {
    ScopedEnvVar flag("INFERFLUX_DISABLE_SHARED_MMQ_LAYOUT", nullptr);
    const auto policy =
        NativeExecutionPolicy::FromEnv();
    REQUIRE_FALSE(policy.disable_shared_mmq_layout);
  }
  {
    ScopedEnvVar flag("INFERFLUX_DISABLE_SHARED_MMQ_LAYOUT", "1");
    const auto policy =
        NativeExecutionPolicy::FromEnv();
    REQUIRE(policy.disable_shared_mmq_layout);
  }
  {
    ScopedEnvVar flag("INFERFLUX_DISABLE_SHARED_MMQ_LAYOUT", "0");
    const auto policy =
        NativeExecutionPolicy::FromEnv();
    REQUIRE_FALSE(policy.disable_shared_mmq_layout);
  }
}

} // namespace inferflux
