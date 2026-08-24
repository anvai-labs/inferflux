#include <catch2/catch_amalgamated.hpp>
#include "server/diagnostics/crash_handler.h"

using namespace inferflux::diagnostics;

TEST_CASE("Crash breadcrumb set and clear", "[diagnostics]") {
  ClearCrashBreadcrumb();

  SetCrashBreadcrumb("ExecuteUnifiedBatch: prefill M=32");
  // No public accessor, but verify no crash on set/clear cycle.
  ClearCrashBreadcrumb();
}

TEST_CASE("ScopedBreadcrumb RAII", "[diagnostics]") {
  {
    ScopedBreadcrumb b("test operation");
    // Breadcrumb is set within scope.
  }
  // Breadcrumb is cleared after scope.
}

TEST_CASE("CUDA error counter", "[diagnostics]") {
  uint64_t baseline = GetCudaErrorCount();
  RecordCudaError();
  RecordCudaError();
  REQUIRE(GetCudaErrorCount() == baseline + 2);
}

TEST_CASE("Active requests tracking", "[diagnostics]") {
  SetActiveRequests(5);
  SetActiveRequests(0);
  // No assertion needed — just verify no crash.
}
