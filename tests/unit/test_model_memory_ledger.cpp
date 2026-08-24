#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/model_memory_ledger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

TEST_CASE("ModelMemoryLedger aggregates startup allocations by domain",
          "[model_memory_ledger]") {
  ModelMemoryLedger ledger;
  ledger.SetModelLabel("qwen2.5-3b");
  ledger.UpsertItem("weights.loader", MemoryDomain::kWeights,
                    MemoryLifetime::kModel, 1024, 1024);
  ledger.UpsertItem("workspace.forward.primary",
                    MemoryDomain::kWorkspaceDevice, MemoryLifetime::kModel,
                    512, 512);
  ledger.UpsertItem("workspace.forward.primary.host",
                    MemoryDomain::kWorkspaceHostPinned,
                    MemoryLifetime::kModel, 128, 128);
  ledger.UpsertItem("kv_cache.primary", MemoryDomain::kKvCache,
                    MemoryLifetime::kPool, 2048, 1536, 2048);

  REQUIRE(ledger.TotalReservedBytes() == 3712);
  REQUIRE(ledger.TotalInUseBytes() == 3200);
  REQUIRE(ledger.TotalHighWaterBytes() == 3712);

  const auto summaries = ledger.AggregateByDomain();
  REQUIRE(summaries.size() == 4);

  MemoryLedgerItem kv_item;
  REQUIRE(ledger.FindItem("kv_cache.primary", &kv_item));
  REQUIRE(kv_item.usage.reserved_bytes == 2048);
  REQUIRE(kv_item.usage.in_use_bytes == 1536);
  REQUIRE(kv_item.usage.high_water_bytes == 2048);

  const std::string report = ledger.Describe();
  REQUIRE(report.find("ModelMemoryLedger[model=qwen2.5-3b]") !=
          std::string::npos);
  REQUIRE(report.find("weights: reserved=") != std::string::npos);
  REQUIRE(report.find("workspace_device: reserved=") != std::string::npos);
  REQUIRE(report.find("kv_cache: reserved=") != std::string::npos);
  REQUIRE(report.find("workspace.forward.primary") != std::string::npos);
}

TEST_CASE("ModelMemoryLedger upsert replaces item state",
          "[model_memory_ledger]") {
  ModelMemoryLedger ledger;
  ledger.UpsertItem("workspace.logits.primary",
                    MemoryDomain::kWorkspaceDevice, MemoryLifetime::kModel, 64,
                    64);
  ledger.UpsertItem("workspace.logits.primary",
                    MemoryDomain::kWorkspaceDevice, MemoryLifetime::kModel, 96,
                    80, 112);

  REQUIRE(ledger.TotalReservedBytes() == 96);
  REQUIRE(ledger.TotalInUseBytes() == 80);
  REQUIRE(ledger.TotalHighWaterBytes() == 112);

  MemoryLedgerItem item;
  REQUIRE(ledger.FindItem("workspace.logits.primary", &item));
  REQUIRE(item.usage.reserved_bytes == 96);
  REQUIRE(item.usage.in_use_bytes == 80);
  REQUIRE(item.usage.high_water_bytes == 112);
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
