#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

enum class MemoryDomain {
  kWeights,
  kWeightsDequantized,
  kWorkspaceDevice,
  kWorkspaceHostPinned,
  kKvCache,
  kKvPrefixCache,
  kSessionMetadata,
  kBatchEphemeral,
};

enum class MemoryLifetime {
  kModel,
  kPool,
  kSession,
  kBatch,
};

struct MemoryUsageSnapshot {
  std::size_t reserved_bytes{0};
  std::size_t in_use_bytes{0};
  std::size_t high_water_bytes{0};
  std::size_t evictable_bytes{0};
};

struct MemoryLedgerItem {
  std::string label;
  MemoryDomain domain{MemoryDomain::kWorkspaceDevice};
  MemoryLifetime lifetime{MemoryLifetime::kModel};
  MemoryUsageSnapshot usage;
};

struct MemoryDomainSummary {
  MemoryDomain domain{MemoryDomain::kWorkspaceDevice};
  MemoryUsageSnapshot usage;
};

const char *ToString(MemoryDomain domain);
const char *ToString(MemoryLifetime lifetime);

class ModelMemoryLedger {
public:
  void Clear();
  void SetModelLabel(std::string label);
  const std::string &ModelLabel() const { return model_label_; }

  void UpsertItem(std::string label, MemoryDomain domain,
                  MemoryLifetime lifetime, std::size_t reserved_bytes,
                  std::size_t in_use_bytes, std::size_t high_water_bytes = 0,
                  std::size_t evictable_bytes = 0);

  std::vector<MemoryLedgerItem> Items() const { return items_; }
  std::vector<MemoryDomainSummary> AggregateByDomain() const;

  std::size_t TotalReservedBytes() const;
  std::size_t TotalInUseBytes() const;
  std::size_t TotalHighWaterBytes() const;
  std::size_t TotalEvictableBytes() const;

  bool FindItem(std::string_view label, MemoryLedgerItem *out) const;
  std::string Describe() const;

private:
  std::string model_label_;
  std::vector<MemoryLedgerItem> items_;
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
