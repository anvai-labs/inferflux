#include "runtime/backends/cuda/native/model_memory_ledger.h"

#include <algorithm>
#include <sstream>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

namespace {

constexpr std::size_t kKiB = 1024ULL;
constexpr std::size_t kMiB = 1024ULL * kKiB;
constexpr std::size_t kGiB = 1024ULL * kMiB;

std::string FormatBytes(std::size_t bytes) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(2);
  if (bytes >= kGiB) {
    out << static_cast<double>(bytes) / static_cast<double>(kGiB) << " GiB";
  } else if (bytes >= kMiB) {
    out << static_cast<double>(bytes) / static_cast<double>(kMiB) << " MiB";
  } else if (bytes >= kKiB) {
    out << static_cast<double>(bytes) / static_cast<double>(kKiB) << " KiB";
  } else {
    out.unsetf(std::ios::fixed);
    out.precision(0);
    out << bytes << " B";
  }
  return out.str();
}

} // namespace

const char *ToString(MemoryDomain domain) {
  switch (domain) {
  case MemoryDomain::kWeights:
    return "weights";
  case MemoryDomain::kWeightsDequantized:
    return "weights_dequantized";
  case MemoryDomain::kWorkspaceDevice:
    return "workspace_device";
  case MemoryDomain::kWorkspaceHostPinned:
    return "workspace_host_pinned";
  case MemoryDomain::kKvCache:
    return "kv_cache";
  case MemoryDomain::kKvPrefixCache:
    return "kv_prefix_cache";
  case MemoryDomain::kSessionMetadata:
    return "session_metadata";
  case MemoryDomain::kBatchEphemeral:
    return "batch_ephemeral";
  }
  return "unknown";
}

const char *ToString(MemoryLifetime lifetime) {
  switch (lifetime) {
  case MemoryLifetime::kModel:
    return "model";
  case MemoryLifetime::kPool:
    return "pool";
  case MemoryLifetime::kSession:
    return "session";
  case MemoryLifetime::kBatch:
    return "batch";
  }
  return "unknown";
}

void ModelMemoryLedger::Clear() {
  model_label_.clear();
  items_.clear();
}

void ModelMemoryLedger::SetModelLabel(std::string label) {
  model_label_ = std::move(label);
}

void ModelMemoryLedger::UpsertItem(std::string label, MemoryDomain domain,
                                   MemoryLifetime lifetime,
                                   std::size_t reserved_bytes,
                                   std::size_t in_use_bytes,
                                   std::size_t high_water_bytes,
                                   std::size_t evictable_bytes) {
  auto it = std::find_if(
      items_.begin(), items_.end(),
      [&](const MemoryLedgerItem &item) { return item.label == label; });
  MemoryLedgerItem next;
  next.label = std::move(label);
  next.domain = domain;
  next.lifetime = lifetime;
  next.usage.reserved_bytes = reserved_bytes;
  next.usage.in_use_bytes = in_use_bytes;
  next.usage.high_water_bytes =
      std::max(high_water_bytes, std::max(reserved_bytes, in_use_bytes));
  next.usage.evictable_bytes = evictable_bytes;
  if (it == items_.end()) {
    items_.push_back(std::move(next));
    return;
  }
  *it = std::move(next);
}

std::vector<MemoryDomainSummary> ModelMemoryLedger::AggregateByDomain() const {
  std::vector<MemoryDomainSummary> out;
  for (const auto &item : items_) {
    auto it = std::find_if(out.begin(), out.end(),
                           [&](const MemoryDomainSummary &summary) {
                             return summary.domain == item.domain;
                           });
    if (it == out.end()) {
      it = out.insert(out.end(), MemoryDomainSummary{item.domain, {}});
    }
    it->usage.reserved_bytes += item.usage.reserved_bytes;
    it->usage.in_use_bytes += item.usage.in_use_bytes;
    it->usage.high_water_bytes += item.usage.high_water_bytes;
    it->usage.evictable_bytes += item.usage.evictable_bytes;
  }
  return out;
}

std::size_t ModelMemoryLedger::TotalReservedBytes() const {
  std::size_t total = 0;
  for (const auto &item : items_) {
    total += item.usage.reserved_bytes;
  }
  return total;
}

std::size_t ModelMemoryLedger::TotalInUseBytes() const {
  std::size_t total = 0;
  for (const auto &item : items_) {
    total += item.usage.in_use_bytes;
  }
  return total;
}

std::size_t ModelMemoryLedger::TotalHighWaterBytes() const {
  std::size_t total = 0;
  for (const auto &item : items_) {
    total += item.usage.high_water_bytes;
  }
  return total;
}

std::size_t ModelMemoryLedger::TotalEvictableBytes() const {
  std::size_t total = 0;
  for (const auto &item : items_) {
    total += item.usage.evictable_bytes;
  }
  return total;
}

bool ModelMemoryLedger::FindItem(std::string_view label,
                                 MemoryLedgerItem *out) const {
  auto it = std::find_if(
      items_.begin(), items_.end(),
      [&](const MemoryLedgerItem &item) { return item.label == label; });
  if (it == items_.end()) {
    return false;
  }
  if (out) {
    *out = *it;
  }
  return true;
}

std::string ModelMemoryLedger::Describe() const {
  std::ostringstream out;
  out << "ModelMemoryLedger";
  if (!model_label_.empty()) {
    out << "[model=" << model_label_ << "]";
  }
  out << " reserved=" << FormatBytes(TotalReservedBytes())
      << " in_use=" << FormatBytes(TotalInUseBytes())
      << " high_water=" << FormatBytes(TotalHighWaterBytes())
      << " evictable=" << FormatBytes(TotalEvictableBytes());

  const auto summaries = AggregateByDomain();
  for (const auto &summary : summaries) {
    out << "\n  " << ToString(summary.domain)
        << ": reserved=" << FormatBytes(summary.usage.reserved_bytes)
        << " in_use=" << FormatBytes(summary.usage.in_use_bytes)
        << " high_water=" << FormatBytes(summary.usage.high_water_bytes)
        << " evictable=" << FormatBytes(summary.usage.evictable_bytes);
  }
  for (const auto &item : items_) {
    out << "\n    - " << item.label << " [" << ToString(item.lifetime)
        << "]: reserved=" << FormatBytes(item.usage.reserved_bytes)
        << " in_use=" << FormatBytes(item.usage.in_use_bytes)
        << " high_water=" << FormatBytes(item.usage.high_water_bytes);
    if (item.usage.evictable_bytes > 0) {
      out << " evictable=" << FormatBytes(item.usage.evictable_bytes);
    }
  }
  return out.str();
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
