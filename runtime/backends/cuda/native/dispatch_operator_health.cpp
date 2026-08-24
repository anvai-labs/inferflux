#include "runtime/backends/cuda/native/dispatch_operator_health.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace inferflux {

InferfluxCudaOperatorHealth &InferfluxCudaOperatorHealth::Instance() {
  static InferfluxCudaOperatorHealth instance;
  return instance;
}

void InferfluxCudaOperatorHealth::MarkUnhealthy(FfnProjOperator op,
                                                const char *reason) {
  if (op == FfnProjOperator::kFallback) {
    return; // not a kernel; never down-rank the dense fallback
  }
  const uint16_t bit = uint16_t{1} << static_cast<unsigned>(op);
  if (ffn_mask_.fetch_or(bit, std::memory_order_relaxed) & bit) {
    return; // already marked
  }
  std::lock_guard<std::mutex> lock(describe_mutex_);
  marked_.emplace_back(std::string("ffn:") + FfnSelectionLabel(op), reason);
}

void InferfluxCudaOperatorHealth::MarkUnhealthy(DownProjOperator op,
                                                const char *reason) {
  if (op == DownProjOperator::kFallback) {
    return;
  }
  const uint16_t bit = uint16_t{1} << static_cast<unsigned>(op);
  if (down_mask_.fetch_or(bit, std::memory_order_relaxed) & bit) {
    return;
  }
  std::lock_guard<std::mutex> lock(describe_mutex_);
  marked_.emplace_back(std::string("down:") + DownSelectionLabel(op), reason);
}

std::string InferfluxCudaOperatorHealth::Describe() const {
  std::lock_guard<std::mutex> lock(describe_mutex_);
  if (marked_.empty()) {
    return {};
  }
  std::ostringstream out;
  for (std::size_t i = 0; i < marked_.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << marked_[i].first << "(" << marked_[i].second << ")";
  }
  return out.str();
}

void InferfluxCudaOperatorHealth::ApplyForceList(const char *csv) {
  if (!csv || !*csv) {
    return;
  }
  std::string text(csv);
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t comma = text.find(',', pos);
    const std::string entry = text.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos);
    pos = comma == std::string::npos ? text.size() : comma + 1;
    if (entry.empty()) {
      continue;
    }
    const std::string::size_type colon = entry.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string layer = entry.substr(0, colon);
    const std::string label = entry.substr(colon + 1);
    if (layer == "ffn") {
      for (const auto &row : kFfnOpTable) {
        if (label == row.label && row.in_rules_table) {
          MarkUnhealthy(row.op, "forced");
        }
      }
    } else if (layer == "down") {
      for (const auto &row : kDownOpTable) {
        if (label == row.label && row.in_rules_table) {
          MarkUnhealthy(row.op, "forced");
        }
      }
    }
  }
}

} // namespace inferflux
