#pragma once

// Dispatch catalog: the single source of truth for the native CUDA
// projection-dispatch operator universe.
//
// The selector (native_dispatch_registry rules), the executors
// (native_linear_executor.h), the metrics layer (metrics.cpp allowlists and
// render grids), the reachability tests, and the load-time probe all derive
// from the tables below. Adding an operator is exactly: one enum member,
// one table row, one registry rule — everything else follows, or the build
// fails (missing table row -> static_assert; unwired switch -> -Werror=
// switch in the label helpers and the executors).
//
// This header is deliberately CUDA-free so CPU-only CI builds can compile
// the decision layer and run the dispatch reachability tests (the CUDA CI
// job is compile-only; only the CPU job executes inferflux_tests).
//
// Incident history: the FFN executor's operator allowlist once omitted
// kQ81GroupMmq3, so the selector's choice was silently discarded and a
// catch-all ran a ~5x slower kernel for months (fixed in 6f94cec; detected
// by Prometheus operator counters showing packed_group at 100%). The
// catalog exists to make that bug class impossible to write, merge, or
// deploy silently.

#include "runtime/backends/cuda/native/gguf_util.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

namespace inferflux {

// ============================================================================
// Operator enums (namespace scope so CPU-only TUs can reference them
// without pulling in CUDA headers via fused_quant_gemm.h).
// ============================================================================

enum class FfnProjOperator {
  kFallback = 0,
  kQ81Group,
  kQ81GroupHotQ4K,
  kQ81GroupRowPairW4,
  kQ81GroupRowQuadM4,
  kQ81GroupMmq3,
  kPackedGroup,
};

enum class DownProjOperator {
  kFallback = 0,
  kQ81Gemv,
  kQ81GemvHotFixed,
  kQ81GemvRowPairHotFixed,
  kQ81GemvRowPair,
  kQ81GemvRowQuad,
  kPackedGemv,
  kMmq,
};

constexpr int kFfnProjOperatorCount = 7;
constexpr int kDownProjOperatorCount = 8;
static_assert(kFfnProjOperatorCount <= 16 && kDownProjOperatorCount <= 16,
              "operator health bitmasks are uint16");

// What actually executes for an operator. Divergence between the selected
// tier and the executed tier is the signal the runtime telemetry watches.
enum class DispatchTier : uint8_t {
  kFallback = 0, // dense fallback (dequant + cuBLAS)
  kQ81,          // Q8_1 activation kernels (MMVQ / grouped / MMQ3)
  kPacked,       // packed int8 dp4a kernels
  kMmq,          // tiled quantized GEMM (MMQ)
};

// ============================================================================
// Capability tables
// ============================================================================

struct FfnOpInfo {
  FfnProjOperator op;
  const char *label;       // canonical selection/trace/divergence label
  DispatchTier tier;       // what executes when this op wins selection
  bool distinct_execution; // false => label-only variant of a shared kernel
  bool in_rules_table;     // emitted by >= 1 registry rule
};

struct DownOpInfo {
  DownProjOperator op;
  const char *label;
  DispatchTier tier;
  bool distinct_execution;
  bool in_rules_table;
};

constexpr std::array<FfnOpInfo, kFfnProjOperatorCount> kFfnOpTable = {{
    {FfnProjOperator::kFallback, "fallback", DispatchTier::kFallback, true,
     false},
    {FfnProjOperator::kQ81Group, "q8_1_group", DispatchTier::kQ81, true, true},
    {FfnProjOperator::kQ81GroupHotQ4K, "q8_1_group_hot_q4k", DispatchTier::kQ81,
     true, true},
    {FfnProjOperator::kQ81GroupRowPairW4, "q8_1_group_row_pair_w4",
     DispatchTier::kQ81, true, true},
    {FfnProjOperator::kQ81GroupRowQuadM4, "q8_1_group_row_quad_m4",
     DispatchTier::kQ81, true, true},
    {FfnProjOperator::kQ81GroupMmq3, "q8_1_group_mmq3", DispatchTier::kQ81,
     true, true},
    {FfnProjOperator::kPackedGroup, "packed_group", DispatchTier::kPacked, true,
     true},
}};

constexpr std::array<DownOpInfo, kDownProjOperatorCount> kDownOpTable = {{
    {DownProjOperator::kFallback, "fallback", DispatchTier::kFallback, true,
     false},
    {DownProjOperator::kQ81Gemv, "q8_1_gemv", DispatchTier::kQ81, true, true},
    {DownProjOperator::kQ81GemvHotFixed, "q8_1_gemv_hot_fixed",
     DispatchTier::kQ81, false, true},
    {DownProjOperator::kQ81GemvRowPairHotFixed, "q8_1_gemv_row_pair_hot_fixed",
     DispatchTier::kQ81, false, true},
    {DownProjOperator::kQ81GemvRowPair, "q8_1_gemv_row_pair",
     DispatchTier::kQ81, false, true},
    {DownProjOperator::kQ81GemvRowQuad, "q8_1_gemv_row_quad",
     DispatchTier::kQ81, false, true},
    {DownProjOperator::kPackedGemv, "packed_gemv", DispatchTier::kPacked, true,
     true},
    {DownProjOperator::kMmq, "mmq", DispatchTier::kMmq, true, true},
}};

// ============================================================================
// Table integrity gates (compile-time)
// ============================================================================

static_assert(kFfnOpTable.size() == kFfnProjOperatorCount,
              "FfnProjOperator gained/lost a member without a catalog row");
static_assert(kDownOpTable.size() == kDownProjOperatorCount,
              "DownProjOperator gained/lost a member without a catalog row");

namespace dispatch_catalog_detail {
constexpr bool
DenseOrdered(const std::array<FfnOpInfo, kFfnProjOperatorCount> &t) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (static_cast<int>(t[i].op) != static_cast<int>(i)) {
      return false;
    }
  }
  return true;
}
constexpr bool
DenseOrdered(const std::array<DownOpInfo, kDownProjOperatorCount> &t) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (static_cast<int>(t[i].op) != static_cast<int>(i)) {
      return false;
    }
  }
  return true;
}
constexpr bool
UniqueLabels(const std::array<FfnOpInfo, kFfnProjOperatorCount> &t) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    for (std::size_t j = i + 1; j < t.size(); ++j) {
      if (std::string_view(t[i].label) == std::string_view(t[j].label)) {
        return false;
      }
    }
  }
  return true;
}
constexpr bool
UniqueLabels(const std::array<DownOpInfo, kDownProjOperatorCount> &t) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    for (std::size_t j = i + 1; j < t.size(); ++j) {
      if (std::string_view(t[i].label) == std::string_view(t[j].label)) {
        return false;
      }
    }
  }
  return true;
}
} // namespace dispatch_catalog_detail

static_assert(dispatch_catalog_detail::DenseOrdered(kFfnOpTable),
              "kFfnOpTable must be indexed by enum value");
static_assert(dispatch_catalog_detail::DenseOrdered(kDownOpTable),
              "kDownOpTable must be indexed by enum value");
static_assert(dispatch_catalog_detail::UniqueLabels(kFfnOpTable),
              "duplicate FFN catalog label");
static_assert(dispatch_catalog_detail::UniqueLabels(kDownOpTable),
              "duplicate down-proj catalog label");

// ============================================================================
// Label helpers. Switches deliberately carry NO default so an unhandled
// enum member is a compile error under -Werror=switch.
// ============================================================================

constexpr const char *FfnSelectionLabel(FfnProjOperator op) {
  switch (op) {
  case FfnProjOperator::kFallback:
    return kFfnOpTable[0].label;
  case FfnProjOperator::kQ81Group:
    return kFfnOpTable[1].label;
  case FfnProjOperator::kQ81GroupHotQ4K:
    return kFfnOpTable[2].label;
  case FfnProjOperator::kQ81GroupRowPairW4:
    return kFfnOpTable[3].label;
  case FfnProjOperator::kQ81GroupRowQuadM4:
    return kFfnOpTable[4].label;
  case FfnProjOperator::kQ81GroupMmq3:
    return kFfnOpTable[5].label;
  case FfnProjOperator::kPackedGroup:
    return kFfnOpTable[6].label;
  }
}

constexpr const char *DownSelectionLabel(DownProjOperator op) {
  switch (op) {
  case DownProjOperator::kFallback:
    return kDownOpTable[0].label;
  case DownProjOperator::kQ81Gemv:
    return kDownOpTable[1].label;
  case DownProjOperator::kQ81GemvHotFixed:
    return kDownOpTable[2].label;
  case DownProjOperator::kQ81GemvRowPairHotFixed:
    return kDownOpTable[3].label;
  case DownProjOperator::kQ81GemvRowPair:
    return kDownOpTable[4].label;
  case DownProjOperator::kQ81GemvRowQuad:
    return kDownOpTable[5].label;
  case DownProjOperator::kPackedGemv:
    return kDownOpTable[6].label;
  case DownProjOperator::kMmq:
    return kDownOpTable[7].label;
  }
}

constexpr DispatchTier TierOf(FfnProjOperator op) {
  return kFfnOpTable[static_cast<std::size_t>(op)].tier;
}

constexpr DispatchTier TierOf(DownProjOperator op) {
  return kDownOpTable[static_cast<std::size_t>(op)].tier;
}

constexpr const char *DispatchTierLabel(DispatchTier tier) {
  switch (tier) {
  case DispatchTier::kFallback:
    return "fallback";
  case DispatchTier::kQ81:
    return "q8_1";
  case DispatchTier::kPacked:
    return "packed";
  case DispatchTier::kMmq:
    return "mmq";
  }
}

// ============================================================================
// Metric label universes. Record* and the Prometheus render loops both
// derive from these arrays — the hand-written allowlists in metrics.cpp
// were the reason kQ81GroupMmq3 executions mapped to "unknown" and were
// never exported.
// ============================================================================

// Labels produced by FfnProjOperatorMetricName (remap included). Keep as a
// flat closed set so the exposition grid is deterministic.
constexpr std::array<const char *, 11> kFfnMetricLabels = {{
    "fallback",
    "q8_1_group_generic",
    "q8_1_group_mmvq",
    "q8_1_group_hot_q4k",
    "q8_1_group_row_pair_w4",
    "q8_1_group_row_quad_m4",
    "q8_1_group_mmq3",
    "q8_1_group_row_pair",
    "q8_1_group_row_quad",
    "q8_1_group_v2",
    "packed_group",
}};

constexpr std::array<const char *, 12> kDownMetricLabels = {{
    "fallback",
    "q8_1_gemv",
    "q8_1_mmvq",
    "q8_1_mmq",
    "q8_1_gemv_hot_fixed",
    "q8_1_gemv_row_pair_hot_fixed",
    "q8_1_gemv_row_pair",
    "q8_1_gemv_row_quad",
    "q8_1_gemv_v2",
    "q8_1_gemv_row_pair_v2",
    "packed_gemv",
    "mmq",
}};

constexpr bool FfnMetricLabelKnown(std::string_view label) {
  for (const char *known : kFfnMetricLabels) {
    if (label == known) {
      return true;
    }
  }
  return false;
}

// Metric label for an FFN selection. The kQ81Group M>8 label names the V1
// grouped kernel that actually runs (grouped MMQ above M=8 is not
// implemented) — the old "q8_1_group_mmq" label overstated execution.
inline const char *FfnMetricLabel(FfnProjOperator op, int quant_type, int m) {
  if (op == FfnProjOperator::kQ81Group) {
    const auto qtype =
        static_cast<runtime::cuda::native::GGUF::TensorType>(quant_type);
    if (qtype == runtime::cuda::native::GGUF::TensorType::Q4_K ||
        qtype == runtime::cuda::native::GGUF::TensorType::Q6_K) {
      return (m <= 8) ? "q8_1_group_mmvq" : "q8_1_group_generic";
    }
  }
  return FfnSelectionLabel(op);
}

// Metric label for a down-proj selection. The single-projection path runs
// MMVQ for M<=8 and real MMQ for M 9-64, so the split label is honest here.
inline const char *DownMetricLabel(DownProjOperator op, int quant_type, int m) {
  (void)quant_type;
  if (op == DownProjOperator::kQ81Gemv) {
    return (m <= 8) ? "q8_1_mmvq" : "q8_1_mmq";
  }
  return DownSelectionLabel(op);
}

constexpr bool DownMetricLabelKnown(std::string_view label) {
  for (const char *known : kDownMetricLabels) {
    if (label == known) {
      return true;
    }
  }
  return false;
}

// ----------------------------------------------------------------------------
// Divergence reporting. A tier mismatch (or a non-fallback selection
// landing on the dense fallback) is counted once per tuple and logged once
// per tuple with a fix hint; repeated occurrences only increment the
// counter.
// ----------------------------------------------------------------------------
inline void ReportDispatchDivergence(const char *layer, const char *selected,
                                     const char *actual, const char *reason) {
  GlobalMetrics().RecordInferfluxCudaDispatchDivergence(layer, selected, actual,
                                                        reason);
  static std::mutex once_mutex;
  static std::set<std::string> logged;
  const std::string key =
      std::string(layer) + "|" + selected + "|" + actual + "|" + reason;
  {
    std::lock_guard<std::mutex> lock(once_mutex);
    if (logged.insert(key).second) {
      log::Warn("dispatch",
                std::string("operator divergence: layer=") + layer +
                    " selected=" + selected + " actual=" + actual +
                    " reason=" + reason +
                    " — the selected operator did not execute; check the "
                    "executor switch against kFfnOpTable/kDownOpTable in "
                    "dispatch_catalog.h and the tier guards in "
                    "native_linear_executor.h");
    }
  }
}

} // namespace inferflux
