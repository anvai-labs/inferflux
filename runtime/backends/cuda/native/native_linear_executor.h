#pragma once

#include "runtime/backends/cuda/native/dispatch_catalog.h"

#include "server/metrics/metrics.h"

#include <string>
#include <string_view>
#include <utility>

namespace inferflux {

struct NativeFfnExecutionSummary {
  FfnProjOperator actual_op{FfnProjOperator::kFallback};
  bool used_q81{false};
  bool used_packed{false};
};

struct NativeGroupedProjectionSummary {
  bool used_q81{false};
  bool used_packed{false};
};

template <typename EnsureNormFn, typename TryGemvFn, typename DenseFallbackFn>
bool ExecuteNativeNormalizedProjectionStage(
    bool *norm_computed, EnsureNormFn &&ensure_norm, TryGemvFn &&try_gemv,
    DenseFallbackFn &&run_dense_fallback) {
  if (!*norm_computed) {
    if (!std::forward<EnsureNormFn>(ensure_norm)()) {
      return false;
    }
    *norm_computed = true;
  }

  if (std::forward<TryGemvFn>(try_gemv)()) {
    return true;
  }

  return std::forward<DenseFallbackFn>(run_dense_fallback)();
}

template <typename TryQ81Fn, typename TryPackedFn, typename FallbackFn>
bool ExecuteNativeGroupedProjectionStage(
    TryQ81Fn &&try_q81_group, TryPackedFn &&try_packed_group,
    FallbackFn &&run_fallback,
    NativeGroupedProjectionSummary *summary = nullptr) {
  NativeGroupedProjectionSummary local_summary;
  local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81_group)();
  if (!local_summary.used_q81) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed_group)();
  }
  if (summary) {
    *summary = local_summary;
  }
  if (!local_summary.used_q81 && !local_summary.used_packed) {
    return std::forward<FallbackFn>(run_fallback)();
  }
  return true;
}

template <typename TryQ81Fn, typename TryPackedFn, typename FallbackFn>
bool ExecuteInferfluxCudaFfnProjectionStage(
    FfnProjOperator selected_op, const char *phase,
    const std::string &quant_label, int quant_type, int batch_rows,
    int intermediate_size, int hidden_size, TryQ81Fn &&try_q81_group,
    TryPackedFn &&try_packed_group, FallbackFn &&run_fallback,
    NativeFfnExecutionSummary *summary = nullptr) {
  NativeFfnExecutionSummary local_summary;

  switch (selected_op) {
  case FfnProjOperator::kQ81Group:
  case FfnProjOperator::kQ81GroupHotQ4K:
  case FfnProjOperator::kQ81GroupRowPairW4:
  case FfnProjOperator::kQ81GroupRowQuadM4:
  case FfnProjOperator::kQ81GroupMmq3:
    local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81_group)();
    if (local_summary.used_q81) {
      local_summary.actual_op = selected_op;
    }
    break;
  case FfnProjOperator::kPackedGroup:
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed_group)();
    if (local_summary.used_packed) {
      local_summary.actual_op = selected_op;
    }
    break;
  case FfnProjOperator::kFallback:
    // Selector requested the dense fallback (force_cublas or no fused tier
    // ready) — skip the fused catch-alls entirely.
    break;
  }

  if (!local_summary.used_q81 && !local_summary.used_packed &&
      selected_op != FfnProjOperator::kPackedGroup &&
      selected_op != FfnProjOperator::kFallback) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed_group)();
    if (local_summary.used_packed) {
      local_summary.actual_op = FfnProjOperator::kPackedGroup;
    }
  }

  const char *metric_op =
      FfnMetricLabel(local_summary.actual_op, quant_type, batch_rows);
  GlobalMetrics().RecordInferfluxCudaFfnProjOperator(phase, metric_op);
  GlobalMetrics().RecordInferfluxCudaFfnProjGeometry(
      phase, metric_op, quant_label, batch_rows, intermediate_size, hidden_size,
      /*grouped_outputs=*/2);
  const std::string_view metric_op_view(metric_op);
  if (metric_op_view.find("row_pair") != std::string_view::npos) {
    GlobalMetrics().RecordInferfluxCudaRowPairSelection(phase, metric_op,
                                                        batch_rows);
  }

  if (summary) {
    *summary = local_summary;
  }

  if (!local_summary.used_q81 && !local_summary.used_packed) {
    return std::forward<FallbackFn>(run_fallback)();
  }
  return true;
}

struct NativeDownProjExecutionSummary {
  DownProjOperator actual_op{DownProjOperator::kFallback};
  bool used_mmq{false};
  bool used_q81{false};
  bool used_packed{false};
};

template <typename TryMmqFn, typename TryQ81Fn, typename TryPackedFn,
          typename FallbackFn, typename LogFn>
bool ExecuteInferfluxCudaDownProjStage(
    DownProjOperator selected_op, const char *phase,
    const std::string &quant_label, int quant_type, int batch_rows,
    int hidden_size, int intermediate_size, TryMmqFn &&try_mmq,
    TryQ81Fn &&try_q81, TryPackedFn &&try_packed, FallbackFn &&run_fallback,
    LogFn &&log_selected_operator,
    NativeDownProjExecutionSummary *summary = nullptr) {
  NativeDownProjExecutionSummary local_summary;

  switch (selected_op) {
  case DownProjOperator::kMmq:
    local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
    if (local_summary.used_mmq) {
      local_summary.actual_op = DownProjOperator::kMmq;
    }
    if (!local_summary.used_mmq) {
      local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81)();
      if (local_summary.used_q81) {
        // Report what actually ran, not what was selected — the old code
        // set actual_op = selected_op (kMmq) while the Q8_1 GEMV executed,
        // mislabeling the operator counters.
        local_summary.actual_op = DownProjOperator::kQ81Gemv;
      }
    }
    break;
  case DownProjOperator::kPackedGemv:
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed)();
    if (local_summary.used_packed) {
      local_summary.actual_op = selected_op;
    }
    break;
  case DownProjOperator::kQ81Gemv:
  case DownProjOperator::kQ81GemvHotFixed:
  case DownProjOperator::kQ81GemvRowPairHotFixed:
  case DownProjOperator::kQ81GemvRowPair:
  case DownProjOperator::kQ81GemvRowQuad:
    local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81)();
    if (local_summary.used_q81) {
      local_summary.actual_op = selected_op;
    }
    if (!local_summary.used_q81) {
      local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
      if (local_summary.used_mmq) {
        local_summary.actual_op = DownProjOperator::kMmq;
      }
    }
    break;
  case DownProjOperator::kFallback:
    // Selector requested the dense fallback — skip the fused catch-alls.
    break;
  }

  if (!local_summary.used_mmq && !local_summary.used_q81 &&
      !local_summary.used_packed &&
      selected_op != DownProjOperator::kPackedGemv &&
      selected_op != DownProjOperator::kFallback) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed)();
    if (local_summary.used_packed) {
      local_summary.actual_op = DownProjOperator::kPackedGemv;
    }
  }

  const char *metric_op =
      DownMetricLabel(local_summary.actual_op, quant_type, batch_rows);
  GlobalMetrics().RecordInferfluxCudaDownProjOperator(phase, metric_op);
  GlobalMetrics().RecordInferfluxCudaDownProjGeometry(
      phase, metric_op, quant_label, batch_rows, hidden_size,
      intermediate_size);
  if (local_summary.actual_op == DownProjOperator::kQ81GemvRowPair ||
      local_summary.actual_op == DownProjOperator::kQ81GemvRowPairHotFixed) {
    GlobalMetrics().RecordInferfluxCudaRowPairSelection(phase, metric_op,
                                                        batch_rows);
  }

  if (local_summary.actual_op != DownProjOperator::kFallback) {
    std::forward<LogFn>(log_selected_operator)(local_summary.actual_op);
  }

  if (summary) {
    *summary = local_summary;
  }

  if (!local_summary.used_mmq && !local_summary.used_q81 &&
      !local_summary.used_packed) {
    return std::forward<FallbackFn>(run_fallback)();
  }
  return true;
}

} // namespace inferflux
