#pragma once

#include "runtime/backends/cuda/native/fused_quant_gemm.h"

#include "server/metrics/metrics.h"

#include <string>
#include <string_view>
#include <utility>

namespace inferflux {

struct NativeFfnExecutionSummary {
  FusedQuantGemm::FfnProjOperator actual_op{
      FusedQuantGemm::FfnProjOperator::kFallback};
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
    FusedQuantGemm::FfnProjOperator selected_op, const char *phase,
    const std::string &quant_label, int quant_type, int batch_rows,
    int intermediate_size, int hidden_size, TryQ81Fn &&try_q81_group,
    TryPackedFn &&try_packed_group, FallbackFn &&run_fallback,
    NativeFfnExecutionSummary *summary = nullptr) {
  NativeFfnExecutionSummary local_summary;

  if (selected_op == FusedQuantGemm::FfnProjOperator::kQ81Group ||
      selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K ||
      selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupRowPairW4 ||
      selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4 ||
      selected_op == FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3) {
    local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81_group)();
    if (local_summary.used_q81) {
      local_summary.actual_op = selected_op;
    }
  } else if (selected_op == FusedQuantGemm::FfnProjOperator::kPackedGroup) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed_group)();
    if (local_summary.used_packed) {
      local_summary.actual_op = selected_op;
    }
  }

  if (!local_summary.used_q81 && !local_summary.used_packed &&
      selected_op != FusedQuantGemm::FfnProjOperator::kPackedGroup) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed_group)();
    if (local_summary.used_packed) {
      local_summary.actual_op = FusedQuantGemm::FfnProjOperator::kPackedGroup;
    }
  }

  const char *metric_op = FusedQuantGemm::FfnProjOperatorMetricName(
      local_summary.actual_op, quant_type, batch_rows, hidden_size);
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
  FusedQuantGemm::DownProjOperator actual_op{
      FusedQuantGemm::DownProjOperator::kFallback};
  bool used_mmq{false};
  bool used_q81{false};
  bool used_packed{false};
};

template <typename TryMmqFn, typename TryQ81Fn, typename TryPackedFn,
          typename FallbackFn, typename LogFn>
bool ExecuteInferfluxCudaDownProjStage(
    FusedQuantGemm::DownProjOperator selected_op, const char *phase,
    const std::string &quant_label, int quant_type, int batch_rows,
    int hidden_size, int intermediate_size, TryMmqFn &&try_mmq,
    TryQ81Fn &&try_q81, TryPackedFn &&try_packed, FallbackFn &&run_fallback,
    LogFn &&log_selected_operator,
    NativeDownProjExecutionSummary *summary = nullptr) {
  NativeDownProjExecutionSummary local_summary;

  if (selected_op == FusedQuantGemm::DownProjOperator::kMmq) {
    local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
    if (local_summary.used_mmq) {
      local_summary.actual_op = FusedQuantGemm::DownProjOperator::kMmq;
    }
    if (!local_summary.used_mmq) {
      local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81)();
      if (local_summary.used_q81) {
        local_summary.actual_op = selected_op;
      }
    }
  } else if (selected_op == FusedQuantGemm::DownProjOperator::kPackedGemv) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed)();
    if (local_summary.used_packed) {
      local_summary.actual_op = selected_op;
    }
  } else {
    local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81)();
    if (local_summary.used_q81) {
      local_summary.actual_op = selected_op;
    }
    if (!local_summary.used_q81) {
      local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
      if (local_summary.used_mmq) {
        local_summary.actual_op = FusedQuantGemm::DownProjOperator::kMmq;
      }
    }
  }

  if (!local_summary.used_mmq && !local_summary.used_q81 &&
      !local_summary.used_packed) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed)();
    if (local_summary.used_packed) {
      local_summary.actual_op = FusedQuantGemm::DownProjOperator::kPackedGemv;
    }
  }

  const char *metric_op = FusedQuantGemm::DownProjOperatorMetricName(
      local_summary.actual_op, quant_type, batch_rows, intermediate_size);
  GlobalMetrics().RecordInferfluxCudaDownProjOperator(phase, metric_op);
  GlobalMetrics().RecordInferfluxCudaDownProjGeometry(
      phase, metric_op, quant_label, batch_rows, hidden_size,
      intermediate_size);
  if (local_summary.actual_op ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPair ||
      local_summary.actual_op ==
          FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed) {
    GlobalMetrics().RecordInferfluxCudaRowPairSelection(phase, metric_op,
                                                        batch_rows);
  }

  if (local_summary.actual_op != FusedQuantGemm::DownProjOperator::kFallback) {
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
