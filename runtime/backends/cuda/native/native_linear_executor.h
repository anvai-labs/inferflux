#pragma once

#include "runtime/backends/cuda/native/dispatch_catalog.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"

#include "server/logging/logger.h"

#include "server/metrics/metrics.h"
#include <cstdio>

#include <string>
#include <string_view>
#include <utility>

namespace inferflux {

// Budgeted dispatch-trace emission (thread_local budget resets when the
// limit changes, mirroring ConsumeOperatorSelectionBudget).
inline bool ConsumeDispatchTraceBudget(const NativeExecutionPolicy *policy) {
  if (!policy || !policy->dispatch_trace) {
    return false;
  }
  struct Budget {
    int limit;
    int used;
  };
  thread_local Budget budget{0, 0};
  if (budget.limit != policy->dispatch_trace_limit) {
    budget = Budget{policy->dispatch_trace_limit, 0};
  }
  if (budget.used >= budget.limit) {
    return false;
  }
  ++budget.used;
  return true;
}

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
    NativeFfnExecutionSummary *summary = nullptr,
    const NativeExecutionPolicy *policy = nullptr) {
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

  const bool ffn_divergent =
      TierOf(selected_op) != TierOf(local_summary.actual_op) ||
      (local_summary.actual_op == FfnProjOperator::kFallback &&
       selected_op != FfnProjOperator::kFallback);
  if (ffn_divergent) {
    ReportDispatchDivergence("ffn", FfnSelectionLabel(selected_op),
                             FfnSelectionLabel(local_summary.actual_op),
                             "tier_mismatch");
  }
  if (ConsumeDispatchTraceBudget(policy)) {
    // Direct stderr like [phase_timing]: a diagnostic trace that must
    // survive warning-level logging in benchmark harnesses.
    std::fprintf(stderr,
                 "[dispatch_trace] [ffn]: phase=%s M=%d N=%d K=%d "
                 "selected=%s actual=%s tier=%s%s\n",
                 phase, batch_rows, intermediate_size, hidden_size,
                 FfnSelectionLabel(selected_op),
                 FfnSelectionLabel(local_summary.actual_op),
                 DispatchTierLabel(TierOf(local_summary.actual_op)),
                 ffn_divergent ? " reason=tier_mismatch" : "");
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
  bool used_mmq_mma{false};
  bool used_mmq{false};
  bool used_q81{false};
  bool used_packed{false};
};

template <typename TryMmqMmaFn, typename TryMmqFn, typename TryQ81Fn,
          typename TryPackedFn, typename FallbackFn, typename LogFn>
bool ExecuteInferfluxCudaDownProjStage(
    DownProjOperator selected_op, const char *phase,
    const std::string &quant_label, int quant_type, int batch_rows,
    int hidden_size, int intermediate_size, TryMmqMmaFn &&try_mmq_mma,
    TryMmqFn &&try_mmq, TryQ81Fn &&try_q81, TryPackedFn &&try_packed,
    FallbackFn &&run_fallback, LogFn &&log_selected_operator,
    NativeDownProjExecutionSummary *summary = nullptr,
    const NativeExecutionPolicy *policy = nullptr) {
  NativeDownProjExecutionSummary local_summary;

  switch (selected_op) {
  case DownProjOperator::kMmqMma:
    // Tensor-core tier selected: MMA first, then the dp4a MMQ, then Q8_1.
    local_summary.used_mmq_mma = std::forward<TryMmqMmaFn>(try_mmq_mma)();
    if (local_summary.used_mmq_mma) {
      local_summary.actual_op = DownProjOperator::kMmqMma;
      break;
    }
    local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
    if (local_summary.used_mmq) {
      local_summary.actual_op = DownProjOperator::kMmq;
      break;
    }
    local_summary.used_q81 = std::forward<TryQ81Fn>(try_q81)();
    if (local_summary.used_q81) {
      local_summary.actual_op = DownProjOperator::kQ81Gemv;
    }
    break;
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
      // MMA before dp4a — the prefill sites allow MMA above the decode
      // window (mmq_mma_max_prefill_batch), so an MMA attempt can still
      // succeed under a Q8_1 selection. Keep the historical order.
      local_summary.used_mmq_mma = std::forward<TryMmqMmaFn>(try_mmq_mma)();
      if (local_summary.used_mmq_mma) {
        local_summary.actual_op = DownProjOperator::kMmqMma;
      }
      if (!local_summary.used_mmq_mma) {
        local_summary.used_mmq = std::forward<TryMmqFn>(try_mmq)();
        if (local_summary.used_mmq) {
          local_summary.actual_op = DownProjOperator::kMmq;
        }
      }
    }
    break;
  case DownProjOperator::kFallback:
    // Selector requested the dense fallback — skip the fused catch-alls.
    break;
  }

  if (!local_summary.used_mmq_mma && !local_summary.used_mmq &&
      !local_summary.used_q81 && !local_summary.used_packed &&
      selected_op != DownProjOperator::kPackedGemv &&
      selected_op != DownProjOperator::kFallback) {
    local_summary.used_packed = std::forward<TryPackedFn>(try_packed)();
    if (local_summary.used_packed) {
      local_summary.actual_op = DownProjOperator::kPackedGemv;
    }
  }

  const bool down_divergent =
      TierOf(selected_op) != TierOf(local_summary.actual_op) ||
      (local_summary.actual_op == DownProjOperator::kFallback &&
       selected_op != DownProjOperator::kFallback);
  if (down_divergent) {
    ReportDispatchDivergence("down_proj", DownSelectionLabel(selected_op),
                             DownSelectionLabel(local_summary.actual_op),
                             "tier_mismatch");
  }
  if (ConsumeDispatchTraceBudget(policy)) {
    std::fprintf(stderr,
                 "[dispatch_trace] [down_proj]: phase=%s M=%d N=%d K=%d "
                 "selected=%s actual=%s tier=%s%s\n",
                 phase, batch_rows, hidden_size, intermediate_size,
                 DownSelectionLabel(selected_op),
                 DownSelectionLabel(local_summary.actual_op),
                 DispatchTierLabel(TierOf(local_summary.actual_op)),
                 down_divergent ? " reason=tier_mismatch" : "");
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

  if (!local_summary.used_mmq_mma && !local_summary.used_mmq &&
      !local_summary.used_q81 && !local_summary.used_packed) {
    return std::forward<FallbackFn>(run_fallback)();
  }
  return true;
}

} // namespace inferflux
