#pragma once

#include "runtime/string_utils.h"

#include <cstdlib>
#include <limits>
#include <string>

namespace inferflux {

struct NativeExecutionPolicy {
  bool enable_batched_decode{true};
  bool disable_cuda_graph{false};
  // Burst-pipelined decode: enqueue K decode steps (forward + sample +
  // on-device token feed) per ExecuteUnifiedBatchBurst() call and consume
  // tokens from a pinned ring while the GPU runs ahead. Default off until
  // parity + benchmark gates pass (S1 rollout).
  bool enable_decode_burst{false};
  int decode_burst_chunk{8};
  int decode_burst_max_ms{40};
  // Split-parallel decode attention (S3): parallelize FlashDecode over
  // Q-heads (short context) and KV chunks (long context). Default off.
  bool enable_attn_split_kv{false};
  int attn_split_chunk{512};
  int attn_split_qsplit_override{-1};
  // Prefer the packed dp4a tier over Q8_1 activations for tiny decode
  // batches (measured faster at M=1-2 on Ada; MMQ keeps winning at larger
  // M). Applied by switching the WHOLE compute-section policy to
  // disable_q81_activations for the step — gating individual dispatch
  // sites desyncs operator selection onto cuBLAS fallbacks. -1 = off.
  int packed_tier_max_batch{-1};
  // Use the fused gate+up+SiLU MMVQ kernel only for batches up to this
  // size. The fused kernel is memory-bandwidth-inefficient as M grows
  // (32% of peak at M=16 vs 63% at M=1); the unfused grouped path wins
  // by ~13% end to end at c=16. -1 = always (historic default).
  int fused_gate_up_silu_max_batch{-1};
  // Fold Q/K bias into BatchedRoPE's register load and V bias into the KV
  // append store, deleting the three per-layer BiasAdd launches. Off by
  // default; null-bias models take the identical old path either way.
  bool enable_bias_rope_fusion{false};
  // Unified dispatch trace: one machine-parseable line per projection
  // dispatch (phase, geometry, selected, actual, tier). Budgeted like the
  // other debug traces; parsed by scripts/parse_dispatch_trace.py.
  bool dispatch_trace{false};
  int dispatch_trace_limit{256};
  bool phase_timing_enabled{false};
  bool force_cublas{false};
  bool disable_prepacked_activations{false};
  bool disable_q81_activations{false};
  bool disable_fused_gemv{false};
  bool debug_decode_mapping{false};
  int debug_decode_mapping_limit{32};
  bool debug_logits{false};
  int debug_logits_limit{64};
  bool debug_operator_selection{false};
  int debug_operator_selection_limit{64};
  int timing_sample_rate{0};
  bool require_fused_quantized_matmul_override{false};
  bool require_fused_quantized_matmul{false};
  std::string dequantized_cache_policy_override;
  bool enable_experimental_q81_triple_rowpair{false};
  bool enable_experimental_q81_downproj_hot_fixed{false};
  bool enable_experimental_q81_downproj_rowpair_hot_fixed{false};
  bool enable_experimental_q81_grouped_hot_q4k{true};
  bool enable_experimental_q81_grouped_rowpair_w4{true};
  bool enable_experimental_q81_grouped_rowquad_m4{false};
  bool enable_experimental_q81_grouped_mmq3{true};
  bool enable_downproj_mmq{false};
  int downproj_mmq_min_batch_override{-1};
  // mma.sync Q6_K down-projection (S7): int8 tensor-core MMQ with
  // deterministic K-split. Default off; A/B gate pending. When on, the
  // down-proj MMQ tier prefers the MMA kernel for Q6_K weights.
  bool enable_mmq_mma{false};
  // MMA only wins where the dp4a MMVQ tier stops being at the bandwidth
  // floor: measured -13% at c4 (M=4) vs the incumbent MMVQ, so the floor
  // M is 9 (the same batch range the dp4a MMQ threshold targets).
  int mmq_mma_min_batch{9};
  int mmq_mma_max_batch{16};
  bool use_vectorized_loads{false};
  bool enable_fused_gate_up_silu{true};
  bool enable_adaptive_mmvq_threads{true};
  int mmvq_min_warps_override{-1};
  int mmvq_max_warps_override{-1};
  bool enable_fused_residual_norm{true};
  bool enable_fused_bias_add{true};
  bool enable_gemv_accumulate{true};
  bool enable_batch_dequant_cache{false};

  // Fused kernel redesign flags (P1+P2 validated: parity-exact, zero
  // regression)
  bool enable_fused_rope_kv_append{true};           // P1: validated
  bool enable_fused_gemv_norm_quant_epilogue{true}; // P2: validated
  bool enable_mmvq_bias_epilogue{false};            // P3
  bool enable_q6k_vectorized{false};                // P4
  bool enable_gate_up_silu_q81_epilogue{false};     // P5

  // Precision: keep residual stream in FP32 to match llama.cpp numerical
  // behavior.  FP16 roundtripping across 36 layers compounds quantization
  // error, causing multi-token response divergence (Jaccard ~0.10).
  bool enable_fp32_residual{true};

  static NativeExecutionPolicy FromEnv() {
    NativeExecutionPolicy policy;
    policy.enable_batched_decode =
        ParseBoolEnv("INFERFLUX_ENABLE_BATCHED_DECODE", true);
    policy.enable_decode_burst =
        ParseBoolEnv("INFERFLUX_CUDA_DECODE_BURST", false);
    policy.decode_burst_chunk =
        ParseIntEnv("INFERFLUX_CUDA_DECODE_BURST_CHUNK", 8, 1, 32);
    policy.decode_burst_max_ms =
        ParseIntEnv("INFERFLUX_CUDA_DECODE_BURST_MAX_MS", 40, 1, 1000);
    policy.enable_attn_split_kv =
        ParseBoolEnv("INFERFLUX_CUDA_ATTN_SPLIT_KV", false);
    policy.attn_split_chunk =
        ParseIntEnv("INFERFLUX_CUDA_ATTN_SPLIT_CHUNK", 512, 64, 8192);
    policy.attn_split_qsplit_override =
        ParseIntEnv("INFERFLUX_CUDA_ATTN_QSPLIT", -1, 1, 8);
    policy.packed_tier_max_batch =
        ParseIntEnv("INFERFLUX_CUDA_PACKED_TIER_MAX_BATCH", -1, -1, 8);
    policy.fused_gate_up_silu_max_batch =
        ParseIntEnv("INFERFLUX_CUDA_FUSED_GATE_UP_SILU_MAX_BATCH", -1, -1, 64);
    policy.enable_bias_rope_fusion =
        ParseBoolEnv("INFERFLUX_CUDA_FUSE_BIAS_ROPE", false);
    policy.dispatch_trace =
        ParseBoolEnv("INFERFLUX_CUDA_DISPATCH_TRACE", false);
    policy.dispatch_trace_limit =
        ParseIntEnv("INFERFLUX_CUDA_DISPATCH_TRACE_LIMIT", 256, 1,
                    std::numeric_limits<int>::max());
    // CUDA graph capture: cudaDeviceSynchronize drains async work before
    // capture to prevent heap corruption. Disable with
    // INFERFLUX_DISABLE_CUDA_GRAPH=1 if issues arise.
    policy.disable_cuda_graph =
        ParseBoolEnv("INFERFLUX_DISABLE_CUDA_GRAPH", false);
    policy.phase_timing_enabled =
        ParseBoolEnv("INFERFLUX_CUDA_PHASE_TIMING", false);
    policy.force_cublas = ParseBoolEnv("INFERFLUX_FORCE_CUBLAS", false);
    policy.disable_prepacked_activations =
        ParseBoolEnv("INFERFLUX_DISABLE_PREPACKED_ACTIVATIONS", false);
    policy.disable_q81_activations =
        ParseBoolEnv("INFERFLUX_DISABLE_Q8_1_ACTIVATIONS", false);
    policy.disable_fused_gemv =
        ParseBoolEnv("INFERFLUX_DISABLE_FUSED_GEMV", false);
    policy.debug_decode_mapping =
        ParseBoolEnv("INFERFLUX_CUDA_DEBUG_DECODE_MAPPING", false);
    policy.debug_decode_mapping_limit =
        ParseIntEnv("INFERFLUX_CUDA_DEBUG_DECODE_MAPPING_LIMIT", 32, 1,
                    std::numeric_limits<int>::max());
    policy.debug_logits = ParseBoolEnv("INFERFLUX_DEBUG_LOGITS", false);
    policy.debug_logits_limit = ParseIntEnv("INFERFLUX_DEBUG_LOGITS_LIMIT", 64,
                                            1, std::numeric_limits<int>::max());
    policy.debug_operator_selection =
        ParseBoolEnv("INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION", false);
    policy.debug_operator_selection_limit =
        ParseIntEnv("INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION_LIMIT", 64, 1,
                    std::numeric_limits<int>::max());
    policy.timing_sample_rate =
        ParseIntEnv("INFERFLUX_CUDA_TIMING_SAMPLE_RATE", 0, 0,
                    std::numeric_limits<int>::max());
    if (std::getenv("INFERFLUX_CUDA_REQUIRE_FUSED_MATMUL")) {
      policy.require_fused_quantized_matmul_override = true;
      policy.require_fused_quantized_matmul =
          ParseBoolEnv("INFERFLUX_CUDA_REQUIRE_FUSED_MATMUL", false);
    }
    if (const char *dequant_override =
            std::getenv("INFERFLUX_CUDA_DEQUANT_CACHE_POLICY")) {
      policy.dequantized_cache_policy_override = dequant_override;
    }
    policy.enable_experimental_q81_triple_rowpair = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_TRIPLE_ROWPAIR", false);
    policy.enable_experimental_q81_downproj_hot_fixed = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_HOT_FIXED", false);
    policy.enable_experimental_q81_downproj_rowpair_hot_fixed = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_DOWNPROJ_ROWPAIR_HOT_FIXED", false);
    policy.enable_experimental_q81_grouped_hot_q4k = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_HOT_Q4K", true);
    policy.enable_experimental_q81_grouped_rowpair_w4 = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWPAIR_W4", true);
    policy.enable_experimental_q81_grouped_rowquad_m4 = ParseBoolEnv(
        "INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWQUAD_M4", false);
    policy.enable_experimental_q81_grouped_mmq3 =
        ParseBoolEnv("INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_MMQ3", true);
    policy.enable_downproj_mmq =
        ParseBoolEnv("INFERFLUX_ENABLE_DOWNPROJ_MMQ", false);
    policy.downproj_mmq_min_batch_override =
        ParseIntEnv("INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH", -1, 1, 64);
    policy.enable_mmq_mma = ParseBoolEnv("INFERFLUX_CUDA_MMQ_MMA", false);
    policy.mmq_mma_min_batch =
        ParseIntEnv("INFERFLUX_CUDA_MMQ_MMA_MIN_BATCH", 9, 1, 64);
    policy.mmq_mma_max_batch =
        ParseIntEnv("INFERFLUX_CUDA_MMQ_MMA_MAX_BATCH", 16, 1, 64);
    policy.use_vectorized_loads =
        ParseBoolEnv("INFERFLUX_USE_VECTORIZED_LOADS", false);
    policy.enable_fused_gate_up_silu =
        ParseBoolEnv("INFERFLUX_ENABLE_FUSED_GATE_UP_SILU", true);
    policy.enable_adaptive_mmvq_threads =
        ParseBoolEnv("INFERFLUX_ENABLE_ADAPTIVE_MMVQ_THREADS", true);
    policy.mmvq_min_warps_override =
        ParseIntEnv("INFERFLUX_MMVQ_MIN_WARPS", -1, 1, 8);
    policy.mmvq_max_warps_override =
        ParseIntEnv("INFERFLUX_MMVQ_MAX_WARPS", -1, 1, 8);
    policy.enable_fused_residual_norm =
        ParseBoolEnv("INFERFLUX_ENABLE_FUSED_RESIDUAL_NORM", true);
    policy.enable_fused_bias_add =
        ParseBoolEnv("INFERFLUX_ENABLE_FUSED_BIAS_ADD", true);
    policy.enable_gemv_accumulate =
        ParseBoolEnv("INFERFLUX_ENABLE_GEMV_ACCUMULATE", true);
    policy.enable_batch_dequant_cache =
        ParseBoolEnv("INFERFLUX_BATCH_DEQUANT_CACHE", false);
    policy.enable_fused_rope_kv_append =
        ParseBoolEnv("INFERFLUX_ENABLE_FUSED_ROPE_KV_APPEND", true);
    policy.enable_fused_gemv_norm_quant_epilogue =
        ParseBoolEnv("INFERFLUX_ENABLE_FUSED_GEMV_NORM_QUANT_EPILOGUE", true);
    policy.enable_mmvq_bias_epilogue =
        ParseBoolEnv("INFERFLUX_ENABLE_MMVQ_BIAS_EPILOGUE", false);
    policy.enable_q6k_vectorized =
        ParseBoolEnv("INFERFLUX_ENABLE_Q6K_VECTORIZED", false);
    policy.enable_gate_up_silu_q81_epilogue =
        ParseBoolEnv("INFERFLUX_ENABLE_GATE_UP_SILU_Q81_EPILOGUE", false);
    policy.enable_fp32_residual =
        ParseBoolEnv("INFERFLUX_CUDA_FP32_RESIDUAL", true);
    return policy;
  }
};

} // namespace inferflux
