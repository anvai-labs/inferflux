# Q8_1 Row-Pair / Down-Projection Investigation

## 1. Why this matters
- The InferFlux CUDA path still trails `llama.cpp` in concurrent throughput despite the active `q8_1_group` and `q8_1_gemv` kernels.
- Historical experiments (Ada RTX 4000 bench, `decode_burst_tokens=0`, `continuoustools`) showed that when the experimental `q8_1_group_row_pair_w4` / `q8_1_group_row_pair` and matching down-proj row-pair kernels were active, scaling to M=2 decoded sequences improved throughput even though accuracy/regression blockers forced the path to be disabled.
- Single decode lane architecture (decode pool = 1) already exposes the hot `M=2` geometry for gate/up; stabilizing the grouped row-pair kernel unlocks the next throughput milestone while keeping per-request alignment.

## 2. Current code review
### Gate/Up grouped kernels
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh` defines:
  - `fused_dequant_gemv_q4k_q8_1_group` (generic grouping) and `..._group_rowpair` / `..._group_rowquad` variants. The rowpair version is guarded in dispatch because of the comment: `Kept for rework ... decoded output corrupted on RTX 4000 Ada`.
  - Row pair kernel folds two activations using `paired` warps, reuses `block_q8_1` activations, and shares `block_q4_k` weight indices across both rows via `pair` and `get_scale_min_k4` helpers.
  - The guard width in the rowpair kernel may require tight bounds on lens (M=2, `kGemvWarpsPerBlock`, `pair`, `lane`), which may have drifted from real `qwen2.5-3b` shapes after geometry adjustments.
- Dispatch decisions are recorded via `inferflux_cuda_ffn_proj_operator_total` metrics (`q8_1_group`, `q8_1_group_row_pair_w4`, `q8_1_group_hot_q4k`, etc.) making it possible to confirm the row-pair path still gets selected once stability is restored.

### Down-projection kernels
- `fused_dequant_gemv_q4k_q8_1_group_rowpair_fixed_blocks` handles M=2 down-proj with fixed block counts; there is also `fused_dequant_gemv_q4k_q8_1_group_rowpair` (experimental, not currently used) and similar functions for `q6/q8` quantizations.
- These kernels assume exact shapes (`K=11008`, `N=2048`, `Q4_K` block alignment). The comment `row-pair/row-quad for M>1` suggests the down-proj dispatcher falls back to row pair when `forward_batch_size` buckets hit `3_4` or `2`.
- The `down_proj` operator summary from benchmarks now shows `q8_1_gemv` and `q8_1_gemv_row_pair` working for M=1/2, but the more fused row-pair path (mirroring gated fused kernel) may still be silent due to dispatch gating or stability concerns.

## 3. Stability hypotheses & verification steps
1. **Row-pair geometry mismatches**: The active `kGemvWarpsPerBlock` was tuned for `M=1` decode (64/2). The fused row-pair kernel locks the grid to `M/2` warps, which might mis-index the `PackedProjectionGroupParams` structures when the warp IDs exceed `row` range (e.g., due to `WarpsPerBlock` template parameters). We should confirm `row`/`row_base` math matches the actual `total_rows` values emitted by `InferfluxCudaExecutor` for `M=2` (FFN gate/up) and `down_proj` (M=1 vs M=2). Logging `forward_batch_size` bucket >1 and `kGemvWarpsPerBlock` at runtime will help.
2. **Activation/weight reuse**: The row-pair kernels rely on `act_q8_1` being stored sequentially per row block. If `PackedProjectionGroupParams` changed (e.g., `num_super_blocks` now derived from `NumSuperBlocks` template), the indexing in `a_row0`, `a_row1` or `params.weights` may no longer match the layout produced by the quantized loader (`quantized/block_utils`). Validating these pointers by replaying the same shapes in a unit test (mock `PackedProjectionGroupParams` for `qwen2.5-3b` values) helps catch mismatched strides.
3. **Down-proj weight alignment**: In `block_q4_k`, each weight row is stored in `scales`/`qs`; the row-pair kernel that handles two outputs uses `pair*32+offs` to index these bytes. If `packed_projection_group_params.weights[i]` now has padding or `NumSuperBlocks` is not consistent across `q4/q6` quant types, the two rows will read stale data, explaining the corrupted output earlier. Inspect `model/gguf/quantized_loader` to ensure `block_q4_k` layout hasn't changed.
4. **Dispatcher gating**: InferFlux CUDA kernel selection is controlled by `InferfluxCudaExecutor::DispatchFusedGemv`. The row-pair path may still be opt-in via flags such as `INFERFLUX_GEMV_ENABLE_ROW_PAIR` or similar; the previous harness comment indicates `row-pair/row-quad` is used if multi-row bucket is active. We should re-check `runtime/backends/cuda/inferflux_cuda_executor.cpp` around `DecodeForwardBatch` to confirm the gating logic still covers `M=2` and not only `M=1`.

## 4. Immediate next actions (Phase 1)
- [ ] Force `decode_pool_size=1` (done) so we maintain a single decode lane and consistently observe the `M=2` geometry without scheduler-induced splitting.
- [ ] Add instrumentation around the row-pair kernels to log when they become eligible (via metrics or `INFERFLUX_DEBUG_NATIVE_KERNELS=1`). Capture `forward_batch_size` buckets 2/3/4 to confirm the workload sees rows pairing.
- [ ] Write a small kernel unit test (possibly in `tests/native/`) that allocates a mock `PackedProjectionGroupParams` for `qwen2.5-3b` shapes and runs the row-pair path with deterministic weights (identical to `q8_1_group`) to confirm output matches known results.
- [ ] Review the existing `quantized/block_q4_k` (GGUF loader) code to ensure `d`, `scales`, `qs` offsets remain consistent for fused row pair down-proj. Extend `scripts/run_gguf_comparison_benchmark.sh` logs (via `NATIVE_PHASE_TIMING`) to include `down_proj` geometry per bucket.

## 5. Future phases (not part of this pass)
- Phase 2: re-enable `fused_dequant_gemv_q4k_q8_1_group_rowpair` for gate/up once geometry/logging confirms shapes align, and hold it behind `INFERFLUX_GEMV_ROW_PAIR=1` until accuracy matches `llama.cpp`.
- Phase 3: Pair the down-proj row-pair operator with the fused FFN path for `M=2` sequences and benchmark to ensure latencies stay consistent.
- Phase 4: Document any hardware-imperfection-specific fixes (e.g., warp-control adjustments) discovered while debugging.
