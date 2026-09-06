# Safetensors Decode Performance: Measurements, Falsified Hypotheses, and Plan

**Status:** Active (co-design baseline, Sep 6 2026)
**Evidence:** nsys profiles (`nsys_profile_ifx_st_c16_clean`, `..._c1`, local),
scheduler metrics from live loads, and the `tests/tools/bf16_gemv_bench.cu`
spike. Model: Qwen2.5-3B bf16 safetensors, RTX 4000 Ada (48 SMs, ~576 GB/s,
48 MB L2).

## 1) Where the throughput goes (verified)

Gap to vLLM/SGLang at c=16 is ~2.4-2.7x (278.7 vs 753.4/659.8 tok/s).
Decomposition, per decode step (avg width ~7-10, batch-histogram-driven):

| Factor | Measured | Headroom |
|---|---|---|
| GEMM kernel efficiency | cuBLAS bf16 skinny GEMMs at 30-44% of DRAM roofline cold-L2 (in-server ~43%, matches); 87% of GPU time | ~2x vs physics |
| Per-step host syncs | ~7 chunky `cudaStreamSynchronize`/step, GPU ~88.5% busy | ~1.12x |
| Decode batch width | avg 5.3-10.7 across runs (EOS-staggered completions + client arrivals); 38% of steps at batch=1 in the nsys run | workload-shaped, see §3 |

Cross-check: width-ratio x kernel-ratio x sync-ratio reproduces the observed
gap within variance. At c=1 the engines are only ~1.13x apart — the gap is
concurrency behavior, not single-stream kernel quality.

## 2) Falsified hypotheses (do not re-chase without new evidence)

1. **"A simple custom GEMV beats cuBLAS on skinny bf16 shapes."** Falsified
   by the spike (`tests/tools/bf16_gemv_bench.cu`): a warp-per-row,
   16-byte-vector-load kernel with M register accumulators is numerically
   correct and beats warm-L2 cuBLAS 1.5-2.9x at M=1 — but under cold L2
   (the realistic condition; attention kernels evict L2 between
   projections) it only matches cuBLAS (0.86-1.04x) on fat shapes and loses
   everywhere at M>=4 (register-accumulator FLOPs scale with M while
   bandwidth doesn't). cuBLAS cold-L2: gate/up 41-45%, down 44%, qkv/o_proj
   30-33%, lm_head M=1 54%. A higher-MLP 4-rows-per-warp variant was also
   attempted and failed (implementation bug, 0.04x — removed from the tool).
   Beating cuBLAS here requires the heavyweight design: multi-row block
   tiles, cp.async double-buffered weight streaming, deep unrolled MLP —
   i.e. a specialist kernel project (the llama.cpp MMVQ philosophy applied
   to bf16), not a quick rewrite.
2. **"Widening `batch_accumulation_ms` raises decode batch width."**
   Falsified by A/B sweep (2 runs per point, 32 req x 64 tok, c=16):
   accum 2/8/16 ms -> 320.7 / 329.3 / 292.4 tok/s average. Width is limited
   by EOS-staggered completions (sequences now stop at natural EOS at
   different lengths — a direct consequence of the PR #81 stop-token fix)
   plus closed-loop client arrivals, not by the merge window. 16 ms is
   strictly worse (steps wait, nothing merges).
3. **"GPU idle dominates."** The ~55% idle seen in early profiles was an
   artifact of `INFERFLUX_CUDA_PHASE_TIMING=1` (stream syncs between every
   phase of every layer, ~324/step) — it halves measured throughput. Clean
   runs show 88.5% GPU busy; sync overhead is ~1.12x, not 2x+.

## 3) Batch width is mostly workload, not a scheduler bug

The scheduler refills decode cohorts correctly when pending work exists
(85/206 steps at width 15-16 in a live-metrics run; sticky merge works).
Low-width steps concentrate at wave transitions where the closed-loop
driver only admits a new request as an old one finishes — vLLM sees the
same arrival pattern. Remaining scheduler-side lever worth one experiment:
admitting *prefilled-and-waiting* requests into the running cohort the
moment decode begins rather than at cohort rebuild; expect small gains
(<10%) on this workload, larger on open-loop arrivals.

## 4) Ranked plan (status: Sep 6 2026)

1. **CUDA-graph the safetensors decode step — DONE.** The batched decode
   path already had full capture scaffolding; the safetensors path was
   excluded by `DecodeGraphCaptureSafe` (quantized weights required)
   plus per-projection capture aborts at every cuBLAS fallback. Changes:
   cuBLAS projection fallbacks are now capture-safe when the cuBLAS
   workspace is pinned (`HasPinnedWorkspace()`, already 4 MB via the
   executor), so the aborts only fire when unsafe; `GemmTypedLt`'s
   heuristic is cached per shape so no host-side query runs under
   capture; and the single-slot graph (which would thrash destroy+
   recapture on EOS-staggered width changes) became a per-width LRU set
   (cap 4). Verified: graphs capture for safetensors (B=1..7, ~509-581
   nodes), greedy outputs bit-identical with graphs on vs
   `INFERFLUX_DISABLE_CUDA_GRAPH=1` (both safetensors and GGUF paths),
   8-way concurrent identical prompts -> 1 distinct output, 32/32
   success. End-to-end (c=16 profiled, 2 runs): 266.8/286.2 ->
   331.3/316.6 tok/s (**+17.2% avg**), whole decode workload executed as
   ~200 `cudaGraphLaunch` calls; kernel time itself also dropped
   (cutlass 3490 -> 2860 ms) — capture-time algo selection beats
   live-heuristic picks under launch pressure.
2. **cublasLt algo search — DONE (scoped to lm_head).** Experiment result:
   Lt's first heuristic is neutral (~±1%) vs `cublasGemmEx` on the
   FFN/QKV/O shapes, but **1.6-2.1x faster on the lm_head shape
   (N=vocab)**, where `cublasGemmEx`'s pick is both slow and unstable
   cold-L2 (3.0-4.2 ms vs a consistent ~2.0 ms). Landed as
   `CublasGemm::GemmTypedLt` (cached per-shape heuristic, GemmTyped
   fallback), routed for the vocab projection (single + batched).
   End-to-end after review fix: 252.6 -> 266.8/286.2 tok/s profiled
   (+5.6%/+13.3%, avg ~+9.5%), GPU busy 7.18 -> 6.78 s, 32/32 success,
   output coherent. **Falsified for other shapes** — do not bother
   routing them.
   **Review lesson (adversarial round 1):** the first version of
   `GemmTypedLt` created its matmul descriptor with the matrix dtype
   (bf16) as `scaleType` while passing `float*` alpha/beta — cublasLt
   requires `CUDA_R_32F` scale type for `CUBLAS_COMPUTE_32F`, so the
   heuristic query returned zero results and every call silently
   permanently fell back to `GemmTyped` (a no-op shipped as an
   optimization; the spike had it right, the production wiring dropped
   it). The reviewer caught it by reproducing the descriptor setup
   standalone at the real lm_head shape. Fix: scaleType = `CUDA_R_32F`,
   plus a one-time warn when a shape's heuristic fails so this class of
   silent fallback can't recur unnoticed. Measurement honesty note: the
   pre-fix "+7.2%" was run-to-run variance, not the code — the no-op
   build measured 270.9/269.9 vs pre-change 252.6, which is why the
   post-fix claim is re-measured from scratch.
3. **Fuse gate+up into one [2N, K] GEMM** — **FALSIFIED**: cold-L2 spike
   shows fused [22016, 2048] is 1.01-1.02x two [11008, 2048] calls
   (noise). Two back-to-back GEMMs are already fine; skip.
4. **Specialist bf16 decode kernel** (the 2x-vs-physics prize): next
   after 1, as a real project — large row-tiles per block (64-128 rows),
   cp.async double-buffered K-stream, tensor-core-free FMUL pipeline sized
   to 48 SMs, dispatch rule M<=8. Spike tool already provides the honest
   cold-L2 benchmark harness to iterate against.
5. **Re-run the two-stage benchmark after each landing** (2x runs, per the
   variance protocol) and update `docs/benchmarks.md`.

## 5) Measurement protocol (keep using it)

- nsys captures without env instrumentation; treat
  `INFERFLUX_CUDA_PHASE_TIMING=1` as attribution-only (it halves throughput).
- Kernel roofline claims only under cold L2 (use the spike's eviction
  pattern); warm-L2 microbenchmarks overstate small shapes >100%.
- Any throughput comparison: 2 runs minimum; the harness shows up to ~30%
  run-to-run variance on llama_cpp_cuda.
