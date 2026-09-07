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
4. **Specialist bf16 decode kernel — FALSIFIED at reachable effort
   (Sep 6).** The deep-MLP design (warp-per-row, K-loop unrolled x8 with
   per-iteration partial accumulators breaking the load->FMA dependency
   chain, stride-32 tail) is now numerically correct (d=0.000 vs cuBLAS on
   all 5 shapes x M<=4) and lands at **38-39% of DRAM roofline cold-L2 —
   the same wall cuBLAS hits (38-45%)**. Two kernel bugs were found and
   fixed en route (stride-1 tail overlapping lane ranges; x offsets not
   tracking the unrolled weight offsets), plus a harness bug (variants
   clobbered each other's output buffers and the reference). Conclusion:
   ~40% of the 576 GB/s spec number is the practical streaming wall for
   this access pattern on this part under cold L2 — the "2x vs physics"
   framing was against an unreachable ceiling. Remaining (heavyweight,
   uncertain payoff): TMA/cp.async.bulk pipelined tiles or wgmma-based
   paths; revisit only with vendor-grade kernel-engineering effort. The
   corrected spike (`tests/tools/bf16_gemv_bench.cu`) is the reference
   harness for any future attempt.
5. **Re-run the two-stage benchmark after each landing** (2x runs, per the
   variance protocol) and update `docs/benchmarks.md`.

## 4b) Width-tail quantification (Sep 6, definitive): the last big lever

With the phase-timing width logging fixed (`BatchForwardDevice` had
hardcoded `tokens=1` in its report -- every device-path decode forward
logged width=1, poisoning earlier width analyses), a 2x32-request c=16
capture gives the definitive decode-width distribution over 703 sampled
forwards:

- **width=1: 266 forwards (38%)** -- solo decode during wave tails
- width=2-14: ~227 (32%) -- EOS-staggered drain/refill transitions
- width=15-21: 185 (26%) -- full cohorts (20-21 = mixed decode+prefill)
- Mean width 7.7; prefill compute is only ~1% of logged time (prefill
  is NOT the stall); mixed decode+prefill batches already occur (9%).

Mechanism: with EOS enabled, sequences finish at varied lengths (7-64
tokens), so each client wave's cohort drains over a long tail where the
last survivors decode solo -- and newly admitted requests wait behind a
prefill/refill boundary instead of joining the running cohort. Decode
is memory-bound: a width-1 step costs nearly as much GPU time as a
width-16 step, so every solo step is ~15 tokens of foregone throughput.

**Next scheduler lever (moderate effort, +25-40% potential at c=16):**
admit/overlap waiting prefills during tail stretches -- e.g., proactively
prefill newly arrived requests into the running cohort (mixed batches)
instead of at cohort-rebuild boundaries, and cap solo-cohort drain by
refilling from pending_decode_ mid-tick. Measurable target: width=1
fraction from 38% to <10% on the same load.

**Refined with the fixed logging (chronological run-length analysis):**
the width sequence is NOT a smooth drain-ramp. It alternates between two
modes: (a) "solo rounds" -- a single-sequence prefill forward (~20-token
prompt) followed by ~63 consecutive width=1 decode forwards (~2.1 s),
i.e. the executor presents B=1 to the burst path while it cycles the
cohort sequence-by-sequence through its burst budget; and (b) "cohort
mode" -- steady width=15-16 stretches of 13-40 forwards with natural
drain ramps. Batched execution (mode b) demonstrably works, so the open
question is precisely why the executor presents B=1 during mode (a)
(candidate: decode-group assembly/chunking during mixed prefill+decode
phases; `decode_batch_capacity = kv_cache_->MaxBatchSize()` interaction
with auto-tune is not yet ruled out). The fix must live in the shared
executor/scheduler layer (both GGUF-quantized and safetensors serving
run through the same `ExecuteUnifiedBatchStep`/burst machinery), so a
single fix covers both model formats. First step of that session: log
the decode group size at burst invocation and the capacity, under INFO,
in one instrumented run.

## 5) Open follow-up: the decode relay fingerprint is provably inert (and a naive fix was falsified)

The executor arms a per-step device relay after each decode step (sampled
tokens + n_past+1 written into device metadata by `DeviceTokenRelay`, plus
an identity fingerprint) so the next step can replay the just-captured
graph without the H2D metadata upload. Review instrumentation showed **0
relay matches across ~2k decode steps**: the arm stores the *fed* tokens
and a `+1`-offset n_past, but the check compares against the *sampled*
tokens the next step feeds and the already-advanced host n_past — both can
never hold, so the primary-path relay has been dead code since it landed
(only the burst path's closed device loop relays).

A naive fix (arm stores `sampled_tokens`; compare `armed.next_n_past ==
presented_n_past`) was first attempted and **reverted**: the
`inferflux_batched_isolation_probe` reported 16 divergent sequences with
the relay live -- but the identical count occurs with graphs disabled,
i.e. the probe cannot distinguish relay corruption from the pre-existing
bf16 batch-vs-single numerics divergence, so the probe cannot validate
the relay either way (it counts numerics divergence, not corruption).

**Unblocked and landed (Sep 6)** with a validation signal that separates
the two: a relay kill switch (`INFERFLUX_DISABLE_DECODE_RELAY=1`) enables
a bit-identical ON/OFF comparison through identical batch compositions,
where any output difference is attributable to the relay alone (the
kernel, buffers, and graph are unchanged). The fix was re-applied
(arm stores `sampled_tokens`; the check compares `armed.next_n_past ==
presented_n_past` directly, matching DeviceTokenRelayKernel's actual
device-side writes line-for-line), the fingerprint unit tests were
rewritten for the corrected contract, and a
`inferflux_scheduler_decode_relay_replays_total` counter proves
engagement (404 replays on a 2x32-request safetensors load, 275 on GGUF).
Validation: relay ON vs OFF **bit-identical** across 32 concurrent
requests spanning 3 distinct batch-width contexts; GGUF path
deterministic and matching the pre-fix output set; 525/525 unit tests.
Throughput: neutral-to-marginal on this driver (the skipped H2D upload
is microseconds; most of the relay's saving overlaps host work that was
already hidden by the graphs), so this lands primarily as a
correctness/observability fix that re-activates dead code.

## 5) Measurement protocol (keep using it)

- nsys captures without env instrumentation; treat
  `INFERFLUX_CUDA_PHASE_TIMING=1` as attribution-only (it halves throughput).
- Kernel roofline claims only under cold L2 (use the spike's eviction
  pattern); warm-L2 microbenchmarks overstate small shapes >100%.
- Any throughput comparison: 2 runs minimum; the harness shows up to ~30%
  run-to-run variance on llama_cpp_cuda.
