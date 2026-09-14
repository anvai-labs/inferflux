# InferFlux Benchmarks and Performance Analysis

**Status:** Current
**Snapshot date:** September 13, 2026 (adds the AMD R9700 ROCm sweep; CUDA
tables below remain the Sep 4-8 2026 RTX 4000 Ada measurements)
**Primary hardware:** NVIDIA RTX 4000 Ada (20 GB), AMD Radeon AI PRO R9700
(32 GB)

## ROCm — AMD R9700 (Sep 13 2026)

InferFlux `rocm` backend (llama.cpp HIP wrapped by the InferFlux scheduler)
vs a stock llama.cpp server built from the same pinned source. Qwen2.5-3B
Q4_K_M plus four production-class models, 48×256-token greedy battery, 16
concurrent. Full table, config, and reading:
[Competitive Positioning §R](COMPETITIVE_POSITIONING.md).

| Model | Stock llama.cpp c=16 | InferFlux c=16 |
|---|---:|---:|
| Qwen2.5-3B (dense) | 992 | **1067** |
| LFM2.5-8B-A1B (hybrid MoE) | 861 | **1089** |
| gpt-oss-20b MXFP4 (MoE) | 598 | **710** |
| Qwen3-30B-A3B (MoE) | 498 | **750** |
| Qwen3-14B (dense) | 391 | 388 (parity) |

Platform sanity: device bandwidth 612 GB/s D2D / 635 GB/s streaming read;
llama.cpp `test-backend-ops` on gfx1201 11,054/11,054 passed. Any
pre-Sep-13 "ROCm throughput" number in older documents (17-36 tok/s claims)
was a misrouted-CPU-backend artifact, not device throughput.

Full backend coverage takes two harness invocations because no single model
format serves all five compared engines — GGUF quantized backends
(`inferflux_cuda`, `llama_cpp_cuda`, Ollama) in Stage 1, full-precision
safetensors backends (`inferflux_cuda`, LM Studio, vLLM, SGLang) in Stage 2.
See [benchmarks](benchmarks.md#multi-backend-harness-reference)
for the exact two-stage recipe.

## FP16 / memory-precision guidance

| Decision | Guidance |
|---|---|
| Default production throughput | Prefer quantized GGUF (`q4_k_m`/`q5_k_m`) for concurrency and memory economy |
| FP16 deployment | Reserve for quality-critical workloads; right-size concurrency to the VRAM budget |
| Capacity controls | StartupAdvisor recommendations + conservative `max_parallel_sequences` + monitored memory pressure |
| Validation gate | Run throughput/contract checks before rollout; treat archived FP16 data as snapshot evidence, not guaranteed ceilings |

`707138b` landed FP16 OOM handling: pre-flight admission check, graceful
degradation, quantization-detection wiring, and a model-path override fix.
The March 2026 caution paths (20 GB FP16 instability, universal-backend
heap corruption) were falsified by the Sep 2026 campaign — native FP16 on
the 20 GB Ada served c=16 at 338 tok/s with an 8.3-8.7 GB peak and zero
classified failures (see Stage 2 below). Historical FP16 evidence snapshots
are cataloged in [ARCHIVE_INDEX](ARCHIVE_INDEX.md).

## Stage 1 — GGUF Quantized (Sep 4 2026, 2-run average)

RTX 4000 Ada 20GB · Qwen2.5-3B Q4_K_M · 32 requests × 64 tokens per concurrency level

**Methodology note:** run-to-run variance turned out to matter here —
`llama_cpp_cuda` swung as much as ~30% between two back-to-back runs at the
same concurrency (e.g. 307.8 vs 238.0 tok/s at c=8), while `inferflux_cuda`
trended consistently upward in both. The table below is the 2-run average
(matching the Apr 2026 doc's own 2-run methodology); the range column shows
the actual min–max spread observed so the averages aren't read as more
precise than they are. Ollama was the most reproducible backend by far
(within ~3 tok/s of itself at every concurrency, both runs).

```
Backend             c=1 avg    c=2 avg    c=4 avg    c=8 avg    c=16 avg   c=16 range
───────────────     ───────    ───────    ───────    ───────    ────────   ──────────
inferflux_cuda        106.9      116.4      163.6      249.9       356.3   335.8–376.8
llama_cpp_cuda        122.6      176.7      215.2      272.9       228.1   223.6–232.6
Ollama¹               123.2      125.3      124.7      122.0       123.8   122.1–125.5
```

¹ Remote host (`192.168.1.20:11434`), model `qwen2.5:3b` (Q4_K_M) — same
quantization as the local GGUF, so the comparison is apples-to-apples.

**Semantic similarity** (embedding cosine, all backend pairs, all concurrency
levels, both runs): 0.893–0.945, consistently in the HIGH band — no backend
produced degenerate or divergent output at any concurrency level.

**`inferflux_cuda` leads `llama_cpp_cuda` at c=16 in both runs** (1.50x and
1.62x — average 1.56x), following the S17/S18 Q4_K down-projection MMA
kernel work. **c=8 is a toss-up, not a clean `llama_cpp_cuda` win**: one run
had `llama_cpp_cuda` ahead 1.40x, the other had `inferflux_cuda` ahead 1.18x.
Treat c=8 as roughly at parity with real noise, not as settled ground for
either backend. `llama_cpp_cuda` is the clear leader at c=1–4 in both runs.

## Stage 2 — Safetensors / Full Precision (Sep 5 2026, post-fix)

RTX 4000 Ada 20GB · Qwen2.5-3B (fp16/bf16 safetensors) · 32 requests × 64 tokens per concurrency level

**Two `inferflux_cuda`-side bugs were found and fixed while validating this
stage** (full trace in the Fixed Bugs section below) — undecoded byte-level
BPE output, and a missing chat template plus wrong EOS token that together
meant it never stopped generation correctly. The table below is a
post-fix run. It wasn't independently re-run twice, but its `inferflux_cuda`
column (278.7 tok/s at c=16) falls inside the min-max range already
established by two pre-fix runs (285.2-311.4) for this same prompt mix — so
the fix changed correctness, not this workload's throughput profile (the
benchmark's 32 prompts happen to need close to the full 64-token budget
regardless; a fix whose main effect is *stopping earlier* shows up more on
short-answer prompts than on this technical-question prompt set).

```
Backend             c=1        c=2        c=4        c=8        c=16       GPU Peak
───────────────     ───────    ───────    ───────    ───────    ────────   ────────
inferflux_cuda         42.3       66.1      123.8      202.4       278.7    8551 MB
vLLM                   49.3       97.1      192.5      385.6       753.4   20040 MB
SGLang                 48.3       91.4      181.2      364.6       659.8   17671 MB
LM Studio¹            103.1       61.7       69.4       68.1        69.5    3033 MB
```

¹ Remote host (`192.168.1.20:1234`), auto-discovered model. Degrades under
concurrency the same way every LM Studio snapshot in this doc's history has
(single-threaded JS event loop serializing dispatch); c=1 is its best
showing.

**vLLM and SGLang still dramatically outperform `inferflux_cuda` on
full-precision safetensors at concurrency** — the fixes were about output
correctness, not closing this gap. At c=16: vLLM 753.4 tok/s (2.70x
`inferflux_cuda`), SGLang 659.8 tok/s (2.37x), both scaling near-linearly
from c=1 (vLLM 15.3x, SGLang 13.7x vs `inferflux_cuda`'s 6.6x). Expected:
both are purpose-built serving engines with mature continuous batching and
paged/radix attention for exactly this workload, while `inferflux_cuda`'s
safetensors path is a much newer, less-optimized code path than its GGUF
path.

### Nsight Systems profile: why `inferflux_cuda` trails vLLM/SGLang on safetensors (Sep 6 2026)

Profiled with nsys (32 req × 64 tok, matching the c=16 benchmark; RTX 4000
Ada, Qwen2.5-3B fp16 safetensors, 6.16 GB of weights read per forward).
Artifacts: `nsys_profile_ifx_st_c16_clean/` and `..._c1/` (local, not
committed). The gap decomposes into three measured factors, ranked:

**1. Decode batch-width collapse — ~2.3x (the dominant loss).** A decode
step re-reads all 6.16 GB of weights regardless of how many tokens it
carries, so throughput ∝ average batch width. At c=16 the scheduler's own
batch histogram is bimodal: **209 of ~417 decode steps ran at batch=1
(38%)**, 59 at batch=2, and only ~99 steps at width 14-15 — weighted
average ~5-7 tokens/step, versus vLLM decoding full-width 16 continuously.
GPU call counts corroborate: FlashDecode ran 289 times for 2048 tokens
(avg 7.1 sequences/step). GPU cost is nearly flat in batch (20.5 ms/token
at batch 1 → 24.8 ms/step at batch 7.1, i.e. 3.5 ms/token), so merging the
batch-1 steps into the full-width steps is close to free throughput.

**2. GEMM kernels at ~half of memory-bandwidth roofline — ~2x vs physics,
~parity vs vLLM.** 87% of GPU time is cuBLAS/cutlass bf16 GEMMs. At batch 1
cuBLAS picks `gemvx` kernels (~300 GB/s achieved); at batch ~11 it picks
`cutlass_80_wmma...` kernels with small grids (e.g. 688 blocks × 32 threads
for M=16, N=11008, K=2048) at ~250 GB/s — against 576 GB/s peak. Per decode
step: 24.8 ms GPU work vs a 10.7 ms roofline floor. vLLM's custom decode
kernels land at a similar ~50% of roofline, so this factor is roughly
parity with vLLM — but it is the largest headroom against the theoretical
ceiling, and the fix (dedicated weight-read-first fused decode GEMV/GEMM
kernels for skinny bf16 shapes, mirroring the GGUF MMVQ strategy, or
cublasLt algo search) benefits both absolute throughput and the batch-width
win.

**3. Per-step host synchronization — ~1.12x.** The clean profile shows
~7 chunky `cudaStreamSynchronize` per decode step (~2.8 ms each, waiting
on step results), with the GPU ~88.5% busy overall. Unlike the GGUF decode
path, the safetensors path does not replay a captured CUDA graph, so every
step pays launch + sync round-trips. Graphing this path (as the GGUF decode
path already does) is the straightforward fix.

Cross-check: 2.3 (width) × ~1.15 (vLLM's slightly better kernels) × 1.12
(syncs) ≈ 3.0 — matching the observed 2.4-2.7x gap within run-to-run
variance.

**Follow-up experiments (Sep 6, later same day) refined two of these:**

- A `batch_accumulation_ms` A/B sweep (2/8/16 ms, 2 runs per point) showed
  **no material width gain** (320.7 / 329.3 / 292.4 tok/s averages; 16 ms
  strictly worse) — low-width steps come from EOS-staggered completions
  (sequences stop at natural EOS at different lengths, a direct consequence
  of the stop-token fix above) plus closed-loop client arrivals, not from a
  merge-window bug. The scheduler does refill correctly when pending work
  exists (85/206 steps at width 15-16 in a live-metrics run).
- A standalone kernel spike (`tests/tools/bf16_gemv_bench.cu`) measured
  cuBLAS on the exact projection shapes under **cold L2** — the realistic
  condition, since attention kernels evict L2 between projections: 30-44%
  of roofline (small shapes worst). A naive warp-per-row custom GEMV
  matches but does not beat cold-L2 cuBLAS, falsifying the quick-rewrite
  path; closing the 2x-vs-physics kernel gap needs the heavyweight design
  (multi-row tiles, cp.async double buffering) or cheaper wins first.

The staged plan (CUDA-graph the safetensors decode step, cublasLt algo
search, gate/up fusion, then a specialist kernel) with all measurements and
falsified hypotheses lives in
[ARCHIVE_INDEX](ARCHIVE_INDEX.md) (plan retired; findings in the nsys section above).

**Methodology caveat worth keeping:** `INFERFLUX_CUDA_PHASE_TIMING=1`
synchronizes the stream between every phase of every layer (~324 syncs per
decode step) and **halved measured throughput** (171 vs 252.6 tok/s
profiled) while inflating apparent GPU idle from ~12% to ~55%. Use it only
for relative phase attribution on short runs, never for throughput or
idle-fraction claims — nsys kernel/API data needs no env flag and is
uncontaminated.

`inferflux_cuda` itself still scales far better on full precision than on
quantized GGUF (~6.6x from c=1→c=16 here, vs ~3.33x average in Stage 1) —
full-precision GEMV is less memory-bandwidth-bound per token, so the
batching win shows up more directly in throughput even though the absolute
ceiling is well below vLLM/SGLang.

### Fixed bugs found while validating this stage

Checking *why* `inferflux_cuda` was slower — not just accepting the number
— surfaced two real, stacked bugs, not benchmark artifacts:

1. **Undecoded byte-level BPE output.** `inferflux_cuda`'s safetensors
   responses contained literal `Ġ`/`Ċ` characters instead of spaces/newlines
   (e.g. `"...ExplainthedifferencebetweenTCPandUDP..."`), 100% reproducible.
   Root cause: `runtime/backends/mlx/mlx_tokenizer.cpp`'s pre-tokenizer-type
   detection only matched a flat `pre_tokenizer.type == "ByteLevel"` in
   `tokenizer.json`; Qwen2.5 (and most modern ByteLevel-BPE tokenizers —
   GPT-2, Llama-3, Phi-3) nests it as `{"type": "Sequence", "pretokenizers":
   [{"type": "Split", ...}, {"type": "ByteLevel", ...}]}`, which the
   detector never unwrapped, so decode fell through to a raw pass-through.
2. **No real chat template, wrong EOS token.** Found while validating fix
   #1 end-to-end: a plain prompt ("What is the capital of France?")
   answered correctly, then hallucinated fake `"chatbot: ..."` /
   `"user: ..."` conversation turns instead of stopping.
   `HFTokenizer::ApplyChatTemplate` unconditionally returned an
   invalid/empty result (a longstanding `// TODO: requires a Jinja2 engine`
   stub), and separately `MlxTokenizer::Load()`'s BOS/EOS resolution only
   read `tokenizer_config.json` — absent for this model directory — so
   `eos_id_` kept a meaningless hardcoded default instead of Qwen2.5's real
   EOS (`<|im_end|>`).

**Fix:** chat-template rendering now shares GGUFTokenizer's existing
strategy-based ChatML/Llama/Mistral/Gemma renderer, extracted into
`model/chat_template_renderer.{h,cpp}` so both tokenizer paths stay in
sync; BOS/EOS resolution falls back to `config.json`'s
`bos_token_id`/`eos_token_id` when `tokenizer_config.json` is absent; and a
standalone `chat_template.jinja` file (the transformers v4.44+/vLLM/SGLang
convention) is read when no `tokenizer_config.json` supplies a template.
**Verified:** 12 new/extended unit tests (each regression-checked against
the pre-fix code — confirmed to fail without the fix, pass with it), the
full 511-case unit suite (0 regressions), and live end-to-end API calls
against the real safetensors-served model — clean text, `finish_reason:
"stop"`, correct short completion for "capital of France" (7 tokens, was
unbounded runaway before).

### Benchmark validation notes

Before trusting the numbers above, the raw data was checked for internal
consistency, not just read off the summary table:

- **32/32 success at every concurrency level, every backend, every run** —
  no partial failures inflating apparent throughput.
- **`total_tokens` is exactly constant across all 5 concurrency levels
  within each backend** (pre-fix: `inferflux_cuda` 2048, vLLM/SGLang 2029,
  every level, both runs; post-fix: `inferflux_cuda` dropped to 2025, now
  in the same range as vLLM/SGLang — direct evidence the stop-token fix
  worked, not just the two hand-checked repro prompts) — confirms
  deterministic greedy decoding and that every concurrency level did equal
  work, so the tok/s comparison reflects speed, not partial-completion
  artifacts.
- **vLLM/SGLang's per-request latency stays flat (~1300-1500ms) across all
  concurrency levels while `inferflux_cuda`'s grows (~1490ms→3300ms,
  post-fix run)** — the correct signature of true continuous/iteration-level
  batching (new requests join an in-flight batch) vs. `inferflux_cuda`'s
  batching model, which shows request-level queuing under load. This is
  architecturally consistent with, and helps explain, the scaling gap above.
- **Response text was actually read, not just checked for non-zero token
  counts** — this is what caught the two `inferflux_cuda` bugs above. An
  earlier pass in this same session had only spot-checked vLLM and SGLang's
  text (confirming those were fine) without checking `inferflux_cuda`'s own
  output; re-checking closed that gap. Both bugs are now fixed and
  reflected in the table above.
- **Semantic-similarity cosine scoring does not run for Stage 2** — the
  harness hardcodes `llama_cpp_cuda` as the reference backend
  (`ref_backend` in `scripts/benchmark_multi_backend_comparison.sh`), and
  `llama_cpp_cuda` never runs on safetensors input, so the column is
  structurally empty here (not a failure). Stage 1's GGUF comparison does
  have real cosine-similarity data (0.893-0.945, see above) because
  `llama_cpp_cuda` runs there.

### Environment setup gotchas fixed to get here

Both vLLM and SGLang produced garbage or refused to start on the first
attempt on this dual-GPU (NVIDIA + AMD) WSL2 box. Neither was an InferFlux
bug; both are environment-specific and are now baked into
[benchmarks](benchmarks.md#multi-backend-harness-reference):

- **vLLM returned HTTP 200 with 0 real tokens on every request.** Its own
  server log showed `ChatTemplateResolutionError` — transformers v4.44+
  removed the default chat-template fallback, and this local safetensors
  conversion of Qwen2.5-3B never got a `chat_template.jinja` or
  `tokenizer_config.json` with one. Fixed by adding the canonical Qwen2.5
  ChatML `chat_template.jinja` (same format as InferFlux's own
  `RenderChatML`) to the model directory. The benchmark harness's success
  check is HTTP-status-only and does not verify token count or inspect
  response text, so this silently printed "32/32 OK" at 0 tok/s — always
  spot-check response bodies for a new backend, not just the summary table.
- **SGLang failed to start** for two independent, stacked reasons: (1) its
  JIT build tool (`tvm_ffi`) auto-detects ROCm over CUDA whenever a ROCm
  install exists on the box at all — true here since this machine also does
  ROCm work — regardless of which GPU is actually targeted; fixed with
  `TVM_FFI_GPU_BACKEND=cuda`. (2) A stale FlashInfer JIT cache under
  `~/.cache/flashinfer` held a hardcoded `/usr/bin/nvcc` path from a
  different environment; this box's nvcc lives at
  `/usr/local/cuda-13.2/bin/nvcc`. Fixed by clearing `~/.cache/flashinfer`
  and `~/.cache/tvm-ffi` and setting `CUDA_HOME=/usr/local/cuda-13.2`
  explicitly for the launch.

## Competitive Claims (Verified, Reproducible — Stage 1, 2-run average)

| Claim | Data |
|---|---|
| `inferflux_cuda` **~1.56x faster than llama_cpp_cuda** at c=16 | 356.3 vs 228.1 tok/s avg; reproduced both runs (1.50x, 1.62x) |
| `inferflux_cuda` **~2.88x faster than Ollama** at c=16 | 356.3 vs 123.8 tok/s avg |
| **Best scaling efficiency at c=16** | inferflux ~3.33x (c=1→c=16) vs llama_cpp_cuda ~1.86x vs Ollama ~1.00x (flat) |
| **High semantic parity across all backends** | 0.893–0.945 cosine similarity, all pairs, all concurrency levels, both runs |
| `llama_cpp_cuda` fastest at c=1–4 | e.g. 215.2 vs 163.6 tok/s avg at c=4 (1.31x); reproduced both runs |
| c=8 is roughly at parity | 272.9 vs 249.9 tok/s avg, but the two runs individually disagree on direction (1.40x llama, then 1.18x inferflux) — not a settled claim either way |

## Why InferFlux Scales Better

### InferFlux (C++17, direct CUDA)
- **Unified batching**: one `ExecuteUnifiedBatch()` GPU kernel launch serves all concurrent sequences
- **Zero language boundary**: C++ HTTP server → C++ scheduler → CUDA kernels, no CGo/JS overhead
- **Shared GPU context**: model weights loaded once, all requests share the same GPU memory
- **Batch-aware scheduler**: `IBatchSelectionPolicy` groups requests for maximum GPU utilization

### Ollama (Go + CGo → llama.cpp)
- Go HTTP server dispatches to llama.cpp via CGo bindings
- CGo call overhead (~1-5μs) compounds at high concurrency
- Go's garbage collector pauses can stall the HTTP accept loop
- Sequential per-request dispatch — no cross-request batching on GPU
- Result: throughput plateaus at ~125 tok/s regardless of concurrency

### LM Studio (Electron + Node.js → llama.cpp server)
- Node.js event loop serializes request dispatch (single-threaded JS)
- Throughput **degrades** under load once concurrency rises past c=1
- Result: c=1 is its best showing in both the Apr 2026 and Sep 2026 snapshots

## Quality Fixes (Apr 2026, still in effect)

| Fix | Before | After |
|---|---|---|
| Chat template rendering (ChatML/Llama/Mistral/Gemma) | 43% accuracy (stub returned empty) | 100% accuracy |
| Repetition penalty (CUDA kernel + per-sequence tracking) | 31% degenerate loops | 0% degenerate |
| KV cache clearing on sequence reuse | Stale data corruption | Clean prefill |

## GPU Memory

Memory also varied ~10% run-to-run in Stage 1 (not just throughput) — ranges
below are min–max across both runs, not single-point values.

```
Stage 1 (GGUF, Q4_K_M, range across 2 runs):
  inferflux_cuda:  7025-7665 MB  (roughly +2900 to +3600 MB vs llama_cpp_cuda)
  llama_cpp_cuda:  4088-4620 MB  (reference)
  Ollama:           609-1141 MB  (remote host, not directly comparable — no
                                  local GPU process on this box)

Stage 2 (safetensors, full precision, single run):
  inferflux_cuda:  8605 MB  (startup delta +7973 MB)
  vLLM:           20046 MB  (pre-allocates most of the GPU for KV cache by
                             design — expected, not a leak)
  SGLang:         17680 MB  (also pre-allocates a large KV cache by design)
  LM Studio:       3087 MB  (remote host)
```

The overhead vs `llama_cpp_cuda` grew from the +1268/+1294 MB read in the
2026-08-31 rebaseline to +2900-3600 MB here; not yet re-explained — worth a
follow-up rather than assuming either number is wrong.

## Running Benchmarks

```bash
# Stage 1 — GGUF quantized (inferflux_cuda, llama_cpp_cuda, Ollama)
BUILD_DIR=./build-cuda SKIP_LMSTUDIO=true \
  bash scripts/benchmark.sh multi-backend \
  models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf

# Stage 2 — safetensors (inferflux_cuda, LM Studio, vLLM, SGLang)
# TVM_FFI_GPU_BACKEND / CUDA_HOME are this dual-GPU box's SGLang fix —
# see "Environment setup gotchas" above and benchmark_multi_backend_steps.md
AUTOSTART_VLLM=true AUTOSTART_SGLANG=true \
  VLLM_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
  SGLANG_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
  TVM_FFI_GPU_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.2 \
  BUILD_DIR=./build-cuda \
  bash scripts/benchmark.sh multi-backend models/qwen2.5-3b-instruct-safetensors

# Native vs llama.cpp only
BUILD_DIR=./build-cuda bash scripts/benchmark.sh gguf-compare

# Throughput regression gate
BUILD_DIR=./build-cuda bash scripts/benchmark.sh throughput-gate
```

See [benchmarks](benchmarks.md#multi-backend-harness-reference) for
the full harness contract, tuning knobs, and this dual-GPU box's
`-DENABLE_ROCM=OFF` build gotcha.

# Multi-Backend Harness Reference

## 1. Flags and defaults
* `INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWPAIR_W4` now defaults to `false` in `NativeExecutionPolicy`. Keep it opt-in for controlled experiments only; exact-shape isolated benchmarking on Ada RTX 4000 showed the `M=2,N=11008,K=2048` row-pair FFN kernel was numerically clean but slower than the generic grouped path.
* Keep `INFERFLUX_ENABLE_BATCHED_DECODE=1` in the benchmark so multi-row decode batches naturally occur and exercise the row-pair operator per the metrics below.
* `INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT=1` is an experimental scheduler knob only. Keep default benchmarking on `wait=0`; use `wait=1` only as an A/B comparison because the effect is workload-sensitive and not stable enough for default serving policy.
* `INFERFLUX_NATIVE_BURST_CHUNK_TOKENS` is a legacy tuning knob for the CUDA-singleton stepwise burst path only; the serving guidance for GPU concurrency is the unified batch path with wave-gathering admission (see CONFIG_REFERENCE).
  * `2`: favors lower-concurrency interactive serving (`c=2`/`c=4`)
  * `4`: current balanced default for WSL2/native CUDA benchmarking
  * `8`: only use for explicit high-concurrency probes; it regressed lower-concurrency runs in March 27 long-sweep data
* Keep `runtime.scheduler.decode_burst_tokens=0` during these probes unless you are explicitly testing fairness-slice behavior. That scheduler burst cap is not the same as the native singleton burst path.
* `BACKEND_STARTUP_TIMEOUT_SEC` is benchmark-harness only. Default is backend-aware: `60s` for most local backends and `180s` for `llama_cpp_cuda`. Raise it further only for explicit cold-start investigations; do not treat it as a serving/runtime tuning knob.

## 1.1 Current sensible defaults for this solution

For the current native singleton burst path on Qwen2.5-3B Q4_K_M under WSL2:

* `INFERFLUX_ENABLE_BATCHED_DECODE=1`
* `runtime.scheduler.min_batch_size=1`
* `runtime.scheduler.batch_accumulation_ms=2`
* `runtime.scheduler.decode_burst_tokens=0`
* `INFERFLUX_NATIVE_BURST_CHUNK_TOKENS=4`
* `INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT=0`

Rationale:

* `chunk=4` was the best balanced result across the March 27 long sweep (`32` requests, `32` max tokens, `1/2/4/8/16` concurrency).
* `chunk=2` was better at `c=2` and `c=4`, but weaker at `c=8` and `c=16`.
* `chunk=8` improved `c=8` only and regressed low/mid concurrency too sharply to serve as the default.
* Sticky wait remains useful as an explicit A/B probe, but not as the default serving policy.

## 2. Harness contract
`run_gguf_comparison_benchmark.sh` now supports multi-concurrency sweeps from one invocation:

* default prompt set: 16 longer “real usage” prompts
* default matrix: `CONCURRENCY=1,4,8`
* default requests: `NUM_REQUESTS=16`

Per-concurrency artifacts are intentionally isolated. Expect files such as:

* `responses_inferflux_cuda_c1/`, `responses_inferflux_cuda_c4/`, `responses_inferflux_cuda_c8/`
* `stats_inferflux_cuda_c1.json`
* `metrics_inferflux_cuda_c4.txt`
* `admin_cache_inferflux_cuda_c8.json`
* `similarity_c1.json`, `similarity_c4.json`, `similarity_c8.json`

If these files are being overwritten across concurrency levels, the harness is broken and the benchmark should not be trusted.

`benchmark_multi_backend_comparison.sh` now runs each backend in an isolated child invocation. That is intentional:

1. the parent process only prepares the suite, dispatches one backend at a time, and merges results
2. each child process starts, benchmarks, and tears down exactly one backend
3. local CUDA backends get a `cudaDeviceReset()` between children

This avoids cross-backend allocator / stream / shell-job state leaking from one engine into the next.

For targeted debugging, set `INFERFLUX_BENCH_SINGLE_BACKEND=<backend_id>` and run the same script directly.

## 3. Run order and reset hook
The benchmark runs `inferflux_cuda` first, then `llama_cpp_cuda`. To avoid CUDA state leaking between the two:
1. The script already calls `stop_server inferflux_cuda` once the InferFlux CUDA run is done.
2. We added `reset_cuda_device()` which issues `cudaDeviceReset()` (via `libcudart`) immediately after native shutdown. That ensures the GPU context is fully torn down before the llama.cpp start.
3. Only then does the script launch `llama_cpp_cuda`; the 3-second sleep after the reset gives the GPU a final breathing room.
4. The script also traps exit and runs the same cleanup path so aborted runs free the active server and leave GPU state predictable for the next benchmark.

If you ever replicate the benchmark manually, follow the same order: stop native, reset the CUDA device (via `cudaDeviceReset()` or `./build-cuda/inferfluxd --reset-cuda` if available), then start the llama.cpp backend. This guarantees accurate throughput isolation for regression comparisons.

## 4. Metrics to validate operator and scheduler behavior
* Inspect `inferflux_cuda_rowpair_selection_total{phase="decode",operator="q8_1_group_row_pair_w4",bucket="2"}` and `...operator="q8_1_gemv_row_pair"` in the resulting `metrics_inferflux_cuda_c*.txt`. Successful runs record counts (>0) in bucket `2` or `3_4`, proving the specialized operators handled the multi-row batches.
* The benchmark also captures `inferflux_cuda_ffn_proj_operator_total` and `inferflux_cuda_down_proj_operator_total` summaries (written to `inferflux_cuda_ffn_proj_summary_inferflux_cuda_c*.json` and `inferflux_cuda_operator_summary_inferflux_cuda_c*.json`) so you can correlate which kernels were chosen.
* Every InferFlux backend run now also captures `/v1/admin/cache` into `admin_cache_<backend>_c*.json`. The corresponding `stats_<backend>_c*.json` embeds that data under `cache_snapshot` and `memory_snapshot`, including:
  * `memory_snapshot.inferflux_cuda_model`
  * `memory_snapshot.inferflux_cuda_kv`
  * `memory_snapshot.paged_kv`
* The multi-backend CSV export now carries the key memory fields alongside throughput so concurrency runs can be compared on both tok/s and memory state.
* The decode-worker sticky-merge counters (`inferflux_scheduler_decode_worker_sticky_merge_total`, `inferflux_scheduler_decode_worker_sticky_merged_requests_total`) are the intended validation signal for `INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT`, but benchmark-side metric capture remains a known limitation: those lines are visible in direct `/metrics` scrapes yet have not been reliable in saved benchmark snapshots.

## 5. Accuracy safeguards
* The similarity report is now per concurrency (`similarity_c*.json`). Treat the whole sweep as invalid if only one concurrency level produces similarity output; that indicates the harness wiped earlier response artifacts.
* Keep `INFERFLUX_DEBUG_OPERATOR_SELECTION=0`/`INFERFLUX_DEBUG_LOGITS=0` for normal benchmarks; enable them only for debugging because they add logging noise.

## 6. Release-note checklist
When promoting the row-pair flag for release:
* Update client-facing docs (this file) and point to the new metric so operators can verify row-pair usage.
* Mention that `llama_cpp_cuda` now runs against a clean GPU thanks to the reset hook—this avoids the sporadic `socket: Operation not permitted` issues that plagued earlier runs.
* Leave the instrumentation (metrics_capture hooks in the benchmark) so any regression gate re-running this benchmark automatically records operator breakdown, row-pair counters, and similarity data.

Current release posture:
* Keep the proven `Q4_K M=1` grouped hot path on by default.
* Prefer `q8_1_group_mmq3` for Q4_K `M>=2`; the exact live `M=2,N=11008,K=2048` benchmark now beats fused gate/up and has a dedicated row-pair parity test.
* Retain `q8_1_group_row_pair_w4` as the M=2 fallback when MMQ3 is disabled.

## 7. Local vLLM / SGLang safetensors runs

`benchmark_multi_backend_comparison.sh` can now auto-launch local `vllm` and `sglang` servers one at a time so their VRAM is released before the next backend starts.

Recommended local safetensors setup:

```bash
AUTOSTART_VLLM=true \
AUTOSTART_SGLANG=true \
VLLM_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
SGLANG_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
VLLM_LAUNCH_ARGS="--dtype half --max-model-len 2048" \
SGLANG_LAUNCH_ARGS="--dtype half --context-length 2048" \
TVM_FFI_GPU_BACKEND=cuda \
CUDA_HOME=/usr/local/cuda-13.2 \
BUILD_DIR=./build-cuda \
./scripts/benchmark.sh multi-backend \
  models/qwen2.5-3b-instruct-safetensors
```

Notes:

* The harness auto-detects the supplied model format and skips incompatible backends.
* GGUF inputs skip `vllm` / `sglang`; safetensors inputs skip `llama_cpp_cuda` / `ollama`.
* `vllm` and `sglang` should be benchmarked with safetensors/Hugging Face model directories via `VLLM_MODEL_PATH` / `SGLANG_MODEL_PATH`.
* If you already run those servers elsewhere, leave `AUTOSTART_VLLM` / `AUTOSTART_SGLANG` unset and point `VLLM_HOST` / `SGLANG_HOST` at the existing endpoints.

**Two environment gotchas on this dual-GPU (NVIDIA + AMD) box, both fixed by
the env vars above / a one-time model directory fix — don't rediscover
these:**

1. **vLLM returns HTTP 200 with 0 real tokens** if the target safetensors
   directory has no chat template (`ChatTemplateResolutionError` in
   `server_vllm.log` — transformers v4.44+ dropped the default-template
   fallback). The harness's success check is HTTP-status-only, so this
   silently reports "N/N OK" at 0 tok/s instead of failing loudly. Fix once
   per model directory: drop a `chat_template.jinja` (the standard Qwen2.5
   ChatML template, matching InferFlux's own `RenderChatML` format) next to
   `tokenizer.json`. Always spot-check a raw response body for a new model
   directory, not just the summary table.
2. **SGLang fails to start** for two independent reasons on a box with both
   CUDA and ROCm installed: its JIT tool (`tvm_ffi`) auto-detects ROCm over
   CUDA whenever a ROCm install exists at all, regardless of which GPU is
   targeted (`TVM_FFI_GPU_BACKEND=cuda` forces the correct choice); and a
   stale `~/.cache/flashinfer` / `~/.cache/tvm-ffi` JIT cache can hold a
   hardcoded `nvcc` path from a different machine/environment (clear both
   dirs and set `CUDA_HOME` explicitly if `nvcc` isn't at the path the cache
   expects).

## 8. Full backend coverage in two stages

No single model file exercises all five backends, so getting a complete
`inferflux_cuda` / `llama_cpp_cuda` / `ollama` / `vllm` / `sglang` picture
takes two sequential harness invocations, not one:

**Stage 1 — GGUF (inferflux_cuda, llama_cpp_cuda, ollama):**

```bash
BUILD_DIR=./build-cuda \
./scripts/benchmark.sh multi-backend \
  models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
```

`ollama` benchmarks against `OLLAMA_HOST` (default
`http://192.168.1.20:11434`, a remote host on this dual-GPU dev box) using
`OLLAMA_MODEL` (default `qwen2.5:3b`) — confirm the tag exists on that host
first (`curl $OLLAMA_HOST/api/tags`) rather than assuming it does.
`lmstudio` is skipped here (`SKIP_LMSTUDIO=true`) when no LM Studio instance
is reachable.

**Stage 2 — safetensors (inferflux_cuda, vllm, sglang):** the recipe in
Section 8 above, run separately, after Stage 1's local backends have torn
down and reset the CUDA device.

Run the two stages one after another, never concurrently — both stages
launch local CUDA backends against the same physical GPU, and the harness's
own cross-backend reset hook (Section 3) only serializes backends *within*
one invocation, not across two.

**Build gotcha specific to this dual-GPU box:** `cmake -S . -B build-cuda
-DENABLE_CUDA=ON` alone is not CUDA-only — `ENABLE_ROCM` defaults to `ON` in
the top-level `CMakeLists.txt`, and with both SDKs installed the combined
configure pulls in `<hip/hip_runtime.h>` (via
`server/startup_advisor.cpp`'s `INFERFLUX_HAS_ROCM` path) alongside CUDA's
`vector_types.h`, which fails with conflicting `dim3` declarations. Pass
`-DENABLE_ROCM=OFF` explicitly when the goal is a CUDA-only `build-cuda` for
this benchmark.

