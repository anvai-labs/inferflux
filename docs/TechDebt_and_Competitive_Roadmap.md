# InferFlux Tech Debt and Competitive Roadmap

**Snapshot date:** September 7, 2026
**Current overall grade:** B+

```
Grade trajectory:  B- (Mar 31) → B+ (Apr 9) → B (Apr 14) → B (Apr 15)
                → B+ (Sep 7)

Sep advances:
  ✓ Safetensors tokenizer correctness: ByteLevel BPE decode, chat
    template, EOS resolution (was: garbled output + runaway generation)
  ✓ Decode CUDA graphs on the safetensors path (+17% c=16 profiled)
  ✓ cublasLt lm_head dispatch (+7%)
  ✓ Decode relay fingerprint fixed and engaging (was provably inert)
  ✓ Projection dispatch single-sourced via ProjectionCtx (both formats)
  ✓ Multi-backend validation extended to vLLM + SGLang
  ✓ All performance hypotheses tested; falsifications documented

Open: width-1 decode tail during closed-loop EOS stagger (workload-
  shaped; scheduler refill levers falsified), practical streaming
  ceiling ~40% of spec bandwidth (both InferFlux and cuBLAS), GGUF
  multi-variant decode numerics under concurrent load (benign, all
  outputs coherent; documented in the performance plan doc).
```

## 1) Dimension Grades

| Dimension | Grade | Evidence |
|---|---|---|
| Vision and product coherence | A- | Server-first, dual-CUDA strategy; native path now leads llama.cpp at high concurrency on full precision (c=16: 1.56x, 2 runs) |
| Capabilities | A- | Streaming, embeddings, logprobs, chat templates shared across formats, decode CUDA graphs + relay with kill switches, GGUF metadata API |
| Scalability and economy | B+ | Safetensors 3.16x scaling (c=1→c=16, 2-run avg); decode graphs remove launch storms; memory footprint ~2.3x less than vLLM's pre-allocation |
| Resource efficiency | A- | Memory-overhead campaign (Sep): GGUF overhead root-caused and fixed (-1.6 GB measured, #108/#109/#110); scratch aliasing, FlashDecode splits, KV budget tuning; known residuals tracked (#111-#113) |
| Design and implementation | A- | ProjectionCtx single-sources the projection dispatch chain for both model formats; falsification-driven perf campaign; 525 CPU tests |
| TDD and CI maturity | B+ | 525 CPU tests (2833 assertions), parity gates per refactor stage, isolation probe, engagement counters for perf changes |
| OSS release readiness | B+ | Canonical docs current (check_docs_contract enforced), 4-backend + vLLM/SGLang benchmark coverage, kill switches documented |

## 2) Competitive Benchmark

### Safetensors (full precision) — Sep 7 2026, RTX 4000 Ada, Qwen2.5-3B

2-run average per cell (multi-backend harness; 0 classified failures):

```
Backend        c=1    c=4    c=8    c=16   scale    GPU peak
─────────────  ────   ────   ────   ─────  ──────   ─────────
inferflux_cuda 47.7   143.8  206.2  338.2  ~7.1x    8.3-8.7 GB
vLLM           36.9   160.2  336.0  675.3  ~18.3x   ~20.1 GB
SGLang         38.1   156.9  307.0  515.3  ~13.5x   18.2-20.1 GB
LM Studio      115.8  75.2   73.0   71.5   ~0.6x    2.9-3.1 GB
```

- Gap to vLLM/SGLang narrowed to **1.52-2.00x at c=16** (was 2.37-2.70x on
  Sep 4-6), while serving at 2.3-2.4x less GPU memory. The post-#110 scratch
  right-sizing likely contributed to the inferflux gain (338 vs 298-311).
- SGLang on this host requires `TVM_FFI_GPU_BACKEND=cuda`: with the ROCm
  toolchain installed system-wide, its JIT misdetects HIP and fails to build
  kernels (`/usr/bin/hipcc` shadows the CUDA path).

### GGUF (Q4_K_M) — Sep 7 2026 (post memory campaign), RTX 4000 Ada

2-run average per cell (multi-backend harness; 0 classified failures;
gguf-compare harness confirms the memory and c=1/4/8 ordering):

```
Backend          c=1    c=4    c=8    c=16   scale   GPU peak
───────────────  ────   ────   ────   ─────  ─────   ─────────────
inferflux_cuda   103.4  163.8  265.8  332.7  3.22x   5.5 GB (ledger)
llama_cpp_cuda   119.8  198.7  284.0  231.4  1.93x   4.1 GB (same harness)
Ollama (local)   121.3  124.1  124.2  123.1  ~1.0x   ~1.0 GB
LM Studio        115.4  71.7   76.2   75.0   ~0.65x  ~2.9 GB
```

- `inferflux_cuda` leads llama.cpp at c=16 on the same harness (1.44x, both
  runs agree) after the memory campaign; llama.cpp stays ahead at c=1-4 and
  c=8 is contested (runs split around parity).
- GGUF memory overhead vs llama.cpp is now **+1,392 MB** on the identical
  workload (was +3,006 MB pre-campaign, +1,268-1,294 MB in the Apr/Aug
  readings) — see performance plan §4e-results.
- ROCm cells (inferflux_rocm / llama_cpp_rocm, GGUF and safetensors): the
  R9700 dropped out of WSL passthrough mid-session (`/dev/kfd` absent).
  Last spot measurements from Sep 7 morning: inferflux_rocm ~17-36 tok/s at
  c=1-8 on GGUF; ST-on-ROCm remains unbuilt (no HIP bf16 forward).
  Re-run when the host restores the device.

## 3) Debt Register

| Priority | Item | Status | Notes |
|---|---|---|---|
| ~~P0~~ | ~~Chat template stub~~ | **FIXED** | Strategy-based renderer, now shared via `model/chat_template_renderer` for GGUF and HF tokenizers |
| ~~P0~~ | ~~Missing repetition penalty~~ | **FIXED** | CUDA kernel + per-sequence tracking + 1.15x greedy default |
| ~~P0~~ | ~~KV cache corruption on reuse~~ | **FIXED** | ClearSequenceAsync before prefill when n_past==0 |
| ~~P0~~ | ~~Safetensors tokenizer: garbled output + runaway generation~~ | **FIXED (Sep)** | ByteLevel BPE decode (Sequence-wrapped pre-tokenizer), chat template, EOS from config.json fallback; concurrent determinism verified |
| ~~P1~~ | ~~Decode relay fingerprint provably inert~~ | **FIXED (Sep)** | Contract matched to DeviceTokenRelayKernel; engagement counter `inferflux_scheduler_decode_relay_replays_total`; kill switch `INFERFLUX_DISABLE_DECODE_RELAY` |
| ~~P2~~ | ~~Decode CUDA graphs on safetensors path~~ | **DONE (Sep)** | Per-width LRU graph set (+17%); kill switch `INFERFLUX_DISABLE_CUDA_GRAPH` |
| P1 | GGUF multi-variant decode numerics | Documented, benign | Concurrent GGUF loads produce 2-3 coherent output variants (decode-composition-dependent kernel selection). Present with relay/graphs on and off. Investigate only on user report |
| ~~P1~~ | ~~GGUF overhead +2.9-3.6 GB vs llama.cpp~~ | **FIXED (Sep)** | Root-caused via nsys memory trace (#107 §4e): 3x per-replica MMQ layout copies + full-window scratch + unguarded KV. #108/#109/#110 land shared layouts, demand-sized scratch, and admission bounds: netted steady 6,391 -> 4,762 MB (-1.6 GB), throughput parity-or-better (422-456 tok/s c=16). Residual vs llama.cpp = worst-case KV reserve |
| P1 | Native structured output | Not started | Grammar-constrained generation still delegates to the llama.cpp parity backend |
| P1 | Speculative decoding integration | Partial | Draft+validate wired; not production-validated on either format |
| P2 | Distributed sequence ownership cleanup | In progress | KV channel + SHM transport production-tested; cleanup hardening remains |
| P2 | llama_cpp_cuda c=1 request failures | Diagnosed | llama.cpp internal graph optimization >120s on fresh load; benchmark-side workaround in place |

## 4) Performance Campaign Record (Sep 2026)

Full record with falsifications: `docs/design/SAFETENSORS_DECODE_PERFORMANCE_PLAN.md`.

| Lever | Result |
|---|---|
| cublasLt lm_head dispatch | +7% — engagement counter added after review caught a silent scaleType no-op |
| Decode CUDA graphs (safetensors) | +17% — per-width LRU set; ~200 graph launches replace ~105k kernel launches |
| Decode relay fingerprint | Re-activated dead code; neutral throughput, correctness + observability |
| Gate/up fusion | Falsified (1.01-1.02x, noise) |
| Quick bf16 specialist kernel | Falsified — correct deep-MLP implementation lands at the same ~40% practical streaming ceiling as cuBLAS cold-L2 |
| Solo-burst cap (width-tail monopolization theory) | Falsified — width-1 tails are closed-loop EOS stagger, not pipeline starvation |
| Prefill-stall theory | Falsified — prefill is ~1% of decode-path compute time |

Methodology notes that must survive: run A/B comparisons twice (harness
variance ~30% on llama_cpp); `INFERFLUX_CUDA_PHASE_TIMING=1` halves
throughput (never use it for perf claims); warm-L2 microbenchmarks
overstate small shapes (>100% roofline); engagement of any perf change
must be proven via counters or kernel traces, not inferred.

## 5) Recommended Next Execution Order

| # | Work item | Impact |
|---|---|---|
| 1 | Width-tail refill policy (admit waiting prefills into running cohorts during EOS-staggered tails) | Last moderate lever: +25-40% at c=16 if width-1 fraction drops below 10% — mechanism identified, falsification record shows naive caps fail; needs cohort-refill design |
| 2 | Re-measure GGUF path on the current build | The Apr GGUF snapshot predates CUDA graphs, relay, cublasLt — all now apply to GGUF too |
| ~~3~~ | ~~GGUF memory-overhead investigation~~ | **DONE** — root cause + fixes landed (#107/#108/#109/#110); follow-ups #111-#113 |
| 4 | Heavyweight kernel path (TMA/wgmma) | Only with a concrete customer case; quick-win class falsified |
| 5 | Differentiation features | Memory footprint, quantized serving, single binary — the measured strengths |

## 6) OSS Release Readiness

| Area | Grade | Notes |
|---|---|---|
| Licensing | A- | Apache 2.0, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT |
| Docs | A- | Canonical docs current, check_docs_contract enforced; kill switches documented |
| Benchmark | B+ | 6-engine coverage (inferflux/llama_cpp/Ollama/LM Studio/vLLM/SGLang), semantic similarity, falsification-aware methodology |
| Tests | A- | 525 CPU tests / 2833 assertions, isolation probe, parity gates per refactor stage |
| Release process | B+ | Protected hosted checks + trusted dual-GPU evidence gate |
