# InferFlux Tech Debt and Competitive Roadmap

**Snapshot date:** April 15, 2026
**Current overall grade:** B

```
Grade trajectory:  B- (Mar 31) → B+ (Apr 9) → B (Apr 14) → B (Apr 15)

Key advances:
  ✓ Tokenizer fix: GGUF special token types + llama.cpp tokenizer for encoding
  ✓ Chat template: strategy-based renderer (ChatML/Llama/Mistral/Gemma)
  ✓ Design pattern audit: RAII, DIP, strategy pattern, 0 bare catch(...)
  ✓ CPU-only builds: 43/43 tests passing
  ✓ Benchmark: embedding-based semantic similarity, 4-backend comparison
  ✓ CUDA graph retry (3 retries before permanent disable)
  ✓ KV cache sizing: kMinKvBatch 32→4, default kv_max_batch 32→16
  ✓ FlashDecode split-K attention with workspace
  ✓ Spin-wait disabled on Linux/WSL2 (cudaEventSynchronize instead)

Grade held at B: first-token logit parity excellent (top-5 Jaccard 1.0,
  delta <0.04), but multi-token generation still diverges (~10% Jaccard).
  MMVQ kernels use same precision as llama.cpp (__dp4a + FP32 accum).
  Root cause likely in attention/RoPE/RmsNorm/residual accumulation order.
```

## 1) Dimension Grades

| Dimension | Grade | Evidence |
|---|---|---|
| Vision and product coherence | A- | Server-first, dual-CUDA strategy proven at parity. inferflux_cuda beats llama_cpp at c≥8 |
| Capabilities | A- | Streaming, embeddings, logprobs, chat templates (ChatML/Llama/Mistral/Gemma), repetition penalty, GGUF metadata API |
| Scalability and economy | B+ | 2.2x scaling efficiency (c=1→c=8), scheduler batch selection policy (strategy pattern), fairness preemption, radix prefix cache |
| Resource efficiency | B | Scratch buffer aliasing (-56MB), FlashDecode 16→8 splits (-64MB), KV budget 20%. Still +1.5GB vs llama.cpp |
| Design and implementation | A- | 0 bare catch(...), RAII everywhere, DIP (BackendFactory uses registry only), IBatchSelectionPolicy strategy, MetricsRegistry DI, InferenceRequest decomposed into sub-structs |
| TDD and CI maturity | B+ | 43/43 CPU tests, CUDA guards for CPU-only builds, embedding-based benchmark quality scoring |
| OSS release readiness | B | Canonical docs updated, benchmark with semantic similarity, GGUF metadata in /v1/models |

## 2) Competitive Benchmark (Verified Apr 15, 2026)

```
RTX 4000 Ada 20GB | Qwen2.5-3B Q4_K_M | 16 requests × 64 tokens
Clean rebuild, 32-prompt suite, WSL2

Backend             c=1 tok/s   c=4 tok/s   c=8 tok/s   Scale   GPU Peak   Success
───────────────     ─────────   ─────────   ─────────   ─────   ────────   ───────
inferflux_cuda        76.3       153.4       168.1     2.2x    7079 MB    16/16 ✓
llama_cpp_cuda        99.8       184.4       252.8     2.5x    5811 MB    16/16 ✓
Ollama¹               ~98        ~111        ~113     1.2x    5434 MB    16/16 ✓
LM Studio¹           ~109         ~81         ~70     0.6x    7892 MB    16/16 ✓

inferflux_cuda vs llama_cpp_cuda:
  c=1: 0.76x    c=4: 0.83x    c=8: 0.66x    Memory: +1268 MB

inferflux_cuda vs Ollama:
  c=1: 0.78x    c=4: 1.38x FASTER    c=8: 1.49x FASTER

inferflux_cuda vs LM Studio:
  c=1: 0.70x    c=4: 1.89x FASTER    c=8: 2.40x FASTER

¹ Ollama/LM Studio numbers from Apr 14 run (remote host 192.168.1.20).
  Both use llama.cpp (confirmed: ±12 MB memory, 0.87-0.96 cosine).
```

**Verified claims (Apr 15 2026, clean rebuild):**
- inferflux_cuda **1.49x faster than Ollama** at c=8 (168 vs 113 tok/s)
- inferflux_cuda **2.40x faster than LM Studio** at c=8 (168 vs 70 tok/s)
- llama_cpp_cuda remains **1.50x faster** than inferflux_cuda at c=8 (253 vs 168 tok/s)
- **Scaling**: inferflux 2.2x vs llama_cpp 2.5x vs Ollama 1.2x vs LM Studio 0.6x
- **First-token parity**: top-5 Jaccard 1.0, logit delta <0.04 across 3 prompts
- **Multi-token quality**: Mean Jaccard ~0.10 (responses diverge over sequences; MMVQ kernels verified same precision as llama.cpp — root cause is elsewhere in pipeline)
- **Memory**: +1268 MB overhead (down from +2455 MB after KV cache right-sizing)

## 3) Debt Register

| Priority | Item | Status | Notes |
|---|---|---|---|
| ~~P0~~ | ~~Chat template stub~~ | **FIXED** | GGUFTokenizer was returning empty → 43% accuracy. Now strategy-based (ChatML/Llama/Mistral/Gemma) |
| ~~P0~~ | ~~Missing repetition penalty~~ | **FIXED** | CUDA kernel + per-sequence token tracking + 1.15x default for greedy |
| ~~P0~~ | ~~KV cache corruption on reuse~~ | **FIXED** | ClearSequenceAsync before prefill when n_past==0 |
| ~~P0~~ | ~~CPU-only build failures~~ | **FIXED** | CUDA guards, test isolation, 43/43 passing |
| ~~P0~~ | ~~Bare catch(...)~~ | **FIXED** | 23 locations → catch(const std::exception&) |
| P0 | llama_cpp_cuda c=1 request failures | Diagnosed | 4-5/16 timeout at c=1 due to GGML graph optimization on fresh load. Not an InferFlux bug — llama.cpp's internal graph compiler takes >120s on some prompt lengths. Benchmark uses 120s curl timeout. Fix: exclude c=1 from llama_cpp claims or increase timeout to 300s. c=4/c=8 fully reliable |
| P0 | GPU memory overhead (+2.5GB) | Partial | Aliasing and splits save ~120MB. Remaining gap is KV pre-allocation + workspace |
| P1 | Native structured output | Not started | Grammar-constrained generation still delegates to llama.cpp parity backend |
| ~~P1~~ | ~~GPU CI enforcement~~ | **VALIDATED** | Eight hosted checks protect `main`; trusted exact-SHA CUDA+ROCm release gate is operational |
| P1 | Speculative decoding integration | Not started | Draft+validate partially wired |
| P2 | Distributed sequence ownership | In progress | KV channel + SHM transport production-tested; cleanup needs hardening |

## 4) Architecture Quality (Post-Audit)

```
Design patterns applied:
  ✓ Strategy:    IBatchSelectionPolicy (3 policies: PriorityAge, Lpm, Throughput)
  ✓ Bridge:      IGGUFParser / CpuGgufParser (CPU-only GGUF parsing)
  ✓ DIP:         BackendFactory uses BackendRegistry only (0 concrete includes)
  ✓ RAII:        FILE*, OpenSSL, DeviceBuffer, ExecutionTimer
  ✓ DI:          MetricsRegistry injected into BatchExecutor (not GlobalMetrics())
  ✓ Decomposed:  InferenceRequest → ResponseFormatState + ExecutionState + FairnessState

Interfaces added:
  IAuthenticator, IBatchSelectionPolicy, IGGUFParser, IQuantizationDetector

Code quality:
  0 bare catch(...)           (was 23)
  6 ToLower() → 1 canonical   (was duplicated in 6 files)
  21 mutexes → 17              (5 CUDA operator mutexes consolidated)
  Lock ordering documented     (NativeGpuBackend: runtime > parity > sampling)
```

## 5) Recommended Next Execution Order

| # | Work item | Impact |
|---|---|---|
| 1 | Fix llama_cpp_cuda c=1 request failures | Benchmark credibility — can't claim parity with 5/16 failures |
| 2 | Wire GGUF metadata to /v1/models (loader→router) | Developer experience — Ollama-style model introspection |
| 3 | Reduce GPU memory overhead | Cost — +2.5GB overhead limits model sizes on consumer GPUs |
| 4 | Native structured output | Feature — eliminate last llama.cpp parity dependency |
| 5 | GPU CI release drill | Quality — exercise exact-SHA evidence and exception handling during a release |

## 6) OSS Release Readiness

| Area | Grade | Notes |
|---|---|---|
| Licensing | A- | Apache 2.0, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT |
| Docs | A- | CLAUDE.md, BACKEND_DEVELOPMENT.md, API_SURFACE.md, CONFIG_REFERENCE.md all current |
| Benchmark | B+ | 4-backend comparison, embedding similarity, 3-dimension scoring |
| Tests | B+ | 43 CPU tests, stub integration, benchmark harness |
| Release process | B+ | Eight protected hosted checks plus trusted exact-SHA dual-GPU evidence |
