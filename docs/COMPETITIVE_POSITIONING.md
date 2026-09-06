# Competitive Positioning

**Snapshot date:** September 4, 2026 (2-run average, GGUF stage — see
[benchmarks](benchmarks.md) for the full methodology note on run-to-run
variance)

```
InferFlux Positioning (Sep 2026, GGUF Q4_K_M, Qwen2.5-3B, 2-run avg):

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │    InferFlux inferflux_cuda ★                    │
  │    ├─ 356 tok/s at c=16 (~1.56x vs llama.cpp)    │
  │    ├─ c=8 roughly at parity with llama.cpp       │
  │    ├─ ~2.88x faster than Ollama at c=16          │
  │    ├─ High semantic parity (0.89-0.95 cosine)    │
  │    └─ Best scaling: ~3.33x (c=1→c=16)            │
  │                                                  │
  │    InferFlux llama_cpp_cuda                       │
  │    ├─ Fastest at c=1-4 (both runs agree)          │
  │    └─ ~228 tok/s at c=16 (avg; range 224-233)     │
  │                                                  │
  │    Ollama (remote, Go + llama.cpp)                │
  │    └─ ~124 tok/s flat, all concurrency (plateaus)│
  │                                                  │
  └──────────────────────────────────────────────────┘
```

## 1) Current Position

| Category | Reading |
|---|---|
| Native CUDA serving at high concurrency | **Leads llama.cpp at c=16 in both of two runs** (avg 356.3 vs 228.1 tok/s = 1.56x; individual runs 1.50x and 1.62x). Still clearly behind at c=1-4. c=8 is genuinely contested — the two runs disagreed on which backend led. |
| vs Ollama | **~2.88x faster** at c=16 (356.3 vs 123.8 tok/s avg). Ollama is the most reproducible backend measured — plateaus flat (~122-126 tok/s) across all concurrency in both runs, no cross-request batching. |
| vs LM Studio | Full-precision safetensors run (post-fix): LM Studio's best showing is c=1 (~103 tok/s), then degrades; `inferflux_cuda` overtakes it from c=4 onward and reaches ~279 tok/s at c=16. |
| vs vLLM / SGLang on full precision | **Both clearly beat `inferflux_cuda`'s safetensors path** at concurrency — vLLM 753.4 tok/s and SGLang 659.8 tok/s at c=16 vs `inferflux_cuda`'s 278.7 (2.70x and 2.37x). Both scale near-linearly (~15.3x vLLM, ~13.7x SGLang vs `inferflux_cuda`'s ~6.6x, c=1→c=16). Expected: both are purpose-built serving engines with mature paged/radix attention; `inferflux_cuda`'s safetensors path is newer and less optimized than its GGUF path. Not a claim to contest today — a clear improvement target instead. |
| Output quality (GGUF) | High semantic parity across all backends (0.893-0.945 cosine similarity, all concurrency levels, both runs) — no backend produced degenerate output. |
| Output quality (safetensors) | **Fixed.** `inferflux_cuda`'s safetensors output had undecoded byte-level BPE artifacts (`Ġ`/`Ċ` instead of spaces/newlines) *and*, found while fixing that, never applied a real chat template or resolved the correct EOS token — so it also failed to stop generation, hallucinating fake conversation turns after answering. Both root-caused and fixed (see [benchmarks](benchmarks.md) for the full trace); verified via 12 new/extended unit tests, a live end-to-end API check returning clean text with `finish_reason: "stop"`, and a post-fix benchmark re-run (numbers above) whose `total_tokens` dropped from 2048 to 2025 — direct evidence of correct stopping, not just the two hand-checked repro prompts. |
| Measurement caution | Stage 1 (GGUF) throughput varied up to ~30% between two back-to-back runs at the same concurrency, especially for `llama_cpp_cuda`. Single-run benchmark numbers on this harness should not be treated as precise; average multiple runs before citing a specific ratio. |
| Operator rigor | Production-grade: metrics, audit, RBAC, guardrails, health probes |
| Architecture quality | RAII, DIP, strategy pattern; extensive unit test suite, 0 bare `catch(...)` |

## 2) Why InferFlux Wins at Concurrency

| Factor | InferFlux | Ollama | LM Studio |
|---|---|---|---|
| Request dispatch | C++ unified batch → single GPU kernel | Go goroutine → CGo → llama.cpp per-request | Node.js event loop → llama.cpp server |
| Language boundary | None (C++→CUDA) | Go→C (CGo, ~1-5μs/call × N) | JS→HTTP→C++ (subprocess) |
| Batching | IBatchSelectionPolicy groups N sequences into 1 forward pass | No cross-request batching | No batching (sequential) |
| Scaling (GGUF, c=1→c=16, 2-run avg) | ~3.33x | ~1.00x (flat) | not measured on GGUF this run |
| GC/runtime pauses | None | Go GC stop-the-world | V8 GC + event loop stalls |

## 3) What Is Distinctive

| Trait | Why it matters |
|---|---|
| Two-CUDA-backend strategy | `inferflux_cuda` (native kernels) + `llama_cpp_cuda` (fallback) — separates innovation from compatibility |
| Machine-visible backend identity | Policy, benchmarks, and automation see which kernel ran |
| Production-grade native CUDA | FlashAttention-2, 50+ fused GEMV, CUDA graphs, repetition penalty kernel, chat template rendering |
| Chat template auto-detection | ChatML/Llama/Mistral/Gemma, shared logic across both GGUF (`tokenizer.chat_template`) and safetensors (`tokenizer_config.json` / `chat_template.jinja`) — no manual config |
| GGUF metadata in /v1/models | Ollama-style model introspection via OpenAI-compatible API |

## 4) Remaining Gaps

| Gap | Status |
|---|---|
| GPU memory overhead (+2.9 to +3.6 GB vs llama_cpp_cuda on GGUF, 2-run range) | Partially mitigated (aliasing, splits, budget). Structural from pre-allocated workspace; grew from the +1.3 GB 2026-08-31 reading — not yet re-explained, worth a follow-up rather than assumed regression. |
| `inferflux_cuda` still behind llama_cpp_cuda at c=1-4 | Consistent both runs; the c=1-4 gap is the active competitive target |
| c=8 crossover point is noisy | Don't rely on a specific c=8 ratio in either direction until more runs are collected |
| `inferflux_cuda` well behind vLLM/SGLang on full-precision safetensors | 2.37-2.70x behind at c=16 (post-fix), widening with concurrency (both scale near-linearly, `inferflux_cuda` doesn't). |
| ~~`inferflux_cuda` safetensors output malformed / never stops~~ | **Fixed.** Byte-level BPE decode, chat-template rendering (now shared with GGUFTokenizer via `model/chat_template_renderer.{h,cpp}`), and config.json-based BOS/EOS fallback all landed together — see [benchmarks](benchmarks.md) for the full trace. |
| Native structured output | Still delegates to llama.cpp parity backend |
| GPU CI enforcement | Eight hosted checks protect `main`; trusted CUDA+ROCm release gate operational |
| Speculative decoding | Partially integrated |

## 5) Release-Facing Guidance

| Question | Answer |
|---|---|
| What can we claim? | Leads llama.cpp at c=16 (~1.56x, reproduced across two runs), ~2.88x faster than Ollama at c=16, high semantic parity across all backends on GGUF |
| What should not be oversold? | Throughput at c=1-4 (llama_cpp_cuda clearly wins there), any single-run c=8 number (it flipped direction between runs), GPU memory efficiency (+2.9-3.6 GB and not yet re-explained since the Apr/Aug readings), and full-precision safetensors speed — vLLM/SGLang beat `inferflux_cuda` there by 2.4-2.7x even post-fix |
| Developer pitch | "The performance of custom CUDA kernels with the compatibility of llama.cpp, in a single binary with OpenAI-compatible APIs" |

## 6) References

- [Benchmark details](benchmarks.md)
- [Multi-backend benchmark harness checklist](benchmark_multi_backend_steps.md)
- [Tech Debt & Roadmap](TechDebt_and_Competitive_Roadmap.md)
- [Backend Development Guide](BACKEND_DEVELOPMENT.md)
