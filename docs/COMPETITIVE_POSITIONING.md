# Competitive Positioning

**Snapshot date:** April 9, 2026

```
InferFlux Positioning (Apr 2026):

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │    InferFlux inferflux_cuda ★                    │
  │    ├─ 160 tok/s at c=8 (parity with llama.cpp)   │
  │    ├─ 1.87x faster than Ollama                   │
  │    ├─ 2.23x faster than LM Studio                │
  │    ├─ 100% accuracy, 0% degenerate               │
  │    └─ Best scaling: 2.2x (c=1→c=8)              │
  │                                                  │
  │    InferFlux llama_cpp_cuda                       │
  │    └─ 156 tok/s at c=8 (fallback path)           │
  │                                                  │
  │    Ollama (Go + llama.cpp)                        │
  │    └─ 86 tok/s at c=8 (plateaus)                 │
  │                                                  │
  │    LM Studio (Electron + llama.cpp)               │
  │    └─ 72 tok/s at c=8 (degrades under load)      │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

## 1) Current Position

| Category | Reading |
|---|---|
| Native CUDA serving | **At parity with llama.cpp at c=8** (159.9 vs 156.4 tok/s = 1.02x). 100% accuracy. |
| vs Ollama | **1.87x faster** at c=8 (160 vs 86 tok/s). Ollama plateaus due to Go CGo overhead. |
| vs LM Studio | **2.23x faster** at c=8 (160 vs 72 tok/s). LM Studio degrades under load (Node.js event loop). |
| Operator rigor | Production-grade: metrics, audit, RBAC, guardrails, health probes |
| Architecture quality | RAII, DIP, strategy pattern. 43 unit tests, 0 bare catch(...) |

## 2) Why InferFlux Wins at Concurrency

| Factor | InferFlux | Ollama | LM Studio |
|---|---|---|---|
| Request dispatch | C++ unified batch → single GPU kernel | Go goroutine → CGo → llama.cpp per-request | Node.js event loop → llama.cpp server |
| Language boundary | None (C++→CUDA) | Go→C (CGo, ~1-5μs/call × N) | JS→HTTP→C++ (subprocess) |
| Batching | IBatchSelectionPolicy groups N sequences into 1 forward pass | No cross-request batching | No batching (sequential) |
| Scaling at c=8 | 2.2x throughput gain | 1.2x (plateaus) | 0.7x (degrades) |
| GC/runtime pauses | None | Go GC stop-the-world | V8 GC + event loop stalls |

## 3) What Is Distinctive

| Trait | Why it matters |
|---|---|
| Two-CUDA-backend strategy | `inferflux_cuda` (native kernels) + `llama_cpp_cuda` (fallback) — separates innovation from compatibility |
| Machine-visible backend identity | Policy, benchmarks, and automation see which kernel ran |
| Production-grade native CUDA | FlashAttention-2, 50+ fused GEMV, CUDA graphs, repetition penalty kernel, chat template rendering |
| Chat template auto-detection | ChatML/Llama/Mistral/Gemma from GGUF metadata — no manual config |
| GGUF metadata in /v1/models | Ollama-style model introspection via OpenAI-compatible API |

## 4) Remaining Gaps

| Gap | Status |
|---|---|
| GPU memory overhead (+2.5 GB) | Partially mitigated (aliasing, splits, budget). Structural from pre-allocated workspace. |
| Native structured output | Still delegates to llama.cpp parity backend |
| GPU CI enforcement | Trusted CUDA+ROCm main gate operational; branch protection pending |
| Speculative decoding | Partially integrated |

## 5) Release-Facing Guidance

| Question | Answer |
|---|---|
| What can we claim? | Throughput parity with llama.cpp, 1.87x faster than Ollama, 2.23x faster than LM Studio — all verified with 100% accuracy |
| What should not be oversold? | GPU memory efficiency (still +2.5 GB), distributed runtime maturity |
| Developer pitch | "The performance of custom CUDA kernels with the compatibility of llama.cpp, in a single binary with OpenAI-compatible APIs" |

## 6) References

- [Benchmark details](benchmarks.md)
- [Tech Debt & Roadmap](TechDebt_and_Competitive_Roadmap.md)
- [Backend Development Guide](BACKEND_DEVELOPMENT.md)
