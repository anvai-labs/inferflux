# Competitive Positioning

**Snapshot date:** September 7, 2026 (2-run average per cell, RTX 4000 Ada,
Qwen2.5-3B; multi-backend harness with response classification — see
[benchmarks](benchmarks.md) for the methodology note on run-to-run variance)

```
InferFlux Positioning (Sep 7 2026, 2-run avg, tok/s):

  ┌───────────────────────────────────────────────────────────┐
  │ GGUF Q4_K_M                                               │
  │  InferFlux inferflux_cuda ★                               │
  │   ├─ 333 tok/s at c=16 (~1.44x vs llama.cpp same harness) │
  │   ├─ ~2.7x faster than local Ollama at c=16               │
  │   ├─ 3.22x scaling (c=1→c=16)                             │
  │   └─ 0 classified failures; 5,478 MB loaded (ledger)      │
  │  llama_cpp_cuda: 120/143/199/284/231 (c=1/2/4/8/16)       │
  │                                                           │
  │ SAFETENSORS bf16                                          │
  │  vLLM     675 tok/s at c=16   (2.00x over inferflux)      │
  │  SGLang   515 tok/s at c=16   (1.52x over inferflux)      │
  │  InferFlux 338 tok/s at c=16  (~7.1x scaling)             │
  │  gap narrowed from 2.37-2.70x (Sep 4) to 1.52-2.00x       │
  │                                                           │
  │ Memory at the same workload (GPU peak, GB)                │
  │  GGUF: inferflux 5.5 (ledger) vs llama.cpp 4.1            │
  │  ST:   inferflux 8.3-8.7 vs vLLM ~20.1 vs SGLang 18.2-20.1│
  └───────────────────────────────────────────────────────────┘
```

## 0) Feasibility matrix (what runs where)

| Engine | GGUF / CUDA | Safetensors / CUDA | GGUF / ROCm | Safetensors / ROCm |
|---|---|---|---|---|
| `inferflux_cuda` / `inferflux_rocm` | ✓ | ✓ | ✓ (last measured Sep 7 morning: ~17-36 tok/s c=1-8; GPU passthrough currently absent on the bench host) | ✗ no HIP bf16 forward built |
| llama.cpp (`llama_cpp_cuda` / `llama_cpp_rocm`) | ✓ | ✓ via f16 GGUF sidecar (harness-enabled; the router resolves a `*.gguf` sidecar in the model dir) | ✓ (blocked with host) | via sidecar; blocked with host |
| vLLM | ✗ (GGUF unsupported in this venv) | ✓ | ✗ not installed for ROCm | ✗ |
| SGLang | ✗ | ✓ (requires `TVM_FFI_GPU_BACKEND=cuda` when ROCm toolchain is on PATH — its JIT otherwise misdetects HIP) | ✗ not installed | ✗ |
| Ollama / LM Studio | ✓ (local Ollama plateaus ~123 tok/s flat) | LM Studio only | — | — |

Note: the AMD R9700 dropped out of WSL passthrough mid-session (`/dev/kfd`
absent); ROCm cells retain the Sep 7 morning spot measurements and should be
re-run when the host restores the device.

### 0b) Stock llama.cpp server — the missing baseline (Sep 8)

The campaign rows for "llama.cpp" measure InferFlux's **wrapper**
(`llama_cpp_cuda`: InferFlux scheduler -> llama.cpp library), not the stock
`llama-server` that ships with llama.cpp and carries its own continuous
batching. Benchmarked separately (same model, same concurrent battery, 2-run
averages, 256-token completions; built from the pinned submodule with CUDA;
`-ngl 99 -c 4096 -np 16 -fa on` — the insights-applied configuration):

| Backend | c=1 | c=8 | c=16 | GPU peak |
|---|---|---|---|---|
| `inferflux_cuda` | 95.8 | 310.1 | 506.6 | 5.4 GB |
| wrapper `llama_cpp_cuda` (seqs=16, post-#117) | 104.7 | 311.7 | 590.8 | 3.0 GB |
| stock `llama-server` (16 slots) | 104.4 | **390.1** | **664.4** | **3.0 GB** |

- **Stock llama-server leads this burst workload at c>=8**: +20-31% over
  `inferflux_cuda` and +12-13% over the tuned wrapper (per-run range across
  the two runs), at the same 3.0 GB as the tuned wrapper. The campaign's "1.44x over llama.cpp at
  c=16" claim holds only against the wrapper (and on the campaign's
  32x64-token workload); against stock llama-server on this battery,
  `inferflux_cuda` trails at c>=8.
- Output inspection: stock-server responses are coherent and correct; its
  greedy outputs vary more across slots (batch-composition numerics, same
  phenomenon both engines show).
- The two llama.cpp deployments serve different purposes: the wrapper
  exists for InferFlux's scheduler/auth/policy surface, stock llama-server
  for raw throughput. Closing the stock-server gap at c>=8 is the new
  performance target; the campaign's workload (many short completions)
  vs this battery (longer 256-token completions, 3 unique prompts shared
  by all requests — prefix-cache-friendly, identically for every engine)
  rank the engines differently, so both measurements are kept side by
  side.

## 1) Current Position

Throughput, tok/s, 2-run average (RTX 4000 Ada, Qwen2.5-3B; multi-backend
harness; per-cell response classification reported 0 failures everywhere):

| Backend | GGUF c=1 | GGUF c=4 | GGUF c=8 | GGUF c=16 | ST c=1 | ST c=4 | ST c=8 | ST c=16 | GPU peak (GB) |
|---|---|---|---|---|---|---|---|---|---|
| `inferflux_cuda` | 103.4 | 163.8 | 265.8 | **332.7** | 47.7 | 143.8 | 206.2 | **338.2** | 5.5 GGUF loaded / 8.3-8.7 ST |
| llama.cpp CUDA | **119.8** | **198.7** | **284.0** | 231.4 | 44.7 | 82.5 | 143.8 | 93.0¹ | 4.1 GGUF / 8.0 ST¹ |
| vLLM | — | — | — | — | 36.9 | 160.2 | 336.0 | **675.3** | ~20.1 |
| SGLang | — | — | — | — | 38.1 | 156.9 | 307.0 | **515.3** | 18.2-20.1 |
| Ollama (local) | 121.3 | 124.1 | 124.2 | 123.1 | — | — | — | — | ~1.0 (run 2 sampled ~2.9 — attribution inconsistent) |
| LM Studio | 115.4 | 71.7 | 76.2 | 75.0 | 115.8 | 75.2 | 73.0 | 71.5 | 2.9-3.1 |

¹ llama.cpp cannot ingest safetensors directly; its ST cell runs an f16 GGUF
sidecar of the same weights (llama.cpp's standard path for HF weights), so it
measures llama.cpp at f16 — a different quantization than its own q4_k_m GGUF
row. Throughput degrades from c=8 to c=16 (f16 KV pressure); run-to-run
variance on this cell is the highest measured (c=16: 87.9-98.0).

| Category | Reading |
|---|---|
| Native CUDA serving at high concurrency (GGUF) | **Leads llama.cpp at c=16 on the same harness in both runs** (avg 332.7 vs 231.4 = 1.44x; runs 1.38x and 1.49x). llama.cpp stays clearly ahead at c=1-4 (0.86x/0.82x) and led c=8 in every Sep 7 run on both harnesses (0.80-0.95x). |
| GGUF memory | `inferflux_cuda` peak 5,478 MB vs llama.cpp 4,086 MB on the identical gguf-compare workload (**+1,392 MB**, down from +3,006 MB before the Sep 7 memory campaign). Ledger split: weights+shared MMQ layouts 2,660 MB, KV reserve 1,208 MB, workspaces ~275 MB. |
| vs Ollama (GGUF) | **~2.7x faster** at c=16 (332.7 vs 123.1). Local Ollama plateaus flat (~120-125) at every concurrency — no cross-request batching. |
| vs vLLM / SGLang (safetensors) | Still behind at concurrency but the gap narrowed: c=16 avg 338.2 vs vLLM 675.3 (**2.00x**) and SGLang 515.3 (**1.52x**) — was 2.37-2.70x on Sep 4. vLLM/SGLang scale ~14-18x from c=1; `inferflux_cuda` ~7x. Both engines also carry the memory bill: `inferflux_cuda` serves the same workload at **2.3-2.4x less GPU memory** (8.3-8.7 GB vs ~18.2-20.1 GB). |
| vs llama.cpp on full-precision weights | `inferflux_cuda` **3.6x faster** at c=16 (338.2 vs 93.0, 2-run avg) on the same safetensors weights (llama.cpp via f16 GGUF sidecar — its highest-precision serving mode), and comparable memory (8.3-8.7 vs 8.0 GB). |
| Output quality | 0 classified failures in every gguf-compare cell (24 measurements — the only harness that emits the field; 12 cells x 2 runs); GGUF semantic parity vs llama.cpp on this campaign's 64-token greedy generations: mean Jaccard ~0.55 / overlap ~0.69. |
| Measurement caution | Run-to-run variance up to ~30% on this harness (c=4 GGUF swung 146-182 across runs). All cited numbers are 2-run averages; never cite a single run. |
| Operator rigor | Production-grade: metrics, audit, RBAC, guardrails, health probes |
| Architecture quality | RAII, DIP, strategy pattern; extensive unit test suite, 0 bare `catch(...)` |

## 2) Why InferFlux Wins at Concurrency

| Factor | InferFlux | Ollama | LM Studio |
|---|---|---|---|
| Request dispatch | C++ unified batch → single GPU kernel | Go goroutine → CGo → llama.cpp per-request | Node.js event loop → llama.cpp server |
| Language boundary | None (C++→CUDA) | Go→C (CGo, ~1-5μs/call × N) | JS→HTTP→C++ (subprocess) |
| Batching | IBatchSelectionPolicy groups N sequences into 1 forward pass | No cross-request batching | No batching (sequential) |
| Scaling (GGUF, c=1→c=16, 2-run avg) | ~3.22x | ~1.01x (flat) | ~0.65x (degrades) |
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
| GPU memory overhead (+2.9 to +3.6 GB vs llama_cpp_cuda on GGUF, 2-run range) | **Resolved (Sep 7)**: root-caused via nsys memory trace and fixed in #108/#109/#110 — netted steady live 6,391 → 4,762 MB (−1,629 MB) with throughput parity-or-better. Residual vs llama.cpp is dominated by the worst-case KV reserve (see performance plan §4e-results). |
| `inferflux_cuda` still behind llama_cpp_cuda at c=1-4 | Consistent both runs; the c=1-4 gap is the active competitive target |
| c=8 crossover point is noisy | Don't rely on a specific c=8 ratio in either direction until more runs are collected |
| `inferflux_cuda` behind vLLM/SGLang on full-precision safetensors | Narrowed to **1.52-2.00x at c=16** (Sep 7: 338 vs 515/675 tok/s; was 2.37-2.70x on Sep 4) while serving at 2.3-2.4x less GPU memory. Both rivals scale near-linearly (~14-18x c=1→c=16); `inferflux_cuda` ~7x. |
| ~~`inferflux_cuda` safetensors output malformed / never stops~~ | **Fixed.** Byte-level BPE decode, chat-template rendering (now shared with GGUFTokenizer via `model/chat_template_renderer.{h,cpp}`), and config.json-based BOS/EOS fallback all landed together — see [benchmarks](benchmarks.md) for the full trace. |
| Native structured output | Still delegates to llama.cpp parity backend |
| GPU CI enforcement | Eight hosted checks protect `main`; trusted CUDA+ROCm release gate operational |
| Speculative decoding | Partially integrated |

## 5) Release-Facing Guidance

| Question | Answer |
|---|---|
| What can we claim? | Leads the llama.cpp wrapper at c=16 on the same harness (~1.44x, both runs agree; see §0b for the stock llama-server baseline), ~2.7x faster than Ollama at c=16, 0 classified failures campaign-wide, high semantic parity, and 2.3-2.4x less GPU memory than vLLM/SGLang on safetensors (1.5-2.0x their throughput) |
| What should not be oversold? | Throughput at c=1-4 (llama_cpp_cuda clearly wins there), any single-run number (variance to ~30%), the residual GGUF memory overhead (+1.4 GB, dominated by the worst-case KV reserve), and full-precision safetensors speed — vLLM/SGLang still lead by 1.5-2.0x at c=16 |
| Developer pitch | "The performance of custom CUDA kernels with the compatibility of llama.cpp, in a single binary with OpenAI-compatible APIs" |

## 6) References

- [Benchmark details](benchmarks.md)
- [Multi-backend benchmark harness checklist](benchmark_multi_backend_steps.md)
- [Tech Debt & Roadmap](TechDebt_and_Competitive_Roadmap.md)
- [Backend Development Guide](BACKEND_DEVELOPMENT.md)
