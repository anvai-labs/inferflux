# InferFlux Vision

**Snapshot date:** April 9, 2026

```mermaid
flowchart LR
    A[OpenAI-compatible APIs] --> B[Operator-grade control plane]
    B --> C[Portable runtime across CPU/CUDA/ROCm/MPS/MLX]
    C --> D[Native memory-efficient execution]
    D --> E[Deterministic distributed contracts]
```

## 1) Product Contract

| Pillar | Concrete meaning |
|---|---|
| API compatibility | Core request/admin surfaces remain OpenAI-style and scriptable |
| Operator control plane | Auth, policy, routing, audit, metrics, and admin APIs are part of the product, not sidecars |
| Dual CUDA strategy | `inferflux_cuda` is the headroom path; `llama_cpp_cuda` is the compatibility/safety path |
| Memory economy | Shared weights, separate KV lifecycle, prefix reuse, and explicit memory policy knobs |
| Stateless default | Baseline API stays stateless; `session_id` reuse is optional and bounded |
| Honest scale path | Single-node rigor first, distributed claims only when lifecycle and failure contracts exist |

## 2) Current Reality

| State | Code-aligned reading |
|---|---|
| Strong today | API/admin/CLI contracts, backend/provider identity, policy-visible fallback, prefix/KV reuse, admin pools visibility, and operator observability |
| Foundation now | Native loader detection, memory-first GGUF dequant policy, KV auto-tune planning/metrics, optional session leases, distributed ticket lifecycle, timeout debt, optional fail-closed generation admission, and stepwise native burst decode on the live phased path |
| **Leadership achieved** | Apr 9 verified: c=1 73.3, c=4 133.6, c=8 159.9 tok/s (16/16 OK, 100% accuracy). At parity with llama.cpp at c=8 (1.02x), 1.87x faster than Ollama, 2.23x faster than LM Studio. Chat template auto-detection (ChatML/Llama/Mistral/Gemma), repetition penalty kernel, KV cache clearing. 0% degenerate responses. Architecture: RAII, DIP, strategy pattern, 43 unit tests. |

## 3) Modern Serving Posture

| Modern practice | InferFlux reading today | What still has to close |
|---|---|---|
| Sync batching over naive async fragmentation | Adopted | Only re-enable native async if it preserves the same batched execution core |
| Quantized serving as first-class runtime path | Adopted foundation | Finish fused GGUF hot paths so memory-first mode is also the fast path |
| Paged KV plus budget-aware planning | Adopted foundation | Mature allocator/ownership behavior under concurrency |
| PD disaggregation with transport health | Adopted foundation | Close sequence ownership, cleanup, and multi-process fault matrix |
| Explicit provider/fallback identity | Adopted | Keep every API/CLI/admin surface aligned as backends evolve |
| Contract gates before grade moves | Adopted stance | Add required GPU/provider lanes before claiming release-grade runtime maturity |

## 4) Old Practices to Retire

| Retire | Replace with |
|---|---|
| Hidden compatibility fallback | Explicit backend/provider/fallback metadata and policy decisions |
| “Async means faster” | Measure batch quality, hot-path residency, and end-to-end throughput |
| Persistent dequant buffers by default | Policy-scoped dequant with memory-first `none` as the native GGUF default |
| Fixed KV reservations | Budgeted KV planning with exported decisions |
| Passive readiness only | Readiness plus optional fail-closed admission where degraded transport should stop new generation work |
| Benchmark-only product claims | Contract tests plus representative runtime evidence |

## 5) Grade Stance

| Area | Reading |
|---|---|
| Overall | `B-` |
| Why not lower | Control-plane, identity, observability, and memory-policy foundations are real and tested |
| Why not higher | MMQ accumulate kernels and lane overlap fixes (0ccbad3) closed the decode down-proj bottleneck and improved concurrent stability. CUDA graphs re-enabled on primary forward. March 31 baseline shows progress (c=1 65.6, c=4 148.3, c=8 174.6 tok/s) but residual c=8 instability (~75% pass rate) and distributed ownership semantics are not yet release-enforced |

## 6) Canonical Source Map

| Need | Source of truth |
|---|---|
| Runtime contract | [Architecture](Architecture.md) |
| Grade and next moves | [Roadmap](Roadmap.md) |
| Debt and migration order | [TechDebt_and_Competitive_Roadmap](TechDebt_and_Competitive_Roadmap.md) |
| Modernization table | [ARCHIVE_INDEX](ARCHIVE_INDEX.md) (MODERNIZATION_AUDIT removed as a point-in-time audit) |
| Product envelope | [PRD](PRD.md) |

Archived long-form narratives stay under [ARCHIVE_INDEX](ARCHIVE_INDEX.md).
