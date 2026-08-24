# Competitive Positioning

**Snapshot date:** March 9, 2026

```mermaid
quadrantChart
    title InferFlux Positioning
    x-axis "Compatibility / Control" --> "Peak Native Throughput"
    y-axis "Single-node" --> "Distributed Runtime"
    quadrant-1 "Throughput leaders"
    quadrant-2 "Target zone"
    quadrant-3 "Local/runtime-only"
    quadrant-4 "Control-plane-heavy"
    InferFlux: [0.60, 0.56]
```

## 1) Where InferFlux Competes Today

| Category | Typical strength | InferFlux status now |
|---|---|---|
| Compatibility-first local servers | Broad GGUF support and low friction | Covered through `llama_cpp_cuda` and CPU paths; not the main differentiator |
| Native-kernel-first GPU servers | Peak throughput and batching efficiency | Native path is real but still behind best-in-class maturity on quantized heavy-batch workloads |
| Operator/control-plane servers | Auth, policy, routing, audit, metrics, admin APIs | Strong today and part of the shipped product contract |
| Distributed inference stacks | Multi-node reliability and ownership semantics | Transport-health foundation exists, but not yet a competitive strength |

## 2) What Is Distinctive

| Trait | Why it matters |
|---|---|
| Two-CUDA-backend strategy | Separates compatibility risk from performance experimentation without hiding fallback behavior |
| Machine-visible backend identity | Makes policy, automation, and benchmarking deterministic |
| API/admin/CLI contract rigor | Control-plane behavior is explicit and testable, not best-effort |
| Transport-aware ops semantics | `/readyz`, `/v1/admin/pools`, and optional fail-closed generation admission understand distributed KV degradation |
| Portable runtime scope | One server surface spans CPU/CUDA/ROCm/MPS/MLX targets |

## 3) Best-In-Class Gap Map

| Gap | What closes it |
|---|---|
| Quantized native throughput | Fused quantized kernels, graph capture, and batch-quality-preserving scheduling |
| Memory economy on edge GPUs | Immutable quantized weights, budgeted KV sizing, paged KV maturity, and no persistent dequant by default |
| Distributed runtime credibility | Sequence ownership cleanup, worker-loss handling, and explicit failure-path CI on top of the new transport-health contract |
| Release confidence on GPU paths | Required GPU behavior lane instead of optional perf smoke runs |

## 4) Competitive Reading

| Area | InferFlux strength now | InferFlux gap now |
|---|---|---|
| Control plane | Already unusually strong for an OSS inference server | Must keep docs and tests aligned as runtime changes |
| Native runtime | Clear architectural direction and growing memory/identity rigor | Still needs throughput proof on quantized hot paths |
| Distributed runtime | More honest and more explicit than “scaffold-only” claims | Still not ownership-safe enough for broad scale claims |

## 5) Canonical Source Map

| Need | Source |
|---|---|
| Product intent | [PRD](PRD.md) |
| Grade and execution plan | [Roadmap](Roadmap.md) |
| Debt and migration priority | [TechDebt_and_Competitive_Roadmap](TechDebt_and_Competitive_Roadmap.md) |
| Practice modernization | [MODERNIZATION_AUDIT](MODERNIZATION_AUDIT.md) |
