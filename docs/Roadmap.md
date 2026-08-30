# InferFlux Product and Design Roadmap

**Status:** Proposed for co-design

**Planning epoch:** August 21, 2026

**Horizon:** 16 engineering weeks; sequencing is a dependency model, not a delivery promise

```mermaid
flowchart LR
  A[TD-001 Green baseline] --> B[TD-002 Required evidence gates]
  B --> Q[ADR-0005 Trusted GPU release evidence]
  B --> O[TD-006 Dependency/runtime currency]
  O --> P[TD-007 Context and sequence capacity]
  P --> R[TD-008 Unified burst sampling parity]
  R --> C[ADR-0001 Runtime proof-or-pivot]
  C --> D[TD-003 Reproducible runtime evidence]
  D --> E[FTR-002 Agent contract]
  E --> M[ADR-0003 Tenant boundary]
  M --> F[FTR-003 Tenant-safe efficiency]
  F --> G[FTR-004 SLO-aware operation]
  G --> H[ADR-0004 Ownership model]
  H --> N[TD-005 Ownership and dispatch debt]
  N --> I[FTR-005 Distributed maturity]
  A --> J[ADR-0002 Local transport]
  J --> K[TD-004 CLI seams]
  K --> L[FTR-001 One-command local UX]
```

## Outcomes We Work Backward From

| Outcome | Release evidence |
|---|---|
| First useful response in under five minutes | Fresh-machine quickstart test and actionable failure diagnostics |
| Trustworthy agent execution | Structured output/tool contract pass rate >=99% across supported providers |
| Predictable cost and latency | TTFT, ITL, throughput, memory, cache isolation, and saturation gates |
| Honest portability | Provider identity is explicit; unsupported behavior fails without silent fallback |
| Scale without state corruption | One owner for sequence/KV state, idempotent cleanup, fault-injection proof |

## Priority and Dependency Order

| Order | Artifact | Why it is here | Exit gate |
|---|---|---|---|
| 1 | [TD-001](technical-debt/TD-001-green-mainline-baseline.md) | Every later claim depends on a reproducible baseline | CPU build and all model-free tests pass from a non-default build directory |
| 2 | [TD-002](technical-debt/TD-002-required-evidence-gates.md) | Prevents correctness and performance regressions from becoming release claims | CPU contracts required; one stable CUDA behavioral lane defined |
| 3 | [TD-006](technical-debt/TD-006-dependency-runtime-currency.md) | Runtime evidence is invalid when accelerator support silently compiles out or uses obsolete interfaces | Versioned CPU/CUDA/ROCm matrix passes on supported pins |
| 4 | [TD-007](technical-debt/TD-007-context-sequence-capacity.md) | Load evidence is invalid until context capacity, errors, and request completion are deterministic | Mixed-prompt lifecycle and capacity matrix passes on CPU/CUDA/ROCm |
| 5 | [TD-008](technical-debt/TD-008-unified-burst-sampling-parity.md) | Graph replay cannot be a supported optimization while it changes greedy sampling semantics | Stepwise parity, deadline, and heterogeneous lifecycle matrix passes |
| 6 | [ADR-0001](adr/ADR-0001-evidence-gated-runtime-portfolio.md) + [TD-003](technical-debt/TD-003-runtime-evidence-drift.md) | Decides whether native CUDA deserves continued proprietary-kernel investment | Representative correctness, throughput, memory, and reliability matrix |
| 7 | [FTR-001](features/FTR-001-managed-local-and-endpoint-cli.md) | Removes the largest adoption failure without duplicating the runtime | `inferctl run` and explicit endpoint/context flows pass end-to-end tests |
| 8 | [FTR-002](features/FTR-002-agent-api-contract.md) | Agent workloads require stronger contracts than basic chat | Native structured output plus Responses/tool compatibility gates |
| 9 | [FTR-003](features/FTR-003-tenant-safe-adapters-and-cache.md) | Efficiency is unsafe unless tenant identity scopes reusable state | Per-request adapters and tenant-salted cache isolation are measured |
| 10 | [FTR-004](features/FTR-004-slo-aware-serving.md) | Operators buy predictable service, not peak-token anecdotes | SLO admission, routing, and capacity signals validated under load |
| 11 | [FTR-005](features/FTR-005-distributed-lifecycle.md) | Distribution magnifies lifecycle errors and follows single-node rigor | Worker-loss and ownership matrix passes in multi-process CI |

## Critical-Path Gantt

```mermaid
gantt
  title InferFlux evidence-to-scale critical path
  dateFormat YYYY-MM-DD
  axisFormat %b %d
  section Release truth
  Green mainline baseline (TD-001)       :crit, t1, 2026-08-24, 10d
  Required evidence gates (TD-002)       :crit, t2, after t1, 10d
  Dependency/runtime currency (TD-006)   :crit, t6, after t2, 10d
  Context and sequence capacity (TD-007) :crit, t7, after t6, 10d
  Unified burst sampling parity (TD-008) :crit, t8, after t7, 10d
  Runtime proof and decision (ADR-0001/TD-003) :crit, t3, after t8, 10d
  section Product adoption
  Local transport and CLI (ADR-0002/FTR-001)   :p1, after t1, 20d
  Agent API contract (FTR-002)                 :crit, p2, after t3, 25d
  section Economy and scale
  Tenant-safe adapters/cache (FTR-003)         :crit, p3, after p2, 25d
  SLO-aware serving (FTR-004)                  :crit, p4, after p3, 20d
  Ownership and distributed faults (ADR-0004/FTR-005) :crit, p5, after p4, 30d
```

## Decision Gates

1. **Baseline gate:** no feature implementation merges while the portable build/test contract is red.
2. **Runtime gate:** continue native CUDA investment only when retained evidence meets ADR-0001 thresholds; otherwise use the compatibility provider as the primary data plane.
3. **Release gate:** require exact-SHA CUDA and ROCm evidence without exposing the persistent GPU host to pull-request code (ADR-0005).
4. **Adoption gate:** local autostart must preserve the same API/policy semantics as remote serving.
5. **Tenant gate:** reusable cache, adapter, session, and audit state must share one tenant identity boundary.
6. **Scale gate:** no distributed maturity claim precedes deterministic ownership and cleanup tests.

## Deliberate Deferrals

- Embedded inference inside `inferctl`; a managed daemon preserves one runtime contract.
- Broad TP/EP or prefill/decode claims before sequence/KV ownership is closed.
- New kernel families before the native-runtime proof-or-pivot decision.
- Desktop UI work before the five-minute local workflow and agent contracts are reliable.

## Artifact Map

- [Planning method and scorecard](planning/PRINCIPLES_AND_PRIORITIZATION.md)
- [Architecture decisions](adr/README.md)
- [Feature specifications](features/README.md)
- [Technical-debt register](technical-debt/README.md)
- [Issue import snapshots](issues/README.md)
