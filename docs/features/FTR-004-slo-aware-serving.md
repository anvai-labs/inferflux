# FTR-004: SLO-Aware Admission, Routing, and Capacity

Status: Proposed
Priority: P1
Owners: Scheduler, Observability, Operations
Dependencies: FTR-003, TD-003

## User Outcome

Operators choose a latency/cost objective and receive predictable behavior under
load instead of relying on peak throughput.

## MVP Scope

- Measure TTFT, inter-token latency, queue time, tokens/sec, cache reuse,
  saturation, rejection reason, and model/adapter residency.
- Add service classes with token budgets, maximum queue delay, and fairness weights.
- Predict admission cost from prompt length, requested output, cache state, and backend envelope.
- Route only among providers that satisfy capability, tenant, and SLO policy.
- Expose capacity advice and overload state through metrics and admin APIs.

## Acceptance Evidence

- Mixed short/long prompt tests bound p95 TTFT and ITL for each service class.
- Overload rejects early with retryable, machine-readable reasons.
- Fairness tests prevent one tenant or long prefill from starving decode.
- Capacity predictions are calibrated against retained load-test results.
- No routing decision hides provider fallback or weakens tenant isolation.

## Non-Goals

- A Kubernetes autoscaler implementation; InferFlux supplies reliable signals.
- Machine-learned scheduling before the deterministic cost model is proven.
