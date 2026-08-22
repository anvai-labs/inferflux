# FTR-003: Tenant-Safe Adapters and Prefix Cache

Status: Proposed
Priority: P1
Owners: Runtime, Security, Model Lifecycle
Dependencies: FTR-002, ADR-0003

## User Outcome

Teams serve specialized models efficiently without loading one base model per
tenant or sharing reusable state across security boundaries.

## MVP Scope

- Load signed/allowlisted LoRA adapters and select one adapter per request.
- Add tenant, base-model revision, adapter revision, tokenizer/template version,
  and policy salt to prefix-cache keys.
- Enforce tenant-scoped adapter load/unload/use permissions and quotas.
- Export bounded hit, miss, eviction, adapter residency, and memory metrics.
- Make cache invalidation explicit on model, adapter, template, or policy change.

## Acceptance Evidence

- Cross-tenant collision and timing-oriented isolation tests show no state reuse.
- Adapter A/B requests interleave without output or KV contamination.
- Load/unload rollback is atomic under concurrent requests.
- Memory and latency overhead are measured for 1, 4, and 16 resident adapters.
- Audit events identify tenant, adapter revision, actor, and outcome.

## Non-Goals

- Training or fine-tuning adapters.
- Arbitrary untrusted runtime adapter paths.
- Cross-tenant cache sharing by default.
