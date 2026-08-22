# ADR-0003: Tenant Identity Is the Reusable-State Boundary

Status: Proposed
Date: 2026-08-21
Owners: Security, Scheduler, Runtime
Dependencies: TD-001

## Context

Prefix cache, session reuse, adapters, quotas, metrics, and audit events can
cross request boundaries. API-key scopes alone do not define whether two
requests may share model-derived state.

## Decision

Resolve an immutable `tenant_id` during authentication and carry it through the
request execution context. Include tenant identity and model revision in cache,
session, adapter, quota, and audit keys. Cross-tenant reuse is denied by default
and requires an explicit trusted sharing policy.

Admin operations identify both tenant and resource. Metrics use bounded labels;
high-cardinality tenant detail remains in structured audit/event storage.

## Consequences

- FTR-003 can add LoRA and prefix reuse without accidental data disclosure.
- Authentication adapters must map API keys/OIDC claims to stable tenant IDs.
- Cache-hit measurements may fall, but isolation is measurable and explainable.

## Rejected Alternatives

- Global cache by model/token hash: maximizes reuse but violates default isolation.
- API key as tenant ID: key rotation would fragment ownership and accounting.
- Per-request isolation: safe but eliminates legitimate tenant-local reuse.
