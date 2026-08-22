# ADR-0004: Lease-Based Distributed Sequence Ownership

Status: Proposed
Date: 2026-08-21
Owners: Scheduler, Distributed Runtime
Dependencies: ADR-0003, TD-002

## Context

Transport tickets and health signals exist, but distributed expert routing is
simulated and sequence/KV cleanup lacks one authoritative owner. More topology
without ownership would multiply leaks, duplicates, and stale generations.

## Decision

Assign each sequence generation a monotonically increasing ownership epoch and
a renewable lease held by exactly one scheduler authority. KV pages, prefix
references, session handles, and worker tickets carry `{tenant, sequence,
epoch}`. Commit, transfer, cancellation, timeout, and cleanup are idempotent.

A worker may execute only the current epoch. Lease expiry makes work ineligible
until the authority either renews or transfers ownership. Recovery prioritizes
safe reclamation over transparent continuation.

## Consequences

- FTR-005 begins with lifecycle and fault tests, not TP/EP performance work.
- Duplicate/late messages become observable stale-epoch events.
- Stateful session reuse in decode-worker mode remains disabled until it honors leases.

## Rejected Alternatives

- Best-effort cleanup: cannot establish deterministic failure behavior.
- Worker-local ownership: cannot safely resolve worker loss or reassignment.
- Consensus system in phase one: disproportionate before single-authority proof.
