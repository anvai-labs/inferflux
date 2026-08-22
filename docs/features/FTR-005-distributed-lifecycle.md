# FTR-005: Distributed Lifecycle and Fault Maturity

Status: Proposed
Priority: P2
Owners: Distributed Runtime, Scheduler, QA
Dependencies: FTR-004, TD-005, ADR-0004

## User Outcome

Multi-process serving survives worker loss and transport degradation without
leaking KV state, duplicating generation, or claiming false readiness.

## MVP Scope

- Implement the sequence ownership epoch/lease contract across scheduler,
  workers, KV tickets, sessions, cancellation, and cleanup.
- Replace simulated distributed dispatch with a transport-neutral interface and
  one real multi-process execution path chosen from measured need.
- Make worker loss, stale messages, queue pressure, and cleanup debt visible to
  readiness, admission, admin pools, and metrics.
- Add deterministic retry versus fail-fast rules for each lifecycle state.

## Acceptance Evidence

- Fault matrix covers crash before/after enqueue, acknowledgement, KV commit,
  first token, cancellation, and lease transfer.
- Every test ends with balanced sequence, KV-page, ticket, and session accounting.
- Stale epochs cannot emit tokens or mutate current state.
- Required multi-process CI runs repeated fault injection without hangs.
- The selected topology demonstrates a measured SLO or cost improvement.

## Non-Goals

- Supporting every TP/EP/PD topology in the first release.
- Transparent continuation when correctness cannot be proven.
