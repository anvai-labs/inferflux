# TD-005: Replace Distributed Ownership and Dispatch Stubs

Status: Proposed
Priority: P1
Owners: Distributed Runtime, Scheduler
Dependencies: TD-002, ADR-0004

## Liability

Distributed expert dispatch performs synchronization/all-gather but returns
local hidden states. Sequence eviction and decode-worker sessions lack a single
ownership authority, so topology claims exceed executable semantics.

## Remediation

- Add ownership epoch/lease types and lifecycle state-machine tests.
- Route cleanup through one idempotent authority across scheduler and backends.
- Define transport messages for ownership transfer and stale-epoch rejection.
- Select and implement one real distributed execution path only after profiling.
- Rename or gate simulated capabilities so APIs cannot advertise them as production.

## Exit Criteria

- No production capability flag resolves to a simulated dispatch path.
- Fault tests prove balanced resources after worker loss, timeout, cancellation,
  duplicate messages, and ownership transfer.
- Multi-process CI validates the selected topology repeatedly without hangs.
