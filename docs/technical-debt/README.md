# Technical-Debt Register

**Status:** Active register

| ID | Debt | Priority | Status | Dependencies |
|---|---|---|---|---|
| [TD-001](TD-001-green-mainline-baseline.md) | Restore a green, portable mainline baseline | P0 | Validated | None |
| [TD-002](TD-002-required-evidence-gates.md) | Make release evidence required and coherent | P0 | In Progress | TD-001 |
| [TD-003](TD-003-runtime-evidence-drift.md) | Eliminate benchmark and product-claim drift | P0 | Proposed | TD-001, ADR-0001 |
| [TD-004](TD-004-cli-transport-seams.md) | Extract CLI transport, context, and daemon seams | P1 | Proposed | TD-001, ADR-0002 |
| [TD-005](TD-005-distributed-ownership-stubs.md) | Replace distributed ownership and dispatch stubs | P1 | Proposed | TD-002, ADR-0004 |
| [TD-006](TD-006-dependency-runtime-currency.md) | Restore dependency and accelerator runtime currency | P0 | In Progress | TD-002 |
| [TD-007](TD-007-context-sequence-capacity.md) | Make context capacity and completion lifecycle explicit | P0 | In Progress | TD-002 |
| [TD-008](TD-008-unified-burst-sampling-parity.md) | Restore unified CUDA burst sampling parity | P1 | Proposed | TD-002, TD-007 |

Statuses are `Proposed`, `In Progress`, `Blocked`, `Validated`, `Closed`, or
`Superseded`. Closing debt requires the evidence listed in its exit criteria.
