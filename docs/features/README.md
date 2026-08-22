# Feature Specification Index

**Status:** Active register

| ID | Feature | Priority | Status | Dependencies |
|---|---|---|---|---|
| [FTR-001](FTR-001-managed-local-and-endpoint-cli.md) | Managed local and explicit-endpoint CLI | P0 | Proposed | TD-001, TD-004, ADR-0002 |
| [FTR-002](FTR-002-agent-api-contract.md) | Agent API and native structured-output contract | P0 | Proposed | TD-002, TD-003, ADR-0001 |
| [FTR-003](FTR-003-tenant-safe-adapters-and-cache.md) | Tenant-safe adapters and prefix cache | P1 | Proposed | FTR-002, ADR-0003 |
| [FTR-004](FTR-004-slo-aware-serving.md) | SLO-aware admission, routing, and capacity | P1 | Proposed | FTR-003, TD-003 |
| [FTR-005](FTR-005-distributed-lifecycle.md) | Distributed lifecycle and fault maturity | P2 | Proposed | FTR-004, TD-005, ADR-0004 |

Feature status is `Proposed`, `Discovery`, `In Progress`, `Validated`, `Released`,
or `Withdrawn`. A feature reaches `Validated` only when its listed evidence is retained.
