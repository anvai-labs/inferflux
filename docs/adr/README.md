# Architecture Decision Record Index

**Status:** Active register

| ID | Decision | Status | Dependencies |
|---|---|---|---|
| [ADR-0001](ADR-0001-evidence-gated-runtime-portfolio.md) | Evidence-gated native runtime portfolio | Proposed | TD-001, TD-006, TD-007 |
| [ADR-0002](ADR-0002-server-first-managed-local-transport.md) | Server-first CLI with managed local transport | Accepted | TD-001 |
| [ADR-0003](ADR-0003-tenant-resource-boundary.md) | Tenant identity as the reusable-state boundary | Proposed | TD-001 |
| [ADR-0004](ADR-0004-distributed-sequence-ownership.md) | Lease-based distributed sequence ownership | Proposed | ADR-0003, TD-002 |
| [ADR-0005](ADR-0005-trusted-gpu-release-evidence.md) | Trusted GPU evidence runs after merge | Accepted | TD-002 |

Statuses are `Proposed`, `Accepted`, `Rejected`, `Superseded`, or `Deprecated`.
Dependent implementation may prototype a proposed ADR, but release behavior
must not depend on it until the co-design checkpoint records acceptance.
