# [P1-4] Resource-Economy Metrics + Autoscaling Signals

Priority: P1
Owner: Observability + Scheduler
Effort: 1.5 eng-weeks
Risk: Medium
Dependencies: P0-3
Labels: observability, scheduler, economics, metrics

## Problem
Current performance view is still too token-per-second centric; economy decisions need first-class efficiency signals.

## Scope Files
- `server/metrics/metrics.cpp`
- `server/metrics/metrics.h`
- `scheduler/scheduler.cpp`
- `runtime/execution/batch_executor.cpp`
- `tests/unit/test_metrics.cpp`
- `tests/integration/throughput_gate_contract_test.py`
- `docs/MONITORING.md`
- `docs/PRD.md`

## Test Plan
1. Add unit tests for new metrics registration and update semantics.
2. Extend throughput gate contract tests to parse and validate new metrics.
3. Add docs examples for alert thresholds and autoscaling interpretation.
4. Run unit and throughput contract suites.

## Acceptance Checklist
- [ ] Batch-packing loss and token-budget skip ratio metrics are exported.
- [ ] KV reuse efficiency metric is exported.
- [ ] Throughput gate can assert at least one new efficiency contract.
- [ ] Monitoring docs include concrete threshold guidance.
