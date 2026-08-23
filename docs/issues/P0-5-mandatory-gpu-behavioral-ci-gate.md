# [P0-5] Trusted GPU Behavioral Release Gate

Priority: P0
Owner: QA + Runtime
Effort: 1 eng-week (+ runner infra)
Risk: Medium
Dependencies: P0-3
Labels: ci, qa, throughput, release-gate

## Problem
GPU validation is partially advisory and environment-dependent, reducing confidence in throughput and native-path regressions.

## Scope Files
- `.github/workflows/ci.yml`
- `scripts/run_throughput_gate.py`
- `docs/ReleaseProcess.md`
- `docs/DeveloperGuide.md`

## Test Plan
1. Require hosted portable checks for pull requests and protect `main`.
2. Ensure the lane executes throughput gate plus focused capability and contract assertions.
3. Verify CI artifact publication for perf diagnostics on pass or fail.
4. Validate branch protection docs reference the required check name.
5. Require exact-revision CUDA and ROCm evidence before release promotion.

## Acceptance Checklist
- [x] Persistent GPU runner is restricted to trusted `main` workflow code.
- [x] CUDA job runs throughput gate and fails on threshold breach.
- [x] CUDA and ROCm logs are consistently uploaded by exact revision.
- [x] Release docs define GPU gate expectations and exception handling.
- [x] CI names are stable for repository protection and release automation.
- [ ] Hosted checks protect `main` from unvalidated merges.

Pre-merge execution on the persistent GPU host was superseded by
[ADR-0005](../adr/ADR-0005-trusted-gpu-release-evidence.md). A disposable,
isolated runner is required before untrusted pull-request code can run on GPU.
