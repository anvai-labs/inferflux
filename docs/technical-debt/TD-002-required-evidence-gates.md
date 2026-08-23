# TD-002: Make Release Evidence Required and Coherent

Status: In Progress
Priority: P0
Owners: QA, Release Engineering
Dependencies: TD-001

## Liability

CI comments say CPU tests are skipped while later steps invoke selected tests;
the main unit executable is not built in that job. GPU behavior is optional or
environment-dependent, allowing provider regressions to outlive the change.

## Remediation

- Build and run the complete model-free suite in the required CPU job.
- Add trusted CUDA and ROCm behavioral jobs with pinned runner/toolchain/model assets.
- Separate correctness/reliability gates from performance trend reporting.
- Upload machine-readable configs, revisions, raw results, and failure artifacts.
- Document an explicit, time-bounded release exception process.

## Exit Criteria

- Branch protection names stable hosted checks as required.
- CI configuration contains no contradictory skip/run contract.
- A deliberately broken CPU contract and CUDA provider assertion both block CI.
- Release checklist requires exact-SHA CUDA and ROCm artifacts and provenance.

## Current Evidence and Remaining Work

The required CPU job builds and runs the full model-free suite. The restricted
`aiserver1-dual-gpu` runner executes serial CUDA and ROCm model-backed gates only
for trusted `main` workflow code. Four consecutive dual-GPU runs passed on
August 22-23, 2026, with exact-SHA artifacts. Node 24-native action upgrades are
also validated on hosted and self-hosted runners.

Remaining work is enforcement: `main` has no branch protection as of August 23,
2026. Require stable hosted checks, then exercise the exact-SHA release checklist.
ADR-0005 intentionally makes persistent-GPU evidence a post-merge release gate
rather than exposing the host to pull-request code.
