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
- Add one required CUDA behavioral job with pinned runner/toolchain/model assets.
- Separate correctness/reliability gates from performance trend reporting.
- Upload machine-readable configs, revisions, raw results, and failure artifacts.
- Document an explicit, time-bounded release exception process.

## Exit Criteria

- Branch protection names stable CPU and CUDA checks as required.
- CI configuration contains no contradictory skip/run contract.
- A deliberately broken CPU contract and CUDA provider assertion both block CI.
- Release checklist links retained artifacts and their provenance.

## Current Evidence and Remaining Work

The required CPU job now builds and runs the full model-free suite. The
organization runner `aiserver1-dual-gpu` is registered with CUDA and ROCm labels,
but remains staged with public-repository access and gate variables disabled.
Local CUDA CTest and model-backed native-provider evidence pass. ROCm now builds
real HIP kernels after correcting `GGML_HIP`; its model-backed timeout remains
open. Promotion and branch protection wait for the trusted workflow on `main`
and three consecutive dual-GPU runs described in
[GPU CI Bootstrap](../GPU_CI_BOOTSTRAP.md).
