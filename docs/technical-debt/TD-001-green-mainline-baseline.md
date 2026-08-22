# TD-001: Restore a Green, Portable Mainline Baseline

Status: Validated
Priority: P0
Owners: Build, QA
Dependencies: None

## Liability

The reviewed main revision builds `inferfluxd` and `inferctl` CPU-only but fails
to link `inferflux_tests`. Aggregate stub tests assume `./build/inferctl`, and a
duplicate exception handler produces a compiler warning. Existing docs still
claim a fully green CPU suite.

## Remediation

- Remove CPU test linkage to CUDA-only RTTI.
- Inject both server and CLI binary paths into every relevant CTest registration.
- Remove the unreachable duplicate handler.
- Build and test from a clean non-default build directory.
- Record model-backed and CUDA tests as unverified until their required assets run.

## Exit Criteria

- CPU-only `inferflux_tests`, `inferfluxd`, and `inferctl` compile from scratch.
- All model-free CTest entries pass with `--output-on-failure`.
- Documentation contract and planning-artifact checks pass.
- No known build warning is described as fixed while still reproducible.
- Reconciliation instructions preserve the dirty legacy checkout.

## Validation Evidence

Validated on 2026-08-21 from `/tmp/inferflux-critical-path-build` with a fresh
CPU-only Release configure. `inferflux_tests`, `inferfluxd`, and `inferctl`
built successfully; `ctest --output-on-failure --timeout 120` passed all 43
registered model-free suites. Model-backed and CUDA runtime gates were not run.
