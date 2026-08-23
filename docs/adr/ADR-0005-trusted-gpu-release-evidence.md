# ADR-0005: Trusted GPU Evidence Runs After Merge

Status: Accepted
Date: 2026-08-23
Owners: Runtime, QA, Release Engineering
Dependencies: TD-002

## Context

InferFlux needs real CUDA and ROCm behavioral evidence, but its persistent
self-hosted runner has accelerator access and runner-local model assets. Running
arbitrary pull-request revisions there would give untrusted code access to that
host. Hosted runners can safely validate portable behavior and compile CUDA,
but cannot prove model-backed behavior on the maintained devices.

## Decision

Keep pull-request gates on GitHub-hosted runners. Restrict the dual-GPU runner
group to `.github/workflows/gpu-gates.yml@refs/heads/main`, and trigger that
workflow only for trusted `main` revisions or an authorized manual run of
`main`. Run CUDA before ROCm because both devices share one host.

Treat `Dual-GPU gate result` and its retained artifacts as mandatory release
evidence for the exact commit being promoted. Branch protection requires the
hosted portable checks; it does not send pull-request code to the GPU host.

## Consequences

- A merge can precede GPU behavioral evidence, but a release cannot.
- A failed main gate is a release blocker and must be fixed or reverted.
- Strict pre-merge GPU execution requires a disposable, isolated runner; it
  must not weaken the trusted runner-group restriction.
- CUDA and ROCm logs are retained under revision-qualified artifact names.

## Rejected Alternatives

- Run `pull_request` code on the persistent GPU host: unacceptable trust
  boundary for a public repository.
- Make GPU validation advisory: permits regressions to become release claims.
- Register one runner per device on the same host: risks resource contention
  without increasing trustworthy coverage.
