# TD-006: Restore Dependency and Accelerator Runtime Currency

Status: In Progress
Priority: P0
Owners: Runtime, Build, Release Engineering
Dependencies: TD-002

## Liability

Accelerator support depends on fast-moving toolchains and llama.cpp APIs. The
pinned llama.cpp commit is from 2026-03-05 and is 2,353 commits behind upstream
v0.2.0 at this snapshot. InferFlux still used the retired `LLAMA_HIP` CMake flag,
so a nominal ROCm build silently compiled CPU-only. Other pins also drift:
Catch2 3.7.1 versus 3.15.3, nlohmann/json 3.11.3 versus 3.12.0, MLX-C 0.5.0
versus 0.6.0, and host ROCm 7.2 versus 7.14. yaml-cpp is intentionally pinned
11 commits ahead of its 0.9.0 release and is not presently stale.

## Remediation

- Maintain a machine-readable dependency manifest with version, source, digest,
  supported toolchains, and upgrade owner.
- Upgrade llama.cpp using the isolated
  [v0.2.0 compatibility plan](../planning/LLAMA_CPP_V0_2_0_UPGRADE.md); adapt
  APIs and compare CPU, CUDA `sm_89`, and ROCm `gfx1201` behavior against the
  retained baseline.
- Upgrade ROCm/driver pairs only from AMD-supported WSL combinations; keep CUDA
  and ROCm toolchain changes separate from library changes.
- Update Catch2, nlohmann/json, and MLX-C in independent low-risk changes.
- Treat OpenSSL as a supported system dependency and test the packaged minimum
  plus current release rather than vendoring it.

## Exit Criteria

- Clean CPU, CUDA, and ROCm builds identify the intended accelerator backend.
- Model-backed CUDA and ROCm gates pass without fallback on documented hardware.
- Dependency versions and digests are emitted in CI evidence and SBOM output.
- Each upgrade has rollback instructions and no unexplained correctness,
  throughput, memory, or startup regression.

## Current Evidence

The ROCm flag is corrected to `GGML_HIP`, and a clean `gfx1201` build now compiles
the HIP backend. CPU and ROCm each pass 43 model-free tests; CUDA passes 53.
Model-backed evidence passes on native CUDA (32/32 requests) and ROCm (two
independent 16/16-request runs) without provider fallback. The scheduler
lost-wakeup cause of the prior ROCm timeout is fixed under TD-007.

`config/dependencies.json` is now the machine-readable evidence manifest for
six direct dependencies. It records immutable refs/digests, vendored artifact
hashes, supported toolchains, ownership, upgrade targets, and rollback. CI
checks the manifest against the llama.cpp gitlink, CMake fetch pins, and local
artifact hashes; CycloneDX and SPDX generation consumes the same data. The
llama.cpp `v0.2.0` upgrade remains an isolated follow-up because its 2,353-commit
API delta must not be mixed with the baseline and lifecycle corrections.
