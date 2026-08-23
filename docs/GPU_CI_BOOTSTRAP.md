# Trusted Dual-GPU CI Bootstrap

**Status:** Operational; hosted checks protect `main`

```mermaid
flowchart LR
  A[Register restricted runner] --> B[Pin model asset]
  B --> C[CUDA gate]
  C --> D[ROCm gate]
  D --> E[Retain exact-SHA evidence]
  E --> F[Protect main hosted checks]
```

## Required Infrastructure

| Item | Contract |
|---|---|
| Runner | Dedicated Linux x64 runner labeled `self-hosted,linux,x64,gpu,cuda,rocm,dual-gpu` |
| NVIDIA | RTX 4000 Ada, compute capability 8.9, 20 GB |
| AMD | Radeon AI PRO R9700, `gfx1201`, 32 GB, ROCm 7.2 |
| Model | Read-only TinyLlama GGUF pinned by digest outside the repository |
| Toolchain | Pinned driver/CUDA/CMake versions recorded in job output |
| Isolation | Clean build and result directory per workflow run |

## Repository Configuration

1. Register one runner in `inferflux-gpu-trusted-staging`; do not register two
   agents that could contend for the same host.
2. Restrict the group to `gpu-gates.yml@refs/heads/main`. The workflow has no
   `pull_request` trigger, so public fork code never reaches the persistent host.
3. Set `INFERFLUX_GPU_MODEL_PATH` to a pinned runner-local GGUF model.
4. Run CUDA and then ROCm jobs; the shared runner serializes them.
5. Retain logs and raw evidence on success and failure.
6. Protect `main`; require hosted CPU checks on pull requests and require the
   dual-GPU workflow as post-merge/release evidence.

The runner group is restricted to
`gpu-gates.yml@refs/heads/main`; `INFERFLUX_ENABLE_DUAL_GPU_GATE=true` enables
the serial CUDA/ROCm jobs. This trust boundary follows
[ADR-0005](adr/ADR-0005-trusted-gpu-release-evidence.md).

## Protected Branch Contract

The following GitHub-hosted checks are required, strict, and enforced for
administrators. Force pushes and branch deletion are disabled.

| Required check | Contract |
|---|---|
| `Build & Test (ubuntu-latest)` | Complete model-free suite and contract assertions |
| `Build & Test (macos-latest, MPS)` | macOS runtime and unit coverage |
| `Build (macos-latest, MLX flag)` | MLX configuration compiles |
| `CUDA compile check (ubuntu-latest)` | CUDA sources compile on a hosted runner |
| `Build check (Vulkan)` | Vulkan configuration compiles |
| `GGUF & Quantization Tests (ubuntu-latest)` | Portable GGUF contracts pass |
| `Coverage (ubuntu-latest)` | Coverage build, tests, and upload pass |
| `clang-format check` | Touched C++ remains formatted |

`Dual-GPU gate result` is deliberately absent from pull-request requirements;
it is required by the release process for the exact promoted SHA.

## WSL Listener Lifecycle

The current WSL execution environment has no systemd bus, so the runner's
`svc.sh install/start` path is unavailable. Registration is persistent, but the
listener must be started again after each host restart:

```bash
cd /home/vsingh/actions-runner-inferflux-gpu
./run.sh
```

Keep that command in a durable host terminal. Confirm GitHub reports the runner
online before enabling a gate:

```bash
gh api orgs/anvai-labs/actions/runners/9054 \
  --jq '{name,status,busy,labels:[.labels[].name]}'
```

Do not rerun `config.sh` during ordinary startup. Recovery registration requires
a fresh token from `POST /orgs/anvai-labs/actions/runners/registration-token`
and these values: group `inferflux-gpu-trusted-staging`, name
`aiserver1-dual-gpu`, and labels
`gpu,cuda,rocm,dual-gpu,rtx4000-ada,radeon-ai-pro-r9700,compute-89,gfx1201,gpu-20gb,gpu-32gb`.
Never write the token to documentation, logs, or source control.

## Promotion Evidence

The runner, model variable, dual-GPU jobs, and protected hosted checks are
operational. Four consecutive trusted-main runs passed on August 22-23, 2026,
including retained CUDA and ROCm artifacts. TD-002 remains open until the
release checklist is exercised against an exact-SHA GPU result.
