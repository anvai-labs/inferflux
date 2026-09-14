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

## Bench Host Hardware


## Reference System: AI Server 1

| Component | Detail |
|-----------|--------|
| **CPU** | AMD Ryzen 9 7950X (AM5, 16C/32T) |
| **Motherboard** | Gigabyte X870E AORUS MASTER (BIOS F11, AMI) |
| **GPU 1** | AMD Radeon AI PRO R9700 (32GB VRAM, RDNA 4, gfx1201) |
| **GPU 2** | NVIDIA RTX 4000 Ada Generation (20GB VRAM, Ada Lovelace) |
| **OS** | Windows 11 + WSL2 (Ubuntu) |
| **ROCm** | 7.2 (Windows native + WSL) |
| **CUDA** | 12.4 (WSL via dxgkrnl paravirtualization) |

## PCIe Slot Layout

The X870E AORUS MASTER has three physical x16 slots with different electrical configurations:

| Slot | Physical | Electrical | Root Port | Current GPU |
|------|----------|-----------|-----------|-------------|
| **Slot 1** (top) | x16 | PCIe 5.0 x16 | CPU GPP0 | AMD R9700 |
| **Slot 2** (middle) | x16 | PCIe 4.0 x4 | Chipset GPP7 | NVIDIA RTX 4000 Ada |
| **Slot 3** (bottom) | x16 | PCIe 3.0 x4 (configurable to Gen4 in BIOS) | Chipset GPP8 | Empty |

**Important:** Slots 2 and 3 are chipset-connected (X870E chipset switch), NOT directly from the CPU. They are independent from Slot 1's CPU root port. Slot 2/3 bandwidth (PCIe 4.0 x4 = 8 GB/s) is sufficient for LLM inference since token generation is compute-bound, not PCIe-bandwidth-bound.

## Dual GPU Setup

### Required BIOS Settings

These settings **must** be enabled for dual GPU operation:

1. **Settings -> IO Ports -> Above 4G Decoding** -> **Enabled** (required for two large-VRAM GPUs: 32GB + 20GB)
2. **Settings -> IO Ports -> IOMMU** -> **Enabled**
3. **Settings -> IO Ports -> PCIEX4_1** -> **Enabled** (explicitly enable Slot 2)
4. **Settings -> Miscellaneous -> Slot 3 Gen** -> Set to **Gen4** if using Slot 3

### Ghost Device Issue (Windows)

When a GPU is moved between slots, Windows caches the old PCI device entry as a "ghost" device with **Status: Disconnected**. This prevents the GPU from being detected in the new slot. The ghost device blocks driver initialization because Windows tries to match the cached instance ID (which includes the old bus topology) instead of creating a new one.

**Symptoms:**
- Device shows `Status: Unknown` or `Status: Disconnected` in Device Manager
- `DEVPKEY_Device_IsPresent: False`
- `nvidia-smi` fails: "couldn't communicate with the NVIDIA driver"
- No new errors in Event Log (driver doesn't even attempt initialization)
- Fan may not spin (GPU stuck in pre-init power state)

**Fix — Remove ghost devices before moving GPUs between slots:**

```powershell
# Run from Admin PowerShell on Windows
# Remove the ghost GPU device (use actual Instance ID from Device Manager)
pnputil /remove-device "PCI\VEN_10DE&DEV_27B2&SUBSYS_181B10DE&REV_A1\4&D0BDF66&0&0009"

# Remove associated ghost audio devices
pnputil /remove-device "PCI\VEN_10DE&DEV_22BC&SUBSYS_181B10DE&REV_A1\4&D0BDF66&0&0109"

# Rescan for hardware changes
pnputil /scan-devices
```

**Prevention:** Always remove ghost device entries before physically moving a GPU to a different slot and rebooting.

### Verified Working Topology

```
PCIROOT(0)
  +-- GPP0 (CPU, PCIe 5.0 x16)
  |     +-- AMD PCIe Switch (1002:1478 upstream / 1002:1479 downstream)
  |           +-- AMD Radeon AI PRO R9700 (Bus 3, Gen5 x16)
  |
  +-- GPP1 (CPU, NVMe)
  |     +-- Samsung NVMe SSD
  |
  +-- GPP7 (Chipset, PCIe 4.0)
  |     +-- AMD X870E Chipset Switch (1022:43F4/43F5)
  |           +-- DP40 -> NVIDIA RTX 4000 Ada (Gen4 x4)
  |           +-- (other ports: USB, SATA, empty expansion)
  |
  +-- GPP8 (Chipset)
        +-- ASMedia USB4 Switch (Slot 3, currently empty)
```

### Diagnostic Commands

Check GPU status from WSL:
```bash
# List all display adapters with status
powershell.exe -Command "Get-PnpDevice -Class Display | Select-Object Status,FriendlyName | Format-Table -AutoSize"

# Check PCIe link for a specific device
powershell.exe -Command "Get-PnpDeviceProperty -InstanceId '<INSTANCE_ID>' -KeyName 'DEVPKEY_PciDevice_CurrentLinkWidth','DEVPKEY_PciDevice_CurrentLinkSpeed','DEVPKEY_Device_IsPresent' | Select-Object KeyName,Data | Format-List"

# Check NVIDIA GPU from elevated PowerShell
nvidia-smi

# List devices with problems
pnputil /enum-devices /problem

# List all display class devices with driver status
pnputil /enum-devices /class Display
```

Check GPU status from WSL (Linux side):
```bash
# AMD ROCm
rocm-smi

# NVIDIA CUDA (via WSL dxgkrnl)
nvidia-smi
```

## InferFlux Backend Mapping

| GPU | InferFlux Backend | Config |
|-----|------------------|--------|
| AMD R9700 (32GB) | `rocm` (llama.cpp HIP) | `config/server.rocm.qwen14b.yaml` |
| NVIDIA RTX 4000 Ada (20GB) | `cuda` / `inferflux_cuda` / `llama_cpp_cuda` | `config/server.cuda.yaml` |

Both GPUs can run InferFlux simultaneously on different ports for multi-model serving or A/B testing between backends.

## Bifurcation Note

The X870E AORUS MASTER does **not** support PCIe x16 -> x8/x8 bifurcation for dual GPU in the top slot. The second and third slots are hardwired x4 from the chipset. If M2B_CPU or M2C_CPU M.2 slots are populated, the top x16 GPU slot drops to x8, but those freed lanes go to M.2 storage, not to another GPU slot.

