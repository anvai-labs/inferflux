# Hardware Setup Guide

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
