# ROCm Development on WSL2

Verified on: Ubuntu 24.04 (Noble), WSL2 kernel 5.15.167.4, ROCm 7.2, AMD Radeon AI PRO R9700 (gfx1201/RDNA 4).

## Prerequisites

### Windows Side

1. **Windows 11 23H2 or later** (required for AMD GPU passthrough to WSL2)
2. **AMD Adrenalin driver** with WSL2 support installed on Windows
   - Download from AMD's driver page for your GPU
   - The driver must include the WSL2/DirectX 12 compute bridge

### WSL2 Distro

Ubuntu 24.04 is the tested baseline. Other distros may work but package names will differ.

```bash
# Confirm you're on WSL2 (not WSL1)
uname -r
# Should show: 5.15.x-microsoft-standard-WSL2 or newer
```

## Step 1: Install ROCm 7.x

AMD provides a WSL2-specific HSA runtime package (`hsa-runtime-rocr4wsl-amdgpu`) that
allows GPU access without `/dev/kfd`. Standard ROCm Docker containers will NOT work
in WSL2 because they require `/dev/kfd`.

```bash
# Add the ROCm apt repository (Ubuntu 24.04 / Noble)
sudo mkdir -p /etc/apt/keyrings
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/7.2 noble main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
```

### Minimum packages (build InferFlux with ROCm)

```bash
sudo apt install -y \
  rocm-core \
  hip-dev \
  hip-runtime-amd \
  hipcc \
  hipblas \
  hipblas-dev \
  rocblas \
  rocblas-dev \
  hsa-rocr-dev \
  hsa-runtime-rocr4wsl-amdgpu \
  rocm-device-libs \
  rocminfo
```

### Full developer toolkit (profiling, all math libs)

```bash
sudo apt install -y \
  rocm \
  rocm-developer-tools \
  rocprofiler \
  rocprofiler-sdk \
  roctracer-dev \
  miopen-hip-dev \
  rccl-dev \
  rocfft-dev \
  rocsolver-dev \
  rocsparse-dev \
  hipfft-dev \
  hipsparse-dev \
  hipsolver-dev
```

### Environment setup

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

Then `source ~/.bashrc`.

## Step 2: Verify GPU Access

```bash
# Should list your AMD GPU (e.g., gfx1201 for R9700)
rocminfo | grep -E "Name:|Marketing Name:|Device Type:"

# Should report HIP version and device count
hipcc --version
hipconfig --full
```

Expected output (R9700 example):

```
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
```

### Quick HIP compile test

```bash
cat > /tmp/test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    hipGetDeviceCount(&count);
    printf("HIP devices: %d\n", count);
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        printf("  [%d] %s  arch=%s  mem=%zu MB\n",
               i, prop.name, prop.gcnArchName,
               prop.totalGlobalMem / (1024 * 1024));
    }
    return count > 0 ? 0 : 1;
}
EOF

hipcc /tmp/test_hip.cpp -o /tmp/test_hip && /tmp/test_hip
```

## Step 3: Build InferFlux with ROCm

```bash
cd /path/to/inferflux

# Init submodules (llama.cpp)
git submodule update --init --recursive

# Configure with ROCm enabled, other GPU backends off
cmake -S . -B build \
  -DENABLE_ROCM=ON \
  -DENABLE_CUDA=OFF \
  -DENABLE_MPS=OFF \
  -DENABLE_VULKAN=OFF

# Build (uses all cores)
cmake --build build -j$(nproc)
```

### Verify the build linked HIP

```bash
# Should show libamdhip64 linkage
ldd build/inferfluxd | grep -i hip

# Check CMake detected ROCm
grep -i "INFERFLUX_HAS_ROCM\|hip_FOUND" build/CMakeCache.txt
```

## Step 4: Run the Server

Use the ROCm config or create one:

```bash
# Copy and adjust the provided ROCm config
cp config/server.cuda.yaml config/server.rocm.yaml
```

Edit `config/server.rocm.yaml` — key fields:

```yaml
server:
  host: 0.0.0.0
  http_port: 8080
  enable_metrics: true

models:
  - id: tinyllama
    path: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    format: auto
    backend: rocm
    default: true

runtime:
  backend_priority: [rocm, cpu]
  rocm:
    enabled: true
    flash_attention:
      enabled: true   # FA2 supported on gfx9/10/11/12
    device_id: 0
```

Start the server:

```bash
./build/inferfluxd --config config/server.rocm.yaml
```

Test it:

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{"prompt": "Hello from ROCm!", "max_tokens": 20, "model": "tinyllama"}'
```

## Step 5: Run Tests

```bash
# Unit tests
ctest --test-dir build --output-on-failure

# Smoke tests (no model required)
bash scripts/smoke.sh gguf-native
bash scripts/smoke.sh backend-identity
```

## WSL2-Specific Notes

### No `/dev/kfd` — This Is Normal

WSL2 uses a paravirtualized GPU path through the Windows display driver, not the
native Linux `amdgpu` kernel module. The HSA runtime communicates with the GPU
through `/dev/dri/renderD128` and Windows' DirectX compute bridge. This means:

- `rocminfo` works (uses HSA runtime)
- `hipcc` compilation works
- HIP runtime API works (`hipGetDeviceCount`, kernel launches, etc.)
- `/dev/kfd` does NOT exist — this is expected
- `rocm-smi` may show errors about missing `amdgpu` module — ignore these
- `dmesg | grep amdgpu` will be empty — the kernel module is not loaded

### Docker Does NOT Work for ROCm in WSL2

Standard ROCm Docker containers (`rocm/dev-ubuntu-*`) require `/dev/kfd` for GPU
access. Since WSL2 doesn't expose `/dev/kfd`, containers cannot see the GPU.
Build and run natively in WSL2 instead.

### Performance Expectations

WSL2 adds a small overhead vs bare-metal Linux due to the paravirtualized GPU path.
For inference workloads this is typically <5%. Profiling tools (`rocprof`, `omniperf`)
may not work reliably — use bare-metal Linux for serious profiling.

### Memory

The WSL2 VM has a default memory limit (usually 50% of host RAM). If you're loading
large models, increase it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=32GB
swap=8GB
```

Then restart WSL: `wsl --shutdown` from PowerShell.

### ROCr runtime spin threads (2 CPUs burned at idle)

ROCm 7.x `libhsa-runtime64` starts two unnamed poller threads at `hipInit()`
(the AsyncEventsLoop) that busy-spin at 100% CPU each for the life of the
process. The upstream fix (rocm-systems#7898) is not in a released ROCm yet,
and no HSA environment knob disables the loop (`HSA_ENABLE_SLEEP`,
`HSA_ENABLE_INTERRUPT`, `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM`, and
`HSA_ENABLE_SDMA` were all tested without effect). On WSL2, `SCHED_IDLE` and
nice are additionally not enforced for service-cgroup threads, so CPU
affinity is the only effective mitigation.

With `INFERFLUX_ROCR_TAME=1`, `inferfluxd` confines these threads **at
creation**. The server binary defines `pthread_create`, so every thread
`libhsa-runtime64` starts goes through it; a thread whose start routine lies
in `libhsa-runtime64` is started through a small trampoline that, in the new
thread, pins it to one shared core and names it `rocr-hsa-N`.
Identification is exact (the start routine's owning library), so no other
thread — InferFlux's own or a third-party threadpool — is ever touched, and
there are no signals, `/proc` scans, or background threads. ROCr starts its
threads with an explicit all-CPU affinity, and the pin is applied after that,
inside the new thread; libhsa never changes a thread's affinity afterwards.

All `libhsa-runtime64` threads are pinned (about three per runtime
initialization: the two pollers and a short-lived helper), because they share
one ROCr-internal start routine. If ROCr itself narrows a thread's mask (for
example to the GPU's NUMA node) and the chosen core is outside it, the thread
is pinned to the highest allotted CPU inside its own mask instead. The tamer
only pins: it does not lower priority, because threads inherit their
creator's nice value and an unprivileged thread cannot raise it back, and
`SCHED_IDLE` can starve the HSA signal-handler work these threads run on
native Linux. A thread that a tamed thread starts (and that is not itself a
ROCr thread) gets the CPU mask it would have inherited without the tamer
(the tamed thread's mask before its pin) unless it asked for an explicit
affinity, so the pin never leaks. GPU sync latency is unaffected.

The core is chosen from the server's own affinity allotment (its
`taskset`/cgroup mask, read at process start): a valid
`INFERFLUX_ROCR_TAME_CORE` inside it, else its highest CPU. A server allotted
a single CPU pins nothing (two spinners must not share its only CPU).

Knobs (see `docs/CONFIG_REFERENCE.md` section 10):

- `INFERFLUX_ROCR_TAME=1` enables the tamer. It is **off by default**, since
  it changes the scheduling of threads InferFlux does not own; enable it on
  hosts that show the spin (e.g. WSL2 + gfx1201).
- `INFERFLUX_ROCR_TAME_CORE=<n>` pins to CPU `n` (default: highest CPU in the
  server's affinity mask).

Verify: the startup log line from `rocr_tamer` reports the core and that the
interposer is active; `top -H -p $(pidof inferfluxd)` shows the `rocr-hsa-N`
pollers sharing a single core (~100% combined instead of ~200%); and
`inferflux_runtime_spin_threads_confined_total` in `/metrics` counts the
tamed threads (`inferflux_runtime_spin_threads_unconfined_total` counts ROCr
threads whose own CPU mask had no allotted CPU, which are left alone).

The interposer is linked only into ROCm-enabled (`ENABLE_ROCM=ON`) server
builds; CUDA-only and CPU builds never route thread creation through it.
Enable the tamer on ROCm hosts only: elsewhere every thread creation pays one
`dladdr` lookup for nothing. Helper threads that a tamed thread starts get
back its original name and CPU mask.

Limits: a cgroup cpuset later shrunk to exclude the chosen core moves the
threads off it; a future ROCr that starts threads via `std::thread`, or that
is loaded with `RTLD_DEEPBIND`, would not be recognized (nothing is tamed,
nothing breaks), and neither would threads of a second, separately loaded
copy of `libhsa-runtime64`; the allotment is read after shared-library constructors, so
`OMP_PROC_BIND` narrowing the main thread first can shrink it (fails safe);
the cached ROCr address range assumes libhsa is never unloaded (InferFlux
never unloads its backends); sanitizer builds compile the interposer out
(the startup log says so). Linux threads inherit their creator's name, so
with the tamer off, spinning ROCr threads started from a first-party thread
show that thread's `if-*` name in `top -H`; the counters are process-wide
and not reset by `MetricsRegistry::Reset()`.

## Supported AMD GPUs

The InferFlux ROCm backend supports FlashAttention-2 on these architectures:

| Architecture | GPUs | FA2 Support |
|-------------|------|-------------|
| GFX9 (CDNA/Vega) | MI200, MI250X, MI300X | Yes |
| GFX10 (RDNA 2) | RX 6000 series | Yes |
| GFX11 (RDNA 3) | RX 7000 series | Yes |
| GFX12 (RDNA 4) | RX 9000 series, Radeon AI PRO | Yes |

For GFX12 (RDNA 4) GPUs, you may need to set `HSA_OVERRIDE_GFX_VERSION` if the
ROCm version doesn't yet have native gfx1201 support:

```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.0
```

## Troubleshooting

### `rocminfo` shows "ROCk module is NOT loaded"

This warning appears because the `amdgpu` kernel module isn't loaded in WSL2.
If `rocminfo` still lists your GPU after the warning, it's working via the WSL2
HSA bridge. If no GPU appears, check that:

1. Your Windows AMD driver supports WSL2
2. The `hsa-runtime-rocr4wsl-amdgpu` package is installed
3. You're on Windows 11 23H2+

### CMake says "ENABLE_ROCM=ON but ROCm toolkit not found"

```bash
# Ensure ROCm is on your PATH
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH

# Verify hip cmake config exists
ls /opt/rocm/lib/cmake/hip/

# Clean and reconfigure
rm -rf build
cmake -S . -B build -DENABLE_ROCM=ON -DENABLE_CUDA=OFF
```

### HIP compilation errors about unsupported architecture

For newer GPUs (gfx1201, etc.) that your ROCm version doesn't fully support:

```bash
# Override the target architecture
export HIP_ARCHS="gfx1200"
# Or set the HSA override
export HSA_OVERRIDE_GFX_VERSION=12.0.0
```

### Build fails linking against hipBLAS/rocBLAS

```bash
sudo apt install -y hipblas-dev rocblas-dev
# Verify libraries exist
ls /opt/rocm/lib/libhipblas.so /opt/rocm/lib/librocblas.so
```

### Poor performance or hangs

```bash
# Check GPU utilization
rocm-smi  # May error in WSL2, try:
cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null

# Reduce context size if OOM
# In server config: runtime.rocm.context_size: 2048
```
