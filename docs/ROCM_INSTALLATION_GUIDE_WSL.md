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
(the tamed thread's mask before its pin) whenever it is still on exactly
that pin, so the pin never leaks (barring an allocation failure during
that thread's creation, when it is created untouched). GPU sync latency is
unaffected; measured tail latency improves (1-token requests, p99 ~7.7 ms
on vs 11-13 ms off), because the spinners no longer land on request-path
cores.

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
threads that could not be pinned: no allotted CPU in their own mask, or
the kernel refused the pin; they are left alone).

The interposer is linked only into server builds where CMake actually found
ROCm (`ENABLE_ROCM=ON` and the toolkit present; the configure log says
"ROCm/HIP found"); CUDA-only and CPU builds never route thread creation through it.
With the tamer off it passes every creation straight through. When on, each
creation is classified with glibc's lock-free `_dl_find_object` (glibc
2.35+; older glibc falls back to `dladdr`, which takes the loader lock).
Helper threads that a tamed thread starts get
back its original name and CPU mask.

Limits: a cgroup cpuset later shrunk to exclude the chosen core moves the
threads off it; a future ROCr that starts threads via `std::thread`, or that
is loaded with `RTLD_DEEPBIND`, would not be recognized (nothing is tamed,
nothing breaks); the allotment is read after shared-library constructors,
so `OMP_PROC_BIND` narrowing the main thread first can shrink it (fails
safe); a process forked from a ROCr thread starts on its pin; thread
creation from a tamed thread allocates a small block, which an allocator
that starts threads under its own lock could deadlock on; sanitizer builds compile the interposer out
(the startup log says so). Linux threads inherit their creator's name, so
with the tamer off, spinning ROCr threads started from a first-party thread
show that thread's `if-*` name in `top -H`; the counters are process-wide
and not reset by `MetricsRegistry::Reset()`.

### Patched ROCr runtime (near-zero idle CPU)

Confining the spin threads to one core still leaves about one core busy,
because they never sleep. The real fix is upstream
([rocm-systems#7898](https://github.com/ROCm/rocm-systems/pull/7898)), which
makes the polling loop nap with a 20 µs → 2 ms backoff. Until a released ROCm
contains it, build ROCr 7.2.0 with the fix backported and point
`inferfluxd` at it:

```bash
scripts/rocm/build_patched_rocr.sh --out ~/rocr-patched   # ~1 min
```

The script clones `ROCm/rocm-systems` at `rocm-7.2.0` (`projects/rocr-runtime`
only), applies `scripts/rocm/patches/rocr-7.2.0-async-events-poll-backoff.patch`,
builds `libhsa-runtime64` in a clean environment (host code with the
system C++ compiler, which `PATCHED_ROCR.json` records; ROCm's LLVM only
for the GPU blit and trap-handler code objects), and adds AMD's prebuilt WSL
thunk `librocdxg` (from the `ROCm/librocdxg` release package,
checksum-pinned). It installs nothing system-wide; the output directory
holds the two libraries and `PATCHED_ROCR.json` (source commit, compiler,
patch and library checksums).

The script requires the host ROCm to be exactly 7.2.0 (the backport's base)
and refuses to stage a library that carries a RUNPATH or has unresolved
dependencies. Unlike AMD's WSL build, the upstream runtime links `libdrm`,
`libdrm_amdgpu`, `libnuma` and `librocprofiler-register`; on a ROCm 7.2 WSL
install these resolve through the loader configuration (`/opt/amdgpu/lib`,
`/opt/rocm/lib`, the system `libnuma1`). Deploy on the host it was built on,
or one with the same ROCm and `amdgpu` packages.

Deploy by copying the output into a directory on the server's
`LD_LIBRARY_PATH` and setting `HSA_ENABLE_DXG_DETECTION=1`, which makes
upstream ROCr load `librocdxg`. That directory must come **before**
`/opt/rocm/lib` in `LD_LIBRARY_PATH` (the setup step above prepends
`/opt/rocm/lib` for interactive shells; a service should set its own, e.g.
`Environment=LD_LIBRARY_PATH=<release>/runtime/bin` in the unit), or the
stock runtime shadows the patched one. Back up anything already there, once:

```bash
D=<release>/runtime/bin
(  # Subshell: `exit` stops the step without closing your terminal. Each
   # command checks itself (set -e is ignored in a `( ... ) ||` subshell).
  B="$D/pre-patched-rocr"
  if [ ! -e "$B" ]; then  # Taken once; a redeploy never overwrites it.
    T="$(mktemp -d "$D/.pre-patched-rocr.XXXXXX")" || exit 1
    for f in "$D"/libhsa-runtime64.so* "$D"/librocdxg.so*; do
      [ -e "$f" ] || continue
      cp -a "$f" "$T/" || exit 1
    done
    mv "$T" "$B" || exit 1  # Only a complete backup gets the final name.
  fi
  cp -a ~/rocr-patched/lib* ~/rocr-patched/PATCHED_ROCR.json "$D/" || exit 1
) || echo "deploy step failed (if the backup failed, nothing was replaced)"
# systemd unit: Environment=HSA_ENABLE_DXG_DETECTION=1
```

`libamdhip64` finds `libhsa-runtime64.so.1` through `LD_LIBRARY_PATH` ahead
of `/opt/rocm/lib`, so only that process uses the patched runtime. The
backoff means events handled by ROCr's async-event threads (HIP host
callbacks, async signal handlers, and the registration of a new handler,
which is signalled through the same polled loop) can be observed up to
~2 ms late after a long idle wait, i.e. on the first one after a pause;
while busy the nap restarts at its floor (20 µs requested, ~70-100 µs in
practice with Linux timer slack). Synchronous waits (`hipStreamSynchronize`)
do not go through these threads.

Verify: `grep -E 'libhsa|librocdxg' /proc/$(pidof inferfluxd)/maps` shows
both from the release directory, and idle CPU drops to about 1%
(`top -H` shows no thread above a few percent). Measured on an R9700
(gfx1201) with qwen2.5-0.5b: idle 200% → 1.1%; greedy outputs identical to
the stock runtime; c=8 and c=1 throughput and per-request latency unchanged
(1-token request p50 7.1 ms on both).

The backup is taken once per `runtime/bin` and is that directory's rollback
target (a new release has its own `runtime/bin`, so it starts without one);
after upgrading ROCm, delete `pre-patched-rocr/` there before deploying
again, since it would restore 7.2.0-era libraries.

Every process that loads the patched runtime needs
`HSA_ENABLE_DXG_DETECTION=1`: without it upstream ROCr does not load
`librocdxg`, finds no KFD device on WSL2, and the ROCm backend is
unavailable (the server falls back or fails to start the ROCm model). This
applies to manual `inferfluxd` runs and tests that put the same directory on
`LD_LIBRARY_PATH`, not only the service.

`INFERFLUX_ROCR_TAME=1` can stay on alongside it: the `libhsa-runtime64`
threads it pins now sleep, and it bounds the cost if the patched library is
ever missing. (Threads that the separate `librocdxg` might start are not
recognized by the tamer; none are busy at idle in the measurements above.)
Rollback: remove the copied libraries and `PATCHED_ROCR.json` (restoring
anything saved in `pre-patched-rocr/`) and the `HSA_ENABLE_DXG_DETECTION`
line, then restart.
Retire the patch once a ROCm release includes #7898 (check the release notes
for rocm-systems#7898 / librocdxg#60).

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
