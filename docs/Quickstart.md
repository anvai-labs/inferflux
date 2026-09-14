# InferFlux Quickstart (OSS)

```mermaid
flowchart LR
    A[Build] --> B[Run inferfluxd]
    B --> C[Call OpenAI API]
    C --> D["Inspect models/admin"]
```

## 0) Prerequisites

- Linux/macOS shell
- CMake >= 3.22
- C++17 toolchain
- A local model path (GGUF or other configured format)

## 1) Build

```bash
./scripts/build.sh
```

`scripts/build.sh` prefers Ninja when available and uses `nvcc` with GCC (`/usr/bin/g++`) as CUDA host compiler by default.
For arch-specific CUDA builds (example Ada RTX 4000), set:

```bash
INFERFLUX_CUDA_ARCHS=89 ./scripts/build.sh
```

Optional CUDA kernel tuning toggles passed through to CMake:
- `INFERFLUX_CUDA_USE_FAST_MATH=ON` (throughput vs precision tradeoff)
- `INFERFLUX_CUDA_DEVICE_LTO=ON` (advanced profile; requires explicit `code=lto_<arch>` gencode wiring)

CPU-only fallback build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF -DENABLE_MPS=OFF
cmake --build build -j$(nproc)
```

## 2) Start Server

Use default config and override model path via env:

```bash
INFERFLUX_MODEL_PATH=models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  ./build/inferfluxd --config config/server.yaml
```

GPU-focused config example:

```bash
./build/inferfluxd --config config/server.cuda.yaml
```

## 3) Verify Health + Metrics

```bash
curl -s http://127.0.0.1:8080/livez
curl -s http://127.0.0.1:8080/readyz
curl -s http://127.0.0.1:8080/metrics | head -40
```

## 4) Send First Inference Request

### Option A: `inferctl`

```bash
./build/inferctl completion \
  --prompt "Give me 3 reasons to use continuous batching" \
  --max-tokens 64 \
  --api-key dev-key-123
```

### Option B: OpenAI-compatible HTTP

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-key-123' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Hello from OSS quickstart"}],
    "max_tokens": 64,
    "stream": false
  }'
```

## 5) Inspect Model Inventory

```bash
./build/inferctl models --api-key dev-key-123
./build/inferctl models --json --api-key dev-key-123
./build/inferctl models --id llama3-8b --json --api-key dev-key-123
```

## 6) Admin Examples

```bash
./build/inferctl admin models --list --api-key dev-key-123
./build/inferctl admin pools --get --api-key dev-key-123
./build/inferctl admin routing --get --api-key dev-key-123
```

## 7) Endpoint Matrix

| Category | Endpoint | Method | Auth scope |
|---|---|---|---|
| Health | `/livez`, `/readyz`, `/healthz` | `GET` | none |
| Metrics | `/metrics` | `GET` | none |
| OpenAI | `/v1/completions`, `/v1/chat/completions` | `POST` | `generate` |
| OpenAI | `/v1/models`, `/v1/models/{id}` | `GET` | `read` |
| OpenAI | `/v1/embeddings` | `POST` | `read` |
| Admin | `/v1/admin/*` | mixed | `admin` |

Full contract: [API Surface](API_SURFACE.md)

## 8) Next Paths

- Operator path: [Admin Guide](AdminGuide.md)
- Config path: [CONFIG_REFERENCE](CONFIG_REFERENCE.md)
- Contributor path: [Developer Guide](DeveloperGuide.md)
- Runtime internals: [Architecture](Architecture.md)

## Installing a Release (Packages)

Instead of building from source, install a packaged release:

| Platform | CPack generator | Primary outputs | Local install/test command |
|---|---|---|---|
| Linux Debian/Ubuntu | `DEB` | `.deb` | `sudo apt install ./inferflux-*.deb` |
| Linux RHEL/Fedora | `RPM` | `.rpm` | `sudo rpm -i inferflux-*.rpm` |
| Linux/macOS generic | `TGZ` | `.tar.gz` | extract + run `inferfluxd` |
| macOS Installer | `productbuild` | `.pkg` | open package installer |
| macOS App bundle | `DragNDrop` | `.dmg` | mount DMG and copy app |
| Windows | `WIX` | `.msi` | run MSI installer |

## 3) Packaging Commands

```bash
# Linux
cpack --config build/CPackConfig.cmake -G DEB
cpack --config build/CPackConfig.cmake -G RPM
cpack --config build/CPackConfig.cmake -G TGZ

# macOS
cpack --config build/CPackConfig.cmake -G productbuild
cpack --config build/CPackConfig.cmake -G DragNDrop

# Windows (PowerShell with WiX on PATH)
cpack --config build/CPackConfig.cmake -G WIX
```

## 4) Verification Contract

| Check | Command |
|---|---|
| Binary starts | `inferfluxd --help` |
| CLI works | `inferctl --help` |
| Health endpoint | `curl -s http://127.0.0.1:8080/livez` |
| Model listing | `./build/inferctl models --api-key dev-key-123` |

## 5) Release Integration

- Homebrew template: `installers/homebrew/inferflux.rb`
- Winget template: `installers/winget/inferencial.inferflux.yaml`
- Release workflow: [ReleaseProcess](ReleaseProcess.md)

## 6) Notes

- Rebuild with `-DENABLE_MLX=ON` if MLX backend support is needed in macOS artifacts.
- Docker remains optional (`docker/`) and is not required for package-native setup.

## 7) Platform Backend Notes

| Platform | Note | Reference |
|---|---|---|
| NVIDIA CUDA | use CUDA profile and throughput gate for behavior checks | [MONITORING](MONITORING.md) |
| AMD ROCm | WSL2 has hard limitations; native Linux or cloud is preferred | [ROCM_INSTALLATION_GUIDE_WSL](ROCM_INSTALLATION_GUIDE_WSL.md) |
| Apple Silicon | enable MPS/MLX build flags as needed | [CONFIG_REFERENCE](CONFIG_REFERENCE.md), [Architecture](Architecture.md) |


Packaging commands and the build-once
flow: see the [Installer history](ARCHIVE_INDEX.md) (consolidated here from
Installer.md).
