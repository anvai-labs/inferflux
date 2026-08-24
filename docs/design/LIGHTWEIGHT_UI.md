# Lightweight Distribution UI

**Status:** Roadmap (P2)
**Target:** 2027-H1

## Goal

Ship a lightweight, self-contained web UI bundled with `inferfluxd` for model serving management, backend monitoring, and multi-GPU orchestration. Zero external dependencies — single binary distribution.

## Design Principles

1. **Embedded** — Static assets compiled into `inferfluxd` binary (no separate frontend deploy)
2. **Zero-config** — Enabled by default on `/ui`, disabled via `ui.enabled: false`
3. **Read-heavy** — Dashboard and monitoring are primary; mutation operations require confirmation
4. **Multi-GPU native** — First-class support for heterogeneous GPU setups (AMD + NVIDIA)

## Pages

### Dashboard (`/ui/`)
- Server status, uptime, active backends
- Per-GPU utilization bars (VRAM, compute, temperature)
- Request throughput chart (tok/s over time)
- Active sequences / batch occupancy

### Models (`/ui/models`)
- Loaded models with backend assignment
- Load/unload controls
- Format detection (GGUF, safetensors, HF)
- Model file browser (local paths + HuggingFace search)

### Backends (`/ui/backends`)
- All registered backends with capabilities
- Per-backend throughput, latency percentiles
- Provider identity (native vs llama.cpp fallback)
- Backend priority configuration
- GPU device info (PCIe link, VRAM, driver version)

### Multi-GPU (`/ui/gpus`)
- Device topology visualization (PCIe tree)
- Per-GPU model assignment
- Memory allocation breakdown
- KV cache utilization per device
- Health signals (temperature, power draw, ECC errors)

### Monitoring (`/ui/metrics`)
- Prometheus metrics browser (existing `/metrics` endpoint)
- Grafana-lite: configurable charts for key metrics
- Log viewer (tail `logs/server.log`)

### Admin (`/ui/admin`)
- Configuration editor (server.yaml with validation)
- Pool management (disaggregated runtime)
- API key management
- Policy store viewer

## Technical Approach

### Phase 1: Static HTML + HTMX (minimal)
- Server-rendered HTML via C++ template engine (e.g., inja or mustache)
- HTMX for dynamic updates without JavaScript framework
- Tailwind CSS (CDN or embedded)
- Total bundle: <500 KB

### Phase 2: React SPA (if complexity demands)
- Vite + React + Tailwind
- Build artifacts embedded via `cmrc` or `xxd` into C++ binary
- WebSocket for real-time metrics streaming

### API Surface

All UI operations go through existing REST endpoints:
- `GET /v1/models` — model list
- `POST /v1/models/load`, `POST /v1/models/unload` — model management
- `GET /metrics` — Prometheus metrics
- `GET /healthz`, `/readyz`, `/livez` — health probes
- `GET /admin/pools` — disaggregated pool status
- New: `GET /api/ui/gpu-topology` — PCIe device tree
- New: `GET /api/ui/backend-stats` — per-backend throughput/latency
- New: `WS /api/ui/stream` — real-time metrics push

## Distribution

- **Binary:** UI assets embedded in `inferfluxd` — no separate install
- **Docker:** Same binary, UI available on configured port
- **Helm:** `ui.enabled` and `ui.basePath` chart values
- **CLI:** `inferctl ui open` opens browser to server UI
