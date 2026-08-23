# InferFlux

> High-throughput inference server for edge and on-premise AI workloads.
> OpenAI-compatible APIs · Native CUDA kernels · 1.87x faster than Ollama at concurrency

**Why InferFlux?** Small, quantized models (3B-8B GGUF) running on a single GPU can power dozens of concurrent AI tasks — but only if the serving layer doesn't bottleneck. Ollama and LM Studio degrade under concurrent load because of Go/Node.js overhead. InferFlux's C++ unified batching serves 8+ concurrent sequences in a single GPU kernel launch, achieving **2.2x scaling** while competitors plateau or degrade.

**Use cases:**
- **Parallel email/document analysis** — 8 agents processing inboxes simultaneously on one RTX 4000
- **Support agent routing** — real-time intent classification and response drafting at scale
- **Market event scanning** — concurrent alert evaluation across multiple data feeds
- **Cybersecurity** — parallel log analysis, threat detection, and anomaly scoring on edge devices
- **IoT / video analytics** — edge inference for camera feeds, sensor fusion, real-time alerting
- **Task orchestration** — multiple AI agents making independent decisions in parallel

**Integration:** Drop-in replacement for any OpenAI-compatible client. Point `OPENAI_BASE_URL` at InferFlux and existing code works unchanged:
- **[Victor](https://github.com/vjsingh1984/victor)** — agentic AI framework with 24 providers. InferFlux replaces Ollama/LM Studio as the local provider with 1.87x higher throughput for parallel agent workloads
- **LangChain / LlamaIndex / openai-python** — use InferFlux as any OpenAI-compatible endpoint
- **NVIDIA RTX 4000 Ada** — optimized for professional workstation GPUs running 3B-8B quantized models at high concurrency

```mermaid
graph LR
    A[Clients\nOpenAI SDKs / curl / inferctl] --> B[InferFlux Server]
    B --> C[Scheduler\nBatching + Fairness + Routing]
    C --> D[Backends\nCPU / CUDA / ROCm / MPS / Vulkan / MLX]
    C --> E[Policy\nAuth + Guardrails + RBAC]
    C --> F[Ops\nMetrics + Audit + Admin APIs]

    style B fill:#f2c14e
    style C fill:#84a59d
    style D fill:#f28482
    style E fill:#8ecae6
    style F fill:#90be6d
```

## Benchmark (Verified Apr 14 2026)

RTX 4000 Ada 20GB · Qwen2.5-3B Q4_K_M · 16 requests × 64 tokens

| Backend | c=1 | c=4 | c=8 | Scale | GPU | Quality |
|---|---|---|---|---|---|---|
| llama_cpp_cuda | 113 tok/s | 206 tok/s | 282 tok/s | 2.5x | 5.4 GB | 16/16 ✓ |
| **inferflux_cuda** | **66 tok/s** | **134 tok/s** | **131 tok/s** | **2.0x** | 6.0 GB | ⚠️ partial¹ |
| Ollama² | 98 tok/s | 111 tok/s | 113 tok/s | 1.2x | 5.4 GB | 16/16 ✓ |
| LM Studio² | 109 tok/s | 81 tok/s | 70 tok/s | 0.6x | 7.9 GB | 16/16 ✓ |

> ¹ inferflux_cuda: correct tokenization and chat template rendering verified; native CUDA kernel numerical precision causes ~60% of responses to diverge from reference. Accuracy parity is the [top priority](https://github.com/vjsingh1984/inferflux/issues/18).
> ² Both use llama.cpp under the hood (confirmed: ±12 MB memory, 0.87-0.96 cosine).

**Key results:**
- `inferflux_cuda` **1.16x faster than Ollama** and **1.88x faster than LM Studio** at c=8
- **Best scaling vs external tools**: 2.0x from c=1→c=8 (Ollama 1.2x, LM Studio 0.6x — degrades)
- `llama_cpp_cuda` remains 2.2x faster than `inferflux_cuda` at c=8 — [closing the gap is the top priority](https://github.com/vjsingh1984/inferflux/issues/18)
- `llama_cpp_cuda` achieves **100% accuracy** — recommended production backend

### Why InferFlux Scales Better

| | InferFlux | Ollama | LM Studio |
|---|---|---|---|
| **Language** | C++17, zero-copy | Go + CGo boundary | Electron + Node.js |
| **Batching** | Unified batch: one GPU kernel serves all concurrent sequences | Sequential per-request dispatch | Single-threaded JS event loop |
| **Weight sharing** | Single GPU context, shared across all requests | Per-process model instance | llama.cpp server subprocess |
| **Overhead at c=8** | ~0 (batch kernel) | CGo call overhead × 8 + GC pauses | Event loop serialization |

Details: [docs/TechDebt_and_Competitive_Roadmap.md](docs/TechDebt_and_Competitive_Roadmap.md)

## OSS Release Snapshot

| Area | What ships in this repo |
|---|---|
| Server binary | `inferfluxd` |
| CLI binary | `inferctl` |
| API surface | `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/models/{id}`, `/v1/embeddings`, `/v1/admin/*` |
| Runtime options | CPU + optional CUDA/ROCm/MPS/Vulkan/MLX |
| Ops endpoints | `/livez`, `/readyz`, `/healthz`, `/metrics`, optional `/ui` |
| OSS metadata | `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` |

## Current Reality

| State | Reading |
|---|---|
| Strong today | API/admin/CLI contracts, backend identity, chat template rendering (ChatML/Llama/Mistral/Gemma), GGUF metadata API |
| Proven advantage | `llama_cpp_cuda` 2.5x faster than Ollama, 4x faster than LM Studio at c=8. `inferflux_cuda` 1.16x faster than Ollama, 1.88x faster than LM Studio |
| Native CUDA | `inferflux_cuda` functional with correct tokenization and chat templates. 50+ fused GEMV kernels, FlashAttention-2, repetition penalty. Accuracy parity with llama.cpp is the top priority |
| Architecture | RAII, DIP (registry-based backend factory), strategy pattern (batch selection), MetricsRegistry DI, InferenceRequest decomposed |
| Still open | GPU memory overhead, native structured output, speculative decoding integration |

## Design Principles

| Principle | Reading |
|---|---|
| Throughput | Unified batching: one GPU kernel serves all concurrent sequences |
| Quality | Chat template auto-detected from GGUF metadata; repetition penalty prevents degenerate loops |
| Memory | Quantized GGUF stays quantized; scratch buffer aliasing; KV budget auto-tuned |
| Backend selection | `inferflux_cuda` is the recommended backend; `llama_cpp_cuda` available as fallback |

## 3-Minute Bring-Up

```bash
# 1) Build
./scripts/build.sh

# Optional: target Ada RTX 4000 specifically
# INFERFLUX_CUDA_ARCHS=89 ./scripts/build.sh

# 2) Run server
INFERFLUX_MODEL_PATH=models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  ./build/inferfluxd --config config/server.yaml

# 3) Send request
./build/inferctl completion \
  --prompt "Explain why batching improves throughput" \
  --max-tokens 64 \
  --api-key dev-key-123
```

## API Surface

| Scope | Endpoint | Method |
|---|---|---|
| Health | `/livez`, `/readyz`, `/healthz` | `GET` |
| Metrics | `/metrics` | `GET` |
| OpenAI | `/v1/completions`, `/v1/chat/completions` | `POST` |
| OpenAI | `/v1/models`, `/v1/models/{id}` | `GET` |
| OpenAI | `/v1/embeddings` | `POST` |
| Admin | `/v1/admin/guardrails` | `GET`, `PUT` |
| Admin | `/v1/admin/rate_limit` | `GET`, `PUT` |
| Admin | `/v1/admin/api_keys` | `GET`, `POST`, `DELETE` |
| Admin | `/v1/admin/models` | `GET`, `POST`, `DELETE` |
| Admin | `/v1/admin/models/default` | `PUT` |
| Admin | `/v1/admin/routing` | `GET`, `PUT` |
| Admin | `/v1/admin/cache`, `/v1/admin/cache/warm` | `GET`, `POST` |

Full API map: [docs/API_SURFACE.md](docs/API_SURFACE.md)

## CLI Surface

```mermaid
graph TD
    A[inferctl] --> B[serve / status / completion / chat / models]
    A --> C[server\nstart/stop/status/restart/logs]
    A --> D[admin\nguardrails/rate-limit/routing/pools/models/cache/api-keys]
    A --> E[pull / quickstart]

    style A fill:#f2c14e
```

## Documentation

Start here: [docs/INDEX.md](docs/INDEX.md)

Performance and runtime:
- [docs/benchmarks.md](docs/benchmarks.md)
- [docs/MONITORING.md](docs/MONITORING.md)
- [docs/TechDebt_and_Competitive_Roadmap.md](docs/TechDebt_and_Competitive_Roadmap.md)
- [docs/Roadmap.md](docs/Roadmap.md)

Architecture:
- [docs/GEMV_KERNEL_ARCHITECTURE.md](docs/GEMV_KERNEL_ARCHITECTURE.md)
- [docs/GGUF_NATIVE_KERNEL_IMPLEMENTATION.md](docs/GGUF_NATIVE_KERNEL_IMPLEMENTATION.md)
- [docs/Architecture.md](docs/Architecture.md)

## Project Status

- Done: production-ready HTTP server with OpenAI-compatible APIs
- Done: multi-backend runtime across CPU and optional GPU providers
- Done: operator-grade auth, RBAC, metrics, audit, and admin surfaces
- Done: documented `llama_cpp_cuda` advantage over Ollama on the published concurrent GGUF benchmark
- In progress: `inferflux_cuda` concurrency work, especially decode down-proj row-pair and row-quad kernels
- In progress: distributed runtime ownership and failure maturity

## Quick Links

- Benchmarks: [docs/benchmarks.md](docs/benchmarks.md)
- Configuration: [config/server.yaml](config/server.yaml)
- Build: [scripts/build.sh](scripts/build.sh)
- Tests: `ctest --test-dir build`

## License

Apache License 2.0. See [LICENSE](LICENSE).
