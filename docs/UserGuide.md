# User Guide (Canonical OSS)

**Status:** Canonical

## 1) User Journey Map

```mermaid
flowchart LR
    A[Start inferfluxd] --> B[Run first completion/chat]
    B --> C[Inspect /v1/models]
    C --> D[Use admin controls]
    D --> E[Watch /metrics and /readyz]
```

## 2) Task Matrix

| Goal | CLI | HTTP | Required scope |
|---|---|---|---|
| Completion | `inferctl completion` | `POST /v1/completions` | `generate` |
| Chat completion | `inferctl chat` | `POST /v1/chat/completions` | `generate` |
| List models | `inferctl models` | `GET /v1/models` | `read` |
| Model detail | `inferctl models --id <id>` | `GET /v1/models/{id}` | `read` |
| Embeddings | n/a | `POST /v1/embeddings` | `read` |
| Admin models/routing/cache | `inferctl admin ...` | `/v1/admin/*` | `admin` |

## 3) Fast Start (4 Steps)

### 1. Build

```bash
./scripts/build.sh
```

### 2. Run server

```bash
INFERFLUX_MODEL_PATH=models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  ./build/inferfluxd --config config/server.yaml
```

GPU profile:

```bash
./build/inferfluxd --config config/server.cuda.yaml
```

### 3. Verify health

```bash
curl -s http://127.0.0.1:8080/livez
curl -s http://127.0.0.1:8080/readyz
curl -s http://127.0.0.1:8080/metrics | head -40
```

### 4. Send first request

```bash
./build/inferctl completion \
  --prompt "List 3 benefits of continuous batching" \
  --max-tokens 64 \
  --api-key dev-key-123
```

## 4) Core Usage Patterns

### Chat (CLI)

```bash
./build/inferctl chat \
  --message "user:Give 3 latency tuning ideas" \
  --max-tokens 96 \
  --api-key dev-key-123
```

### Chat (HTTP)

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-key-123' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false,
    "max_tokens": 64
  }'
```

### Embeddings

```bash
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-key-123' \
  -d '{
    "model": "llama3-8b",
    "input": ["first sentence", "second sentence"]
  }'
```

## 5) Model and Identity Inspection

```bash
./build/inferctl models --api-key dev-key-123
./build/inferctl models --json --api-key dev-key-123
./build/inferctl models --id llama3-8b --json --api-key dev-key-123
```

Backend identity contract fields (`--json`):
- `backend_exposure.requested_backend`
- `backend_exposure.exposed_backend`
- `backend_exposure.provider`
- `backend_exposure.fallback`
- `backend_exposure.fallback_reason` (optional)

## 6) Admin Essentials

```bash
./build/inferctl admin models --list --api-key dev-key-123
./build/inferctl admin routing --get --api-key dev-key-123
./build/inferctl admin pools --get --api-key dev-key-123
./build/inferctl admin cache --status --api-key dev-key-123
```

## 7) Frequently Tuned Knobs

| Area | YAML key | Env override |
|---|---|---|
| Model path | `models[].path` | `INFERFLUX_MODEL_PATH` |
| Backend priority | `runtime.backend_priority` | `INFERFLUX_BACKEND_PRIORITY` |
| Batch size | `runtime.scheduler.max_batch_size` | `INFERFLUX_SCHED_MAX_BATCH_SIZE` |
| Batch token cap | `runtime.scheduler.max_batch_tokens` | `INFERFLUX_SCHED_MAX_BATCH_TOKENS` |
| CUDA kernel policy | `runtime.cuda.attention.kernel` | `INFERFLUX_CUDA_ATTENTION_KERNEL` |
| Phase overlap | `runtime.cuda.phase_overlap.enabled` | `INFERFLUX_CUDA_PHASE_OVERLAP` |
| InferFlux strict mode | `runtime.backend_exposure.strict_inferflux_request` | `INFERFLUX_BACKEND_STRICT_INFERFLUX_REQUEST` |

## 8) Quick Failure Matrix

| Symptom | First check | Typical fix |
|---|---|---|
| `401`/`403` | key + scope | use correct bearer key (`generate`/`read`/`admin`) |
| `404 model_not_found` | `inferctl models` | load model or set correct `model` id |
| `422 backend_policy_violation` | backend exposure policy | disable strict mode or use compatible backend |
| High latency / low throughput | `/metrics` batch counters | tune batch size/token cap and KV pages |
| Not ready | `/readyz` | inspect model path, backend readiness, logs |

## 9) Related Docs

- [Quickstart](Quickstart.md)
- [API Surface](API_SURFACE.md)
- [CONFIG_REFERENCE](CONFIG_REFERENCE.md)
- [Admin Guide](AdminGuide.md)
- [Troubleshooting](Troubleshooting.md)
