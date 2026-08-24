# InferFlux API Surface (Source-Aligned)

> This page reflects the HTTP endpoints implemented in `server/http/http_server.cpp` and the CLI contracts in `cli/main.cpp`.

```mermaid
graph TD
    A[InferFlux HTTP API] --> B[Public OpenAI-compatible]
    A --> C[Admin Control Plane]
    A --> D[Health & Metrics]

    B --> B1[/v1/completions]
    B --> B2[/v1/chat/completions]
    B --> B3[/v1/models]
    B --> B4[/v1/models/{id}]
    B --> B5[/v1/embeddings]

    C --> C1[/v1/admin/guardrails]
    C --> C2[/v1/admin/rate_limit]
    C --> C3[/v1/admin/api_keys]
    C --> C4[/v1/admin/models]
    C --> C5[/v1/admin/models/default]
    C --> C6[/v1/admin/routing]
    C --> C7[/v1/admin/cache]
    C --> C8[/v1/admin/cache/warm]

    D --> D1[/livez /readyz /healthz]
    D --> D2[/metrics]
```

## 1) Public Endpoints

| Endpoint | Method | Scope | Notes |
|---|---|---|---|
| `/v1/completions` | `POST` | `generate` | Prompt-style completion |
| `/v1/chat/completions` | `POST` | `generate` | Chat-format completion + streaming |
| `/v1/models` | `GET` | `read` | OpenAI-compatible model list |
| `/v1/models/{id}` | `GET` | `read` | Single model descriptor |
| `/v1/embeddings` | `POST` | `read` | Embedding generation with capability routing |

## 2) Admin Endpoints

| Endpoint | Method(s) | Scope | Intent |
|---|---|---|---|
| `/v1/admin/guardrails` | `GET`, `PUT` | `admin` | Read/update blocklist settings |
| `/v1/admin/rate_limit` | `GET`, `PUT` | `admin` | Read/update API key rate limit |
| `/v1/admin/api_keys` | `GET`, `POST`, `DELETE` | `admin` | List/add/remove API keys |
| `/v1/admin/models` | `GET`, `POST`, `DELETE` | `admin` | List/load/unload models |
| `/v1/admin/models/default` | `PUT` | `admin` | Set default model |
| `/v1/admin/routing` | `GET`, `PUT` | `admin` | Capability fallback policy |
| `/v1/admin/cache` | `GET` | `admin` | Prefix/KV cache status |
| `/v1/admin/cache/warm` | `POST` | `admin` | Seed cache entry |
| `/v1/admin/pools` | `GET` | `admin` | Pool readiness plus scheduler and distributed KV transport state |

## 3) Ops Endpoints

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/livez` | `GET` | none | Process liveness |
| `/readyz` | `GET` | none | Readiness + role/state |
| `/healthz` | `GET` | none | General health snapshot |
| `/metrics` | `GET` | none | Prometheus metrics |
| `/ui` | `GET` | none | Optional embedded web UI (if built with `ENABLE_WEBUI=ON`) |

## 4) CLI ↔ API Map

| CLI command | Primary endpoint |
|---|---|
| `inferctl completion` | `POST /v1/completions` |
| `inferctl chat` | `POST /v1/chat/completions` |
| `inferctl models` | `GET /v1/models` or `GET /v1/models/{id}` |
| `inferctl admin models --list` | `GET /v1/admin/models` |
| `inferctl admin models --load` | `POST /v1/admin/models` |
| `inferctl admin models --unload` | `DELETE /v1/admin/models` |
| `inferctl admin models --set-default` | `PUT /v1/admin/models/default` |
| `inferctl admin cache --status` | `GET /v1/admin/cache` |
| `inferctl admin cache --warm` | `POST /v1/admin/cache/warm` |
| `inferctl admin routing --get/--set` | `GET`/`PUT /v1/admin/routing` |
| `inferctl admin pools --get` | `GET /v1/admin/pools` (CLI falls back to `/readyz` + `/metrics` for older servers) |

## 5) Compatibility Notes

- HTTP interface is OpenAI-style for client interoperability.
- Scope checks are enforced server-side (`generate`, `read`, `admin`).
- `/v1/models` and `/v1/models/{id}` are distinct from admin model lifecycle endpoints.
- `session_id` is an optional InferFlux extension for `/v1/completions` and `/v1/chat/completions`;
  it is ignored unless `runtime.scheduler.session_handles.enabled=true` and does not change default stateless behavior.

`GET /v1/admin/pools` returns three top-level objects for automation symmetry:

| Field | Meaning |
|---|---|
| `pool_health` | readiness contract (`ready`, `reason`, role, decode-pool state, distributed timeout streak/debt) |
| `scheduler` | queue-depth and batch-cap metrics |
| `distributed_kv` | enqueue rejection counts and ticket lifecycle totals |

When `INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED=true`, generation endpoints fail closed with:

| Endpoint class | Status | Body |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` | `503` | `{"error":"distributed_kv_transport_degraded"}` |

## 6) Model Identity Contract

`GET /v1/models` and `GET /v1/models/{id}` include backend identity metadata for
automation and policy validation.

| Field | Meaning |
|---|---|
| `backend_exposure.requested_backend` | backend hint requested by config/admin load path |
| `backend_exposure.exposed_backend` | backend actually exposed by the router |
| `backend_exposure.provider` | provider path (`inferflux` or `llama_cpp`) |
| `backend_exposure.fallback` | `true` when selected backend differs due fallback |
| `backend_exposure.fallback_reason` | optional fallback diagnostic string |

Default `inferctl models` table output now surfaces the same contract columns
(`EXPOSED-BE`, `REQ-BE`, `PROVIDER`, `FALLBACK`) so human and machine views
stay aligned.

## 7) Strict Native-Request Error Contract

When backend exposure inferflux-strict mode is enabled
(`runtime.backend_exposure.strict_inferflux_request: true`), explicit admin model
loads that request `inferflux_cuda` fail fast if native kernels are not ready.

- Endpoint: `POST /v1/admin/models`
- Status: `422 Unprocessable Entity`
- Body:
  - `error: "backend_policy_violation"`
  - `reason: "<policy diagnostic>"`
