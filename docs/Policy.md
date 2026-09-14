# Policy Engine & Store

## Overview
InferFlux ships with a native policy store that persists API keys, scopes, rate limits, guardrail blocklists, and capability-routing policy. Policies live in `config/policy_store.conf` (override via `INFERFLUX_POLICY_STORE`) and are loaded on startup, then updated atomically via admin APIs.

## Capabilities
- **RBAC Scopes**: Keys (and future OIDC claims) map to scopes (`generate`, `read`, `admin`), enforced per endpoint.
- **Dynamic Guardrails**: `/v1/admin/guardrails` updates shared blocklists instantly across replicas.
- **Rate Limiting**: `/v1/admin/rate_limit` tunes per-minute quotas without restarts.
- **Routing Policy**: `/v1/admin/routing` controls capability-aware fallback behavior (`allow_default_fallback`, `require_ready_backend`, `fallback_scope`) and now persists those settings across restarts.
- **API Key Lifecycle**: `/v1/admin/api_keys` lists/adds/removes keys, keeping the store in sync with in-memory auth.
- **AES-GCM Encryption**: Optional `INFERFLUX_POLICY_PASSPHRASE` transparently encrypts the policy store on disk.
- **Atomic Durability**: Policy writes use a temp-file + rename flow with `.bak` backups, and load falls back to backup if the primary file is corrupt.
- **OPA-only Guardrail Configs**: `Guardrail::Enabled()` returns `true` when either the blocklist is non-empty **or** an OPA endpoint is configured — so deployments that rely solely on an OPA policy engine (no local blocklist) are correctly detected as having guardrails active. A guardrail is disabled only when both the blocklist and the OPA endpoint are absent.

## Startup precedence
- Policy-store values for guardrails, rate limits, and routing policy override static YAML/env defaults when present.
- If a routing policy is not yet stored, InferFlux seeds the policy store from current YAML/env routing config at startup.

## Admin API persistence semantics
- Mutating admin endpoints (`/v1/admin/guardrails`, `/v1/admin/rate_limit`, `/v1/admin/api_keys`, `/v1/admin/routing`) now return HTTP 500 with `{"error":"policy_persist_failed"}` when on-disk persistence fails.
- On persistence failure, in-memory policy changes are rolled back so runtime and disk remain consistent.
- Integration coverage validates fail-closed rollback semantics for all mutating policy endpoints.

## Roadmap
1. **Persistent backends**: swap the INI file for pluggable stores (SQLite, Postgres, Vault) with watch streams.
2. **Versioning & Audit**: append-only history and signed commits for compliance.
3. **Policy plugins**: optional Cedar/advanced-rule adapters beyond the shipped OPA client for complex rules (tenant quotas, contextual guardrails).
4. **UI & CLI parity**: extend `inferctl admin api-keys` and future dashboards to edit policies with RBAC + audit.
