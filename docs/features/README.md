# Feature Specifications

All five specs are **Proposed** (aspirational, unshipped). They are wired into
[Roadmap](../Roadmap.md) in dependency order; the EP/TP topology detail retired
from design_ep_tp folds into FTR-005's distributed lifecycle scope.

| ID | Spec | Priority | Status | Dependencies |
|---|---|---|---|---|
| FTR-001 | Managed local and explicit-endpoint CLI | P0 | Proposed | (section below) |
| FTR-002 | Agent API and native structured-output contract | P0 | Proposed | (section below) |
| FTR-003 | Tenant-safe adapters and prefix cache | P1 | Proposed | (section below) |
| FTR-004 | SLO-aware admission, routing, and capacity | P1 | Proposed | (section below) |
| FTR-005 | Distributed lifecycle and fault maturity | P2 | Proposed | (section below) |

---

## FTR-001: Managed Local and Explicit-Endpoint CLI

*Proposed · P0 · Owners: CLI, Developer Experience · Deps: TD-004, ADR-0002*

## User Outcome

A new user gets a first response in under five minutes, while an operator can
target local or remote InferFlux explicitly and safely.

## MVP Scope

- Add `--endpoint` and `INFERFLUX_ENDPOINT` with URL/TLS-aware parsing.
- Add named contexts with endpoint and credential references; never store raw
  credentials in world-readable configuration.
- Add `inferctl run MODEL [chat options]` to configure/start a managed daemon,
  wait on readiness, send the request, and print recovery guidance on failure.
- Prefer a user-scoped Unix socket on Linux/macOS and named pipe on Windows;
  retain loopback TCP for SDK compatibility.
- Replace `failed to connect` with destination, diagnosis, and exact next commands.

## Acceptance Evidence

- Fresh-home end-to-end test reaches a response with one user command.
- Endpoint precedence and conflicting flag cases have table-driven tests.
- Non-interactive `chat` never autostarts unless explicitly requested.
- Stale PID/socket recovery, port collision, readiness timeout, and daemon crash are tested.
- Local and remote paths produce the same request/auth/policy semantics.

## Non-Goals

- Linking the inference runtime into `inferctl`.
- Replacing the public HTTP/OpenAI-compatible endpoint.
- Building a desktop UI.


---

## FTR-002: Agent API and Native Structured-Output Contract

*Proposed · P0 · Owners: API, Runtime, QA · Deps: TD-002, TD-003, ADR-0001*

## User Outcome

Agent developers receive valid structured data and tool calls without depending
on hidden provider fallback.

## MVP Scope

- Implement a documented `/v1/responses` text-generation subset by mapping one
  canonical internal request/stream model to existing chat/completion execution.
- Own JSON-schema/grammar token filtering on the native path.
- Validate tool-call names and arguments before emitting terminal events.
- Expose capability/provider/fallback metadata consistently for Responses,
  chat, and model inventory.
- Define stable streaming events, cancellation, usage, and error contracts.

## Acceptance Evidence

- Maintained schema corpus achieves >=99% valid outputs and 100% deterministic
  rejection of unsupported schema features.
- Tool contract suite covers zero, one, parallel, malformed, and cancelled calls.
- Native-supported cases run with the compatibility delegate disabled.
- Cross-provider golden contract tests verify API equivalence, not token identity.
- Documentation states the supported Responses subset and unsupported fields.

## Non-Goals

- Stateful response storage in the first release.
- Audio, image generation, or computer-use tools.
- Pretending unsupported model tool behavior can be repaired by the server.


---

## FTR-003: Tenant-Safe Adapters and Prefix Cache

*Proposed · P1 · Owners: Runtime, Security, Model Lifecycle · Deps: FTR-002, ADR-0003*

## User Outcome

Teams serve specialized models efficiently without loading one base model per
tenant or sharing reusable state across security boundaries.

## MVP Scope

- Load signed/allowlisted LoRA adapters and select one adapter per request.
- Add tenant, base-model revision, adapter revision, tokenizer/template version,
  and policy salt to prefix-cache keys.
- Enforce tenant-scoped adapter load/unload/use permissions and quotas.
- Export bounded hit, miss, eviction, adapter residency, and memory metrics.
- Make cache invalidation explicit on model, adapter, template, or policy change.

## Acceptance Evidence

- Cross-tenant collision and timing-oriented isolation tests show no state reuse.
- Adapter A/B requests interleave without output or KV contamination.
- Load/unload rollback is atomic under concurrent requests.
- Memory and latency overhead are measured for 1, 4, and 16 resident adapters.
- Audit events identify tenant, adapter revision, actor, and outcome.

## Non-Goals

- Training or fine-tuning adapters.
- Arbitrary untrusted runtime adapter paths.
- Cross-tenant cache sharing by default.


---

## FTR-004: SLO-Aware Admission, Routing, and Capacity

*Proposed · P1 · Owners: Scheduler, Observability, Operations · Deps: FTR-003, TD-003*

## User Outcome

Operators choose a latency/cost objective and receive predictable behavior under
load instead of relying on peak throughput.

## MVP Scope

- Measure TTFT, inter-token latency, queue time, tokens/sec, cache reuse,
  saturation, rejection reason, and model/adapter residency.
- Add service classes with token budgets, maximum queue delay, and fairness weights.
- Predict admission cost from prompt length, requested output, cache state, and backend envelope.
- Route only among providers that satisfy capability, tenant, and SLO policy.
- Expose capacity advice and overload state through metrics and admin APIs.

## Acceptance Evidence

- Mixed short/long prompt tests bound p95 TTFT and ITL for each service class.
- Overload rejects early with retryable, machine-readable reasons.
- Fairness tests prevent one tenant or long prefill from starving decode.
- Capacity predictions are calibrated against retained load-test results.
- No routing decision hides provider fallback or weakens tenant isolation.

## Non-Goals

- A Kubernetes autoscaler implementation; InferFlux supplies reliable signals.
- Machine-learned scheduling before the deterministic cost model is proven.


---

## FTR-005: Distributed Lifecycle and Fault Maturity

*Proposed · P2 · Owners: Distributed Runtime, Scheduler, QA · Deps: FTR-004, TD-005, ADR-0004*

## User Outcome

Multi-process serving survives worker loss and transport degradation without
leaking KV state, duplicating generation, or claiming false readiness.

## MVP Scope

- Implement the sequence ownership epoch/lease contract across scheduler,
  workers, KV tickets, sessions, cancellation, and cleanup.
- Replace simulated distributed dispatch with a transport-neutral interface and
  one real multi-process execution path chosen from measured need.
- Make worker loss, stale messages, queue pressure, and cleanup debt visible to
  readiness, admission, admin pools, and metrics.
- Add deterministic retry versus fail-fast rules for each lifecycle state.

## Acceptance Evidence

- Fault matrix covers crash before/after enqueue, acknowledgement, KV commit,
  first token, cancellation, and lease transfer.
- Every test ends with balanced sequence, KV-page, ticket, and session accounting.
- Stale epochs cannot emit tokens or mutate current state.
- Required multi-process CI runs repeated fault injection without hangs.
- The selected topology demonstrates a measured SLO or cost improvement.

## Non-Goals

- Supporting every TP/EP/PD topology in the first release.
- Transparent continuation when correctness cannot be proven.

