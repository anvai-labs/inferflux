# ADR-0006: Three-way origin contract — InferFlux as the authoritative producer

Status: Accepted
Date: 2026-09-15
Owners: Runtime (InferFlux), Gateway (Sandhi), Agent (Victor)
Dependencies: ADR-0003 (tenant boundary), Sandhi ADR-0008/TD-0008, Sandhi TD-0013

## Context

InferFlux serves inference behind Sandhi (a Rust usage-metering gateway) which is driven by
Victor (an agent harness); all three speak OpenAI-compatible HTTP. A three-way audit
(2026-09-14) found that the layering was right — Sandhi already trusted InferFlux's exact
usage for completed calls rather than re-tokenizing — but the *producer* side had gaps that
forced consumers to guess: no reasoning/content separation, non-unique completion ids, a
non-OpenAI error envelope, an inconsistent streaming usage frame, and no way to price a
prompt without generating from it. Each gap is a contract decision that one party cannot
make alone; this ADR records the agreed ownership and the wire contract InferFlux now ships.

The audit's premise — "Sandhi approximates tokens" — held only in two narrow Sandhi-internal
sites (budget-admission ceiling, interrupted-stream fallback), both Sandhi's to fix
(TD-0027); nothing in this ADR asks InferFlux to serve Sandhi's estimation hot path.

## Decision

InferFlux is the single authoritative source for every quantity only the process running the
tokenizer and the generation loop can observe. Consumers trust these fields verbatim and
never re-derive them:

| Capability | Owner | Wire shape (shipped) |
|---|---|---|
| Token counts at the call | InferFlux | `usage.prompt_tokens`, `completion_tokens`, `total_tokens` |
| Reasoning vs. content | InferFlux | `message.reasoning_content` / `delta.reasoning_content`; `usage.completion_tokens_details.reasoning_tokens` |
| Cache-hit attribution | InferFlux | `usage.prompt_tokens_details.cached_tokens` (explicit 0 on miss) |
| Request identity | InferFlux mints; Sandhi owns meter-side correlation | unique completion `id`s; `client_request_id` echo |
| Resolved model | InferFlux (only it performs capability fallback) | `model` reports the model that actually served |
| Latency semantics | InferFlux for origin timings | `usage.duration_ms`, `usage.time_to_first_token_ms` |
| Streaming usage frame | InferFlux emits on every path | terminal SSE frame when `stream_options.include_usage=true` |
| Error shape / retryability | InferFlux for origin errors | OpenAI envelope `{"error":{message,type,code}}`; 429 adds `Retry-After` |
| Budget admission estimate | Sandhi (statistics over settled events — never a hot-path tokenize call) | — |
| Prompt/tool policy, retry ownership, reasoning display | Victor | — |

Reasoning separation is format-aware and format-agnostic on the wire: models emitting
`<think>` tags (Qwen3, LFM2.5) and gpt-oss's harmony channels produce the same
`reasoning_content`/`reasoning_tokens` shape; the response-side splitter is selected per
request from the detected chat-template family and never leaks control tokens into `content`.

`POST /v1/tokenize` (read scope) exposes exact counts through the same resolved-model
routing as generation: `input` (string or array), response `tokens[]` per input plus
`input_tokens` total. It exists for gateway calibration validation and diagnostics — by
explicit agreement it is **not** a hot-path dependency; Sandhi's admission estimates remain
statistical (TD-0013).

Kill switch: `INFERFLUX_DISABLE_REASONING_SPLIT` restores legacy verbatim `content`.

## Consequences

- Every reasoning-capable model family InferFlux supports needs a response-side splitter and
  a prompt-side renderer; adding a family without both degrades quality silently (the
  gpt-oss incident is the recorded example).
- Sandhi can delete its per-call approximation code paths only for *completed* calls;
  interrupted-stream fallback remains until Sandhi's own W7 lands.
- The conformance suite (Sandhi `tests/sdk-conformance/`) pins this contract against drift;
  InferFlux contract changes must update the pin in the same or a same-day follow-up PR.
- `reasoning_tokens` currently counts reasoning *pieces* (streaming) or 1 (buffered) — a
  "billed reasoning present" indicator, not an exact token count; tightening it is a
  follow-up, not a contract break.

## Implementation record

- #170 gateway-facing producer contract (ids, echo, error envelope, rate headers, usage
  frame, TTFT)
- #171 reasoning separation, buffered path (`ReasoningSplitter`)
- #172 llama.cpp v0.2.0 + streaming reasoning separation, `INFERFLUX_STUB_COMPLETION`
- #174 gpt-oss/harmony renderer + `HarmonySplitter`, `ChatTemplateFamily` plumbing
- `/v1/tokenize` (this change): unit tests (parse/build), `IntegrationTokenizeContract`
  stub suite, live-verified counts on Qwen2.5-3B
- Cross-repo records: Sandhi `docs/td/TD-0027-three-way-origin-codesign.md`; Victor
  `docs/architecture/inferflux-reasoning-separation-handoff.md`

## Rejected Alternatives

- **Sandhi re-tokenizes prompts for exactness** — rejected: a second tokenizer is a second
  source of truth that drifts from the serving model's; TD-0013 forbids it.
- **/v1/tokenize as a gateway hot-path dependency** — rejected: adds a round trip per
  request to recover information the settled usage event already carries.
- **Delegating all chat-template rendering to llama.cpp's own converter list** — partially
  rejected: llama.cpp's built-in gpt-oss handler omits the system preamble the model was
  trained on; InferFlux keeps its own harmony renderer (see #174) and uses llama.cpp's list
  for the families it does cover.
