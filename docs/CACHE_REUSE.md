# Cache execution and usage contract

`usage.prompt_tokens` includes cached input tokens.
`usage.prompt_tokens_details.cached_tokens` reports tokens skipped by accepted KV
reuse; a lookup candidate or a shorter elapsed time is not sufficient evidence.
Fresh input is `prompt_tokens - cached_tokens`. This contract applies to ordinary
responses and terminal SSE usage frames (`stream_options.include_usage=true`).

| Execution mode | Prefix reuse policy | Session reuse policy |
|---|---|---|
| Plain chat or completion, phased execution | Eligible when a compatible donor survives admission and copying succeeds | Eligible when configured and a compatible lease is available |
| Tools present, phased execution | Same as plain chat; tool schemas change the tokenized prefix | Same as plain chat |
| `response_format` requesting JSON or grammar | Unsupported on the full Generate path | Unsupported on the full Generate path |
| `logprobs=true` | Unsupported on the full Generate path | Unsupported on the full Generate path |

Structured output and logprobs deliberately use one Generate call to preserve a
single sampler/grammar state. The llama.cpp Generate path clears sequence 0.
Removing the scheduler guard to improve cache numbers would change execution
correctness. Stateful structured reuse requires a separate design and tests for
output equivalence, grammar validity, concurrent requests, and cancellation.
This policy does not change those generation paths or promise cache hits.

Global prefix caching and optional session leases are independent. Session IDs
are not evidence that leases are enabled. Misses may result from different
tokenized prefixes, incompatible models/backends, unavailable donors, eviction,
capacity contention, or execution fallback. Diagnose them using correlated
requests and accepted prefill extent. On an exact prefix hit, one input token is
replayed to refresh logits and is counted as fresh input. Failed copies and
full-prefill recovery report zero reused tokens.

Use the [bounded replay procedure](planning/CACHE_REPLAY_2026-09-18.md) to preserve
warm/cold provenance. Do not clear a shared cache or change a serving model to
force a cold sample. A direct call can warm a later gateway call, so label that
ordering explicitly. Sandhi's ledger should satisfy fresh + cached = upstream
inclusive prompt tokens. Gateway request records alone are not distributed
traces; configure an OTEL collector explicitly if one is required.

This document is a policy clarification, with no backend, sampler, API, or
configuration behavior change. Reverting it rolls back the clarification only.
