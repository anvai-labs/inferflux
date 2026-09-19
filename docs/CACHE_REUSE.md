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

## Effective capacity and policy

`GET /v1/models`, `GET /v1/models/{id}`, and `GET /v1/admin/models` include a
`runtime` object for each loaded model:

```json
{
  "sequence_capacity": 2,
  "context_tokens_per_sequence": 32768,
  "session_handles_enabled": false,
  "cache_reuse": {
    "phased": {"prefix": true, "session": false},
    "full_generate": {
      "prefix": false,
      "session": false,
      "reason": "sampler_grammar_continuity"
    }
  }
}
```

This example illustrates a configured two-sequence llama.cpp backend. Capacity
comes from that model's active backend: llama.cpp context/sequence APIs for
CPU/ROCm/CUDA wrappers, and the allocated native KV cache for native CUDA. A
missing, unloaded, or non-reporting backend produces `null` capacity, never a
guessed CUDA default. `details.context_length` remains training/model metadata.
Sequence capacity is a configured limit, not a live count of free slots.

The phased policy combines backend transfer support with configured radix/session
availability. It describes eligibility, not guaranteed reuse for an individual
request. Structured/logprob requests use `full_generate`. Native per-model
sequence capacity also bounds admission; no model-global CUDA gauge is used to
invent another backend's capacity. Revert the capability commit to remove these
additive fields and restore the previous admission-capacity source.
