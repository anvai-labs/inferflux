# ADR-0007: Phased execution for logprobs and structured-output requests

Status: Proposed
Date: 2026-10-21
Owners: Runtime (InferFlux); consumers: Gateway (Sandhi metering), Agent (Victor)
Dependencies: #241 (stateless-Generate decode-burst fix), #248 (stateless
admission cost), ADR-0005 (trusted-GPU release evidence). Tracks #243 item 4.

## Context

`Scheduler::ProcessBatch` routes any request carrying `logprobs` or a
`response_format` onto the legacy stateless `Generate()` executor path
(`policy_bypass`, scheduler.cpp ~2121). The recorded rationale: the backend
applies "a single consistent sampler/grammar state" there, while phased
prefill + stepwise decode "would split sampler state across phases."

The consequences of that bypass are now measured, not hypothetical:

- **No cache participation, by construction.** The bypass forces
  `cache_reused_tokens = 0` and skips sequence admission, so these requests
  never join the prefix cache — as donor or beneficiary. Gateway clients that
  attach `logprobs` for metering (the Victor/Sandhi profile) observe zero
  cache ratios no matter how repetitive their traffic is, and cannot approach
  the ≈80% ratios the same clients see on providers whose equivalent path is
  cache-eligible.
- **An order of magnitude slower decode.** The legacy path runs ~15–44
  ms/token on `qwen3-coder-30b` where the phased/stepwise path runs ~1.2–1.5
  ms/token on identical hardware, because it re-prefills per executor pass
  (bounded to one pass since #241) and samples through the heavyweight
  per-call chain.
- **Slot-slotting.** Stateless requests bypass `max_parallel_sequences`
  slots entirely and serialize on the backend's single context.

Since #241 (stateless passes run their whole budget unsliced) and #248
(admission charges `prompt + max_tokens` for them), the bypass's costs are
honest in accounting but unchanged in latency and caching.

### Why this is more tractable than the original rationale assumed

The unified batch executor **already** does per-request sampling correctly:

- `ExecuteUnifiedBatch` builds per-sequence logits rows
  (`llama_get_logits_ith(context_, logit_indices[i])`) — the input struct has
  carried `request_logits` (default true) since inception, and the sampling
  loop already constructs a per-request sampler chain from
  `inp.sampling` per step, samples, and accepts
  (llama_cpp_backend.cpp unified sampling block).
- A greedy fast path already reads the raw logits row directly for argmax.

What the unified path lacks is exactly two things: (1) returning logprob
data per step, and (2) sampler state that persists across steps for a given
sequence (needed only for grammar, and for penalty history). This ADR adds
(1) as Stage 1 and (2) as Stage 2.

## Decision

Move logprobs-carrying requests onto the phased path first (Stage 1), then
grammar/structured-output requests (Stage 2), with per-sequence sampler
lifetime as the enabling primitive. The legacy `Generate()` path remains as
the images fallback and behind kill switches.

### Stage 1 — logprobs on the phased path (the cache-ratio unlock)

1. **Wire contract** (`runtime/backends/common/backend_types.h`):
   extend `UnifiedBatchInput` with `int logprob_top_n{-1}` (-1 = not
   requested; 0 = selected-token logprob only; N = top-N alternatives) and
   `UnifiedBatchOutput` with `TokenLogprob step_logprob;` (populated only
   when requested).
2. **Backend sampling block**: after `llama_sampler_sample`, when
   `logprob_top_n >= 0`, compute the selected token's log-softmax value and
   top-N alternatives from the **already-fetched** logits row. Factor the
   existing `CollectLogprob` math into one shared helper (e.g.
   `ComputeTokenLogprob(const float *row, int n_vocab, int token, int
   top_n)`) used by both the legacy and unified paths — one derivation, per
   the dedup convention; the legacy path keeps byte-identical output.
3. **Scheduler**: remove `collect_logprobs` from the `policy_bypass`
   predicate. `response_format.has_format` stays bypassed until Stage 2.
4. **Executor**: remove the `collect_logprobs` exclusion from the phased
   eligibility filter (batch_executor.cpp ~438-441); in the phased decode
   loops, append each step's `step_logprob` into the request-owned logprob
   vector exactly where pieces are appended today, so `InferenceResult`
   carries the same logprob list shape the legacy path produced.
5. **Kill switch**: `INFERFLUX_DISABLE_PHASED_LOGPROBS=1` restores the
   Stage-0 bypass (env-parsing convention in `NativeExecutionPolicy`-style
   bootstrap), so a serving regression is one restart away from old
   behavior.
6. **Known fidelity note (pre-existing, out of scope)**: the phased path
   re-creates the sampler chain per step, so repetition-penalty history
   resets each step — the same limitation plain phased requests have today.
   Stage 1 changes nothing about it for either population; penalty-history
   parity is tracked separately.

Stage 1 is deliberately **without** grammar: `logprobs` alone needs no
cross-step sampler state (temperature/top-p sampling is stateless per step;
penalties are the documented exception above).

### Stage 2 — structured output on the phased path

1. **Per-sequence sampler lifetime** in `LlamaCppBackend`: a backend-owned
   `sequence_id → llama_sampler*` registry. Created at phased-prefill
   admission from the request's `StructuredConstraint` (grammar + root +
   sampling), advanced with `llama_sampler_accept` for every token the
   unified loop emits for that sequence, and freed on sequence retirement
   (both the immediate and deferred-retirement paths, alongside KV frees).
2. **Generation discipline**: the registry key carries the sequence
   generation; a reused slot number must never inherit a stale chain —
   mirror the slot-generation validation used for KV sequences
   (`SequenceSlotManager` lease generations).
3. **Sampling loop**: when a per-sequence chain exists, the unified loop
   samples through it instead of the per-step `active_sampler_`; otherwise
   behavior is unchanged.
4. **Scheduler**: remove `response_format.has_format` from the bypass
   predicate. JSON-mode output validation stays at completion, unchanged.
5. **Kill switch**: `INFERFLUX_DISABLE_PHASED_GRAMMAR=1`.

### Observability

- Expose the execution-path mix: a counter labeled by `cache_execution_path`
  (`phased` / `full_generate` / `full_generate_images`) so the bypass
  population is visible in `/metrics` before and after each stage.
- The existing prefix-lookup/reuse counters (#244) become the success
  signal: logprobs traffic should start registering lookups, hits, and
  accepted reuse.
- MONITORING.md updated with the new counter and the kill switches.

## Alternatives considered

- **Keep the bypass; make `Generate()` cache-aware.** Rejected: duplicates
  the cache machinery onto a second execution path, leaves the ~15–44
  ms/token decode penalty and slot-bypass in place, and re-opens exactly
  the drift surface the dedup rounds just closed.
- **Batch grammar via llama.cpp-level shared state.** Rejected: llama.cpp
  exposes no per-sequence grammar sampler suitable for interleaved batch
  steps; the backend-owned per-sequence registry is the minimal ownership
  change at our layer.
- **Defer entirely.** Rejected: this is the remaining blocker to
  gateway-side cache ratios for metering clients (the #184/C5 cache
  expectations), and Stage 1's risk is bounded (plumbing + a kill switch).

## Acceptance gates

Per stage, in order (CPU TDD first, then trusted-main gates per ADR-0005,
then serving cutover with recorded rollback):

**Stage 1**
1. Red→green unit tests: a logprobs request takes the phased path; the
   collected logprob list equals the legacy `Generate()` output for the
   same stub logits (exact-value parity, both for `logprobs=true` and
   `top_logprobs=N`); a repeated logprobs request reports
   `cached_tokens > 0` (cache eligibility, the point of the change).
2. Live: the prefix-metrics probe with `logprobs: true`; Victor-scale
   latency within 2× the plain-request baseline (logprob collection adds
   per-step top-k work, not a path change); gateway `cached_tokens`
   observed on a repeated metered call.
3. Kill switch round-trip: env set → request bypasses; unset → phased.

**Stage 2**
1. Property test: every token emitted under a grammar constraint satisfies
   the grammar (existing grammar fixtures); slot retirement frees the chain
   (leak assertion); slot reuse never sees a stale chain (generation test).
2. Live: `json_object`/`json_schema` requests complete with valid JSON on
   the phased path and become cache-eligible.

## Consequences

- Metering clients' cache ratios stop being zero-by-design; the DeepSeek-
  class ≈80% ratios become reachable for repetitive agent traffic.
- Logprobs/structured decode moves to phased speed (~1.2–1.5 ms/token class)
  and participates in slot fairness and continuous batching.
- The legacy `Generate()` path's remaining population shrinks to the images
  fallback and kill-switch cases, simplifying future retirement decisions.
- Two new backend-owned state registries (per-sequence samplers) inherit
  the sequence-lifecycle discipline (generations, deferred retirement) —
  the cost of Stage 2, and the reason it is a separate stage.
