# ADR-0007: Phased execution for logprobs and structured-output requests

Status: Proposed
Date: 2026-10-21 (rev 2 — incorporates adversarial design review; rev 1
gaps listed under Revision history)
Owners: Runtime (InferFlux); consumers: Gateway (Sandhi metering), Agent (Victor)
Dependencies: ADR-0005 (trusted-GPU release evidence). Tracks #243 item 4
and the #241/#248 scheduler repairs it builds on.

## Context

`Scheduler::ProcessBatch` routes any request carrying `logprobs` or a
`response_format` onto the legacy stateless `Generate()` executor path
(`policy_bypass`, scheduler.cpp ~2117–2136). The recorded rationale: the
backend applies "a single consistent sampler/grammar state" there, while
phased prefill + stepwise decode "would split sampler state across phases."

The consequences of that bypass are measured, not hypothetical:

- **No cache participation, by construction.** The bypass forces
  `cache_reused_tokens = 0` and skips sequence admission, so these requests
  never join the prefix cache — as donor or beneficiary. Gateway clients
  that attach `logprobs` for metering (the Victor/Sandhi profile) observe
  zero cache ratios regardless of traffic repetition.
- **An order of magnitude slower decode.** The legacy path re-prefills per
  executor pass (one pass since #241) and samples through the heavyweight
  per-call chain; the phased path decodes stepwise at roughly a tenth of
  the per-token cost on the same hardware.
- **Slot-slotting.** Stateless requests bypass `max_parallel_sequences`
  slots entirely and serialize on the backend's single context.

### Why this is more tractable than the original rationale assumed

For **llama-family backends** the unified batch executor already does
per-request sampling correctly (design-review-verified):

- `ExecuteUnifiedBatch` builds per-sequence logits rows
  (`llama_get_logits_ith(context_, logit_indices[i])`); the input struct
  has carried `request_logits` (default true) since inception.
- The sampling loop constructs a per-request sampler chain from
  `inp.sampling` per step, samples, and accepts
  (llama_cpp_backend.cpp ~1616–1666); a greedy fast path reads the raw
  logits row directly (~1587–1613).
- Every step path the scheduler drives on llama backends reaches that
  block (`ExecuteUnifiedBatchStep`/`ExecuteUnifiedBatchPhased`/
  `ExecutePhasedPrefillStep` all funnel into `ExecuteUnifiedBatch`; the
  async submit is synchronous for these backends). The one sampler-free
  path (`BatchDecodeStep`, greedy argmax) has no callers today.
- Logprob math is row-local: llama logits are raw (un-log-softmaxed) and a
  stable log-softmax over the row already exists as
  `ComputeLogprob(const float*, …)` in `runtime/backends/backend_utils.h`
  (used by the native and MLX paths); the legacy `CollectLogprob` is a
  context-coupled wrapper over the same math. Reported logprobs reflect
  the pre-processor distribution on both paths — temperature/top-p change
  the sampled token, not the reported value.

What the unified path lacks: (1) returning logprob data per step, (2)
sampler state persisting across steps for a sequence (needed by grammar;
see the fidelity note for penalties), and (3) awareness of these request
populations in the burst/greedy short-circuits and the scheduler's
eligibility sites — enumerated below, because they are numerous.

## Scope boundary (design-review finding 1, 7)

**Stages 1 and 2 apply to llama-family backends only.** Native backends
(`NativeGpuBackend`/`InferfluxCudaRuntime`) override `ExecuteUnifiedBatch`
and never reach the sampling block this design extends, yet advertise
`supports_logprobs`/`supports_structured_output`; unscoped, they would
return HTTP 200 with empty logprob lists or, post-Stage-2, emit
unconstrained grammar output. Therefore:

- The `policy_bypass` predicate is evaluated per resolved backend: on
  native backends, `collect_logprobs`/`response_format` requests **remain**
  on the legacy path (which delegates to the parity backend for grammar
  today). A follow-up may add native support; this ADR does not budget it.
- The burst and greedy short-circuits (`TryGreedyBurstDecodeTokens`,
  `ExecuteUnifiedBatchBurst`, `BurstStepEligible`, and
  `CanSampleGreedyArgmax`) gain explicit ineligibility for
  logprob-collecting (Stage 1) and grammar-constrained (Stage 2) requests —
  their comments currently assume these populations "are excluded upstream
  by the scheduler," which is exactly what the stages remove.

## Decision

Two stages. Legacy `Generate()` remains for the images fallback, native
backends, and behind kill switches.

### Stage 1 — logprobs on the phased path (llama backends)

**Wire & plumbing:**
1. `UnifiedBatchInput` gains `int logprob_top_n{-1}` (-1 = not requested);
   `UnifiedBatchOutput` gains `TokenLogprob step_logprob`. In
   `ExecuteUnifiedBatch`'s per-sequence sampling block, when requested,
   compute the selected token's logprob and top-N from the already-fetched
   logits row via the existing `ComputeLogprob` (no new math derivation;
   the legacy wrapper continues to produce byte-identical output).
2. **Per-sampled-token list semantics** (review finding 3): the legacy list
   contains one `TokenLogprob` for EVERY sampled non-EOS token — including
   empty-piece tokens and the stop-triggering token. The phased loops
   therefore append at sampling time, not at visible-piece emission
   (`IsVisibleGeneratedPiece` filters empty pieces and UTF-8 assembly
   drops incomplete ones — appending "where pieces are appended" would
   under-report). `PrefillResult` gains the prefill-sampled first token's
   logprob (it originates in prefill today and is emitted at
   prime/unprime); `FinalizeUnifiedDecodeStepResult` copies the accumulated
   list into the final `InferenceResult`; the accumulator lives in
   `execution.result.logprobs` with reprime safety verified (yields reset
   `execution.result` today — the accumulator must survive or be flushed at
   yield).
3. **Streaming**: `AppendGeneratedPiece`'s `on_token(emit_piece, nullptr)`
   is threaded to carry the step's logprob so `stream:true` + `logprobs`
   SSE chunks keep per-token logprobs (the metering profile streams; silent
   null-logprob chunks would be a regression).

**Gate sites — all seven must change together** (review finding 2; touching
only the bypass and executor filter routes these requests through the
slowest possible path with mis-sized admission):
a. `policy_bypass` predicate: `collect_logprobs` term drops; predicate
   becomes `has_images || response_format.has_format` (**the images clause
   is new** — review finding 10: without it, images+logprobs requests would
   flow into text-only phased prefill and generate garbage, where today
   they at least run `GenerateWithImages`).
   Native-backend scoping per the Scope boundary.
b. Executor phased eligibility filter (batch_executor.cpp ~580–583): drop
   the `collect_logprobs` exclusion.
c. `CanAppendToStickyStepBatchLocked` (scheduler.cpp ~1003): drop the
   `collect_logprobs` exclusion so sticky cohorts can merge them.
d. `resolve_direct_step_backend` (~1371): drop `collect_logprobs`.
e. `use_stepwise_decode` (~1513): drop `collect_logprobs` (today one
   logprob request flips the whole batch to the general executor).
f. `can_defer_to_unified_prefill` (~2372): drop `collect_logprobs`.
g. `EstimateQueueTokenCost` (~303): Stage-1 requests decode stepwise again,
   so their admission charge reverts to prompt-width (decode demand is
   charged from the decode queue as for plain requests). The #248
   prompt+max_tokens charge remains only for populations still stateless
   (`has_format` until Stage 2, `has_images`, native backends).
4. **Kill switch**: `INFERFLUX_DISABLE_PHASED_LOGPROBS=1` restores the
   Stage-0 routing at sites (a) and (b).

**Fidelity note (corrected per review finding 4):** the unified path
re-creates the sampler chain per step, so penalties and RNG state do NOT
accumulate across a generation — true today for plain phased requests.
Legacy `Generate()` DOES accumulate them (one chain per call, auto-accept
on sample). **Stage 1 therefore deliberately changes sampling semantics
for the migrated population**: clients setting repetition/frequency/
presence penalties or a fixed seed may see different sampled tokens after
migration (SamplingParams default neutral, so default traffic is
unaffected). The parity gate below is scoped to neutral-penalty greedy
requests; the divergence is documented here rather than hidden. Related
wart not to propagate: the unified loop's explicit `llama_sampler_accept`
after `llama_sampler_sample` double-accepts (harmless today only because
chains are torn down per step); Stage 2's persistent chains must not
inherit that line.

### Stage 2 — structured output on the phased path (llama backends)

1. **Per-sequence sampler registry** on `LlamaCppBackend`:
   `sequence_id + generation → llama_sampler*`, created at phased-prefill
   admission from the `StructuredConstraint`, advanced per emitted token,
   freed wherever KV sequences free (`FreeSeqSlot`-centralized retirement
   covers immediate and deferred paths; `UnifiedBatchInput` already carries
   `sequence_generation` for key validation). The registry rides
   `backend_state_mutex_` — the sampling block, `SetupSampler`, and
   `FreeSequence` already serialize there, and decode workers exist only
   in disaggregated mode. Slot-0 legacy-clear edge: legacy `Generate()`
   clears KV sequence 0 with no slot discipline; the registry must drop
   any chain for a sequence cleared outside the slot lifecycle.
2. **The sampling loop consults the per-sequence chain when one exists** —
   including the **greedy fast path**, which today argmaxes with no
   sampler at all (review finding 5: `temperature 0` json_schema is the
   common case and would otherwise emit unconstrained tokens that
   completion-time validation then rejects — breaking requests that
   succeed today). `CanSampleGreedyArgmax` gains grammar awareness.
3. **JSON-mode validation moves with the population** (review finding 6):
   the parse/`require_json_object` rejection/`{"output": …}` wrapping/
   retokenization block currently lives only in the legacy executor
   finalization (batch_executor.cpp ~945–979). It must apply in phased
   finalization (`FinalizeUnifiedDecodeStepResult`) with identical error
   strings and token-count semantics.
4. `policy_bypass` predicate becomes `has_images` (plus native scoping);
   the same gate-site cleanup (c/d/e/f) applies to `has_format`.
5. **Kill switch**: `INFERFLUX_DISABLE_PHASED_GRAMMAR=1`.

## Alternatives considered

- **Keep the bypass; make `Generate()` cache-aware.** Rejected: duplicates
  cache machinery onto a second path, leaves the decode penalty and slot
  bypass, re-opens the drift surface the dedup rounds closed.
- **Batch grammar via llama.cpp-level shared state.** Rejected: no
  per-sequence grammar sampler suitable for interleaved batch steps; the
  backend-owned registry is the minimal ownership change at our layer.
- **Include native backends in the stages.** Rejected for scope: the native
  runtime has its own sampling and logprob plumbing; unifying it is a
  separate follow-up once the llama path proves out.
- **Defer entirely.** Rejected: this is the remaining blocker to
  gateway-side cache ratios for metering clients (#184/C5 expectations),
  and Stage 1's risk is bounded by the kill switch.

## Acceptance gates

Restated to match what existing test seams can actually exercise (review
finding 8): no current double produces controllable logits rows, so
exact-value parity is proven at three levels — helper units on synthetic
rows, list-shape/routing units via doubles, real-model integration.

**Stage 1**
1. Unit: `ComputeLogprob`-based step-logprob construction on synthetic
   logits rows (selected value + top-N ordering/content); list-shape tests
   via doubles asserting one entry per sampled token (including empty-piece
   and stop-trigger tokens) and first-token capture from prefill.
2. Routing units: logprob requests join sticky cohorts, direct stepwise,
   and deferred prefill (gates c/d/e/f); admission charge reverts (g);
   images+logprobs stays on `GenerateWithImages` (a); burst ineligibility;
   kill-switch round-trip.
3. Real-model integration (trusted-main gates per ADR-0005): exact-value
   logprob parity vs the legacy path for neutral-penalty greedy prompts on
   the pinned model; a repeated logprob request reports
   `cached_prompt_tokens > 0`; streaming chunks carry non-null logprobs;
   Victor-scale latency within 2× the plain-request baseline.

**Stage 2**
1. Property test: every emitted token satisfies the grammar (existing
   fixtures); greedy fast path is grammar-gated; registry freed on
   immediate and deferred retirement (leak assertion); stale-chain
   rejection on slot reuse (generation test); slot-0 legacy-clear edge.
2. Phased-finalization JSON validation produces identical error strings,
   wrapping, and token counts to legacy for the same completions.
3. Real-model: `json_object`/`json_schema` complete with valid JSON on the
   phased path and become cache-eligible.

## Observability

- Execution-path counter labeled by `cache_execution_path`
  (`phased`/`full_generate`/`full_generate_images` — the string values
  already exist at the decision sites) so the bypass population is visible
  before and after each stage.
- The #244 prefix-lookup/reuse counters are the success signal: metered
  traffic should start registering lookups, hits, and accepted reuse.
- MONITORING.md updated with the counter and both kill switches.

## Consequences

- Metering clients' cache ratios stop being zero-by-design; DeepSeek-class
  ratios become reachable for repetitive agent traffic.
- Logprobs/structured decode moves to phased speed and participates in
  slot fairness and continuous batching — on llama-family backends.
- The migrated population's penalty/seed semantics change (documented
  above); default-neutral traffic is unaffected.
- The legacy `Generate()` population shrinks to images fallback, native
  backends, and kill-switch cases.
- Stage 2 introduces a backend-owned sampler registry that inherits the
  sequence-lifecycle discipline (generations, deferred retirement,
  slot-0 edge) — the cost of Stage 2, and the reason it is separate.

## Revision history

- rev 2 (2026-10-21): adversarial design review findings incorporated —
  native-backend and burst/greedy scoping (F1/F7), the seven-gate
  enumeration and admission-charge revert (F2), per-sampled-token list
  semantics with first-token/finalize plumbing (F3), corrected fidelity
  note (F4), greedy-fast-path grammar gating (F5), JSON-validation
  relocation (F6), restated parity gate against real test seams + reuse of
  existing `ComputeLogprob` (F8), streaming logprob threading (F9), images
  bypass clause (F10), stale line references fixed.
- rev 1 (2026-10-21): initial proposal.
