# InferFlux resolution: 2026-10-18 gateway 504 execution timeout

Answers the Victor timeout handoff dated 2026-10-18
(`VICTOR_PREFIX_CACHE_TIMEOUT_HANDOFF_2026-10-18.md`, delivered to the
repository working tree; see also [the September cache acceptance
handoff](CACHE_ACCEPTANCE_HANDOFF_2026-09-19.md) and #184).
Session: 2026-10-18/19, aiserver1, `/home/vsingh/code/inferflux`. Root cause
found, reproduced live, fixed with TDD, and mitigated on the serving unit with a
recorded config-only change. **#184/C5 remains open** for the Mac-owned
actual-member verdict; nothing here replaces Victor-side evidence.

## Identities

| Item | Value |
|---|---|
| Serving unit | user `inferfluxd.service`, restarted **2026-10-18 22:42:31 CDT**, PID 2834176 (was 979, started Oct 15 04:52:39 CDT) |
| Binary (unchanged by mitigation) | `/home/vsingh/.local/share/inferflux/releases/v0.5.0-bfdf32c00ccf/runtime/inferfluxd`, SHA-256 `66f2fa90fef39d7bea8a67b18c275745c9961c12add9da7470eded7cf2c46d71`, verified equal for `/proc/<pid>/exe` before and after |
| Release provenance | manifest `source=bfdf32c00ccf40c7af73b86df2eedabc20a7ae85` (PR #237 promotion; trusted-main mixed-GPU workflow run 37935509802), llama submodule `bb4caa75` |
| Repo refs | checkout stayed detached at `c5c2475c1` with untracked files preserved; fix branch `fix/decode-burst-stateless-generate` from `origin/develop` = `62e191640`; fix commit `41e29411d`; PR **#241** |
| Config | private yaml line 45 `runtime.scheduler.decode_burst_tokens: 8 → 0`; backup `config.private.yaml.pre-decode-burst-mitigation-20261018T2145CDT` beside it; rollback = restore + restart (binary identical) |
| Deployment/models | qwen3-coder-30b Q4_K_XL GGUF on `llama_cpp_rocm`/rocm:0 (ctx 65536, 2 seqs, default), qwen2.5-coder-14b on `llama_cpp_cuda`/cuda:0, bge-small-en-v1.5 on cuda:0 (embedding_batch_size 1) |
| Effective deadline settings | origin: none changed; gateway: stock Sandhi 120 s buffered upstream deadline (observed firing at exactly +120.0 s on both failed calls); no retry/timeout knobs altered anywhere |
| Evidence record | `~/.local/state/inferflux-consolidation-20260922/diagnosis-20261018-decode-burst/evidence.json` (private-side, sanitized) |

## Root cause (verified, not inferred)

The deployed config set `runtime.scheduler.decode_burst_tokens: 8` (a benchmark
knob from March 2026). Burst slicing — and fairness timeslicing through the same
code — assumes every decode pass resumes from retained KV state. Requests forced
onto the **legacy `Generate()`/`GenerateWithImages()` executor path**
(`logprobs` requests, `response_format`/structured output) have no `n_past` /
`sequence_id` continuity: `Generate()` clears sequence 0 and re-evaluates the
whole prompt, and each slice's tokens continue from the *original* prompt. So:

- every 8-token slice re-prefilled the **entire prompt**;
- the loop is bounded only by `max_tokens/8` (1024→128 passes, 512→64);
- the returned text is the same 8-token continuation repeated (garbage);
- one 8,429-token request = **128 full prefills ≈ 9 minutes** of GPU work.

### Correlated failure evidence

- Gateway ledger (private SQLite): Victor's two calls arrived
  `2026-10-19T00:47:00.403Z` / `00:49:00.439Z` (= 19:47/19:49 CDT), each
  settled with zero usage at **exactly +120.0 s** (the buffered 504); same
  session id, model qwen3-coder-30b.
- Origin log: from 19:47 to 20:53 CDT (last write), **640 cycles** of
  full-prefill + exactly-8-token generations: 8,429×128 (call 1, tools),
  7,925×128 (call 2, tools omitted — 504-token delta ≈ the tool schema), plus
  six ~0.9–1.7 k-token co-victim requests ×64 each. The two Victor requests
  **strictly alternate** after 33 solo cycles (~132 s = the gap between the two
  calls) — two concurrent requests round-robined by `min_batch_size=2`.
- A later victor-mac call created 01:40:22Z was served 01:53:44Z (origin
  duration 2.1 s) — an ~800 s queue wait while the origin ground through the
  backlog.
- Usage-side prompt tokens since Oct 15: 87,855; prompt tokens actually
  prefilled in the burst window alone: ~2.55 M.

Historical decode failures in the same log file (`llama_decode failed`,
KV-position inconsistencies, 4 occurrences) predate the current process
generation — the runtime log is append-mode across service generations; they
are preserved as prior-evidence, not part of this incident.

### Live reproduction (pre-fix, same binary)

- `logprobs:true, max_tokens:64`, 18-token prompt → **8 full-prefill cycles**;
  output `"Red, blue, green, yellow,"` repeated six times; finish=length.
- `response_format:json_object` → single pass truncated at the 8-token cap →
  incomplete JSON → "Model output violated response_format constraints"
  (previously misread as a grammar defect; it was the same cap).
- Plain requests took the phased/stepwise path and were unaffected — matching
  why ordinary traffic looked healthy while Victor's calls died.

## Fix

PR #241 (`41e29411d`): slice caps apply only when `n_past`/`sequence_id` give
the next pass a KV-continuous `Decode()` resume; stateless passes run their
whole decode budget in one executor call. Red→green regressions in
`tests/unit/test_unified_batching.cpp` (`[decode_burst]`, burst + timeslice
variants; verified failing on the unfixed executor) and the updated
`tests/unit/test_scheduler.cpp` single-pass contract. CPU: 655 cases / 4,257
assertions, ctest 55/55. CI on #241: ubuntu + macOS MPS (blocking) passed;
advisory compile checks as per rollup. **Trusted-runner GPU gates have not run**
— they are main-only per ADR-0005 and must follow promotion; CPU results do not
establish GPU execution.

## Recorded mitigation and qualification (unchanged limits)

Config-only mitigation applied after evidence preservation (details above);
ready after ~230 s; binary SHA unchanged. Post-mitigation probes, all on the
exact ready model:

| Probe | Result |
|---|---|
| Prior reproduction (logprobs, max_tokens 64) | 1 prefill; coherent output; 1.4 s |
| `response_format:json_object`, agent-shaped, max_tokens 1024 | 1 prefill; complete valid JSON, 80 tokens, 3.1 s |
| **Synthetic Victor-scale** 40,600-byte payload → 9,274 prompt tokens, logprobs, max_tokens 1024, temp 0 | 1 prefill (3.31 s) + 214 coherent JSON tokens, **6.71 s end-to-end** — well inside the 120 s gateway deadline |
| Gateway `https://aiserver1:18788/readyz` | 200 |

The Victor-scale row is **new synthetic evidence**, not a replay of the
unretained original request bodies.

## Cache/lifecycle observations for the next gate

- **Executed reuse works on the phased path**: two identical 718-token plain
  prompts back-to-back reported `cached_tokens` 0 then 717/718 (second call).
  However `inferflux_prefix_hits_total / misses / matched_tokens` for
  `llama_cpp_rocm` stayed 0/0/0 — those Prometheus counters are not wired for
  this backend path; per-request `usage.prompt_tokens_details.cached_tokens`
  is the working signal. Metrics coverage gap, not absent reuse.
- **Logprobs/structured requests are cache-ineligible by design on this
  binary** (the full_generate path sets `cache_reused_tokens = 0` and
  re-prefills). Gateway C5 cache-ratio expectations must account for this:
  DeepSeek-class ratios (≈80%) apply only to plain requests until
  logprobs/structured output joins the phased path (same follow-up as the
  slow-path note below).
- `inferflux_llama_prompt_tokens_total` is inflated by cumulative
  `llama_perf_context` counters across mixed execution paths (phased paths do
  not reset them; the next full-`Generate()` request absorbs the drift). Do not
  use it alone as an amplification measure — pair it with backend log counts.
- The legacy `Generate()` path decoded at ~14–35 ms/token vs ~1.2 ms/token on
  the stepwise path for the same model — a separate follow-up (why logprobs/
  structured requests are parked on the slow path at all).
- Session handles remain disabled; no cancellation/lease claims are made here.
  The 20:53 CDT request demonstrates requests continue executing after client
  deadline expiry (no origin cancellation) — consistent with the prior
  handoffs' boundary.

## Handback to the Mac session

The InferFlux origin now completes Victor-scale logprobs/structured requests in
single-digit seconds with unchanged deadlines. The buffered 120 s policy can
stay. After #241 promotes and the fixed binary deploys, `decode_burst_tokens`
may be restored to a nonzero value if the phased fairness cadence is wanted
(the fix makes that safe); until then it must stay 0 on any binary without #241.
The actual Victor agent/member run, its usage joins and deliverable verdicts —
and the six-Qwen/one-ZAI C5 verdict — remain Mac-owned. **#184 stays open.**

## Closing addendum (2026-10-28): the bystander-eviction fix is accepted

The October 2026 investigation into zero-observed reuse on this server
ended in a fix, a release, and a measurement window; this addendum closes
the chain with the verdict and the evidence.

**Root cause.** Slot-pressure admission retires LRU radix donors
(`Scheduler::AllocSeqSlot`); `ProcessBatch` then discarded the request's
own prefix candidate after ANY eviction, although `Lookup` refreshes
`last_used` on the nodes it walks — the victim is almost always an
unrelated bystander whose eviction leaves the matched donor, its backend
sequence, and its block references fully intact. Fixed by PR #276
(`EvictOneSequence` reports the retired id; the candidate is discarded
only when the matched donor itself was retired; diagnostic stages split
`capacity_eviction_donor`/`capacity_eviction_bystander`). Shipped in
v0.6.0 (promoted, dual-GPU gates green at `9a09cfa`, aiserver1 deployed
from the exact tagged SHA 2026-10-27).

**Evidence.**
- Probe signature before the fix: three identical 433-token requests
  reported `cached_tokens` 0, 0, 432 (the first repeat always fell into
  the discard; walls 0.21–0.22 s). After: 432, 432, 432 with 0.10 s
  walls; gpt-oss-20b: 0, 491, 491.
- Production window (43.6 h post-deploy): organic traffic on this
  homelab server is near-zero (15 requests total), so the ratio cannot
  be read from organic traffic — recorded as a finding, not a gap.
- Bounded multi-turn replay (the victor pattern: each turn re-sends the
  full conversation; 3 conversations × 4 turns, growing 658→943-token
  prefixes): **9 of 9 lookup hits accepted as reuse (100%), 6,777
  tokens skipped**; every multi-turn turn reused the entire prior
  prefix. Logs: aiserver1 `/tmp/replay_window.py` output in the closing
  PR body.

**ADR-0008 trigger verdict.** The recorded implementation trigger
("accepted-reuse ratio < 50% of lookup hits on repeat-heavy traffic")
**does not fire** — the measured ratio is 100%. The serialized KV tier
stays data-gated and unimplemented; revisit only if a future workload
shows sustained loss the fix cannot explain.

**Companion hardening shipped with the window:** eviction-decision
counters `inferflux_capacity_eviction_donor/bystander_total` (#284) make
the discard split observable without the env-gated decision log, and the
honest `model_in_use` unload error (#284, issue #283) removed the
misleading 404 that originally obscured runtime model swap during the
bake-off.
