# Cache co-design: remaining acceptance and session handoff

Current status: [September 19 acceptance addendum](#september-19-cache-acceptance-addendum--current-acceptance). The historical evidence below retains its original provenance and limitations.

This supersedes the implementation-pending status in the dated September 18
handoff. InferFlux PRs #185–#191 are merged into `develop` at `c13d3a52e`.
The independently reviewed promotion [PR #192](https://github.com/anvai-labs/inferflux/pull/192)
merged into `main` at `26d96fbe3` after green hosted CI. Runtime evidence remains
a separate exact-main-SHA gate.
The current execution tracker is [issue 184](https://github.com/anvai-labs/inferflux/issues/184).
The [replay report](CACHE_REPLAY_2026-09-18.md) records the evidence and its limits.

## Review follow-up before promotion

Independent review found an additional warm-session failure case: failed deferred
prefill retired its active sequence and pages but left a copied retained table in
the session manager. Expiry or shutdown could release those pages again after
another owner recycled them. The focused fix discards that retained state before
retiring the active references. It preserves a retained state when admission never
transferred it to an active sequence.

The regression covers warm success followed by an appended request with empty,
failed, or cancelled prefill, intermediate/final chunk sizes, cold retry, and
teardown after all pages have been recycled. Before the fix, teardown freed four
pages belonging to another owner; after it, all 114 assertions pass. This is
another verified defect, not retrospective proof of the original 40-call cause.

## Prompt-token units after runtime review

A new CPU writer run exposed a pre-existing measurement mismatch: phased responses
reported SimpleTokenizer word/punctuation counts, while cache reuse used backend
BPE tokens. Its first request reported 2,167 prompt tokens for a 1,765-token backend
prompt. Wire/SQLite/dashboard conservation established reporting consistency, not
tokenizer-exact totals. This is tracked in [issue 193](https://github.com/anvai-labs/inferflux/issues/193).

The follow-up reports text prompt usage from the backend-tokenized input sequence,
including that backend's BOS policy. Phased, deferred, split-decode and full-generation
paths retain the same count; fairness charges the original prompt once across slices.
Scheduling/admission token estimates are unchanged. Unavailable tokenization and
multimodal requests retain their previous fallback; this does not define image-token
accounting or change completion token counting.

The regression leaves the real admission path intact: a three-token SimpleTokenizer
estimate represents 40 backend tokens and 39 accepted cached tokens. Before the fix,
usage incorrectly reported 3/3; it now reports 40/39. Earlier runtime evidence below
and in local artifacts must retain its pre-fix provenance.

## Dependencies and priority

1. Independently review the final repair and promotion commit before pushing.
   Run hosted CI, resolve findings/failures, and satisfy the protected `main`
   approving-review requirement. Keep existing main-only release changes.
   Promotion alone does not authorize a version/tag/package publication.
2. After promotion, obtain exact-main-SHA evidence from `GPU Behavioral Gates`.
   The trusted `aiserver1-dual-gpu` runner must execute CUDA then ROCm serially.
   See [ADR-0005](../adr/ADR-0005-trusted-gpu-release-evidence.md). Do not dispatch
   a feature/develop revision through the trusted-main runner restriction.
3. Own the diagnostic deployment and rollback, then capture an actual local
   member's successive requests. Preserve correlation, hashed token/session
   identity, exact tokenized common-prefix length, accepted reuse, execution
   mode, sequence generation, lease state, and copy/eviction decisions.
4. Run direct/gateway comparisons and the acceptance matrix: unique/repeat/append,
   stream/non-stream, tools/JSON/logprobs, same/different sessions, contention,
   eviction/model changes, copy failure and cancellation. Check inclusive prompt
   counts, fresh+cached conservation, reporting availability and timing sources.
5. Finish the full six-Qwen/one-ZAI WS-E run using the original session's approved
   ZAI connection. A local single-writer replay does not close this gate.

Sandhi C1 regression fixtures and C2/C3 availability/dashboard changes are merged
through PRs [268](https://github.com/anvai-labs/sandhi/pull/268) and
[269](https://github.com/anvai-labs/sandhi/pull/269). C4 bounded diagnostics and its
handoff are also merged through [270](https://github.com/anvai-labs/sandhi/pull/270)
and [271](https://github.com/anvai-labs/sandhi/pull/271), at `647d7d5`.
Use `POST /admin/usage/diagnostics` or `sandhi diagnose --request|--session|--run`.
This is an admin-only bounded projection of persisted rows, with no pagination or
prompt/body capture. It explicitly identifies unavailable historical lifecycle
metadata. C5 actual-member target-runtime acceptance remains tracked by TD-0028.

## Preserved service and rollback boundary

The verified serving process is PID 554660, cwd
`/home/vsingh/code/inferflux-worktrees/fix-victor-codesign`, source `aea7a24d0`.
Its executable is `build-rocm/inferfluxd`, config
`config/server.rocm.qwen3coder30b.yaml`, port 8080, model `qwen3-coder-30b`.
`INFERFLUX_LLAMA_CTX_SIZE=65536` and two sequences are configured; session handles
are disabled. Its output is `/tmp/inferflux_qwen3coder.log`.

Keep this executable/config/worktree intact. Before any replacement, verify their
identity again, record all launch settings privately, and arrange a controlled
handover with a health check and return to the preserved executable on failure.
Never copy credential-bearing environment/config contents into the handoff.
Do not kill/restart Qwen or flush its cache merely to obtain a cold sample.
If Qwen must stay uninterrupted and a second instance cannot fit, defer GPU
deployment to an agreed handover; CPU/model-free evidence has a different scope.

The trusted GPU gate uses the separate pinned model
`/home/vsingh/code/inferflux/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
with CUDA/ROCm listeners on 18081/18082. Confirm those ports are free and check
GPU headroom before the gate; existing Qwen health must remain good throughout.
Integration tests also use port 18083. Keep temporary acceptance services outside
the test port range, for example on 28083, and verify the chosen port is free.

## Local and Mac connection recipes (no credentials)

On aiserver1, the reusable member replay is `scripts/cache_member_replay.py`.
It launches an isolated loopback Sandhi gateway at 18789 and observer at 18084;
both ports can be overridden. Its upstream defaults to `http://127.0.0.1:8080`.
Use a Victor-capable Python environment, a freshly built Sandhi proxy, and keys
supplied privately via the documented environment variables. The current observer
buffers SSE, so it must not be used to claim cancellation/streaming-latency coverage.

The original Mac gateway was `http://127.0.0.1:18788`, dashboard `/dashboard`.
Its InferFlux tunnel was Mac `127.0.0.1:18080` to `aiserver1:8080`.
Retained state was `/Users/vijaysingh/code/codingagent/var/sandhi-zai/`.
These are historical Mac-local locations, not aiserver1 loopback endpoints; the
originating session must verify their current state without sharing credentials.

Victor's full harness is `scripts/validation/multiagent_gateway_live.py --mixed`.
Use its current `--help` for gateway-state/output/proxy-port arguments and the
repository's documented environment. Match the approved Qwen/ZAI models and
preserve distinct member sessions. Transfer only bounded, sanitized evidence:
repo/binary/config identities, hashes/correlation IDs, counters, reporting
availability, timing sources, and generated-deliverable/test verdicts.

## Evidence needed to close the tracker

Record reviewed commit/tree identities, hosted CI URLs, exact-SHA CUDA/ROCm gate
URLs/artifacts, post-deployment model/capacity responses, member diagnostic events,
wire-to-ledger-to-dashboard reconciliation, cancellation/late-usage results and
the full mixed-team verdict. Original request bodies were not retained: explain
new reproductions on their own evidence and keep historical causality unproven.

## Disconnect acceptance follow-up

An unbuffered early-disconnect check against an owned CPU Qwen server caused that
process to exit. A separate model-free subprocess reproduced termination by
SIGPIPE (exit -13) after the peer closed its SSE connection. This is a verified
HTTP response-lifecycle defect, separate from cache accounting and the original
40-call symptom. The shared GPU service was not used for this reproduction.

The repair suppresses broken-pipe signals on server-owned sockets where the OS
provides that option, or scopes signal blocking/draining to the current socket
operation's thread. It covers plain response writes and TLS writes, reads,
handshakes and shutdown, preserves the caller's signal state and errno, and uses
a nonblocking drain. Failed writes still reach the existing cancellation path.
A subprocess regression exercises plain/TLS closed peers, default signal
disposition, pre-existing pending signals and a subsequent healthy connection.
Rollback is a revert of this HTTP-only repair; it does not change model or cache
configuration. Runtime cancellation acceptance remains distinct from observing
client-side closure or late gateway usage.

## September 19 cache acceptance addendum — current acceptance

This addendum records the current verified scope. Earlier sections remain historical.
The trusted-main GPU gate, isolated deployment and frozen-member replay passed.
**C5 streaming accounting remains open:** terminal wire output does not match
Sandhi's persisted output counts. Final acceptance requires the Sandhi repair and
streaming rerun. The bounded disconnect/recovery result is independently clean.
Historical and failed attempts retain their original scope.

| Activity | Status |
|---|---|
| Prompt-token accounting #194/#195 | Merged: develop `930f580e0`, main `9edab96b4`; issue #193 remains open for its remaining acceptance scope. |
| Trusted-runner PATH #196/#197 | Merged: develop `d33494c06`, main `67689ef4b`. |
| Ordinary serialized CUDA → ROCm gates | Passed at [9edab96b4 / 35420962763](https://github.com/anvai-labs/inferflux/actions/runs/35420962763) and [67689ef4b / 35425803441](https://github.com/anvai-labs/inferflux/actions/runs/35425803441). Small-model gates do not establish Qwen deployment acceptance. |
| Frozen five-payload direct/Sandhi CPU acceptance | Passed: ten responses and five gateway rows reconciled; instances now stopped. |
| Unbuffered disconnect/recovery CPU acceptance | Passed for all four direct/gateway calls on the HTTP repair; instances now stopped. |
| Optional Qwen GPU workflow #198/#199 | Merged: develop `f37d0d655ef58aae6da7f326c4fc02daf46431c2`, main `8efae1facde551d55dc641441d057a1b0dae6e68`, after independent review before pushing and all required hosted CI passed before merging. Exact-main runtime acceptance passed at `c5d4eb89f`. |
| HTTP SIGPIPE repair #200/#201 | Merged: develop `38d02f4b9ac1cec20614f0d9775f84408de32447`, main `252c78ef4961ecc8ecec1815ec48bd6c87320bb3`, after independent review before pushing and all required hosted CI passed before merging. |
| First optional Qwen GPU attempt | [Run 35429444326](https://github.com/anvai-labs/inferflux/actions/runs/35429444326) at `252c78ef4961ecc8ecec1815ec48bd6c87320bb3` failed aggregate acceptance: optional Qwen reconciliation rejected the harness's incorrect backend name. CUDA compile/unit/ordinary-model checks and ROCm passed. |
| Harness correction #202/#203 | Independently reviewed feature `4293cdabe` / promotion `21ec6dc7c`; 19 CPU contracts passed. All required hosted CI passed before merge: develop `b32a8f0a208f4975a949b7a5d64725f2a591dcef`, main `c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1`. Corrected trusted-main acceptance [35431794400](https://github.com/anvai-labs/inferflux/actions/runs/35431794400) and main hosted CI [35431758591](https://github.com/anvai-labs/inferflux/actions/runs/35431758591) passed. |
| Updated Qwen GPU serving target | Ready on `aiserver1:8081`, PID 2254917, instance `c527a7a4d4eb`, from accepted main `c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1`. Identical-binary rollback/relaunch was rehearsed; shared 8080 remained unchanged. |
| Deployed frozen-member acceptance | Retry `6ge9ssew` passed and independent review is clean: ten wire calls, 36 diagnostics and five SQLite/C4/wire/dashboard joins. Gateway cache total was zero; six explicit capacity evictions explain this run. The earlier `zf2anjr6` timeout remains a preserved failed attempt. |
| Deployed 24-case matrix | Passed; independent audit is clean. Twelve streaming and twelve non-streaming cases, four positive-cache responses and 1,906 cached tokens match global and per-model counter deltas. |
| Gateway streaming accounting / C5 | **Open:** all five wire streams completed validly, but SQLite output totals were 14,387 versus 288 reported on wire. Sandhi owns the repair and regression; repeat deployed streaming acceptance after its reviewed fix. |
| Deployed disconnect/recovery | Four-call probe passed for observed early client closure and successful recovery; independent review is clean. No origin-cancellation claim. |
| Full six-Qwen/one-ZAI run | Mac engineering can resume using its approved connection and the ready sidecar. Final acceptance still requires the Sandhi streaming fix/rerun and completed full-team evidence. |

### Failed first GPU attempt and correction

Trusted-main [run 35429444326](https://github.com/anvai-labs/inferflux/actions/runs/35429444326)
used source `252c78ef4961ecc8ecec1815ec48bd6c87320bb3`. CUDA compilation, unit tests
and the ordinary small-model gate passed; the optional Qwen step failed with
`backend_execution_identity`. ROCm passed; the aggregate workflow failed.
The harness required the nonexistent identity `llama_cuda`, while the actual
`CUDABackend::Name()` and scheduler diagnostic identity are `llama_cpp_cuda`.
Ten direct/gateway calls and the SQLite/C4/dashboard checks completed before this
assertion. Those facts do not constitute accepted GPU replay or deployment.

The unaccepted executable SHA-256 was
`431922fed0ec9c27c2d24337cf1641b503fe78644b0ffb723560266479e706b8`.
Preserved failure artifact:
`/tmp/inferflux-qwen-gate-35429444326/cache-acceptance.json`.
The first failed artifact does not retain per-call counters or diagnostic rows.
Completion of ten calls and gateway reconciliation is inferred from the failure
location in that exact harness, rather than preserved per-call evidence.
Owned processes stopped and shared port 8080 remained healthy. This failed
attempt is separate from the accepted source, gate and deployment recorded below.

[PR #202](https://github.com/anvai-labs/inferflux/pull/202) (`4293cdabe`) and
[promotion #203](https://github.com/anvai-labs/inferflux/pull/203) (`21ec6dc7c`)
correct that identity and preserve completed calls plus bounded allowlisted
failure diagnostics. Their 19 CPU contracts include checks against actual C++
backend names and scheduler stage literals. Independent review was clean and all
required hosted CI passed before both PRs merged. See the [coordination update](https://github.com/anvai-labs/inferflux/issues/184#issuecomment-5740307253).

### Accepted GPU gate and isolated deployment

Accepted main source: `c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1`.
[Main hosted CI 35431758591](https://github.com/anvai-labs/inferflux/actions/runs/35431758591)
and [trusted GPU run 35431794400](https://github.com/anvai-labs/inferflux/actions/runs/35431794400)
passed. The latter includes CUDA, optional frozen Qwen replay, serialized ROCm and
the aggregate gate. Its independently audited artifact is
`/tmp/inferflux-qwen-gate-35431794400/cache-acceptance.json`.
All ten wire calls and 34 diagnostic records reconcile; exact backend prompt and
accepted cache counts match the CPU per-ordinal table below. The gateway reports
**11,501 fresh + 1,700 cached = 13,201 inclusive prompt tokens; 740 output tokens**.
Four explicit capacity-eviction records explain this run's four gateway cache zeros.
That measured cause is not retrospective proof of the original forty-call symptom.

Accepted executable SHA-256:
`312f49e63dae7609dd804e3b74aa4a4b2dd06b72aebf8aa6a322ed619e974de6`.
Immutable deployment bundle:
`/home/vsingh/code/inferflux/build-accepted-qwen-cuda/c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1`.
The ready sidecar on port 8081 is PID 2254917, instance `c527a7a4d4eb`, using
`llama_cpp_cuda` with eight CUDA layers plus CPU, two sequences with 16,384 context
tokens each, and session handles disabled. The deployment uses that same accepted
executable fingerprint; it does not replace the shared ROCm service on port 8080.

Rollback rehearsal stopped the first sidecar PID 2245858, verified port 8081
closed, then relaunched the same accepted binary as PID 2254917. Evidence:
`/tmp/inferflux-c5-rollback-rehearsal.json`. Baseline PID 554660, executable and
configuration hashes, process start ticks and health remained unchanged.

### Post-deployment timeout and verified retry

The first post-deployment replay failed its 180-second I/O timeout after six
completed wire calls, on direct ordinal 3. Preserve its failed artifact:
`/tmp/inferflux-frozen-member-joint-zf2anjr6/acceptance.json`.
The same origin PID 2254917 stayed healthy and later finalized the outstanding
request. Its reported completion counter increased by 4,096, matching that
request's `max_tokens=4096`; this is not proof of exact completion tokenization or
origin cancellation. Observation record:
`/tmp/inferflux-c5-timeout-observation.json`.

Independent review found gate and sidecar configurations identical except for
port. The unchanged captured payloads use temperature 0.7 without a seed and allow
up to 4,096 output tokens, so completion lengths can vary. This timeout does not
establish a cache defect or explain the historical forty-call symptom. The failed
post-deployment attempt remains distinct from the successful trusted-main gate.

The reviewed retry used a wrapper allowing 900 seconds per I/O and 3,600 seconds
overall, with a bounded diagnostic reader that handles log rotation. The same
five payloads on the same serving instance passed at
`/tmp/inferflux-frozen-member-joint-6ge9ssew/acceptance.json`; independent review
is clean. The cache was warmed by the failed attempt, and no cache clear occurred,
so the retry is not an independent or cold-cache control. It retained 36 diagnostic
records within the 41 remaining slots of the process's 64-event budget.
The pinned Sandhi binary has a separate 120-second buffered upstream timeout,
with no stock per-provider override; the wrapper's 900-second I/O timeout does
not change that gateway limit. No gateway timeout occurred in the successful replay.

All ten wire calls reconciled with backend diagnostics. Five gateway requests
joined uniquely through SQLite, bounded C4 export, wire usage and dashboard:
**13,201 fresh + 0 cached = 13,201 inclusive prompt tokens; 751 output tokens**.
All five cache observations were reported origin usage, including reported zeros.
Direct accepted cache counts were **0 / 1,701 / 1,932 / 2,443 / 3,099**, totaling
**9,175**. Six capacity evictions occurred: direct ordinal 0 and every gateway
request each had an exact-length lookup, one evicted sequence and zero accepted/
completed reuse. The remaining four direct calls reused their growing prefixes.

This reproduces a zero-dashboard-cache symptom on the deployed runtime while
also observing positive direct reuse. The retained capacity-eviction decisions
explain this run's zeros; they do not establish the original forty-call cause.
Lookup candidates were not counted as accepted reuse.

### Deployed matrix and open streaming-accounting defect

The deployed 24-case matrix passed at
`/tmp/inferflux-deployed-cache-matrix-5oosv1hs/summary.json`; independent audit is
clean. Twelve streaming and twelve non-streaming cases produced four positive
cache responses totaling 1,906 cached tokens, exactly matching global and per-model
metric deltas. The matrix does not establish strict SSE terminal-protocol behavior.

The separate five-call gateway SSE/overlap probe **failed accounting acceptance**:
`/tmp/inferflux-deployed-stream-overlap-fh_gryvj/report.json`, with
`failure-supplement.md` and `failure-supplement.json` in the same directory.
All five streams returned HTTP 200, a finish event, exactly one terminal usage
object and DONE. Measured submitted/in-flight overlap was 5,279.3 ms for the same
session and 5,322.575 ms for different sessions. Delivered content was serialized;
that is valid and does not itself fail the overlap check or prove concurrent GPU
execution.

| Calls | Wire completion tokens per call | SQLite output tokens per call |
|---|---:|---:|
| Single stream | 32 | 1,655 |
| Four overlapping requests | 64 | 3,183 |
| Total across five calls | 288 | 14,387 |

Prompt tokens matched at 24 per request and cache tokens matched at zero. The
harness failed its C4 comparison before preserving a C4 page or querying the
dashboard. The supplement independently captures **SQLite evidence only**; no
captured C4/dashboard response or aggregate is claimed for this failed probe.
`reported/origin_usage` describes cache-read availability, not proof that output
counts are final provider-reported values.

Inspection of pinned Sandhi `647d7d5` strongly supports an EOF-finalization
explanation: terminal usage and DONE are forwarded before transport EOF finalizes
metering. Closing at DONE can retain a partial raw-SSE-byte output estimate; all
five persisted values match that estimate. This remains a **strong inference**,
since no physical-attempt EOF trace was retained. The originating Sandhi session
owns the fix and deterministic fake-upstream regression: emit known terminal
usage and DONE, delay EOF, then compare immediate client close with draining EOF
across chunk boundaries. Preserve real preterminal-disconnect semantics.
The finding and regression handoff were posted to the
[originating Sandhi session](https://github.com/anvai-labs/sandhi/pull/272#issuecomment-5740849386).

**C5 streaming accounting remains open.** Merge the independently reviewed Sandhi
repair after green CI, then rerun this five-call deployed oracle and reconcile
wire, persisted rows, C4 and dashboard before closing that acceptance dependency.
Do not relax the accounting comparison to make the current failure pass.

The separate four-call deployed disconnect/recovery probe passed at
`/tmp/inferflux-deployed-cancellation-8jvdiobs/report.json`; independent review is
clean. Both clients closed after first content and before observing terminal
usage/finish/DONE; both subsequent recoveries succeeded and the origin stayed
healthy. This supports client early closure and recovery only, not origin/backend
cancellation or delivery of terminal usage.

### Verified CPU frozen-member result

The approved capture `/tmp/victor-member-replay-63ad80a3f502` supplies five ordered
actual-member payloads. All twelve listed checksums remained unchanged. Payloads
were replayed directly, then through Sandhi at each ordinal; generated responses
were not fed into subsequent requests and no captured or returned tools executed.
The direct and gateway arms shared the isolated origin cache; neither arm is a
cold-cache or independent-cache control.

| Ordinal | Backend prompt tokens | Direct accepted cache | Gateway accepted cache |
|---|---:|---:|---:|
| 0 | 1701 | 0 | 1700 |
| 1 | 1932 | 1701 | 0 |
| 2 | 2443 | 1932 | 0 |
| 3 | 3099 | 2443 | 0 |
| 4 | 4026 | 3099 | 0 |

All ten wire counts matched backend prompt lengths and accepted/finalized reuse;
paired token hashes matched. Five gateway rows joined uniquely through SQLite,
bounded C4 export and dashboard: **11,501 fresh + 1,700 cached = 13,201 inclusive
prompt tokens; 764 output tokens**. Four gateway zero-cache responses followed
logged donor-capacity eviction. This explains those new CPU calls, not the original
forty-call symptom. Zero dashboard cache tokens alone do not prove absent caching.

API backend: `llama_cpp_cpu` (diagnostic backend name `llama_cpu`), two sequences,
16,384 context tokens per sequence,
no backend fallback, **session handles disabled**. Session/token hash correlation
was verified; this is not evidence of active session leases or warm lease reuse.
Source: `486bf2a0b99a97e6cb02ba1b078c61aceb96e089`.
Origin executable SHA-256:
`b092852d18abb302d6d69f1cd081aa98f06a3ba723c6a6b466610ea84c4d9259`.
Sandhi source: `647d7d5df064b3b3c3e74ee8635521f82d3552cb`; executable SHA-256:
`1d3c234aa10f11a9a6a70e4888d10b314f8fe0972b3586b709b972118b2aee8e`.
Evidence: `/tmp/inferflux-frozen-member-joint-pkioglpo/acceptance.json` and adjacent
sanitized projections. Port 28083 and its temporary gateway are no longer serving.

### Disconnect result and validation limits

The separate model-free reproduction terminated by SIGPIPE (exit `-13`) after a
peer disconnected. The reviewed HTTP-only repair was tested on an owned CPU Qwen
at source `ebfa7ea75631ddfbb8abe196537404725cede442`. Both direct and gateway clients
closed after first content, before observing terminal usage/finish/DONE; both
subsequent recovery calls completed and the origin remained healthy.
Evidence: `/tmp/inferflux-cpu-cancellation-s6vuhclk/report.json` (four calls).

C4 persisted the disconnected gateway row with input 0, output 48, cache 0 and
`cache_read_observation=absent/origin_usage`; recovery reported origin usage and
conserved counts. Completeness, basis, outcome and physical-attempt provenance
remain unavailable. This proves observed client closure, reporting availability
and recovery, **not provider cancellation or delivery of terminal usage**.
Sandhi TD-0028 retains that lifecycle boundary; no new Sandhi runtime patch is
claimed. HTTP repair validation passed 47 CPU CTest targets; the combined HTTP/
optional-workflow stack passed 48. The initially merged harness passed 15 CPU
contracts; the reviewed correction above passes 19. All required hosted CI passed
for all merged changes, including #202/#203. Accepted GPU evidence is recorded
above; the deployed frozen replay passed independently reviewed acceptance.
The matrix and bounded deployed disconnect/recovery result are independently
clean; C5 streaming accounting remains open.

### Remaining runtime coverage

The optional Qwen target uses partial CUDA offload; no Qwen ROCm acceptance or
full GPU offload is claimed. Session handles remain disabled, so correlation
checks do not validate active leases. GPU copy-failure and model-swap fault
injection remain untested. The disconnect checks do not prove backend/provider
cancellation; origin TTFT provenance and exact completion-token accounting also
remain unverified. Gateway terminal SSE protocol and same/different-session
submitted-request overlap were observed, but persisted streaming output accounting
failed and remains open. Delivered-stream overlap is optional; valid serialization
is not a failure. The full six-Qwen/one-ZAI mixed team still requires the originating
Mac session's approved connection and cannot close final acceptance before the
Sandhi repair and streaming rerun.

### Durable evidence location

Sanitized evidence is archived under
`/home/vsingh/code/inferflux/build-accepted-qwen-cuda/c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1/evidence/`.
The local `/tmp` paths above identify the original attempts and their distinct
success/failure scope. All 16 metadata JSON files plus README and manifest passed
hash verification and private-content checks. Manifest SHA-256:
`987b579cb6c3701d35f263d19555dea5eb548def29e7b19f93f66c7b1611b4c2`.
Credentials, private gateway stores, prompts, raw logs and response bodies are
excluded; originals remain unchanged. These artifacts are local and not committed.

### Preserved service and originating-Mac handoff

Shared Qwen remains port 8080, PID 554660, associated checkout
`aea7a24d035fd386ce24db4f8cc68710b4b11543`, cwd
`/home/vsingh/code/inferflux-worktrees/fix-victor-codesign`, executable
`build-rocm/inferfluxd`, config `config/server.rocm.qwen3coder30b.yaml`.
Verified `/proc/554660/exe` SHA-256:
`cff60d525a60d26574013696f012d27a63a1ad1c7cfa1cbde634bbec19ed499d`.
Checkout association and executable fingerprint are separate facts; no embedded
build attestation is claimed. No shared restart, reconfiguration, model replacement
or cache clear occurred. Isolated CPU success did not upgrade this service.

Final verification after all probes is recorded in
`/tmp/inferflux-c5-final-runtime.json`: both services were ready. Baseline PID
554660 retained start ticks `9455753`, its executable fingerprint, configuration
hash and associated checkout. Accepted sidecar PID 2254917 retained start ticks
`28515471` and executable fingerprint
`312f49e63dae7609dd804e3b74aa4a4b2dd06b72aebf8aa6a322ed619e974de6`.

Coordination records: [Mac readiness](https://github.com/anvai-labs/inferflux/issues/184#issuecomment-5739881527),
[accepted-runtime owner handback](https://github.com/anvai-labs/inferflux/issues/184#issuecomment-5740855755),
and [Sandhi #272 follow-up](https://github.com/anvai-labs/sandhi/pull/272#issuecomment-5739991254).
Sandhi #272 (`566f8199a79c64495fbacc18c169581d996d5c08`) documents the new baseline;
its runtime-relative changes are documentation only. The accepted GPU serving
identity, endpoint and deployment/rollback record are available above. Mac
engineering may resume, but final acceptance remains blocked on the Sandhi
streaming repair/rerun and full-team evidence.

The former missing-new-payload dependency is resolved by the available bundle.
The original forty upstream bodies were not retained and historical causality
remains unproven. The Mac session can use the accepted sidecar for its approved
member replay, then execute Victor's
`scripts/validation/multiagent_gateway_live.py --mixed` for six Qwen/one ZAI.
Historical Mac locations to verify: gateway `127.0.0.1:18788`, state
`/Users/vijaysingh/code/codingagent/var/sandhi-zai/`, tunnel `127.0.0.1:18080` →
`aiserver1:8080`. That tunnel still targets the old service; do not label its runs
updated-runtime acceptance. Confirm the accepted endpoint before changing routing.
For the accepted target, open a separate tunnel from the originating Mac:

```bash
ssh -N -L 18081:127.0.0.1:8081 aiserver1
```

Set the local Sandhi InferFlux upstream to `http://127.0.0.1:18081/v1`, reusing the
original private InferFlux key already held in that session. Do not copy credentials
into the handoff. Keep the original `18080` tunnel available for rollback to 8080.
To stop only the accepted sidecar on aiserver1:

```bash
bundle=/home/vsingh/code/inferflux/build-accepted-qwen-cuda/c5d4eb89f71dcfb3064b974d63b88c4ce88f42a1
/usr/bin/python3 "$bundle/sidecar.py" stop --bundle "$bundle"
```

Then restore the Mac Sandhi upstream to its original `http://127.0.0.1:18080/v1`.
Use the existing approved ZAI connection locally; transfer no credentials.
Return only bounded sanitized identities, counts, timing/availability sources,
reconciliation, lifecycle limits and generated-deliverable/test verdicts.
