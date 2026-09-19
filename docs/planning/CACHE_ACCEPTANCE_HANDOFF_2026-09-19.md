# Cache co-design: remaining acceptance and session handoff

This supersedes the implementation-pending status in the dated September 18
handoff. InferFlux PRs #185–#190 are merged into `develop` at `868c6ba5f`.
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
[269](https://github.com/anvai-labs/sandhi/pull/269). C4 bounded diagnostics and C5
actual-member joint acceptance are tracked by Sandhi TD-0028. C4 can be implemented
and tested on aiserver1 independently of the main promotion and Mac credentials.

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
