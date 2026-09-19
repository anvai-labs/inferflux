# Cache replay and implementation evidence

The original WS-E zero-cache workload remains **unattributed**. Its original
upstream bodies were not retained. A controlled wire probe is not an exact replay
of those calls, and dashboard zeros alone do not establish missing caching.

## Verified serving identity

On 2026-09-18, PID 554660 served port 8080 from
`/home/vsingh/code/inferflux-worktrees/fix-victor-codesign`, clean commit
`aea7a24d035fd386ce24db4f8cc68710b4b11543`. Its launch command was
`./build-rocm/inferfluxd --config config/server.rocm.qwen3coder30b.yaml`.
`/v1/models` reported ready `qwen3-coder-30b`, `llama_cpp_rocm`, fallback false.
The process had `INFERFLUX_LLAMA_CTX_SIZE=65536`; the YAML set two parallel
sequences. Startup logs confirmed session handles disabled (TTL 300000 ms,
limit 1024); no session environment override was present. The model's metadata
context length 262144 is not the configured runtime context extent.

The development checkout started at `a9f9a720f`. Qwen was not restarted,
unloaded, or reconfigured. No shared cache was cleared. Runner 9054,
`aiserver1-dual-gpu`, was online and idle at the initial check.

## Bounded wire probe

```bash
python3 scripts/cache_reuse_probe.py --model qwen3-coder-30b \
  --output /tmp/inferflux-cache-probe.json
```

Set `INFERCTL_API_KEY` privately before invoking it. The probe sends 24 sequential
requests: plain/tools/JSON/logprobs, stream/non-stream, unique prefix/repeat/append.
It stores correlation IDs, system-prefix SHA-256, and inclusive wire usage only.
A unique prefix is not a guarantee of a completely cold template prefix, and
production traffic can evict entries between requests.

Run `cache-probe-20ea69f73e58` produced these cached-token counts in both streaming
modes (unique/repeat/append):

| Shape | Cached tokens |
|---|---|
| Plain | 3 / 3 / 3 |
| Tools | 0 / 0 / 0 |
| JSON object | 0 / 0 / 0 |
| Logprobs | 0 / 0 / 0 |

Credential-free raw evidence is at
`/tmp/inferflux-cache-probe-20260918.json` on the rig. The results differ from the
earlier handoff's warm plain/tool probe. They establish a current symptom, not its
cause. Every response satisfied `0 <= cached_tokens <= prompt_tokens`.

## Original workload replay

Use Victor's `scripts/validation/multiagent_gateway_live.py --mixed` from merged
commit `27c63afa1f640a24a1548420a7be1ff3a395322f`
([Victor PR 1115](https://github.com/anvai-labs/victor/pull/1115)). The harness
requires the retained Sandhi gateway state and a working Victor environment.
After fetching both repositories, the local writer member was replayed from
Victor `609aaf5fb` through an isolated Sandhi `91a08d1` gateway built locally.
The replay used the WS-E writer goal, provider/model, tools, reasoning effort,
and budgets in a one-member pipeline. It is a local-member reproduction, not the
original six-member/ZAI concurrency run. No ZAI credential is configured here.

Run `inferflux-member-5430646e95` passed its writer deliverables and independent
pytest check. Four successive same-session requests used three tools and neither
structured output nor logprobs. Their inclusive input/output/cache usage was
1731/55/0, 1785/65/0, 1839/67/0, and 1961/110/0. Sandhi SQLite reconciled every
row exactly: fresh + cached = upstream prompt, and output counts matched.
Evidence, ephemeral gateway state, and generated test files are under
`/tmp/inferflux-member-5430646e95`; the gateway was stopped afterward.

At the post-run snapshot, the unchanged serving process had only 24 free paged
blocks out of 4096, versus 34 blocks retained by one live prefix sequence. The
aggregate KV reuse counters were still 28 requests / 50747 tokens, unchanged from
the earlier handoff. This establishes resource pressure and a current reproduction;
it does not retrospectively identify the cause of the original 40 calls. The
wire observer hashed messages and measured serialized-message common-prefix
bytes (2499, 2961, 3478); **these are not tokenized common-prefix lengths**.
Tokenized comparisons and exact allocation/copy decisions require the diagnostic
build, which has not been deployed to the shared GPU service.

For a separately scheduled deployment of the diagnostic build, set
`INFERFLUX_CACHE_DIAGNOSTIC_REQUEST_PREFIX` to the correlation prefix of one local
member's requests. Capture emits at most 64 JSON events per process, requires a
nonempty matching `x-inferflux-client-request-id`, and hashes token sequences and
session IDs. It records policy bypass, lookup common-prefix length, copy failure,
partial-prefill failure, and accepted synchronous prefill reuse with sequence
generation. Lookup length is a candidate, never proof of successful reuse.
The token hash uses comma-separated decimal token IDs before SHA-256.

These events are request diagnostics, not distributed traces. An OTEL collector
requires explicit configuration. Capture does not expose prompt text or token
IDs. The diagnostic binary has not been deployed to the shared Qwen service;
deployment requires the owned launch recipe and rollback. Disable the environment
setting or revert the diagnostic commit to remove capture.

GPU runtime evidence must be collected on trusted main through the serialized
`aiserver1-dual-gpu` gates; do not run pull-request code on that runner. CPU tests
and production-server wire probes do not certify a modified GPU binary.

## Execution accounting increment

Usage now reads the accepted prefill reuse extent, independently of the radix
lookup candidate. Exact hits replay one token for logits and report one fewer
cached token. Failed copies, unavailable donors, full-prefill recovery, and
fallback to full Generate report zero. Deferred prefill checks copy success and
commits the pending reuse extent only after its last successful prefill chunk.
The scheduler records the aggregate KV reuse counters from the same finalized
value as response usage, once per completed request, excluding zero-token reuse.

Deterministic scheduler cases cover synchronous/deferred exact hits and failed
copies, matches without donors, full-prefill recovery, and Generate fallback.
The CPU test build is `build-cpu-ci`; these tests use deterministic backends and
do not claim GPU coverage. Rollback is a revert of the accounting commit; no
model, config, grammar, or wire-field removal is involved.
