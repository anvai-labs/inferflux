# Cache replay and implementation evidence

Current status: [September 19 acceptance addendum](CACHE_ACCEPTANCE_HANDOFF_2026-09-19.md#september-19-cache-acceptance-addendum--current-acceptance). The historical evidence below retains its original provenance and limitations. The deployed frozen replay passed; Sandhi #274 and the unchanged five-call oracle resolved the scoped terminal-stream accounting finding. C5/#184 remains open for the originating Mac six-Qwen/one-ZAI verdict and explicit broader limitations.

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

The reusable local replay is:

```bash
python scripts/cache_member_replay.py \
  --victor-repo /tmp/inferflux-victor-cache-replay \
  --sandhi-binary /tmp/inferflux-cache-sandhi-target/debug/sandhi-proxy
```

Use a Python environment with Victor's dependencies and privately set
`INFERCTL_API_KEY`. The script bounds capture at 32 requests, uses only the Qwen
writer portion of WS-E, launches loopback gateway/observer ports 18789/18084,
stops its gateway on exit, and checks each SQLite row against wire usage. No
Mac credentials or ZAI calls are needed. The validated reusable-script run,
`inferflux-member-973c745aa4`, completed two calls (1720/155/0 and 1929/246/0),
passed the generated pytest test, and automatically reconciled both ledger rows.

For a separately scheduled deployment of the diagnostic build, set
`INFERFLUX_CACHE_DIAGNOSTIC_REQUEST_PREFIX` to the correlation prefix of one local
member's requests. Capture emits at most 64 JSON events per process, requires a
nonempty matching `x-inferflux-client-request-id`, and hashes token sequences and
session IDs. It records policy bypass, lookup common-prefix length, copy failure,
partial-prefill failure, and accepted synchronous prefill reuse with sequence
generation. Lookup length is a candidate, never proof of successful reuse.
The token hash uses comma-separated decimal token IDs before SHA-256.
The completed diagnostic implementation also records capacity-eviction counts,
admission failure, effective session configuration/lease outcome, actual execution
path, and tokenized common-prefix length against the previous captured request
in the same model/session. First observations report a null previous-prefix
length. The bounded comparison retains token IDs in memory but never logs them.

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

## Block ownership follow-up

A deterministic regression reproduced a radix reference leak: two 40-token
donations diverging at token 20 reserved eight blocks, but the trie retained only
seven in its ownership tables. The new leaf stored suffix blocks while the
scheduler had acquired references to its complete table. Eviction therefore
could not release all donated references. Nodes now retain complete donor tables;
lookup selects the chosen donor's table without concatenating ancestor tables.
Hits cannot extend past the selected donor's token extent.

Separate deterministic scheduler cases reproduced four leaked blocks when a
zero-capacity cache declined donation. Cleanup now releases rejected donation
references and unowned reservations when slot admission fails. Full-prefill
failure also drops scheduler references to warm blocks. Tests require all 32
blocks to become free again after donation rejection, occupied-slot fallback,
and page-pressure fallback. Page-pressure fallback calls full Generate and
reports zero reused tokens, consistent with the current replay's resource
snapshot. This is a verified defect and mechanism, not retrospective proof of
what caused the original 40 calls.

Rollback: revert the ownership commit together with its donor-table lookup
change. Do not mix old suffix-only insertion with complete-table lookup. The
running Qwen process still uses the old binary; repairing its retained state
requires a separately scheduled deployment/rollback, not a live cache flush.

## Local adversarial review

The author review checked donor ownership, slot admission, failed prefill,
response/metric agreement, per-model capacity, and capture privacy. It is not
an independent approval. Two additional deterministic regressions failed before
their fixes:

- With one sequence slot, admission evicted the selected donor but its successful
  empty-slot copy still reported 39 cached tokens. Admission now discards prefix
  candidates after slot eviction and reserves a fresh table. Synchronous and
  deferred repetitions must report zero, avoid copying the retired donor, and
  return every block after eviction.
- A failed deferred prefill still created a live donor. Radix donation and session
  retention now require an explicit successful-prefill flag, set only after all
  prompt chunks complete. Intermediate/final prefill failures must leave no
  retained blocks, and the next same-session request must start cold.

These are review findings in the implementation paths; neither establishes the
historical cause of the original WS-E calls. The deployed GPU binary is unchanged.
