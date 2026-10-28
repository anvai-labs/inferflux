# ADR-0008: Two-tier KV prefix store (serialized tier under the radix cache)

Status: Proposed
Date: 2026-10-27
Owners: Runtime (InferFlux), Scheduler
Dependencies: PR #276 (retired-id eviction reporting), ADR-0006 (three-way
origin contract — InferFlux remains the authoritative reuse producer).
Implementation is data-gated: build only if the post-#276 production
hit-rate window still disappoints (accepted-reuse ratio < 50% of lookup
hits on repeat-heavy traffic, or long-context TTFT regression).

## Context

GPU prefix reuse is bounded by sequence slots. `max_sequences` defaults to
12 radix donors (`runtime/prefix_cache/radix_prefix_cache.h:38-41`;
`server/main.cpp:1620-1636` sets only `capacity=256`), and on a 32 GB GPU
with 4 × 24K-token slots, warm donors are evicted as soon as admission
pressure needs their slots (`Scheduler::AllocSeqSlot` eviction loop,
`scheduler/scheduler.cpp:3122`). Once
`RadixPrefixCache::EvictOneSequenceLocked` runs, the donor's KV is
destroyed — `FreeSequence` fires before the eviction callback
(`radix_prefix_cache.cpp:341`) — and a later request sharing that prefix
re-prefills from scratch.

The serialization primitives already exist, abstracted at the
`BackendInterface` level (`runtime/backends/common/backend_interface.h:211-220`):
`SerializeSequence(int seq_id) -> std::vector<uint8_t>` and
`HydrateSequence(int dest_seq_id, const std::vector<uint8_t>&)`. Llama
implementation at `runtime/backends/llama/llama_cpp_backend.cpp:1728-1757`
(`llama_state_seq_get_size/get_data/set_data`); native CUDA at
`runtime/backends/native/native_gpu_backend.cpp:376-402`. One consumer
already proves the end-to-end pattern — the disaggregated KV transport
hydrates serialized blobs into freshly allocated slots
(`scheduler/scheduler.cpp:1225-1247`) — but nothing consumes eviction.

Blob semantics (verified in the vendored llama.cpp):
`llama_state_seq_set_data` restores stored KV cells **with their original
positions** into the destination sequence id
(`external/llama.cpp/src/llama-context.cpp:3001-3033`; API contract
`external/llama.cpp/include/llama.h:860-878`). Consequences: (a) hydration
into a fresh, empty slot generation is position-consistent by construction;
(b) the blob does not self-describe its token count — the tier must record
`n_tokens` in its own header; (c) the blob's internal magic/version check
makes incompatible hydration fail closed (`set_data` returns 0 →
`HydrateSequence` false), a natural full-prefill fallback.

Economics: qwen3-coder-30b KV is ~192 KiB/token f16 / ~102 KiB q8_0
(4K-token donor ≈ 0.4–0.8 GB; 24K ≈ 2.4–4.6 GB). GPU re-prefill of a
500-token prefix is ~50–100 ms; hydrating 1 GB from NVMe is ~0.3 s, much
faster from warm page cache. Hydration beats re-prefill only at roughly
≥10K-token prefixes or on a saturated GPU — the design enforces this with
guards, not operator vigilance. Cross-model reuse is out of scope (KV
tensors are architecture-specific); same-model unload/load via
`DELETE`/`POST /v1/admin/models` (`server/http/http_server.cpp:2500-2623`)
is in scope.

## Decision

Add a **serialized KV tier**: a byte-budgeted, LRU-evicted store of
per-donor blob files at a configured path (tmpfs ⇒ RAM tier, NVMe
directory ⇒ disk tier — one mechanism), with an in-process side-table
index keyed by a model-identity-bound token-prefix key. A pre-evict
observer captures `SerializeSequence` output when the radix cache evicts a
warm donor; an async writer persists it. On a radix miss, the scheduler
probes the tier; a compatible entry covering enough tokens to beat
re-prefill is hydrated directly into a freshly admitted slot (no
`CopySequencePrefix`). Default OFF; every hook is a no-op when disabled;
all failures fall back to today's full-prefill behavior.

The radix trie's `Lookup` semantics — the hottest admission path, protected
by frozen GPU evidence — are **not modified**. The tier is a strictly
colder second chance consulted only on trie miss.

## Design

### Tier store (`runtime/prefix_cache/kv_tier_store.{h,cpp}`)

- File-per-donor, named `<tier_key_hex>.kvblob`, where
  `tier_key = sha256(identity_blob || token_prefix_bytes)` (decimal token
  encoding, as in `LogCacheDecision`'s token hashing,
  `scheduler/scheduler.cpp:97-101`). Two-level directory sharding (256
  subdirs) keeps startup scans and budget eviction cheap.
- File format: 8-byte magic `IFKVT1\0\0`, u32 header length, compact JSON
  header `{version, tier_key, identity{...}, n_tokens, blob_bytes,
  blob_sha256, created_unix_ms}`, raw blob payload. `n_tokens` is mandatory
  (the blob does not carry it). Atomic write via `.tmp` + `rename()`;
  crash leftovers deleted by the startup scan. Full-payload sha256
  verification is opt-in (`tier_kv_verify_checksum`, default false —
  software hashing ~0.5–1 s/GB would double hydrate latency for a
  corruption class the blob's own magic check mostly catches).
- Budget: in-memory index tracks per-entry bytes; LRU eviction by minimum
  `last_used` (relaxed atomics, mirroring `RadixNode::last_used` at
  `radix_prefix_cache.cpp:130-135`). Corrupt/mismatched/oversized files
  are deleted at startup; the directory is always safe to `rm -rf`.

### Index: side-table, not a trie extension

`KvTierIndex` = `unordered_map<tier_key, TierEntry>` with
`TierEntry = {token_prefix, n_tokens, identity, path, bytes, last_used}`.
Longest-match lookup is a bounded linear scan (≤ `tier_kv_max_entries`
default 256; microsecond-scale) with early exit. Rationale: extending
`RadixPrefixCache` nodes would touch the shared-lock hottest path and
could lie about `matched_tokens`; the entry count is far too small to
need a trie. Revisit trigger: if `max_entries` must exceed ~1K, move the
tier reference into the radix instead. Strict precedence: radix GPU hit
always wins; the tier is consulted only when `prefix_hit == false` and
not `reused_session_state`. Tier entries carry no backend weak_ptr, block
table, or slot dependency, so the `capacity_eviction` candidate-discard
logic does not apply.

### Export path

- `RadixPrefixCache::SetPreEvictObserver(std::function<bool(int seq_id,
  std::shared_ptr<BackendInterface>, std::vector<int> tokens)>)`, invoked
  in `EvictOneSequenceLocked` and in `EvictOne` (leaf-with-sequence case)
  **before** `FreeSequence`, with the victim's full token prefix
  reconstructed by walking parent edges (O(depth), export-only). Returning
  false declines (guards below). Covers both eviction triggers: admission
  pressure and Insert-time overflow.
- Serialization is synchronous in the observer — deferred-free handoff is
  rejected because the eviction exists to free that slot *now*. The
  dominant cost is a GB-scale copy under `BackendStateLock`
  (~40–460 ms), bounded by: `min_export_tokens` (default 2048 — below
  that, export cost ≈ re-prefill benefit), a staging-buffer pool that
  skips on exhaustion (never blocks), a per-export byte cap, and
  `export_under_pressure` (default false) as the admission-latency kill
  switch for this hook specifically.
- Disk I/O is async on a writer thread. `io/AsyncFileWriter::Enqueue`
  **blocks** on a condition variable when full
  (`io/async_file_writer.cpp`) — unacceptable on the scheduler path — so
  add a drop-on-full `TryEnqueue` and run a private writer instance for
  the tier (own queue, isolated from `PagedKVCache`'s writer at
  `runtime/kv_cache/paged_kv_cache.cpp:153`). Header assembly and optional
  checksum run on the writer thread.
- Amplification controls: dedupe by `tier_key` (skip existing; replace
  only if longer), in-flight bytes counted against the budget, single io
  thread by default. Teardown donation overflow needs no separate hook:
  `Insert` never refuses (it evicts internally), so every donor death
  under pressure already flows through `EvictOneSequenceLocked`.

### Hydration path

Probe after the radix lookup block (`scheduler/scheduler.cpp:2190-2197`),
hydrate after slot admission, mirroring the KV-transport pattern
(`scheduler/scheduler.cpp:1229-1246`):

1. `FindLongestPrefix(identity, bpe_prompt_tokens)` → accept only if
   `n_tokens ≥ min_hydrate_tokens` (default 8192, the crossover) and
   `n_tokens ≤ min(prompt_len, max_hydrate_tokens)`. Emit
   `LogCacheDecision(..., "tier_lookup", n_tokens)`.
2. Reserve blocks with `warm_blocks = 0` (the tier contributes no paged
   blocks; native-path accounting is Phase 3).
3. `blob = Read(tier_key)`; `HydrateSequence(seq_id, blob)` — no
   `CopySequencePrefix`. On success: `prefill_start = n_tokens` (minus one
   for exact-hit replay, as at `scheduler/scheduler.cpp:2396-2400`),
   `prefill_offset = n_past = prefill_start`,
   `cache_reuse_pending_tokens = prefill_start`, then
   `SyncSequenceSlotProgress`; the suffix flows through the existing
   deferred unified prefill unchanged. Emit `"tier_hydrate"`.
4. On any failure (read error, header/identity mismatch,
   `HydrateSequence` false — including llama's internal magic/version
   rejection): drop the entry, `FreeSequence(seq_id)` (a mid-throw
   `set_data` can leave partial cells), `prefill_start = 0`, fall through
   to the existing full-prefill ladder (same shape as `copy_failed`,
   `scheduler/scheduler.cpp:2386-2390`). Emit `"tier_hydrate_failed"`.

Slot/generation safety: the destination is the freshly acquired
`AllocSeqSlot` lease — a new, empty generation (retirement fences complete
before reuse, `scheduler/scheduler.cpp:3105-3143`) — and alloc→hydrate has
no yield between them, the same invariant the KV-transport path relies on.
Hydrated tokens count as accepted reuse (`cached_tokens`) exactly like
radix reuse, preserving ADR-0006 attribution.

### Model-identity binding

`identity_blob` = canonical serialization of: backend family + `Name()`
(blob formats are backend-family-specific), GGUF KV geometry from
`ModelInfo::gguf` (`scheduler/model_router.h:41-59`), `kv_cache_type`
(`runtime/backends/common/backend_config.h:84`), `n_ctx`/`n_seq_max`
geometry, state-format tag + vendored llama.cpp state-io version, model
id + path, tokenizer identity when overridden. Rejected: full GGUF sha256
(30–60 s added load latency for redundancy with the geometry tuple and
the blob's internal version check). The identity is also stored per-file
and re-verified at hydrate time. Tier files persist across a same-model
`DELETE`/`POST /v1/admin/models` cycle; after reload, tier hits resume
(Phase 3 adds the e2e test and docs).

### Configuration (defaults OFF; env > CLI > YAML > defaults)

Plumbed like `prefix_cache_capacity` (`server/main.cpp:341-343`,
`:821-826`) onto `Scheduler::Config` (`scheduler/scheduler.h:64`);
documented in `docs/CONFIG_REFERENCE.md`:

| Knob | Default | Env |
|---|---|---|
| `tier_kv_enabled` | false | `INFERFLUX_TIER_KV_ENABLED` |
| `tier_kv_path` | "" (empty = disabled) | `INFERFLUX_TIER_KV_PATH` |
| `tier_kv_budget_bytes` | 8 GiB | `INFERFLUX_TIER_KV_BUDGET_BYTES` |
| `tier_kv_min_export_tokens` | 2048 | `INFERFLUX_TIER_KV_MIN_EXPORT_TOKENS` |
| `tier_kv_min_hydrate_tokens` | 8192 | `INFERFLUX_TIER_KV_MIN_HYDRATE_TOKENS` |
| `tier_kv_max_hydrate_tokens` | 0 (= min(slot ctx, 32768)) | `INFERFLUX_TIER_KV_MAX_HYDRATE_TOKENS` |
| `tier_kv_io_threads` | 1 | `INFERFLUX_TIER_KV_IO_THREADS` |
| `tier_kv_queue_depth` | 4 | `INFERFLUX_TIER_KV_QUEUE_DEPTH` |
| `tier_kv_max_entries` | 256 | `INFERFLUX_TIER_KV_MAX_ENTRIES` |
| `tier_kv_verify_checksum` | false | `INFERFLUX_TIER_KV_VERIFY_CHECKSUM` |
| `tier_kv_export_under_pressure` | false | `INFERFLUX_TIER_KV_EXPORT_UNDER_PRESSURE` |

### Observability

Counters following the `RecordPrefixLookup` pattern
(`server/metrics/metrics.cpp:290-296`):
`inferflux_tier_kv_{lookups,hits,hydrate,hydrate_failed,export,export_bytes,
export_skipped{reason},hydrate_tokens}_total`; hydrate-duration histogram
(existing `LatencyHistogram`); gauges `size_bytes`, `entries`,
`in_flight_exports`, `budget_bytes`. Request-scoped `LogCacheDecision`
stages `tier_lookup` / `tier_hydrate` / `tier_hydrate_failed`; export
events via `LogSequenceSlotEvent("tier_export_begin"/"tier_export_end")`.
The frozen acceptance projector whitelist
(`scripts/frozen_gpu_cache_acceptance.py:265-286`) must gain the three
stages **in the same implementation PR** (reconciliation only consumes
`completed`/`prefill_accepted` — verified unaffected).

### Test strategy (Catch2)

- New `tests/unit/test_kv_tier_store.cpp`: header round-trip, atomic-write
  visibility, corrupt/truncated/foreign-identity blobs dropped, LRU
  byte-budget eviction, dedupe/replace-only-if-longer, startup scan,
  `TryEnqueue` drop-on-full never blocks.
- Extend `tests/unit/test_radix_prefix_cache.cpp`: observer fires before
  `FreeSequence` (ordering via instrumented backend), prefix
  reconstruction across split edges, declining changes nothing.
- Extend `PositionCheckingBackend` (`tests/unit/test_scheduler.cpp:2540-2637`):
  fake `SerializeSequence` (synthetic blob encoding `n_tokens`) and
  `HydrateSequence` (sets `pos_max[dest] = n_tokens - 1`, mirroring llama
  semantics) with injectable partial-failure. Assert: zero
  `position_violations` through `ExecuteUnifiedBatch`; hydrate failure
  clears the slot and completes via full prefill; sub-crossover prefixes
  never hydrate; a radix GPU hit always preempts a longer tier entry;
  exact-hit hydration replays one token.
- `tests/unit/test_metrics.cpp`: new series render.
- Phase-2 integration gate: tmpfs tier, trie eviction forced between two
  same-long-prefix requests; `tier_kv_hits_total ≥ 1` and
  `usage.prompt_tokens_details.cached_tokens` reflects tier reuse.

## Phased rollout

- **Phase 1 — export + index + observability** (hydration unreachable):
  store, index, writer, observer hook behind `export_under_pressure`,
  metrics, config, whitelist, tests. Gate: green tests; zero behavior
  change when disabled (TTFT benchmark unchanged); adversarial review;
  export-stall histogram reviewed on a saturated-GPU rehearsal.
- **Phase 2 — hydration** (default OFF, opt-in): admission probe, hydrate
  + fallback, crossover guards. Gate: position-consistency and
  corrupt-blob tests; TTFT A/B on a ≥10K-token shared-prefix workload
  showing hydration wins on a saturated GPU; tier hit rate > 0.
- **Phase 3 — unload/load preservation + polish**: admin unload/load e2e;
  optional native-path block accounting; optional
  `LLAMA_STATE_SEQ_FLAGS_ONDEVICE` exploration to cut the D2H export
  stall.

Risks: the Phase-1 admission stall from synchronous serialize under
pressure (default-off pressure hook + histogram before Phase 2 leans on
it); disk exhaustion on small hosts (budget gauge + alerts); stale blobs
after upgrades (identity tuple + blob magic both must pass);
frozen-contract drift (whitelist change lands with the first stage
addition, never after).

## Consequences

- Long shared prefixes survive slot pressure and same-model unload/load;
  recovery cost shifts from GPU recompute (~1–2 s per 10K tokens) to blob
  IO (~0.3–0.9 s/GB), a win only above the crossover.
- The radix trie and GPU reuse path remain untouched; all tier behavior
  is additive and default-off.
- New operational surface: a tier directory to provision and monitor;
  per-entry files contain KV tensors derived from user prompts — the
  directory inherits the server process's access controls and must be
  treated as prompt-derived data.
- `AsyncFileWriter` gains `TryEnqueue`; `RadixPrefixCache` gains a
  pre-evict observer — both small, separately testable.

## Rejected alternatives

- **Extending `RadixPrefixCache` nodes with tier references**: touches the
  shared-lock hottest path and risks lying about `matched_tokens`; entry
  counts are far below trie thresholds.
- **Pure-RAM in-process tier (no files)**: loses unload/load persistence
  and restart warmth; tmpfs gives RAM-tier behavior through one mechanism
  with page-cache reuse for free.
- **One big file / page-cache reliance without per-entry layout**: no
  per-entry LRU or atomic replacement; one corrupt region poisons all.
- **mmap of GPU memory / GPUDirect**: no such primitive in
  `BackendInterface`; llama.cpp owns context memory.
- **Deferred-free export (serialize on the writer thread, free after)**:
  defeats the eviction's purpose — admission needs that slot now.
- **Full GGUF sha256 in the identity key**: 30–60 s of load latency for
  redundancy.
- **Reusing `PagedKVCache` page offload** (`runtime/kv_cache/paged_kv_cache.cpp:120-220`):
  persists the scheduler's logical float pages, not backend-resident KV
  tensors; `SerializeSequence` is the honest full-state path.
