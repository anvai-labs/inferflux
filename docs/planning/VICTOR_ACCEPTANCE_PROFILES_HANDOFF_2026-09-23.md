# Victor / InferFlux acceptance profiles and deadline co-design

Status: discovered requirements and implementation handoff, **not implemented or accepted**.
Continue [C5 / #184](https://github.com/anvai-labs/inferflux/issues/184),
`docs/planning/CACHE_ACCEPTANCE_HANDOFF_2026-09-19.md` and
`docs/planning/DUAL_GPU_MODEL_PINNING_IMPLEMENTATION.md`.
The originating Mac session owns Victor's actual-member evidence and optional Sandhi integration.

## Verified baseline

- Re-fetched origin on 2026-09-23: main and develop are
  `02cf22addb9f78309bb5b2807427f215353a7085`. The mutable root checkout is an older,
  dirty branch; do not use or overwrite it as the accepted source.
- Last recorded serving binary SHA-256:
  `5c806f3b652e5f1e50f293fc00eefd3763b9ee9be252c915ecb1ed5f051353c4`.
  PID 59055 / loopback8080 is the recorded deployment, not a fresh identity claim.
  Recheck PID, executable, effective configuration and model readiness before testing.
- `ModelLoadSpec` already resolves global configuration and per-model placement,
  context, GPU-layer, sequence-capacity and KV-type overrides. Extend this owner;
  do not introduce a competing serving registry or dispatch path.
- `scripts/dual_gpu_acceptance.py` is a supplemental setup gate. It currently
  generates two models with 2048 total context, eight GPU layers and two sequences,
  uses fixed test ports and harness timeouts, and explicitly reports
  `actual_victor_cohort=false`, `c5_accepted=false`, cancellation pending and executed
  cache reuse pending. Its successful run35840245559 does not validate the full
  three-model serving geometry.
- `config/server.dual-gpu.yaml` expresses the intended three-model geometry:
  Qwen3-Coder-30B on AMD (65536 total / two sequences), Qwen2.5-Coder-14B on NVIDIA
  (32768 / two), and BGE-small on NVIDIA (16384 /32). GPU layers are requested as
  all; verify actual placement and memory. Context per sequence is32768,16384,512,
  respectively. The example itself is not deployment acceptance.
- HTTP socket send/receive timers exist. They must not be labeled scheduler queue,
  total generation, GPU-preemption or end-to-end deadlines. Those semantics need
  an explicit implementation/acceptance audit before configuration promises.

## One configuration-driven acceptance surface

Add a strict, versioned acceptance-profile contract to the existing harness and
its existing tests in `tests/integration/dual_gpu_acceptance_test.py`. Keep the
lightweight CI profile distinct from a full consolidated-service profile. Audit
duplicates before adding tests; reuse the existing pinning, usage-conservation,
placement and cleanup helpers from `frozen_gpu_cache_acceptance.py`.

The profile should reference the canonical serving specification and public model
IDs, expected source/binary/backend-module/asset pins, allowed endpoints and auth
references, workload cohort, bounds, and required evidence. Derive generated launch
settings and expected effective model geometry from the same parsed specification;
do not hand-maintain a second launch dictionary. Compare derived expectations with
live `/v1/models` capabilities. Reject unknown fields, duplicate model IDs,
unavailable required capabilities and configuration drift explicitly.

Private credentials and private filesystem paths stay local. Configuration carries
references, not credential values. Reports include a profile version/digest and a
sanitized resolved configuration, including the source of inherited settings.
Preserve no-profile behavior as an explicit compatibility path until migration is
reviewed; a missing or invalid requested profile must never silently fall back.

## Deadline ownership

| Layer | Responsibility |
|---|---|
| Victor | Member/task/client limits, approved model selection, distinct sessions |
| Sandhi | Authorized endpoint/model routing; buffered, stream-setup and stream-idle bounds; accounting |
| InferFlux | Model admission, queue/execution policy, socket I/O and cooperative cancellation |
| Acceptance runner | Startup, request, cohort and cleanup bounds; honest failure evidence |

Use global policy with per-model overrides where the server actually enforces the
policy. Endpoint/transport policy belongs to the listener or upstream entry, not GPU
selection. Exact models inherit within their selected endpoint; clients cannot raise
operator ceilings. Validate positive finite bounds and reject unsupported overrides.
Resolve once and report effective values; do not wrap an unchanged shorter inner
timer with a larger outer timer or rebuild connection pools per request.

Where a bounded origin queue/execution policy exists, leave response/settlement
headroom before the gateway and client expire. Record each timer's start, scope and
retry ownership instead of assuming scalar ordering establishes an end-to-end bound.
Stream idle is a gap bound, not total lifetime. Long streaming requires an explicit
total-lifetime or budget-lease renewal contract. A504 or client disconnect does not
prove a running GPU operation stopped. Do not automatically replay ambiguous POSTs.

Sandhi's existing acceptance baseline remains buffered120s/setup30s/idle90s.
Operator global/endpoint/model overrides are planned separately (Victor G53), not
shipped by the 0.9.1 repair. A changed-deadline cohort is new evidence and cannot
replace a prior failure. Align contracts; do not copy identical timeout numbers
into every layer.

## Ordered execution gates

1. TDD the profile parser/resolver and effective-config comparison using existing
   test owners; prove invalid/ambiguous profiles fail and defaults remain unchanged.
2. Resolve the current origin-liveness blocker before adding load. Victor's new
   Qwen3 simple cohort has four final gateway504 calls; cancellation-time accounting
   did not reconcile. Readiness and verified weight placement were insufficient.
   Capture request/model-correlated admission, dispatch, backend completion and
   queue ownership on the exact binary. A shared embedding/generation worker join
   is a hypothesis, not a proven cause. The Mac session lacked passwordless general
   sudo for bounded profiling; do not weaken ptrace policy or transfer credentials.
3. Validate the full three-model geometry, real memory, generation progress and BGE
   compatibility under mixed traffic. Keep setup probes labeled supplemental.
4. Run approved actual-member evidence directly and through optional Sandhi using
   the same ready model/profile and ordered payloads. Preserve request/session/run
   correlation, explicit cache reporting, tokenizer units and usage conservation.
   Reported zero cache is not proof of zero executed reuse.
5. Run the15 unique Victor formation/ensemble cases with appropriately simple
   tasks for each local chat model, then the unchanged six-Qwen/one-ZAI C5 verdict.
   Require complete member artifacts, passing pytest/oracles and distinct member
   sessions. Gateway-enabled cohorts require wire/SQLite/C4/dashboard reconciliation;
   direct-origin cohorts require applicable wire/origin usage and correlation checks
   and explicitly mark gateway accounting unexercised. A direct cohort cannot replace
   the required gateway C5 verdict. Keep partial failures.
6. Keep streaming, cancellation, enabled-session leases, executed reuse and GPU
   kernel overlap as separate acceptance claims until their own evidence passes.

OIDC remains the default Sandhi user/UI posture with roles and separate accounting
authority. Direct InferFlux currently uses the explicitly configured private scoped
API-key path; it must not be called verified OIDC interoperability. Test both declared
paths without silent auth downgrade, gateway bypass or credential transfer.

Follow AGENTS.md trusted-main GPU gates and serial CUDA/ROCm runner ownership.
Work in linked branches from fresh origin/develop, review before promotion, require
all actual CI gates, and preserve shared cache, dirty files, rollback configuration,
prior failed evidence and private state. Keep #184/C5 open until the full verdict is
reviewed. Do not repeat the preserved five-call streaming replay merely to recreate
this handoff.


## 2026-09-24 cancellation and embedding-memory findings

The existing embedding admission test missed cancellation during the only or final
native slice: its 33-input case cancelled the first slice and was caught on the next
iteration. New single/final-slice scenarios reproduced a false successful result.
The scheduler now checks cancellation at result publication, after retaining tokens
measured for completed work and before requeueing or returning vectors. Cancelled
results are aborted and contain no embedding vectors. This does not interrupt native
GPU execution or promise an atomic cancellation acknowledgement after publication.
It also does not make measured usage available on a disconnected HTTP transport.

The regression extends the existing fixture, rather than adding a parallel suite.
The failing cases were observed before the production fix; the updated admission
suite passes 93 assertions in four cases, and all 54 configured CPU CTest targets
pass. These are CPU results, not a GPU runtime or C5 acceptance claim.

Source inspection also found that `EnsureEmbedBatchCtx` independently hardcodes
32 sequences and 512 tokens per sequence for context, batch and microbatch. Changing
model placement or the configured generation context does not bound this allocation.
The recorded serving log includes a 13432 MiB CUDA embedding compute reservation;
AMD model/KV/compute allocations already total approximately 23 GiB on a 32 GiB card.
Moving BGE to AMD therefore needs measured capacity and bounded embedding geometry,
not an assumption that its small weight file implies a small execution footprint.

Follow-up: derive embedding execution geometry from one validated model-owned
configuration, test bounds and defaults on CPU, and measure the accepted trusted-main
runtime before changing placement. Preserve ordered batching, token accounting and
pooling semantics. The cross-device executor also waits for all device-group futures
before returning results to the scheduler; this is a possible source of shared
head-of-line blocking, not a demonstrated diagnosis of the stalled live process.
Neither the cancellation fix nor a same-binary restart closes origin liveness,
embedding compatibility, executed reuse, session leases or #184/C5.


### Bounded embedding geometry implementation follow-up

The opt-in `embedding_batch_size` model override now uses one validated geometry
for the dedicated llama.cpp embedding context and input grouping. Omission retains
32 sequences; values 1–32 select groups of up to 512 tokens per sequence. See
[the configuration contract](../CONFIG_REFERENCE.md#per-model-embedding-batch-geometry).
Native/MLX and the legacy name-only manager reject explicit geometry. Failed or
rejected reloads involving explicit geometry preserve the prior backend state;
model registry changes still require unload/new identity. Diagnostics describe
configured limits, not allocated capacity or completed GPU execution.

The existing model/parser/registry/HTTP/native test owners are extended. Invalid
parser cases and rejected-reload behavior failed before their fixes. A real BGE
GGUF regression on a CPU-only build checks ordered groups, per-input output parity
and preserved output after reconfiguration rejection. Candidate code has not run
on either GPU. The example BGE model opts into one-sequence groups but keeps its
existing NVIDIA placement. Measure accepted-runtime GPU allocation, BGE compatibility
and mixed traffic before considering AMD relocation or closing G52/G59 and C5.
