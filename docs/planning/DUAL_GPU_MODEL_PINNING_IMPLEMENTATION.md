# Same-process CUDA/ROCm model placement

Status: implementation and model-free validation; **hardware and Victor acceptance
pending**. C5 / [#184](https://github.com/anvai-labs/inferflux/issues/184) stays open.
This implements the September 21 Victor handoff against develop `b293d5719`.
The pinned llama.cpp revision remains `bb4caa7540188872173c44d161602d9271386413`.

## Load contract

Startup `models`, `INFERFLUX_MODELS`, watched registries and
`POST /v1/admin/models` use `ModelLoadSpec`. Existing ID/path/backend/format fields
remain. The following optional model fields override the resolved global runtime
configuration, including global environment settings, before backend tuning:

| Field | Contract |
|---|---|
| `device` | `cuda:N` or `rocm:N`; nonnegative vendor-local ordinal |
| `context_size` | Positive total context; divided across sequence capacity by llama.cpp |
| `gpu_layers` | Nonnegative layer count, or `-1` for all; explicit zero remains zero |
| `max_parallel_sequences` | 1–256; total context must cover every sequence |
| `kv_cache_type` | `f16`, `q8_0`, `q4_0`; explicit quantized KV requires flash attention |

Omitted fields retain legacy defaults. CUDA ordinal zero and ROCm ordinal zero
identify different devices. With a selector and an omitted/auto backend, the
router chooses the corresponding llama.cpp provider. Explicit selectors never
fall back across vendors or to CPU. Native CUDA rejects explicit selectors and
llama-specific offload/KV overrides rather than silently ignoring them.
`runtime.cuda.device_id` and `runtime.rocm.device_id` now supply vendor-specific
default selectors; per-model `device` takes precedence. A model-level bare
`device_id` is rejected. Existing native configurations containing the previously
ignored runtime selector must remove it or choose the llama.cpp provider.

Use [the two-model example](https://github.com/anvai-labs/inferflux/blob/develop/config/server.dual-gpu.yaml). Both clients send
requests to the same `/v1/chat/completions`, changing only `model`. Sandhi needs one
origin with both IDs in its virtual-key allowlist. No deployment manifest or
existing service is changed by this example. Container/Helm operators must deploy
the mixed binary **and its backend modules**, then mount a config using this schema;
single-vendor images cannot implement this recipe.

## Placement and failure semantics

The loader resolves a vendor registry and ordinal to a llama.cpp device handle,
passes a null-terminated one-device list, sets `LLAMA_SPLIT_MODE_NONE`, and uses
`main_gpu=0` within that restricted list. CUDA/ROCm strategy discovery uses the
same ordinal. ggml's vendor module binds its own execution/allocation threads to
that handle; InferFlux does not change process-wide visibility variables.

An adapter confined to `llama_device_placement.cpp` inspects the pinned llama
model's actual tensor buffers after loading. Any weight buffer on a different GPU,
or no GPU weight residency for an explicit GPU selection, rejects the load.
The public llama API has no weight-buffer inventory, so this adapter deliberately
depends on the pinned internal model layout. Review it on every submodule update.
Failed context creation frees loaded weights before returning.

Model-list, model-detail and admin responses expose `placement`:
requested/effective selector, vendor, device name, nullable PCI identity,
`state=verified_weights`, GPU/CPU weight-tensor bytes, observed GPU layer count,
requested offload, effective context/sequence capacity and KV dtype. Bytes are
logical weight-tensor bytes, **not total allocator/VRAM usage**. CPU-resident
weights remain explicit. `verified_weights` does not prove KV/activation placement,
kernel execution, kernel overlap, or executed cache reuse. Missing physical identity
is null and fails the strict setup gate; a device label is never substituted.

Mixed builds use upstream `GGML_BACKEND_DL` modules with local symbol scope:
the pinned CUDA and HIP libraries otherwise export the same registration symbol.
Modules are colocated with the build's executable, and upstream installs them in
`bin`. Portable CPU-module compilation disables `GGML_NATIVE` in mixed builds.
The CUDA and HIP startup-advisor probes are separate translation units because
their vendor headers define conflicting vector types.

Duplicate IDs/artifacts are rejected, including filesystem aliases. Replicas of
one artifact are not supported by this contract. Registry snapshots with duplicate
identities or changed specifications are rejected before unloading anything.
To update, remove the entry, wait for backend leases to drain, then add the new
specification under a new ID. A refused unload retains registry ownership and can
be retried. IDs retired in a server lifetime cannot be reused, preventing stale
session state from naming a replacement. A retained default/backend owner may
require a coordinated restart. Invalid startup overrides fail startup; admin
validation/conflicts return 400/422/409. Registry failures log their reason.

## Admission and concurrency boundary

The scheduler retains the conservative minimum sequence bound across loaded
models. Hot loads below the running slot bound are rejected; configure all model
capacities at startup or coordinate a restart to reduce it. There is no claim of
independent per-device admission quotas.

Within an admitted batch whose backends all have verified placement, different
devices execute concurrently using the existing batch executor. Requests on one
device stay together, and a device-group exception produces errors for that group
while other groups finish. Shared speculative execution retains its existing path.
Batch accumulation admits simultaneous requests together; bounded decode slices
allow later work to join. This does not promise overlap for every arrival pattern,
or parallelize the scheduler's initial prefill loop. Single-model behavior retains
its existing execution path. The optional `INFERFLUX_PLACEMENT_DIAGNOSTICS=1` emits
bounded-by-workload host-call intervals and request correlations; it is intended
for isolated acceptance, not enabled on shared production services.

## Validation and promotion

TDD first reproduced duplicate registry acceptance and lost ownership after refused
unload. A separate concurrency regression demonstrated serial device groups and an
escaping simulated allocation failure before the repair. Added coverage checks
parser parity, malformed selectors/resources, wrong vendors, unchanged defaults,
explicit-zero offload, admin propagation, live-update rejection, hot-load capacity,
unload leases, cross-model session isolation and honest placement reporting.
Original red logs remain locally under `/tmp/inferflux-dual-gpu-red-*.log`.

A clean mixed build enabled CUDA architecture 89 and HIP architecture gfx1201,
with CUDA and ROCm both ON. **Compilation is not GPU acceptance.** The CPU suite
and contract harnesses can run before promotion. Runtime work must follow
[the trusted-main process](../GPU_CI_BOOTSTRAP.md). No feature/PR code is dispatched
to the trusted runner. Independent review and hosted CI precede promotion.

The opt-in `same_process_dual_gpu` input in `gpu-gates.yml` adds one serial job
after the normal CUDA and ROCm jobs, inside the existing exclusive workflow group.
It clean-builds one server plus both modules. Configure two distinct small GGUFs
(each at most 2 GiB) via `INFERFLUX_DUAL_{CUDA,ROCM}_MODEL_{PATH,SHA256}` and the
existing pinned `INFERFLUX_GPU_CACHE_SANDHI_{BINARY,SHA256,SOURCE}` variables.
The harness checks exact source, submodule, model, binary and module identities;
requires at least 2 GiB free on each GPU; and uses private credentials/state on
loopback 28085 and 18794. It monitors preserved services on 8080/8081/8090 and
stops only owned processes. It never clears shared caches or edits existing keys.

The supplemental scenario checks both models' weight residency before requests,
concurrent plain/SSE calls directly and through one isolated Sandhi origin,
explicit cache reporting and wire → SQLite → C4 → dashboard conservation, including
per-model/session attribution. It records host execution overlap separately from
GPU kernel overlap. Sanitized JSON is retained on both success and failure; raw
logs, request/response bodies, private SQLite and credentials are not uploaded.
This does not provision or alter the retained Mac gateway.

**Still required:** exact promoted-main runtime evidence; GPU kernel timelines
showing simultaneous work; negative hardware-device/OOM tests; cancellation and
late accounting; executed reuse diagnostics; and fresh actual Victor member
cohorts (deliverables, pytest, numeric oracle, structured decisions, distinct member
sessions and per-model attribution). Retain the 120-second buffered gateway deadline.
The setup harness always reports `c5_accepted=false` and `actual_victor_cohort=false`.
Request correlations and usage records alone are not distributed OTEL traces.

## Migration and rollback

1. Keep current 8080 ROCm, accepted 8081 CUDA and 8090 embeddings processes,
   binaries/configs, caches, credentials and dirty worktrees unchanged.
2. Review/promote the implementation and obtain exact-main setup evidence using
   small isolated models without reclaiming another owner's VRAM.
3. Coordinate any production-model unload/restart with owners only after artifacts
   and placement/concurrency/tracing evidence are reviewable. Preserve original
   successful and failed Victor evidence. Provision the one-origin Sandhi route
   with local credentials and explicit model allowlists.
4. Roll back by removing only the new route/tunnel and stopping only the new owned
   server; restore the preserved route if an approved migration changed it.
   Do not clear caches or delete prior evidence. C5/#184 closes only after full
   mixed-team acceptance, not after this setup gate.
