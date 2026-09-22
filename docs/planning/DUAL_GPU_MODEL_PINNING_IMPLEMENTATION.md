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

Use [the three-model example](https://github.com/anvai-labs/inferflux/blob/develop/config/server.dual-gpu.yaml). Both clients send
requests to the same `/v1/chat/completions`, changing only `model`. Sandhi needs one
origin with both IDs in its virtual-key allowlist. No deployment manifest or
existing service is changed by this example. Container/Helm operators must deploy
the mixed binary **and its backend modules**, then mount a config using this schema;
single-vendor images cannot implement this recipe.

## Contract for consuming applications

Victor and other OpenAI-compatible clients can use InferFlux directly or through
Sandhi. Sandhi is optional, not a requirement for multi-model serving. In either
case clients select a public model ID and the appropriate API path; no request
contains a GPU selector or a vendor-specific port. Configure placement, offload,
context and sequence capacity only through InferFlux's operator load contract.
Placement diagnostics may be inspected operationally, but consumers must not use
them to decide request destinations. Public IDs should describe model identity
rather than device placement. The chosen public IDs below identify models only.

| Responsibility | Owner |
|---|---|
| Choose a model for a task; send chat/embedding requests | Victor or consuming application |
| Optional gateway policy, authentication, quotas, retries, caching, routing and accounting | Sandhi, according to its configured capabilities |
| Model loading, GPU placement, memory budgets, execution scheduling and backend token usage | InferFlux |

Direct clients set their base URL to InferFlux's `/v1`; gateway clients set it to
Sandhi's compatible API and use its locally managed credentials. Both paths must
preserve model selection, response model identity and usage accounting. Sandhi
maps public model IDs to an InferFlux origin (or an explicit configured alias),
not to CUDA/ROCm devices. InferFlux must provide correct model selection and usage
without gateway involvement; gateway-specific features remain optional additions.

The consolidated deployment uses **one InferFlux origin on port 8080**. The example
binds loopback; remote consumers use an authenticated gateway or an explicitly
configured reachable address. The isolated acceptance harness uses 28085 to avoid
replacing a running service. That test port is not part of the application contract.
The user selected Qwen3-Coder-30B on AMD, Qwen2.5-Coder-14B on NVIDIA and the
existing BGE embedding model on the same origin. Replacement of 8080/8081/8090 is
authorized for consolidation; the user subsequently authorized stopping all three
for the co-design test window to release VRAM. Preserve their launch configurations, credentials and disk state for
rollback. Restarting loses in-memory cache warmth.

| Application setting | Contract |
|---|---|
| OpenAI-compatible base URL | `http://127.0.0.1:8080/v1` on the server host |
| Discovery | Authenticated `GET /v1/models`; use the exact returned model ID |
| Generation | `POST /v1/chat/completions` with an explicit `model` |
| Chat model ID | `qwen3-coder-30b` |
| Chat model ID | `qwen2.5-coder-14b` |
| Existing embedding model ID | `bge-small-en-v1.5`; target route `/v1/embeddings` on the same origin |
| Authentication | Existing locally managed bearer credentials; do not copy keys into this document |
| Correlation | Unique `x-inferflux-client-request-id` for each call |
| Sessions | `x-inferflux-session-id`; use distinct IDs for Victor members and model histories |

The launch specification requests all layers on each assigned GPU. Qwen3 uses
65536 total context tokens across two sequences (32768 each); Qwen2.5 uses 32768
across two sequences (16384 each). The artifact metadata implies approximately
6 GiB of f16 KV storage per chat model at these capacities, in addition to weights,
compute buffers and embedding allocations. This estimate is not a VRAM measurement;
actual startup and mixed-load memory validation are required before cutover.

The model IDs above match the example configuration. Exact artifact hashes and
resource settings must be recorded before migration;
clients must not infer model identity from a port or a GPU vendor. Strict routing
rejects an unavailable requested model rather than silently choosing the default.
Both plain and streaming calls use the same URL and model IDs. For streaming, send
`"stream": true, "stream_options": {"include_usage": true}` and consume the final
usage chunk before `[DONE]`.

`usage.prompt_tokens` includes the cached portion. Read
`usage.prompt_tokens_details.cached_tokens` explicitly, including zero on a miss;
fresh input is `prompt_tokens - cached_tokens`. `completion_tokens` counts output,
and `total_tokens = prompt_tokens + completion_tokens`. Attribute usage to the
resolved response `model` and request/session identity, not the listener or device.
An interrupted stream without final usage is incomplete accounting, not zero usage.
Cache token reporting alone does not establish executed backend reuse.

Embedding requests use `{"model":"bge-small-en-v1.5","input":["text"]}`.
Their response reports `usage.prompt_tokens` and `usage.total_tokens` (equal input
counts); there are no generated completion tokens or SSE generation chunks.
Do not apply the chat usage schema to embedding responses. Encoder-only models
report `capabilities.generation=false`; chat/completion requests naming them fail
capability admission before reaching backend generation.

Embedding work now enters the existing scheduler and yields between slices of at
most 32 inputs (reduced for the scheduler token budget). Arrays contain at most
4096 strings. Queued chat work can run between slices; model ownership remains
leased across them, and embedding work never borrows generation session/KV state.
Unified scheduler mode (`decode_pool_size: 0`) is required; split decode mode
returns an explicit service error for embeddings. Vendor-separated admitted batches
retain the independent-device execution path. This is bounded cooperative service,
not kernel preemption or a promise of independent per-device admission quotas.

Batching and model-selected pooling from the existing embedding deployment's
commit `f0789a242` are preserved. Its dedicated embedding context remains 32 × 512
tokens; the BGE example explicitly records that geometry. Inputs longer than 512
backend tokens retain that deployment's truncation behavior, and usage now counts
the evaluated tokens after truncation. The existing generation context is separate;
placement metadata does not measure the lazily allocated embedding context's VRAM.
Allocation failure in an embedding slice returns an error without failing chat.
Cancellation is checked between slices; an active GPU call is not interrupted.
POSIX socket errors cancel remaining work, as do Linux full-hangup indications.
Darwin reports a hangup even for valid read-side half-closes, so that indication
alone cannot cancel work there. A clean peer close may remain undetected until
the response write; Windows disconnect detection and late accounting on failed
or disconnected calls still need acceptance coverage. Request-ID and W3C trace
headers work directly; neither requires Sandhi.

Model-free regression coverage checks interleaved chat, successful array ordering,
zero generation/cache usage for embeddings, cancellation, simulated allocation
failure, a generic embedding backend, evaluated-token counts and HTTP half-closes.
Actual BGE vector compatibility, shared-GPU memory and latency under mixed traffic,
and direct/gateway Victor cohorts remain hardware acceptance requirements.

Sandhi retains its existing client-facing gateway address and needs only one
InferFlux origin/tunnel with the required model IDs allowed. Victor members select their
assigned ID through Sandhi; they do not select an origin port per GPU. Retain the
120-second buffered gateway deadline. Preserve wire, SQLite, C4 and dashboard
correlations during migration; request/session accounting is not an OTEL trace.

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
Linux mixed modules export only the upstream loader entry points. This also
hides shared GNU-unique template statics, which can escape `RTLD_LOCAL` isolation.
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
loopback 28085 and 18794. It records the initial listening state of 8080/8081/8090
and requires it to remain unchanged, including intentionally stopped origins.
Any initially running origin must remain healthy. It stops only owned processes
and never clears shared caches or edits existing keys.

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
Run the multi-model Victor acceptance both directly against InferFlux and through
Sandhi; direct acceptance must not depend on gateway-only headers or state.

### Review corrections before promotion

The September 22 review reproduced a macOS HTTP half-close cancellation error,
unrequested default pinning, and acceptance-harness false positives for wrong
response models and direct request IDs. Regressions now retain the returned model
identity, check every SSE frame, and compare direct correlation exactly. Omitted
selectors preserve upstream default placement; only explicit selectors constrain
devices. The original HTTP half-close regression is reused rather than duplicated.
The macOS CPU suite also exposed a 30-second eviction-worker shutdown delay:
changing its stop predicate under the waiter's mutex prevents lost notifications.
Finally, a monitor shutdown timeout no longer skips owned-server cleanup; cleanup
failures are reported separately without replacing the original execution failure.
These are model-free checks and do not constitute hardware acceptance.

The broader Mac stub integration run exposed a separate isolation defect: its
fixed port 18081 overlapped the preserved CUDA SSH tunnel, and startup readiness
accepted the existing listener. Those results are invalid as stub or acceptance
evidence. The test issued completion, embedding, tokenization, cache-warm and
admin requests to the retained runtime. Routing settings were checked afterward
against its preserved configuration and matched; the scope is `any_compatible`.
The cache-warm request used tokens `[1,2,3]` and block table `[100]`; there is no
scoped removal API, so no shared cache was cleared to undo it. Treat that runtime's
subsequent in-memory cache state as potentially contaminated, retaining earlier
evidence separately. No service was restarted and no credentials were changed.
The process helper now rejects an occupied configured port before launching or
sending application data. A bounded TCP handshake catches Darwin wildcard
listeners that a reuse-enabled bind alone misses; it sends no HTTP or credentials.
Only connection refusal permits the bind preflight. `INFERFLUX_TEST_PORT_BASE` permits isolated stub ports;
the Mac rerun uses 28091-28094. This bind preflight does not reserve the port
through child startup, so exclusive test-port ownership remains required.

## Migration and rollback

1. The user authorized stopping 8080 ROCm, accepted 8081 CUDA and 8090 embeddings
   before replacement testing. All three stopped cleanly on September 22.
   Their private launch/environment/config snapshots and binary copies are in
   `/home/vsingh/.local/state/inferflux-consolidation-20260922` on aiserver1.
   Persisted caches and dirty worktrees remain untouched. Old in-memory cache
   warmth is lost on shutdown; earlier evidence remains preserved.
2. Review/promote the implementation and obtain exact-main setup evidence using
   small isolated models in the authorized test window.
3. The user authorized consolidating all three services on 8080 with the model IDs
   above. Cut over only after artifacts and placement/concurrency/tracing evidence
   are reviewable under the trusted-main process. Preserve original
   successful and failed Victor evidence. Provision the one-origin Sandhi route
   with local credentials and explicit model allowlists.
4. Roll back by removing only the new route/tunnel and stopping only the new owned
   server; restore the preserved route if an approved migration changed it.
   Do not clear caches or delete prior evidence. C5/#184 closes only after full
   mixed-team acceptance, not after this setup gate.
