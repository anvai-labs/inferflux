# TD-007: Make Context Capacity and Completion Lifecycle Explicit

Status: In Progress
Priority: P0
Owners: Runtime, Scheduler, Server
Dependencies: TD-002

## Liability

`LlamaBackendConfig` defaults to a 2,048-token context and 128 parallel
sequences. llama.cpp interprets that context as shared capacity and exposes only
256 padded tokens per sequence, while the scheduler advertises 128 logical
slots without validating prompt plus decode demand against the effective
per-sequence limit. The prefill and decode pools also shared one condition
variable despite having disjoint work predicates; a notification could wake
the wrong pool and leave a request stalled before response headers. Backend
failures can currently surface as empty successful responses.

## Remediation

- Define whether configured context is aggregate or per sequence, expose the
  effective value, and derive backend sequence capacity from one owner.
- Reject prompt plus decode budgets that cannot fit before device execution.
- Propagate prefill/decode failures as structured API errors, never empty 200s.
- Bound non-emitting token loops and add repeated-prompt, prefix-reuse, socket
  completion, and sequence-retirement stress tests.
- Validate the corrected contract on CPU, CUDA `sm_89`, and ROCm `gfx1201`.

## Exit Criteria

- `/v1/models` and metrics report aggregate context, per-sequence context, and
  live/maximum sequences from the same runtime state.
- Oversized requests fail deterministically with an actionable 4xx response.
- A 100-request mixed-prompt gate completes without timeout, silent fallback,
  empty-success masking, slot leak, or unbounded non-emitting-token work.
- Sanitizer and accelerator runs prove one cleanup path for success, error,
  cancellation, prefix donation, and session retention.

## Current Evidence

CPU unit coverage now bounds repeated non-emitting unified-batch tokens and
exercises 32 sequential split-pool requests. Separate prefill/decode condition
variables close the lost-wakeup path, while the HTTP probe reports request,
header, and body failure phases. On `gfx1201`, the formerly flaky sequential
probe completed 5/5 requests and two independent concurrent probes each
completed 16/16 requests (128 output tokens) without fallback or timeout. The
ROCm CI lane now uses that 16-request behavioral gate. Context admission,
capacity exposure, structured backend errors, and the 100-request lifecycle
matrix remain open exit criteria.
