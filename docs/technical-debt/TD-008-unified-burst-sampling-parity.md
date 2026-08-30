# TD-008: Restore Unified CUDA Burst Sampling Parity

Status: Proposed
Priority: P1
Owners: CUDA Runtime, Scheduler
Dependencies: TD-002, TD-007

## Liability

The graph-replayed unified CUDA burst samples with raw batched argmax. It does
not apply the normal per-sequence repetition, frequency, or presence penalties
and does not update sampling history between graph replays. Native greedy also
applies an implicit repetition penalty once history exists, so no current
multi-token eligibility subset preserves stepwise token semantics. The path is
therefore hard-disabled even when `INFERFLUX_CUDA_DECODE_BURST=1` is requested.
Its `max_wall_ms` deadline is also reserved but unenforced.

## Remediation

- Keep penalty history and counts on device per sequence.
- Apply penalties and update history between every replayed sample and forward.
- Enforce per-sequence token/KV budgets and the burst wall-clock deadline.
- Preserve EOS, stop-string, cancellation, and non-emitting-token semantics.
- Re-enable capability reporting only after parity and latency gates pass.

## Exit Criteria

- Multi-token burst matches stepwise greedy output with default and explicit
  penalties across homogeneous and heterogeneous batches.
- Deadline, decode-limit, KV-headroom, EOS, and cancellation tests pass.
- CUDA graph and non-graph traces report identical sampling state transitions.
- Model-backed tests show no output or residual divergence, and benchmarks
  demonstrate a material throughput gain before the feature is enabled.
