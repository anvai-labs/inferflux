# FFN Kernel Fusion Status: Original Layout Blocker Resolved

**Date**: March 10, 2026
**Task**: #12 - FFN kernel fusion
**Status**: Partial unblock complete; runtime rollout rejected by benchmark

## What changed

The original blocker was incorrect.

- `down_proj` is valid in `Q4_K`
- `Q4_K` packs along the reduction dimension
- for `down_proj`, the reduction dimension is `N_inter`
- `5632 / 256 = 22`, so the shape is legal

## What is now working

- `down_proj` indexing in the fused FFN bring-up kernel is fixed
- the kernel now uses an output-tiled design:
  - one CTA owns one batch row
  - one CTA owns an output tile
  - activated intermediate values are computed once per tile and reused across
    the output tile
- parity coverage exists in `test_native_forward.cpp`
- the focused fused-FFN CUDA test passes
- the full `[native_forward]` suite passes

## What is still blocked

Runtime rollout is still blocked.

Reason:
- the new tiled kernel is parity-correct, but the isolated benchmark shows it
  loses badly against the current FFN path on the real decode geometry
- measured results on RTX 4000 Ada:
  - `M=1, K=2048, N_inter=11008, N_hidden=2048`
  - baseline current path: `0.118 ms`
  - fused tiled path: `32.741 ms`
  - speedup: `0.004x`
  - `M=2`
  - baseline current path: `0.181 ms`
  - fused tiled path: `59.791 ms`
  - speedup: `0.003x`
- it is not yet wired into `transformer_forward.cu`

## Correct next step

1. Keep the current FFN runtime path unchanged
2. Redesign fusion around a different operator strategy
3. Benchmark any replacement against the current path before adding a rollout
   flag or touching `transformer_forward.cu`

## Incorrect conclusion retired

This statement is no longer valid:

- "FFN fusion is blocked because `down_proj` uses `N_inter=5632`, which is not
  compatible with `Q4_K`"

That was a misunderstanding of the weight layout.
