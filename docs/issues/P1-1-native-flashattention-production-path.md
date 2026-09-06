# [P1-1] Native FlashAttention Production Path Completion

Priority: P1
Owner: Runtime (CUDA kernels)
Effort: 2 eng-weeks
Risk: High
Dependencies: P0-3
Labels: runtime, cuda, kernels, performance

## Problem
Native flash-attention implementation still contains placeholder and TODO paths, limiting native backend maturity.

## Scope Files
- `runtime/backends/cuda/kernels/flash_attention.cpp`
- `runtime/backends/cuda/kernels/flash_attention.cu`
- `runtime/backends/cuda/kernels/flash_attention.h`
- `tests/unit/test_flash_attn.cpp`
- `tests/integration/native_metrics_test.py`
- `docs/MONITORING.md`

## Test Plan
1. Add kernel correctness tests against reference attention outputs.
2. Add architecture-conditional tests for kernel selection behavior.
3. Add integration metric checks for native kernel activation and fallback counters.
4. Run flash-attention and native metrics suites.

## Acceptance Checklist
- [ ] Placeholder TODO attention paths are removed or explicitly unsupported with deterministic behavior.
- [ ] Kernel selection logic is validated by tests.
- [ ] Native attention metrics show expected kernel usage.
- [ ] Performance regression threshold is documented and enforced.
