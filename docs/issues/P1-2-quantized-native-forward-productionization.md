# [P1-2] Quantized Native Forward Productionization

Priority: P1
Owner: Runtime (CUDA kernels)
Effort: 3 eng-weeks
Risk: High
Dependencies: P0-4
Labels: runtime, cuda, quantization, throughput

## Problem
Quantized forward path still uses sequential and non-fused fallbacks that limit throughput and efficiency.

## Scope Files
- `runtime/backends/cuda/native/quantized_forward.cpp`
- `runtime/backends/cuda/native/quantized_forward.h`
- `runtime/backends/cuda/native/quantized_gemm.cpp`
- `runtime/backends/cuda/native/quantized_gemm.h`
- `runtime/backends/cuda/native/model_forward_factory.cpp`
- `tests/unit/test_native_forward.cpp`
- `tests/unit/test_native_batching.cpp`
- `docs/MONITORING.md`

## Test Plan
1. Add unit tests validating true batched quantized forward path selection.
2. Add correctness tests across representative quantization formats.
3. Add stress test for KV attention with cached K and V on quantized models.
4. Run native forward and native batch suites.

## Acceptance Checklist
- [ ] Batched quantized forward is active for batch size > 1.
- [ ] Core hot-path kernels are fused where designed.
- [ ] KV cached attention is implemented and validated.
- [ ] Quantized model correctness tests pass.
- [ ] Throughput improves vs previous sequential fallback baseline.
