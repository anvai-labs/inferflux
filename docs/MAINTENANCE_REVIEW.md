# Maintenance Review

> Rear-view-mirror pass after the InferFlux CUDA, scheduler, distributed-runtime, and benchmark-control work. Focus: reduce accidental complexity without removing required serving capabilities.

## 1) Keep vs Simplify

| Area | Keep | Why |
|---|---|---|
| Dual CUDA backends (`inferflux_cuda`, `llama_cpp_cuda`) | Keep | They serve different purposes: native owns long-term runtime architecture; llama.cpp is the compatibility/perf baseline. |
| Sequence slots + generation leases | Keep | They are required for KV ownership, prefix reuse, and bounded GPU memory. The problem was stale reuse semantics, not the concept. |
| Optional `session_id` layer | Keep optional | Stateless API by default is correct; sticky reuse should remain opt-in and isolated from baseline OpenAI-compatible flows. |

## 2) Highest-Value Simplifications

| Priority | Problem | Evidence | Simplification target |
|---|---|---|---|
| P0 | Native execution policy is split across startup config, runtime env, benchmark env, and deep kernel files. | [server/main.cpp](../server/main.cpp), [transformer_forward.cu](../runtime/backends/cuda/native/transformer_forward.cu), [fused_quant_gemm.cu](../runtime/backends/cuda/native/fused_quant_gemm.cu), [run_gguf_comparison_benchmark.sh](../scripts/run_gguf_comparison_benchmark.sh) | Introduce `NativeExecutionPolicy` loaded once at backend startup and passed down explicitly. Remove deep `getenv()` reads from forward/dispatch code. |
| P0 | Native linear dispatch policy and kernel launching are too interleaved. | [fused_quant_gemm.cu](../runtime/backends/cuda/native/fused_quant_gemm.cu), [transformer_forward.cu](../runtime/backends/cuda/native/transformer_forward.cu) | Split into `native_dispatch_policy` and `native_linear_executor` layers. Selection, experimental gates, and thresholding should not live beside kernel launch tables. |
| P1 | Scheduler has too many responsibilities in one file. | [scheduler.cpp](../scheduler/scheduler.cpp) | Split by concern: admission/batch building, decode workers, sequence lifecycle, distributed transport, eviction/session cleanup. |
| P1 | Batch executor mixes request lifecycle, fairness slicing, prompt/decode assembly, and backend orchestration. | [batch_executor.cpp](../runtime/execution/batch_executor.cpp) | Split into `single_request_executor`, `phased_batch_executor`, and `output_assembly` helpers. Keep `BatchExecutor` as orchestration only. |
| P1 | Benchmark harness is a shell orchestrator that now also acts like a metrics parser and experiment controller. | [run_gguf_comparison_benchmark.sh](../scripts/run_gguf_comparison_benchmark.sh), [classify_benchmark_response.py](../scripts/classify_benchmark_response.py), [compare_decode_traces.py](../scripts/compare_decode_traces.py) | Move orchestration to Python and keep shell only as a thin wrapper or remove it entirely. Existing Python helpers already show the direction. |
| P1 | Config/bootstrap parsing is monolithic and hard to reason about. | [server/main.cpp](../server/main.cpp) | Extract config loading + env normalization into a dedicated server config loader. `main()` should build and run, not parse every knob directly. |
| P2 | Metrics registry is one large monolith with unrelated domains coupled together. | [metrics.cpp](../server/metrics/metrics.cpp) | Split metrics recording/exposition by domain: request, scheduler, InferFlux CUDA, distributed transport, admin/readiness. |

## 3) Most Important Architectural Cleanup

### Native execution policy surface

Current state:
- startup config owns some behavior
- benchmark env overrides some behavior
- forward path reads env directly
- dispatch path reads env directly
- experimental gates are partly selector-owned and partly launch-owned

Result:
- hard to reason about the active runtime mode
- benchmark defaults can accidentally measure the wrong execution mode
- TDD has to manipulate env vars deep in the call stack

Target:
- one `NativeExecutionPolicy`
- backend constructs it once
- forward/dispatch get it as explicit input
- metrics expose the resolved policy

## 4) Refactor Order

1. Extract `NativeExecutionPolicy`
2. Extract native dispatch policy from `fused_quant_gemm.cu`
3. Split `transformer_forward.cu` prefill/decode linear-projection orchestration helpers
4. Split `scheduler.cpp` by responsibility
5. Replace the benchmark shell monolith with a Python orchestrator
6. Split metrics by domain

## 5) Low-Risk Rules For The Maintenance Cycle

- Do not remove sequence leases or paged KV ownership.
- Do not merge native and llama.cpp backends into one abstraction that hides provider identity.
- Do not add more environment-driven behavior inside kernel/forward files.
- Prefer explicit policy structs over more flags.
- Prefer small extraction refactors with TDD over broad rewrites.

## 6) Immediate Next Refactor

`NativeExecutionPolicy` is the best next maintenance refactor because it:
- removes hidden runtime behavior
- reduces benchmark/runtime policy drift
- simplifies testing
- creates a clean handoff point for future dispatch policy modularization
