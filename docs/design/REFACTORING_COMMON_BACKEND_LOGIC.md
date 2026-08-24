# Common Backend Logic Refactor Note

**Snapshot date:** March 9, 2026  
**Status:** partially landed, lower priority than runtime/perf work

## 1) Intent

```mermaid
flowchart LR
    A[Shared backend types] --> B[Shared backend interface]
    B --> C[Backend-specific runtimes]
    C --> D[Scheduler and router stay backend-agnostic]
```

## 2) What Has Already Landed

| Area | Current state |
|---|---|
| Shared types | `runtime/backends/common/backend_types.h` exists |
| Shared interface | `runtime/backends/common/backend_interface.h` exists |
| Shared batch helpers | `runtime/backends/common/batching_utils.h` exists |
| Backend inheritance | `LlamaCppBackend` implements the common backend interface |

## 3) What Is Still Incomplete

| Gap | Why it is still deferred |
|---|---|
| Residual `LlamaCppBackend` type coupling in some runtime/execution paths | Untangling it is broad churn with limited near-term throughput value |
| Cleaner native/runtime interface boundaries | Current priority is native quantized hot-path maturity, not hierarchy surgery |
| Full removal of compatibility aliases | Safe once batch execution and backend contracts settle further |

## 4) Current Guidance

1. Reuse the common backend types for new behavior.
2. Avoid large inheritance/interface rewrites unless they remove active duplication in touched code.
3. Prefer incremental extraction over architectural rewrites that compete with throughput and distributed work.

## 5) Definition of Done

This refactor is only worth finishing when it delivers one of:

1. less duplicated batching/runtime logic across active backends,
2. clearer native-runtime ownership boundaries,
3. lower regression risk for new backend features.

## 6) Related Docs

- [Backend_Parity_LlamaCpp_CUDA_MLX](Backend_Parity_LlamaCpp_CUDA_MLX.md)
- [../Architecture](../Architecture.md)
