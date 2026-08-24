# Backend Parity: llama.cpp, CUDA, and MLX

## Goal
Keep backend behavior consistent across hardware targets while preserving
target-specific optimization paths and minimizing duplicated control logic.

## Current Parity Snapshot (March 9, 2026)

| Area | llama.cpp provider path (`llama_cpp_cuda`) | InferFlux provider path (`inferflux_cuda`) | MLX path |
| --- | --- | --- | --- |
| Model load lifecycle | Shared in `LlamaCppBackend` | InferFlux CUDA runtime via `InferfluxCudaRuntime` + strict-inferflux policy hooks | Separate MLX-native loader/engine |
| Phased prefill/decode APIs | Supported | Supported (sync batched path + decode/prefill overlap path) | Supported (native overrides) |
| Unified batch async lanes | Supported | Contract intentionally disabled today (`SupportsAsyncUnifiedBatch()==false`) | Supported (native overrides) |
| GGUF quantized overlap safety | N/A (llama.cpp path) | Active lane-local quantized map/adapter ownership (no hard GGUF overlap disablement) | N/A |
| Memory policy | Compatibility/runtime-managed | Native dequant policy and KV planning are explicit and load-scoped | Hardware-specific |
| Capability surface | Broad compatibility baseline | Endpoint parity contracts are closed for completion/chat/embeddings through native parity delegate paths; speculative decoding contract is native-owned | Hardware-specific capabilities |
| Backend selection | Factory + router | Factory + router (native-preferred with explicit strict/fallback policy) | Factory + router |
| Primary role | Stability + model-format compatibility | Performance/control core with active kernel maturity roadmap | Apple Silicon performance path |

## Design Principles

1. Single control-plane path:
   - Scheduler, executor, and router should not branch on backend internals.
2. Sharded backend policy modules:
   - Backend selection and tuning live outside concrete backend implementations.
3. Hardware optimization stays local:
   - CUDA/MLX optimization hooks remain inside their backend classes.
4. Capability-first evolution:
   - New backend features are surfaced via traits/capabilities, not hard-coded
     checks in scheduler/executor.

## Current Value Split (Why Keep Two CUDA Backends)

| Backend | What it gives today | Why it stays |
| --- | --- | --- |
| `inferflux_cuda` | InferFlux-owned runtime control, strict provider identity, sync-first batched overlap path, and explicit memory-policy controls | Throughput and architecture headroom |
| `llama_cpp_cuda` | Stable compatibility baseline with mature feature surface and broad GGUF behavior | Production safety net and fallback path |

## Current Gaps to Close

1. Native throughput-path maturity:
   - `SupportsAsyncUnifiedBatch()` is still disabled in native runtime because sync batched execution is currently the faster path.
2. Native-first parity independence:
   - completion/chat/embeddings parity currently depends on delegate availability
     for some model-format layouts.
3. Native quantized heavy-batch maturity:
   - GGUF overlap safety is now in place, but fused-kernel + large-model perf
     proof is still pending.

## Next Steps

1. Keep async optional and only re-enable it if it feeds the same sync batched core without fragmenting work.
2. Move parity surfaces from delegate-backed implementation to native-first
   implementation where practical.
3. Validate quantized GGUF throughput on larger batches/models and wire
   non-regression checks into GPU CI.
