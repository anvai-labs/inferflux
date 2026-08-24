# Memory Accounting Strategy

## Goal

Track native-runtime memory by **ownership** and **lifetime**, so VRAM policy is explicit:

- model weights are loaded once and shared
- decoder workspace is reused across requests
- KV pages have a separate lifecycle from model weights
- session/slot managers own metadata, not large GPU buffers

## Memory Domains

| Domain | Examples | Owner | Lifetime |
| --- | --- | --- | --- |
| `weights` | quantized GGUF buffer, dense safetensors buffer | model loader | model |
| `weights_dequantized` | optional persistent dense cache for GGUF fallback paths | model loader | model or batch |
| `workspace_device` | forward scratch, sampler scratch, logits, overlap-lane scratch | native executor / forward runtime | model |
| `workspace_host_pinned` | pinned metadata staging, pinned pointer tables | native executor / forward runtime | model |
| `kv_cache` | active KV pages | KV allocator | pool |
| `kv_prefix_cache` | prefix-retained KV pages | prefix cache | pool |
| `session_metadata` | `session_id -> sequence_id/generation/block_table` | scheduler/session manager | session |
| `batch_ephemeral` | temporary fallback dequant buffers, temporary batch scratch | workspace arena | batch |

## Ownership Rules

### Model lifecycle

- allocate weights at model load
- allocate reusable decoder workspace at model load
- allocate pinned host staging at model load
- keep these alive until model unload

### KV lifecycle

- allocate KV pages under the KV allocator
- sequence slots and session handles only reference KV state
- session expiry releases leases and refs; it does not own decoder scratch

### Batch lifecycle

- fallback-only temporary dequant buffers should come from a reusable arena
- the hot path should not issue `cudaMalloc/cudaFree`

## Policy Mapping

Existing `DequantizedCachePolicy` remains narrow:

- `kNone`: no persistent dense GGUF cache
- `kBatchLifetime`: dense fallback buffers may exist only for batch scope
- `kModelLifetime`: dense fallback buffers may remain model-resident

This policy does **not** control decoder scratch, pinned staging, sampler buffers, or logits. Those are runtime workspace and should exist whenever the model is loaded.

## First Implementation Slice

Implemented now:

- `MemoryDomain` and `MemoryLifetime`
- `ModelMemoryLedger`
- startup accounting in InferFlux CUDA startup
- live batch-ephemeral GGUF scratch accounting for quantized fallback paths
- native KV slot-state accounting split into `active`, `prefix_retained`, and `free`
- export through:
  - Prometheus metrics
  - `GET /v1/admin/cache`
- exact startup snapshots for:
  - model weights
  - KV cache
  - primary forward device workspace
  - primary forward pinned host workspace
  - sampler device workspace
  - logits buffers
  - overlap-lane forward/sampler/logits workspaces when enabled

Not yet implemented:

- live high-water updates during runtime
- paged-KV prefix-cache accounting for the non-native paged path
- reusable arena accounting beyond the current GGUF scratch buffer
- per-session metadata accounting
- persistent GGUF dequant-cache accounting beyond loader-visible startup state

## Current Wiring

Startup wiring lives in:

- [model_memory_ledger.h](/home/vsingh/code/inferflux/runtime/backends/cuda/native/model_memory_ledger.h)
- [model_memory_ledger.cpp](/home/vsingh/code/inferflux/runtime/backends/cuda/native/model_memory_ledger.cpp)
- [inferflux_cuda_executor.cpp](/home/vsingh/code/inferflux/runtime/backends/cuda/inferflux_cuda_executor.cpp)

Workspace byte reporting is provided by:

- [llama_forward.h](/home/vsingh/code/inferflux/runtime/backends/cuda/native/llama_forward.h)
- [transformer_forward.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/transformer_forward.cu)
- [gpu_sampler.h](/home/vsingh/code/inferflux/runtime/backends/cuda/native/gpu_sampler.h)
- [gpu_sampler.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/gpu_sampler.cu)

## Export Contract

At native pipeline initialization and on batch-scratch allocate/release transitions, InferFlux now refreshes one canonical ledger snapshot that answers:

- how much memory is model-resident
- how much is KV-resident
- how much is decoder workspace
- how much temporary GGUF fallback scratch is currently resident
- how much pinned host memory is reserved for metadata staging

That snapshot is exported in three ways:

- startup log line from the native executor
- Prometheus gauges in `/metrics`
- JSON under `memory.inferflux_cuda_model` and `memory.inferflux_cuda_kv` in `GET /v1/admin/cache`

This makes the next optimization steps measurable:

1. move remaining per-batch CUDA allocations into reusable arenas
2. extend KV accounting from fixed-slot InferFlux CUDA into paged-KV ownership accounting
3. expose session metadata and per-batch arena high-water accounting
4. tie future memory policies to the ledger instead of ad hoc logs
