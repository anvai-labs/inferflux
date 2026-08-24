# Native KV Device Addressing

## Goal

Remove the remaining host-built KV pointer slabs from native CUDA decode so
batched append and read paths derive addresses on device from compact metadata.

This is the next throughput step after:

- restoring the native Q4_K hot paths
- collapsing batch metadata H2D copies into one slab
- collapsing KV pointer H2D copies from 4 uploads to 2 uploads
- separating startup from steady-state `nsys` profiling

## Current State

Native decode now derives both append and read addresses on device from the
regular KV layout in
[transformer_forward.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/transformer_forward.cu)
and
[flash_attention.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/kernels/flash_attention.cu).

Per-batch pointer slab generation/upload has been removed from the live native
decode path.

The backing KV layout is already regular in
[kv_cache_gpu.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/kv_cache_gpu.cu):

- `GetK(layer, seq_id) = buffer + seq_id * slot_stride + layer * layer_stride`
- `GetV(layer, seq_id) = buffer + seq_id * slot_stride + layer * layer_stride + kv_stride`

So the pointer slabs are not encoding sparse topology. They are encoding simple
address arithmetic that the device can compute directly.

## Why This Matters

Steady-state `nsys` profiling shows startup allocator churn is no longer the
main problem. The remaining native-vs-llama gap is now a real decode hot-path
gap, and `cudaMemcpyAsync` is still dominated by runtime metadata traffic.

That means the next safe win was not another host-side slab compression. It was
removing the host-built pointer slabs entirely.

## Safe First Slice

Do this in two phases, not one.

### Phase A: Device-side append addressing

Scope:

- keep KV storage layout unchanged
- keep FlashDecode read path unchanged
- remove only append pointer slab generation/upload

Implementation:

1. Extend native batched KV append to accept:
   - base KV buffer
   - `slot_stride`
   - `layer_stride`
   - `kv_stride`
   - `seq_ids`
   - `n_past`
   - `layer`
2. Compute `k_dst` and `v_dst` inside the append kernel:
   - `k_base = buffer + seq_id * slot_stride + layer * layer_stride`
   - `v_base = k_base + kv_stride`
   - `dst = base + n_past * kv_dim`
3. Delete the append host slab and its H2D upload.

Status:

- implemented
- `BatchedKvAppendStrided(...)` now derives append addresses on device from
  `kv_buffer + seq_id/stride + layer/stride + n_past`
- native `BatchForward()` no longer builds or uploads append pointer slabs
- CUDA tests cover both the isolated strided append kernel and the synthetic
  batched decode pipeline

Why first:

- append path is simpler than FlashDecode
- destination rows are written once and consumed later
- correctness is easy to verify against the existing path

### Phase B: Device-side read addressing for FlashDecode

Scope:

- keep the same KV storage layout
- stop uploading read pointer slabs

Implementation:

1. Modify `FlashDecodeMultiSeq` to accept layout/stride metadata plus:
   - `seq_ids`
   - `kv_lens`
   - `layer`
2. Compute `k_rd` and `v_rd` inside the kernel from the same address formula.
3. Remove the remaining host read slab and its H2D upload.

Status:

- implemented
- `FlashDecodeMultiSeqStrided(...)` now derives read bases on device from
  `kv_buffer + seq_id/stride + layer/stride`
- native `BatchForward()` no longer builds or uploads read pointer slabs
- direct parity coverage exists for both generic and Qwen decode geometry

Why second:

- FlashDecode is the more sensitive path
- it touches read addressing on every decode step
- it should be changed only after append-side parity is proven

## Constraints

- no KV storage layout change in this slice
- no paged-KV ABI change in this slice
- no session/slot ownership change in this slice
- no per-request CUDA allocation
- preserve CUDA graph safety for fixed device addresses

## Required Contracts

Before rollout:

1. Unit parity:
   - append path writes identical KV rows vs current pointer-slab path
   - FlashDecode outputs match baseline for the same logits/token sequence
2. Integration:
   - short native vs llama.cpp benchmark keeps current accuracy envelope
3. Profiling:
   - steady-state native `cudaMemcpyAsync` count drops again
   - no new steady-state `cudaMalloc/cudaFree`

## Current Evidence

Phase A/Phase B correctness is now covered by:

1. `./build-cuda/inferflux_tests "[batched_decode]"`
2. the new `BatchedKvAppendStrided` parity test in
   [test_batched_decode.cpp](/home/vsingh/code/inferflux/tests/unit/test_batched_decode.cpp)
3. `FlashDecodeMultiSeqStrided matches per-sequence FlashAttention decode for Qwen geometry`
   in [test_native_forward.cpp](/home/vsingh/code/inferflux/tests/unit/test_native_forward.cpp)

Short end-to-end benchmark after Phase B on the source branch:

- native: `113.9 tok/s`
- `llama_cpp_cuda`: `115.0 tok/s`
- exact match: `6/8`
- artifact:
  [comparison_20260311_122203.json](/home/vsingh/code/inferflux/gguf_benchmark_results/comparison_20260311_122203.json)

Serialized steady-state `nsys` profiling on this branch is now trustworthy
because [profile_backend.sh](/home/vsingh/code/inferflux/scripts/profile_backend.sh)
uses backend-specific default ports plus a lock file to prevent overlapping
profiling sessions.

Current API-level result:

- native: `cudaMemcpyAsync calls=754 total=1548.199 ms avg=2.053 ms`
- llama.cpp: `cudaMemcpyAsync calls=924 total=3.179 ms avg=3.441 us`
- both backends now show `cudaMalloc=0` and `cudaFree=0` in steady state

That means the remaining native gap is no longer allocator churn or KV pointer
slab upload. The dominant unresolved issue is the cost of the remaining
`cudaMemcpyAsync` path itself.

## Follow-up: Pinned Host Sampling and Copy Trace

After adding:

- pinned host buffers for sampler result/logits staging in
  [gpu_sampler.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/gpu_sampler.cu)
- per-site native copy tracing in
  [cuda_copy_trace.h](/home/vsingh/code/inferflux/runtime/backends/cuda/native/cuda_copy_trace.h)

the short `c=4` benchmark reported:

- native: `123.6 tok/s`
- `llama_cpp_cuda`: `127.9 tok/s`
- ratio: `0.97x`
- artifact:
  [comparison_20260311_123017.json](/home/vsingh/code/inferflux/gguf_benchmark_results/comparison_20260311_123017.json)

The native server log now shows the exact live copy sites:

- `batch.meta_h2d`: `128 calls`, `1.677 ms total`
- `sampler.batch_result_d2h`: `128 calls`, `0.612 ms total`
- `forward.token_ids_h2d`: `10 calls`, `0.174 ms total`
- `forward.residual_d2d`: `10 calls`, `0.100 ms total`
- `batch.residual_d2d`: `9 calls`, `0.094 ms total`
- `sampler.greedy_result_d2h`: `10 calls`, `0.063 ms total`

Serialized steady-state `nsys` then moved native to:

- `cudaMemcpyAsync calls=767 total=5.697 ms avg=7.428 us`

which is now effectively at parity with llama.cpp:

- `cudaMemcpyAsync calls=924 total=5.614 ms avg=6.075 us`

This confirms the old native copy bottleneck was mostly pageable host staging
in sampling, not the remaining metadata slab itself.

The next native bottleneck is now `cudaStreamSynchronize`, not
`cudaMemcpyAsync`.

## Success Metric

The steady-state profiler has now moved from:

- host-built pointer slabs
- two KV pointer H2D uploads per batch

to:

- metadata-only batch upload
- zero KV pointer H2D uploads per batch in the live native decode path

That is the cleanest next step toward exceeding llama.cpp on native decode
throughput without changing model math or destabilizing batching.
