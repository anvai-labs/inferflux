# InferFlux CUDA Profiling Analysis — 2026-03-18

**Hardware:** NVIDIA RTX 4000 Ada (sm_89, 48 SMs, 20 GB, 360 GB/s DRAM)
**Model:** Qwen2.5-3B-Instruct Q4_K_M (2 GB weights, 36 layers, 2048 hidden, 16/2 heads)
**Config:** Q8_1 disabled, fused gate_up OFF, CUDA graphs OFF, single-sequence decode
**Tools:** nsys 2025.6.3 (timing), ncu 2026.1.0 (hardware counters, admin)

## Executive Summary

GEMV kernels achieve **89-94% memory bandwidth** utilization — they are near-peak.
The 0.63x throughput gap vs llama.cpp is NOT in kernel efficiency. It's in:

1. **Host-GPU synchronization** (9.87ms sampler sync per token, includes all GPU work)
2. **Kernel launch overhead** (~685 launches/token vs llama.cpp ~300-400)
3. **Overhead kernels** (~1.5ms in QuantizeRows, BiasAdd, RmsNorm, ResidualAdd per token)

## Key Numbers

| Metric | Value |
|--------|-------|
| Decode GPU kernel time | ~8.5-9.0 ms/token |
| Sampler sync wall-clock | 9.87 ms/token (avg over 49 decode steps) |
| Probe throughput (50 tokens) | ~101 tok/s (kernel-limited) |
| Server throughput (with HTTP) | ~64 tok/s |
| Theoretical peak (360 GB/s, 2 GB weights) | 180 tok/s (5.56 ms/token) |
| llama.cpp reference | 119.9 tok/s (8.34 ms/token) |

## ncu Hardware Counter Results (1 prefill pass, 500 kernels)

| Kernel | Count | Mem BW% | SM% | Occupancy% | Regs | Block | Grid |
|--------|-------|---------|-----|-----------|------|-------|------|
| q4k_packed_group<2> | 40 | **94.0%** | 77.9% | 91.6% | 40 | 256 | (256,5) |
| q4k_packed | 45 | **89.7%** | 74.7% | 90.6% | 33 | 256 | (256,5) |
| q4k_packed_group<3> | 17 | **84.6%** | 73.1% | 90.0% | 40 | 256 | (256,5) |
| q6k_packed | 24 | **92.3%** | 62.1% | 71.5% | 40 | 256 | (32,5) |
| FlashAttention2 GQA | 29 | 5.6% | 1.9% | 8.3% | 72 | 128 | (5,2) |
| SiluMulQuantize | 28 | 11.0% | 1.1% | 16.7% | 16 | 256 | (5,1) |
| RmsNorm | 57 | 6.7% | 3.1% | 65.0% | 16 | 1024 | (5,1) |
| ResidualAdd | 57 | 20.9% | 1.1% | 16.1% | 16 | 256 | (40,1) |

## nsys Kernel Time Breakdown (20 forward passes, 1 prefill + 19 decode)

| Kernel | Total Time | % of Kernel Time | Instances | Avg/Instance |
|--------|-----------|-----------------|-----------|-------------|
| q4k_packed_group<2> | 69.7 ms | 29.8% | 1080 | 64.6 µs |
| q6k_packed | 41.9 ms | 17.9% | 740 | 56.6 µs |
| q4k_packed | 24.4 ms | 10.5% | 1080 | 22.6 µs |
| dequantize_q4_k (warmup) | 24.4 ms | 10.4% | 1 | 24.4 ms |
| FlashDecodeMultiSeq GQA | 15.6 ms | 6.7% | 684 | 22.8 µs |
| transform_downproj_q6 (warmup) | 11.7 ms | 5.0% | 18 | 651 µs |
| SiluMulQuantize | 10.7 ms | 4.6% | 720 | 14.8 µs |
| transform_downproj_q4 (warmup) | 8.2 ms | 3.5% | 18 | 454 µs |
| QuantizeRowsSym | 6.7 ms | 2.9% | 2180 | 3.1 µs |
| q4k_packed_group<3> | 6.6 ms | 2.8% | 360 | 18.4 µs |
| RmsNorm | 4.9 ms | 2.1% | 1460 | 3.4 µs |
| BiasAdd | 2.9 ms | 1.2% | 2160 | 1.3 µs |
| ResidualAdd | 1.9 ms | 0.8% | 1440 | 1.3 µs |
| BatchedArgmax | 1.7 ms | 0.7% | 19 | 91.9 µs |

## Sync Trace (50 decode tokens)

```
site=sampler.result_ready       calls=1    avg_us=1289.600   (prefill)
site=sampler.batch_result_ready calls=49   avg_us=9870.065   (decode — THIS IS THE BOTTLENECK)
site=sampler.logits_ready       calls=51   avg_us=149.529    (logprob D2H copy)
```

The `sampler.batch_result_ready` sync blocks the host for **9.87 ms per decode token**.
This sync waits for the completion_event_ which follows the entire forward pass + argmax + D2H,
making it the pipeline serialization point. All GPU work from forward kernels through sampling
must complete before the next token can begin.

## NVTX Phase Breakdown (20 forward passes)

| Phase | Total Time | Per Decode (19 avg) | % of Decode |
|-------|-----------|--------------------|----|
| BatchedDecode | 217.9 ms | 11.47 ms | 100% |
| BatchForward (kernel launch) | 72.3 ms | 3.80 ms | 33% |
| Host overhead (sync + scheduling) | 145.6 ms | 7.67 ms | 67% |

## Per-Decode-Token Budget

```
GPU kernel execution:      ~8.5 ms  (measurable via nsys kernel sum)
Inter-kernel GPU idle:     ~1.0 ms  (gaps between 685 kernel launches)
Sampler argmax + D2H:      ~0.1 ms
Host-side sync overhead:   ~0.4 ms  (cudaEventSynchronize API call itself)
------------------------------------------------------
Total decode latency:      ~10.0 ms → ~100 tok/s (probe-measured)
```

## Optimization Priorities

### P0: Enable CUDA Graphs for Decode (~2-3 ms savings → +30-40 tok/s)

CUDA graphs eliminate per-launch overhead and inter-kernel gaps for the fixed
decode graph structure (same kernels, same grid/block dims every token).

- Currently disabled on Windows (`INFERFLUX_DISABLE_CUDA_GRAPH=1`)
- Eliminates ~685 individual launches, replacing with single `cudaGraphLaunch`
- Removes inter-kernel GPU idle gaps (~1.0 ms)
- Removes per-launch CPU-side overhead (~0.3 ms)
- **Expected: 8.5ms → ~6.5ms per token → ~154 tok/s**

### P1: Fix Q8_1 Kernel Crash on Windows (~15% improvement)

Q8_1 pre-quantized activations eliminate QuantizeRowsSym overhead and
allow grouped Q8_1 pair GEMV kernels (validated at 75.5 tok/s on Linux
vs 64 tok/s without). The crash is `illegal memory access` in the
Q8_1 group_row_pair kernel path. Not a struct size issue (static_assert
confirms 36 bytes). Likely a buffer sizing or alignment issue.

### P2: Fuse Small Overhead Kernels (~0.5-1.0 ms savings)

Current overhead kernels (1.5ms/token): QuantizeRows (6.7ms/20=335µs),
BiasAdd (2.9ms/20=145µs), ResidualAdd (1.9ms/20=95µs), RmsNorm (4.9ms/20=245µs).

Opportunities:
- Fuse `RmsNorm + QuantizeRows` into single kernel (saves 1 launch + L2 round-trip)
- Fuse `BiasAdd + BiasAdd + BiasAdd` (3 per layer for Q/K/V) into one vectorized kernel
- Fuse `ResidualAdd` into preceding down_proj kernel output write

### P3: Reduce Kernel Launch Count (Long-term)

685 kernels/token is 1.7-2.3x more than llama.cpp (~300-400).
With CUDA graphs this matters less, but without graphs, each launch adds ~5-10µs.
Further fusion opportunities: embed norm+quant into GEMV epilogue.

## Theoretical Ceiling

| Scenario | Time/Token | Tok/s |
|----------|-----------|-------|
| Current (Q8_1 OFF, graphs OFF) | 9.87 ms | ~101 |
| + CUDA graphs | ~7.0 ms | ~143 |
| + Q8_1 fix | ~6.5 ms | ~154 |
| + Kernel fusion | ~6.0 ms | ~167 |
| Theoretical (100% BW, zero overhead) | 5.56 ms | ~180 |
| llama.cpp reference | 8.34 ms | 119.9 |
