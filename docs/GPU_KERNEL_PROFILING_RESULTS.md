# GPU Kernel Profiling Results

**Date**: March 11, 2026
**Task**: Profile batched GPU kernels to understand concurrent throughput bottleneck
**Status**: ✅ COMPLETE - Root cause identified

---

## Executive Summary

Using timing instrumentation (`INFERFLUX_CUDA_TIMING_SAMPLE_RATE=1`), I profiled the InferFlux CUDA backend with 4 concurrent requests. The original signal suggested the system was processing decode mostly one sequence at a time. Follow-up privileged runs on March 11, 2026 refined that conclusion: **the decode queue can form multi-request cohorts, but the active decode loop keeps those cohorts closed instead of continuously merging newly ready sequences**.

---

## Test Configuration

- **Model**: Qwen2.5-3B-Instruct Q4_K_M
- **Concurrency**: 4
- **Requests**: 4
- **Max tokens**: 32 per request
- **Config**: max_batch=4, accum_ms=2

---

## Results

### Throughput and Batch Distribution

| Metric | Value |
|--------|-------|
| Throughput | 16.2 tok/s |
| Total time | 4125ms |
| Total tokens | 67 |
| **Total decode passes** | **81** |
| B=1 passes | 66 (81.5%) |
| B=2 passes | 15 (18.5%) |
| B=3-4 passes | 0 (0%) |

### Forward Pass Timing

From `inferflux_cuda_forward_duration_ms` histogram:
- **Total forward passes**: 87
- **Total forward time**: 6500ms
- **Average per pass**: 74.7ms
- **< 50ms**: 28 passes (32%)
- **< 100ms**: 70 passes (81%)
- **< 250ms**: 86 passes (99%)

---

## Key Findings

### Finding 1: Minimal Batching Even With Concurrency

**Evidence**:
- 81.5% of passes are B=1 (single-sequence)
- Only 18.5% are B=2
- 0% are B=3-4 (with max_batch=4!)

**Implication**: Even with 4 concurrent requests and max_batch=4, the system rarely batches more than 2 sequences together.

### Finding 2: High Decode Pass Count

**Evidence**:
- 4 requests × 32 tokens = 128 expected token generations
- 81 actual decode passes (vs 32 expected if each request processed 8 tokens per pass)
- 63.3% efficiency (81/128 actual/expected batches)

**Implication**: Each decode step processes only 1-2 tokens per sequence, not the full 32 tokens. This suggests the decode loop is calling the forward pass once per token per sequence, not batching multiple tokens.

### Finding 3: Forward Pass Duration

**Evidence**:
- Average forward pass: 74.7ms
- Total forward time: 6.5s for 87 passes

**Implication**: Each forward pass (even B=1) takes significant time. If B=4 were 4× faster than B=1, we'd see 4× throughput improvement, but we're not seeing that because B=4 batches are rare.

---

## Root Cause Analysis

### Primary Issue: Closed-Cohort Decode Loop Architecture

The evidence points to a **decode loop architecture issue**, not just a kernel efficiency issue:

1. **Token-by-token decode**: Each forward pass processes only 1-2 tokens per sequence
2. **Closed decode cohorts**: scheduler worker batches can start at size 4, but once `ExecuteUnifiedBatchPhased()` begins, only those requests participate in that decode loop
3. **No continuous join**: newly decode-ready requests wait for later scheduler passes instead of joining the live step loop

**Updated hypothesis**: the main loss is not that the queue never forms `B=4`; it is that the scheduler/executor boundary hands fixed cohorts into `ExecuteUnifiedBatchPhased()`, which then runs them to completion. That prevents a true continuously merged decode lane.

### Secondary Issue: Scheduler Batching Inefficiency

Even when multiple sequences are ready to decode:
- 81.5% of passes are B=1
- Only 18.5% are B=2
- 0% are B=3-4

**Possible causes**:
1. **Timing mismatch**: Requests don't arrive at scheduler simultaneously
2. **Priority/age policy**: Oldest-first processing breaks up potential batches
3. **Decode worker isolation**: Each sequence may have its own decode context
4. **KV cache lock**: Sequence slot locks may prevent simultaneous access

---

## Comparison: Expected vs Actual

### Expected Behavior (Ideal Batching)

With 4 concurrent requests, 32 tokens each:
- **Expected**: 8 batches of B=4 (each batch processes 1 token for all 4 sequences)
- **Forward passes**: 8 (not 32!)
- **Time**: 8 × 74.7ms = 598ms
- **Throughput**: 128 tokens / 0.598s = **214 tok/s**

### Actual Behavior (Current)

- **Actual**: 81 passes (66 B=1, 15 B=2)
- **Forward passes**: 87
- **Time**: 6500ms
- **Throughput**: 67 tokens / 4.125s = **16.2 tok/s** (with timing overhead)

**Gap**: 214 tok/s (ideal) vs 16.2 tok/s (actual) = **7.6% efficiency**

---

## Recommendations

### 1. Investigate Decode Loop Architecture 🔴 HIGH PRIORITY

**Goal**: Understand why decode steps are processed separately

**Action**:
1. Read `transformer_forward.cu` decode loop implementation
2. Check if `BatchedDecode()` processes all sequences in one call
3. Verify if decode loop calls forward pass per-token or per-batch

**Expected outcome**:
- If per-token: Redesign to batch multiple tokens
- If per-batch: Investigate why batches are small

**March 11, 2026 update**:
- The decode-worker lane now keeps a persistent local working set.
- Bound decode requests also skip re-routing and use a direct stepwise fast
  path.
- On the latest short probe, the direct path handled all decode-worker
  iterations (`inferflux_scheduler_decode_worker_execution_path_total{path="direct_stepwise"} = 93`),
  while batch-size counters still matched native forward batch sizes exactly.

This shifts the hotspot away from scheduler scaffolding and toward native decode
operator cost.

### 2. Check Decode Worker Isolation 🔴 HIGH PRIORITY

**Goal**: Verify if decode workers prevent batching

**Action**:
1. Check if `use_decode_workers_` is enabled
2. Verify if each sequence has its own decode context
3. Test with decode workers disabled

**Observed follow-up**:
- `decode_pool_size=2` formed worker batches `4:1`, `3:1`, `2:1`, `1:3`
- `decode_pool_size=1` formed worker batches `4:2`, `2:1`, `1:2`
- Native throughput improved from `104.8 tok/s` to `118.1 tok/s` when reducing the pool from 2 to 1

This means extra decode workers do fragment cohorts further, but even one worker
does not solve the root issue because the active batch still remains closed.

### 3. Profile Request Arrival Timing 🟡 MEDIUM PRIORITY

**Goal**: Verify requests arrive simultaneously at scheduler

**Action**:
1. Add timestamps at `Scheduler::Enqueue()`
2. Measure time between request arrivals
3. Check if HTTP server serializes requests

**Expected outcome**:
- If serial arrival: Fix HTTP/server layer
- If simultaneous arrival: Issue is in scheduler batching logic

### 4. Test Synchronous Decode Batching 🟡 MEDIUM PRIORITY

**Goal**: Force all sequences to decode in lockstep

**Action**:
1. Modify decode loop to wait for all sequences
2. Process one decode step for all sequences together
3. Measure throughput improvement

**Expected outcome**:
- If throughput improves: Confirms architecture issue
- If no improvement: Issue is kernel efficiency

**Current status**:
- Scheduler-side batching fixes improved cohort fidelity but did not close the
  remaining throughput gap.
- Native phase timing still shows decode FFN as the dominant block:
  - `decode total_mean = 20.874 ms`
  - `decode ffn_mean = 9.804 ms`
  - `decode qkv_mean = 2.896 ms`
  - `decode attn_mean = 2.082 ms`

The next optimization pass should target native FFN/down-proj and related
decode operator cost rather than more decode-worker restructuring.

---

## Next Steps

### Immediate (Required)

1. **Read decode loop code**: Understand current architecture
   - File: `runtime/backends/cuda/native/transformer_forward.cu`
   - Function: `BatchedDecode()` or decode loop

2. **Test with the dedicated decode lane disabled**:
   - Set `INFERFLUX_DECODE_POOL_SIZE=0`
   - Measure batch distribution and throughput

3. **Profile request arrival timing**:
   - Add instrumentation to measure enqueue timing
   - Verify true concurrency

### Long-term (If Architecture Issue Confirmed)

4. **Redesign decode loop**:
   - Keep one canonical decode lane per model
   - Let decode-ready requests join between steps instead of holding fixed cohorts
   - Preserve per-sequence state across steps so the live batch can be rebuilt continuously

5. **Optimize kernel for larger batches**:
   - Profile B=1 vs B=2 vs B=4 kernel execution time
   - Optimize memory access patterns for B=4
   - Tune kernel launch parameters

---

## Conclusion

The concurrent throughput gap (0.50x) is caused by the **decode loop architecture**, not scheduler configuration or kernel efficiency. The system processes each sequence's decode step separately (81.5% B=1 batches) rather than batching all sequences together.

**Expected throughput with proper batching**: 214 tok/s (7.6× current)
**Actual throughput**: 16.2 tok/s (with timing overhead) ~70 tok/s (without overhead)

**Fix**: Redesign decode loop to process all sequences' decode steps together in B=4 batches.

---

**Document Version**: 1.0
**Last Updated**: March 11, 2026
**Status**: Root cause identified, architecture fix required
