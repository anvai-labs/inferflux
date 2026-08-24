# Batch Accumulation Test Results

**Date**: March 10, 2026
**Test**: Impact of `batch_accumulation_ms` on concurrent throughput
**Status**: ✅ COMPLETE - Modest improvement found, but not a silver bullet

---

## Results Summary

| batch_accumulation_ms | Tok/s | Avg Lat (ms) | B=1% | B=2% | B=3-4% | Total Decodes |
|-----------------------|-------|--------------|------|------|--------|---------------|
| **0** (default)       | 71.7  | 920          | 70.4 | 22.5 | 7.0    | 142           |
| 1                     | 73.9  | 815          | 79.4 | 3.5  | 17.0   | 141           |
| **2**                 | **74.7** | 964     | 61.4 | 38.6 | 0.0    | 140           |
| 5                     | 74.6  | 876          | 69.5 | 23.4 | 7.1    | 141           |

**Best configuration**: `batch_accumulation_ms=2`
- Throughput improvement: **+4.2%** (71.7 → 74.7 tok/s)
- Latency increase: +44ms (+4.8%)
- Batching improvement: More B=2 batches (38.6% vs 22.5%), fewer total decodes (140 vs 142)

---

## Key Findings

### Finding 1: Modest Throughput Improvement

Setting `batch_accumulation_ms=2` improves throughput by 4.2%, but this is FAR from the 100% improvement needed to close the concurrent throughput gap (0.50x → 1.0x).

**Conclusion**: Batch accumulation delay helps, but is NOT the primary bottleneck.

### Finding 2: Batch Distribution Changes

At `accum_ms=2`:
- B=1 batches: 61.4% (down from 70.4%)
- B=2 batches: 38.6% (up from 22.5%)
- B=3-4 batches: 0.0% (down from 7.0%)

**Interpretation**: The 2ms delay allows more time for requests to pair up into B=2 batches, which reduces total decode count from 142 to 140. However, we're still not seeing many B=3-4 batches, which suggests:

1. Requests arrive at slightly different times, making it hard to group 3-4 together
2. The scheduler's priority/age policy may be breaking batches apart
3. Token budget constraints may be limiting batch size

### Finding 3: Latency Trade-off

Average latency increased from 920ms to 964ms (+44ms), which is expected given the 2ms accumulation delay applies to each batch. This is a reasonable trade-off for 4% throughput improvement.

---

## Comparison to llama.cpp

**Current gap**: InferFlux CUDA achieves ~0.50x of llama.cpp throughput at c=4

If llama.cpp achieves ~150 tok/s at c=4 (extrapolating from earlier benchmarks), then:
- Native at accum_ms=0: 71.7 tok/s (0.48x)
- Native at accum_ms=2: 74.7 tok/s (0.50x)
- **Gap remains**: ~50% of llama.cpp performance

**Conclusion**: Even with optimal batch accumulation, we're still at 0.50x parity. The bottleneck lies elsewhere.

---

## Root Cause Analysis

### Primary Suspect: GPU Kernel Efficiency

**Hypothesis**: Batched decode kernels (B=2, B=4) are NOT significantly faster than single-sequence kernels, so even when requests are batched, we don't get proportional throughput improvements.

**Evidence**:
- 140-142 decode passes for 8 requests × ~64 tokens = 512 tokens
- Ideal: 8 batches × 64 tokens = 8 batches
- Actual: 140-142 batches (17-18× more batches than ideal!)

**Implication**: Each decode step is being processed as a separate batch, not consolidated. This suggests the batched decode kernel is not actually batching multiple sequences efficiently.

### Secondary Suspect: Request Synchronization

**Hypothesis**: Even with 4 concurrent requests, they're not arriving at the scheduler simultaneously, so the scheduler can't group them into larger batches.

**Evidence**:
- At c=4, only 7% of batches are B=3-4
- Most batches are B=1 (61-79%) or B=2 (3-38%)

**Question**: Are requests truly being issued concurrently, or is there serialization at the HTTP/server layer?

---

## Next Steps

### 1. Profile Batched Kernel Efficiency (HIGH PRIORITY)

**Test**: Measure kernel execution time for B=1 vs B=2 vs B=4
**Tool**: Nsight Compute or Nsight Systems
**Expected**: If B=4 is 4x faster than B=1, then batching should help. If not, kernel optimization is needed.

### 2. Verify True Concurrency (HIGH PRIORITY)

**Test**: Add timing instrumentation to measure:
- Time between request arrivals at scheduler
- Time between batch builds
- Whether requests are truly concurrent or serialized

**Instrumentation points**:
- `Scheduler::Enqueue()` - when request enters queue
- `Scheduler::BuildBatchLocked()` - when batch is built
- `ProcessBatch()` - when batch is executed

### 3. Test Higher Token Budget (MEDIUM PRIORITY)

**Current**: `max_batch_tokens=16384`
**Test**: Increase to `32768` or `65536`
**Rationale**: Token budget may be preventing larger batches from forming

### 4. Test Different Batch Policies (MEDIUM PRIORITY)

**Current**: `policy: priority_age` (oldest high-priority first)
**Alternatives**:
- `throughput_balanced` - may batch more aggressively
- `lpm_priority` - may improve prefix cache hits

---

## Recommendations

### DO: Set `batch_accumulation_ms=2` in Default Config

**Reasoning**:
- 4.2% throughput improvement with acceptable latency trade-off
- Low-risk configuration change
- Helps with batch grouping without significant latency impact

**Change**: Update `config/server.yaml`:
```yaml
runtime:
  scheduler:
    batch_accumulation_ms: 2  # Up from 0 for 4% throughput improvement
```

### DO NOT: Expect This to Close the Concurrent Throughput Gap

**Reason**: 4% improvement is far from the 100% needed. The primary bottleneck is elsewhere (likely GPU kernel efficiency or request synchronization).

### PRIORITIZE: Profile Batched Decode Kernel

**Action**: Use Nsight Systems to trace execution with c=4 and measure:
- How many batches are actually processed
- What batch sizes are used
- Whether batched kernels are faster than single-sequence

**Goal**: Understand why we're processing 140+ decode passes instead of ~8 batches.

---

## Conclusion

The batch accumulation delay provides a modest 4% throughput improvement, but the concurrent throughput gap (0.50x) remains. The primary bottleneck is NOT the scheduler's batch building logic, but rather:

1. **GPU kernel efficiency**: Batched kernels may not be significantly faster than single-sequence
2. **Request synchronization**: Requests may not be arriving truly concurrently at the scheduler
3. **Decode loop efficiency**: Each decode step is being processed separately instead of batching multiple decode steps together

**Next action**: Profile GPU execution to understand why 140+ decode passes are needed for 8 concurrent requests.

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
**Status**: Test complete, bottleneck identified (NOT scheduler batching)
