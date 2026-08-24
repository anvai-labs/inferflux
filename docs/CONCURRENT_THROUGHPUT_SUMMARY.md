# Concurrent Throughput Investigation: Summary & Recommendations

**Date**: March 10, 2026
**Issue**: InferFlux CUDA achieves 0.83x sequential parity but only 0.50x at concurrency=4
**Status**: Investigation complete, modest improvement identified

---

## Executive Summary

The concurrent throughput gap (0.50x at c=4) is **NOT caused by**:
- Scheduler `max_batch_size` limit (testing showed max_batch=4 performs BEST)
- Insufficient batch accumulation delay (though `accum_ms=2` provides 4% improvement)

**Primary bottleneck**: GPU kernel efficiency and/or decode loop architecture. The system processes 140+ decode passes for 8 concurrent requests, suggesting each decode step is processed separately rather than batching multiple sequences together.

---

## Investigation Results

### Test 1: Batch Size Limit (max_batch_size)

**Hypothesis**: Default `max_batch_size=4` limits concurrent throughput

**Results** (c=4, 16 requests):
| max_batch | Tok/s | Max Batch Observed | B=1% | B=2% | B=3-4% |
|-----------|-------|-------------------|------|------|--------|
| 4         | **69.9** | 3 | 89.4 | 3.2 | 5.0 |
| 8         | 66.8   | 4 | - | - | - |
| 16        | 69.5   | 4 | 86.3 | 8.5 | 6.3 |
| 32        | 67.1   | 4 | - | - | - |

**Conclusion**: `max_batch_size=4` performs BEST. Increasing to 16 or 32 REDUCES throughput by 0.4-4.2%.

### Test 2: Batch Accumulation Delay (batch_accumulation_ms)

**Hypothesis**: `accum_ms=0` causes immediate single-sequence batching

**Results** (c=4, 8 requests):
| accum_ms | Tok/s | Improvement | B=1% | B=2% | B=3-4% |
|----------|-------|-------------|------|------|--------|
| 0        | 71.7  | baseline    | 70.4 | 22.5 | 7.0 |
| 1        | 73.9  | +3.1%       | 79.4 | 3.5  | 17.0 |
| **2**    | **74.7** | **+4.2%**   | 61.4 | 38.6 | 0.0 |
| 5        | 74.6  | +4.1%       | 69.5 | 23.4 | 7.1 |

**Conclusion**: `accum_ms=2` provides **4.2% throughput improvement** with acceptable latency trade-off (+44ms, +4.8%).

---

## Root Cause Analysis

### The Real Problem: Decode Loop Inefficiency

**Evidence**:
- 8 concurrent requests × ~64 tokens = 512 total tokens
- Expected: ~8 batches (one per request per decode step)
- Actual: **140-142 decode passes** (17× more than expected!)

**Implication**: The system is NOT processing multiple sequences in a single batch. Instead, it's processing each sequence's decode step separately, even when multiple sequences are ready.

**Hypothesis**: The batched decode kernel (B=2, B=4) is NOT significantly faster than single-sequence kernels, so there's no incentive to batch. OR, the decode loop architecture prevents effective batching.

### Why Scheduler Doesn't Batch

Even with `max_batch_size=16` and `accum_ms=2`, we see:
- 61-79% single-sequence batches (B=1)
- Only 0-17% batches with 3-4 sequences

**Possible causes**:
1. **Request arrival timing**: Requests don't arrive simultaneously, so scheduler can't group them
2. **Priority/age policy**: Oldest requests are processed first, breaking up potential batches
3. **Token budget constraints**: `max_batch_tokens=16384` may limit batch size
4. **Decode worker isolation**: Each sequence may have its own decode worker, preventing batching

---

## Recommendations

### 1. Set `batch_accumulation_ms=2` in Default Config ✅ DO THIS

**Reasoning**:
- 4.2% throughput improvement with minimal risk
- Low-latency trade-off (+44ms on 920ms baseline = +4.8%)
- Helps with batch grouping without architectural changes

**Implementation**: Update `config/server.yaml`:
```yaml
runtime:
  scheduler:
    batch_accumulation_ms: 2  # Up from 0 for 4% throughput improvement
```

**Expected impact**: 71.7 → 74.7 tok/s at c=4 (4.2% improvement)

### 2. DO NOT Change `max_batch_size` ❌ DON'T DO THIS

**Reasoning**: Testing showed `max_batch_size=4` performs BEST at c=4. Increasing to 16 or 32 REDUCES throughput.

### 3. Profile Batched Decode Kernel Efficiency 🔬 HIGH PRIORITY

**Goal**: Understand why batched kernels aren't providing proportional speedup

**Test**:
1. Use Nsight Systems to trace execution with c=4
2. Measure kernel execution time for B=1 vs B=2 vs B=4
3. Check if memory bandwidth or compute is the bottleneck

**Expected outcome**:
- If B=4 is 4× faster than B=1: Problem is in scheduler/batching logic
- If B=4 is only 1.1× faster: Problem is in kernel efficiency

### 4. Verify True Request Concurrency 🔬 HIGH PRIORITY

**Goal**: Confirm requests are truly concurrent at scheduler level

**Instrumentation**:
- Add timestamps at `Scheduler::Enqueue()`
- Measure time between request arrivals
- Check if HTTP server or scheduler is serializing requests

**Expected outcome**:
- If requests arrive simultaneously: Problem is in batch building
- If requests arrive serially: Problem is upstream (HTTP server, connection handling)

### 5. Test Higher Token Budget (Optional)

**Current**: `max_batch_tokens=16384`
**Test**: Increase to `32768` or `65536`
**Rationale**: Token budget may prevent larger batches from forming

### 6. Test Different Batch Policies (Optional)

**Current**: `policy: priority_age`
**Alternatives**:
- `throughput_balanced`: May batch more aggressively
- `lpm_priority`: May improve prefix cache hits

---

## Expected Impact

### With `batch_accumulation_ms=2`:
- Current (c=4): 71.7 tok/s
- Improved (c=4): 74.7 tok/s
- **Improvement**: +4.2%

### Gap to llama.cpp:
- If llama.cpp achieves ~150 tok/s at c=4 (extrapolated)
- Native at accum_ms=0: 71.7 tok/s (0.48x)
- Native at accum_ms=2: 74.7 tok/s (0.50x)
- **Remaining gap**: 50% (requires kernel optimization or architecture changes)

---

## Conclusion

The concurrent throughput gap is **NOT a configuration issue**. The scheduler's batch size limit and accumulation delay have minimal impact (4% improvement). The primary bottleneck is in the GPU kernel efficiency or decode loop architecture.

**Immediate action**: Set `batch_accumulation_ms=2` for 4% improvement.

**Long-term solution**: Profile batched kernel execution and decode loop to understand why 140+ decode passes are needed for 8 concurrent requests. This likely requires:
1. Kernel optimization to make B=4 significantly faster than 4× B=1
2. OR architecture changes to batch multiple decode steps together
3. OR request coordination to ensure simultaneous arrival at scheduler

---

**Documents Created**:
1. `docs/CONCURRENT_THROUGHPUT_INVESTIGATION.md` - Full investigation details
2. `docs/BATCH_ACCUMULATION_TEST_RESULTS.md` - Accumulation delay test results
3. `scripts/archive/analysis/investigate_concurrent_batch_size.sh` - archived batch-size probe
4. `scripts/archive/test/test_batch_accumulation.sh` - archived accumulation-delay probe

**Status**: Investigation complete, ready to proceed with GPU kernel profiling

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
