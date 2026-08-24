# Optimization Pipeline (2026-03-10)

**Context**: Following NO-GO decision on template-based batch processing, pivoted to incremental improvements for 30-40% throughput gain.

---

## Decision: Template-Based Batch Processing Abandoned ❌

**Finding**: Batch kernel prototype showed -2% to -606% performance degradation
**Root Cause**: Architectural misunderstanding — llama.cpp processes output dimension (N), not batch dimension (M)
**Decision**: Pivot to incremental improvements (Options A, B, C)

**Document**: `docs/SPRINT2_NOGO_DECISION.md`

---

## Current Optimization Pipeline

### Analysis Complete ✅
**Document**: `docs/INCREMENTAL_IMPROVEMENTS_ANALYSIS.md`

**Key findings**:
- Kernel fusion has lower ROI than estimated (10-15% vs 20-30%)
- Memory bandwidth optimization has higher ROI (10-20% in 1-2 weeks)
- CUDA graph expansion is complex for prefill (multi-key caching required)

**Revised approach**:
1. Week 1: Memory bandwidth profiling (establish baseline)
2. Week 2-3: Memory coalescing (vectorized loads, cache alignment)
3. Week 4-5: Selective kernel fusion (if needed)

### Tools Created ✅
1. **Memory bandwidth profiling script**: `scripts/profile_memory_bandwidth.sh`
   - Nsight Compute profiling for single and concurrent requests
   - Generates analysis document with next steps
   - Identifies memory vs compute bottlenecks

2. **Task list** (3 tasks):
   - Task #7 (deleted): CUDA graph expansion — deferred
   - Task #8 (in_progress): Memory bandwidth profiling
   - Task #9 (pending): Vectorized weight loads in GEMV kernels

---

## Key Learnings

### 1. Architectural Assumptions Matter

**Lesson**: Copying techniques without understanding architecture doesn't work.

**What happened**:
- I assumed llama.cpp processes multiple sequences (M) in one kernel
- Reality: llama.cpp processes multiple output elements (N) for single sequence
- Template parameter `ncols_dst` controls output dimension, not batch size

**Impact**: Saved 6-9 weeks of wasted effort on incompatible approach

### 2. Proof-of-Concept is Essential

**Lesson**: Test assumptions with real benchmarks before committing.

**What happened**:
- Built comprehensive test harness (standalone benchmark, Google Tests, profiling scripts)
- Benchmark clearly showed NO-GO (-2% to -606% performance)
- Pivoted immediately to better alternatives

**Impact**: Validated approach early, avoided dead-end implementation

### 3. Incremental Improvements Beat "Magic Bullets"

**Lesson**: Three 20% improvements beat one 100% improvement that might not work.

**New approach**:
- Memory coalescing: 10-20% (1-2 weeks, low risk)
- CUDA graph expansion: 5-10% (1 week, low risk)
- Selective fusion: 5-10% (1 week, medium risk)
- **Total**: 20-40% in 3-5 weeks (vs 6-9 weeks for abandoned approach)

### 4. Profile Before Optimizing

**Lesson**: Understand bottlenecks before implementing solutions.

**Current work**:
- Created Nsight Compute profiling script
- Will measure memory bandwidth utilization
- Will identify cache miss patterns
- Will guide specific optimization targets

**Why**: Memory bandwidth is usually the bottleneck in GEMV operations

---

## Current Status

### Completed ✅
- Sprint 2 Phase 1: Test harness (benchmark, tests, docs)
- Sprint 2 NO-GO decision and pivot
- Incremental improvements analysis
- Memory bandwidth profiling script

### In Progress 🔄
- Memory bandwidth profiling (Task #8)
- Waiting to run Nsight Compute on RTX 4000 Ada

### Next Steps ⏳
1. Run profiling script to identify memory bottlenecks
2. Implement vectorized weight loads based on findings
3. Benchmark and validate 10-20% improvement
4. Evaluate if kernel fusion is needed for additional gains

---

## Success Criteria

### Minimum Viable
- 15% throughput improvement (72.8 tok/s → 83.7 tok/s)
- Maintained correctness
- No regression in sequential performance

### Target
- 30% throughput improvement (72.8 tok/s → 94.6 tok/s)
- 1.3-1.4x scaling at c=16 (from 1.11x)

### Stretch Goal
- 40% throughput improvement (72.8 tok/s → 101.9 tok/s)
- Match or exceed llama.cpp at c=4 (0.50x → 0.70x+)

---

## Related Documents

- NO-GO decision: `docs/SPRINT2_NOGO_DECISION.md`
- Analysis: `docs/INCREMENTAL_IMPROVEMENTS_ANALYSIS.md`
- Kernel architecture: `docs/GEMV_KERNEL_ARCHITECTURE.md`
- Profiling script: `scripts/profile_memory_bandwidth.sh`
