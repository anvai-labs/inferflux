# FFN Kernel Fusion Implementation Status

**Date**: March 10, 2026
**Task**: #12 - Implement FFN kernel fusion (gate+up+SiLU+down)
**Status**: ⏳ IN PROGRESS - Tiled bring-up kernel working, benchmarked slower than baseline

---

## Summary

FFN kernel fusion implementation is in progress. Phase 1 (analysis and
profiling) is complete. Phase 2 now has a parity-tested tiled bring-up kernel
that reuses activated intermediate values across an output tile, but it is not
yet selected by the runtime hot path.

---

## Completed Work

### Phase 1: Analysis and Profiling ✅

**Profiling Results** (from `INFERFLUX_CUDA_PHASE_TIMING=1`):

**Prefill** (tokens=16):
- `ffn_proj` (gate+up): 50.02 ms (42% of FFN time)
- `ffn_silu`: 0.00 ms ✅ **Already fused into down_proj!**
- `ffn_down`: 68.95 ms (58% of FFN time)
- **FFN total**: 118.97 ms (40% of total 299.26 ms)

**Decode** (tokens=1):
- `ffn_proj` (gate+up): ~2.0 ms (48% of FFN time)
- `ffn_silu`: 0.00 ms ✅ **Already fused!**
- `ffn_down`: ~2.1 ms (52% of FFN time)
- **FFN total**: ~4.1 ms (38% of total ~10.7 ms)

**Key Finding**: SiLU activation is already fused into the down_proj kernel via `TryQ8_1SiluMulGemv`. The remaining fusion opportunity is to combine gate+up+down into a single kernel.

### Expected Improvement (Revised)

- **Current**: gate+up (separate kernel) → down_proj (with SiLU fused)
- **Proposed**: gate+up+SiLU+down (single kernel)
- **Eliminate**: Intermediate gate/up writes to global memory
- **Expected savings**: 10-20% of FFN time = 3.8-8% overall improvement ✅

### Documentation Created

1. **`docs/FFN_FUSION_ANALYSIS.md`** - Comprehensive technical analysis
   - Current architecture breakdown
   - Memory traffic patterns
   - Implementation complexity analysis
   - Risk assessment
   - Success criteria

2. **`scripts/archive/profile/profile_ffn_breakdown.sh`** - archived profiling script
   - Automated FFN performance profiling
   - Nsight Systems integration
   - Phase timing extraction

3. **Kernel Design Document** - This document

---

## Implementation Progress

### Files Created

1. **`runtime/backends/cuda/native/kernels/fused_ffn_gemm.cuh`** ⏳ PARTIAL
   - ✅ SwiGLU activation function
   - ✅ Correct `down_proj` Q4_K indexing
   - ✅ Output-tiled kernel structure
   - ✅ Activated intermediate tile reuse across hidden-output tile
   - ⏳ **Needs**: Performance validation on live FFN geometry
   - ⏳ **Needs**: Controlled runtime rollout

2. **Runtime policy** ✅ CLEANED UP
   - ✅ No runtime rollout flag kept
   - ✅ The fused FFN path remains a benchmark/testing path only until it wins

### Files To Be Created

3. **Dispatch wrapper** ✅ DONE
   - `fused_quant_gemm.cu`: `FusedQuantGemm::FusedFfnQ4K(...)`
   - Explicit bring-up/testing entry point
   - Not runtime-selected yet

4. **Integration** ⏸️ DEFERRED
   - `transformer_forward.cu` is intentionally unchanged
   - runtime rollout is blocked on performance, not correctness

5. **Correctness Test** ✅ DONE
   - `tests/unit/test_native_forward.cpp`
   - CUDA parity check against dequantized FP16 reference
   - Exercises multiple intermediate tiles and multiple output tiles

6. **Performance Benchmark** ✅ DONE
   - Isolated benchmark executable: `benchmark_fused_ffn`
   - Compares the real current FFN path vs `FusedFfnQ4K(...)`
   - Measured on the live decode geometry:
     - `M=1, K=2048, N_inter=11008, N_hidden=2048`
     - baseline: `0.118 ms`
     - fused: `32.741 ms`
     - speedup: `0.004x`
     - max abs diff: `0.001463`
     - `M=2`
     - baseline: `0.181 ms`
     - fused: `59.791 ms`
     - speedup: `0.003x`
     - max abs diff: `0.001463`

---

## Kernel Design Challenges

### Challenge 1: Intermediate Dimension Size ✅ Resolved for bring-up

**Problem**: The intermediate dimension (N_inter) is large (e.g., 5632 for Qwen2.5-3B with hidden=2048, expansion=2.75).

**Initial Approach**: Store all intermediate gate+up outputs in shared memory
- **Issue**: 5632 × 4 bytes = 22KB per intermediate value
- **Result**: Doesn't fit in shared memory (typically 48KB max)

**Revised Approach**: Tile intermediate dimensions
- Compute a tile of gate+up activations once
- Store activated tile in shared memory
- Reuse that tile across a hidden-output tile
- **Status**: ✅ Implemented for the Q4_K bring-up kernel

### Challenge 2: Weight Layout Complexity ✅ Resolved

**Problem**: Three weight matrices with different layouts:
- `gate_weight`: [K × N_inter] stored as [N_inter, num_blocks]
- `up_weight`: [K × N_inter] stored as [N_inter, num_blocks]
- `down_weight`: [N_inter × N_hidden] stored as [N_hidden, num_blocks]

**Issue**: Correct indexing for down_weight when iterating through intermediate dimensions

**Resolution**:
- `Q4_K` packs along the reduction dimension
- for `down_proj`, that reduction dimension is `N_inter`
- `5632 / 256 = 22`, so the Qwen FFN shape is valid

**Status**: ✅ Corrected in kernel code

### Challenge 3: Thread Block Organization ⏳

**Current Bring-Up Design**:
- One CTA computes one batch row
- One CTA owns an 8-output hidden tile
- 8 warps per CTA (256 threads)
- Activated intermediate values are produced in 32-element tiles
- The same activated tile is reused across all 8 outputs

**Potential Issues**:
- Load imbalance if N_inter not evenly divisible by 8
- Memory coalescing for down_weight access
- Occupancy and register pressure

**Status**: ⏳ Needs profiling and optimization

---

## Current Kernel Implementation

### File: `fused_ffn_gemm.cuh`

**Components**:
1. ✅ `SwiGLU()` - Activation function
2. ✅ `FusedFFNGemmQ4K()` - Main kernel
3. ✅ Shared memory for activation
4. ✅ Warp-level reductions
5. ⏳ Down projection indexing (needs fix)

**Kernel Signature**:
```cpp
template <typename BlockType>
__global__ void FusedFFNGemmQ4K(
    const BlockType *__restrict__ gate_weight,
    const BlockType *__restrict__ up_weight,
    const BlockType *__restrict__ down_weight,
    const half *__restrict__ activation,
    half *__restrict__ output,
    int N_inter, int N_hidden, int M, int K
);
```

**Algorithm**:
1. Load activation row into shared memory
2. For each 32-element intermediate tile:
   a. Compute gate and up activations once per intermediate
   b. Store `SwiGLU(gate, up)` into shared memory
   c. Reuse that activated tile across an 8-output hidden tile
   d. Accumulate down-proj outputs warp-by-warp
3. Write the 8-output tile

---

## Known Issues and Fixes Needed

### Remaining Issues

### Issue 1: Runtime rollout is intentionally blocked

The tiled kernel is parity-correct, but the isolated benchmark shows it is
dramatically slower than the live FFN path:
- current path: `q8_1_group` + `SiluMulQuantizeQ8_1` + `q8_1_gemv`
- fused path: `FusedFfnQ4K(...)`
- result: no rollout

### Issue 2: No hot-path integration yet

`transformer_forward.cu` still uses the existing FFN path. That is deliberate
until a fused FFN design has evidence of a real throughput win.

**Fix Needed**: redesign around a different operator strategy, not this kernel

---

## Revised Kernel Design (Recommendation)

Given the complexity, I recommend a **simplified two-pass approach**:

### Keep the current runtime FFN path
- grouped `Q8_1` gate/up kernels
- fused `SiluMulQuantizeQ8_1`
- `Q8_1`/packed/MMQ down-proj dispatch

### Future fusion work must beat the current path in isolation first
- the current tiled `FusedFfnQ4K(...)` design does not
- do not reintroduce a runtime rollout flag until a replacement wins
**Drawback**: Still writes intermediate to global memory
**Estimated Improvement**: 2-4% (vs 3-8% for full fusion)

---

## Next Steps

### Immediate (Required for Completion)

1. **Keep runtime rollout off** ✅
   - current FFN hot path remains faster
   - no runtime flag is kept for the fused bring-up path

2. **Use the isolated benchmark as the entry gate** ✅
   - `benchmark_fused_ffn` is now the required proof step
   - no runtime integration without a measured win on the live geometry

3. **Redesign the fused operator** ⏳ REQUIRED
   - current tiled design is parity-correct but throughput-negative
   - next candidate must reduce compute duplication without collapsing occupancy

4. **Integrate only after a benchmark win** ⏳ REQUIRED
   - `transformer_forward.cu` remains unchanged until that happens

### Short-term (Week 1)

5. **Debug and Optimize** ⏳
   - Use cuda-memcheck to detect memory errors
   - Profile with Nsight Compute
   - Optimize memory access patterns
   - Tune thread block configuration

6. **Create Benchmark** ⏳
   - Isolated kernel performance test
   - Full-model throughput comparison
   - Measure actual improvement

### Medium-term (Week 2)

7. **Evaluate Results** ⏳
   - If ≥ 3% improvement: Default-enable
   - If < 3%: Keep as opt-in, document findings
   - Consider alternative optimizations

---

## Timeline Estimate

| Task | Estimate | Status |
|------|----------|--------|
| Analysis and profiling | 4 hours | ✅ Complete |
| Initial kernel implementation | 6 hours | ⏳ In progress (bugs remain) |
| Debug and fix kernel | 4-8 hours | ⏳ TODO |
| Dispatch integration | 2 hours | ⏳ TODO |
| Correctness test | 2 hours | ⏳ TODO |
| Performance benchmark | 2 hours | ⏳ TODO |
| Optimization and tuning | 4-8 hours | ⏳ TODO |
| **Total remaining** | **18-32 hours** | - |

**Estimated completion**: 1-2 weeks of focused work

---

## Risk Assessment

### High Risks

1. **Kernel Correctness**: Complex quantization logic, easy to introduce bugs
   - **Mitigation**: Extensive testing, start with small geometries

2. **Performance May Not Meet Target**: Actual improvement could be < 3%
   - **Mitigation**: Profile early, be ready to pivot to alternative optimizations

3. **Limited Geometry Support**: May not work for all model sizes
   - **Mitigation**: Fallback to current implementation, document constraints

### Medium Risks

1. **Code Complexity**: Harder to maintain and debug
   - **Mitigation**: Clean documentation, opt-in initially

2. **Compatibility**: May not work with all quantization types
   - **Mitigation**: Start with Q4_K only, expand later if successful

### Low Risks

1. **Regressions**: Opt-in flag minimizes impact
   - **Mitigation**: Already implemented

2. **Build Integration**: Straightforward addition
   - **Mitigation**: Following existing patterns

---

## Alternative Approaches

If full fusion proves too complex or doesn't meet targets:

### Option A: Two-Pass Fusion (Simpler)
- Compute gate+up+SiLU in one kernel
- Write intermediate to global memory
- Compute down_proj in separate kernel (current approach)
- **Benefit**: Still reduces 1 kernel launch
- **Expected**: 2-4% improvement
- **Complexity**: Low

### Option B: Kernel Fusion for Other Operations
- Fuse Q+K+V projections in attention
- Or fuse RmsNorm with other projections
- **Expected**: 3-5% improvement
- **Complexity**: Medium
- **Status**: Not investigated

### Option C: Concurrent Throughput Focus
- Fix 0.50x gap at concurrency=4
- May have higher business impact
- **Expected**: 0.50x → 0.70x (40% improvement)
- **Complexity**: Medium-High
- **Status**: Separate investigation needed

---

## Recommendations

### Immediate Actions

1. **Complete current implementation** ⏳
   - Fix kernel bugs (down_proj indexing, quantization)
   - Create correctness test
   - Validate with small geometries first

2. **If bugs prove difficult** ⏳
   - Implement two-pass fusion (simpler)
   - Or pivot to concurrent throughput optimization

3. **Decision gate** ⏳
   - After 1 week: Evaluate progress
   - If < 50% complete: Consider pivot
   - If ≥ 50% complete: Continue to completion

### Long-term Strategy

- **FFN fusion** is one of several optimization paths
- **Vectorized loads** failed (0.55% vs 5% target)
- **Kernel fusion** has higher complexity but proven benefits in other frameworks
- **Concurrent throughput** may have higher ROI

**Recommendation**: Complete FFN fusion proof-of-concept, then evaluate whether to:
- Continue with full FFN fusion (1-2 more weeks)
- Pivot to simpler fusion approaches
- Focus on concurrent throughput gap (0.50x → 0.70x)

---

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Analysis complete | ✅ | ✅ | Done |
| Kernel implemented | ✅ | ⏳ | Has bugs |
| Correctness test | Bit-exact | ⏳ | TODO |
| Micro-kernel speedup | ≥ 5% | ⏳ | TODO |
| Full-model improvement | ≥ 3% | ⏳ | TODO |
| Code quality | Clean, documented | ⏳ | Partial |

---

## Conclusion

FFN kernel fusion implementation is **in progress** with Phase 1 complete and Phase 2 partially complete. The kernel has been designed and implemented but has bugs that need fixing. The expected benefit (3-8% overall improvement) justifies continued work, but the complexity suggests we should evaluate progress after 1 week and consider alternative approaches if needed.

**Current Status**: ⏳ 30% complete (analysis done, kernel skeleton created, bugs remain)

**Estimated Time to Completion**: 1-2 weeks

**Go/No-Go Decision**: Evaluate after fixing bugs and running initial correctness tests

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
**Status**: Work in progress
