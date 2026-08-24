# FFN Kernel Fusion Analysis

**Date**: March 10, 2026
**Task**: #12 - Implement full FFN kernel fusion (gate+up+SiLU+down)
**Status**: Analysis Phase

---

## Current FFN Architecture

### Execution Flow (Current)

The FFN (Feed-Forward Network) in LLaMA models uses SwiGLU activation:
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

### Current Implementation: 3-Stage Approach

**Stage 1: Gate+Up Projections** (FUSED ✅)
```cpp
// Single kernel launch for both gate and up
TryQ8_1ProjectionGroup(ffn_plans)  // or TryPackedProjectionGroup
// Outputs: d_ffn_gate_, d_ffn_up_ (written to global memory)
```

**Stage 2: SiLU Activation + Quantization** (FUSED ✅)
```cpp
// Single kernel for SiLU + Q8_1 quantization
FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream)
// Reads: d_ffn_gate_, d_ffn_up_ (from global memory)
// Writes: act_q8_1 (to global memory)
// Operation: SiLU(gate) * up, then quantize to int8
```

**Stage 3: Down Projection** (Separate ❌)
```cpp
// GEMV kernel for down projection
FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream)
// Reads: act_q8_1 (from global memory)
// Writes: d_ffn_down_ (to global memory)
```

### Memory Traffic Pattern

```
                    ┌─────────────────┐
                    │  Activation x   │
                    │  (Hidden dim)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Gate+Up GEMV   │  Stage 1: FUSED
                    │  (1 kernel)     │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     ┌──────▼──────┐                   ┌─────▼─────┐
     │ d_ffn_gate_ │                   │ d_ffn_up_ │
     │ (Global mem)│                   │(Global mem)│
     └──────┬──────┘                   └─────┬─────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                    ┌────────▼────────┐
                    │ SiLU * Quantize │  Stage 2: FUSED
                    │  (1 kernel)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   act_q8_1      │
                    │  (Global mem)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Down GEMV      │  Stage 3: SEPARATE
                    │  (1 kernel)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  d_ffn_down_    │
                    │  (Global mem)   │
                    └─────────────────┘
```

**Total kernel launches**: 3 (actually 2 kernel launches + grouped GEMV)

**Global memory writes**: 3 intermediate buffers

---

## Full Fusion Opportunity

### Proposed: Single-Kernel FFN

Create a kernel that does ALL operations in one launch:

```cpp
FusedFFNGemm<block_q4_k>(
    gate_weight, up_weight, down_weight,  // 3 weight matrices
    activation,                            // input (hidden dim)
    output,                               // output (hidden dim)
    N_intermediate,                        // FFN expansion ratio
    N_hidden,                             // hidden size
    M, K,                                 // batch, input dims
    stream
);
```

### Execution Flow (Fused)

```
                    ┌─────────────────┐
                    │  Activation x   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Gate+Up+SiLU   │  Stage 1+2: FUSED
                    │  +Down GEMV     │  Stage 3: FUSED
                    │  (1 kernel)     │  ALL IN ONE
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  d_ffn_down_    │
                    │  (Global mem)   │
                    └─────────────────┘
```

**Total kernel launches**: 1

**Global memory writes**: 1 (final output only)

**Benefits**:
1. **Reduce kernel launches**: 3 → 1 (2 fewer launches)
2. **Eliminate intermediate writes**: No `d_ffn_gate_`, `d_ffn_up_`, `act_q8_1` buffers
3. **Better cache utilization**: Activation loaded once, used by all 3 projections
4. **Fused SiLU**: No intermediate kernel synchronization

---

## Implementation Complexity

### Challenges

1. **Triple weight access**: Need to load from 3 different weight matrices simultaneously
   - gate_weight: [hidden_size × intermediate_size]
   - up_weight: [hidden_size × intermediate_size]
   - down_weight: [intermediate_size × hidden_size]

2. **Different quantization types**: Gate, up, and down may have different quantization
   - Q4_K, Q6_K, Q8_0, etc.
   - Need template specialization or runtime dispatch

3. **Memory layout**: Weights are in GGUF format with specific quantization blocks
   - block_q4_k, block_q6_k, etc.
   - Need efficient dequantization within kernel

4. **Activation reuse**: Gate and up both use same input, down uses activated result
   - Need shared memory for activation
   - Need register-level fusion for SiLU

### Design Options

#### Option A: Template-Based Specialization

```cpp
template <typename GateBlockType, typename UpBlockType, typename DownBlockType>
__global__ void FusedFFNGemm(
    const GateBlockType *gate_weight,
    const UpBlockType *up_weight,
    const DownBlockType *down_weight,
    const half *activation,
    half *output,
    int N_intermediate, int N_hidden, int M, int K
);
```

**Pros**:
- Type-safe, efficient
- Compiler can optimize specific quantization combinations

**Cons**:
- Combinatorial explosion of template instantiations
- Large code size

#### Option B: Runtime Dispatch with Variant Weights

```cpp
__global__ void FusedFFNGemmQ4K_Q4K_Q4K(...);  // All Q4_K
__global__ void FusedFFNGemmQ4K_Q4K_Q6K(...);  // Mixed
// ... many variants
```

**Pros**:
- Specialized kernels for each combination
- Optimal performance

**Cons**:
- Many kernel variants to maintain
- Complex dispatch logic

#### Option C: Two-Stage Fusion (Incremental) ✅ RECOMMENDED

**Stage 1**: Fuse SiLU into down_proj (already exists)
- This is already done: `TryQ8_1SiluMulGemv`
- Gate+up computed separately, down reads activated result

**Stage 2**: Fuse gate+up+SiLU into down_proj
- Single kernel that does gate_proj + up_proj + SiLU internally
- Then immediately does down_proj
- All in registers/shared memory

**Pros**:
- Incremental approach, easier to validate
- Builds on existing fusion patterns
- Lower risk

**Cons**:
- Still 2 kernel launches (gate+up fused, then down with SiLU)
- Less benefit than full fusion

### Recommended Approach: Option C with Full Fusion Target

Start with **incremental fusion** (Option C), but design for **full fusion**:

**Phase 1** (1 week): Validate current fusion effectiveness
- Profile to confirm where time is spent
- Measure intermediate memory traffic
- Identify bottlenecks

**Phase 2** (1-2 weeks): Implement full fusion for common case
- Focus on Q4_K (most common quantization)
- Single kernel: gate+up+SiLU+down
- Template for block type

**Phase 3** (1 week): Benchmark and optimize
- Compare to baseline
- Optimize memory access patterns
- Tune shared memory usage

---

## Expected Performance Improvement

### Theoretical Analysis

**Current breakdown** (rough estimates for Qwen2.5-3B Q4_K_M):
- Gate+Up GEMV: ~30% of FFN time
- SiLU + Quantize: ~10% of FFN time
- Down GEMV: ~60% of FFN time

**FFN as fraction of total**: ~25-35%

**Potential savings**:
- Kernel launch overhead: ~2-5 μs per launch × 2 = 4-10 μs
- Memory writes eliminated: 3 × 2MB = 6MB per layer
- Cache efficiency: 5-10% from activation reuse

**Expected improvement**: 3-5% overall (at the low end of estimates)

### Comparison to Vectorized Loads

| Optimization | Expected | Actual | Outcome |
|--------------|----------|--------|---------|
| Vectorized scales | 5-10% | 0.55% | ❌ Failed |
| FFN fusion | 3-5% | TBD | ⏳ Pending |

**Key difference**:
- Vectorized loads: Memory-level optimization (memcpy overhead killed it)
- FFN fusion: Kernel-level optimization (fewer launches, less memory traffic)

---

## Implementation Plan

### Phase 1: Research and Design (Week 1)

1. **Profile current implementation**
   - Use Nsight Compute to measure time breakdown
   - Identify actual bottlenecks
   - Measure memory traffic patterns

2. **Study existing fusion kernels**
   - Analyze `TryQ8_1ProjectionGroup` implementation
   - Understand `SiluMulQuantizeQ8_1` kernel
   - Learn from llama.cpp MMQ approach

3. **Design fused kernel interface**
   - Define kernel signature
   - Plan shared memory layout
   - Design thread block organization

### Phase 2: Implementation (Week 1-2)

1. **Create fused kernel file**
   - `runtime/backends/cuda/native/kernels/fused_ffn_gemm.cuh`

2. **Implement Q4_K variant** (most common)
   - Template for `block_q4_k`
   - Fused gate+up+SiLU+down in one kernel
   - Optimize for decode (M=1) and prefill (M>1)

3. **Add dispatch logic**
   - Update `FusedQuantGemm` class
   - Add opt-in flag: `INFERFLUX_USE_FUSED_FFN=1`
   - Fall back to current implementation if geometry doesn't fit

4. **Create correctness test**
   - `tests/unit/test_fused_ffn.cu`
   - Bit-exact comparison with baseline
   - Test various geometries (M=1, M=2, M=4)

### Phase 3: Benchmark and Optimize (Week 2)

1. **Performance benchmark**
   - Micro-benchmark: Isolated FFN kernel
   - Full-model: TinyLlama, Qwen2.5-3B
   - Measure tok/s improvement

2. **Optimize if needed**
   - Tune shared memory usage
   - Adjust thread block size
   - Optimize memory access patterns

3. **Decision point**
   - If ≥ 3% improvement: Default-enable
   - If < 3%: Keep as opt-in, document findings

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Correctness | Bit-exact match (tolerance 1e-3) | Unit test |
| Micro-kernel speedup | ≥ 5% | Nsight Compute |
| Full-model improvement | ≥ 3% | Throughput gate |
| Code quality | Clean, documented | Code review |
| Zero regressions | No bugs | Integration tests |

---

## Risk Assessment

### High Risks

1. **Complexity**: Kernel fusion is complex, easy to introduce bugs
   - **Mitigation**: Incremental approach, extensive testing

2. **Performance**: May not achieve target improvement
   - **Mitigation**: Profile first, validate assumptions

3. **Maintenance**: More complex code is harder to maintain
   - **Mitigation**: Clear documentation, opt-in initially

### Medium Risks

1. **Compatibility**: May not work for all quantization types
   - **Mitigation**: Fallback to current implementation

2. **Memory**: Shared memory constraints for large models
   - **Mitigation**: Geometry checks, fallback for large cases

### Low Risks

1. **Correctness**: Bit-exact comparison possible
   - **Mitigation**: Already planned

2. **Integration**: Opt-in flag minimizes impact
   - **Mitigation**: Runtime policy check

---

## Next Steps

1. ✅ **Complete analysis** (this document)
2. ⏳ **Profile current implementation** with Nsight Compute
3. ⏳ **Design fused kernel** interface and memory layout
4. ⏳ **Implement Q4_K variant** as proof-of-concept
5. ⏳ **Benchmark and evaluate** against baseline

**Decision gate**: After profiling and initial implementation, decide whether to proceed based on:
- Can we achieve ≥ 3% improvement?
- Is complexity manageable?
- Are there better optimization opportunities?

---

## References

- Current implementation: `runtime/backends/cuda/native/transformer_forward.cu` (lines 1330-1520)
- Dispatch logic: `runtime/backends/cuda/native/fused_quant_gemm.cu` (lines 1180-1256)
- Existing fused kernels: `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`
- llama.cpp MMQ reference: `external/llama.ggml/`

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
**Status**: Ready for implementation phase
