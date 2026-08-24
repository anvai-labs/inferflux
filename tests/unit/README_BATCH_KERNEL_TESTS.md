# Batch GEMV Kernel Test Harness

**Purpose**: Validate template-based batch kernel approach and measure performance improvement potential.

---

## Overview

This test harness validates whether the template-based batch processing approach (inspired by llama.cpp) provides performance benefits for inferflux_cuda.

### What It Tests

1. **Correctness**: Do batch kernels produce the same results as the baseline?
2. **Performance**: How much faster/slower are batch kernels?
3. **Scaling**: Does performance improve with larger batch sizes?

### Files

- `tests/unit/test_batch_gemv_kernel.cpp` - Full test suite with Google Test
- `tests/unit/benchmark_batch_gemv.cu` - Standalone benchmark (recommended)
- `scripts/build_and_test_batch_kernel.sh` - Build and run script
- `scripts/profile_batch_kernel.sh` - Nsight Systems profiling script

---

## Quick Start

### Option 1: Standalone Benchmark (Recommended)

```bash
# Build and run benchmark
./scripts/build_and_test_batch_kernel.sh
```

**Output**: Performance comparison for batch sizes 1, 2, 4, 8

### Option 2: Full Test Suite with Google Test

```bash
# Add to inferflux build
# Add to tests/unit/CMakeLists.txt:

# Add test file
add_executable(test_batch_gemv_kernel test_batch_gemv_kernel.cpp)
target_link_libraries(test_batch_gemv_kernel PRIVATE GTest::gtest_main)
cuda_add_executable(test_batch_gemv_kernel test_batch_gemv_kernel)

# Build and run
cmake --build build --target test_batch_gemv_kernel
./build/tests/unit/test_batch_gemv_kernel
```

### Option 3: Nsight Systems Profiling

```bash
# Profile with Nsight Systems
./scripts/profile_batch_kernel.sh

# View profiles
nsys-ui ./batch_kernel_profile_*/profile_baseline.nsys-rep
nsys-ui ./batch_kernel_profile_*/profile_batch2.nsys-rep
```

---

## Interpreting Results

### Expected Outcomes

#### If Batch Processing Works (GO signal)

**Performance targets**:
- BatchSize=2: >10% faster than baseline
- BatchSize=4: >20% faster than baseline
- BatchSize=8: >30% faster than baseline

**Nsight Systems metrics**:
- Reduced memory bandwidth per sequence
- Better GPU occupancy for larger batches
- Fewer kernel launches (if measuring end-to-end)

**Decision**: ✅ **GO** - Proceed with Phase 2 (architectural refactoring)

#### If Batch Processing Doesn't Work (NO-GO signal)

**Performance results**:
- All batch sizes: Same or slower than baseline
- Performance degrades with larger batch sizes
- Memory bandwidth contention increases

**Nsight Systems metrics**:
- No memory bandwidth improvement
- Lower GPU occupancy
- Higher kernel execution time

**Decision**: ❌ **NO-GO** - Pivot to incremental improvements (kernel fusion, memory coalescing)

---

## Troubleshooting

### Build Errors

**Error**: `fused_dequant_gemv.cuh: No such file`
```
Solution: Ensure you're running from the InferFlux root directory
```

**Error**: `nvcc: command not found`
```
Solution: Install CUDA Toolkit and add to PATH
```

**Error**: `undefined reference to GetGpuProfile`
```
Solution: This is expected for standalone build; remove any external dependencies
```

### Runtime Errors

**Error**: `CUDA error: invalid configuration`
```
Solution: Check GPU architecture compatibility
```

**Error**: `CUDA error: out of memory`
```
Solution: Reduce K or N dimensions in test configuration
```

### Wrong Results

**Issue**: Batch kernel output doesn't match baseline
```
Solution: Check kernel implementation for bugs, especially:
- Activation indexing (x[b*K*blockDim.y + row*K])
- Output indexing (output[b*N*blockDim.y + row*N + out_idx])
```

---

## Implementation Notes

### Kernel Differences

**Baseline kernel** (`fused_dequant_gemv_q4k_baseline`):
- Processes ONE sequence per kernel launch
- Uses `blockIdx.y` for row (sequence in batch)
- Simple accumulator: `float acc`

**Batch kernel** (`fused_dequant_gemv_q4k_batched<BatchSize>`):
- Processes BatchSize sequences in ONE kernel launch
- Template parameter for compile-time optimization
- Batch accumulator: `float batch_acc[BatchSize]`
- Loop unrolling for batch dimension

### Memory Layout

Both kernels use the same memory layout:
- `weight`: `[N][K]` quantized (N=output_dim, K=input_dim)
- `x`: `[M][K]` activations (M=batch_size, K=input_dim)
- `output`: `[M][N]` results (M=batch_size, N=output_dim)

The batch kernel processes multiple sequences (M) cooperatively within a single kernel invocation.

---

## Performance Analysis

### What to Look For

1. **Kernel execution time**:
   - Does batch kernel execute faster per sequence?
   - Is there speedup that increases with batch size?

2. **Memory bandwidth** (via Nsight Systems):
   - Bytes transferred per second
   - Is memory bandwidth better utilized for batches?

3. **GPU occupancy** (via Nsight Systems):
   - SM utilization percentage
   - Are more CUDA cores busy with batch processing?

4. **Kernel launch overhead**:
   - Time spent launching kernels
   - Reduced launch count for batch workloads?

### Success Criteria

**Minimum viable improvement**:
- 10% speedup at BatchSize=2
- 20% speedup at BatchSize=4

**Stretch goal**:
- 30% speedup at BatchSize=8
- Memory bandwidth improvement visible in Nsight Systems

---

## Go/No-Go Decision

After running tests, decide:

### GO: Proceed with Full Implementation

**If**:
- BatchSize=4 shows >20% improvement
- Memory bandwidth clearly better utilized
- No correctness issues

**Then**:
1. Proceed to Phase 2: Architectural refactoring
2. Implement dispatch logic for template variants
3. Integrate into FusedQuantGemm

**Expected outcome**: 1.5-2.0x scaling improvement (6-9 weeks)

### NO-GO: Pivot to Incremental Improvements

**If**:
- All batch sizes ≤10% improvement
- Performance degrades with batch size
- Memory bandwidth contention increases

**Then**:
1. Implement Option A: Kernel fusion (2-3 weeks, 20-30% benefit)
2. Implement Option B: Memory coalescing (1-2 weeks, 10-20% benefit)
3. Implement Option C: CUDA graph optimization (1 week, 5-10% benefit)

**Expected outcome**: 1.3-1.4x scaling improvement (4-6 weeks)

---

## Related Documentation

- `docs/SPRINT2_IMPLEMENTATION_PLAN.md` - Detailed implementation plan
- `docs/SPRINT2_SUMMARY.md` - Sprint 2 summary
- `docs/cuda_native_scaling_roadmap.md` - Overall improvement roadmap

---

## Contact

For questions or issues, refer to:
- Investigation summary: `docs/INVESTIGATION_COMPLETE_SUMMARY.md`
- Kernel analysis: `docs/PHASE3_KERNEL_ANALYSIS.md`
