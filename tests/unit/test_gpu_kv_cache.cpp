#include <catch2/catch_amalgamated.hpp>
#include <vector>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#endif

namespace inferflux {

// GPU KV Cache tests require CUDA hardware.
// Tagged [gpu_kv_cache][cuda] so they can be skipped on CI without GPUs.

TEST_CASE("KvCacheGpu: interface compiles", "[gpu_kv_cache]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Verify the class interface compiles correctly
  KvCacheGpu cache;
  REQUIRE(cache.GetMemoryUsage() == 0);
  REQUIRE(cache.MaxSeqLen() == 0);
  REQUIRE(cache.MaxBatchSize() == 0);
#else
  REQUIRE(true); // Placeholder
#endif
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("KvCacheGpu: Allocate and access", "[gpu_kv_cache][cuda]") {
  // This test requires a CUDA GPU
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  KvCacheGpu cache;
  int num_layers = 2;
  int num_kv_heads = 4;
  int head_dim = 64;
  int max_seq = 128;
  int max_batch = 4;

  REQUIRE(
      cache.Allocate(num_layers, num_kv_heads, head_dim, max_seq, max_batch));
  REQUIRE(cache.GetMemoryUsage() > 0);
  REQUIRE(cache.MaxSeqLen() == max_seq);
  REQUIRE(cache.MaxBatchSize() == max_batch);

  // Verify K/V pointers are non-null and distinct
  half *k0 = cache.GetK(0, 0);
  half *v0 = cache.GetV(0, 0);
  REQUIRE(k0 != nullptr);
  REQUIRE(v0 != nullptr);
  REQUIRE(k0 != v0);

  // Different layers should have different pointers
  half *k1 = cache.GetK(1, 0);
  REQUIRE(k1 != k0);

  // Different batch slots should have different pointers
  half *k0_b1 = cache.GetK(0, 1);
  REQUIRE(k0_b1 != k0);

  // Clear should not crash
  cache.ClearSequence(0);
}

TEST_CASE("KvCacheGpu: copy prefix + serialize hydrate",
          "[gpu_kv_cache][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  KvCacheGpu cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, /*num_kv_heads=*/1, /*head_dim=*/8,
                         /*max_seq=*/8, /*max_batch=*/2));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  // Write deterministic source prefix values for K/V.
  constexpr int kPrefixTokens = 4;
  constexpr int kKvDim = 8;
  std::vector<half> k_src(static_cast<size_t>(kPrefixTokens * kKvDim));
  std::vector<half> v_src(static_cast<size_t>(kPrefixTokens * kKvDim));
  for (size_t i = 0; i < k_src.size(); ++i) {
    k_src[i] = __float2half(static_cast<float>(i + 1));
    v_src[i] = __float2half(static_cast<float>((i + 1) * 2));
  }

  REQUIRE(cudaMemcpyAsync(cache.GetK(0, 0), k_src.data(),
                          k_src.size() * sizeof(half), cudaMemcpyHostToDevice,
                          stream) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(cache.GetV(0, 0), v_src.data(),
                          v_src.size() * sizeof(half), cudaMemcpyHostToDevice,
                          stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  REQUIRE(cache.CopySequencePrefix(/*src_seq=*/0, /*dst_seq=*/1, kPrefixTokens,
                                   stream));

  std::vector<half> k_dst(k_src.size());
  std::vector<half> v_dst(v_src.size());
  REQUIRE(cudaMemcpy(k_dst.data(), cache.GetK(0, 1),
                     k_dst.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(v_dst.data(), cache.GetV(0, 1),
                     v_dst.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < k_src.size(); ++i) {
    REQUIRE(__half2float(k_dst[i]) == Catch::Approx(__half2float(k_src[i])));
    REQUIRE(__half2float(v_dst[i]) == Catch::Approx(__half2float(v_src[i])));
  }

  std::vector<uint8_t> blob;
  REQUIRE(cache.SerializeSequence(/*seq_id=*/0, &blob));
  REQUIRE_FALSE(blob.empty());

  cache.ClearSequence(/*seq_id=*/1);
  REQUIRE(cache.HydrateSequence(/*seq_id=*/1, blob, stream));

  std::vector<half> k_hydrated(k_src.size());
  std::vector<half> v_hydrated(v_src.size());
  REQUIRE(cudaMemcpy(k_hydrated.data(), cache.GetK(0, 1),
                     k_hydrated.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(v_hydrated.data(), cache.GetV(0, 1),
                     v_hydrated.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < k_src.size(); ++i) {
    REQUIRE(__half2float(k_hydrated[i]) ==
            Catch::Approx(__half2float(k_src[i])));
    REQUIRE(__half2float(v_hydrated[i]) ==
            Catch::Approx(__half2float(v_src[i])));
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("KvCacheGpu: out-of-bounds seq_id is safely rejected",
          "[gpu_kv_cache][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  KvCacheGpu cache;
  int max_batch = 4;
  REQUIRE(cache.Allocate(/*num_layers=*/1, /*num_kv_heads=*/2, /*head_dim=*/8,
                         /*max_seq=*/16, max_batch));

  // In-bounds: should succeed
  cache.ClearSequence(0);
  cache.ClearSequence(3);

  // Out-of-bounds: ClearSequence should not crash (it has a bounds check)
  cache.ClearSequence(4);   // seq_id == max_batch
  cache.ClearSequence(100); // far out of range

  // Verify MaxBatchSize is correctly reported
  REQUIRE(cache.MaxBatchSize() == max_batch);
}

TEST_CASE("KvCacheGpu: seq slot IDs must fit within max_batch",
          "[gpu_kv_cache][cuda]") {
  // This test validates the invariant that the scheduler's sequence slot IDs
  // (0..kMaxSequenceSlots-1) must all fit within the KV cache's max_batch.
  // The scheduler uses kMaxSequenceSlots=128, so KV cache needs ≥128 slots.
  // We test with a smaller value here to keep GPU memory usage low in CI.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kSchedulerMaxSlots = 16; // subset of scheduler's kMaxSequenceSlots

  // Allocate with enough slots for the scheduler
  KvCacheGpu cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, /*num_kv_heads=*/1, /*head_dim=*/8,
                         /*max_seq=*/8, /*max_batch=*/kSchedulerMaxSlots));

  // All scheduler slot IDs should produce valid, distinct pointers
  std::vector<half *> k_ptrs(kSchedulerMaxSlots);
  std::vector<half *> v_ptrs(kSchedulerMaxSlots);
  for (int i = 0; i < kSchedulerMaxSlots; ++i) {
    k_ptrs[i] = cache.GetK(0, i);
    v_ptrs[i] = cache.GetV(0, i);
    REQUIRE(k_ptrs[i] != nullptr);
    REQUIRE(v_ptrs[i] != nullptr);
  }

  // Each slot should have distinct K and V pointers
  for (int i = 0; i < kSchedulerMaxSlots; ++i) {
    for (int j = i + 1; j < kSchedulerMaxSlots; ++j) {
      REQUIRE(k_ptrs[i] != k_ptrs[j]);
      REQUIRE(v_ptrs[i] != v_ptrs[j]);
    }
  }

  // Write distinct values to each slot's K cache and read them back
  // to verify no overlap between slots
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  constexpr int kKvDim = 8; // num_kv_heads * head_dim
  for (int i = 0; i < kSchedulerMaxSlots; ++i) {
    std::vector<half> data(kKvDim);
    for (int j = 0; j < kKvDim; ++j) {
      data[j] = __float2half(static_cast<float>(i * 100 + j));
    }
    REQUIRE(cudaMemcpyAsync(cache.GetK(0, i), data.data(),
                            kKvDim * sizeof(half), cudaMemcpyHostToDevice,
                            stream) == cudaSuccess);
  }
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  // Read back and verify each slot has its own data (no corruption)
  for (int i = 0; i < kSchedulerMaxSlots; ++i) {
    std::vector<half> readback(kKvDim);
    REQUIRE(cudaMemcpy(readback.data(), cache.GetK(0, i),
                       kKvDim * sizeof(half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int j = 0; j < kKvDim; ++j) {
      float expected = static_cast<float>(i * 100 + j);
      REQUIRE(__half2float(readback[j]) == Catch::Approx(expected));
    }
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("KvCacheGpu: undersized max_batch causes slot collision",
          "[gpu_kv_cache][cuda]") {
  // Demonstrates the failure mode that caused the correctness regression:
  // if KV cache max_batch < scheduler slots, higher slot IDs alias lower ones
  // or access out-of-bounds GPU memory.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kSmallBatch = 4;
  KvCacheGpu cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, /*num_kv_heads=*/1, /*head_dim=*/8,
                         /*max_seq=*/8, /*max_batch=*/kSmallBatch));

  // Slots 0..3 are valid
  half *k0 = cache.GetK(0, 0);
  half *k3 = cache.GetK(0, 3);
  REQUIRE(k0 != nullptr);
  REQUIRE(k3 != nullptr);
  REQUIRE(k0 != k3);

  // Slot 4 would be out-of-bounds. The pointer arithmetic still "works" but
  // points past the allocated buffer. This is the dangerous case that the
  // executor's max_batch clamp (≥16) prevents.
  half *k4 = cache.GetK(0, 4);
  // k4 is computed as buffer_ + 4*slot_stride_, which is past the allocation
  // (buffer_ only has 4*slot_stride_ elements total). This verifies the
  // pointer would be past the last valid slot.
  REQUIRE(k4 > k3); // Pointer math works, but access would be UB

  // Verify MaxBatchSize correctly reports the limit
  REQUIRE(cache.MaxBatchSize() == kSmallBatch);
}
#endif

} // namespace inferflux
