#include <catch2/catch_amalgamated.hpp>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace inferflux {
namespace {

// Helper: allocate device memory, copy from host
template <typename T> T *ToDevice(const std::vector<T> &host) {
  T *d = nullptr;
  REQUIRE(cudaMalloc(&d, host.size() * sizeof(T)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return d;
}

template <typename T> T *AllocDevice(size_t count) {
  T *d = nullptr;
  REQUIRE(cudaMalloc(&d, count * sizeof(T)) == cudaSuccess);
  REQUIRE(cudaMemset(d, 0, count * sizeof(T)) == cudaSuccess);
  return d;
}

template <typename T>
std::vector<T> FromDevice(const T *d, size_t count) {
  std::vector<T> host(count);
  REQUIRE(cudaMemcpy(host.data(), d, count * sizeof(T),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

std::vector<half> MakeRandom(size_t count, float scale = 0.1f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    float v = scale * std::sin(0.173f * static_cast<float>(i) + 0.031f);
    out[i] = __float2half(v);
  }
  return out;
}

} // namespace

// ============================================================================
// BatchedRoPE: isolated test with synthetic data
// ============================================================================

TEST_CASE("BatchedRoPE produces non-zero output for B>1",
          "[batched_decode][cuda]") {
  constexpr int B = 4;
  constexpr int num_heads = 4;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr float freq_base = 10000.0f;
  constexpr int rope_type = 0; // kNorm

  const int q_size = B * num_heads * head_dim;
  const int k_size = B * num_kv_heads * head_dim;

  auto h_q = MakeRandom(q_size, 0.5f);
  auto h_k = MakeRandom(k_size, 0.3f);
  std::vector<int> h_n_past = {0, 5, 10, 20};

  half *d_q = ToDevice(h_q);
  half *d_k = ToDevice(h_k);
  int *d_n_past = ToDevice(h_n_past);

  cudaError_t err = cuda_kernel::BatchedRoPE<half>(
      d_q, d_k, B, num_heads, num_kv_heads, head_dim, d_n_past, freq_base,
      nullptr, rope_type);
  REQUIRE(err == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Verify output is different from input (RoPE applies rotation)
  auto out_q = FromDevice(d_q, q_size);
  auto out_k = FromDevice(d_k, k_size);
  int q_changed = 0, k_changed = 0;
  for (int i = 0; i < q_size; ++i) {
    if (__half2float(out_q[i]) != __half2float(h_q[i]))
      ++q_changed;
  }
  for (int i = 0; i < k_size; ++i) {
    if (__half2float(out_k[i]) != __half2float(h_k[i]))
      ++k_changed;
  }
  // At least some values should change (except pos=0, pair_idx=0 where angle=0)
  CHECK(q_changed > q_size / 2);
  CHECK(k_changed > k_size / 2);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_n_past);
}

TEST_CASE("BatchedRoPE matches sequential RoPE per-sequence",
          "[batched_decode][cuda]") {
  constexpr int B = 3;
  constexpr int num_heads = 4;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr float freq_base = 10000.0f;
  constexpr int rope_type = 0;

  const int q_per_seq = num_heads * head_dim;
  const int k_per_seq = num_kv_heads * head_dim;
  std::vector<int> h_n_past = {3, 7, 15};

  // Create identical input for batched and sequential paths
  auto h_q = MakeRandom(B * q_per_seq, 0.5f);
  auto h_k = MakeRandom(B * k_per_seq, 0.3f);

  // --- Batched path ---
  half *d_q_batch = ToDevice(h_q);
  half *d_k_batch = ToDevice(h_k);
  int *d_n_past = ToDevice(h_n_past);

  REQUIRE(cuda_kernel::BatchedRoPE<half>(d_q_batch, d_k_batch, B, num_heads,
                                          num_kv_heads, head_dim, d_n_past,
                                          freq_base, nullptr,
                                          rope_type) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  auto batch_q = FromDevice(d_q_batch, B * q_per_seq);
  auto batch_k = FromDevice(d_k_batch, B * k_per_seq);

  // --- Sequential path (per-sequence RoPE) ---
  for (int b = 0; b < B; ++b) {
    half *d_q_seq = ToDevice(
        std::vector<half>(h_q.begin() + b * q_per_seq,
                          h_q.begin() + (b + 1) * q_per_seq));
    half *d_k_seq = ToDevice(
        std::vector<half>(h_k.begin() + b * k_per_seq,
                          h_k.begin() + (b + 1) * k_per_seq));

    REQUIRE(cuda_kernel::RoPE<half>(d_q_seq, d_k_seq, 1, num_heads,
                                     num_kv_heads, head_dim, h_n_past[b],
                                     freq_base, nullptr,
                                     rope_type) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    auto seq_q = FromDevice(d_q_seq, q_per_seq);
    auto seq_k = FromDevice(d_k_seq, k_per_seq);

    // Compare batched vs sequential
    for (int i = 0; i < q_per_seq; ++i) {
      float bv = __half2float(batch_q[b * q_per_seq + i]);
      float sv = __half2float(seq_q[i]);
      CHECK(std::abs(bv - sv) < 1e-3f);
    }
    for (int i = 0; i < k_per_seq; ++i) {
      float bv = __half2float(batch_k[b * k_per_seq + i]);
      float sv = __half2float(seq_k[i]);
      CHECK(std::abs(bv - sv) < 1e-3f);
    }

    cudaFree(d_q_seq);
    cudaFree(d_k_seq);
  }

  cudaFree(d_q_batch);
  cudaFree(d_k_batch);
  cudaFree(d_n_past);
}

// ============================================================================
// BatchedKvAppend: isolated test with synthetic data
// ============================================================================

TEST_CASE("BatchedKvAppend scatters to correct destinations",
          "[batched_decode][cuda]") {
  constexpr int B = 4;
  constexpr int kv_dim = 128; // num_kv_heads * head_dim

  // Source: [B, kv_dim] contiguous
  auto h_k = MakeRandom(B * kv_dim, 0.5f);
  auto h_v = MakeRandom(B * kv_dim, 0.3f);

  half *d_k = ToDevice(h_k);
  half *d_v = ToDevice(h_v);

  // Allocate B separate destination buffers (simulating KV cache slots)
  std::vector<half *> h_k_dst_ptrs(B), h_v_dst_ptrs(B);
  for (int b = 0; b < B; ++b) {
    h_k_dst_ptrs[b] = AllocDevice<half>(kv_dim);
    h_v_dst_ptrs[b] = AllocDevice<half>(kv_dim);
  }

  // Upload pointer arrays to device
  half **d_k_dst = nullptr;
  half **d_v_dst = nullptr;
  REQUIRE(cudaMalloc(&d_k_dst, B * sizeof(half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_v_dst, B * sizeof(half *)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_k_dst, h_k_dst_ptrs.data(), B * sizeof(half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v_dst, h_v_dst_ptrs.data(), B * sizeof(half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  cudaError_t err = cuda_kernel::BatchedKvAppend<half>(d_k, d_v, d_k_dst,
                                                        d_v_dst, B, kv_dim,
                                                        nullptr);
  REQUIRE(err == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Verify each destination buffer has the correct data
  for (int b = 0; b < B; ++b) {
    auto k_out = FromDevice(h_k_dst_ptrs[b], kv_dim);
    auto v_out = FromDevice(h_v_dst_ptrs[b], kv_dim);
    for (int d = 0; d < kv_dim; ++d) {
      CHECK(__half2float(k_out[d]) ==
            Catch::Approx(__half2float(h_k[b * kv_dim + d])).margin(1e-4f));
      CHECK(__half2float(v_out[d]) ==
            Catch::Approx(__half2float(h_v[b * kv_dim + d])).margin(1e-4f));
    }
  }

  for (int b = 0; b < B; ++b) {
    cudaFree(h_k_dst_ptrs[b]);
    cudaFree(h_v_dst_ptrs[b]);
  }
  cudaFree(d_k_dst);
  cudaFree(d_v_dst);
  cudaFree(d_k);
  cudaFree(d_v);
}

TEST_CASE("BatchedKvAppendStrided matches KV cache layout addressing",
          "[batched_decode][cuda]") {
  constexpr int num_layers = 2;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr int max_seq_len = 128;
  constexpr int max_batch = 4;
  constexpr int B = 4;
  constexpr int layer = 1;
  constexpr int kv_dim = num_kv_heads * head_dim;

  KvCacheGpuTyped<half> cache;
  REQUIRE(cache.Allocate(num_layers, num_kv_heads, head_dim, max_seq_len,
                         max_batch));

  std::vector<int> h_seq_ids = {0, 3, 1, 2};
  std::vector<int> h_n_past = {0, 5, 7, 1};
  auto h_k = MakeRandom(B * kv_dim, 0.45f);
  auto h_v = MakeRandom(B * kv_dim, 0.25f);

  half *d_k = ToDevice(h_k);
  half *d_v = ToDevice(h_v);
  int *d_seq_ids = ToDevice(h_seq_ids);
  int *d_n_past = ToDevice(h_n_past);

  cudaError_t err = cuda_kernel::BatchedKvAppendStrided<half>(
      d_k, d_v, cache.Buffer(), d_seq_ids, d_n_past, layer, B, cache.KvDim(),
      cache.SlotStride(), cache.LayerStride(), cache.KvStride(), nullptr);
  REQUIRE(err == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<half *> h_k_ptrs(B), h_v_ptrs(B);
  cache.GetBatchAppendPtrs(layer, h_seq_ids.data(), h_n_past.data(), B,
                           h_k_ptrs.data(), h_v_ptrs.data());
  for (int b = 0; b < B; ++b) {
    auto k_out = FromDevice(h_k_ptrs[b], kv_dim);
    auto v_out = FromDevice(h_v_ptrs[b], kv_dim);
    for (int d = 0; d < kv_dim; ++d) {
      CHECK(__half2float(k_out[d]) ==
            Catch::Approx(__half2float(h_k[b * kv_dim + d])).margin(1e-4f));
      CHECK(__half2float(v_out[d]) ==
            Catch::Approx(__half2float(h_v[b * kv_dim + d])).margin(1e-4f));
    }
  }

  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_seq_ids);
  cudaFree(d_n_past);
}

// ============================================================================
// FlashDecodeMultiSeq: isolated test with synthetic data
// ============================================================================

TEST_CASE("FlashDecodeMultiSeq produces valid attention output",
          "[batched_decode][cuda]") {
  constexpr int B = 4;
  constexpr int num_heads = 4;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Variable KV lengths per sequence
  std::vector<int> h_kv_lens = {10, 5, 20, 1};

  const int q_total = B * num_heads * head_dim;
  auto h_q = MakeRandom(q_total, 0.1f);

  // Create KV cache buffers per sequence: [kv_len, num_kv_heads * head_dim]
  const int kv_stride = num_kv_heads * head_dim;
  std::vector<half *> h_k_ptrs(B), h_v_ptrs(B);
  for (int b = 0; b < B; ++b) {
    auto k_data = MakeRandom(h_kv_lens[b] * kv_stride, 0.1f);
    auto v_data = MakeRandom(h_kv_lens[b] * kv_stride, 0.1f);
    h_k_ptrs[b] = ToDevice(k_data);
    h_v_ptrs[b] = ToDevice(v_data);
  }

  // Upload pointer arrays and kv_lens to device
  const half **d_k_ptrs = nullptr;
  const half **d_v_ptrs = nullptr;
  int *d_kv_lens = nullptr;
  REQUIRE(cudaMalloc(&d_k_ptrs, B * sizeof(half *)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_v_ptrs, B * sizeof(half *)) == cudaSuccess);
  REQUIRE(cudaMemcpy(const_cast<half **>(d_k_ptrs), h_k_ptrs.data(),
                     B * sizeof(half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(const_cast<half **>(d_v_ptrs), h_v_ptrs.data(),
                     B * sizeof(half *),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  d_kv_lens = ToDevice(h_kv_lens);

  half *d_q = ToDevice(h_q);
  half *d_o = AllocDevice<half>(q_total);

  cudaError_t err = cuda_kernel::FlashDecodeMultiSeq<half>(
      d_q, d_k_ptrs, d_v_ptrs, d_o, d_kv_lens, B, num_heads, num_kv_heads,
      head_dim, scale, nullptr);
  REQUIRE(err == cudaSuccess);

  // Check for CUDA errors (illegal memory access etc.)
  err = cudaDeviceSynchronize();
  INFO("FlashDecodeMultiSeq sync error: " << cudaGetErrorString(err));
  REQUIRE(err == cudaSuccess);

  // Verify output is not all zeros (attention should produce non-zero output)
  auto h_o = FromDevice(d_o, q_total);
  int nonzero = 0;
  for (int i = 0; i < q_total; ++i) {
    if (__half2float(h_o[i]) != 0.0f)
      ++nonzero;
  }
  CHECK(nonzero > q_total / 2);

  // Verify output doesn't contain NaN/Inf
  for (int i = 0; i < q_total; ++i) {
    float v = __half2float(h_o[i]);
    CHECK(std::isfinite(v));
  }

  for (int b = 0; b < B; ++b) {
    cudaFree(h_k_ptrs[b]);
    cudaFree(h_v_ptrs[b]);
  }
  cudaFree(const_cast<half **>(d_k_ptrs));
  cudaFree(const_cast<half **>(d_v_ptrs));
  cudaFree(d_kv_lens);
  cudaFree(d_q);
  cudaFree(d_o);
}

// ============================================================================
// KvCacheGpuTyped: test GetBatchAppendPtrs/GetBatchKVPtrs pointer arithmetic
// ============================================================================

TEST_CASE("KvCache batch pointer computation is consistent with Append",
          "[batched_decode][cuda]") {
  constexpr int num_layers = 2;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr int max_seq_len = 128;
  constexpr int max_batch = 4;
  constexpr int kv_dim = num_kv_heads * head_dim;

  KvCacheGpuTyped<half> cache;
  REQUIRE(cache.Allocate(num_layers, num_kv_heads, head_dim, max_seq_len,
                          max_batch));

  // Write known data at specific positions using Append
  auto k_data = MakeRandom(kv_dim, 1.0f);
  auto v_data = MakeRandom(kv_dim, 2.0f);

  constexpr int layer = 0;
  constexpr int seq_id = 2;
  constexpr int pos = 5;

  half *d_k = ToDevice(k_data);
  half *d_v = ToDevice(v_data);
  REQUIRE(cache.Append(layer, seq_id, pos, 1, d_k, d_v, nullptr) ==
          cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // GetBatchAppendPtrs should return pointer to position 5
  std::vector<int> seq_ids = {seq_id};
  std::vector<int> n_past = {pos};
  half *h_k_ptr = nullptr, *h_v_ptr = nullptr;
  cache.GetBatchAppendPtrs(layer, seq_ids.data(), n_past.data(), 1, &h_k_ptr,
                            &h_v_ptr);

  // The append pointer should point exactly where we wrote data
  auto readback = FromDevice(h_k_ptr, kv_dim);
  for (int i = 0; i < kv_dim; ++i) {
    CHECK(__half2float(readback[i]) ==
          Catch::Approx(__half2float(k_data[i])).margin(1e-4f));
  }

  // GetBatchKVPtrs should return the base pointer for the sequence
  const half *h_k_base = nullptr, *h_v_base = nullptr;
  cache.GetBatchKVPtrs(layer, seq_ids.data(), 1, &h_k_base, &h_v_base);

  // Base + pos * kv_dim should equal the append pointer
  CHECK(h_k_ptr == (half *)(h_k_base) + pos * kv_dim);

  cudaFree(d_k);
  cudaFree(d_v);
}

// ============================================================================
// Full batched decode pipeline: Embed → RoPE → KvAppend → FlashDecode
// Tests the full BatchForward kernel sequence with a fake model.
// ============================================================================

TEST_CASE("Batched decode pipeline runs without CUDA errors",
          "[batched_decode][cuda]") {
  constexpr int B = 4;
  constexpr int num_heads = 4;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr int hidden_size = num_heads * head_dim; // 256
  constexpr int kv_dim = num_kv_heads * head_dim;   // 128
  constexpr int max_seq_len = 128;
  constexpr float freq_base = 10000.0f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Allocate KV cache
  KvCacheGpuTyped<half> cache;
  REQUIRE(cache.Allocate(1, num_kv_heads, head_dim, max_seq_len, B));

  // Simulate different past lengths
  std::vector<int> h_n_past = {0, 3, 7, 15};
  std::vector<int> h_seq_ids = {0, 1, 2, 3};

  // Pre-populate KV cache with some data (simulate previous tokens)
  auto kv_fill = MakeRandom(kv_dim, 0.1f);
  half *d_kv_fill = ToDevice(kv_fill);
  for (int b = 0; b < B; ++b) {
    for (int pos = 0; pos < h_n_past[b]; ++pos) {
      REQUIRE(cache.Append(0, h_seq_ids[b], pos, 1, d_kv_fill, d_kv_fill,
                            nullptr) == cudaSuccess);
    }
  }
  cudaFree(d_kv_fill);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Allocate working buffers
  half *d_q = AllocDevice<half>(B * num_heads * head_dim);
  half *d_k_new = AllocDevice<half>(B * kv_dim);
  half *d_v_new = AllocDevice<half>(B * kv_dim);
  half *d_attn_out = AllocDevice<half>(B * num_heads * head_dim);
  int *d_n_past = ToDevice(h_n_past);
  int *d_seq_ids = ToDevice(h_seq_ids);
  int *d_kv_lens = nullptr;
  {
    std::vector<int> h_kv_lens(B);
    for (int b = 0; b < B; ++b)
      h_kv_lens[b] = h_n_past[b] + 1;
    d_kv_lens = ToDevice(h_kv_lens);
  }

  // Fill Q/K/V with synthetic projected data
  {
    auto q_data = MakeRandom(B * num_heads * head_dim, 0.2f);
    auto k_data = MakeRandom(B * kv_dim, 0.15f);
    auto v_data = MakeRandom(B * kv_dim, 0.15f);
    REQUIRE(cudaMemcpy(d_q, q_data.data(),
                       B * num_heads * head_dim * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_k_new, k_data.data(), B * kv_dim * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_v_new, v_data.data(), B * kv_dim * sizeof(half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }

  // Step 1: BatchedRoPE
  {
    cudaError_t err = cuda_kernel::BatchedRoPE<half>(
        d_q, d_k_new, B, num_heads, num_kv_heads, head_dim, d_n_past,
        freq_base, nullptr, 0);
    INFO("BatchedRoPE launch: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
    err = cudaDeviceSynchronize();
    INFO("BatchedRoPE sync: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
  }

  // Step 2: BatchedKvAppendStrided
  {
    cudaError_t err = cuda_kernel::BatchedKvAppendStrided<half>(
        d_k_new, d_v_new, cache.Buffer(), d_seq_ids, d_n_past, 0, B,
        cache.KvDim(), cache.SlotStride(), cache.LayerStride(),
        cache.KvStride(), nullptr);
    INFO("BatchedKvAppendStrided launch: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
    err = cudaDeviceSynchronize();
    INFO("BatchedKvAppendStrided sync: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
  }

  // Step 3: FlashDecodeMultiSeqStrided
  {
    cudaError_t err = cuda_kernel::FlashDecodeMultiSeqStrided<half>(
        d_q, cache.Buffer(), d_attn_out, d_seq_ids, d_kv_lens, 0, B,
        num_heads, num_kv_heads, head_dim, cache.SlotStride(),
        cache.LayerStride(), cache.KvStride(), scale, nullptr);
    INFO("FlashDecodeMultiSeqStrided launch: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
    err = cudaDeviceSynchronize();
    INFO("FlashDecodeMultiSeqStrided sync: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);

    // Verify non-zero output
    auto h_out = FromDevice(d_attn_out, B * num_heads * head_dim);
    int nonzero = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
      float v = __half2float(h_out[i]);
      REQUIRE(std::isfinite(v));
      if (v != 0.0f)
        ++nonzero;
    }
    CHECK(nonzero > 0);

  }

  cudaFree(d_q);
  cudaFree(d_k_new);
  cudaFree(d_v_new);
  cudaFree(d_attn_out);
  cudaFree(d_n_past);
  cudaFree(d_seq_ids);
  cudaFree(d_kv_lens);
}

} // namespace inferflux

#else // !INFERFLUX_NATIVE_KERNELS_READY

TEST_CASE("Batched decode tests require CUDA", "[batched_decode]") {
  SKIP("CUDA not available");
}

#endif
