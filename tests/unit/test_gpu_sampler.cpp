#include <catch2/catch_amalgamated.hpp>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/decode_burst.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include <cuda_runtime.h>
#include <vector>
#endif

namespace inferflux {

TEST_CASE("GpuSampler: interface compiles", "[gpu_sampler]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Verify the class interface compiles
  GpuSampler sampler;
  // Without initialization, we can't call Sample
  REQUIRE(true);
#else
  REQUIRE(true);
#endif
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("GpuSampler: Greedy argmax", "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 100;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Create logits with a known maximum at position 42
  std::vector<float> h_logits(vocab_size, 0.0f);
  h_logits[42] = 10.0f;

  float *d_logits;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Greedy (temperature=0) should return 42
  int token = sampler.Sample(d_logits, 0.0f, 0, 1.0f);
  REQUIRE(token == 42);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Async greedy argmax collect", "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 64;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  std::vector<float> h_logits(vocab_size, -1.0f);
  h_logits[17] = 4.0f;

  float *d_logits = nullptr;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  sampler.EnqueueSample(d_logits, 0.0f, 0, 1.0f);
  REQUIRE(sampler.CollectSample() == 17);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Batched greedy argmax", "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 100;
  int batch_size = 4;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Create batched logits with known maxima at different positions
  std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
  h_logits[0 * vocab_size + 10] = 10.0f; // seq 0 -> token 10
  h_logits[1 * vocab_size + 42] = 10.0f; // seq 1 -> token 42
  h_logits[2 * vocab_size + 0] = 10.0f;  // seq 2 -> token 0
  h_logits[3 * vocab_size + 99] = 10.0f; // seq 3 -> token 99

  float *d_logits;
  cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), batch_size * vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // All greedy temperatures -> should use batched kernel
  std::vector<float> temps(batch_size, 0.0f);
  std::vector<int> top_ks(batch_size, 0);
  std::vector<float> top_ps(batch_size, 1.0f);
  std::vector<uint32_t> seeds(batch_size, UINT32_MAX);
  std::vector<int> out_tokens;

  sampler.SampleBatch(d_logits, batch_size, temps, top_ks, top_ps, seeds,
                      &out_tokens);

  REQUIRE(out_tokens.size() == 4);
  REQUIRE(out_tokens[0] == 10);
  REQUIRE(out_tokens[1] == 42);
  REQUIRE(out_tokens[2] == 0);
  REQUIRE(out_tokens[3] == 99);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Async batched greedy argmax collect",
          "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 100;
  int batch_size = 3;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
  h_logits[0 * vocab_size + 3] = 8.0f;
  h_logits[1 * vocab_size + 15] = 9.0f;
  h_logits[2 * vocab_size + 71] = 7.0f;

  float *d_logits = nullptr;
  cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), batch_size * vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  const std::vector<float> temps(batch_size, 0.0f);
  const std::vector<int> top_ks(batch_size, 0);
  const std::vector<float> top_ps(batch_size, 1.0f);
  const std::vector<uint32_t> seeds(batch_size, UINT32_MAX);
  std::vector<int> out_tokens;

  sampler.EnqueueSampleBatch(d_logits, batch_size, temps, top_ks, top_ps,
                             seeds);
  sampler.CollectSampleBatch(&out_tokens);

  REQUIRE(out_tokens == std::vector<int>{3, 15, 71});

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Stochastic sample returns valid token",
          "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 50;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Uniform logits
  std::vector<float> h_logits(vocab_size, 1.0f);

  float *d_logits;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Stochastic sample should return a valid token
  int token = sampler.Sample(d_logits, 1.0f, 0, 1.0f, 42);
  REQUIRE(token >= 0);
  REQUIRE(token < vocab_size);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE(
    "GpuSampler: Batched stochastic sampling preserves per-sequence seeds",
    "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int vocab_size = 64;
  constexpr int batch_size = 2;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler batch_sampler;
  GpuSampler single_sampler;
  REQUIRE(batch_sampler.Initialize(vocab_size, stream));
  REQUIRE(single_sampler.Initialize(vocab_size, stream));

  std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < vocab_size; ++i) {
      h_logits[b * vocab_size + i] =
          0.01f * static_cast<float>((b + 1) * ((i % 7) - 3));
    }
  }

  float *d_logits = nullptr;
  cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), batch_size * vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  std::vector<float> temps(batch_size, 1.0f);
  std::vector<int> top_ks(batch_size, 0);
  std::vector<float> top_ps(batch_size, 1.0f);
  const std::vector<uint32_t> seeds = {123u, 98765u};
  std::vector<int> batch_tokens;

  batch_sampler.SampleBatch(d_logits, batch_size, temps, top_ks, top_ps, seeds,
                            &batch_tokens);

  REQUIRE(batch_tokens.size() == batch_size);
  for (int i = 0; i < batch_size; ++i) {
    const int single = single_sampler.Sample(
        d_logits + i * vocab_size, temps[i], top_ks[i], top_ps[i], seeds[i]);
    REQUIRE(batch_tokens[i] == single);
  }

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

namespace {
// Device mirror helpers for the DecodeTokenFeed tests: upload state, run the
// kernel, sync, download state.
struct FeedState {
  static constexpr int kB = 4;
  int *sampled = nullptr;
  int *token_ids = nullptr;
  int *n_past = nullptr;
  int *kv_lens = nullptr;
  int *eos = nullptr;
  int *done = nullptr;
  int *steps_left = nullptr;

  std::vector<int> h_sampled, h_token_ids, h_n_past, h_kv_lens, h_eos, h_done,
      h_steps;

  explicit FeedState(cudaStream_t stream) {
    h_sampled = {5, 7, 99, 12};
    h_token_ids = {0, 0, 0, 0};
    h_n_past = {10, 20, 30, 40};
    h_kv_lens = {11, 21, 31, 41};
    h_eos = {99, 3};
    h_done = {0, 0, 0, 0};
    h_steps = {3, 3, 3, 0};
    cudaMalloc(&sampled, sizeof(int) * kB);
    cudaMalloc(&token_ids, sizeof(int) * kB);
    cudaMalloc(&n_past, sizeof(int) * kB);
    cudaMalloc(&kv_lens, sizeof(int) * kB);
    cudaMalloc(&eos, sizeof(int) * 2);
    cudaMalloc(&done, sizeof(int) * kB);
    cudaMalloc(&steps_left, sizeof(int) * kB);
    Upload(stream);
  }
  ~FeedState() {
    cudaFree(sampled);
    cudaFree(token_ids);
    cudaFree(n_past);
    cudaFree(kv_lens);
    cudaFree(eos);
    cudaFree(done);
    cudaFree(steps_left);
  }
  void Upload(cudaStream_t stream) {
    cudaMemcpyAsync(sampled, h_sampled.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(token_ids, h_token_ids.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(n_past, h_n_past.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(kv_lens, h_kv_lens.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eos, h_eos.data(), sizeof(int) * 2, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(done, h_done.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(steps_left, h_steps.data(), sizeof(int) * kB,
                    cudaMemcpyHostToDevice, stream);
  }
  void Download(cudaStream_t stream) {
    cudaMemcpyAsync(h_token_ids.data(), token_ids, sizeof(int) * kB,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_n_past.data(), n_past, sizeof(int) * kB,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_kv_lens.data(), kv_lens, sizeof(int) * kB,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_done.data(), done, sizeof(int) * kB,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_steps.data(), steps_left, sizeof(int) * kB,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};
} // namespace

TEST_CASE("DecodeTokenFeed: advances metadata, freezes on EOS and budget",
          "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  {
    FeedState st(stream);
    // seq0/seq1: normal advance. seq2: sampled 99 == EOS -> freeze.
    // seq3: steps_left 0 -> no-op (already exhausted).
    REQUIRE(runtime::cuda::native::LaunchDecodeTokenFeed(
        st.sampled, st.token_ids, st.n_past, st.kv_lens, st.eos, 2, st.done,
        st.steps_left, FeedState::kB, stream));
    st.Download(stream);

    REQUIRE(st.h_token_ids[0] == 5);
    REQUIRE(st.h_n_past[0] == 11);
    REQUIRE(st.h_kv_lens[0] == 12);
    REQUIRE(st.h_steps[0] == 2);
    REQUIRE(st.h_done[0] == 0);

    REQUIRE(st.h_token_ids[1] == 7);
    REQUIRE(st.h_n_past[1] == 21);
    REQUIRE(st.h_kv_lens[1] == 22);
    REQUIRE(st.h_steps[1] == 2);

    // EOS: frozen before any state update, flag set.
    REQUIRE(st.h_done[2] == 1);
    REQUIRE(st.h_token_ids[2] == 0);
    REQUIRE(st.h_n_past[2] == 30);
    REQUIRE(st.h_kv_lens[2] == 31);
    REQUIRE(st.h_steps[2] == 3);

    // Exhausted budget: no-op.
    REQUIRE(st.h_token_ids[3] == 0);
    REQUIRE(st.h_n_past[3] == 40);
    REQUIRE(st.h_kv_lens[3] == 41);
    REQUIRE(st.h_steps[3] == 0);

    // Idempotence: replaying the same launch (as a graph replay past a
    // freeze would) must not advance seq2 (done) or seq3 (budget), and must
    // advance the live sequences exactly once more.
    REQUIRE(runtime::cuda::native::LaunchDecodeTokenFeed(
        st.sampled, st.token_ids, st.n_past, st.kv_lens, st.eos, 2, st.done,
        st.steps_left, FeedState::kB, stream));
    st.Download(stream);
    REQUIRE(st.h_n_past[0] == 12);
    REQUIRE(st.h_kv_lens[0] == 13);
    REQUIRE(st.h_steps[0] == 1);
    REQUIRE(st.h_done[2] == 1);
    REQUIRE(st.h_n_past[2] == 30);
    REQUIRE(st.h_n_past[3] == 40);

    // Budget exhaustion mid-burst: seq0 hits steps_left 0 on this launch and
    // must stop advancing afterwards.
    REQUIRE(runtime::cuda::native::LaunchDecodeTokenFeed(
        st.sampled, st.token_ids, st.n_past, st.kv_lens, st.eos, 2, st.done,
        st.steps_left, FeedState::kB, stream));
    st.Download(stream);
    REQUIRE(st.h_steps[0] == 0);
    REQUIRE(st.h_n_past[0] == 13);
    REQUIRE(runtime::cuda::native::LaunchDecodeTokenFeed(
        st.sampled, st.token_ids, st.n_past, st.kv_lens, st.eos, 2, st.done,
        st.steps_left, FeedState::kB, stream));
    st.Download(stream);
    REQUIRE(st.h_n_past[0] == 13); // frozen at budget exhaustion
  }
  cudaStreamDestroy(stream);
}
#endif

} // namespace inferflux
