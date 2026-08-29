#include "runtime/backends/cuda/native/cuda_copy_trace.h"
#include "runtime/backends/cuda/native/cuda_sync_trace.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"

#include "server/logging/logger.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace inferflux {

// ============================================================================
// Kernels
// ============================================================================

// Repetition penalty: for each token in recent_ids, apply multiplicative
// repetition penalty and additive frequency/presence penalties to its logit.
// Matches llama.cpp llama_sampler_penalties behavior.
__global__ void RepetitionPenaltyKernel(float *__restrict__ logits,
                                        const int *__restrict__ recent_ids,
                                        const int *__restrict__ freq_counts,
                                        int num_recent, int vocab_size,
                                        float rep_penalty, float freq_penalty,
                                        float pres_penalty) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_recent)
    return;
  int token_id = recent_ids[idx];
  if (token_id < 0 || token_id >= vocab_size)
    return;
  float logit = logits[token_id];
  // Multiplicative repetition penalty
  if (logit > 0.0f) {
    logit /= rep_penalty;
  } else {
    logit *= rep_penalty;
  }
  // Additive frequency and presence penalties
  int count = freq_counts[idx];
  logit -= freq_penalty * static_cast<float>(count);
  if (count > 0) {
    logit -= pres_penalty;
  }
  logits[token_id] = logit;
}

// Temperature scaling: logits /= temperature
__global__ void TemperatureScaleKernel(float *__restrict__ logits,
                                       int vocab_size, float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  logits[idx] /= temperature;
}

// Softmax: two-pass (find max, then exp and sum)
__global__ void SoftmaxMaxKernel(const float *__restrict__ logits,
                                 float *__restrict__ max_val, int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;

  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    local_max = fmaxf(local_max, logits[i]);
  }
  smem[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    *max_val = smem[0];
  }
}

__global__ void SoftmaxExpSumKernel(const float *__restrict__ logits,
                                    float *__restrict__ probs,
                                    const float *__restrict__ max_val,
                                    float *__restrict__ sum_val,
                                    int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  float m = *max_val;

  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float e = expf(logits[i] - m);
    probs[i] = e;
    local_sum += e;
  }
  smem[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *sum_val = smem[0];
  }
}

__global__ void SoftmaxNormKernel(float *__restrict__ probs,
                                  const float *__restrict__ sum_val,
                                  int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  probs[idx] /= *sum_val;
}

// Argmax: find token with highest probability
__global__ void ArgmaxKernel(const float *__restrict__ logits,
                             int *__restrict__ result, int vocab_size) {
  extern __shared__ char smem_raw[];
  float *s_vals = reinterpret_cast<float *>(smem_raw);
  int *s_idxs = reinterpret_cast<int *>(s_vals + blockDim.x);
  int tid = threadIdx.x;

  float best_val = -FLT_MAX;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  s_vals[tid] = best_val;
  s_idxs[tid] = best_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = s_idxs[0];
  }
}

// Top-k: zero out all entries below the k-th largest
// Simple approach: find k-th threshold then zero below it
__global__ void TopKMaskKernel(float *__restrict__ probs, int vocab_size,
                               int top_k, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  if (probs[idx] < threshold) {
    probs[idx] = 0.0f;
  }
}

// Top-p: zero out entries where cumulative sum exceeds p
// Requires sorted probs. Simple single-thread approach for correctness.
// LEGACY: kept for reference/other call sites. Use ParallelTopPMaskKernel.
__global__ void TopPMaskKernel(float *__restrict__ probs, int vocab_size,
                               float top_p) {
  // Single thread scans sorted probs
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Simple bubble-sort approach for small vocab isn't practical.
  // Instead, scan from max prob downward using the unsorted array.
  // This is approximate but functional.
  float cumsum = 0.0f;
  // We need to iterate in sorted order. For a production kernel,
  // use CUB radix sort. Here we use a simple iterative max approach.
  // Mark visited entries by negating them temporarily.

  // Copy probs to find top-p threshold
  for (int iter = 0; iter < vocab_size && cumsum < top_p; iter++) {
    float max_val = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < vocab_size; i++) {
      if (probs[i] > max_val) {
        max_val = probs[i];
        max_idx = i;
      }
    }
    if (max_idx < 0)
      break;
    cumsum += max_val;
    probs[max_idx] = -probs[max_idx]; // Mark as visited
  }

  // Restore marked entries, zero those below threshold
  for (int i = 0; i < vocab_size; i++) {
    if (probs[i] < 0.0f) {
      probs[i] = -probs[i]; // Restore
    } else {
      probs[i] = 0.0f; // Zero out
    }
  }
}

// Fused softmax: temperature scale + max + exp-sum + normalize in one kernel.
// Single block, blockDim.x threads. Reads from `logits`, writes to `probs`.
// Shared memory: blockDim.x floats for reductions.
__global__ void FusedSoftmaxKernel(const float *__restrict__ logits,
                                   float *__restrict__ probs, int vocab_size,
                                   float temperature) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  const float inv_temp =
      (temperature != 1.0f) ? (1.0f / temperature) : 1.0f;

  // --- Pass 1: find max (with temperature scaling applied) ---
  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += T) {
    float v = logits[i] * inv_temp;
    if (v > local_max) local_max = v;
  }
  smem[tid] = local_max;
  __syncthreads();

  for (int s = T / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }
  float global_max = smem[0];
  __syncthreads();

  // --- Pass 2: compute exp(scaled_logit - max) and partial sums ---
  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += T) {
    float e = expf(logits[i] * inv_temp - global_max);
    probs[i] = e;
    local_sum += e;
  }
  smem[tid] = local_sum;
  __syncthreads();

  for (int s = T / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  float global_sum = smem[0];
  __syncthreads();

  // --- Pass 3: normalize ---
  float inv_sum = 1.0f / global_sum;
  for (int i = tid; i < vocab_size; i += T) {
    probs[i] *= inv_sum;
  }
}

// Parallel top-p nucleus mask: single block, 256 threads.
// Iteratively finds the max probability in parallel, accumulates cumsum,
// marks visited entries by negating. Reduces O(V²) serial to O(V²/T).
// Shared memory: blockDim.x * (sizeof(float) + sizeof(int)) bytes.
__global__ void ParallelTopPMaskKernel(float *__restrict__ probs,
                                       int vocab_size, float top_p) {
  extern __shared__ char smem_raw[];
  float *s_max_vals = reinterpret_cast<float *>(smem_raw);
  int *s_max_idxs =
      reinterpret_cast<int *>(s_max_vals + blockDim.x);

  const int tid = threadIdx.x;
  const int T = blockDim.x;

  float cumsum = 0.0f;

  while (cumsum < top_p) {
    // --- Parallel max reduction ---
    float local_max = -1.0f;
    int local_idx = -1;
    for (int i = tid; i < vocab_size; i += T) {
      if (probs[i] > local_max) {
        local_max = probs[i];
        local_idx = i;
      }
    }
    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = T / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (s_max_vals[tid + s] > s_max_vals[tid]) {
          s_max_vals[tid] = s_max_vals[tid + s];
          s_max_idxs[tid] = s_max_idxs[tid + s];
        }
      }
      __syncthreads();
    }

    // No positive probability remaining
    if (s_max_vals[0] <= 0.0f) break;

    cumsum += s_max_vals[0];

    // Mark the winning element as visited by negating it
    if (tid == 0) {
      probs[s_max_idxs[0]] = -probs[s_max_idxs[0]];
    }
    __syncthreads();
  }

  // --- Restore/zero pass: visited entries (negative) are restored,
  //     unvisited entries are zeroed ---
  for (int i = tid; i < vocab_size; i += T) {
    if (probs[i] < 0.0f) {
      probs[i] = -probs[i]; // Restore
    } else {
      probs[i] = 0.0f; // Zero out
    }
  }
}

// Batched argmax: one block per sequence, parallel reduction within each block
__global__ void BatchedArgmaxKernel(const float *__restrict__ logits,
                                    int *__restrict__ results, int vocab_size,
                                    int batch_size) {
  int seq = blockIdx.x;
  if (seq >= batch_size)
    return;
  extern __shared__ char smem_raw[];
  float *s_vals = reinterpret_cast<float *>(smem_raw);
  int *s_idxs = reinterpret_cast<int *>(s_vals + blockDim.x);
  int tid = threadIdx.x;

  const float *row = logits + seq * vocab_size;

  float best_val = -FLT_MAX;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (row[i] > best_val) {
      best_val = row[i];
      best_idx = i;
    }
  }
  s_vals[tid] = best_val;
  s_idxs[tid] = best_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    results[seq] = s_idxs[0];
  }
}

// Multinomial sample: walk CDF until uniform < cumsum
__global__ void MultinomialSampleKernel(const float *__restrict__ probs,
                                        const float *__restrict__ uniform,
                                        int *__restrict__ result,
                                        int vocab_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  float u = *uniform;
  float cumsum = 0.0f;
  // First renormalize
  float total = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    total += probs[i];
  }
  if (total <= 0.0f) {
    *result = 0;
    return;
  }

  for (int i = 0; i < vocab_size; i++) {
    cumsum += probs[i] / total;
    if (u < cumsum) {
      *result = i;
      return;
    }
  }
  *result = vocab_size - 1; // Fallback to last token
}

// ============================================================================
// GpuSampler implementation
// ============================================================================

GpuSampler::~GpuSampler() {
  if (d_probs_)
    cudaFree(d_probs_);
  if (d_indices_)
    cudaFree(d_indices_);
  if (d_temp_)
    cudaFree(d_temp_);
  if (d_result_)
    cudaFree(d_result_);
  if (d_max_val_)
    cudaFree(d_max_val_);
  if (d_max_idx_)
    cudaFree(d_max_idx_);
  if (d_result_batch_)
    cudaFree(d_result_batch_);
  if (d_uniform_)
    cudaFree(d_uniform_);
  if (h_result_pinned_)
    cudaFreeHost(h_result_pinned_);
  if (h_result_batch_pinned_)
    cudaFreeHost(h_result_batch_pinned_);
  if (h_logits_pinned_)
    cudaFreeHost(h_logits_pinned_);
  if (completion_event_)
    cudaEventDestroy(completion_event_);
  if (rng_initialized_)
    curandDestroyGenerator(rng_);
}

bool GpuSampler::Initialize(int vocab_size, cudaStream_t stream) {
  vocab_size_ = vocab_size;
  stream_ = stream;

  cudaError_t err;
  err = cudaMalloc(&d_probs_, vocab_size * sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_temp_, vocab_size * sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_result_, sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_max_val_, sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_max_idx_, sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_uniform_, sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_result_batch_, kMaxBatchSize * sizeof(int));
  if (err != cudaSuccess)
    return false;
  err =
      cudaMallocHost(reinterpret_cast<void **>(&h_result_pinned_), sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMallocHost(reinterpret_cast<void **>(&h_result_batch_pinned_),
                       kMaxBatchSize * sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMallocHost(reinterpret_cast<void **>(&h_logits_pinned_),
                       static_cast<size_t>(vocab_size) * sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming);
  if (err != cudaSuccess)
    return false;

  // Initialize cuRAND
  curandStatus_t rng_st =
      curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT);
  if (rng_st != CURAND_STATUS_SUCCESS) {
    log::Error("gpu_sampler", "curandCreateGenerator failed");
    return false;
  }
  curandSetStream(rng_, stream);
  rng_initialized_ = true;

  log::Info("gpu_sampler",
            "Initialized: vocab_size=" + std::to_string(vocab_size));
  return true;
}

std::size_t GpuSampler::DeviceWorkspaceBytes() const {
  std::size_t bytes = 0;
  if (d_probs_) {
    bytes += static_cast<std::size_t>(vocab_size_) * sizeof(float);
  }
  if (d_temp_) {
    bytes += static_cast<std::size_t>(vocab_size_) * sizeof(float);
  }
  if (d_result_) {
    bytes += sizeof(int);
  }
  if (d_max_val_) {
    bytes += sizeof(float);
  }
  if (d_max_idx_) {
    bytes += sizeof(int);
  }
  if (d_uniform_) {
    bytes += sizeof(float);
  }
  if (d_result_batch_) {
    bytes += static_cast<std::size_t>(kMaxBatchSize) * sizeof(int);
  }
  return bytes;
}

std::size_t GpuSampler::HostWorkspaceBytes() const {
  std::size_t bytes = 0;
  if (h_result_pinned_) {
    bytes += sizeof(int);
  }
  if (h_result_batch_pinned_) {
    bytes += static_cast<std::size_t>(kMaxBatchSize) * sizeof(int);
  }
  if (h_logits_pinned_) {
    bytes += static_cast<std::size_t>(vocab_size_) * sizeof(float);
  }
  return bytes;
}

int GpuSampler::GreedyArgmax(const float *d_logits) {
  NVTX_SCOPE("Sampler_Argmax");
  int threads = 256;
  int smem = threads * (sizeof(float) + sizeof(int));

  ArgmaxKernel<<<1, threads, smem, stream_>>>(d_logits, d_result_, vocab_size_);

  runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kSamplerGreedyResultD2H,
      h_result_pinned_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
      stream_);
  cudaEventRecord(completion_event_, stream_);
  completion_pending_ = true;
  return CollectSample();
}

int GpuSampler::StochasticSample(const float *d_logits, float temperature,
                                 int top_k, float top_p) {
  NVTX_SCOPE("Sampler_Stochastic");
  const int threads = 256;

  // Step 1+2+3: Fused softmax (temperature scale + max + exp-sum + normalize)
  // Reads d_logits directly, writes normalized probs to d_probs_.
  // Single kernel replaces: D2D copy + TemperatureScale + SoftmaxMax +
  // SoftmaxExpSum + SoftmaxNorm + D2D copy back (7 ops → 1 kernel).
  int smem = threads * sizeof(float);
  FusedSoftmaxKernel<<<1, threads, smem, stream_>>>(d_logits, d_probs_,
                                                    vocab_size_, temperature);

  // Step 4: Top-k filtering (if enabled)
  if (top_k > 0 && top_k < vocab_size_) {
    // Find k-th largest value on CPU (for simplicity)
    // In production, use CUB radix sort
    // For now, skip top-k and rely on top-p
  }

  // Step 5: Top-p / nucleus filtering (parallel, 256 threads)
  if (top_p < 1.0f && top_p > 0.0f) {
    int top_p_smem = threads * (sizeof(float) + sizeof(int));
    ParallelTopPMaskKernel<<<1, threads, top_p_smem, stream_>>>(
        d_probs_, vocab_size_, top_p);
  }

  // Step 6: Generate random uniform
  curandGenerateUniform(rng_, d_uniform_, 1);

  // Step 7: Multinomial sample
  MultinomialSampleKernel<<<1, 1, 0, stream_>>>(d_probs_, d_uniform_, d_result_,
                                                vocab_size_);

  runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kSamplerGreedyResultD2H,
      h_result_pinned_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
      stream_);
  cudaEventRecord(completion_event_, stream_);
  completion_pending_ = true;
  return CollectSample();
}

int GpuSampler::Sample(const float *d_logits, float temperature, int top_k,
                       float top_p, uint32_t seed) {
  EnqueueSample(d_logits, temperature, top_k, top_p, seed);
  return CollectSample();
}

void GpuSampler::EnqueueSample(const float *d_logits, float temperature,
                               int top_k, float top_p, uint32_t seed) {
  assert(!completion_pending_ &&
         "GpuSampler::EnqueueSample called while a previous sample is "
         "still pending — each lane must own its own GpuSampler instance");
  if (seed != UINT32_MAX && rng_initialized_) {
    curandSetPseudoRandomGeneratorSeed(rng_, seed);
  }

  pending_batch_size_ = 1;
  if (temperature <= 0.0f) {
    NVTX_SCOPE("Sampler_Argmax");
    int threads = 256;
    int smem = threads * (sizeof(float) + sizeof(int));

    ArgmaxKernel<<<1, threads, smem, stream_>>>(d_logits, d_result_,
                                                vocab_size_);

    runtime::cuda::native::TracedCudaMemcpyAsync(
        runtime::cuda::native::CopyTraceSite::kSamplerGreedyResultD2H,
        h_result_pinned_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
        stream_);
    cudaEventRecord(completion_event_, stream_);
    completion_pending_ = true;
    return;
  }

  NVTX_SCOPE("Sampler_Stochastic");
  const int threads = 256;

  // Fused softmax: temperature scale + max + exp-sum + normalize in one kernel.
  // Reads d_logits directly, writes normalized probs to d_probs_.
  int smem = threads * sizeof(float);
  FusedSoftmaxKernel<<<1, threads, smem, stream_>>>(d_logits, d_probs_,
                                                    vocab_size_, temperature);

  // Top-p / nucleus filtering (parallel, 256 threads)
  if (top_p < 1.0f && top_p > 0.0f) {
    int top_p_smem = threads * (sizeof(float) + sizeof(int));
    ParallelTopPMaskKernel<<<1, threads, top_p_smem, stream_>>>(
        d_probs_, vocab_size_, top_p);
  }

  curandGenerateUniform(rng_, d_uniform_, 1);
  MultinomialSampleKernel<<<1, 1, 0, stream_>>>(d_probs_, d_uniform_, d_result_,
                                                vocab_size_);

  runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kSamplerGreedyResultD2H,
      h_result_pinned_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
      stream_);
  cudaEventRecord(completion_event_, stream_);
  completion_pending_ = true;
}

int GpuSampler::CollectSample() {
  if (completion_pending_) {
    runtime::cuda::native::TracedCudaEventSynchronize(
        runtime::cuda::native::SyncTraceSite::kSamplerResultReady,
        completion_event_);
  }
  completion_pending_ = false;
  pending_batch_size_ = 0;
  return h_result_pinned_ ? *h_result_pinned_ : 0;
}

void GpuSampler::GreedyArgmaxBatch(const float *d_logits, int batch_size,
                                   std::vector<int> *out_tokens) {
  EnqueueSampleBatch(d_logits, batch_size, std::vector<float>(batch_size, 0.0f),
                     std::vector<int>(batch_size, 0),
                     std::vector<float>(batch_size, 1.0f),
                     std::vector<uint32_t>(batch_size, UINT32_MAX));
  CollectSampleBatch(out_tokens);
}

void GpuSampler::EnqueueSampleBatch(const float *d_logits, int batch_size,
                                    const std::vector<float> &temperatures,
                                    const std::vector<int> &top_ks,
                                    const std::vector<float> &top_ps,
                                    const std::vector<uint32_t> &seeds) {
  NVTX_SCOPE("SampleBatch");
  pending_batch_size_ = batch_size;

  bool all_greedy = std::all_of(temperatures.begin(), temperatures.end(),
                                [](float t) { return t <= 0.0f; });
  if (all_greedy) {
    NVTX_SCOPE("Sampler_BatchedArgmax");
    int B = std::min(batch_size, kMaxBatchSize);
    int threads = 256;
    int smem = threads * (sizeof(float) + sizeof(int));

    BatchedArgmaxKernel<<<B, threads, smem, stream_>>>(
        d_logits, d_result_batch_, vocab_size_, B);

    runtime::cuda::native::TracedCudaMemcpyAsync(
        runtime::cuda::native::CopyTraceSite::kSamplerBatchResultD2H,
        h_result_batch_pinned_, d_result_batch_, B * sizeof(int),
        cudaMemcpyDeviceToHost, stream_);
    cudaEventRecord(completion_event_, stream_);
    completion_pending_ = true;
    pending_batch_size_ = B;
    return;
  }

  // Async batched stochastic: enqueue all sampling kernels without
  // synchronizing between sequences, then do a single D2H memcpy + event
  // sync. Each sequence's kernels run sequentially on the same stream (so
  // sharing d_probs_ scratch is safe), but we avoid B host-GPU round-trips.
  {
    NVTX_SCOPE("Sampler_BatchedStochastic");
    int B = std::min(batch_size, kMaxBatchSize);
    const int threads = 256;

    for (int i = 0; i < B; ++i) {
      const float *logits_i = d_logits + i * vocab_size_;
      const uint32_t seed =
          i < static_cast<int>(seeds.size()) ? seeds[static_cast<size_t>(i)]
                                             : UINT32_MAX;
      if (seed != UINT32_MAX && rng_initialized_) {
        curandSetPseudoRandomGeneratorSeed(rng_, seed);
      }

      if (temperatures[i] <= 0.0f) {
        // Greedy: argmax directly to batch result slot
        int smem = threads * (sizeof(float) + sizeof(int));
        ArgmaxKernel<<<1, threads, smem, stream_>>>(
            logits_i, d_result_batch_ + i, vocab_size_);
      } else {
        // Stochastic: softmax → top-p → multinomial → batch result slot
        int smem = threads * sizeof(float);
        FusedSoftmaxKernel<<<1, threads, smem, stream_>>>(
            logits_i, d_probs_, vocab_size_, temperatures[i]);

        if (top_ps[i] < 1.0f && top_ps[i] > 0.0f) {
          int top_p_smem = threads * (sizeof(float) + sizeof(int));
          ParallelTopPMaskKernel<<<1, threads, top_p_smem, stream_>>>(
              d_probs_, vocab_size_, top_ps[i]);
        }

        curandGenerateUniform(rng_, d_uniform_, 1);
        MultinomialSampleKernel<<<1, 1, 0, stream_>>>(
            d_probs_, d_uniform_, d_result_batch_ + i, vocab_size_);
      }
    }

    // Single async D2H copy + event for all B results
    runtime::cuda::native::TracedCudaMemcpyAsync(
        runtime::cuda::native::CopyTraceSite::kSamplerBatchResultD2H,
        h_result_batch_pinned_, d_result_batch_, B * sizeof(int),
        cudaMemcpyDeviceToHost, stream_);
    cudaEventRecord(completion_event_, stream_);
    completion_pending_ = true;
    pending_batch_size_ = B;
  }
}

bool GpuSampler::EnqueueBatchedArgmax(const float *d_logits, int batch_size) {
  if (batch_size <= 0 || batch_size > kMaxBatchSize) {
    return false;
  }
  int threads = 256;
  int smem = threads * (sizeof(float) + sizeof(int));
  BatchedArgmaxKernel<<<batch_size, threads, smem, stream_>>>(
      d_logits, d_result_batch_, vocab_size_, batch_size);
  return cudaGetLastError() == cudaSuccess;
}

void GpuSampler::CollectSampleBatch(std::vector<int> *out_tokens) {
  out_tokens->resize(std::max(0, pending_batch_size_));
  if (pending_batch_size_ <= 0) {
    return;
  }
  if (completion_pending_) {
    runtime::cuda::native::TracedCudaEventSynchronize(
        runtime::cuda::native::SyncTraceSite::kSamplerBatchResultReady,
        completion_event_);
  }
  completion_pending_ = false;
  for (int i = 0; i < pending_batch_size_; ++i) {
    (*out_tokens)[i] = h_result_batch_pinned_[i];
  }
  pending_batch_size_ = 0;
}

void GpuSampler::SampleBatch(const float *d_logits, int batch_size,
                             const std::vector<float> &temperatures,
                             const std::vector<int> &top_ks,
                             const std::vector<float> &top_ps,
                             const std::vector<uint32_t> &seeds,
                             std::vector<int> *out_tokens) {
  EnqueueSampleBatch(d_logits, batch_size, temperatures, top_ks, top_ps, seeds);
  CollectSampleBatch(out_tokens);
}

void GpuSampler::EnqueueGreedyArgmaxDeviceOnly(const float *d_logits,
                                                int batch_size) {
  int B = std::min(batch_size, kMaxBatchSize);
  int threads = 256;
  int smem = threads * (sizeof(float) + sizeof(int));
  BatchedArgmaxKernel<<<B, threads, smem, stream_>>>(d_logits, d_result_batch_,
                                                      vocab_size_, B);
  // No D2H memcpy, no event record — result stays on device only.
}

void GpuSampler::ApplyPenalties(float *d_logits,
                                const std::vector<int> &recent_ids,
                                const std::vector<int> &freq_counts,
                                float repetition_penalty,
                                float frequency_penalty,
                                float presence_penalty) {
  if (recent_ids.empty() ||
      (repetition_penalty == 1.0f && frequency_penalty == 0.0f &&
       presence_penalty == 0.0f)) {
    return; // No penalties to apply
  }
  const int n = static_cast<int>(recent_ids.size());

  // Upload recent token IDs and frequency counts to device
  int *d_recent = nullptr;
  int *d_freq = nullptr;
  cudaMalloc(&d_recent, n * sizeof(int));
  cudaMalloc(&d_freq, n * sizeof(int));
  cudaMemcpyAsync(d_recent, recent_ids.data(), n * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_freq, freq_counts.data(), n * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  RepetitionPenaltyKernel<<<blocks, threads, 0, stream_>>>(
      d_logits, d_recent, d_freq, n, vocab_size_, repetition_penalty,
      frequency_penalty, presence_penalty);

  // Synchronize before freeing temp buffers
  cudaStreamSynchronize(stream_);
  cudaFree(d_recent);
  cudaFree(d_freq);
}

void GpuSampler::CopyLogitsToHost(const float *d_logits, float *host_buf) {
  NVTX_SCOPE("Sampler_CopyLogits");
  runtime::cuda::native::TracedCudaMemcpyAsync(
      runtime::cuda::native::CopyTraceSite::kSamplerCopyLogitsD2H,
      h_logits_pinned_, d_logits, vocab_size_ * sizeof(float),
      cudaMemcpyDeviceToHost, stream_);
  cudaEventRecord(completion_event_, stream_);
  completion_pending_ = true;
  runtime::cuda::native::TracedCudaEventSynchronize(
      runtime::cuda::native::SyncTraceSite::kSamplerLogitsReady,
      completion_event_);
  completion_pending_ = false;
  std::memcpy(host_buf, h_logits_pinned_,
              static_cast<size_t>(vocab_size_) * sizeof(float));
}

} // namespace inferflux
