#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include <cmath>
#include <cstdint>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// RMS Normalization (templated)
// ============================================================================

template <typename T>
__global__ void
RmsNormKernel(const T *__restrict__ input, const T *__restrict__ weight,
              T *__restrict__ output, int hidden_size, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T *x = input + row * hidden_size;
  T *y = output + row * hidden_size;

  extern __shared__ float shared[];

  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = DtypeTraits<T>::to_float(x[i]);
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = DtypeTraits<T>::to_float(x[i]) * rms *
                DtypeTraits<T>::to_float(weight[i]);
    y[i] = DtypeTraits<T>::from_float(val);
  }
}

template <typename T>
cudaError_t RmsNorm(const T *input, const T *weight, T *output, int count,
                    int hidden_size, float eps, cudaStream_t stream) {
  int threads = min(1024, hidden_size);
  int t = 1;
  while (t < threads)
    t <<= 1;
  threads = t;
  int smem = threads * sizeof(float);

  RmsNormKernel<T><<<count, threads, smem, stream>>>(input, weight, output,
                                                     hidden_size, eps);
  return cudaGetLastError();
}

// ============================================================================
// Rotary Position Embedding (RoPE) (templated)
// ============================================================================

// rope_type: 0 = kNorm (consecutive pairs: (0,1),(2,3),...),
//            2 = kNeox (split-half pairs: (0,d/2),(1,d/2+1),...)
template <typename T>
__global__ void RoPEKernel(T *__restrict__ q, T *__restrict__ k, int seq_len,
                           int num_heads, int num_kv_heads, int head_dim,
                           int n_past, float freq_base, int rope_type) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim / 2;
  const int total_q_pairs = seq_len * num_heads * half_dim;
  const int total_k_pairs = seq_len * num_kv_heads * half_dim;
  const int total_pairs = total_q_pairs + total_k_pairs;

  if (idx >= total_pairs)
    return;

  bool is_q = (idx < total_q_pairs);
  int local_idx = is_q ? idx : (idx - total_q_pairs);
  int n_heads = is_q ? num_heads : num_kv_heads;
  T *tensor = is_q ? q : k;

  int pair_idx = local_idx % half_dim;
  int head_idx = (local_idx / half_dim) % n_heads;
  int pos_idx = local_idx / (half_dim * n_heads);

  int position = n_past + pos_idx;
  float freq =
      1.0f / powf(freq_base, 2.0f * pair_idx / static_cast<float>(head_dim));
  float angle = position * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  int offset = pos_idx * n_heads * head_dim + head_idx * head_dim;
  int i0, i1;
  if (rope_type == 2) {
    // kNeox: split-half pairs (0,d/2),(1,d/2+1),...
    i0 = offset + pair_idx;
    i1 = offset + pair_idx + half_dim;
  } else {
    // kNorm: consecutive pairs (0,1),(2,3),(4,5),...
    i0 = offset + 2 * pair_idx;
    i1 = offset + 2 * pair_idx + 1;
  }

  float v0 = DtypeTraits<T>::to_float(tensor[i0]);
  float v1 = DtypeTraits<T>::to_float(tensor[i1]);

  tensor[i0] = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
  tensor[i1] = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);
}

template <typename T>
cudaError_t RoPE(T *q, T *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type) {
  int half_dim = head_dim / 2;
  int total_pairs =
      seq_len * num_heads * half_dim + seq_len * num_kv_heads * half_dim;
  int threads = 256;
  int blocks = (total_pairs + threads - 1) / threads;

  RoPEKernel<T><<<blocks, threads, 0, stream>>>(q, k, seq_len, num_heads,
                                                num_kv_heads, head_dim, n_past,
                                                freq_base, rope_type);
  return cudaGetLastError();
}

// ============================================================================
// SwiGLU: silu(gate) * up (templated)
// ============================================================================

template <typename T>
__global__ void SiluMulKernel(const T *__restrict__ gate,
                              const T *__restrict__ up, T *__restrict__ output,
                              int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  float g = DtypeTraits<T>::to_float(gate[idx]);
  float u = DtypeTraits<T>::to_float(up[idx]);
  float silu = g / (1.0f + expf(-g));
  output[idx] = DtypeTraits<T>::from_float(silu * u);
}

template <typename T>
cudaError_t SiluMul(const T *gate, const T *up, T *output, int count,
                    cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  SiluMulKernel<T><<<blocks, threads, 0, stream>>>(gate, up, output, count);
  return cudaGetLastError();
}

// ============================================================================
// Residual Addition (templated)
// ============================================================================

template <typename T>
__global__ void ResidualAddKernel(T *__restrict__ residual,
                                  const T *__restrict__ input, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  float r = DtypeTraits<T>::to_float(residual[idx]);
  float x = DtypeTraits<T>::to_float(input[idx]);
  residual[idx] = DtypeTraits<T>::from_float(r + x);
}

template <typename T>
cudaError_t ResidualAdd(T *residual, const T *input, int count,
                        cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  ResidualAddKernel<T><<<blocks, threads, 0, stream>>>(residual, input, count);
  return cudaGetLastError();
}

// ============================================================================
// Fused Residual Add + RMS Normalization
// ============================================================================
// Combines residual += input and output = RmsNorm(residual, weight, eps) into
// a single kernel, eliminating one kernel launch per fusion site.

template <typename T>
__global__ void
ResidualAddRmsNormKernel(T *__restrict__ residual, const T *__restrict__ input,
                         const T *__restrict__ weight, T *__restrict__ output,
                         int hidden_size, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  T *res_row = residual + row * hidden_size;
  T *out_row = output + row * hidden_size;

  extern __shared__ float shared[];

  // Pass 1: residual += input, compute sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float r = DtypeTraits<T>::to_float(res_row[i]);
    float x = DtypeTraits<T>::to_float(input[row * hidden_size + i]);
    float val = r + x;
    res_row[i] = DtypeTraits<T>::from_float(val);
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

  // Pass 2: apply RmsNorm (re-read from residual which now has the sum)
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = DtypeTraits<T>::to_float(res_row[i]) * rms *
                DtypeTraits<T>::to_float(weight[i]);
    out_row[i] = DtypeTraits<T>::from_float(val);
  }
}

template <typename T>
cudaError_t ResidualAddRmsNorm(T *residual, const T *input, const T *weight,
                               T *output, int count, int hidden_size, float eps,
                               cudaStream_t stream) {
  int threads = min(1024, hidden_size);
  int t = 1;
  while (t < threads)
    t <<= 1;
  threads = t;
  int smem = threads * sizeof(float);

  ResidualAddRmsNormKernel<T><<<count, threads, smem, stream>>>(
      residual, input, weight, output, hidden_size, eps);
  return cudaGetLastError();
}

// ============================================================================
// Embedding Lookup (templated)
// ============================================================================

template <typename T>
__global__ void EmbeddingLookupKernel(const T *__restrict__ table,
                                      const int *__restrict__ token_ids,
                                      T *__restrict__ output, int seq_len,
                                      int hidden_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq_len * hidden_size;
  if (idx >= total)
    return;

  int pos = idx / hidden_size;
  int dim = idx % hidden_size;
  int token_id = token_ids[pos];

  output[idx] = table[token_id * hidden_size + dim];
}

template <typename T>
cudaError_t EmbeddingLookup(const T *table, const int *token_ids, T *output,
                            int seq_len, int hidden_size, cudaStream_t stream) {
  int total = seq_len * hidden_size;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  EmbeddingLookupKernel<T><<<blocks, threads, 0, stream>>>(
      table, token_ids, output, seq_len, hidden_size);
  return cudaGetLastError();
}

// ============================================================================
// Half/BF16 to Float conversion (templated)
// ============================================================================

template <typename T>
__global__ void HalfToFloatKernel(const T *__restrict__ input,
                                  float *__restrict__ output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  output[idx] = DtypeTraits<T>::to_float(input[idx]);
}

template <typename T>
cudaError_t HalfToFloat(const T *input, float *output, int count,
                        cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  HalfToFloatKernel<T><<<blocks, threads, 0, stream>>>(input, output, count);
  return cudaGetLastError();
}

// ============================================================================
// Symmetric per-row activation quantization (templated)
// ============================================================================

template <typename T>
__global__ void QuantizeRowsSymmetricKernel(const T *__restrict__ input,
                                            int8_t *__restrict__ output,
                                            float *__restrict__ row_scales,
                                            int cols) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T *x = input + static_cast<size_t>(row) * cols;
  int8_t *y = output + static_cast<size_t>(row) * cols;

  extern __shared__ float shared[];

  float local_max = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x) {
    const float v = DtypeTraits<T>::to_float(x[i]);
    local_max = fmaxf(local_max, fabsf(v));
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  const float scale = (shared[0] > 0.0f) ? shared[0] / 127.0f : 0.0f;
  const float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
  if (tid == 0) {
    row_scales[row] = scale;
  }
  __syncthreads();

  for (int i = tid; i < cols; i += blockDim.x) {
    const float scaled = DtypeTraits<T>::to_float(x[i]) * inv_scale;
    const int q = max(-127, min(127, __float2int_rn(scaled)));
    y[i] = static_cast<int8_t>(q);
  }
}

template <typename T>
cudaError_t QuantizeRowsSymmetric(const T *input, int8_t *output,
                                  float *row_scales, int rows, int cols,
                                  cudaStream_t stream) {
  if (!input || !output || !row_scales || rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }

  int threads = min(256, cols);
  int t = 1;
  while (t < threads) {
    t <<= 1;
  }
  threads = t;
  const int smem = threads * sizeof(float);
  QuantizeRowsSymmetricKernel<T>
      <<<rows, threads, smem, stream>>>(input, output, row_scales, cols);
  return cudaGetLastError();
}

template <typename T>
__global__ void SiluMulQuantizeRowsSymmetricKernel(
    const T *__restrict__ gate, const T *__restrict__ up,
    int8_t *__restrict__ output, float *__restrict__ row_scales, int cols) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T *g_row = gate + static_cast<size_t>(row) * cols;
  const T *u_row = up + static_cast<size_t>(row) * cols;
  int8_t *y = output + static_cast<size_t>(row) * cols;

  extern __shared__ float shared[];

  float local_max = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x) {
    const float g = DtypeTraits<T>::to_float(g_row[i]);
    const float u = DtypeTraits<T>::to_float(u_row[i]);
    const float silu = g / (1.0f + expf(-g));
    local_max = fmaxf(local_max, fabsf(silu * u));
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  const float scale = (shared[0] > 0.0f) ? shared[0] / 127.0f : 0.0f;
  const float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
  if (tid == 0) {
    row_scales[row] = scale;
  }
  __syncthreads();

  for (int i = tid; i < cols; i += blockDim.x) {
    const float g = DtypeTraits<T>::to_float(g_row[i]);
    const float u = DtypeTraits<T>::to_float(u_row[i]);
    const float silu = g / (1.0f + expf(-g));
    const float scaled = (silu * u) * inv_scale;
    const int q = max(-127, min(127, __float2int_rn(scaled)));
    y[i] = static_cast<int8_t>(q);
  }
}

template <typename T>
cudaError_t SiluMulQuantizeRowsSymmetric(const T *gate, const T *up,
                                         int8_t *output, float *row_scales,
                                         int rows, int cols,
                                         cudaStream_t stream) {
  if (!gate || !up || !output || !row_scales || rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }

  int threads = min(256, cols);
  int t = 1;
  while (t < threads) {
    t <<= 1;
  }
  threads = t;
  const int smem = threads * sizeof(float);
  SiluMulQuantizeRowsSymmetricKernel<T>
      <<<rows, threads, smem, stream>>>(gate, up, output, row_scales, cols);
  return cudaGetLastError();
}

// ============================================================================
// Bias Add (templated)
// ============================================================================

template <typename T>
__global__ void BiasAddKernel(T *output, const T *bias, int total,
                              int bias_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  float val = DtypeTraits<T>::to_float(output[idx]);
  float b = DtypeTraits<T>::to_float(bias[idx % bias_dim]);
  output[idx] = DtypeTraits<T>::from_float(val + b);
}

template <typename T>
cudaError_t BiasAdd(T *output, const T *bias, int rows, int bias_dim,
                    cudaStream_t stream) {
  int total = rows * bias_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  BiasAddKernel<T>
      <<<blocks, threads, 0, stream>>>(output, bias, total, bias_dim);
  return cudaGetLastError();
}

// ============================================================================
// Mixed-precision kernels (FP32 residual + T weights/activations)
// ============================================================================

// RmsNormMixed: reads FP32 input, normalizes in FP32, writes T output.
template <typename T>
__global__ void RmsNormMixedKernel(const float *__restrict__ input,
                                   const T *__restrict__ weight,
                                   T *__restrict__ output, int hidden_size,
                                   float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float *x = input + row * hidden_size;
  T *y = output + row * hidden_size;

  extern __shared__ float shared[];

  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = x[i]; // already FP32
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = x[i] * rms * DtypeTraits<T>::to_float(weight[i]);
    y[i] = DtypeTraits<T>::from_float(val);
  }
}

template <typename T>
cudaError_t RmsNormMixed(const float *input, const T *weight, T *output,
                         int count, int hidden_size, float eps,
                         cudaStream_t stream) {
  int threads = min(1024, hidden_size);
  int t = 1;
  while (t < threads)
    t <<= 1;
  threads = t;
  int smem = threads * sizeof(float);
  RmsNormMixedKernel<T>
      <<<count, threads, smem, stream>>>(input, weight, output, hidden_size,
                                         eps);
  return cudaGetLastError();
}

// ResidualAddMixed: residual(FP32) += input(T).
template <typename T>
__global__ void ResidualAddMixedKernel(float *__restrict__ residual,
                                       const T *__restrict__ input, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  residual[idx] += DtypeTraits<T>::to_float(input[idx]);
}

template <typename T>
cudaError_t ResidualAddMixed(float *residual, const T *input, int count,
                             cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  ResidualAddMixedKernel<T>
      <<<blocks, threads, 0, stream>>>(residual, input, count);
  return cudaGetLastError();
}

// ResidualAddRmsNormMixed: residual(FP32) += input(T); output(T) =
// RmsNorm(residual).
template <typename T>
__global__ void
ResidualAddRmsNormMixedKernel(float *__restrict__ residual,
                              const T *__restrict__ input,
                              const T *__restrict__ weight,
                              T *__restrict__ output, int hidden_size,
                              float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float *res_row = residual + row * hidden_size;
  T *out_row = output + row * hidden_size;

  extern __shared__ float shared[];

  // Pass 1: residual += input (in FP32), compute sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = res_row[i] + DtypeTraits<T>::to_float(input[row * hidden_size + i]);
    res_row[i] = val; // stays FP32
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

  // Pass 2: normalize and write T output
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = res_row[i] * rms * DtypeTraits<T>::to_float(weight[i]);
    out_row[i] = DtypeTraits<T>::from_float(val);
  }
}

template <typename T>
cudaError_t ResidualAddRmsNormMixed(float *residual, const T *input,
                                    const T *weight, T *output, int count,
                                    int hidden_size, float eps,
                                    cudaStream_t stream) {
  int threads = min(1024, hidden_size);
  int t = 1;
  while (t < threads)
    t <<= 1;
  threads = t;
  int smem = threads * sizeof(float);
  ResidualAddRmsNormMixedKernel<T>
      <<<count, threads, smem, stream>>>(residual, input, weight, output,
                                         hidden_size, eps);
  return cudaGetLastError();
}

// EmbeddingLookupF32: look up embeddings from T table, write FP32 output.
template <typename T>
__global__ void EmbeddingLookupF32Kernel(const T *__restrict__ table,
                                         const int *__restrict__ token_ids,
                                         float *__restrict__ output,
                                         int seq_len, int hidden_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq_len * hidden_size;
  if (idx >= total)
    return;

  int pos = idx / hidden_size;
  int dim = idx % hidden_size;
  int token_id = token_ids[pos];

  output[idx] = DtypeTraits<T>::to_float(table[token_id * hidden_size + dim]);
}

template <typename T>
cudaError_t EmbeddingLookupF32(const T *table, const int *token_ids,
                               float *output, int seq_len, int hidden_size,
                               cudaStream_t stream) {
  int total = seq_len * hidden_size;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  EmbeddingLookupF32Kernel<T><<<blocks, threads, 0, stream>>>(
      table, token_ids, output, seq_len, hidden_size);
  return cudaGetLastError();
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template cudaError_t RmsNorm<half>(const half *, const half *, half *, int, int,
                                   float, cudaStream_t);
template cudaError_t RmsNorm<__nv_bfloat16>(const __nv_bfloat16 *,
                                            const __nv_bfloat16 *,
                                            __nv_bfloat16 *, int, int, float,
                                            cudaStream_t);

template cudaError_t RoPE<half>(half *, half *, int, int, int, int, int, float,
                                cudaStream_t, int);
template cudaError_t RoPE<__nv_bfloat16>(__nv_bfloat16 *, __nv_bfloat16 *, int,
                                         int, int, int, int, float,
                                         cudaStream_t, int);

template cudaError_t SiluMul<half>(const half *, const half *, half *, int,
                                   cudaStream_t);
template cudaError_t SiluMul<__nv_bfloat16>(const __nv_bfloat16 *,
                                            const __nv_bfloat16 *,
                                            __nv_bfloat16 *, int, cudaStream_t);

template cudaError_t ResidualAdd<half>(half *, const half *, int, cudaStream_t);
template cudaError_t ResidualAdd<__nv_bfloat16>(__nv_bfloat16 *,
                                                const __nv_bfloat16 *, int,
                                                cudaStream_t);

template cudaError_t ResidualAddRmsNorm<half>(half *, const half *,
                                              const half *, half *, int, int,
                                              float, cudaStream_t);
template cudaError_t
ResidualAddRmsNorm<__nv_bfloat16>(__nv_bfloat16 *, const __nv_bfloat16 *,
                                  const __nv_bfloat16 *, __nv_bfloat16 *, int,
                                  int, float, cudaStream_t);

template cudaError_t EmbeddingLookup<half>(const half *, const int *, half *,
                                           int, int, cudaStream_t);
template cudaError_t EmbeddingLookup<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const int *,
                                                    __nv_bfloat16 *, int, int,
                                                    cudaStream_t);

template cudaError_t HalfToFloat<half>(const half *, float *, int,
                                       cudaStream_t);
template cudaError_t HalfToFloat<__nv_bfloat16>(const __nv_bfloat16 *, float *,
                                                int, cudaStream_t);
template cudaError_t QuantizeRowsSymmetric<half>(const half *, int8_t *,
                                                 float *, int, int,
                                                 cudaStream_t);
template cudaError_t QuantizeRowsSymmetric<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          int8_t *, float *,
                                                          int, int,
                                                          cudaStream_t);
template cudaError_t SiluMulQuantizeRowsSymmetric<half>(const half *,
                                                        const half *, int8_t *,
                                                        float *, int, int,
                                                        cudaStream_t);
template cudaError_t
SiluMulQuantizeRowsSymmetric<__nv_bfloat16>(const __nv_bfloat16 *,
                                            const __nv_bfloat16 *, int8_t *,
                                            float *, int, int, cudaStream_t);

template cudaError_t BiasAdd<half>(half *, const half *, int, int,
                                   cudaStream_t);
template cudaError_t BiasAdd<__nv_bfloat16>(__nv_bfloat16 *,
                                            const __nv_bfloat16 *, int, int,
                                            cudaStream_t);

// ============================================================================
// Fused Triple Bias Add (Q/K/V in one launch)
// ============================================================================

template <typename T>
__global__ void BiasAddTripleKernel(T *__restrict__ q, T *__restrict__ k,
                                    T *__restrict__ v,
                                    const T *__restrict__ q_bias,
                                    const T *__restrict__ k_bias,
                                    const T *__restrict__ v_bias, int rows,
                                    int q_dim, int k_dim, int v_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_total = rows * q_dim;
  const int k_total = rows * k_dim;
  const int total = q_total + k_total + rows * v_dim;
  if (idx >= total)
    return;

  if (idx < q_total) {
    float val = DtypeTraits<T>::to_float(q[idx]);
    float b = DtypeTraits<T>::to_float(q_bias[idx % q_dim]);
    q[idx] = DtypeTraits<T>::from_float(val + b);
  } else if (idx < q_total + k_total) {
    const int ki = idx - q_total;
    float val = DtypeTraits<T>::to_float(k[ki]);
    float b = DtypeTraits<T>::to_float(k_bias[ki % k_dim]);
    k[ki] = DtypeTraits<T>::from_float(val + b);
  } else {
    const int vi = idx - q_total - k_total;
    float val = DtypeTraits<T>::to_float(v[vi]);
    float b = DtypeTraits<T>::to_float(v_bias[vi % v_dim]);
    v[vi] = DtypeTraits<T>::from_float(val + b);
  }
}

template <typename T>
cudaError_t BiasAddTriple(T *q, T *k, T *v, const T *q_bias, const T *k_bias,
                          const T *v_bias, int rows, int q_dim, int k_dim,
                          int v_dim, cudaStream_t stream) {
  const int total = rows * (q_dim + k_dim + v_dim);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  BiasAddTripleKernel<T><<<blocks, threads, 0, stream>>>(
      q, k, v, q_bias, k_bias, v_bias, rows, q_dim, k_dim, v_dim);
  return cudaGetLastError();
}

template cudaError_t BiasAddTriple<half>(half *, half *, half *, const half *,
                                         const half *, const half *, int, int,
                                         int, int, cudaStream_t);
template cudaError_t BiasAddTriple<__nv_bfloat16>(
    __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, const __nv_bfloat16 *,
    const __nv_bfloat16 *, const __nv_bfloat16 *, int, int, int, int,
    cudaStream_t);

// ============================================================================
// Non-templated backward-compatible overloads (delegate to half instantiation)
// ============================================================================

cudaError_t RmsNorm(const half *input, const half *weight, half *output,
                    int count, int hidden_size, float eps,
                    cudaStream_t stream) {
  return RmsNorm<half>(input, weight, output, count, hidden_size, eps, stream);
}

cudaError_t RoPE(half *q, half *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base,
                 cudaStream_t stream) {
  return RoPE<half>(q, k, seq_len, num_heads, num_kv_heads, head_dim, n_past,
                    freq_base, stream);
}

cudaError_t SiluMul(const half *gate, const half *up, half *output, int count,
                    cudaStream_t stream) {
  return SiluMul<half>(gate, up, output, count, stream);
}

cudaError_t ResidualAdd(half *residual, const half *input, int count,
                        cudaStream_t stream) {
  return ResidualAdd<half>(residual, input, count, stream);
}

cudaError_t ResidualAddRmsNorm(half *residual, const half *input,
                               const half *weight, half *output, int count,
                               int hidden_size, float eps,
                               cudaStream_t stream) {
  return ResidualAddRmsNorm<half>(residual, input, weight, output, count,
                                  hidden_size, eps, stream);
}

cudaError_t EmbeddingLookup(const half *table, const int *token_ids,
                            half *output, int seq_len, int hidden_size,
                            cudaStream_t stream) {
  return EmbeddingLookup<half>(table, token_ids, output, seq_len, hidden_size,
                               stream);
}

cudaError_t HalfToFloat(const half *input, float *output, int count,
                        cudaStream_t stream) {
  return HalfToFloat<half>(input, output, count, stream);
}

cudaError_t QuantizeRowsSymmetric(const half *input, int8_t *output,
                                  float *row_scales, int rows, int cols,
                                  cudaStream_t stream) {
  return QuantizeRowsSymmetric<half>(input, output, row_scales, rows, cols,
                                     stream);
}

cudaError_t SiluMulQuantizeRowsSymmetric(const half *gate, const half *up,
                                         int8_t *output, float *row_scales,
                                         int rows, int cols,
                                         cudaStream_t stream) {
  return SiluMulQuantizeRowsSymmetric<half>(gate, up, output, row_scales, rows,
                                            cols, stream);
}

// ============================================================================
// Batched RoPE: B sequences with different n_past values
// q layout: [B, num_heads * head_dim], k layout: [B, num_kv_heads * head_dim]
// ============================================================================

template <typename T>
__global__ void
BatchedRoPEKernel(T *__restrict__ q, T *__restrict__ k, int batch_size,
                  int num_heads, int num_kv_heads, int head_dim,
                  const int *__restrict__ d_n_past, float freq_base,
                  int rope_type, const T *__restrict__ q_bias,
                  const T *__restrict__ k_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim / 2;
  const int q_pairs_per_seq = num_heads * half_dim;
  const int k_pairs_per_seq = num_kv_heads * half_dim;
  const int pairs_per_seq = q_pairs_per_seq + k_pairs_per_seq;
  const int total_pairs = batch_size * pairs_per_seq;

  if (idx >= total_pairs)
    return;

  int b = idx / pairs_per_seq;
  int local_idx = idx % pairs_per_seq;
  bool is_q = (local_idx < q_pairs_per_seq);
  int pair_in_tensor = is_q ? local_idx : (local_idx - q_pairs_per_seq);
  int n_heads = is_q ? num_heads : num_kv_heads;

  int pair_idx = pair_in_tensor % half_dim;
  int head_idx = pair_in_tensor / half_dim;

  int position = d_n_past[b]; // Each sequence has its own position
  float freq =
      1.0f / powf(freq_base, 2.0f * pair_idx / static_cast<float>(head_dim));
  float angle = position * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  int base_offset = b * n_heads * head_dim + head_idx * head_dim;
  T *tensor = is_q ? q : k;
  // Adjust base for k: k has different stride per batch element
  if (!is_q) {
    base_offset = b * num_kv_heads * head_dim + head_idx * head_dim;
  }

  int i0, i1;
  if (rope_type == 2) {
    i0 = base_offset + pair_idx;
    i1 = base_offset + pair_idx + half_dim;
  } else {
    i0 = base_offset + 2 * pair_idx;
    i1 = base_offset + 2 * pair_idx + 1;
  }

  // Optional bias fold: adding Q/K bias at load reproduces the
  // BiasAdd-then-RoPE sequence with one fewer half-rounding step.
  const T *bias = is_q ? q_bias : k_bias;
  float v0 = DtypeTraits<T>::to_float(tensor[i0]);
  float v1 = DtypeTraits<T>::to_float(tensor[i1]);
  if (bias != nullptr) {
    const int bias_dim0 = i0 - base_offset;
    const int bias_dim1 = i1 - base_offset;
    const int bias_base = head_idx * head_dim;
    v0 += DtypeTraits<T>::to_float(bias[bias_base + bias_dim0]);
    v1 += DtypeTraits<T>::to_float(bias[bias_base + bias_dim1]);
  }

  tensor[i0] = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
  tensor[i1] = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);
}

template <typename T>
cudaError_t BatchedRoPE(T *q, T *k, int batch_size, int num_heads,
                        int num_kv_heads, int head_dim, const int *d_n_past,
                        float freq_base, cudaStream_t stream, int rope_type,
                        const T *q_bias, const T *k_bias) {
  int half_dim = head_dim / 2;
  int pairs_per_seq = num_heads * half_dim + num_kv_heads * half_dim;
  int total_pairs = batch_size * pairs_per_seq;
  int threads = 256;
  int blocks = (total_pairs + threads - 1) / threads;

  BatchedRoPEKernel<T><<<blocks, threads, 0, stream>>>(
      q, k, batch_size, num_heads, num_kv_heads, head_dim, d_n_past, freq_base,
      rope_type, q_bias, k_bias);
  return cudaGetLastError();
}

template cudaError_t BatchedRoPE<half>(half *, half *, int, int, int, int,
                                       const int *, float, cudaStream_t, int,
                                       const half *, const half *);
template cudaError_t
BatchedRoPE<__nv_bfloat16>(__nv_bfloat16 *, __nv_bfloat16 *, int, int, int, int,
                           const int *, float, cudaStream_t, int,
                           const __nv_bfloat16 *, const __nv_bfloat16 *);

// ============================================================================
// Batched KV Append: scatter-copy K/V for B sequences
// k_new/v_new layout: [B, kv_dim]
// d_k_dst/d_v_dst: [B] device pointers to destination rows
// ============================================================================

template <typename T>
__global__ void BatchedKvAppendKernel(const T *__restrict__ k_new,
                                      const T *__restrict__ v_new,
                                      T *const *__restrict__ d_k_dst,
                                      T *const *__restrict__ d_v_dst,
                                      int batch_size, int kv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * kv_dim;
  if (idx >= total)
    return;

  int b = idx / kv_dim;
  int d = idx % kv_dim;

  d_k_dst[b][d] = k_new[b * kv_dim + d];
  d_v_dst[b][d] = v_new[b * kv_dim + d];
}

template <typename T>
cudaError_t BatchedKvAppend(const T *k_new, const T *v_new, T **d_k_dst,
                            T **d_v_dst, int batch_size, int kv_dim,
                            cudaStream_t stream) {
  int total = batch_size * kv_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  BatchedKvAppendKernel<T><<<blocks, threads, 0, stream>>>(
      k_new, v_new, d_k_dst, d_v_dst, batch_size, kv_dim);
  return cudaGetLastError();
}

template cudaError_t BatchedKvAppend<half>(const half *, const half *, half **,
                                           half **, int, int, cudaStream_t);
template cudaError_t BatchedKvAppend<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 **,
                                                    __nv_bfloat16 **, int, int,
                                                    cudaStream_t);

template <typename T>
__global__ void BatchedKvAppendStridedKernel(
    const T *__restrict__ k_new, const T *__restrict__ v_new,
    T *__restrict__ kv_buffer, const int *__restrict__ d_seq_ids,
    const int *__restrict__ d_n_past, int layer, int batch_size, int kv_dim,
    size_t slot_stride, size_t layer_stride, size_t kv_stride,
    const T *__restrict__ v_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * kv_dim;
  if (idx >= total) {
    return;
  }

  const int b = idx / kv_dim;
  const int d = idx % kv_dim;
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const size_t seq_offset = static_cast<size_t>(d_seq_ids[b]) * slot_stride;
  const size_t token_offset =
      static_cast<size_t>(d_n_past[b]) * kv_dim + static_cast<size_t>(d);
  const size_t k_offset = seq_offset + layer_offset + token_offset;
  const size_t v_offset = seq_offset + layer_offset + kv_stride + token_offset;

  kv_buffer[k_offset] = k_new[idx];
  // Optional V bias fold (V feeds only the cache, so adding at the store
  // reproduces BiasAdd-then-append exactly).
  if (v_bias != nullptr) {
    kv_buffer[v_offset] =
        DtypeTraits<T>::from_float(DtypeTraits<T>::to_float(v_new[idx]) +
                                   DtypeTraits<T>::to_float(v_bias[d]));
  } else {
    kv_buffer[v_offset] = v_new[idx];
  }
}

template <typename T>
cudaError_t BatchedKvAppendStrided(const T *k_new, const T *v_new, T *kv_buffer,
                                   const int *d_seq_ids, const int *d_n_past,
                                   int layer, int batch_size, int kv_dim,
                                   size_t slot_stride, size_t layer_stride,
                                   size_t kv_stride, cudaStream_t stream,
                                   const T *v_bias) {
  const int total = batch_size * kv_dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  BatchedKvAppendStridedKernel<T><<<blocks, threads, 0, stream>>>(
      k_new, v_new, kv_buffer, d_seq_ids, d_n_past, layer, batch_size, kv_dim,
      slot_stride, layer_stride, kv_stride, v_bias);
  return cudaGetLastError();
}

template cudaError_t BatchedKvAppendStrided<half>(const half *, const half *,
                                                  half *, const int *,
                                                  const int *, int, int, int,
                                                  size_t, size_t, size_t,
                                                  cudaStream_t, const half *);
template cudaError_t BatchedKvAppendStrided<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, const int *,
    const int *, int, int, int, size_t, size_t, size_t, cudaStream_t,
    const __nv_bfloat16 *);

// ============================================================================
// BatchedKvAppendIndirect: scatter-copy K/V using slot base pointer table
// ============================================================================

template <typename T>
__global__ void BatchedKvAppendIndirectKernel(
    const T *__restrict__ k_new, const T *__restrict__ v_new,
    T *const *__restrict__ slot_base_ptrs,
    const int *__restrict__ d_seq_ids, const int *__restrict__ d_n_past,
    int layer, int batch_size, int kv_dim, size_t layer_stride,
    size_t kv_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * kv_dim;
  if (idx >= total)
    return;

  const int b = idx / kv_dim;
  const int d = idx % kv_dim;
  T *slot = slot_base_ptrs[d_seq_ids[b]];
  const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
  const size_t token_offset =
      static_cast<size_t>(d_n_past[b]) * kv_dim + static_cast<size_t>(d);
  const size_t k_offset = layer_offset + token_offset;
  const size_t v_offset = layer_offset + kv_stride + token_offset;

  slot[k_offset] = k_new[idx];
  slot[v_offset] = v_new[idx];
}

template <typename T>
cudaError_t BatchedKvAppendIndirect(const T *k_new, const T *v_new,
                                    T *const *slot_base_ptrs,
                                    const int *d_seq_ids, const int *d_n_past,
                                    int layer, int batch_size, int kv_dim,
                                    size_t layer_stride, size_t kv_stride,
                                    cudaStream_t stream) {
  const int total = batch_size * kv_dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  BatchedKvAppendIndirectKernel<T>
      <<<blocks, threads, 0, stream>>>(k_new, v_new, slot_base_ptrs, d_seq_ids,
                                       d_n_past, layer, batch_size, kv_dim,
                                       layer_stride, kv_stride);
  return cudaGetLastError();
}

template cudaError_t BatchedKvAppendIndirect<half>(
    const half *, const half *, half *const *, const int *, const int *, int,
    int, int, size_t, size_t, cudaStream_t);
template cudaError_t BatchedKvAppendIndirect<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *const *,
    const int *, const int *, int, int, int, size_t, size_t, cudaStream_t);

// ============================================================================
// Mean-pooling kernel for embedding extraction
// ============================================================================

template <typename T>
__global__ void MeanPoolKernel(const T *__restrict__ input,
                               float *__restrict__ output, int seq_len,
                               int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;
  float sum = 0.0f;
  for (int t = 0; t < seq_len; ++t) {
    sum += __half2float(input[t * hidden_size + idx]);
  }
  output[idx] = sum / static_cast<float>(seq_len);
}

// BF16 specialization
template <>
__global__ void
MeanPoolKernel<__nv_bfloat16>(const __nv_bfloat16 *__restrict__ input,
                              float *__restrict__ output, int seq_len,
                              int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;
  float sum = 0.0f;
  for (int t = 0; t < seq_len; ++t) {
    sum += __bfloat162float(input[t * hidden_size + idx]);
  }
  output[idx] = sum / static_cast<float>(seq_len);
}

template <typename T>
cudaError_t MeanPool(const T *input, float *output, int seq_len,
                     int hidden_size, cudaStream_t stream) {
  if (seq_len <= 0 || hidden_size <= 0) {
    return cudaSuccess;
  }
  int threads = 256;
  int blocks = (hidden_size + threads - 1) / threads;
  MeanPoolKernel<T>
      <<<blocks, threads, 0, stream>>>(input, output, seq_len, hidden_size);
  return cudaGetLastError();
}

template cudaError_t MeanPool<half>(const half *, float *, int, int,
                                    cudaStream_t);
template cudaError_t MeanPool<__nv_bfloat16>(const __nv_bfloat16 *, float *,
                                             int, int, cudaStream_t);

// Mixed-precision instantiations
template cudaError_t RmsNormMixed<half>(const float *, const half *, half *,
                                        int, int, float, cudaStream_t);
template cudaError_t RmsNormMixed<__nv_bfloat16>(const float *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *, int, int,
                                                  float, cudaStream_t);
template cudaError_t ResidualAddMixed<half>(float *, const half *, int,
                                            cudaStream_t);
template cudaError_t ResidualAddMixed<__nv_bfloat16>(float *,
                                                      const __nv_bfloat16 *,
                                                      int, cudaStream_t);
template cudaError_t ResidualAddRmsNormMixed<half>(float *, const half *,
                                                    const half *, half *, int,
                                                    int, float, cudaStream_t);
template cudaError_t ResidualAddRmsNormMixed<__nv_bfloat16>(
    float *, const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *,
    int, int, float, cudaStream_t);
template cudaError_t EmbeddingLookupF32<half>(const half *, const int *,
                                               float *, int, int,
                                               cudaStream_t);
template cudaError_t EmbeddingLookupF32<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const int *, float *,
                                                        int, int, cudaStream_t);

// ============================================================================
// Device-side token relay for zero-copy decode loop
// ============================================================================

__global__ void DeviceTokenRelayKernel(const int *__restrict__ sampled_tokens,
                                       int *__restrict__ batch_meta,
                                       int batch_size, int max_batch_size) {
  const int b = threadIdx.x;
  if (b >= batch_size)
    return;

  // batch_meta layout: [token_ids(max_B)][n_past(max_B)][seq_ids(max_B)][kv_lens(max_B)]
  int *token_ids = batch_meta;
  int *n_past = batch_meta + max_batch_size;
  // seq_ids stays unchanged (same sequences)
  int *kv_lens = batch_meta + max_batch_size * 3;

  // Copy sampled token to graph input buffer
  token_ids[b] = sampled_tokens[b];
  // Increment position for next forward pass
  n_past[b] += 1;
  // kv_lens = n_past + 1
  kv_lens[b] = n_past[b] + 1;
}

cudaError_t DeviceTokenRelay(const int *sampled_tokens, int *batch_meta,
                              int batch_size, int max_batch_size,
                              cudaStream_t stream) {
  if (batch_size <= 0)
    return cudaSuccess;
  // Single block, B threads — tiny kernel, runs in <1us
  DeviceTokenRelayKernel<<<1, batch_size, 0, stream>>>(
      sampled_tokens, batch_meta, batch_size, max_batch_size);
  return cudaGetLastError();
}

__global__ void AppendTokenToBufferKernel(const int *__restrict__ src,
                                          int *__restrict__ dst, int pos) {
  if (threadIdx.x == 0) {
    dst[pos] = src[0];
  }
}

cudaError_t AppendTokenToBuffer(const int *sampled_token, int *token_buffer,
                                 int position, cudaStream_t stream) {
  AppendTokenToBufferKernel<<<1, 1, 0, stream>>>(sampled_token, token_buffer,
                                                   position);
  return cudaGetLastError();
}

__global__ void DeviceCheckEosKernel(const int *__restrict__ sampled_tokens,
                                     int batch_size, int eos_token_id,
                                     int *__restrict__ d_has_eos) {
  const int b = threadIdx.x;
  if (b >= batch_size)
    return;
  if (sampled_tokens[b] == eos_token_id) {
    atomicOr(d_has_eos, 1);
  }
}

cudaError_t DeviceCheckEos(const int *sampled_tokens, int batch_size,
                            int eos_token_id, int *d_has_eos,
                            cudaStream_t stream) {
  if (batch_size <= 0)
    return cudaSuccess;
  // Reset flag, then check
  cudaMemsetAsync(d_has_eos, 0, sizeof(int), stream);
  DeviceCheckEosKernel<<<1, batch_size, 0, stream>>>(
      sampled_tokens, batch_size, eos_token_id, d_has_eos);
  return cudaGetLastError();
}

// Convert half/bfloat16 to float kernel
template <typename T>
__global__ void ConvertHalfToFloatKernel(const T *__restrict__ input,
                                         float *__restrict__ output,
                                         size_t num_elements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if constexpr (std::is_same_v<T, half>) {
      output[idx] = __half2float(input[idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      output[idx] = __bfloat162float(input[idx]);
    }
  }
}

template <typename T>
cudaError_t ConvertHalfToFloat(const T *d_input, float *d_output,
                                size_t num_elements, cudaStream_t stream) {
  if (num_elements == 0) {
    return cudaSuccess;
  }

  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  ConvertHalfToFloatKernel<T><<<num_blocks, block_size, 0, stream>>>(
      d_input, d_output, num_elements);

  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t ConvertHalfToFloat<half>(const half *, float *, size_t,
                                                cudaStream_t);
template cudaError_t ConvertHalfToFloat<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         float *, size_t,
                                                         cudaStream_t);

} // namespace cuda_kernel
} // namespace inferflux
