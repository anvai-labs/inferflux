// Standalone spike: bf16 skinny GEMV/GEMM (M <= 16) vs cuBLAS on the exact
// Qwen2.5-3B safetensors projection shapes, measured under COLD L2 (a 96 MB
// scratch is swept between iterations, mirroring how attention kernels evict
// L2 between projections in the real decode loop). Warm-L2 numbers are
// misleading: back-to-back GEMMs can exceed 100% of DRAM roofline from L2
// hits on the smaller shapes.
//
// Findings (RTX 4000 Ada, Sep 6 2026): cuBLAS lands at 30-44% of roofline
// on these shapes cold; the naive warp-per-row custom kernel here only
// matches it (M=1-4) and loses at larger M. Beating cuBLAS on skinny bf16
// GEMV needs a heavier design (multi-row tiles, cp.async double buffering,
// deep MLP) -- see docs/design/SAFETENSORS_DECODE_PERFORMANCE_PLAN.md.
//
// Build:  nvcc -O3 -arch=native -lcublas bf16_gemv_bench.cu -o bf16_gemv_bench
// Run:    ./bf16_gemv_bench
//
// Not wired into CMake: spike tool, no server integration yet.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x)                                                           \
  do {                                                                     \
    cudaError_t e = (x);                                                   \
    if (e != cudaSuccess) {                                                \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,  \
             __LINE__);                                                    \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(x)                                                    \
  do {                                                                     \
    cublasStatus_t s = (x);                                                \
    if (s != CUBLAS_STATUS_SUCCESS) {                                      \
      printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__);    \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Custom kernel: weight-read-first skinny GEMV, M <= kMaxM.
//
// Shape: y[M, N] = x[M, K] * W^T  where W is [N, K] row-major (TN GEMM),
// i.e. each output row n reads the contiguous row W[n, :].
//
// One warp computes one (or a few) output row(s) for all M inputs: the row's
// K elements are streamed once with 16-byte vector loads and each lane keeps
// kMaxM accumulators in registers, so W traffic is independent of M. Grid is
// sized to flood all SMs (rows/warp tuned so total blocks >= ~4x SM count).
// ---------------------------------------------------------------------------

constexpr int kMaxM = 16;

// One warp per output row. Lane l handles elements l*8 .. l*8+7 of each
// 128-byte segment (8 bf16), striding warp_width*8 elements per iteration.
__global__ void __launch_bounds__(256) GemvBf16WideK(
    const __nv_bfloat16 *__restrict__ w,  // [N, K] row-major
    const __nv_bfloat16 *__restrict__ x,  // [M, K] row-major
    float *__restrict__ y,                // [M, N] row-major
    int N, int K, int M) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int n_rows = (gridDim.x * blockDim.x) >> 5;
  const int warp_inc = (gridDim.x * blockDim.x) >> 5;

  const int vec_per_row = K >> 3;  // 8 bf16 per 16B load

  for (int row = warp; row < N; row += warp_inc) {
    const __nv_bfloat16 *w_row = w + static_cast<long>(row) * K;
    float acc[kMaxM];
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) acc[m] = 0.f;

    const int4 *w_vec = reinterpret_cast<const int4 *>(w_row);
    for (int v = lane; v < vec_per_row; v += 32) {
      int4 wv = __ldg(w_vec + v);
      const __nv_bfloat16 *wh = reinterpret_cast<const __nv_bfloat16 *>(&wv);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const __nv_bfloat16 *x_row = x + static_cast<long>(m) * K;
        // 8 products; compiler pairs these into HFMA2 on the packed halves.
        float p0 = __bfloat162float(wh[0]) * __bfloat162float(x_row[v * 8 + 0]);
        float p1 = __bfloat162float(wh[1]) * __bfloat162float(x_row[v * 8 + 1]);
        float p2 = __bfloat162float(wh[2]) * __bfloat162float(x_row[v * 8 + 2]);
        float p3 = __bfloat162float(wh[3]) * __bfloat162float(x_row[v * 8 + 3]);
        float p4 = __bfloat162float(wh[4]) * __bfloat162float(x_row[v * 8 + 4]);
        float p5 = __bfloat162float(wh[5]) * __bfloat162float(x_row[v * 8 + 5]);
        float p6 = __bfloat162float(wh[6]) * __bfloat162float(x_row[v * 8 + 6]);
        float p7 = __bfloat162float(wh[7]) * __bfloat162float(x_row[v * 8 + 7]);
        acc[m] += (p0 + p1) + (p2 + p3) + (p4 + p5) + (p6 + p7);
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
      if (lane == 0)
        y[static_cast<long>(m) * N + row] = acc[m];
    }
  }
}


// ---------------------------------------------------------------------------
// Reference + harness
// ---------------------------------------------------------------------------

static void CublasGemmTN(cublasHandle_t h, int M, int N, int K,
                         const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                         float *y) {
  // cuBLAS is column-major; compute y^T[N, M] = W^T[N, K] * x^T[K, M].
  // Row-major W[N,K] == column-major W_cm[K,N] with W_cm^T == W...
  // Standard trick: row-major y[M,N] = x[M,K] * W[N,K]^T is
  // column-major y_cm[N,M] = W_cm[N,K] * x_cm[K,M], i.e. gemm(op(A)=N on W
  // viewed as [N,K] col-major? no) -- use op(B)=T form:
  // y_cm[N,M] = W_rowmajor_as_cm[K,N]^T * x_rowmajor_as_cm[K,M]
  float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x,
      CUDA_R_16BF, K, &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

struct Shape {
  const char *name;
  int N, K;
  long weight_bytes() const { return (long)N * K * 2; }
};

int main() {
  Shape shapes[] = {
      {"gate/up", 11008, 2048}, {"down", 2048, 11008},
      {"qkv", 2560, 2048},      {"o_proj", 2048, 2048},
      {"lm_head", 151936, 2048},
  };
  int Ms[] = {1, 2, 3, 4, 8, 16};

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s, SMs=%d, peak BW ~%.0f GB/s\n\n", prop.name,
         prop.multiProcessorCount, 576.0);

  for (const auto &shape : shapes) {
    for (int M : Ms) {
      size_t wn = (size_t)shape.N * shape.K, xn = (size_t)M * shape.K,
             yn = (size_t)M * shape.N;
      __nv_bfloat16 *d_w, *d_x;
      float *d_y_cublas, *d_y_custom;
      CHECK(cudaMalloc(&d_w, wn * 2));
      CHECK(cudaMalloc(&d_x, xn * 2));
      CHECK(cudaMalloc(&d_y_cublas, yn * 4));
      CHECK(cudaMalloc(&d_y_custom, yn * 4));

      std::mt19937 rng(42);
      std::uniform_real_distribution<float> dist(-1.f, 1.f);
      std::vector<__nv_bfloat16> h_w(wn), h_x(xn);
      for (auto &v : h_w) v = __float2bfloat16(dist(rng));
      for (auto &v : h_x) v = __float2bfloat16(dist(rng));
      CHECK(
          cudaMemcpy(d_w, h_w.data(), wn * 2, cudaMemcpyHostToDevice));
      CHECK(
          cudaMemcpy(d_x, h_x.data(), xn * 2, cudaMemcpyHostToDevice));

      // Custom kernel launch geometry: warp handles one row; total warps
      // ~4x what's needed so every SM is saturated; rows strided.
      int warps_wanted = shape.N;  // one warp per row upper bound
      int block = 256;
      int grid = std::min((warps_wanted * 32 + block - 1) / block,
                          prop.multiProcessorCount * 8);
      grid = std::max(grid, 1);

      // 96 MB scratch, swept between iterations to evict L2 (Ada L2=48MB).
      float *d_evict;
      const size_t evict_n = 96u * 1024 * 1024 / sizeof(float);
      CHECK(cudaMalloc(&d_evict, evict_n * sizeof(float)));
      CHECK(cudaMemset(d_evict, 1, evict_n * sizeof(float)));

      cudaEvent_t t0, t1;
      CHECK(cudaEventCreate(&t0));
      CHECK(cudaEventCreate(&t1));
      const int iters = 20;
      float ms_cublas = 0, ms_custom = 0;

      // cuBLAS timing
      float alpha = 1.f, beta = 0.f;
      CUBLAS_CHECK(cublasSetStream(handle, 0));
      CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.N, M, shape.K,
                                &alpha, d_w, CUDA_R_16BF, shape.K, d_x,
                                CUDA_R_16BF, shape.K, &beta, d_y_cublas,
                                CUDA_R_32F, shape.N, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));
      // Evict L2 between iterations (attention kernels run between
      // projections in the real decode loop) but time only the GEMM window.
      ms_cublas = 0;
      for (int i = 0; i < iters; ++i) {
        CHECK(cudaMemsetAsync(d_evict, i & 0xFF, evict_n * sizeof(float)));
        CHECK(cudaEventRecord(t0));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.N, M,
                                  shape.K, &alpha, d_w, CUDA_R_16BF, shape.K, d_x,
                                  CUDA_R_16BF, shape.K, &beta, d_y_cublas,
                                  CUDA_R_32F, shape.N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms_i = 0;
        CHECK(cudaEventElapsedTime(&ms_i, t0, t1));
        ms_cublas += ms_i;
      }

      // Custom timing
      GemvBf16WideK<<<grid, block>>>(d_w, d_x, d_y_custom, shape.N, shape.K, M);
      CHECK(cudaGetLastError());
      ms_custom = 0;
      for (int i = 0; i < iters; ++i) {
        CHECK(cudaMemsetAsync(d_evict, (i + 1) & 0xFF,
                              evict_n * sizeof(float)));
        CHECK(cudaEventRecord(t0));
        GemvBf16WideK<<<grid, block>>>(d_w, d_x, d_y_custom, shape.N, shape.K,
                                       M);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms_i = 0;
        CHECK(cudaEventElapsedTime(&ms_i, t0, t1));
        ms_custom += ms_i;
      }

      ms_cublas /= iters;
      ms_custom /= iters;

      // Correctness (coarse): max abs diff on a sample of outputs.
      std::vector<float> ya(yn), yb(yn);
      CHECK(cudaMemcpy(ya.data(), d_y_cublas, yn * 4, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(yb.data(), d_y_custom, yn * 4, cudaMemcpyDeviceToHost));
      double max_diff = 0;
      size_t stride = std::max<size_t>(1, yn / 4096);
      for (size_t i = 0; i < yn; i += stride) {
        double d = std::abs((double)ya[i] - (double)yb[i]);
        if (d > max_diff) max_diff = d;
      }

      double gbs = shape.weight_bytes() / 1e9;
      printf(
          "%-8s M=%2d | cublas %7.3f ms (%3.0f%% roofline) | warp/row "
          "%7.3f ms (%3.0f%%) | speedup %4.2fx | maxdiff %.3f\n",
          shape.name, M, ms_cublas,
          100.0 * (gbs / (ms_cublas / 1e3)) / 576.0, ms_custom,
          100.0 * (gbs / (ms_custom / 1e3)) / 576.0, ms_cublas / ms_custom,
          max_diff);

      CHECK(cudaEventDestroy(t0));
      cudaFree(d_evict);
      CHECK(cudaEventDestroy(t1));
      cudaFree(d_w);
      cudaFree(d_x);
      cudaFree(d_y_cublas);
      cudaFree(d_y_custom);
    }
    printf("\n");
  }
  cublasDestroy(handle);
  return 0;
}
