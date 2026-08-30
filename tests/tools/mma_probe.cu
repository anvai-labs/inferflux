// Standalone: launch the MMA kernel with printf diagnostics.
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace inferflux::runtime::cuda::native;

__global__ void ProbeVecDot(const int *x_tile, const int *y_tile, float *out) {
  float sum[16 * 128 / (8 * 32)] = {0};
  // emulate one warp's work with threadIdx = flat 256-thread layout
  VecDotQ6KQ8_1Mma<16>(x_tile, y_tile, sum, 0);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    out[0] = sum[0];
    out[1] = sum[1];
  }
}

int main() {
  cudaFree(0); // init context
  const int N = 256, K = 256, M = 1;
  std::vector<block_q6_k> w(static_cast<size_t>(N) * (K / 256));
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i)
      b.ql[i] = (i * 7) & 0xFF;
    for (int i = 0; i < 64; ++i)
      b.qh[i] = (i * 11) & 0xFF;
    for (int i = 0; i < 16; ++i)
      b.scales[i] = 1; // scale 1
    const half d = __float2half(0.01f);
    std::memcpy(&b.d, &d, 2);
  }
  std::vector<half> acts(K);
  for (int i = 0; i < K; ++i)
    acts[i] = __float2half(0.001f * i);
  std::vector<BlockQ8_1Mmq> a(M * (K / 128));
  for (int sub = 0; sub < 4; ++sub) {
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::fmax(amax, std::fabs(__half2float(acts[sub * 32 + i])));
    const float d_inv = 127.0f / amax;
    a[0].d4[sub] = amax / 127.0f;
    for (int i = 0; i < 32; ++i)
      a[0].qs[sub * 32 + i] =
          (int8_t)std::lround(__half2float(acts[sub * 32 + i]) * d_inv);
  }
  block_q6_k *d_w;
  BlockQ8_1Mmq *d_a;
  half *d_out;
  cudaMalloc(&d_w, w.size() * sizeof(block_q6_k));
  cudaMalloc(&d_a, a.size() * sizeof(BlockQ8_1Mmq));
  cudaMalloc(&d_out, M * N * sizeof(half));
  cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q6_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a.data(), a.size() * sizeof(BlockQ8_1Mmq),
             cudaMemcpyHostToDevice);
  dim3 grid((N + kMmqY - 1) / kMmqY, (M + 15) / 16);
  const size_t smem =
      (16 * kMmqTileYK + kMmqY * kMmqMmaTileXKQ6K) * sizeof(int);
  printf("grid=(%d,%d) smem=%zu\n", grid.x, grid.y, smem);
  auto err = cudaFuncSetAttribute(InferfluxMmqQ6KMma<16>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smem);
  printf("setattr: %s\n", cudaGetErrorString(err));
  InferfluxMmqQ6KMma<16>
      <<<grid, 256, smem>>>((const char *)d_w, d_a, d_out, N, K, M);
  err = cudaGetLastError();
  printf("launch: %s\n", cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  printf("sync: %s\n", cudaGetErrorString(err));
  std::vector<half> out(M * N);
  cudaMemcpy(out.data(), d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost);
  // FPU ref
  double ref0 = 0;
  for (int blk = 0; blk < K / 256; ++blk)
    for (int e = 0; e < 256; ++e) {
      const auto &b = w[blk];
      const int lo = b.ql[e / 2] & 0xF, hi = b.ql[e / 2] >> 4;
      const int l = (e % 2 == 0) ? lo : hi, h = (b.qh[e] & 3);
      const int q = ((l | (h << 4)) - 32);
      const int k = blk * 256 + e;
      ref0 += 0.01 * 1.0 * q * (a[0].d4[k / 32]) * a[0].qs[k % 128];
    }
  printf("out[0]=%.6f ref[0]=%.6f rel=%.3e\n", __half2float(out[0]), ref0,
         std::fabs((__half2float(out[0]) - ref0) / ref0));
  return 0;
}
