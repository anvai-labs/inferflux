// Decisive pairing test: uniform weights (all quant bytes = 0x21 → +0x21-0x20
// = +1 low / +2 high after pairing, scales all 1, d such that dequant value
// is exactly representable) and uniform activations (all quant = 3, d chosen).
// If pairing is right, out = N_vals * w_val * a_val exactly.
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace inferflux::runtime::cuda::native;

int main() {
  cudaFree(0);
  const int N = 128, K = 256, M = 16;
  std::vector<block_q6_k> w(N * (K / 256));
  for (auto &b : w) {
    std::memset(b.ql, 0, 128); // low nibbles 0
    std::memset(b.qh, 0, 64);
    for (int i = 0; i < 16; ++i)
      b.scales[i] = 1;
    const half d = __float2half(1.0f);
    std::memcpy(&b.d, &d, 2);
  }
  // weight value = (0|0)-32 = -32 everywhere, scale 1, d 1 → w = -32.
  std::vector<half> acts(M * K, __float2half(1.0f));
  std::vector<BlockQ8_1Mmq> a(M * (K / 128));
  for (auto &g : a) {
    for (int s = 0; s < 4; ++s)
      g.d4[s] = 1.0f / 127.0f;
    for (int i = 0; i < 128; ++i)
      g.qs[i] = 127;
  }
  // a value = 127 * (1/127) = 1.
  block_q6_k *d_w;
  BlockQ8_1Mmq *d_a;
  half *d_out;
  cudaMalloc(&d_w, w.size() * sizeof(block_q6_k));
  cudaMalloc(&d_a, a.size() * sizeof(BlockQ8_1Mmq));
  cudaMalloc(&d_out, (size_t)M * N * sizeof(half));
  cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q6_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a.data(), a.size() * sizeof(BlockQ8_1Mmq),
             cudaMemcpyHostToDevice);
  dim3 grid((N + kMmqY - 1) / kMmqY, (M + 15) / 16);
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(InferfluxMmqQ6KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  InferfluxMmqQ6KMma<16>
      <<<grid, 256, smem>>>((const char *)d_w, d_a, d_out, N, K, M);
  printf("sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  std::vector<half> out(M * N);
  cudaMemcpy(out.data(), d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost);
  const double expect = -32.0 * 1.0 * K; // sum of w*a over K
  printf("out[0]=%.2f expect=%.2f ratio=%.4f\n", __half2float(out[0]), expect,
         __half2float(out[0]) / expect);
  {
    int exact = 0, zero = 0, other = 0;
    double first_other = 0;
    for (size_t idx = 0; idx < out.size(); ++idx) {
      const float v = __half2float(out[idx]);
      if (v == -4096.0f)
        ++exact;
      else if (v == 0.0f)
        ++zero;
      else {
        ++other;
        if (!first_other)
          first_other = v;
      }
    }
    printf("hist: exact=%d zero=%d other=%d (first other=%.1f) of %zu\n", exact,
           zero, other, first_other, out.size());
  }
  return 0;
}
