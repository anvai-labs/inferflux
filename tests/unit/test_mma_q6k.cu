/**
 * Correctness tests for the MMA (int8 tensor-core) Q6_K MMQ kernel.
 *
 * Gates:
 *   1. Uniform-value exactness: constant weights/activations must give
 *      exactly N * w * a in every output (canary for fragment pairing
 *      bugs — permutation errors are invisible to all-ones identity
 *      checks but not to non-uniform constants).
 *   2. Device D4 quantizer (QuantizeRowQ8_1MmqKernel) == host emulation,
 *      byte-exact.
 *   3. Kernel output == double-precision FPU reference over the same
 *      quantized activations, rel tol 1e-2 (fp16 output rounding
 *      dominates), across shapes including tail-N tiles and K-split.
 *
 * Build (standalone, matches the other tests/unit CUDA binaries):
 *   nvcc -std=c++17 -I<repo> -I<repo>/external -o test_mma_q6k \
 *       tests/unit/test_mma_q6k.cu
 */

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace inferflux::runtime::cuda::native;

namespace {

int g_failures = 0;

uint32_t Lcg(uint32_t &s) { return s = s * 1664525u + 1013904223u; }

// Q6_K element map (validated against dequant_q6k_element in
// quant_common.cuh): two 128-value groups of four 32-value sub-chunks.
// Sub 0/1 read low nibbles, sub 2/3 high; qh bits sub*2; scale index
// g*8 + sub*2 + l/16.
double Q6KValue(const block_q6_k &b, int e) {
  const int g = e / 128;
  const int sub = (e % 128) / 32;
  const int l = e % 32;
  const int ql_idx = g * 64 + ((sub & 1) ? 32 : 0) + l;
  const int ql_val = (sub >= 2) ? (b.ql[ql_idx] >> 4) : (b.ql[ql_idx] & 0xF);
  const int qh_val = (b.qh[g * 32 + l] >> (sub * 2)) & 0x03;
  const int q = (ql_val | (qh_val << 4)) - 32;
  const int scale_idx = g * 8 + sub * 2 + l / 16;
  return __half2float(*reinterpret_cast<const half *>(&b.d)) *
         static_cast<int8_t>(b.scales[scale_idx]) * q;
}

// Host emulation of the D4 quantizer, matching device float ops exactly.
void QuantizeHost(const std::vector<half> &x, int K,
                  std::vector<BlockQ8_1Mmq> &row_major, int row) {
  const int groups = K / 128;
  for (int g = 0; g < groups; ++g) {
    BlockQ8_1Mmq &grp = row_major[static_cast<size_t>(row) * groups + g];
    for (int sub = 0; sub < 4; ++sub) {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float v =
            __half2float(x[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        amax = fmaxf(amax, fabsf(v));
      }
      const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      grp.d4[sub] = amax > 0.0f ? 1.0f / d_inv : 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float v =
            __half2float(x[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        grp.qs[sub * 32 + i] = static_cast<int8_t>(nearbyintf(v * d_inv));
      }
    }
  }
}

void Report(const char *name, bool ok, double detail) {
  printf("%-52s %s (%.3e)\n", name, ok ? "PASS" : "FAIL", detail);
  if (!ok) {
    ++g_failures;
  }
}

struct Buffers {
  block_q6_k *w = nullptr;
  BlockQ8_1Mmq *a = nullptr;
  half *acts = nullptr;
  half *out = nullptr;
  float *partials = nullptr;
  size_t mn = 0;
  cudaStream_t s;
};

// Runs quantizer + kernel (with optional K-split) and returns the output.
std::vector<half> RunKernel(const std::vector<block_q6_k> &w,
                            const std::vector<half> &acts,
                            const std::vector<BlockQ8_1Mmq> &row_major, int M,
                            int N, int K, int ksplits, Buffers &buf) {
  const int groups = K / 128;
  // Upload activations group-major (kernel layout).
  std::vector<BlockQ8_1Mmq> gm(row_major.size());
  for (int r = 0; r < M; ++r)
    for (int g = 0; g < groups; ++g)
      gm[static_cast<size_t>(g) * M + r] =
          row_major[static_cast<size_t>(r) * groups + g];

  if (!buf.w) {
    cudaStreamCreate(&buf.s);
    buf.mn = static_cast<size_t>(M) * N;
    cudaMalloc(&buf.w, w.size() * sizeof(block_q6_k));
    cudaMalloc(&buf.a, gm.size() * sizeof(BlockQ8_1Mmq));
    cudaMalloc(&buf.acts, acts.size() * sizeof(half));
    cudaMalloc(&buf.out, buf.mn * sizeof(half));
    cudaMalloc(&buf.partials, 8 * buf.mn * sizeof(float));
  }
  cudaMemcpyAsync(buf.w, w.data(), w.size() * sizeof(block_q6_k),
                  cudaMemcpyHostToDevice, buf.s);
  cudaMemcpyAsync(buf.a, gm.data(), gm.size() * sizeof(BlockQ8_1Mmq),
                  cudaMemcpyHostToDevice, buf.s);
  cudaMemcpyAsync(buf.acts, acts.data(), acts.size() * sizeof(half),
                  cudaMemcpyHostToDevice, buf.s);

  // Gate 2: device quantizer vs host emulation (group-major compare).
  {
    std::vector<BlockQ8_1Mmq> dev(gm.size());
    dim3 qgrid((groups + 3) / 4, M);
    QuantizeRowQ8_1MmqKernel<<<qgrid, 128, 0, buf.s>>>(
        buf.acts, buf.a, K, M);
    cudaMemcpyAsync(dev.data(), buf.a, dev.size() * sizeof(BlockQ8_1Mmq),
                    cudaMemcpyDeviceToHost, buf.s);
    cudaStreamSynchronize(buf.s);
    Report("quantizer == host emulation",
           std::memcmp(dev.data(), gm.data(), dev.size() * sizeof(BlockQ8_1Mmq)) == 0,
           0.0);
  }

  dim3 grid((N + kMmqY - 1) / kMmqY, (M + 15) / 16, ksplits);
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(InferfluxMmqQ6KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  InferfluxMmqQ6KMma<16><<<grid, dim3(32, kMmqWarps, 1), smem, buf.s>>>(
      reinterpret_cast<const char *>(buf.w), buf.a, buf.out, N, K, M,
      buf.partials, ksplits);
  if (ksplits > 1) {
    const int rthreads = 256;
    const size_t rblocks = (buf.mn + rthreads - 1) / rthreads;
    ReduceMmqKSplit<<<rblocks, rthreads, 0, buf.s>>>(buf.partials, buf.out,
                                                     ksplits, buf.mn);
  }
  std::vector<half> out(buf.mn);
  cudaMemcpyAsync(out.data(), buf.out, out.size() * sizeof(half),
                  cudaMemcpyDeviceToHost, buf.s);
  cudaStreamSynchronize(buf.s);
  return out;
}

// Random-data shape: kernel vs FPU reference on quantized activations.
void TestShape(int M, int N, int K, int ksplits, uint32_t seed) {
  const int blocks_per_row = K / 256;
  std::vector<block_q6_k> w(static_cast<size_t>(N) * blocks_per_row);
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i) b.ql[i] = Lcg(seed) & 0xFF;
    for (int i = 0; i < 64; ++i) b.qh[i] = Lcg(seed) & 0xFF;
    for (int i = 0; i < 16; ++i) b.scales[i] = 1 + (Lcg(seed) % 15);
    const half d =
        __float2half(0.001f + 0.004f * (Lcg(seed) % 1000) / 1000.0f);
    std::memcpy(&b.d, &d, 2);
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &v : acts)
    v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 1000.0f);
  std::vector<BlockQ8_1Mmq> row_major(static_cast<size_t>(M) * K / 128);
  for (int r = 0; r < M; ++r) QuantizeHost(acts, K, row_major, r);

  Buffers buf;
  const std::vector<half> out = RunKernel(w, acts, row_major, M, N, K, ksplits, buf);
  cudaStreamDestroy(buf.s);

  double max_rel = 0.0;
  size_t n_checked = 0;
  for (int col = 0; col < N; ++col) {
    for (int j = 0; j < M; ++j) {
      double ref = 0.0;
      for (int kb = 0; kb < blocks_per_row; ++kb) {
        const block_q6_k &b =
            w[static_cast<size_t>(col) * blocks_per_row + kb];
        for (int e = 0; e < 256; ++e) {
          const int k = kb * 256 + e;
          const BlockQ8_1Mmq &g =
              row_major[static_cast<size_t>(j) * (K / 128) + k / 128];
          ref += Q6KValue(b, e) * g.d4[(k % 128) / 32] * g.qs[k % 128];
        }
      }
      const float got = __half2float(out[static_cast<size_t>(j) * N + col]);
      const double rel =
          std::fabs((got - ref) / (ref == 0.0 ? 1.0 : ref));
      max_rel = std::max(max_rel, rel);
      ++n_checked;
    }
  }
  char name[64];
  snprintf(name, sizeof(name), "M=%d N=%d K=%d ks=%d (%zu cols)", M, N, K,
           ksplits, n_checked);
  Report(name, max_rel < 1e-2, max_rel);
  cudaFree(buf.w); cudaFree(buf.a); cudaFree(buf.acts); cudaFree(buf.out);
  cudaFree(buf.partials);
}

// Uniform-value exactness: w = -32 everywhere (quant 0, scale 1, d 1),
// a = +1 (quant 127, d4 = 1/127) -> out = -32 * K exactly.
void TestUniform() {
  const int N = 128, K = 256, M = 16;
  std::vector<block_q6_k> w(N);
  for (auto &b : w) {
    std::memset(b.ql, 0, 128);
    std::memset(b.qh, 0, 64);
    for (int i = 0; i < 16; ++i) b.scales[i] = 1;
    const half d = __float2half(1.0f);
    std::memcpy(&b.d, &d, 2);
  }
  std::vector<half> acts(static_cast<size_t>(M) * K, __float2half(1.0f));
  std::vector<BlockQ8_1Mmq> row_major(static_cast<size_t>(M) * K / 128);
  for (auto &g : row_major) {
    for (int s = 0; s < 4; ++s) g.d4[s] = 1.0f / 127.0f;
    for (int i = 0; i < 128; ++i) g.qs[i] = 127;
  }
  Buffers buf;
  const std::vector<half> out = RunKernel(w, acts, row_major, M, N, K, 1, buf);
  cudaStreamDestroy(buf.s);
  const float expect = -32.0f * K;
  double max_dev = 0.0;
  int n_bad = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (__half2float(out[i]) != expect) ++n_bad;
    max_dev = std::max(max_dev, static_cast<double>(std::fabs(
                                   __half2float(out[i]) - expect)));
  }
  Report("uniform pairing (-8192 exact, 2048 outputs)", n_bad == 0, max_dev);
  cudaFree(buf.w); cudaFree(buf.a); cudaFree(buf.acts); cudaFree(buf.out);
  cudaFree(buf.partials);
}

} // namespace

int main() {
  cudaFree(0);
  std::fesetround(FE_TONEAREST);
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
      prop.major < 7 ||
      (prop.major == 7 && prop.minor < 5)) {
    printf("test_mma_q6k: requires sm_75+ tensor cores, skipping\n");
    return 0;
  }

  TestUniform();
  TestShape(1, 256, 256, 1, 12345);
  TestShape(2, 300, 256, 1, 777);    // tail N tile (300 % 128 != 0)
  TestShape(16, 2048, 2048, 1, 4242);
  TestShape(16, 2048, 11008, 1, 999);
  TestShape(16, 2048, 11008, 3, 555); // K-split with reduce
  TestShape(16, 2048, 11008, 6, 31337);

  printf("%s (%d failures)\n", g_failures ? "RESULT: FAIL" : "RESULT: PASS",
         g_failures);
  return g_failures ? 1 : 0;
}
