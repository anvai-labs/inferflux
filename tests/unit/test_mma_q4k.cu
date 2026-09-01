/**
 * Correctness tests for the Q4_K MMA (int8 tensor-core) kernel — the
 * Q8_1-family path (vec_dot_q8_1_q8_1_mma port).
 *
 * Gates:
 *   1. Uniform-value exactness (structural pairing; permutation-blind
 *      spot-checks cannot catch what this does).
 *   2. DS quantizer byte-exact vs host emulation ({d, d*sum} half2 pair
 *      per 32-value group; the .y term feeds the Q4_K dmin correction).
 *   3. FPU reference with the CORRECT 12-byte 6-bit k4 scale decoding
 *      (get_scale_min_k4 reads q[j+4] up to byte 11 — scales span 12
 *      bytes, not 8), rel tol 2e-2 (half-precision ds scales vs Q6's
 *      float d4), including K-split variants.
 */
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using inferflux::runtime::cuda::native::BlockQ8_1MmqDs;
using inferflux::runtime::cuda::native::InferfluxMmqQ4KMma;
using inferflux::runtime::cuda::native::QuantizeRowQ8_1MmqDsKernel;
using inferflux::runtime::cuda::native::block_q4_k;
using inferflux::runtime::cuda::native::kMmqMmaTileXKQ81;
using inferflux::runtime::cuda::native::kMmqMmaWarps;
using inferflux::runtime::cuda::native::kMmqTileNeK;
using inferflux::runtime::cuda::native::kMmqTileYK;
using inferflux::runtime::cuda::native::kMmqY;
using inferflux::runtime::cuda::native::MmqSmemInts;
using inferflux::runtime::cuda::native::ReduceMmqKSplit;

namespace {

int g_fail = 0;
uint32_t Lcg(uint32_t &s) { return s = s * 1664525u + 1013904223u; }

// Host Q4_K dequant — 6-bit k4 scales (get_scale_min_k4 scheme).
void ScaleMinK4(const unsigned char *q, int j, int *sc, int *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

double Q4KValue(const block_q4_k &b, int e) {
  const float d = __half2float(__ushort_as_half(b.d));
  const float dmin = __half2float(__ushort_as_half(b.dmin));
  const int sb = e / 32;
  int sc = 0, m = 0;
  ScaleMinK4(b.scales, sb, &sc, &m);
  const unsigned char qbyte = b.qs[(sb / 2) * 32 + (e % 32)];
  const int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0xF);
  return d * sc * q - dmin * m;
}

// Host DS quantizer emulation (float ops match device).
void QuantizeDsHost(const std::vector<half> &x, int K,
                    std::vector<BlockQ8_1MmqDs> &row_major, int row) {
  const int groups = K / 128;
  for (int g = 0; g < groups; ++g) {
    BlockQ8_1MmqDs &grp = row_major[static_cast<size_t>(row) * groups + g];
    for (int sub = 0; sub < 4; ++sub) {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i)
        amax = fmaxf(amax, fabsf(__half2float(
                               x[static_cast<size_t>(row) * K + g * 128 +
                                 sub * 32 + i])));
      const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
      const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      float sum32 = 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float v = __half2float(
            x[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        const int q = nearbyintf(v * d_inv);
        grp.qs[sub * 32 + i] = static_cast<int8_t>(q);
        sum32 += q;
      }
      grp.ds[sub] =
          make_half2(__float2half_rn(d), __float2half_rn(d * sum32));
    }
  }
}

struct Buf {
  block_q4_k *w = nullptr;
  BlockQ8_1MmqDs *a = nullptr;
  half *acts = nullptr;
  half *out = nullptr;
  float *partials = nullptr;
  size_t mn = 0;
  cudaStream_t s;
};

std::vector<half> Run(const std::vector<block_q4_k> &w,
                      const std::vector<half> &acts,
                      const std::vector<BlockQ8_1MmqDs> &row_major, int M,
                      int N, int K, int ks, Buf &buf) {
  const int groups = K / 128;
  std::vector<BlockQ8_1MmqDs> gm(row_major.size());
  for (int r = 0; r < M; ++r)
    for (int g = 0; g < groups; ++g)
      gm[static_cast<size_t>(g) * M + r] =
          row_major[static_cast<size_t>(r) * groups + g];

  if (!buf.w) {
    cudaStreamCreate(&buf.s);
    buf.mn = static_cast<size_t>(M) * N;
    cudaMalloc(&buf.w, w.size() * sizeof(block_q4_k));
    cudaMalloc(&buf.a, gm.size() * sizeof(BlockQ8_1MmqDs));
    cudaMalloc(&buf.acts, acts.size() * sizeof(half));
    cudaMalloc(&buf.out, buf.mn * sizeof(half));
    cudaMalloc(&buf.partials, 8 * buf.mn * sizeof(float));
  }
  cudaMemcpyAsync(buf.w, w.data(), w.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, buf.s);
  cudaMemcpyAsync(buf.acts, acts.data(), acts.size() * sizeof(half),
                  cudaMemcpyHostToDevice, buf.s);

  // Device DS quantizer vs host emulation (group-major compare).
  {
    std::vector<BlockQ8_1MmqDs> dev(gm.size());
    dim3 qgrid((groups + 3) / 4, M);
    QuantizeRowQ8_1MmqDsKernel<<<qgrid, 128, 0, buf.s>>>(buf.acts, buf.a, K,
                                                         M);
    cudaMemcpyAsync(dev.data(), buf.a, dev.size() * sizeof(BlockQ8_1MmqDs),
                    cudaMemcpyDeviceToHost, buf.s);
    cudaStreamSynchronize(buf.s);
    const bool ok = std::memcmp(dev.data(), gm.data(),
                                dev.size() * sizeof(BlockQ8_1MmqDs)) == 0;
    printf("  DS quantizer == host: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      ++g_fail;
  }

  cudaMemcpyAsync(buf.a, gm.data(), gm.size() * sizeof(BlockQ8_1MmqDs),
                  cudaMemcpyHostToDevice, buf.s);
  dim3 grid((N + kMmqY - 1) / kMmqY, (M + 15) / 16, ks);
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(InferfluxMmqQ4KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  InferfluxMmqQ4KMma<16><<<grid, dim3(32, kMmqMmaWarps, 1), smem, buf.s>>>(
      reinterpret_cast<const char *>(buf.w), buf.a, buf.out, N, K, M,
      buf.partials, ks);
  if (ks > 1) {
    const int rt = 256;
    const size_t rb = (buf.mn + rt - 1) / rt;
    ReduceMmqKSplit<<<rb, rt, 0, buf.s>>>(buf.partials, buf.out, ks, buf.mn);
  }
  std::vector<half> out(buf.mn);
  cudaMemcpyAsync(out.data(), buf.out, out.size() * sizeof(half),
                  cudaMemcpyDeviceToHost, buf.s);
  cudaStreamSynchronize(buf.s);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("  KERNEL ERROR: %s\n", cudaGetErrorString(err));
    ++g_fail;
  }
  return out;
}

void TestUniform() {
  const int N = 128, K = 256, M = 16;
  std::vector<block_q4_k> w(N);
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i) b.qs[i] = 0x11; // lo=1, hi=1 nibbles
    // 12-byte k4 scale area: sc0-3 = bytes 0-3 (6-bit); m0-3 = bytes 4-7;
    // sc4-7 low nibbles live in bytes 8-11 (get_scale_min_k4 j>=4 reads
    // q[j+4]). sc=1 everywhere, m=0 everywhere.
    for (int i = 0; i < 12; ++i) b.scales[i] = 0;
    for (int j = 0; j < 4; ++j) {
      b.scales[j] = 1;      // sc0-3 = 1
      b.scales[j + 8] = 1;  // sc4-7 low nibbles = 1
    }
    const half d = __float2half(1.0f);
    const half dmin = __float2half(0.0f);
    std::memcpy(&b.d, &d, 2);
    std::memcpy(&b.dmin, &dmin, 2);
  }
  // Every weight value = 1*1*1 - 0 = 1.
  std::vector<half> acts(static_cast<size_t>(M) * K, __float2half(1.0f));
  std::vector<BlockQ8_1MmqDs> row_major(static_cast<size_t>(M) * K / 128);
  // ds.y = d * sum(qs) = (1/127) * (32*127) = 32.0 per group.
  for (auto &g : row_major)
    for (int sub = 0; sub < 4; ++sub) {
      g.ds[sub] = make_half2(__float2half_rn(1.0f / 127.0f),
                             __float2half_rn(32.0f));
      for (int i = 0; i < 32; ++i) g.qs[sub * 32 + i] = 127;
    }
  // act value = d*q = 1 for every element -> out = K exactly.
  Buf buf;
  const std::vector<half> out = Run(w, acts, row_major, M, N, K, 1, buf);
  cudaStreamDestroy(buf.s);
  const float expect = static_cast<float>(K);
  int bad = 0;
  int shown = 0;
  for (size_t idx = 0; idx < out.size(); ++idx)
    if (__half2float(out[idx]) != expect) {
      ++bad;
      if (shown++ < 8) {
        const int j = idx / N, i = idx % N;
        printf("  bad[j=%d i=%d]=%.1f ", j, i, __half2float(out[idx]));
      }
    }
  printf("uniform pairing (out == %d exact, %zu outputs): %s", K, out.size(),
         bad ? "FAIL" : "PASS");
  if (bad)
    ++g_fail;
  printf("\n");
  cudaFree(buf.w); cudaFree(buf.a); cudaFree(buf.acts); cudaFree(buf.out);
  cudaFree(buf.partials);
}

void TestShape(int M, int N, int K, int ks, uint32_t seed) {
  const int bpr = K / 256;
  std::vector<block_q4_k> w(static_cast<size_t>(N) * bpr);
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i) b.qs[i] = Lcg(seed) & 0xFF;
    for (int i = 0; i < 12; ++i) b.scales[i] = Lcg(seed) & 0xFF;
    const half d = __float2half(0.003f + 0.002f * (Lcg(seed) % 1000) / 1000.0f);
    const half dm = __float2half(0.001f);
    std::memcpy(&b.d, &d, 2);
    std::memcpy(&b.dmin, &dm, 2);
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &v : acts)
    v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 2000.0f);
  std::vector<BlockQ8_1MmqDs> row_major(static_cast<size_t>(M) * K / 128);
  for (int r = 0; r < M; ++r) QuantizeDsHost(acts, K, row_major, r);

  Buf buf;
  const std::vector<half> out = Run(w, acts, row_major, M, N, K, ks, buf);
  cudaStreamDestroy(buf.s);

  double max_rel = 0.0;
  for (int col = 0; col < N; ++col)
    for (int j = 0; j < M; ++j) {
      double ref = 0.0;
      for (int kb = 0; kb < bpr; ++kb) {
        const block_q4_k &b = w[static_cast<size_t>(col) * bpr + kb];
        for (int e = 0; e < 256; ++e) {
          const int k = kb * 256 + e;
          const BlockQ8_1MmqDs &g =
              row_major[static_cast<size_t>(j) * (K / 128) + k / 128];
          ref += Q4KValue(b, e) *
                 (__half2float(__low2half(g.ds[(k % 128) / 32])) *
                  g.qs[k % 128]);
        }
      }
      const float got = __half2float(out[static_cast<size_t>(j) * N + col]);
      const double rel = std::fabs((got - ref) / (std::fabs(ref) > 1.0 ? ref : 1.0));
      max_rel = std::max(max_rel, rel);
    }

  // Q8_1-family tolerance: activations carry half-precision {d, d*sum}
  // scales (llama's layout) vs Q6's float d4 — 2e-2 empirical precision
  // bound; structural correctness is gated by the uniform-exact test.
  const bool ok = max_rel < 2e-2;
  printf("M=%-2d N=%-6d K=%-5d ks=%d max_rel=%.3e  %s\n", M, N, K, ks,
         max_rel, ok ? "PASS" : "FAIL");
  if (!ok)
    ++g_fail;
  cudaFree(buf.w); cudaFree(buf.a); cudaFree(buf.acts); cudaFree(buf.out);
  cudaFree(buf.partials);
}

} // namespace

int main() {
  cudaFree(0);
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || prop.major < 7 ||
      (prop.major == 7 && prop.minor < 5)) {
    printf("test_mma_q4k: requires sm_75+ tensor cores, skipping\n");
    return 0;
  }
  printf("== uniform ==\n");
  TestUniform();
  printf("== FPU shapes ==\n");
  TestShape(1, 256, 256, 1, 12345);
  TestShape(2, 300, 256, 1, 777);
  TestShape(16, 1024, 2048, 1, 4242);
  // Production prefill geometries: gate/up (N=11008) and QKV (N=2048) at
  // K=hidden. M=2..8 prefill is a real dispatch state (short prompts) and
  // was previously untested — the zero-output regression lived here.
  TestShape(2, 11008, 2048, 1, 20260831);
  TestShape(3, 11008, 2048, 1, 20260832);
  TestShape(4, 11008, 2048, 1, 20260833);
  TestShape(8, 11008, 2048, 1, 20260834);
  TestShape(2, 2048, 2048, 1, 20260835);
  TestShape(5, 2048, 2048, 1, 20260836);
  // Second y-tile (rows 16+) — prefill M 17..31 lands in a partial second
  // tile with splits>=2 at QKV geometry; never covered before.
  TestShape(17, 2048, 2048, 1, 20260837);
  TestShape(18, 2048, 2048, 2, 20260838);
  TestShape(18, 11008, 2048, 2, 20260839);
  TestShape(20, 11008, 2048, 1, 20260840);
  TestShape(33, 2048, 2048, 1, 20260841);
  TestShape(16, 11008, 2048, 1, 999);
  TestShape(16, 11008, 2048, 3, 555);
  TestShape(16, 11008, 2048, 6, 31337);
  printf("RESULT: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
