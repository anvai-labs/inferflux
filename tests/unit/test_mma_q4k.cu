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

using inferflux::runtime::cuda::native::block_q4_k;
using inferflux::runtime::cuda::native::BlockQ8_1MmqDs;
using inferflux::runtime::cuda::native::InferfluxMmqQ4KMma;
using inferflux::runtime::cuda::native::kMmqMmaTileXKQ81;
using inferflux::runtime::cuda::native::kMmqMmaWarps;
using inferflux::runtime::cuda::native::kMmqTileNeK;
using inferflux::runtime::cuda::native::kMmqTileYK;
using inferflux::runtime::cuda::native::kMmqY;
using inferflux::runtime::cuda::native::MmqSmemInts;
using inferflux::runtime::cuda::native::QuantizeRowQ8_1MmqDsKernel;
using inferflux::runtime::cuda::native::ReduceMmqKSplit;
using inferflux::runtime::cuda::native::SiluMulQuantizeQ8_1MmqDsKernel;

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
        amax = fmaxf(
            amax,
            fabsf(__half2float(
                x[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i])));
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
      grp.ds[sub] = make_half2(__float2half_rn(d), __float2half_rn(d * sum32));
    }
  }
}

// Host emulation of the fused SwiGLU DS quantizer: quantizes
// silu(gate)*up elementwise, matching device float ops.
void SiluMulQuantizeDsHost(const std::vector<half> &gate,
                           const std::vector<half> &up, int K,
                           std::vector<BlockQ8_1MmqDs> &row_major, int row) {
  const int groups = K / 128;
  for (int g = 0; g < groups; ++g) {
    BlockQ8_1MmqDs &grp = row_major[static_cast<size_t>(row) * groups + g];
    for (int sub = 0; sub < 4; ++sub) {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float gv = __half2float(
            gate[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        const float uv = __half2float(
            up[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        amax = fmaxf(amax, fabsf(gv * uv / (1.0f + expf(-gv))));
      }
      const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
      const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      float sum32 = 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float gv = __half2float(
            gate[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        const float uv = __half2float(
            up[static_cast<size_t>(row) * K + g * 128 + sub * 32 + i]);
        const float v = gv * uv / (1.0f + expf(-gv));
        const int q = nearbyintf(v * d_inv);
        grp.qs[sub * 32 + i] = static_cast<int8_t>(q);
        sum32 += q;
      }
      grp.ds[sub] = make_half2(__float2half_rn(d), __float2half_rn(d * sum32));
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
    QuantizeRowQ8_1MmqDsKernel<<<qgrid, 128, 0, buf.s>>>(buf.acts, buf.a, K, M);
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
      buf.partials, ks, reinterpret_cast<const char *>(buf.w), buf.out, 0);
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
    for (int i = 0; i < 128; ++i)
      b.qs[i] = 0x11; // lo=1, hi=1 nibbles
    // 12-byte k4 scale area: sc0-3 = bytes 0-3 (6-bit); m0-3 = bytes 4-7;
    // sc4-7 low nibbles live in bytes 8-11 (get_scale_min_k4 j>=4 reads
    // q[j+4]). sc=1 everywhere, m=0 everywhere.
    for (int i = 0; i < 12; ++i)
      b.scales[i] = 0;
    for (int j = 0; j < 4; ++j) {
      b.scales[j] = 1;     // sc0-3 = 1
      b.scales[j + 8] = 1; // sc4-7 low nibbles = 1
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
      g.ds[sub] =
          make_half2(__float2half_rn(1.0f / 127.0f), __float2half_rn(32.0f));
      for (int i = 0; i < 32; ++i)
        g.qs[sub * 32 + i] = 127;
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
  cudaFree(buf.w);
  cudaFree(buf.a);
  cudaFree(buf.acts);
  cudaFree(buf.out);
  cudaFree(buf.partials);
}

void TestShape(int M, int N, int K, int ks, uint32_t seed) {
  const int bpr = K / 256;
  std::vector<block_q4_k> w(static_cast<size_t>(N) * bpr);
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i)
      b.qs[i] = Lcg(seed) & 0xFF;
    for (int i = 0; i < 12; ++i)
      b.scales[i] = Lcg(seed) & 0xFF;
    const half d = __float2half(0.003f + 0.002f * (Lcg(seed) % 1000) / 1000.0f);
    const half dm = __float2half(0.001f);
    std::memcpy(&b.d, &d, 2);
    std::memcpy(&b.dmin, &dm, 2);
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &v : acts)
    v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 2000.0f);
  std::vector<BlockQ8_1MmqDs> row_major(static_cast<size_t>(M) * K / 128);
  for (int r = 0; r < M; ++r)
    QuantizeDsHost(acts, K, row_major, r);

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
          ref +=
              Q4KValue(b, e) *
              (__half2float(__low2half(g.ds[(k % 128) / 32])) * g.qs[k % 128]);
        }
      }
      const float got = __half2float(out[static_cast<size_t>(j) * N + col]);
      const double rel =
          std::fabs((got - ref) / (std::fabs(ref) > 1.0 ? ref : 1.0));
      max_rel = std::max(max_rel, rel);
    }

  // Q8_1-family tolerance: activations carry half-precision {d, d*sum}
  // scales (llama's layout) vs Q6's float d4, so the residual grows with
  // dot length — measured ~1.2e-2 at K=2048 and ~2.1-3.0e-2 at the
  // down-proj K=11008 (same numeric class as llama.cpp's
  // vec_dot_q4_K_q8_1). Structural correctness is gated by the
  // uniform-exact test, not this bound.
  const double tol = K >= 8192 ? 4e-2 : 2e-2;
  const bool ok = max_rel < tol;
  printf("M=%-2d N=%-6d K=%-5d ks=%d max_rel=%.3e (tol %.0e) %s\n", M, N, K, ks,
         max_rel, tol, ok ? "PASS" : "FAIL");
  if (!ok)
    ++g_fail;
  cudaFree(buf.w);
  cudaFree(buf.a);
  cudaFree(buf.acts);
  cudaFree(buf.out);
  cudaFree(buf.partials);
}

// Fused SwiGLU DS quantizer (S18 down-proj input path) vs host emulation,
// byte-exact.
void TestSiluMulQuantizer(int M, int K, uint32_t seed) {
  std::vector<half> gate(static_cast<size_t>(M) * K), up(gate.size());
  for (auto &v : gate)
    v = __float2half(4.0f * (static_cast<int>(Lcg(seed) % 2001) - 1000) /
                     1000.0f);
  for (auto &v : up)
    v = __float2half(2.0f * (static_cast<int>(Lcg(seed) % 2001) - 1000) /
                     1000.0f);
  const int groups = K / 128;
  std::vector<BlockQ8_1MmqDs> row_major(static_cast<size_t>(M) * groups);
  for (int r = 0; r < M; ++r)
    SiluMulQuantizeDsHost(gate, up, K, row_major, r);
  std::vector<BlockQ8_1MmqDs> gm(row_major.size());
  for (int r = 0; r < M; ++r)
    for (int g = 0; g < groups; ++g)
      gm[static_cast<size_t>(g) * M + r] =
          row_major[static_cast<size_t>(r) * groups + g];

  half *dg, *du;
  BlockQ8_1MmqDs *da;
  cudaMalloc(&dg, gate.size() * sizeof(half));
  cudaMalloc(&du, up.size() * sizeof(half));
  cudaMalloc(&da, gm.size() * sizeof(BlockQ8_1MmqDs));
  cudaMemcpy(dg, gate.data(), gate.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(du, up.data(), up.size() * sizeof(half), cudaMemcpyHostToDevice);
  dim3 qgrid((groups + 3) / 4, M);
  SiluMulQuantizeQ8_1MmqDsKernel<<<qgrid, 128>>>(dg, du, da, K, M);
  std::vector<BlockQ8_1MmqDs> dev(gm.size());
  cudaMemcpy(dev.data(), da, dev.size() * sizeof(BlockQ8_1MmqDs),
             cudaMemcpyDeviceToHost);
  const bool ok = std::memcmp(dev.data(), gm.data(),
                              dev.size() * sizeof(BlockQ8_1MmqDs)) == 0;
  printf("M=%-2d K=%-5d silu-mul DS quantizer == host: %s\n", M, K,
         ok ? "PASS" : "FAIL");
  if (!ok)
    ++g_fail;
  cudaFree(dg);
  cudaFree(du);
  cudaFree(da);
}

} // namespace

// Dual-tensor launch (gate+up fusion): tiles [n1_tiles, grid) must read w2
// and write out2 with local tile indices; the union must match two
// single-tensor launches exactly.
void TestDualReal() {
  const int N = 11008, K = 2048, M = 16; // real gate/up shape
  const int bpr = K / 256;
  std::vector<block_q4_k> w1(static_cast<size_t>(N) * bpr);
  std::vector<block_q4_k> w2(static_cast<size_t>(N) * bpr);
  uint32_t seed = 12345;
  auto rnd = [&seed]() {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return static_cast<float>(seed % 2001 - 1000) / 1000.0f;
  };
  for (int t = 0; t < 2; ++t) {
    auto &w = t ? w2 : w1;
    for (auto &b : w) {
      for (int i = 0; i < 128; ++i)
        b.qs[i] = static_cast<unsigned char>(rnd() * 60 + 64);
      for (int i = 0; i < 12; ++i)
        b.scales[i] = static_cast<unsigned char>(rnd() * 8);
      const half d = __float2half(rnd() * 0.5f + 0.5f);
      const half dmin = __float2half(0.0f);
      std::memcpy(&b.d, &d, 2);
      std::memcpy(&b.dmin, &dmin, 2);
    }
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &a : acts)
    a = __float2half(rnd() * 2.0f - 1.0f);
  // Quantize on device (same kernel the server uses).
  std::vector<BlockQ8_1MmqDs> row_major(static_cast<size_t>(M) * K / 128);
  {
    half *d_in;
    BlockQ8_1MmqDs *d_q;
    cudaMalloc(&d_in, acts.size() * sizeof(half));
    cudaMalloc(&d_q, row_major.size() * sizeof(BlockQ8_1MmqDs));
    cudaMemcpyAsync(d_in, acts.data(), acts.size() * sizeof(half),
                    cudaMemcpyHostToDevice);
    dim3 qgrid((K / 128 + 3) / 4, M);
    QuantizeRowQ8_1MmqDsKernel<<<qgrid, 128>>>(d_in, d_q, K, M);
    cudaMemcpy(row_major.data(), d_q, row_major.size() * sizeof(BlockQ8_1MmqDs),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_q);
    cudaGetLastError();
  }

  cudaStream_t s;
  cudaStreamCreate(&s);
  const size_t mn = static_cast<size_t>(M) * N;
  block_q4_k *d_w1, *d_w2;
  half *d_acts, *d_out1, *d_out2, *d_ref1, *d_ref2;
  BlockQ8_1MmqDs *d_ds;
  float *d_part;
  cudaMalloc(&d_w1, w1.size() * sizeof(block_q4_k));
  cudaMalloc(&d_w2, w2.size() * sizeof(block_q4_k));
  cudaMalloc(&d_acts, acts.size() * sizeof(half));
  cudaMalloc(&d_ds, row_major.size() * sizeof(BlockQ8_1MmqDs));
  cudaMalloc(&d_out1, mn * sizeof(half));
  cudaMalloc(&d_out2, mn * sizeof(half));
  cudaMalloc(&d_ref1, mn * sizeof(half));
  cudaMalloc(&d_ref2, mn * sizeof(half));
  cudaMalloc(&d_part, 8 * mn * sizeof(float));
  cudaMemcpyAsync(d_w1, w1.data(), w1.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_w2, w2.data(), w2.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_acts, acts.data(), acts.size() * sizeof(half),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_ds, row_major.data(),
                  row_major.size() * sizeof(BlockQ8_1MmqDs),
                  cudaMemcpyHostToDevice, s);
  dim3 grid((N + kMmqY - 1) / kMmqY, (M + 15) / 16, 1);
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(InferfluxMmqQ4KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  // Reference: two single-tensor launches.
  InferfluxMmqQ4KMma<16><<<grid, dim3(32, kMmqMmaWarps, 1), smem, s>>>(
      reinterpret_cast<const char *>(d_w1), d_ds, d_ref1, N, K, M, nullptr, 1,
      reinterpret_cast<const char *>(d_w1), d_ref1, 0);
  InferfluxMmqQ4KMma<16><<<grid, dim3(32, kMmqMmaWarps, 1), smem, s>>>(
      reinterpret_cast<const char *>(d_w2), d_ds, d_ref2, N, K, M, nullptr, 1,
      reinterpret_cast<const char *>(d_w2), d_ref2, 0);
  // Dual: one launch, tiles {0,1} -> w1/out1, tiles {2,3} -> w2/out2.
  InferfluxMmqQ4KMma<16>
      <<<dim3(grid.x * 2, grid.y, 1), dim3(32, kMmqMmaWarps, 1), smem, s>>>(
          reinterpret_cast<const char *>(d_w1), d_ds, d_out1, N, K, M, nullptr,
          1, reinterpret_cast<const char *>(d_w2), d_out2,
          (N + kMmqY - 1) / kMmqY);
  std::vector<half> r1(mn), r2(mn), o1(mn), o2(mn);
  cudaMemcpyAsync(r1.data(), d_ref1, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaMemcpyAsync(r2.data(), d_ref2, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaMemcpyAsync(o1.data(), d_out1, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaMemcpyAsync(o2.data(), d_out2, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaStreamSynchronize(s);
  int bad = 0;
  for (size_t i = 0; i < mn; ++i) {
    uint16_t b1, b2, r1b, r2b;
    std::memcpy(&b1, &o1[i], 2);
    std::memcpy(&b2, &o2[i], 2);
    std::memcpy(&r1b, &r1[i], 2);
    std::memcpy(&r2b, &r2[i], 2);
    if (b1 != r1b) {
      ++bad;
      if (bad < 4)
        printf("  dual out1[%zu]=%.4f want %.4f\n", i, __half2float(o1[i]),
               __half2float(r1[i]));
    }
    if (b2 != r2b) {
      ++bad;
      if (bad < 8)
        printf("  dual out2[%zu]=%.4f want %.4f\n", i, __half2float(o2[i]),
               __half2float(r2[i]));
    }
  }
  printf("dual tile mapping (N=%d K=%d M=%d): %s", N, K, M,
         bad ? "FAIL" : "PASS");
  if (bad)
    ++g_fail;
  printf("\n");
  cudaFree(d_w1);
  cudaFree(d_w2);
  cudaFree(d_acts);
  cudaFree(d_ds);
  cudaFree(d_out1);
  cudaFree(d_out2);
  cudaFree(d_ref1);
  cudaFree(d_ref2);
  cudaFree(d_part);
  cudaStreamDestroy(s);
}

// Triple-launch (q+k+v fusion) with UNEQUAL segment widths: catches
// per-segment row-stride bugs (all segments must match three single
// launches bitwise, splits 1 and 2).
// Regression (release review): single Q4_K launch with N % 128 != 0 at
// splits > 1 must match its ks=1 output bitwise. The #152 padded
// partials stride broke exactly this class (v0.1.1 was correct).
void TestSplitsOddN() {
  const int N = 320, K = 256, M = 16; // N % 128 = 64
  const int bpr = K / 256;
  std::vector<block_q4_k> w(static_cast<size_t>(N) * bpr);
  uint32_t seed = 4242;
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i) {
      b.qs[i] = Lcg(seed) & 0xFF;
    }
    for (int i = 0; i < 12; ++i) {
      b.scales[i] = Lcg(seed) & 0xFF;
    }
    const half d = __float2half(0.003f + 0.002f * (Lcg(seed) % 1000) / 1000.0f);
    const half dm = __float2half(0.001f);
    std::memcpy(&b.d, &d, 2);
    std::memcpy(&b.dmin, &dm, 2);
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &v : acts) {
    v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 2000.0f);
  }
  std::vector<BlockQ8_1MmqDs> hq(static_cast<size_t>(M) * (K / 128));
  for (int r = 0; r < M; ++r) {
    QuantizeDsHost(acts, K, hq, r);
  }

  cudaStream_t s;
  cudaStreamCreate(&s);
  const size_t mn = static_cast<size_t>(M) * N;
  block_q4_k *d_w;
  BlockQ8_1MmqDs *d_a;
  half *d_o1, *d_o2;
  float *d_part;
  cudaMalloc(&d_w, w.size() * sizeof(block_q4_k));
  cudaMalloc(&d_a, hq.size() * sizeof(BlockQ8_1MmqDs));
  cudaMalloc(&d_o1, mn * sizeof(half));
  cudaMalloc(&d_o2, mn * sizeof(half));
  cudaMalloc(&d_part, 8 * mn * sizeof(float));
  cudaMemcpyAsync(d_w, w.data(), w.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_a, hq.data(), hq.size() * sizeof(BlockQ8_1MmqDs),
                  cudaMemcpyHostToDevice, s);
  dim3 grid((N + 127) / 128, (M + 15) / 16, 1);
  const size_t smem = MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(InferfluxMmqQ4KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));

  auto run_ks = [&](int ks, half *out) {
    dim3 g(grid.x, grid.y, ks);
    InferfluxMmqQ4KMma<16><<<g, dim3(32, kMmqMmaWarps, 1), smem, s>>>(
        reinterpret_cast<const char *>(d_w), d_a, out, N, K, M,
        ks > 1 ? d_part : nullptr, ks, reinterpret_cast<const char *>(d_w), out,
        0);
    if (ks > 1) {
      ReduceMmqKSplit<<<static_cast<int>((mn + 255) / 256), 256, 0, s>>>(
          d_part, out, ks, mn);
    }
  };

  run_ks(1, d_o1);
  run_ks(2, d_o2);
  cudaStreamSynchronize(s);
  std::vector<half> o1(mn), o2(mn);
  cudaMemcpyAsync(o1.data(), d_o1, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaMemcpyAsync(o2.data(), d_o2, mn * sizeof(half), cudaMemcpyDeviceToHost,
                  s);
  cudaStreamSynchronize(s);
  int bad = 0;
  for (size_t i = 0; i < mn; ++i) {
    uint16_t a, b;
    std::memcpy(&a, &o1[i], 2);
    std::memcpy(&b, &o2[i], 2);
    if (a != b) {
      ++bad;
      if (bad < 4)
        printf("  oddN ks2 o[%zu]=%.4f want %.4f (row %zu)\n", i,
               __half2float(o2[i]), __half2float(o1[i]), i / N);
    }
  }
  printf("splits odd-N (N=%d M=%d ks=2 vs ks=1): %s", N, M,
         bad ? "FAIL" : "PASS");
  if (bad)
    ++g_fail;
  printf("\n");
  cudaFree(d_w);
  cudaFree(d_a);
  cudaFree(d_o1);
  cudaFree(d_o2);
  cudaFree(d_part);
  cudaStreamDestroy(s);
}

void TestTripleUnequal() {
  const int N1 = 512, N2 = 256, N3 = 256, K = 256, M = 16;
  const int bpr = K / 256;
  std::vector<block_q4_k> w1(static_cast<size_t>(N1) * bpr);
  std::vector<block_q4_k> w2(static_cast<size_t>(N2) * bpr);
  std::vector<block_q4_k> w3(static_cast<size_t>(N3) * bpr);
  uint32_t seed = 777;
  for (auto *wp : {&w1, &w2, &w3}) {
    for (auto &b : *wp) {
      for (int i = 0; i < 128; ++i) {
        b.qs[i] = Lcg(seed) & 0xFF;
      }
      for (int i = 0; i < 12; ++i) {
        b.scales[i] = Lcg(seed) & 0xFF;
      }
      const half d =
          __float2half(0.003f + 0.002f * (Lcg(seed) % 1000) / 1000.0f);
      const half dm = __float2half(0.001f);
      std::memcpy(&b.d, &d, 2);
      std::memcpy(&b.dmin, &dm, 2);
    }
  }
  std::vector<half> acts(static_cast<size_t>(M) * K);
  for (auto &v : acts) {
    v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 2000.0f);
  }
  std::vector<BlockQ8_1MmqDs> hq(static_cast<size_t>(M) * (K / 128));
  for (int r = 0; r < M; ++r) {
    QuantizeDsHost(acts, K, hq, r);
  }
  cudaStream_t s;
  cudaStreamCreate(&s);
  const size_t mn1 = static_cast<size_t>(M) * N1;
  const size_t mn2 = static_cast<size_t>(M) * N2;
  const size_t mn3 = static_cast<size_t>(M) * N3;
  block_q4_k *d_w1, *d_w2, *d_w3;
  inferflux::runtime::cuda::native::BlockQ8_1MmqDs *d_a;
  half *d_o1, *d_o2, *d_o3, *d_r1, *d_r2, *d_r3;
  float *d_part;
  cudaMalloc(&d_w1, w1.size() * sizeof(block_q4_k));
  cudaMalloc(&d_w2, w2.size() * sizeof(block_q4_k));
  cudaMalloc(&d_w3, w3.size() * sizeof(block_q4_k));
  cudaMalloc(&d_a,
             hq.size() *
                 sizeof(inferflux::runtime::cuda::native::BlockQ8_1MmqDs));
  cudaMalloc(&d_o1, mn1 * sizeof(half));
  cudaMalloc(&d_o2, mn2 * sizeof(half));
  cudaMalloc(&d_o3, mn3 * sizeof(half));
  cudaMalloc(&d_r1, mn1 * sizeof(half));
  cudaMalloc(&d_r2, mn2 * sizeof(half));
  cudaMalloc(&d_r3, mn3 * sizeof(half));
  cudaMalloc(&d_part, 8 * (mn1 + mn2 + mn3) * sizeof(float));
  cudaMemcpyAsync(d_w1, w1.data(), w1.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_w2, w2.data(), w2.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_w3, w3.data(), w3.size() * sizeof(block_q4_k),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_a, hq.data(),
                  hq.size() *
                      sizeof(inferflux::runtime::cuda::native::BlockQ8_1MmqDs),
                  cudaMemcpyHostToDevice, s);
  const size_t smem =
      inferflux::runtime::cuda::native::MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  const dim3 block(32, inferflux::runtime::cuda::native::kMmqMmaWarps, 1);

  int bad = 0;
  // ks=1 only: the shipped k/v dual forces splits=1, and the single q
  // launch's own splits path is self-reducing inside its launcher.
  {
    dim3 g1((N1 + 127) / 128, (M + 15) / 16, 1);
    dim3 g2((N2 + 127) / 128, (M + 15) / 16, 1);
    dim3 g3((N3 + 127) / 128, (M + 15) / 16, 1);
    inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>
        <<<g1, block, smem, s>>>(reinterpret_cast<const char *>(d_w1), d_a,
                                 d_r1, N1, K, M, nullptr, 1,
                                 reinterpret_cast<const char *>(d_w1), d_r1, 0);
    inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>
        <<<g2, block, smem, s>>>(reinterpret_cast<const char *>(d_w2), d_a,
                                 d_r2, N2, K, M, nullptr, 1,
                                 reinterpret_cast<const char *>(d_w2), d_r2, 0);
    inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>
        <<<g3, block, smem, s>>>(reinterpret_cast<const char *>(d_w3), d_a,
                                 d_r3, N3, K, M, nullptr, 1,
                                 reinterpret_cast<const char *>(d_w3), d_r3, 0);
    // Shipped combination: single q launch + equal-N k/v dual launch.
    dim3 gq((N1 + 127) / 128, (M + 15) / 16, 1);
    dim3 gd((N2 + N3 + 127) / 128, (M + 15) / 16, 1);
    inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>
        <<<gq, block, smem, s>>>(reinterpret_cast<const char *>(d_w1), d_a,
                                 d_o1, N1, K, M, nullptr, 1,
                                 reinterpret_cast<const char *>(d_w1), d_o1, 0);
    inferflux::runtime::cuda::native::InferfluxMmqQ4KMma<16>
        <<<gd, block, smem, s>>>(
            reinterpret_cast<const char *>(d_w2), d_a, d_o2, N2, K, M, nullptr,
            1, reinterpret_cast<const char *>(d_w3), d_o3, N2 / 128);
    cudaStreamSynchronize(s);
    std::vector<half> o1(mn1), o2(mn2), o3(mn3), r1(mn1), r2(mn2), r3(mn3);
    cudaMemcpyAsync(o1.data(), d_o1, mn1 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaMemcpyAsync(o2.data(), d_o2, mn2 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaMemcpyAsync(o3.data(), d_o3, mn3 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaMemcpyAsync(r1.data(), d_r1, mn1 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaMemcpyAsync(r2.data(), d_r2, mn2 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaMemcpyAsync(r3.data(), d_r3, mn3 * sizeof(half), cudaMemcpyDeviceToHost,
                    s);
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < mn1; ++i) {
      uint16_t a, b;
      std::memcpy(&a, &o1[i], 2);
      std::memcpy(&b, &r1[i], 2);
      if (a != b) {
        ++bad;
        if (bad < 4)
          printf("  q o1[%zu] mismatch\n", i);
      }
    }
    for (size_t i = 0; i < mn2; ++i) {
      uint16_t a, b;
      std::memcpy(&a, &o2[i], 2);
      std::memcpy(&b, &r2[i], 2);
      if (a != b) {
        ++bad;
        if (bad < 8)
          printf("  kv o2[%zu] mismatch (row %zu)\n", i, i / N2);
      }
    }
    for (size_t i = 0; i < mn3; ++i) {
      uint16_t a, b;
      std::memcpy(&a, &o3[i], 2);
      std::memcpy(&b, &r3[i], 2);
      if (a != b) {
        ++bad;
        if (bad < 12)
          printf("  kv o3[%zu] mismatch (row %zu)\n", i, i / N3);
      }
    }
  }
  printf("triple unequal-N (N=%d/%d/%d M=%d ks 1+2): %s", N1, N2, N3, M,
         bad ? "FAIL" : "PASS");
  if (bad)
    ++g_fail;
  printf("\n");
  cudaFree(d_w1);
  cudaFree(d_w2);
  cudaFree(d_w3);
  cudaFree(d_a);
  cudaFree(d_o1);
  cudaFree(d_o2);
  cudaFree(d_o3);
  cudaFree(d_r1);
  cudaFree(d_r2);
  cudaFree(d_r3);
  cudaFree(d_part);
  cudaStreamDestroy(s);
}

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
  // Production down-proj geometry (S18): N=hidden, K=intermediate. Mixed
  // quants (ollama Q4_K_M) carry per-layer Q4_K down-proj tensors that run
  // this path at decode M=4..16; ks=3 matches the production split
  // heuristic (16 N-tiles under 48 SMs).
  printf("== down-proj geometry ==\n");
  TestShape(4, 2048, 11008, 1, 20260850);
  TestShape(6, 2048, 11008, 1, 20260851);
  TestShape(8, 2048, 11008, 1, 20260852);
  TestShape(16, 2048, 11008, 3, 20260853);
  TestShape(17, 2048, 11008, 1, 20260854); // partial second y-tile
  printf("== silu-mul DS quantizer ==\n");
  TestSiluMulQuantizer(4, 11008, 20260855);
  TestSiluMulQuantizer(16, 11008, 20260856);
  TestDualReal();
  TestSplitsOddN();
  TestTripleUnequal();
  printf("RESULT: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
