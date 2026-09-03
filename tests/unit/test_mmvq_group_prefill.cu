/**
 * Numeric correctness tests for the M>=2 (multi-column) MMVQ projection
 * kernels used by prefill — the QKV triple and gate/up pair group kernels
 * plus the single-projection ncols variants. These shapes only run at
 * seq_len >= 2, which existing coverage (decode-shaped, M=1) never
 * touched; garbage-first-token triage pointed here.
 */

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace inferflux::runtime::cuda::native;

namespace {

int g_failures = 0;

uint32_t Lcg(uint32_t &s) { return s = s * 1664525u + 1013904223u; }

void Report(const char *name, bool ok, double detail) {
  printf("%-46s %s (%.3e)\n", name, ok ? "PASS" : "FAIL", detail);
  if (!ok) {
    ++g_failures;
  }
}

// Q4_K element dequant (host mirror of dequant_q4k_element).
void ScaleMinK4(int j, const unsigned char *q, unsigned char *d,
                unsigned char *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Dot part only (d*sc*q): the min term is applied per 32-element block via
// ApplyMinTerm using the half-rounded d*sum the kernels consume.
double Q4KDot(const block_q4_k &b, int e) {
  const int sb = e / 32;
  const int ee = e % 32;
  unsigned char sc, m;
  ScaleMinK4(sb, b.scales, &sc, &m);
  const unsigned char qbyte = b.qs[(sb / 2) * 32 + ee];
  const int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0x0F);
  return __half2float(*reinterpret_cast<const half *>(&b.d)) * sc * q;
}

double Q4KMin(const block_q4_k &b, int e) {
  const int sb = e / 32;
  unsigned char sc, m;
  ScaleMinK4(sb, b.scales, &sc, &m);
  return __half2float(*reinterpret_cast<const half *>(&b.dmin)) * m;
}

double Q4KValue(const block_q4_k &b, int e) {
  const int sb = e / 32;
  const int ee = e % 32;
  unsigned char sc, m;
  ScaleMinK4(sb, b.scales, &sc, &m);
  const unsigned char qbyte = b.qs[(sb / 2) * 32 + ee];
  const int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0x0F);
  const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
  const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));
  return d * sc * q - dmin * m;
}

std::vector<block_q4_k> MakeQ4(int rows, int nsb, uint32_t &seed) {
  std::vector<block_q4_k> w(static_cast<size_t>(rows) * nsb);
  for (auto &b : w) {
    const half hd = __float2half(0.006f * ((Lcg(seed) % 5) + 1));
    const half hdm = __float2half(0.002f * ((Lcg(seed) % 4) + 1));
    std::memcpy(&b.d, &hd, sizeof(b.d));
    std::memcpy(&b.dmin, &hdm, sizeof(b.dmin));
    for (int i = 0; i < K_SCALE_SIZE; ++i)
      b.scales[i] = Lcg(seed) & 0xFF;
    for (int i = 0; i < QK_K / 2; ++i)
      b.qs[i] = Lcg(seed) & 0xFF;
  }
  return w;
}

struct Acts {
  std::vector<block_q8_1> h;
  std::vector<double> dequant;   // [M][K] reference activations
  std::vector<float> sum_term;   // [M][K/32] half(d8*sum(qs)) as the kernels read it
};

Acts MakeActs(int M, int K, uint32_t &seed) {
  Acts a;
  const int nq8 = K / QK8_1;
  a.h.resize(static_cast<size_t>(M) * nq8);
  a.dequant.resize(static_cast<size_t>(M) * K);
  a.sum_term.resize(static_cast<size_t>(M) * nq8);
  for (int m = 0; m < M; ++m) {
    for (int blk = 0; blk < nq8; ++blk) {
      auto &b = a.h[static_cast<size_t>(m) * nq8 + blk];
      const float d8 = 0.01f * ((Lcg(seed) % 7) + 1);
      float qsum = 0.0f;
      for (int i = 0; i < QK8_1; ++i) {
        b.qs[i] = static_cast<char>((Lcg(seed) % 251) - 125);
        qsum += static_cast<int8_t>(b.qs[i]);
        a.dequant[static_cast<size_t>(m) * K + blk * 32 + i] =
            d8 * static_cast<int8_t>(b.qs[i]);
      }
      // Q8_1 DS layout: {d, d*sum-of-qs} (llama.cpp convention; the Q4_K
      // min correction consumes the d*sum field, rounded to half).
      const float s_half = __half2float(__float2half(d8 * qsum));
      const half2 ds = __halves2half2(__float2half(d8), __float2half(s_half));
      std::memcpy(&b.ds, &ds, sizeof(b.ds));
      a.sum_term[static_cast<size_t>(m) * nq8 + blk] = s_half;
    }
  }
  return a;
}

} // namespace

int main() {
  const int K = 2048;
  const int nsb = K / QK_K;
  cudaStream_t s;
  cudaStreamCreate(&s);

  for (int M : {1, 2, 4}) {
    uint32_t seed = 31 * M + 5;

    // --- QKV triple group (three projections, distinct N) ---
    const int Ns[3] = {192, 128, 128};
    std::vector<std::vector<block_q4_k>> w;
    for (int p = 0; p < 3; ++p)
      w.push_back(MakeQ4(Ns[p], nsb, seed));
    Acts acts = MakeActs(M, K, seed);

    std::vector<double> ref[3];
    for (int p = 0; p < 3; ++p) {
      ref[p].resize(static_cast<size_t>(M) * Ns[p]);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < Ns[p]; ++n) {
          double sum = 0.0;
          for (int k = 0; k < K; ++k)
            sum += Q4KDot(w[p][static_cast<size_t>(n) * nsb + k / QK_K],
                          k % QK_K) *
                   acts.dequant[static_cast<size_t>(m) * K + k];
          for (int k = 0; k < K; k += 32)
            sum -= Q4KMin(w[p][static_cast<size_t>(n) * nsb + k / QK_K],
                          k % QK_K) *
                   acts.sum_term[static_cast<size_t>(m) * (K / 32) + k / 32];
          ref[p][static_cast<size_t>(m) * Ns[p] + n] = sum;
        }
    }

    block_q8_1 *d_a;
    cudaMalloc(&d_a, acts.h.size() * sizeof(block_q8_1));
    cudaMemcpy(d_a, acts.h.data(), acts.h.size() * sizeof(block_q8_1),
               cudaMemcpyHostToDevice);
    const void *dw[3];
    half *dout[3];
    for (int p = 0; p < 3; ++p) {
      void *d;
      cudaMalloc(&d, w[p].size() * sizeof(block_q4_k));
      cudaMemcpy(d, w[p].data(), w[p].size() * sizeof(block_q4_k),
                 cudaMemcpyHostToDevice);
      dw[p] = d;
      cudaMalloc(&dout[p], static_cast<size_t>(M) * Ns[p] * sizeof(half));
    }

    DispatchMmvqTriple<block_q4_k, inferflux_mmvq_q4k_group<1, 3>,
                       inferflux_mmvq_q4k_group<2, 3>,
                       inferflux_mmvq_q4k_group<4, 3>,
                       inferflux_mmvq_q4k_group<8, 3>>(
        dw[0], dw[1], dw[2], d_a, dout[0], Ns[0], dout[1], Ns[1], dout[2],
        Ns[2], M, K, s);

    double worst = 0.0;
    int nbad = 0;
    for (int p = 0; p < 3; ++p) {
      double scale_p = 0.0;
      for (double r : ref[p])
        scale_p = std::fmax(scale_p, std::fabs(r));
      std::vector<half> got(static_cast<size_t>(M) * Ns[p]);
      cudaMemcpy(got.data(), dout[p], got.size() * sizeof(half),
                 cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < got.size(); ++i) {
        const double r = ref[p][i];
        // Cancellation-safe: near-zero refs blow up pure relative error (the
        // repo convention — see test_mma_q6k). Denominator floors at 1% of
        // the projection's max magnitude, matching the dp4a quantization
        // noise floor of the Q8_1 activation format.
        const double denom =
            std::fabs(r) > 0.01 * scale_p ? std::fabs(r) : 0.01 * scale_p;
        const double rel = std::fabs(__half2float(got[i]) - r) / denom;
        if (rel > worst) {
          worst = rel;
          if (rel > 2e-2)
            printf("  group M=%d p=%d elem %zu: ref=%.3f got=%.3f (scale %.1f, rel %.2e)\n",
                   M, p, i, r, __half2float(got[i]), scale_p, rel);
        }
        nbad += rel > 2e-2;
      }
    }
    char name[64];
    std::snprintf(name, sizeof(name), "M=%d q4k_group<%d,3> vs FPU (%d bad)",
                  M, M, nbad);
    Report(name, worst <= 2e-2, worst);

    // --- single projection at ncols=M ---
    {
      const int N1 = 160;
      auto w1 = MakeQ4(N1, nsb, seed);
      std::vector<double> ref1(static_cast<size_t>(M) * N1);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N1; ++n) {
          double sum = 0.0;
          for (int k = 0; k < K; ++k)
            sum += Q4KDot(w1[static_cast<size_t>(n) * nsb + k / QK_K],
                          k % QK_K) *
                   acts.dequant[static_cast<size_t>(m) * K + k];
          for (int k = 0; k < K; k += 32)
            sum -= Q4KMin(w1[static_cast<size_t>(n) * nsb + k / QK_K],
                          k % QK_K) *
                   acts.sum_term[static_cast<size_t>(m) * (K / 32) + k / 32];
          ref1[static_cast<size_t>(m) * N1 + n] = sum;
        }
      block_q4_k *d_w1;
      half *d_o1;
      cudaMalloc(&d_w1, w1.size() * sizeof(block_q4_k));
      cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(block_q4_k),
                 cudaMemcpyHostToDevice);
      cudaMalloc(&d_o1, static_cast<size_t>(M) * N1 * sizeof(half));
      const int ncols = (M == 1) ? 1 : ((M <= 2) ? 2 : 4);
      dim3 grid(N1, (M + ncols - 1) / ncols);
      if (ncols == 1)
        inferflux_mmvq_q4k<1><<<grid, 128, 0, s>>>(d_w1, d_a, d_o1, N1, K, M);
      else if (ncols == 2)
        inferflux_mmvq_q4k<2><<<grid, 128, 0, s>>>(d_w1, d_a, d_o1, N1, K, M);
      else
        inferflux_mmvq_q4k<4><<<grid, 128, 0, s>>>(d_w1, d_a, d_o1, N1, K, M);
      std::vector<half> got(static_cast<size_t>(M) * N1);
      cudaMemcpy(got.data(), d_o1, got.size() * sizeof(half),
                 cudaMemcpyDeviceToHost);
      double w1worst = 0.0;
      int bad1 = 0;
      double scale_max = 0.0;
      for (double r : ref1)
        scale_max = std::fmax(scale_max, std::fabs(r));
      for (size_t i = 0; i < got.size(); ++i) {
        const double r = ref1[i];
        // Same cancellation-safe denominator as the group section (and the
        // repo convention, test_mma_q6k): near-zero refs blow up a pure
        // relative error.
        const double denom =
            std::fabs(r) > 0.01 * scale_max ? std::fabs(r) : 0.01 * scale_max;
        const double rel = std::fabs(__half2float(got[i]) - r) / denom;
        if (rel > 2e-2) {
          ++bad1;
          if (rel > w1worst) {
            w1worst = rel;
            printf("  single M=%d elem %zu: ref=%.3f got=%.3f (scale %.1f, rel %.2e)\n",
                   M, i, r, __half2float(got[i]), scale_max, rel);
          }
        }
      }
      std::snprintf(name, sizeof(name), "M=%d q4k single ncols=%d (%d bad)", M,
                    ncols, bad1);
      Report(name, w1worst <= 2e-2, w1worst);
      cudaFree(d_w1);
      cudaFree(d_o1);
    }

    for (int p = 0; p < 3; ++p) {
      cudaFree(const_cast<void *>(dw[p]));
      cudaFree(dout[p]);
    }
    cudaFree(d_a);
  }

  printf("%s (%d failures)\n", g_failures ? "FAILED" : "ALL PASS", g_failures);
  return g_failures ? 1 : 0;
}
