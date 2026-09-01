/**
 * Numeric correctness test for the prefill FlashAttention-2 path
 * (FlashAttention2Typed) against an FPU reference — the coverage gap that
 * let the garbage-first-token regression through: test_flash_attn.cpp only
 * checks config flags, never math.
 *
 * Shapes pin both dispatch branches at Qwen2.5 geometry (16 Q heads,
 * 2 KV heads, head_dim 128, causal):
 *   query_len 1, 2, 7   -> scalar GQA kernel
 *   query_len >= 16     -> MMA tensor-core kernel
 *
 * Data generation is deliberately bimodal:
 *   - "flat" case: uniform small values (soft softmax).
 *   - "sharp" case: production-like magnitudes AND a planted dominant key
 *     per query row, so softmax is winner-take-all and the OUTPUT is highly
 *     sensitive to any error in the score matrix. The original version of
 *     this test used only flat data — a swapped MMA A-fragment register
 *     that corrupted every score row still passed, because with
 *     near-uniform softmax weights the output is ~mean(V) regardless of
 *     scores. The sharp case fails loudly (wrong winner -> wrong V row)
 *     for any (row, dim) pairing error in the fragment loads.
 *
 * Also covers kv_len > query_len (chunked prefill with n_past > 0), which
 * exercises a different causal-limit path than fresh prefill.
 */

#include "runtime/backends/cuda/kernels/flash_attention.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace inferflux;
using namespace inferflux::cuda_kernel;

namespace {

int g_failures = 0;

uint32_t Lcg(uint32_t &s) { return s = s * 1664525u + 1013904223u; }

// Signed uniform in [-1, 1). The int cast MUST happen before the
// subtraction: Lcg returns uint32_t, so "(Lcg(s) % n) - n/2" wraps to
// ~4.29e9 whenever the modulo lands below n/2 — after the float divide
// that exceeds half range (65504), every such value becomes inf, the FPU
// reference goes NaN, and every "rel > tol" comparison silently evaluates
// false. An earlier revision of this test generated ~49% infs and passed
// unconditionally, which is exactly how a swapped MMA A-fragment register
// reached production.
float SymRand(uint32_t &s) {
  return (static_cast<int>(Lcg(s) % 2000) - 1000) / 1000.0f;
}

void Report(const char *name, bool ok, double detail) {
  printf("%-52s %s (%.3e)\n", name, ok ? "PASS" : "FAIL", detail);
  if (!ok) {
    ++g_failures;
  }
}

enum class DataKind { kFlat, kSharp, kSharpHi };

} // namespace

int main() {
  const int num_heads = 16;
  const int num_kv_heads = 2;
  const int head_dim = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int q_width = num_heads * head_dim;
  const int kv_width = num_kv_heads * head_dim;

  cudaStream_t s;
  cudaStreamCreate(&s);

  struct Case {
    int qlen;
    int kvlen; // > qlen exercises the n_past > 0 causal path
    DataKind kind;
  };
  const Case cases[] = {
      {1, 1, DataKind::kFlat},
      {2, 2, DataKind::kFlat},
      {7, 7, DataKind::kFlat},
      {16, 16, DataKind::kFlat},
      {35, 35, DataKind::kFlat},
      {17, 17, DataKind::kFlat},
      {1, 1, DataKind::kSharp},
      {7, 7, DataKind::kSharp},
      {16, 16, DataKind::kSharp},
      {17, 17, DataKind::kSharp},
      {33, 33, DataKind::kSharp},
      {35, 35, DataKind::kSharp},
      {16, 40, DataKind::kSharp}, // chunked prefill: n_past = 24
      // Winner signal confined to dims 8..15 of each 16-dim chunk: rows
      // 0..7 of every tile lose their winner entirely under the historical
      // A-fragment register swap (their dims 8..15 were attributed to row+8),
      // so the output collapses toward mean(V) and the case fails loudly.
      {16, 16, DataKind::kSharpHi},
      {17, 17, DataKind::kSharpHi},
      {35, 35, DataKind::kSharpHi},
  };

  for (const Case &c : cases) {
    const int qlen = c.qlen, kvlen = c.kvlen;
    uint32_t seed = 77 + qlen * 131 + kvlen +
                    (c.kind == DataKind::kFlat
                         ? 0
                         : (c.kind == DataKind::kSharp ? 7 : 13));

    std::vector<half> hq(static_cast<size_t>(qlen) * q_width);
    std::vector<half> hk(static_cast<size_t>(kvlen) * kv_width);
    std::vector<half> hv(static_cast<size_t>(kvlen) * kv_width);

    if (c.kind == DataKind::kFlat) {
      for (auto &v : hq)
        v = __float2half(SymRand(seed));
      for (auto &v : hk)
        v = __float2half(SymRand(seed));
      for (auto &v : hv)
        v = __float2half(SymRand(seed));
    } else if (c.kind == DataKind::kSharp) {
      // Production-like magnitudes: post-RoPE K spans ~+-90, Q ~+-20, V ~+-2.
      for (auto &v : hk)
        v = __float2half(90.0f * SymRand(seed));
      for (auto &v : hv)
        v = __float2half(2.0f * SymRand(seed));
      for (int i = 0; i < qlen; ++i) {
        for (int h = 0; h < num_heads; ++h) {
          const int kvh = h / (num_heads / num_kv_heads);
          // Plant a dominant key per row: Q[i,h] ~= K[winner]. The winner
          // score is |K|^2-scale, ~10 sigma above any random competitor.
          const int winner = (kvlen - qlen) + i - (i % 3); // causal-legal, varied
          for (int d = 0; d < head_dim; ++d) {
            const float kw = __half2float(
                hk[((size_t)winner * num_kv_heads + kvh) * head_dim + d]);
            const float noise = 2.0f * SymRand(seed);
            hq[((size_t)i * q_width) + h * head_dim + d] =
                __float2half(kw + noise);
          }
        }
      }
    } else { // kSharpHi: winner signal only in dims 8..15 of each 16-chunk
      for (auto &v : hk)
        v = __float2half(90.0f * SymRand(seed));
      for (auto &v : hv)
        v = __float2half(2.0f * SymRand(seed));
      for (int i = 0; i < qlen; ++i) {
        for (int h = 0; h < num_heads; ++h) {
          const int kvh = h / (num_heads / num_kv_heads);
          const int winner = (kvlen - qlen) + i - (i % 3);
          for (int d = 0; d < head_dim; ++d) {
            const float noise = 2.0f * SymRand(seed);
            float v = noise;
            if ((d % 16) >= 8) {
              v += __half2float(
                  hk[((size_t)winner * num_kv_heads + kvh) * head_dim + d]);
            }
            hq[((size_t)i * q_width) + h * head_dim + d] = __float2half(v);
          }
        }
      }
    }

    // Finiteness guard: non-finite inputs would NaN the reference and make
    // every comparison below vacuously false (NaN > tol is false). Fail
    // loudly instead of silently passing.
    {
      int nfinite_bad = 0;
      for (const auto &v : hq)
        nfinite_bad += !std::isfinite(__half2float(v));
      for (const auto &v : hk)
        nfinite_bad += !std::isfinite(__half2float(v));
      for (const auto &v : hv)
        nfinite_bad += !std::isfinite(__half2float(v));
      if (nfinite_bad != 0) {
        char name[64];
        std::snprintf(name, sizeof(name), "qlen=%-2d kvlen=%-2d finite inputs",
                      qlen, kvlen);
        Report(name, false, static_cast<double>(nfinite_bad));
        continue;
      }
    }

    // FPU reference: causal softmax attention with GQA head mapping.
    std::vector<double> ref(static_cast<size_t>(qlen) * q_width, 0.0);
    const int gqa = num_heads / num_kv_heads;
    for (int i = 0; i < qlen; ++i) {
      for (int h = 0; h < num_heads; ++h) {
        const int kvh = h / gqa;
        const int causal_limit = kvlen - qlen + i + 1;
        double maxw = -1e30;
        double w[128];
        for (int j = 0; j < causal_limit; ++j) {
          double dot = 0.0;
          for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(
                       hq[i * q_width + h * head_dim + d]) *
                   __half2float(hk[j * kv_width + kvh * head_dim + d]);
          }
          w[j] = dot * scale;
          if (w[j] > maxw) {
            maxw = w[j];
          }
        }
        double denom = 0.0;
        for (int j = 0; j < causal_limit; ++j) {
          w[j] = std::exp(w[j] - maxw);
          denom += w[j];
        }
        for (int d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int j = 0; j < causal_limit; ++j) {
            acc += w[j] * __half2float(hv[j * kv_width + kvh * head_dim + d]);
          }
          ref[i * q_width + h * head_dim + d] = acc / denom;
        }
      }
    }

    half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, hq.size() * sizeof(half));
    cudaMalloc(&dk, hk.size() * sizeof(half));
    cudaMalloc(&dv, hv.size() * sizeof(half));
    cudaMalloc(&dout, hq.size() * sizeof(half));
    cudaMemcpy(dq, hq.data(), hq.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), hk.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), hv.size() * sizeof(half),
               cudaMemcpyHostToDevice);

    cudaError_t err = FlashAttention2Typed<half>(
        dq, dk, dv, dout, /*batch=*/1, qlen, kvlen, num_heads, num_kv_heads,
        head_dim, scale, /*causal=*/true, s);
    if (err != cudaSuccess) {
      char name[64];
      std::snprintf(name, sizeof(name), "qlen=%d kvlen=%d launch", qlen,
                    kvlen);
      Report(name, false, static_cast<double>(err));
      continue;
    }
    std::vector<half> got(hq.size());
    cudaMemcpy(got.data(), dout, got.size() * sizeof(half),
               cudaMemcpyDeviceToHost);

    double worst = 0.0;
    int nbad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double v = __half2float(got[i]);
      const double r = ref[i];
      const double rel =
          std::fabs(v - r) / (std::fabs(r) > 0.05 ? std::fabs(r) : 0.05);
      if (rel > worst) {
        worst = rel;
      }
      nbad += rel > 5e-2;
    }
    char name[64];
    const char *kindstr = c.kind == DataKind::kFlat  ? "flat"
                          : c.kind == DataKind::kSharp ? "sharp"
                                                       : "sharp_hi";
    std::snprintf(name, sizeof(name),
                  "qlen=%-2d kvlen=%-2d %s (%d bad/%zu)", qlen, kvlen, kindstr,
                  nbad, ref.size());
    Report(name, worst <= 5e-2, worst);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
  }

  printf("%s (%d failures)\n", g_failures ? "FAILED" : "ALL PASS", g_failures);
  return g_failures ? 1 : 0;
}
