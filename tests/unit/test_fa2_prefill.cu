/**
 * Numeric correctness test for the prefill FlashAttention-2 path
 * (FlashAttention2Typed) against an FPU reference — the coverage gap that
 * let the garbage-first-token regression through: test_flash_attn.cpp only
 * checks config flags, never math.
 *
 * Shapes pin both dispatch branches at Qwen2.5 geometry (16 Q heads,
 * 2 KV heads, head_dim 128, causal):
 *   query_len 1, 2, 7   -> scalar GQA kernel
 *   query_len 16, 35    -> MMA tensor-core kernel
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

void Report(const char *name, bool ok, double detail) {
  printf("%-44s %s (%.3e)\n", name, ok ? "PASS" : "FAIL", detail);
  if (!ok) {
    ++g_failures;
  }
}

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

  const int lens[] = {1, 2, 7, 16, 35};
  for (int qlen : lens) {
    const int kvlen = qlen; // fresh prefill: n_past = 0
    uint32_t seed = 77 + qlen;

    std::vector<half> hq(static_cast<size_t>(qlen) * q_width);
    std::vector<half> hk(static_cast<size_t>(kvlen) * kv_width);
    std::vector<half> hv(static_cast<size_t>(kvlen) * kv_width);
    for (auto &v : hq)
      v = __float2half(((Lcg(seed) % 2000) - 1000) / 1000.0f);
    for (auto &v : hk)
      v = __float2half(((Lcg(seed) % 2000) - 1000) / 1000.0f);
    for (auto &v : hv)
      v = __float2half(((Lcg(seed) % 2000) - 1000) / 1000.0f);

    // FPU reference: causal softmax attention with GQA head mapping.
    std::vector<double> ref(static_cast<size_t>(qlen) * q_width, 0.0);
    const int gqa = num_heads / num_kv_heads;
    for (int i = 0; i < qlen; ++i) {
      for (int h = 0; h < num_heads; ++h) {
        const int kvh = h / gqa;
        double maxw = -1e30;
        double w[64];
        for (int j = 0; j <= i; ++j) {
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
        for (int j = 0; j <= i; ++j) {
          w[j] = std::exp(w[j] - maxw);
          denom += w[j];
        }
        for (int d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int j = 0; j <= i; ++j) {
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
      std::snprintf(name, sizeof(name), "qlen=%d launch", qlen);
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
      const double rel = std::fabs(v - r) / (std::fabs(r) > 0.05 ? std::fabs(r) : 0.05);
      if (rel > worst) {
        worst = rel;
      }
      nbad += rel > 5e-2;
    }
    char name[64];
    std::snprintf(name, sizeof(name), "qlen=%-2d vs FPU (%d bad/%zu)", qlen,
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
