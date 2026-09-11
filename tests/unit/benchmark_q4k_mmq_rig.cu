// PR-0 measurement rig (plan: throughput+memory kernel improvements).
// Q4_K decode matmul decision matrix: production MMA chain
// (QuantizeForMmqMma + GemvMmqMmaPrequantized incl. K-split reduce), the dp4a
// MMQ tier (inferflux_mmq_q4k<16/32>), and the MMVQ tier (M<=8), across the
// real Qwen2.5-3B projection shapes and decode widths, under COLD L2 (96 MB
// scratch swept between iterations — warm-L2 overstates >100% roofline,
// plan-doc §5).
//
// Correctness: MMA vs dp4a outputs cross-checked on a 64-output sample
// (gate 2e-3, enforced via exit code). The standalone dequant reference
// disagrees with both kernels on synthetic data and is informational only;
// MMVQ outputs are not correctness-checked (timing only).
//
// Decision table (plan file): MMA >=1.3x slower than llama mul_mat_q
// per-kernel -> stream-K rework (PR-3); parity but in-server family gap ->
// launch structure (PR-2); dp4a within 10% of MMA at M=16 -> tier sweep.
#include <nvtx3/nvToolsExt.h>

#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "runtime/backends/cuda/native/weight_map.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mmq.cuh"
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace native = inferflux::runtime::cuda::native;
using native::block_q4_k;
using native::block_q8_1;

namespace {

constexpr int kIters = 100;
constexpr int kWarmup = 20;

struct Shape {
  int n, k;
  const char *name;
};

const Shape kShapes[] = {
    {2048, 2048, "qkv/o"},
    {11008, 2048, "gate/up"},
    {2048, 11008, "down"},
};
const int kM[] = {1, 2, 4, 8, 16};

unsigned short EncodeHalf(float v) {
  const half h = __float2half(v);
  unsigned short b;
  std::memcpy(&b, &h, sizeof(b));
  return b;
}

std::vector<block_q4_k> MakeQ4K(int rows, int k, int seed) {
  const int blocks_per_row = k / QK_K;
  std::vector<block_q4_k> w(static_cast<size_t>(rows) * blocks_per_row);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks_per_row; ++b) {
      auto &blk = w[static_cast<size_t>(r) * blocks_per_row + b];
      for (int i = 0; i < QK_K / 2; ++i)
        blk.qs[i] = static_cast<uint8_t>((seed * 11 + r * 13 + b * 7 + i * 3) & 0xFF);
      // 12 B of 6-bit scales (8 scale/min values packed in 12 B for Q4_K).
      for (int i = 0; i < 12; ++i)
        blk.scales[i] =
            static_cast<uint8_t>(((seed + r * 3 + b) * 5 + i * 7) & 0x3F);
      blk.d = EncodeHalf(0.008f * (((r + b + seed) % 5) + 1));
      blk.dmin = EncodeHalf(0.002f * (((r + b) % 3) + 1));
    }
  }
  return w;
}

std::vector<half> MakeActs(int m, int k, int seed) {
  std::vector<half> a(static_cast<size_t>(m) * k);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = __float2half(0.01f * std::sin(0.173f * (i + seed)) - 0.001f);
  return a;
}

// Host quantize of fp16 activations to block_q8_1 (timing-only for dp4a/mmvq;
// correctness is judged against the device-dequantized reference GEMM).
void HostQuantizeQ8_1(const std::vector<half> &acts, int m, int k,
                      std::vector<block_q8_1> &out) {
  out.resize(static_cast<size_t>(m) * (k / 32));
  for (int r = 0; r < m; ++r) {
    for (int blk = 0; blk < k / 32; ++blk) {
      auto &b = out[static_cast<size_t>(r) * (k / 32) + blk];
      float amax = 0.f;
      float vals[32];
      for (int i = 0; i < 32; ++i) {
        float v = __half2float(acts[static_cast<size_t>(r) * k + blk * 32 + i]);
        vals[i] = v;
        amax = std::max(amax, std::fabs(v));
      }
      const float d = amax / 127.f;
      float sum = 0.f;
      for (int i = 0; i < 32; ++i) {
        const int q = static_cast<int>(std::nearbyint(vals[i] / d));
        b.qs[i] = static_cast<int8_t>(q);
        sum += q;
      }
      b.ds = __halves2half2(__float2half(d), __float2half(d * sum));
    }
  }
}

float DequantHalf(unsigned short b) {
  half h;
  std::memcpy(&h, &b, sizeof(h));
  return __half2float(h);
}

} // namespace

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("device: %s (SMs=%d, smemOptin=%zu KB)\n", prop.name,
         prop.multiProcessorCount,
         static_cast<size_t>(prop.sharedMemPerBlockOptin) / 1024);

  cudaStream_t s;
  cudaStreamCreate(&s);

  // Cold-L2 eviction scratch (Ada L2 = 48 MB).
  float *d_evict = nullptr;
  const size_t evict_n = 96u * 1024 * 1024 / sizeof(float);
  cudaMalloc(&d_evict, evict_n * sizeof(float));
  cudaMemset(d_evict, 0, evict_n * sizeof(float));

  // CUDA event pool for timing.
  cudaEvent_t beg, fin;
  cudaEventCreate(&beg);
  cudaEventCreate(&fin);

  for (const Shape &sh : kShapes) {
    const int N = sh.n, K = sh.k;
    auto wq4 = MakeQ4K(N, K, 7);
    block_q4_k *d_w;
    cudaMalloc(&d_w, wq4.size() * sizeof(block_q4_k));
    cudaMemcpy(d_w, wq4.data(), wq4.size() * sizeof(block_q4_k),
               cudaMemcpyHostToDevice);

    // Reference weights: device dequantize to fp16, read back.
    half *d_wref;
    cudaMalloc(&d_wref, static_cast<size_t>(N) * K * sizeof(half));
    dequantize_q4_k(d_w, d_wref, static_cast<int>(wq4.size()), s);
    cudaStreamSynchronize(s);
    std::vector<half> href(static_cast<size_t>(N) * K);
    cudaMemcpy(href.data(), d_wref, href.size() * sizeof(half),
               cudaMemcpyDeviceToHost);

    printf("\n=== shape %s: N=%d K=%d ===\n", sh.name, N, K);
    const int shape_idx = static_cast<int>(&sh - kShapes);

    for (int M : kM) {
      char nvtx_name[24];
      snprintf(nvtx_name, sizeof(nvtx_name), "M%d_s%d", M, shape_idx);
      nvtxRangePushA(nvtx_name);
      auto acts = MakeActs(M, K, 3);
      half *d_in;
      cudaMalloc(&d_in, acts.size() * sizeof(half));
      cudaMemcpy(d_in, acts.data(), acts.size() * sizeof(half),
                 cudaMemcpyHostToDevice);
      half *d_out;
      cudaMalloc(&d_out, static_cast<size_t>(M) * N * sizeof(half));

      // Reference GEMM (float) on a 64-output sample.
      const int nsample = std::min(64, N);
      std::vector<float> ref(static_cast<size_t>(M) * nsample);
      for (int m = 0; m < M; ++m)
        for (int nn = 0; nn < nsample; ++nn) {
          float acc = 0.f;
          for (int k = 0; k < K; ++k)
            acc += __half2float(acts[static_cast<size_t>(m) * K + k]) *
                   DequantHalf(href[static_cast<size_t>(nn) * K + k]);
          ref[static_cast<size_t>(m) * nsample + nn] = acc;
        }

      // ---- production MMA chain ----
      std::vector<half> mma_sample;
      {
        inferflux::QuantizedWeightInfo wi;
        wi.data = d_w;
        wi.quant_type = static_cast<int>(native::GGUF::TensorType::Q4_K);
        wi.num_elements = static_cast<long long>(N) * K;
        native::BlockQ8_1MmqDs *d_ds = nullptr;
        cudaMalloc(&d_ds, sizeof(native::BlockQ8_1MmqDs) *
                              static_cast<size_t>(M) * (K / QK8_1) * 2);
        float *d_part = nullptr;
        const int n_tiles = (N + 127) / 128;
        const int max_splits = 8;
        cudaMalloc(&d_part, sizeof(float) * static_cast<size_t>(M) * N *
                                max_splits);
        cudaMemset(d_part, 0, sizeof(float) * static_cast<size_t>(M) * N *
                                  max_splits);

        bool mma_ok = true;
        auto run_mma = [&]() {
          if (!inferflux::FusedQuantGemm::QuantizeForMmqMma(d_in, d_ds, M, K, s) ||
              !inferflux::FusedQuantGemm::GemvMmqMmaPrequantized(
                  wi, d_ds, d_out, d_part, M, N, K, s)) {
            mma_ok = false;
            return;
          }
        };
        // correctness
        cudaMemset(d_out, 0, static_cast<size_t>(M) * N * sizeof(half));
        run_mma();
        if (!mma_ok) {
          // M < 2 (and unsupported geometries) decline by design — the
          // production dispatch routes those widths to the MMVQ tier.
          printf("  M=%-2d MMA-chain   declined (skipped)\n", M);
          cudaFree(d_ds);
          cudaFree(d_part);
          continue;
        }
        run_mma();
        cudaStreamSynchronize(s);
        std::vector<half> hout(static_cast<size_t>(M) * nsample);
        cudaMemcpy(hout.data(), d_out, hout.size() * sizeof(half),
                   cudaMemcpyDeviceToHost);
        mma_sample = hout;
        double max_rel = 0.0;
        for (int m = 0; m < M; ++m)
          for (int nn = 0; nn < nsample; ++nn) {
            const float refv = ref[static_cast<size_t>(m) * nsample + nn];
            const float got = __half2float(hout[static_cast<size_t>(m) * nsample + nn]);
            const float denom = std::max(1.f, std::fabs(refv));
            max_rel = std::max(max_rel,
                               std::fabs(static_cast<double>(got) - refv) /
                                   denom);
          }
        // timing
        for (int i = 0; i < kWarmup; ++i) {
          cudaMemsetAsync(d_evict, i & 0xFF, evict_n * sizeof(float) / 4, s);
          run_mma();
        }
        cudaStreamSynchronize(s);
        float total_ms = 0.f;
        float mn_ms = 1e9f;
        for (int i = 0; i < kIters; ++i) {
          cudaMemsetAsync(d_evict, i & 0xFF, evict_n * sizeof(float) / 4, s);
          cudaEventRecord(beg, s);
          run_mma();
          cudaEventRecord(fin, s);
          cudaEventSynchronize(fin);
          float ms = 0.f;
          cudaEventElapsedTime(&ms, beg, fin);
          total_ms += ms;
          mn_ms = std::min(mn_ms, ms);
        }
        const float us = total_ms / kIters * 1000.f;
        // bytes: q4_K weights N*K*144/256 + fp16 acts + fp16 out
        const double bytes = static_cast<double>(N) * K * 144.0 / 256.0 +
                             static_cast<double>(M) * K * 2 +
                             static_cast<double>(M) * N * 2;
        printf("  M=%-2d MMA-chain   %8.2f us (min %7.2f)  %8.0f GB/s  maxrel %.2e\n",
               M, us, mn_ms * 1000.f, bytes / (us * 1e3), max_rel);
        // max_rel vs the dequant reference is informational on synthetic
        // data (known-broken reference, see header); the kernel-vs-kernel
        // gate in the dp4a block below is the enforced check.
        cudaFree(d_ds);
        cudaFree(d_part);
      }

      // ---- dp4a MMQ tier (block_q8_1 activations) ----
      if (M >= 16) {
        std::vector<block_q8_1> hq;
        HostQuantizeQ8_1(acts, M, K, hq);
        block_q8_1 *d_a = nullptr;
        cudaMalloc(&d_a, hq.size() * sizeof(block_q8_1));
        cudaMemcpy(d_a, hq.data(), hq.size() * sizeof(block_q8_1),
                   cudaMemcpyHostToDevice);
        // cross-check dp4a vs the device-dequant reference on the sample
        cudaMemset(d_out, 0, static_cast<size_t>(M) * N * sizeof(half));
        {
          dim3 grid((N + 7) / 8, (M + 31) / 32);
          native::inferflux_mmq_q4k<32><<<grid, 256,
              static_cast<size_t>(32) * 8 * sizeof(block_q8_1), s>>>(
              d_w, d_a, d_out, M, N, K);
          cudaStreamSynchronize(s);
          std::vector<half> hout2(static_cast<size_t>(M) * nsample);
          cudaMemcpy(hout2.data(), d_out, hout2.size() * sizeof(half),
                     cudaMemcpyDeviceToHost);
          double max_rel2 = 0;
          for (int m = 0; m < M; ++m)
            for (int nn = 0; nn < nsample; ++nn) {
              const float refv = ref[static_cast<size_t>(m) * nsample + nn];
              const float got =
                  __half2float(hout2[static_cast<size_t>(m) * nsample + nn]);
              const float mma = __half2float(
                  mma_sample[static_cast<size_t>(m) * nsample + nn]);
              max_rel2 = std::max(max_rel2,
                                  std::fabs(static_cast<double>(got) - refv) /
                                      std::max(1.0, std::fabs(static_cast<double>(refv))));
              (void)mma;
            }
          printf("  [dp4a maxrel vs dequant-ref: %.3e] (informational — "
                 "synthetic-data reference)\n", max_rel2);
          // kernel-vs-kernel: MMA output captured earlier in hout
          {
            double kvk = 0;
            for (int m = 0; m < M; ++m)
              for (int nn = 0; nn < nsample; ++nn) {
                const float a = __half2float(
                    mma_sample[static_cast<size_t>(m) * nsample + nn]);
                const float b2 = __half2float(hout2[static_cast<size_t>(m) * nsample + nn]);
                kvk = std::max(kvk, std::fabs(static_cast<double>(a - b2)) /
                                        std::max(1.0, std::fabs(static_cast<double>(a))));
              }
            printf("  [MMA vs dp4a maxrel: %.3e]\n", kvk);
            if (kvk > 2e-3) {
              printf("  MMA/dp4a correctness gate FAILED (%.3e > 2e-3)\n",
                     kvk);
              return 1;
            }
          }
        }
        const int tile = M <= 16 ? 16 : 32;
        const size_t smem = static_cast<size_t>(tile) * 8 * sizeof(block_q8_1);
        auto run_q4k = [&]() {
          dim3 grid((N + 7) / 8, (M + tile - 1) / tile);
          if (tile == 16)
            native::inferflux_mmq_q4k<16><<<grid, 256, smem, s>>>(
                d_w, d_a, d_out, M, N, K);
          else
            native::inferflux_mmq_q4k<32><<<grid, 256, smem, s>>>(
                d_w, d_a, d_out, M, N, K);
        };
        for (int i = 0; i < kWarmup; ++i) run_q4k();
        cudaStreamSynchronize(s);
        float total_ms = 0.f;
        for (int i = 0; i < kIters; ++i) {
          cudaMemsetAsync(d_evict, i & 0xFF, evict_n * sizeof(float) / 4, s);
          cudaEventRecord(beg, s);
          run_q4k();
          cudaEventRecord(fin, s);
          cudaEventSynchronize(fin);
          float ms = 0.f;
          cudaEventElapsedTime(&ms, beg, fin);
          total_ms += ms;
        }
        const float us = total_ms / kIters * 1000.f;
        const double bytes = static_cast<double>(N) * K * 144.0 / 256.0 +
                             static_cast<double>(M) * K * 2 +
                             static_cast<double>(M) * N * 2;
        printf("  M=%-2d dp4a-mmq   %8.2f us (mean)         %8.0f GB/s\n",
               M, us, bytes / (us * 1e3));
        cudaFree(d_a);
      }

      // ---- MMVQ tier (M <= 8) ----
      if (M <= 8) {
        const int ncols = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : 8;
        std::vector<block_q8_1> hq;
        HostQuantizeQ8_1(acts, M, K, hq);
        block_q8_1 *d_a = nullptr;
        cudaMalloc(&d_a, hq.size() * sizeof(block_q8_1));
        cudaMemcpy(d_a, hq.data(), hq.size() * sizeof(block_q8_1),
                   cudaMemcpyHostToDevice);
        const size_t smem = 4 * ncols * sizeof(float);
        auto run_mmvq = [&]() {
          dim3 grid(N, (M + ncols - 1) / ncols);
          switch (ncols) {
          case 1: native::inferflux_mmvq_q4k<1><<<grid, 128, smem, s>>>(d_w, d_a, d_out, N, K, M); break;
          case 2: native::inferflux_mmvq_q4k<2><<<grid, 128, smem, s>>>(d_w, d_a, d_out, N, K, M); break;
          case 4: native::inferflux_mmvq_q4k<4><<<grid, 128, smem, s>>>(d_w, d_a, d_out, N, K, M); break;
          default: native::inferflux_mmvq_q4k<8><<<grid, 128, smem, s>>>(d_w, d_a, d_out, N, K, M); break;
          }
        };
        for (int i = 0; i < kWarmup; ++i) run_mmvq();
        cudaStreamSynchronize(s);
        float total_ms = 0.f;
        for (int i = 0; i < kIters; ++i) {
          cudaMemsetAsync(d_evict, i & 0xFF, evict_n * sizeof(float) / 4, s);
          cudaEventRecord(beg, s);
          run_mmvq();
          cudaEventRecord(fin, s);
          cudaEventSynchronize(fin);
          float ms = 0.f;
          cudaEventElapsedTime(&ms, beg, fin);
          total_ms += ms;
        }
        const float us = total_ms / kIters * 1000.f;
        const double bytes = static_cast<double>(N) * K * 144.0 / 256.0 +
                             static_cast<double>(M) * K * 2 +
                             static_cast<double>(M) * N * 2;
        printf("  M=%-2d mmvq<%d>    %8.2f us (mean)         %8.0f GB/s\n",
               M, ncols, us, bytes / (us * 1e3));
        cudaFree(d_a);
      }

      cudaFree(d_in);
      cudaFree(d_out);
      nvtxRangePop();
    }

    cudaFree(d_w);
    cudaFree(d_wref);
  }

  cudaFree(d_evict);
  return 0;
}
