/**
 * Isolated benchmark for the Q4_K gate+up projection family — the dominant
 * decode kernel family (inferflux_mmvq_q4k_fused_gate_up_silu, 45-51% of
 * c16 GPU time per nsys). Measures the production fused path
 * (FusedQuantGemm::FusedGateUpSiluGemvQ8_1) against the bandwidth floor
 * (both weight matrices read once) at decode batch sizes M 1-16, plus an
 * FPU correctness spot-check of the fused output.
 *
 * Geometry: Qwen 2.5 3B — N (intermediate) = 11008, K (hidden) = 2048.
 * Floor = 2 * N * (K/256) * sizeof(block_q4_k) / bandwidth.
 */

#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using inferflux::runtime::cuda::native::block_q4_k;
using inferflux::runtime::cuda::native::block_q8_1;

constexpr int kIters = 100;
constexpr int kWarmup = 20;
constexpr int kN = 11008; // intermediate size
constexpr int kK = 2048;  // hidden size

uint32_t Lcg(uint32_t &s) { return s = s * 1664525u + 1013904223u; }

double BenchMs(cudaEvent_t start, cudaEvent_t stop) {
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  return ms / kIters;
}

// Host Q4_K dequant reference — 6-bit k4 scales across 12 bytes
// (get_scale_min_k4 scheme; see K_SCALE_SIZE=12).
void ScaleMinK4(const unsigned char *q, int j, int *sc, int *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

float Q4KValue(const block_q4_k &b, int e) {
  const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
  const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));
  const int sb = e / 32;
  int sc = 0, m = 0;
  ScaleMinK4(b.scales, sb, &sc, &m);
  const unsigned char qbyte = b.qs[(sb / 2) * 32 + (e % 32)];
  const int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0xF);
  return d * sc * q - dmin * m;
}

} // namespace

int main() {
  using namespace inferflux;
  using namespace inferflux::runtime::cuda::native;
  using inferflux::runtime::cuda::native::BlockQ8_1MmqDs;
  using inferflux::runtime::cuda::native::InferfluxMmqQ4KMma;
  using inferflux::runtime::cuda::native::QuantizeRowQ8_1MmqDsKernel;
  using inferflux::runtime::cuda::native::block_q8_1;
  constexpr int kWideWarps = 4; // incumbent launch: (128,1,1) blocks
  cudaFree(0);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  const double bw_gbs = 360.0; // RTX 4000 Ada effective (measured floor)
  printf("device: %s  SMs=%d  ~%.0f GB/s\n", prop.name, prop.multiProcessorCount,
         bw_gbs);

  uint32_t seed = 4242;
  const int blocks_per_row = kK / 256;

  // Two Q4_K weight matrices: gate and up, [kN, kK].
  std::vector<block_q4_k> host_w(2 * static_cast<size_t>(kN) * blocks_per_row);
  for (auto &b : host_w) {
    for (int i = 0; i < QK_K / 2; ++i)
      b.qs[i] = Lcg(seed) & 0xFF;
    // Valid 6-bit k4 scale encoding across 12 bytes: bytes 0-3 = sc0-3
    // (6-bit), bytes 4-7 = m0-3 (6-bit), bytes 8-11 = sc4-7 low nibbles
    // (high 2 bits of sc4-7 live in bytes 0-3 bit 6-7, kept 0 here).
    for (int i = 0; i < K_SCALE_SIZE; ++i)
      b.scales[i] = 0;
    for (int j = 0; j < 4; ++j) {
      b.scales[j] = 1 + (Lcg(seed) % 60);      // sc0-3, top 2 bits 0
      b.scales[j + 4] = Lcg(seed) % 60;        // m0-3
      b.scales[j + 8] = 1 + (Lcg(seed) % 15);  // sc4-7 low nibbles
    }
    const half d = __float2half(0.002f);
    const half dmin = __float2half(0.001f);
    std::memcpy(&b.d, &d, 2);
    std::memcpy(&b.dmin, &dmin, 2);
  }

  block_q4_k *d_gate = nullptr, *d_up = nullptr;
  cudaMalloc(&d_gate, host_w.size() / 2 * sizeof(block_q4_k));
  cudaMalloc(&d_up, host_w.size() / 2 * sizeof(block_q4_k));
  cudaMemcpy(d_gate, host_w.data(), host_w.size() / 2 * sizeof(block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_up, host_w.data() + host_w.size() / 2,
             host_w.size() / 2 * sizeof(block_q4_k), cudaMemcpyHostToDevice);

  QuantizedWeightInfo gate_info{d_gate, static_cast<int>(GGUF::TensorType::Q4_K),
                                static_cast<int64_t>(kN) * kK};
  QuantizedWeightInfo up_info{d_up, static_cast<int>(GGUF::TensorType::Q4_K),
                              static_cast<int64_t>(kN) * kK};

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t s;
  cudaStreamCreate(&s);

  const double floor_us = 2.0 * kN * blocks_per_row * sizeof(block_q4_k) /
                          (bw_gbs * 1e3);
  printf("geometry: N=%d K=%d  weights=%.1f MB  floor=%.1f us (both mats)\n\n",
         kN, kK, 2.0 * kN * blocks_per_row * sizeof(block_q4_k) / 1e6, floor_us);

  for (int M : {1, 4, 8, 16}) {
    std::vector<half> acts(static_cast<size_t>(M) * kK);
    for (auto &v : acts)
      v = __float2half((static_cast<int>(Lcg(seed) % 2001) - 1000) / 4000.0f);
    half *d_act = nullptr;
    cudaMalloc(&d_act, acts.size() * sizeof(half));
    cudaMemcpy(d_act, acts.data(), acts.size() * sizeof(half),
               cudaMemcpyHostToDevice);

    const size_t q8_blocks = static_cast<size_t>(M) * (kK / 32);
    void *d_act_q8 = nullptr;
    cudaMalloc(&d_act_q8, q8_blocks * 36);
    half *d_out = nullptr;
    cudaMalloc(&d_out, static_cast<size_t>(M) * kN * sizeof(half));

    // Quantize activations once (Q8_1, 36B blocks).
    FusedQuantGemm::QuantizeRowQ8_1(d_act, d_act_q8, M, kK, s);

    // Production fused gate+up+SiLU path.
    for (int i = 0; i < kWarmup; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1(gate_info, up_info, d_act_q8,
                                              d_out, M, kN, kK, s);
    }
    cudaEventRecord(start, s);
    for (int i = 0; i < kIters; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1(gate_info, up_info, d_act_q8,
                                              d_out, M, kN, kK, s);
    }
    cudaEventRecord(stop, s);
    const double fused_ms = BenchMs(start, stop);

    // Wide-load variant (uint4 weight loads).
    {
      using namespace inferflux::runtime::cuda::native;
      const dim3 grid(kN, M);
      const dim3 block(kWideWarps * 32);
      auto run_wide = [&] {
        if (M <= 1) {
          inferflux_mmvq_q4k_fused_gate_up_silu_wide<1>
              <<<grid, block, 0, s>>>(
                  static_cast<const block_q4_k *>(gate_info.data),
                  static_cast<const block_q4_k *>(up_info.data),
                  static_cast<const block_q8_1 *>(d_act_q8), d_out, kN, kK, M);
        } else if (M <= 2) {
          const dim3 g2(kN, (M + 1) / 2);
          inferflux_mmvq_q4k_fused_gate_up_silu_wide<2>
              <<<g2, block, 0, s>>>(
                  static_cast<const block_q4_k *>(gate_info.data),
                  static_cast<const block_q4_k *>(up_info.data),
                  static_cast<const block_q8_1 *>(d_act_q8), d_out, kN, kK, M);
        } else if (M <= 4) {
          const dim3 g4(kN, (M + 3) / 4);
          inferflux_mmvq_q4k_fused_gate_up_silu_wide<4>
              <<<g4, block, 0, s>>>(
                  static_cast<const block_q4_k *>(gate_info.data),
                  static_cast<const block_q4_k *>(up_info.data),
                  static_cast<const block_q8_1 *>(d_act_q8), d_out, kN, kK, M);
        } else {
          const dim3 g8(kN, (M + 7) / 8);
          inferflux_mmvq_q4k_fused_gate_up_silu_wide<8>
              <<<g8, block, 0, s>>>(
                  static_cast<const block_q4_k *>(gate_info.data),
                  static_cast<const block_q4_k *>(up_info.data),
                  static_cast<const block_q8_1 *>(d_act_q8), d_out, kN, kK, M);
        }
      };
      if (M <= 8) {
        // Reference: incumbent output.
        std::vector<half> ref(static_cast<size_t>(M) * kN);
        FusedQuantGemm::FusedGateUpSiluGemvQ8_1(gate_info, up_info, d_act_q8,
                                                d_out, M, kN, kK, s);
        cudaDeviceSynchronize();
        cudaMemcpy(ref.data(), d_out, ref.size() * sizeof(half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < kWarmup; ++i) run_wide();
        cudaEventRecord(start, s);
        for (int i = 0; i < kIters; ++i) run_wide();
        cudaEventRecord(stop, s);
        cudaDeviceSynchronize();
        const double wide_ms = BenchMs(start, stop);
        printf("M=%-2d  fused wide (hot)  : %7.1f us   (%.2fx floor)\n", M,
               wide_ms * 1000, wide_ms * 1000 / floor_us);
        std::vector<half> got(ref.size());
        cudaMemcpy(got.data(), d_out, got.size() * sizeof(half),
                   cudaMemcpyDeviceToHost);
        int bad = 0, shown = 0;
        double max_rel2 = 0;
        for (size_t i2 = 0; i2 < ref.size(); ++i2) {
          const double r = __half2float(ref[i2]);
          const double g = __half2float(got[i2]);
          const double rel = std::fabs(g - r) / (std::fabs(r) > 1.0 ? r : 1.0);
          max_rel2 = std::max(max_rel2, rel);
          if (rel > 1e-2) {
            ++bad;
            if (shown++ < 6)
              printf("  wide mismatch out[%zu]: ref=%.3f wide=%.3f\n", i2,
                     r, g);
          }
        }
        printf("  wide vs incumbent: %d/%zu bad, max_rel=%.3e\n", bad,
               ref.size(), max_rel2);
      }
    }

    // FPU spot-check at M=1: silu(gate dot) * up dot for sampled columns.
    double max_rel = 0;
    if (M == 1) {
      std::vector<half> out(kN);
      cudaMemcpy(out.data(), d_out, kN * sizeof(half),
                 cudaMemcpyDeviceToHost);
      // Q8_1 reference activations: quantize on host (mirrors kernel input).
      std::vector<block_q8_1> q8(kK / 32);
      cudaMemcpy(q8.data(), d_act_q8, q8.size() * sizeof(block_q8_1),
                 cudaMemcpyDeviceToHost);
      auto act_val = [&](int k) {
        const auto &blk = q8[k / 32];
        // ds = {d, d*sum}: scale is the low half of the packed half2.
        const half2 ds = blk.ds;
        const half d_h = __low2half(ds);
        return __half2float(d_h) * static_cast<int8_t>(blk.qs[k % 32]);
      };
      for (int col : {0, 97, 1000, 5555, 11007}) {
        double g = 0, u = 0;
        for (int k = 0; k < kK; ++k) {
          const block_q4_k &gb =
              host_w[static_cast<size_t>(col) * blocks_per_row + k / 256];
          const block_q4_k &ub =
              host_w[host_w.size() / 2 + static_cast<size_t>(col) *
                                                blocks_per_row + k / 256];
          g += Q4KValue(gb, k % 256) * act_val(k);
          u += Q4KValue(ub, k % 256) * act_val(k);
        }
        const double ref = g / (1.0 + std::exp(-g)) * u;
        const double got = __half2float(out[col]);
        max_rel = std::max(max_rel, std::fabs((got - ref) / ref));
      }
    }

    // Q4_K MMA variant: DS quantize + one MMA GEMM per matrix, K-split
    // only when the N-tile grid under-occupies (N=11008 -> 86 tiles).
    {
      using namespace inferflux::runtime::cuda::native;
      BlockQ8_1MmqDs *d_act_ds = nullptr;
      cudaMalloc(&d_act_ds,
                 static_cast<size_t>(M) * (kK / 128) * sizeof(BlockQ8_1MmqDs));
      dim3 qgrid((kK / 128 + 3) / 4, M);
      QuantizeRowQ8_1MmqDsKernel<<<qgrid, 128, 0, s>>>(d_act, d_act_ds, kK, M);
      const int n_tiles = (kN + kMmqY - 1) / kMmqY;
      for (int ks : {1, 2}) {
        float *d_part = nullptr;
        if (ks > 1)
          cudaMalloc(&d_part, static_cast<size_t>(ks) * M * kN * sizeof(float));
        const size_t smem = MmqSmemInts(16) * sizeof(int);
        cudaFuncSetAttribute(InferfluxMmqQ4KMma<16>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
        const int gate_out = kN; // separate halves of d_out per matrix
        auto run = [&]() {
          for (int mat = 0; mat < 2; ++mat) {
            const char *w = mat ? reinterpret_cast<const char *>(d_up)
                                : reinterpret_cast<const char *>(d_gate);
            dim3 grid(n_tiles, (M + 15) / 16, ks);
            InferfluxMmqQ4KMma<16><<<grid, dim3(32, kMmqMmaWarps, 1), smem, s>>>(
                w, d_act_ds, d_out + mat * gate_out, kN, kK, M, d_part, ks);
            if (ks > 1) {
              const int rt = 256;
              const size_t mn = static_cast<size_t>(M) * kN;
              ReduceMmqKSplit<<<(mn + rt - 1) / rt, rt, 0, s>>>(d_part,
                  d_out + mat * gate_out, ks, mn);
            }
          }
        };
        for (int i = 0; i < kWarmup; ++i) run();
        cudaEventRecord(start, s);
        for (int i = 0; i < kIters; ++i) run();
        cudaEventRecord(stop, s);
        const double mma_ms = BenchMs(start, stop);
        printf("M=%-2d  mma gate+up s=%d   : %7.1f us   (%.2fx floor)\n", M,
               ks, mma_ms * 1000, mma_ms * 1000 / floor_us);
        cudaFree(d_part);
      }
      cudaFree(d_act_ds);
    }

    printf("M=%-2d  fused gate+up+silu : %7.1f us   (%.2fx floor)\n", M,
           fused_ms * 1000, fused_ms * 1000 / floor_us);
    if (M == 1) {
      printf("      M=1 FPU spot-check max_rel = %.3e\n", max_rel);
    }

    cudaFree(d_act);
    cudaFree(d_act_q8);
    cudaFree(d_out);
  }

  cudaFree(d_gate);
  cudaFree(d_up);
  return 0;
}
