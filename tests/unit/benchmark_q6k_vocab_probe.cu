// Vocab-head (lm_head) shape probe for the Q6_K MMA kernel.
//
// The 4i re-profile found the vocab matmul ([M<=16, 2048] x
// [2048, 151936] Q6_K) running at ~1.4 ms/launch in-server, ~4x above
// its ~325 us DRAM floor. This probe measures the SAME production
// kernel in isolation at the vocab shape (plus the down-proj shape as
// a control) across split counts, reporting achieved GB/s so the
// kernel-quality vs launch-conditions question can be answered before
// any specialized kernel is written.
//
// Build: standalone target benchmark_q6k_vocab_probe (CMakeLists.txt).
// Run:   ./build-cuda/benchmark_q6k_vocab_probe

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include "runtime/backends/cuda/native/kernels/mmq_mma.cuh"
#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kK = 2048;
constexpr int kBlocksPerRow = kK / 256; // q6_k: 256 elems per block

// Deterministic weights in a plausible q6_k range; scales/d chosen so no
// NaN/Inf appears (correctness is not the question here — throughput is).
using inferflux::runtime::cuda::native::block_q6_k;
std::vector<block_q6_k> MakeWeights(int rows, uint32_t seed) {
  std::vector<block_q6_k> w(static_cast<size_t>(rows) * kBlocksPerRow);
  auto u8 = [&seed](int span) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return static_cast<unsigned char>(seed % span);
  };
  for (auto &b : w) {
    for (int i = 0; i < 128; ++i) {
      b.ql[i] = static_cast<unsigned char>(u8(192) + 32);
    }
    for (int i = 0; i < 64; ++i) {
      b.qh[i] = static_cast<unsigned char>(u8(256));
    }
    for (int i = 0; i < 16; ++i) {
      b.scales[i] = static_cast<unsigned char>(u8(32) + 1);
    }
    const half d = __float2half(0.05f);
    std::memcpy(&b.d, &d, 2);
  }
  return w;
}

std::vector<half> MakeActs(int m, uint32_t seed) {
  std::vector<half> a(static_cast<size_t>(m) * kK);
  for (auto &x : a) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    x = __float2half(static_cast<float>(seed % 2001 - 1000) / 1000.0f);
  }
  return a;
}

void RunShape(const char *name, int N, int M, int splits) {
  printf("=== %s: N=%d K=%d M=%d ===\n", name, N, kK, M);
  auto w = MakeWeights(N, 0xDEADBEEFu);
  auto acts = MakeActs(M, 0x12345678u);

  // Host-quantize activations to the D4 layout (same as the q6k rig).
  std::vector<inferflux::runtime::cuda::native::BlockQ8_1Mmq> hq(
      static_cast<size_t>(M) * (kK / 128));
  {
    auto hq_at =
        [&](int r,
            int grp) -> inferflux::runtime::cuda::native::BlockQ8_1Mmq & {
      return hq[static_cast<size_t>(grp) * M + r];
    };
    for (int r = 0; r < M; ++r) {
      for (int grp = 0; grp < kK / 128; ++grp) {
        auto &g = hq_at(r, grp);
        for (int sub = 0; sub < 4; ++sub) {
          float amax = 0;
          float vals[32];
          for (int i = 0; i < 32; ++i) {
            const float v = __half2float(
                acts[static_cast<size_t>(r) * kK + grp * 128 + sub * 32 + i]);
            vals[i] = v;
            amax = std::fmax(amax, std::fabs(v));
          }
          const float d_inv = amax > 0 ? 127.0f / amax : 0.0f;
          g.d4[sub] = amax > 0 ? 1.0f / d_inv : 0.0f;
          for (int i = 0; i < 32; ++i) {
            g.qs[sub * 32 + i] =
                static_cast<int8_t>(std::lround(vals[i] * d_inv));
          }
        }
      }
    }
  }

  block_q6_k *d_w;
  inferflux::runtime::cuda::native::BlockQ8_1Mmq *d_a;
  half *d_out;
  float *d_part = nullptr;
  cudaMalloc(&d_w, w.size() * sizeof(block_q6_k));
  cudaMalloc(&d_a, hq.size() *
                       sizeof(inferflux::runtime::cuda::native::BlockQ8_1Mmq));
  const size_t mn = static_cast<size_t>(M) * N;
  cudaMalloc(&d_out, mn * sizeof(half));
  if (splits > 1) {
    cudaMalloc(&d_part, static_cast<size_t>(splits) * mn * sizeof(float));
  }
  cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q6_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, hq.data(),
             hq.size() * sizeof(inferflux::runtime::cuda::native::BlockQ8_1Mmq),
             cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  dim3 grid((N + inferflux::runtime::cuda::native::kMmqY - 1) /
                inferflux::runtime::cuda::native::kMmqY,
            (M + 15) / 16, splits);
  const size_t smem =
      inferflux::runtime::cuda::native::MmqSmemInts(16) * sizeof(int);
  cudaFuncSetAttribute(inferflux::runtime::cuda::native::InferfluxMmqQ6KMma<16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  const dim3 block(32, inferflux::runtime::cuda::native::kMmqMmaWarps, 1);

  auto run = [&]() {
    inferflux::runtime::cuda::native::InferfluxMmqQ6KMma<16>
        <<<grid, block, smem, s>>>(reinterpret_cast<const char *>(d_w), d_a,
                                   d_out, N, kK, M, d_part, splits);
    if (splits > 1) {
      const int rthreads = 256;
      const size_t rblocks = (mn + rthreads - 1) / rthreads;
      inferflux::runtime::cuda::native::
          ReduceMmqKSplit<<<static_cast<int>(rblocks), rthreads, 0, s>>>(
              d_part, d_out, splits, mn);
    }
  };

  for (int i = 0; i < 20; ++i) {
    run();
  }
  cudaStreamSynchronize(s);
  cudaEvent_t beg, fin;
  cudaEventCreate(&beg);
  cudaEventCreate(&fin);
  float total = 0.f;
  int iters = 100;
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(beg, s);
    run();
    cudaEventRecord(fin, s);
    cudaEventSynchronize(fin);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, beg, fin);
    total += ms;
  }
  const float us = total / iters * 1000.f;
  // Bytes: Q6_K weights N*K*210/256 + fp16 acts + fp16 out.
  const double bytes = static_cast<double>(N) * kK * 210.0 / 256.0 +
                       static_cast<double>(M) * kK * 2 +
                       static_cast<double>(M) * N * 2;
  printf("  splits=%-2d grid=(%d,%d,%d)  %8.1f us  %7.1f GB/s\n", splits,
         grid.x, grid.y, grid.z, us, bytes / (us * 1e3));

  cudaEventDestroy(beg);
  cudaEventDestroy(fin);
  cudaStreamDestroy(s);
  cudaFree(d_w);
  cudaFree(d_a);
  cudaFree(d_out);
  if (d_part) {
    cudaFree(d_part);
  }
}

} // namespace

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Q6_K vocab-shape probe - %s (%d SMs)\n", prop.name,
         prop.multiProcessorCount);
  // Control: the down-proj shape (kernel quality known-good).
  RunShape("down (control)", 2048, 16, 6);
  // The vocab head: splits sweep around the single-wave point.
  RunShape("vocab", 151936, 16, 1);
  RunShape("vocab", 151936, 16, 2);
  RunShape("vocab", 151936, 8, 1);
  RunShape("vocab", 151936, 1, 1);
  return 0;
}
