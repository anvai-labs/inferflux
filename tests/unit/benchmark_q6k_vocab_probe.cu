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
std::vector<block_q6_k> MakeWeights(int rows, int kK2, uint32_t seed) {
  const int kBlocksPerRow = kK2 / 256;
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

std::vector<half> MakeActs(int m, int kk2, uint32_t seed) {
  std::vector<half> a(static_cast<size_t>(m) * kk2);
  for (auto &x : a) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    x = __float2half(static_cast<float>(seed % 2001 - 1000) / 1000.0f);
  }
  return a;
}

void RunShape(const char *name, int N, int KK, int M, int splits) {
  printf("=== %s: N=%d K=%d M=%d ===\n", name, N, KK, M);
  auto w = MakeWeights(N, KK, 0xDEADBEEFu);
  auto acts = MakeActs(M, KK, 0x12345678u);

  // Host-quantize activations to the D4 layout (same as the q6k rig).
  std::vector<inferflux::runtime::cuda::native::BlockQ8_1Mmq> hq(
      static_cast<size_t>(M) * (KK / 128));
  {
    auto hq_at =
        [&](int r,
            int grp) -> inferflux::runtime::cuda::native::BlockQ8_1Mmq & {
      return hq[static_cast<size_t>(grp) * M + r];
    };
    for (int r = 0; r < M; ++r) {
      for (int grp = 0; grp < KK / 128; ++grp) {
        auto &g = hq_at(r, grp);
        for (int sub = 0; sub < 4; ++sub) {
          float amax = 0;
          float vals[32];
          for (int i = 0; i < 32; ++i) {
            const float v = __half2float(
                acts[static_cast<size_t>(r) * KK + grp * 128 + sub * 32 + i]);
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
                                   d_out, N, KK, M, d_part, splits);
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
  const double bytes = static_cast<double>(N) * KK * 210.0 / 256.0 +
                       static_cast<double>(M) * KK * 2 +
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


// Bandwidth-optimal vocab GEMM (M <= 16): one warp per output row.
// Lanes partition each row's 8 sub-blocks (4 lanes per sub-block, 8
// elements each, ql bytes coalesced). The D4 activations (at most
// 16 x 8 super-blocks x 128 B = 32 KB) are L1/L2-resident after the
// first super-block. fp32 accumulators, warp shfl reduction. No mma
// tile machinery - pure streaming dequant+FMA.
struct ActT16 {  // 16 per-row int8 values for one element (fp16-pad to 16B)
  int8_t v[16];
};

__global__ void VocabGemmQ6KKernel(
    const inferflux::runtime::cuda::native::block_q6_k *__restrict__ w,
    const ActT16 *__restrict__ act_t,
    const inferflux::runtime::cuda::native::BlockQ8_1Mmq *__restrict__ act,
    half *__restrict__ out, int M, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  const int sub = lane >> 2;         // 0..7 sub-block index
  const int g = sub >> 2;            // 0..1 ql/qh group
  const int sb = sub & 3;            // sub within group
  const int ebase = (lane & 3) << 3; // 8 elements per lane
  const int bpr = K >> 8;            // super-blocks per row

  const inferflux::runtime::cuda::native::block_q6_k *wrow =
      w + static_cast<size_t>(row) * bpr;

  float acc[16];
#pragma unroll
  for (int m = 0; m < 16; ++m) {
    acc[m] = 0.f;
  }

  for (int blk = 0; blk < bpr; ++blk) {
    const inferflux::runtime::cuda::native::block_q6_k &b = wrow[blk];
    const float d = __half2float(__ushort_as_half(b.d));
#pragma unroll
    for (int hg = 0; hg < 2; ++hg) {
      const float scale =
          static_cast<float>(b.scales[(hg << 3) | (sb << 1) | (ebase >> 4)]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float wv = inferflux::runtime::cuda::native::dequant_q6k_element(
                             b, d, hg, sb, ebase + i) *
                         scale;
        const int elem = blk * 256 + hg * 128 + sb * 32 + ebase + i;
        // One uint4 carries the element's 16 per-row quantized values.
        const uint4 packed =
            *reinterpret_cast<const uint4 *>(&act_t[elem]);
        const int8_t *av = reinterpret_cast<const int8_t *>(&packed);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
          const int grp = elem / 128;
          const float aval =
              static_cast<float>(av[m]) * __half2float(act[grp * M + m].d4[sb]);
          acc[m] += wv * aval;
        }
      }
    }
  }

  // Warp reduce the 4 lanes sharing each sub-block position.
#pragma unroll
  for (int m = 0; m < 16; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane % 4 == 0) {
      // 8 sub-block owners (lanes 0,4,8,...28) hold the 8 partial sums;
      // combine across sub-blocks through smem-free final pass: each of
      // the 4 lane-groups wrote one of 4 lanes... use shfl across the
      // sub-block lanes (offsets 4, 8, 16).
    }
    acc[m] = v;
  }
#pragma unroll
  for (int m = 0; m < 16; ++m) {
    float v = acc[m];
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    if (lane == 0) {
      out[static_cast<size_t>(m) * gridDim.x * (blockDim.x >> 5) + row] =
          __float2half(v);
    }
  }
}

} // namespace

int RunStreamingVocab(int N, int M) {
  auto w = MakeWeights(N, kK, 0xDEADBEEFu);
  auto acts = MakeActs(M, kK, 0x12345678u);
  std::vector<inferflux::runtime::cuda::native::BlockQ8_1Mmq> hq(
      static_cast<size_t>(M) * (kK / 128));
  {
    auto hq_at = [&](int r, int grp) ->
        inferflux::runtime::cuda::native::BlockQ8_1Mmq & {
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
  ActT16 *d_at;
  half *d_out, *d_ref;
  // Element-major transpose: act_t[elem].v[m] = qs value of (elem, m).
  std::vector<ActT16> at(static_cast<size_t>(kK));
  for (int grp = 0; grp < kK / 128; ++grp) {
    for (int m = 0; m < M; ++m) {
      const auto &g = hq[static_cast<size_t>(grp) * M + m];
      for (int sub = 0; sub < 4; ++sub) {
        for (int i = 0; i < 32; ++i) {
          const int elem = grp * 128 + sub * 32 + i;
          at[elem].v[m] = g.qs[sub * 32 + i];
        }
      }
    }
  }
  cudaMalloc(&d_w, w.size() * sizeof(block_q6_k));
  cudaMalloc(&d_a, hq.size() * sizeof(inferflux::runtime::cuda::native::
                                         BlockQ8_1Mmq));
  cudaMalloc(&d_at, at.size() * sizeof(ActT16));
  const size_t mn = static_cast<size_t>(M) * N;
  cudaMalloc(&d_out, mn * sizeof(half));
  cudaMalloc(&d_ref, mn * sizeof(half));
  cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q6_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, hq.data(), hq.size() * sizeof(inferflux::runtime::cuda::
                                                   native::BlockQ8_1Mmq),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_at, at.data(), at.size() * sizeof(ActT16),
             cudaMemcpyHostToDevice);
  cudaStream_t s;
  cudaStreamCreate(&s);
  const int warps_per_block = 8;
  const int grid_x = N / warps_per_block;
  dim3 grid(grid_x, 1, 1);
  const dim3 block(warps_per_block * 32, 1, 1);
  auto run = [&]() {
    VocabGemmQ6KKernel<<<grid, block, 0, s>>>(d_w, d_at, d_a, d_out, M, kK);
  };
  for (int i = 0; i < 20; ++i) {
    run();
  }
  cudaStreamSynchronize(s);
  cudaEvent_t beg, fin;
  cudaEventCreate(&beg);
  cudaEventCreate(&fin);
  float total = 0.f;
  const int iters = 100;
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
  const double bytes =
      static_cast<double>(N) * kK * 210.0 / 256.0 +
      static_cast<double>(M) * kK * 2 + static_cast<double>(M) * N * 2;
  printf("  streaming M=%-2d grid=%d  %8.1f us  %7.1f GB/s\n", M, grid.x, us,
         bytes / (us * 1e3));
  // Correctness vs the production Q6_K MMA kernel.
  {
    dim3 mgrid((N + inferflux::runtime::cuda::native::kMmqY - 1) /
                   inferflux::runtime::cuda::native::kMmqY,
               (M + 15) / 16, 1);
    const size_t smem =
        inferflux::runtime::cuda::native::MmqSmemInts(16) * sizeof(int);
    cudaFuncSetAttribute(
        inferflux::runtime::cuda::native::InferfluxMmqQ6KMma<16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    inferflux::runtime::cuda::native::InferfluxMmqQ6KMma<16>
        <<<mgrid, dim3(32, inferflux::runtime::cuda::native::kMmqMmaWarps, 1),
           smem, s>>>(reinterpret_cast<const char *>(d_w), d_a, d_ref, N, kK,
                      M, nullptr, 1);
    cudaStreamSynchronize(s);
  }
  std::vector<half> got(mn), ref(mn);
  cudaMemcpy(got.data(), d_out, mn * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), d_ref, mn * sizeof(half), cudaMemcpyDeviceToHost);
  double worst = 0.0;
  for (size_t i = 0; i < mn; ++i) {
    worst = std::fmax(
        worst, std::fabs(__half2float(got[i]) - __half2float(ref[i])) /
                   std::fmax(std::fabs(__half2float(ref[i])), 0.05));
  }
  printf("  streaming vs MMA max_rel %.4f %s\n", worst,
         worst <= 5e-2 ? "(PASS)" : "(FAIL)");
  cudaEventDestroy(beg);
  cudaEventDestroy(fin);
  cudaStreamDestroy(s);
  cudaFree(d_w);
  cudaFree(d_a);
  cudaFree(d_at);
  cudaFree(d_out);
  cudaFree(d_ref);
  return worst <= 5e-2 ? 0 : 1;
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Q6_K vocab-shape probe - %s (%d SMs)\n", prop.name,
         prop.multiProcessorCount);
  // Control: the down-proj shape (kernel quality known-good).
  RunShape("down real (K=11008)", 2048, 11008, 16, 6);
  // The vocab head: splits sweep around the single-wave point.
  RunShape("vocab", 151936, kK, 16, 1);
  RunShape("vocab", 151936, kK, 16, 2);
  RunShape("vocab", 151936, kK, 8, 1);
  RunShape("vocab", 151936, kK, 1, 1);
  int fails = RunStreamingVocab(151936, 16);
  return fails;
}
