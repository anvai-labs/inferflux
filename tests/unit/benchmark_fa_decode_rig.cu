// FA decode kernel-vs-kernel rig (Qwen2.5-3B geometry).
//
// Measures the shipped decode attention entries against an mma.m16n8k16
// candidate (llama.cpp flash_attn_ext_f16 decode shape: GQA heads packed
// into the mma N dimension, 4 warps splitting each 128-row KV chunk,
// fp16 PV accumulation with online-softmax rescale) on identical inputs.
// Follows the benchmark_q4k_mmq_rig protocol: cold-L2 primary (96 MB
// eviction before every timed iteration), warm-L2 secondary, cudaEvent
// timing, FPU reference on device, exit code 1 on any gate failure.
//
// Debugging notes from bring-up (do not re-learn these):
//   - Within-warp row offsets (warp*32) must be applied to the K/V
//     fragment loads; the mma itself stays silent when all 4 warps
//     compute the same rows (only the merge result is wrong).
//   - The merge dump accumulates the two 16-row groups: rg1 must ADD to
//     rg0's slot, not overwrite it.
//   - Q rows must be indexed by (b, kvh*kGqa + n); a missing kvh offset
//     only shows up in kv-head-1 blocks.
//
// Build: standalone target benchmark_fa_decode_rig (see CMakeLists.txt).
// Run:   ./build-cuda/benchmark_fa_decode_rig

#include "runtime/backends/cuda/kernels/flash_attention.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using inferflux::cuda_kernel::FlashDecodeMultiSeqStrided;
using inferflux::cuda_kernel::FlashDecodePacked;

namespace {

constexpr int kB = 16;          // sequences (decode batch)
constexpr int kH = 16;          // Q heads
constexpr int kKVH = 2;         // KV heads
constexpr int kD = 128;         // head dim
constexpr int kGqa = kH / kKVH; // 8
constexpr int kSplitsCap = 8;   // workspace sized for up to 8 splits
constexpr int kWarmup = 20;
constexpr int kIters = 100;

constexpr int kThreads = 128;      // 4 warps
constexpr int kChunk = 128;        // KV rows per chunk per block
constexpr int kRowStride = kD + 8; // padded tile row stride in halves
constexpr int kPStride = 9;        // P transpose tile row stride in halves

// mma fragment geometry (per warp, within its 32-row slice of a chunk):
//   QK^T: 2 row-groups (16 rows each) x 8 k-steps, f32 accum
//   PV  : 8 v-tiles (16 dims each) x 2 row-groups, f16 accum
// D-fragment element i (f32 or f16 m16n8) sits at
//   row = lane/4 + 8*(i/2), col = 2*(lane%4) + (i%2).
// B fragment (m16n8k16, col-major [16k x 8n]) register j packs
//   {B[(lane%4)*2 + 8*j, lane/4], B[(lane%4)*2 + 1 + 8*j, lane/4]}.

__device__ __forceinline__ uint32_t PackHalfs(float lo, float hi) {
  __half2 h2 = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t *>(&h2);
}

__device__ __forceinline__ void MmaQk(int (&d)[4], const int (&a)[4],
                                      const int (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void MmaPv(int (&d)[2], const int (&a)[4],
                                      const int (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Flush rescale factors below 2^-20 (llama.cpp SOFTMAX_FTZ_THRESHOLD) so
// the fp16 accumulator never enters the denormal range.
__device__ __forceinline__ float Ftz(float x) {
  return x < 9.53674316e-7f ? 0.f : x;
}

// K A-fragment: [16 rows x 16 dims] of the padded K tile (row-major).
__device__ __forceinline__ void LoadKFrag(int (&a)[4], const half *s_k,
                                          int row0, int k0) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int b = (lane % 4) * 2;
  a[0] = *reinterpret_cast<const int *>(s_k + (row0 + g) * kRowStride + k0 + b);
  a[1] = *reinterpret_cast<const int *>(s_k + (row0 + g + 8) * kRowStride + k0 +
                                        b);
  a[2] = *reinterpret_cast<const int *>(s_k + (row0 + g) * kRowStride + k0 + b +
                                        8);
  a[3] = *reinterpret_cast<const int *>(s_k + (row0 + g + 8) * kRowStride + k0 +
                                        b + 8);
}

// A = V^T view: element (r, c) = V[row0 + c, d0 + r]. Eight smem half
// loads packed into four registers (ldmatrix.trans is the v2 upgrade).
__device__ __forceinline__ void LoadVTFrag(int (&a)[4], const half *s_v,
                                           int row0, int d0) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int b = lane % 4;
  a[0] = PackHalfs(__half2float(s_v[(row0 + 2 * b) * kRowStride + d0 + g]),
                   __half2float(s_v[(row0 + 2 * b + 1) * kRowStride + d0 + g]));
  a[1] = PackHalfs(
      __half2float(s_v[(row0 + 2 * b) * kRowStride + d0 + g + 8]),
      __half2float(s_v[(row0 + 2 * b + 1) * kRowStride + d0 + g + 8]));
  a[2] = PackHalfs(__half2float(s_v[(row0 + 2 * b + 8) * kRowStride + d0 + g]),
                   __half2float(s_v[(row0 + 2 * b + 9) * kRowStride + d0 + g]));
  a[3] = PackHalfs(
      __half2float(s_v[(row0 + 2 * b + 8) * kRowStride + d0 + g + 8]),
      __half2float(s_v[(row0 + 2 * b + 9) * kRowStride + d0 + g + 8]));
}

// Candidate mma decode attention. Grid (batch, kv_heads, splits), 128
// threads; every warp owns 32 of each 128-row chunk with its own online
// softmax state; warps merge through smem at the end. Partial layout
// matches FlashDecodeGQASplitKernel; the launcher always runs the
// combine (num_splits == 1 degenerates to a plain normalize).
__global__ void FlashDecodeMmaGqaKernel(
    const half *__restrict__ Q, const half *__restrict__ kv_buffer,
    float *__restrict__ partial_O, float *__restrict__ partial_max,
    float *__restrict__ partial_sum, const int *__restrict__ seq_ids,
    const int *__restrict__ kv_lens, int num_splits, size_t slot_stride,
    size_t kv_stride, float scale) {
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  const int kv_len = kv_lens[b];
  const int split_len = (kv_len + num_splits - 1) / num_splits;
  const int kv_begin = split * split_len;
  const int kv_end = min(kv_begin + split_len, kv_len);

  extern __shared__ char smem_raw[];
  half *s_k = reinterpret_cast<half *>(smem_raw);
  half *s_v = s_k + kChunk * kRowStride;
  half *s_p = s_v + kChunk * kRowStride; // [4][32][kPStride] P transpose
  float *merge = reinterpret_cast<float *>(
      s_p + 4 * 32 * kPStride); // 4192 floats, see layout below

  if (kv_begin >= kv_end) {
    // Empty split: identity partials (combine-kernel contract).
    if (tid < kGqa) {
      const int split_base = ((b * kKVH + kvh) * num_splits + split) * kGqa;
      partial_max[split_base + tid] = -INFINITY;
      partial_sum[split_base + tid] = 0.f;
    }
    return;
  }

  const int g = b * kH + kvh * kGqa; // Q row base: this kv head's heads
  const size_t kvh_base = static_cast<size_t>(seq_ids[b]) * slot_stride +
                          static_cast<size_t>(kvh) * kD;

  // Q register fragments (B operand of QK^T; invariant over KV).
  // B[k, n] = Q[n, k] scaled: B[0] = {Q[n, k0], Q[n, k0+1]}.
  int q_b[8][2];
  {
    const int bb = (lane % 4) * 2;
    const int n = lane / 4;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k0 = ks * 16 + bb;
      q_b[ks][0] = PackHalfs(__half2float(Q[(g + n) * kD + k0]) * scale,
                             __half2float(Q[(g + n) * kD + k0 + 1]) * scale);
      q_b[ks][1] = PackHalfs(__half2float(Q[(g + n) * kD + k0 + 8]) * scale,
                             __half2float(Q[(g + n) * kD + k0 + 9]) * scale);
    }
  }

  // Per-warp online state; each lane tracks the column pair
  // c = 2*(lane%4) + {0, 1} of its warp's 8 heads.
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};
  int vkq[8][2][2]; // [v_tile][row_group][2 regs of half2]
#pragma unroll
  for (int v = 0; v < 8; ++v) {
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      vkq[v][rg][0] = 0;
      vkq[v][rg][1] = 0;
    }
  }

  for (int chunk = kv_begin; chunk < kv_end; chunk += kChunk) {
    const int rows = min(kChunk, kv_end - chunk);
    for (int i = tid; i < kChunk * kD; i += kThreads) {
      const int r = i / kD;
      const int d = i % kD;
      half kv = __float2half(0.f), vv = __float2half(0.f);
      if (r < rows) {
        const size_t off =
            kvh_base + static_cast<size_t>(chunk + r) * (kKVH * kD) + d;
        kv = kv_buffer[off];
        vv = kv_buffer[off + kv_stride]; // V plane follows the K plane
      }
      s_k[r * kRowStride + d] = kv;
      s_v[r * kRowStride + d] = vv;
    }
    __syncthreads();

    // ---- QK^T over this warp's 32 rows ----
    // p[rg*4 + l]: score at row warp*32 + rg*16 + lane/4 + 8*(l/2),
    // col 2*(lane%4) + (l%2).
    float p[8];
    float vmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      int d[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        int a[4];
        LoadKFrag(a, s_k, warp * 32 + rg * 16, ks * 16);
        MmaQk(d, a, q_b[ks]);
      }
      const float *df = reinterpret_cast<const float *>(d);
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        p[rg * 4 + l] = df[l];
        vmax[l % 2] = fmaxf(vmax[l % 2], df[l]);
      }
    }
    // Column max across the 8 lanes sharing each column pair
    // (lanes with equal lane%4 differ in bits 2..4).
#pragma unroll
    for (int mask = 4; mask < 32; mask <<= 1) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        vmax[e] = fmaxf(vmax[e], __shfl_xor_sync(0xffffffffu, vmax[e], mask));
      }
    }

    // Online update per lane column pair.
    const float new_m[2] = {fmaxf(m_run[0], vmax[0]), fmaxf(m_run[1], vmax[1])};
    float rsc[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      rsc[e] = Ftz(__expf(m_run[e] - new_m[e]));
      m_run[e] = new_m[e];
      l_run[e] *= rsc[e];
    }
    {
      const __half2 rs = __floats2half2_rn(rsc[0], rsc[1]);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
          __half2 *d2 = reinterpret_cast<__half2 *>(vkq[v][rg]);
          d2[0] *= rs;
          d2[1] *= rs;
        }
      }
    }

    // Exponentiated P in D-fragment layout; rows past the valid count
    // (zero-padded K) must contribute nothing to the softmax.
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int row = warp * 32 + rg * 16 + lane / 4 + 8 * (l / 2);
        p[rg * 4 + l] = row < rows ? __expf(p[rg * 4 + l] - new_m[l % 2]) : 0.f;
      }
    }
    float esum[2] = {0.f, 0.f};
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      esum[0] += p[rg * 4 + 0] + p[rg * 4 + 2];
      esum[1] += p[rg * 4 + 1] + p[rg * 4 + 3];
    }
#pragma unroll
    for (int mask = 4; mask < 32; mask <<= 1) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        esum[e] += __shfl_xor_sync(0xffffffffu, esum[e], mask);
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      l_run[e] += esum[e];
    }

    // Transpose P through this warp's smem tile so the PV B operand can
    // be read in B-fragment order: s_p[w][row][col] = P[row][col].
    half *pw = s_p + warp * (32 * kPStride);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rg = i / 4;
      const int row = rg * 16 + lane / 4 + 8 * ((i % 4) / 2);
      const int col = (lane % 4) * 2 + (i % 2);
      pw[row * kPStride + col] = __float2half(p[i]);
    }
    __syncwarp();

    // ---- PV: A = V^T, B = P^T ----
    // B[k, n] = P[rg*16 + k, n]: B[j] packs rows (lane%4)*2 + 8*j and +1
    // at column lane/4.
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      const int kb = (lane % 4) * 2;
      const int n = lane / 4;
      int bfr[2];
      bfr[0] = PackHalfs(__half2float(pw[(rg * 16 + kb) * kPStride + n]),
                         __half2float(pw[(rg * 16 + kb + 1) * kPStride + n]));
      bfr[1] = PackHalfs(__half2float(pw[(rg * 16 + kb + 8) * kPStride + n]),
                         __half2float(pw[(rg * 16 + kb + 9) * kPStride + n]));
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        int a[4];
        LoadVTFrag(a, s_v, warp * 32 + rg * 16, v * 16);
        MmaPv(vkq[v][rg], a, bfr);
      }
    }
    __syncthreads();
  }

  // ---- Merge the 4 warps through smem ----
  // merge: m[4][8] | l[4][8] | rf[4][8] | acc[4][8 heads][128 dims].
  float *mg_m = merge;
  float *mg_l = mg_m + 32;
  float *mg_rf = mg_l + 32;
  float *mg_o = mg_rf + 32;

  if (lane / 4 == 0) {
    const int head = (lane % 4) * 2;
    mg_m[warp * 8 + head] = m_run[0];
    mg_m[warp * 8 + head + 1] = m_run[1];
    mg_l[warp * 8 + head] = l_run[0];
    mg_l[warp * 8 + head + 1] = l_run[1];
  }
  // Dump fp16 accumulators to fp32 (D-fragment index map). The two
  // row-groups hold disjoint KV-row contributions: rg0 stores, rg1 adds
  // (each lane addresses the same slot in both).
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const __half *d2 = reinterpret_cast<const __half *>(vkq[v][pass]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int vd = v * 16 + lane / 4 + 8 * (i / 2);
        const int head = (lane % 4) * 2 + (i % 2);
        float *slot = &mg_o[(warp * 8 + head) * kD + vd];
        if (pass == 0) {
          *slot = __half2float(d2[i]);
        } else {
          *slot += __half2float(d2[i]);
        }
      }
    }
  }
  __syncthreads();

  const int split_base = ((b * kKVH + kvh) * num_splits + split) * kGqa;
  if (warp == 0) {
    for (int h = tid; h < kGqa; h += 32) {
      float bm = -INFINITY;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        bm = fmaxf(bm, mg_m[w * 8 + h]);
      }
      float bl = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float f = Ftz(__expf(mg_m[w * 8 + h] - bm));
        mg_rf[w * 8 + h] = f;
        bl += mg_l[w * 8 + h] * f;
      }
      partial_max[split_base + h] = bm;
      partial_sum[split_base + h] = bl;
    }
    __syncwarp();
    for (int i = tid; i < kGqa * kD; i += 32) {
      const int h = i / kD;
      const int d = i % kD;
      float acc = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        acc += mg_o[(w * 8 + h) * kD + d] * mg_rf[w * 8 + h];
      }
      partial_O[(split_base + h) * kD + d] = acc;
    }
  }
}

// Combine across splits (FlashDecodeGQASplitKernel partial layout).
__global__ void CombinePartialsKernel(const float *__restrict__ partial_O,
                                      const float *__restrict__ partial_max,
                                      const float *__restrict__ partial_sum,
                                      half *__restrict__ O, int num_splits) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int kvh = h / kGqa;
  const int hg = h % kGqa;
  const int d = threadIdx.x;
  float bm = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    const int idx = ((b * kKVH + kvh) * num_splits + s) * kGqa + hg;
    bm = fmaxf(bm, partial_max[idx]);
  }
  if (bm == -INFINITY) {
    O[(b * kH + h) * kD + d] = __float2half(0.f);
    return;
  }
  float acc = 0.f, lsum = 0.f;
  for (int s = 0; s < num_splits; ++s) {
    const int idx = ((b * kKVH + kvh) * num_splits + s) * kGqa + hg;
    if (partial_max[idx] == -INFINITY) {
      continue;
    }
    const float w = __expf(partial_max[idx] - bm);
    acc += w * partial_O[idx * kD + d];
    lsum += w * partial_sum[idx];
  }
  O[(b * kH + h) * kD + d] = __float2half(lsum > 0.f ? acc / lsum : 0.f);
}

// FPU reference: one thread per (b, h, d), two-pass fp32 softmax.
__global__ void RefAttentionKernel(const half *__restrict__ Q,
                                   const half *__restrict__ kv_buffer,
                                   half *__restrict__ O,
                                   const int *__restrict__ kv_lens,
                                   size_t slot_stride, size_t kv_stride,
                                   float scale) {
  const int d = threadIdx.x % kD;
  const int h = blockIdx.y;
  const int b = blockIdx.x;
  const int kvh = h / kGqa;
  const int kv_len = kv_lens[b];
  const half *qp = Q + (b * kH + h) * kD;
  const size_t kbase =
      static_cast<size_t>(b) * slot_stride + static_cast<size_t>(kvh) * kD;
  const size_t vbase = kbase + kv_stride;
  const size_t row_stride = static_cast<size_t>(kKVH) * kD;
  float m = -INFINITY;
  for (int t = 0; t < kv_len; ++t) {
    float dot = 0.f;
    for (int e = 0; e < kD; ++e) {
      dot += __half2float(qp[e]) *
             __half2float(kv_buffer[kbase + t * row_stride + e]);
    }
    m = fmaxf(m, dot * scale);
  }
  float lsum = 0.f, acc = 0.f;
  for (int t = 0; t < kv_len; ++t) {
    float dot = 0.f;
    for (int e = 0; e < kD; ++e) {
      dot += __half2float(qp[e]) *
             __half2float(kv_buffer[kbase + t * row_stride + e]);
    }
    const float w = __expf(dot * scale - m);
    lsum += w;
    acc += w * __half2float(kv_buffer[vbase + t * row_stride + d]);
  }
  O[(b * kH + h) * kD + d] = __float2half(lsum > 0.f ? acc / lsum : 0.f);
}

void CudaChecked(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
  }
}

float TimeKernel(const std::function<void()> &launch, bool cold_l2, void *evict,
                 size_t evict_bytes) {
  cudaEvent_t beg, fin;
  CudaChecked(cudaEventCreate(&beg), "event");
  CudaChecked(cudaEventCreate(&fin), "event");
  float total = 0.f;
  for (int i = 0; i < kWarmup + kIters; ++i) {
    if (cold_l2 && evict) {
      CudaChecked(cudaMemsetAsync(evict, i & 0xFF, evict_bytes), "evict");
    }
    CudaChecked(cudaEventRecord(beg), "record");
    launch();
    CudaChecked(cudaEventRecord(fin), "record");
    CudaChecked(cudaEventSynchronize(fin), "sync");
    float ms = 0.f;
    CudaChecked(cudaEventElapsedTime(&ms, beg, fin), "elapsed");
    if (i >= kWarmup) {
      total += ms;
    }
  }
  cudaEventDestroy(beg);
  cudaEventDestroy(fin);
  return (total / kIters) * 1000.f; // mean us
}

constexpr size_t POFloats() {
  return static_cast<size_t>(kB) * kKVH * kSplitsCap * kGqa * kD;
}
constexpr size_t PScalars() {
  return static_cast<size_t>(kB) * kKVH * kSplitsCap * kGqa;
}

int RigMain() {
  cudaDeviceProp prop;
  CudaChecked(cudaGetDeviceProperties(&prop, 0), "props");
  printf("FA decode rig - %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
  printf("geometry: B=%d H=%d KVH=%d D=%d GQA=%d\n\n", kB, kH, kKVH, kD, kGqa);

  const int kvlens[] = {128, 256, 512, 1024, -1}; // -1 = staggered lens
  const size_t max_seq = 1024;
  const size_t kv_dim = kKVH * kD;
  const size_t kv_stride = max_seq * kv_dim; // per K or V plane
  const size_t slot_stride = 2 * kv_stride;  // K then V (single layer)
  const size_t kv_bytes = kB * slot_stride * sizeof(half);
  const size_t q_bytes = kB * kH * kD * sizeof(half);

  std::vector<half> h_q(kB * kH * kD);
  std::vector<half> h_kv(kB * slot_stride);
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  auto rnd = [&seed]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return static_cast<float>(static_cast<int32_t>(seed >> 24) % 2001 - 1000) /
           1000.f;
  };
  for (auto &x : h_q) {
    x = __float2half(rnd());
  }
  for (auto &x : h_kv) {
    x = __float2half(rnd());
  }

  half *d_q, *d_kv, *d_o, *d_ref;
  int *d_lens, *d_ids;
  CudaChecked(cudaMalloc(&d_q, q_bytes), "q");
  CudaChecked(cudaMalloc(&d_kv, kv_bytes), "kv");
  CudaChecked(cudaMalloc(&d_o, q_bytes), "o");
  CudaChecked(cudaMalloc(&d_ref, q_bytes), "ref");
  CudaChecked(cudaMalloc(&d_lens, kB * sizeof(int)), "lens");
  CudaChecked(cudaMalloc(&d_ids, kB * sizeof(int)), "ids");
  CudaChecked(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice),
              "cpy q");
  CudaChecked(cudaMemcpy(d_kv, h_kv.data(), kv_bytes, cudaMemcpyHostToDevice),
              "cpy kv");

  const size_t evict_bytes = 96u << 20;
  void *d_evict = nullptr;
  CudaChecked(cudaMalloc(&d_evict, evict_bytes), "evict");

  void *d_ws = nullptr;
  const size_t ws_bytes = (POFloats() + 2 * PScalars()) * sizeof(float);
  CudaChecked(cudaMalloc(&d_ws, ws_bytes), "ws");
  float *p_o = static_cast<float *>(d_ws);
  float *p_max = p_o + POFloats();
  float *p_sum = p_max + PScalars();

  int fails = 0;
  const float scale = 1.f / sqrtf(static_cast<float>(kD));
  cudaStream_t s = nullptr;

  printf("%-8s %-18s %10s %10s %13s\n", "kvlen", "kernel", "cold_us", "warm_us",
         "max_rel_err");
  for (int kv_len : kvlens) {
    // Staggered case: decode cohorts are EOS-staggered, so per-sequence
    // lengths span the full window rather than matching exactly.
    std::vector<int> h_lens(kB), h_ids(kB);
    for (int b = 0; b < kB; ++b) {
      h_ids[b] = b;
      h_lens[b] =
          (kv_len > 0) ? kv_len : (128 + (b * 113 + 7) % (max_seq - 128));
    }
    CudaChecked(cudaMemcpy(d_lens, h_lens.data(), kB * sizeof(int),
                           cudaMemcpyHostToDevice),
                "lens");
    CudaChecked(cudaMemcpy(d_ids, h_ids.data(), kB * sizeof(int),
                           cudaMemcpyHostToDevice),
                "ids");

    {
      dim3 grid(kB, kH);
      RefAttentionKernel<<<grid, kD>>>(d_q, d_kv, d_ref, d_lens, slot_stride,
                                       kv_stride, scale);
      CudaChecked(cudaGetLastError(), "ref");
      CudaChecked(cudaDeviceSynchronize(), "ref sync");
    }

    auto run_case = [&](const char *name, const std::function<void()> &launch) {
      CudaChecked(cudaMemset(d_o, 0, q_bytes), "zero o");
      launch();
      CudaChecked(cudaGetLastError(), name);
      CudaChecked(cudaDeviceSynchronize(), "sync");
      std::vector<half> got(kB * kH * kD), ref(kB * kH * kD);
      CudaChecked(cudaMemcpy(got.data(), d_o, q_bytes, cudaMemcpyDeviceToHost),
                  "d2h");
      CudaChecked(
          cudaMemcpy(ref.data(), d_ref, q_bytes, cudaMemcpyDeviceToHost),
          "d2h");
      double worst = 0.0;
      for (size_t i = 0; i < got.size(); ++i) {
        const double gv = __half2float(got[i]);
        const double rv = __half2float(ref[i]);
        worst = fmax(worst, fabs(gv - rv) / fmax(fabs(rv), 0.05));
      }
      const float cold_us = TimeKernel(launch, true, d_evict, evict_bytes);
      const float warm_us = TimeKernel(launch, false, nullptr, 0);
      printf("%-8s %-18s %10.1f %10.1f %13.4f %s\n",
             (kv_len > 0 ? std::to_string(kv_len) : "staggered").c_str(), name,
             cold_us, warm_us, worst, worst <= 5e-2 ? "" : "  <-- FAIL");
      if (worst > 5e-2) {
        ++fails;
      }
    };

    const auto strided = [&]() {
      CudaChecked(FlashDecodeMultiSeqStrided<half>(
                      d_q, d_kv, d_o, d_ids, d_lens, 0, kB, kH, kKVH, kD,
                      slot_stride, 2 * kv_stride, kv_stride, scale, s, d_ws,
                      ws_bytes, max_seq),
                  "strided");
    };
    const auto packed = [&]() {
      CudaChecked(FlashDecodePacked<half>(d_q, d_kv, d_o, d_ids, d_lens, 0, kB,
                                          kH, kKVH, kD, slot_stride,
                                          2 * kv_stride, kv_stride, scale, s,
                                          d_ws, ws_bytes, max_seq),
                  "packed");
    };
    const auto mma = [&](int splits) {
      const int smem = (2 * kChunk * kRowStride * 2) + (4 * 32 * kPStride * 2) +
                       static_cast<int>(4192 * sizeof(float));
      static bool configured = false;
      if (!configured) {
        CudaChecked(cudaFuncSetAttribute(
                        FlashDecodeMmaGqaKernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024),
                    "smem attr");
        configured = true;
      }
      FlashDecodeMmaGqaKernel<<<dim3(kB, kKVH, splits), kThreads, smem, s>>>(
          d_q, d_kv, p_o, p_max, p_sum, d_ids, d_lens, splits, slot_stride,
          kv_stride, scale);
      CudaChecked(cudaGetLastError(), "mma kernel");
      CombinePartialsKernel<<<dim3(kB, kH), kD, 0, s>>>(p_o, p_max, p_sum, d_o,
                                                        splits);
      CudaChecked(cudaGetLastError(), "mma combine");
    };

    run_case("strided(default)", strided);
    run_case("packed", packed);
    run_case("mma s1", [&] { mma(1); });
    run_case("mma s2", [&] { mma(2); });
    run_case("mma s4", [&] { mma(4); });
    run_case("mma s8", [&] { mma(8); });
    printf("\n");
  }

  cudaFree(d_q);
  cudaFree(d_kv);
  cudaFree(d_o);
  cudaFree(d_ref);
  cudaFree(d_lens);
  cudaFree(d_ids);
  cudaFree(d_evict);
  cudaFree(d_ws);
  if (fails == 0) {
    printf("ALL GATES PASS\n");
    return 0;
  }
  printf("%d GATE FAILURES\n", fails);
  return 1;
}

} // namespace

int main() { return RigMain(); }
