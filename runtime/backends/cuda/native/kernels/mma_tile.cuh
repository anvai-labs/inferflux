#pragma once

// MMA tile primitives: warp-level fragment containers and the int8 tensor
// core mma.sync for NVIDIA sm_75+ (Turing and later).
//
// Minimal port of the NVIDIA/TURING branch of llama.cpp's ggml-cuda/mma.cuh
// (MIT license, ggml contributors) covering exactly the tiles the MMQ MMA
// path needs: A/B operand fragments <16,8,int>, the <64,2> load view, and
// the C accumulator <16,16,int> in j-major indexing. Fragment layouts and
// lane mappings are the PTX m16n8k16 register semantics — do not modify
// them in isolation; the mma() asm and the get_i/get_j mappings are a
// matched set.

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define INFERFLUX_TURING_MMA 1
#endif

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace mma {

enum DataLayout {
  kIMajor,
  kJMajor,
};

template <int I, int J, typename T, DataLayout DL = kIMajor> struct Tile;

// NVIDIA int-tile fragment (m8n8k16 register semantics). ne = I*J/32
// registers per lane of the owning warp.
template <int I_, int J_>
struct Tile<I_, J_, int, kIMajor> {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I_ * J_ / 32;
  int x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x / 4;
    } else if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return ((l / 2) * 8) + (threadIdx.x / 4);
    } else if constexpr (I == 16 && J == 16) {
      return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
    } else if constexpr (I == 64 && J == 2) {
      // Special load view: a <16,4> region fetched as <16,8>.
      return threadIdx.x % 16;
    } else {
      return -1;
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x % 4;
    } else if constexpr (I == 8 && J == 8) {
      return (l * 4) + (threadIdx.x % 4);
    } else if constexpr (I == 16 && J == 8) {
      return ((threadIdx.x % 4) * 2) + (l % 2);
    } else if constexpr (I == 16 && J == 16) {
      return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
    } else if constexpr (I == 64 && J == 2) {
      return 2 * ((threadIdx.x / 16) % 2) + l;
    } else {
      return -1;
    }
  }
};

// j-major view: same registers, transposed logical indexing.
template <int I_, int J_>
struct Tile<I_, J_, int, kJMajor> {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = Tile<I_, J_, int, kIMajor>::ne;
  int x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    return Tile<I, J, int, kIMajor>::get_j(l);
  }
  static __device__ __forceinline__ int get_j(const int l) {
    return Tile<I, J, int, kIMajor>::get_i(l);
  }
};

// Element-wise fragment load via the lane mapping.
template <int I, int J, typename T, DataLayout DL>
__device__ __forceinline__ void LoadGeneric(Tile<I, J, T, DL> &t,
                                            const T *__restrict__ xs,
                                            int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs[t.get_i(l) * stride + t.get_j(l)];
  }
}

// ldmatrix load for the A operand fragment (m8n8.x2 form used by the
// <16,4,int> tile). Falls back to element-wise loads pre-Turing.
template <typename T>
__device__ __forceinline__ void LoadLdmatrix(Tile<16, 4, T> &t,
                                              const T *__restrict__ xs,
                                              int stride) {
#ifdef INFERFLUX_TURING_MMA
  int *xi = reinterpret_cast<int *>(t.x);
  const int *xs32 = reinterpret_cast<const int *>(xs) +
                     (threadIdx.x % t.I) * stride;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
               : "=r"(xi[0]), "=r"(xi[1])
               : "l"(xs32));
#else
  LoadGeneric(t, xs, stride);
#endif
}

// D(16x8) = A(16x4) x B(8x4) on int8 fragments, accumulate int32 —
// mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32. Register counts:
// a.ne=2, b.ne=1, c.ne=4 (the PTX fragment layout for m16n8k16).
__device__ __forceinline__ void MmaS8(Tile<16, 8, int> &c,
                                      const Tile<16, 4, int> &a,
                                      const Tile<8, 4, int> &b) {
  static_assert(a.ne == 2 && b.ne == 1 && c.ne == 4, "fragment sizes");
#ifdef INFERFLUX_TURING_MMA
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c.x[0]), "+r"(c.x[1]), "+r"(c.x[2]), "+r"(c.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(b.x[0]));
#else
  // sm_75: two m8n8k16 instructions.
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(c.x[0]), "+r"(c.x[1])
      : "r"(a.x[0]), "r"(b.x[0]));
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(c.x[2]), "+r"(c.x[3])
      : "r"(a.x[1]), "r"(b.x[0]));
#endif
#else
  (void)c;
  (void)a;
  (void)b;
#endif
}

} // namespace mma
} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
