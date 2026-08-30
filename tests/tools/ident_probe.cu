// Step-1 ladder: MmaS8 primitive identity check.
#include "runtime/backends/cuda/native/kernels/mma_tile.cuh"
#include <cstdio>
using namespace inferflux::runtime::cuda::native;

__global__ void ident_test(int *out) {
  namespace m = mma;
  __shared__ int a_s[16 * 4];
  __shared__ int b_s[8 * 4];
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int i = tid; i < 64; i += 64)
    a_s[i] = 0x01010101;
  for (int i = tid; i < 32; i += 64)
    b_s[i] = 0x01010101;
  __syncthreads();
  if (threadIdx.y != 0)
    return; // single warp computes
  m::Tile<16, 4, int> a;
  m::Tile<8, 4, int> b;
  m::Tile<16, 8, int> c;
  // A has no get_i/get_j mapping (fragment comes only from ldmatrix) —
  // all-ones is permutation-invariant for the identity check.
  m::LoadLdmatrix(a, a_s, 4);
  m::LoadGeneric(b, b_s, 4);
  m::MmaS8(c, a, b);
  // C[i][j] = sum_k 1*1 over 4 int8 k = 4 for all i,j
  for (int l = 0; l < c.ne; ++l) {
    const int i = c.get_i(l), j = c.get_j(l);
    out[i * 8 + j] = c.x[l];
  }
}

int main() {
  cudaFree(0);
  int *d;
  cudaMalloc(&d, 16 * 8 * sizeof(int));
  cudaMemset(d, -1, 16 * 8 * sizeof(int));
  ident_test<<<1, 64>>>(d);
  printf("sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  int h[128];
  cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 128; ++i)
    if (h[i] != 4) {
      if (bad < 6)
        printf("C[%d][%d]=%d (want 4)\n", i / 8, i % 8, h[i]);
      ++bad;
    }
  printf("identity check: %d/128 wrong\n", bad);
  return 0;
}
