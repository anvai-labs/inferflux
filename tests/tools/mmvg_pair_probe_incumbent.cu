#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "runtime/backends/cuda/native/kernels/mmvq.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace inferflux::runtime::cuda::native;

// One nonzero weight nibble: qs[BYTE] = NIB<<4 | 0 (lo) or 0x0N (hi).
// One nonzero activation: block ACT_BLK value ACT_E = 1. out reveals the
// pairing: g != 0 iff the kernel pairs that nibble with that value.
int main() {
  cudaFree(0);
  const int N = 1, K = 2048, M = 1;
  std::vector<block_q4_k> w(K / 256);
  std::vector<block_q8_1> a(K / 32);
  auto reset = [&] {
    for (auto &b : w) {
      std::memset(&b, 0, sizeof(b));
      for (int j = 0; j < 4; ++j) { b.scales[j] = 1; b.scales[j + 8] = 1; }
      const half d = __float2half(1.0f), dm = __float2half(0.0f);
      std::memcpy(&b.d, &d, 2); std::memcpy(&b.dmin, &dm, 2);
    }
    for (auto &blk : a) {
      blk.ds = make_half2(__float2half(1.0f), __float2half(0.0f));
      for (int i = 0; i < 32; ++i) blk.qs[i] = 0;
    }
  };
  block_q4_k *dw; block_q8_1 *da; half *dout;
  cudaMalloc(&dw, w.size()*sizeof(block_q4_k));
  cudaMalloc(&da, a.size()*sizeof(block_q8_1));
  cudaMalloc(&dout, N*sizeof(half));

  // Ground truth from the layout: byte B lo -> value (B%32) of q8 blk B/32;
  // hi -> value (B%32) of blk B/32+1 (sub 2*(B/32) / +1 => q8 blk 2*(B/32)).
  for (int B : {5, 40}) {
    for (int hi : {0, 1}) {
      for (int act_blk = 0; act_blk < 8; ++act_blk) {
        reset();
        w[0].qs[B] = hi ? 0x30 : 0x03; // nibble value 3
        a[act_blk].qs[B % 32] = 1;
        a[act_blk].ds = make_half2(__float2half(1.0f), __float2half(1.0f));
        cudaMemcpy(dw, w.data(), w.size()*sizeof(block_q4_k), cudaMemcpyHostToDevice);
        cudaMemcpy(da, a.data(), a.size()*sizeof(block_q8_1), cudaMemcpyHostToDevice);
        inferflux_mmvq_q4k_fused_gate_up_silu<1><<<dim3(N,1), 128, 0>>>(
            dw, dw, da, dout, N, K, M);
        cudaDeviceSynchronize();
        half o; cudaMemcpy(&o, dout, 2, cudaMemcpyDeviceToHost);
        const float g = __half2float(o) > 0 ? 3.0f : 0.0f; // silu(g)*g with g=3 -> 2.83
        printf("byte=%2d %s nibble, act blk=%d val=%d -> %s (expect blk=%d)\n",
               B, hi ? "HI" : "LO", act_blk, B % 32,
               __half2float(o) > 0.5f ? "HIT " : "miss", 2 * (B / 32) + hi);
      }
    }
  }
  return 0;
}
