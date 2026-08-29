#pragma once

#include <cublas_v2.h>
// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif

namespace inferflux {

/**
 * CublasGemm: cuBLAS handle manager + typed GEMM with FP32 accumulation.
 *
 * Wraps cublasGemmEx for y = x * W^T (safetensors row-major convention).
 */
class CublasGemm {
public:
  CublasGemm() = default;
  ~CublasGemm();

  CublasGemm(const CublasGemm &) = delete;
  CublasGemm &operator=(const CublasGemm &) = delete;

  /**
   * Create cuBLAS handle and bind to stream.
   */
  bool Initialize(cudaStream_t stream);

  /**
   * Switch cuBLAS handle to a different stream.
   */
  void SetStream(cudaStream_t stream);

  /**
   * FP16 GEMM with FP32 accumulation: C = A * B^T
   *
   * A: [M, K] row-major (half)
   * B: [N, K] row-major (half) — transposed during multiply
   * C: [M, N] row-major (half)
   */
  bool Gemm(int M, int N, int K, const half *A, const half *B, half *C);

  /**
   * Typed GEMM with FP32 accumulation: C = A * B^T
   * Supports half (FP16) and __nv_bfloat16 (BF16) via DtypeTraits.
   */
  template <typename T>
  bool GemmTyped(int M, int N, int K, const T *A, const T *B, T *C);

  /**
   * Typed GEMM with accumulation: C = A * B^T + C (beta = 1.0)
   * Eliminates the need for a separate ResidualAdd kernel when the output
   * buffer already contains the residual to accumulate into.
   */
  template <typename T>
  bool GemmTypedAccum(int M, int N, int K, const T *A, const T *B, T *C);

  /**
   * Strided batched GEMM for GQA attention.
   * Each batch: C_i = A_i * B_i^T
   */
  bool GemmBatched(int M, int N, int K, const half *A, const half *B, half *C,
                   long long stride_A, long long stride_B, long long stride_C,
                   int batch_count);

  /**
   * Typed strided batched GEMM.
   */
  template <typename T>
  bool GemmBatchedTyped(int M, int N, int K, const T *A, const T *B, T *C,
                        long long stride_A, long long stride_B,
                        long long stride_C, int batch_count);

  /**
   * Pre-allocate cuBLAS workspace to avoid dynamic allocation during
   * CUDA graph capture. Must be called after Initialize() and before
   * any graph capture begins.
   */
  bool PreallocateWorkspace(size_t workspace_bytes = 4 * 1024 * 1024);

private:
  cublasHandle_t handle_{nullptr};
  void *workspace_{nullptr};
  size_t workspace_size_{0};
};

} // namespace inferflux
