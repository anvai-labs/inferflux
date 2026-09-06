#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "server/logging/logger.h"

#include <cublasLt.h>

#include <cstring>

namespace inferflux {

CublasGemm::~CublasGemm() {
  if (lt_workspace_) {
    cudaFree(lt_workspace_);
    lt_workspace_ = nullptr;
  }
  if (lt_handle_) {
    cublasLtDestroy(reinterpret_cast<cublasLtHandle_t>(lt_handle_));
    lt_handle_ = nullptr;
  }
  if (workspace_) {
    cudaFree(workspace_);
    workspace_ = nullptr;
  }
  if (handle_) {
    cublasDestroy(handle_);
  }
}

bool CublasGemm::Initialize(cudaStream_t stream) {
  stream_ = stream;
  cublasStatus_t st = cublasCreate(&handle_);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "cublasCreate failed: " + std::to_string(st));
    return false;
  }

  st = cublasSetStream(handle_, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "cublasSetStream failed: " + std::to_string(st));
    return false;
  }

  st = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Warn("cublas_gemm", "cublasSetMathMode failed, using default");
  }

  log::Info("cublas_gemm", "cuBLAS initialized");
  return true;
}

bool CublasGemm::PreallocateWorkspace(size_t workspace_bytes) {
  if (!handle_)
    return false;
  if (workspace_) {
    cudaFree(workspace_);
    workspace_ = nullptr;
  }
  cudaError_t err = cudaMalloc(&workspace_, workspace_bytes);
  if (err != cudaSuccess) {
    log::Warn("cublas_gemm", "Workspace pre-allocation failed (" +
                                 std::to_string(workspace_bytes / 1024) +
                                 " KB), CUDA graph capture may not work");
    return false;
  }
  workspace_size_ = workspace_bytes;
  cublasStatus_t st = cublasSetWorkspace(handle_, workspace_, workspace_size_);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Warn("cublas_gemm",
              "cublasSetWorkspace failed: " + std::to_string(st));
    cudaFree(workspace_);
    workspace_ = nullptr;
    workspace_size_ = 0;
    return false;
  }
  log::Info("cublas_gemm", "Pre-allocated " +
                               std::to_string(workspace_bytes / 1024) +
                               " KB workspace for CUDA graph capture");
  return true;
}

void CublasGemm::SetStream(cudaStream_t stream) {
  stream_ = stream;
  if (handle_) {
    auto status = cublasSetStream(handle_, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      log::Error("cublas_gemm",
                 "cublasSetStream failed: " + std::to_string(status));
    }
  }
}

bool CublasGemm::Gemm(int M, int N, int K, const half *A, const half *B,
                      half *C) {
  return GemmTyped<half>(M, N, K, A, B, C);
}

template <typename T>
bool CublasGemm::GemmTyped(int M, int N, int K, const T *A, const T *B, T *C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  cublasStatus_t st = cublasGemmEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, dtype, K, A, dtype,
      K, &beta, C, dtype, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmTyped failed: " + std::to_string(st) +
                                  " (M=" + std::to_string(M) +
                                  " N=" + std::to_string(N) +
                                  " K=" + std::to_string(K) + ")");
    return false;
  }
  return true;
}

bool CublasGemm::EnsureLt() {
  if (lt_handle_)
    return true;
  cublasLtHandle_t lt = nullptr;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS)
    return false;
  void *ws = nullptr;
  if (cudaMalloc(&ws, kLtWorkspaceBytes) != cudaSuccess) {
    cublasLtDestroy(lt);
    return false;
  }
  lt_handle_ = lt;
  lt_workspace_ = ws;
  return true;
}

namespace {
// Layout-safe cache key: M/N/K each under 2^20 for any realistic model, and
// the dtype enum folded into the top bits where it cannot overlap.
inline uint64_t LtCacheKey(int M, int N, int K, cudaDataType_t dtype) {
  return (static_cast<uint64_t>(M & 0xFFFFF) << 40) |
         (static_cast<uint64_t>(N & 0xFFFFF) << 20) |
         static_cast<uint64_t>(K & 0xFFFFF) |
         (static_cast<uint64_t>(dtype) << 61);
}

// cublasLtMatmulAlgo_t is a 64-byte opaque struct; the header caches it as
// raw bytes so cublasLt.h stays out of the header.
inline cublasLtMatmulAlgo_t LtAlgoFromBytes(const unsigned char *bytes) {
  cublasLtMatmulAlgo_t algo;
  static_assert(sizeof(algo) <= 64, "cublasLtMatmulAlgo_t grew past 64 bytes");
  std::memcpy(&algo, bytes, sizeof(algo));
  return algo;
}
inline void LtAlgoToBytes(const cublasLtMatmulAlgo_t &algo,
                          unsigned char *bytes) {
  std::memcpy(bytes, &algo, sizeof(algo));
}
} // namespace

template <typename T>
bool CublasGemm::GemmTypedLt(int M, int N, int K, const T *A, const T *B,
                             T *C) {
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;
  if (!EnsureLt()) {
    return GemmTyped<T>(M, N, K, A, B, C);
  }

  const uint64_t key = LtCacheKey(M, N, K, dtype);
  cublasLtMatmulAlgo_t algo{};
  bool have_algo = false;
  bool cached = false;
  {
    std::lock_guard<std::mutex> lock(lt_mutex_);
    auto it = lt_algo_cache_.find(key);
    if (it != lt_algo_cache_.end()) {
      if (it->second.valid)
        algo = LtAlgoFromBytes(it->second.algo_bytes);
      have_algo = it->second.valid;
      cached = true;
    }
  }

  static const cublasLtOrder_t kRowOrder = CUBLASLT_ORDER_ROW;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, dd = nullptr;
  bool layouts_ok = false;
  if (!cached) {
    // A = x [M, K] row-major with op N; B = W [N, K] row-major with op T;
    // D = C [M, N] row-major. Descriptors are cheap host objects, rebuilt
    // per call; only the chosen algo is cached per shape.
    cublasStatus_t st =
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, dtype);
    if (st == CUBLAS_STATUS_SUCCESS) {
      cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_T;
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                     sizeof(ta));
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                     sizeof(tb));
      st = cublasLtMatrixLayoutCreate(&ad, dtype, M, K, K);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(ad, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&bd, dtype, N, K, K);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&dd, dtype, M, N, N);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(dd, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      layouts_ok = (st == CUBLAS_STATUS_SUCCESS);
    }
    if (layouts_ok) {
      cublasLtMatmulPreference_t pref = nullptr;
      if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
        size_t ws = kLtWorkspaceBytes;
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        cublasLtMatmulHeuristicResult_t heur[4];
        int returned = 0;
        if (cublasLtMatmulAlgoGetHeuristic(
                reinterpret_cast<cublasLtHandle_t>(lt_handle_), op, ad, bd, dd,
                dd, pref, 4, heur, &returned) == CUBLAS_STATUS_SUCCESS &&
            returned > 0) {
          algo = heur[0].algo;
          have_algo = true;
        }
        cublasLtMatmulPreferenceDestroy(pref);
      }
    }
    std::lock_guard<std::mutex> lock(lt_mutex_);
    LtAlgoCacheEntry entry;
    entry.valid = have_algo;
    if (have_algo)
      LtAlgoToBytes(algo, entry.algo_bytes);
    lt_algo_cache_[key] = entry;
  } else {
    // Cache hit: still need descriptors for the call.
    cublasStatus_t st =
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, dtype);
    if (st == CUBLAS_STATUS_SUCCESS) {
      cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_T;
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                     sizeof(ta));
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                     sizeof(tb));
      st = cublasLtMatrixLayoutCreate(&ad, dtype, M, K, K);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(ad, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&bd, dtype, N, K, K);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&dd, dtype, M, N, N);
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(dd, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &kRowOrder, sizeof(kRowOrder));
      layouts_ok = (st == CUBLAS_STATUS_SUCCESS);
    }
  }

  bool ok = false;
  if (have_algo && layouts_ok) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    ok = cublasLtMatmul(reinterpret_cast<cublasLtHandle_t>(lt_handle_), op,
                        &alpha, A, ad, B, bd, &beta, C, dd, C, dd,
                        have_algo ? &algo : nullptr, lt_workspace_,
                        kLtWorkspaceBytes, stream_) == CUBLAS_STATUS_SUCCESS;
  }
  if (op)
    cublasLtMatmulDescDestroy(op);
  if (ad)
    cublasLtMatrixLayoutDestroy(ad);
  if (bd)
    cublasLtMatrixLayoutDestroy(bd);
  if (dd)
    cublasLtMatrixLayoutDestroy(dd);

  if (!ok) {
    return GemmTyped<T>(M, N, K, A, B, C);
  }
  return true;
}

template <typename T>
bool CublasGemm::GemmTypedAccum(int M, int N, int K, const T *A, const T *B,
                                T *C) {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  cublasStatus_t st = cublasGemmEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, dtype, K, A, dtype,
      K, &beta, C, dtype, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmTypedAccum failed: " + std::to_string(st) +
                                  " (M=" + std::to_string(M) +
                                  " N=" + std::to_string(N) +
                                  " K=" + std::to_string(K) + ")");
    return false;
  }
  return true;
}

bool CublasGemm::GemmBatched(int M, int N, int K, const half *A, const half *B,
                             half *C, long long stride_A, long long stride_B,
                             long long stride_C, int batch_count) {
  return GemmBatchedTyped<half>(M, N, K, A, B, C, stride_A, stride_B, stride_C,
                                batch_count);
}

template <typename T>
bool CublasGemm::GemmBatchedTyped(int M, int N, int K, const T *A, const T *B,
                                  T *C, long long stride_A, long long stride_B,
                                  long long stride_C, int batch_count) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  cublasStatus_t st = cublasGemmStridedBatchedEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, dtype, K, stride_B,
      A, dtype, K, stride_A, &beta, C, dtype, N, stride_C, batch_count,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmBatchedTyped failed: " + std::to_string(st) +
                                  " (batch=" + std::to_string(batch_count) +
                                  ")");
    return false;
  }
  return true;
}

// Explicit template instantiations
template bool CublasGemm::GemmTypedLt<half>(int, int, int, const half *,
                                            const half *, half *);
template bool CublasGemm::GemmTypedLt<__nv_bfloat16>(int, int, int,
                                                     const __nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     __nv_bfloat16 *);

template bool CublasGemm::GemmTyped<half>(int, int, int, const half *,
                                          const half *, half *);
template bool CublasGemm::GemmTyped<__nv_bfloat16>(int, int, int,
                                                   const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *);

template bool CublasGemm::GemmTypedAccum<half>(int, int, int, const half *,
                                               const half *, half *);
template bool CublasGemm::GemmTypedAccum<__nv_bfloat16>(int, int, int,
                                                        const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *);

template bool CublasGemm::GemmBatchedTyped<half>(int, int, int, const half *,
                                                 const half *, half *,
                                                 long long, long long,
                                                 long long, int);
template bool CublasGemm::GemmBatchedTyped<__nv_bfloat16>(
    int, int, int, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *, long long, long long, long long, int);

} // namespace inferflux
