#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/inferflux_cuda_executor.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#endif

namespace inferflux {

// ============================================================================
// DtypeTraits compile-time property tests
// ============================================================================

TEST_CASE("DtypeTraits: FP16 cublas_type is CUDA_R_16F", "[dtype_traits]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  REQUIRE(DtypeTraits<half>::cublas_type == CUDA_R_16F);
#else
  REQUIRE(true); // Placeholder
#endif
}

TEST_CASE("DtypeTraits: BF16 cublas_type is CUDA_R_16BF", "[dtype_traits]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  REQUIRE(DtypeTraits<__nv_bfloat16>::cublas_type == CUDA_R_16BF);
#else
  REQUIRE(true);
#endif
}

TEST_CASE("DtypeTraits: FP16 name is fp16", "[dtype_traits]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  REQUIRE(std::string(DtypeTraits<half>::name) == "fp16");
#else
  REQUIRE(true);
#endif
}

TEST_CASE("DtypeTraits: BF16 name is bf16", "[dtype_traits]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  REQUIRE(std::string(DtypeTraits<__nv_bfloat16>::name) == "bf16");
#else
  REQUIRE(true);
#endif
}

// ============================================================================
// InferenceDtype enum tests
// ============================================================================

TEST_CASE("InferfluxCudaExecutor: InferenceDtype defaults to FP16",
          "[dtype_traits]") {
  InferfluxCudaExecutor executor;
  REQUIRE(executor.GetInferenceDtype() ==
          InferfluxCudaExecutor::InferenceDtype::kFP16);
}

} // namespace inferflux
