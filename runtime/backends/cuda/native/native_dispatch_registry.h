#pragma once

#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"

#include <string>
#include <string_view>

namespace inferflux {

enum class InferfluxCudaDispatchPhase {
  kUnknown = 0,
  kDecode,
  kPrefill,
};

enum class InferfluxCudaDispatchBucket {
  k1 = 0,
  k2,
  k3_4,
  k5_8,
  k9_16,
  k17Plus,
};

const char *InferfluxCudaDispatchPhaseName(InferfluxCudaDispatchPhase phase);
InferfluxCudaDispatchPhase
ParseInferfluxCudaDispatchPhase(std::string_view phase);

InferfluxCudaDispatchBucket BucketBatchRows(int rows);
InferfluxCudaDispatchBucket BucketExtent(int extent);
const char *InferfluxCudaDispatchBucketName(InferfluxCudaDispatchBucket bucket);

struct InferfluxCudaFfnDispatchProfile {
  InferfluxCudaDispatchPhase phase{InferfluxCudaDispatchPhase::kUnknown};
  int quant_type0{0};
  int quant_type1{0};
  FusedDispatchGeometry geometry{};
  bool same_quant{false};
  bool q81_ready{false};
  bool packed_ready{false};
  InferfluxCudaDispatchBucket m_bucket{InferfluxCudaDispatchBucket::k1};
  InferfluxCudaDispatchBucket n_bucket{InferfluxCudaDispatchBucket::k1};
  InferfluxCudaDispatchBucket k_bucket{InferfluxCudaDispatchBucket::k1};
};

struct InferfluxCudaFfnDispatchDecision {
  FusedQuantGemm::FfnProjOperator op{
      FusedQuantGemm::FfnProjOperator::kFallback};
  const char *reason{nullptr};
};

struct InferfluxCudaDownProjDispatchProfile {
  InferfluxCudaDispatchPhase phase{InferfluxCudaDispatchPhase::kUnknown};
  int quant_type{0};
  FusedDispatchGeometry geometry{};
  bool q81_ready{false};
  bool packed_ready{false};
  bool mmq_ready{false};
  InferfluxCudaDispatchBucket m_bucket{InferfluxCudaDispatchBucket::k1};
  InferfluxCudaDispatchBucket n_bucket{InferfluxCudaDispatchBucket::k1};
  InferfluxCudaDispatchBucket k_bucket{InferfluxCudaDispatchBucket::k1};
};

struct InferfluxCudaDownProjDispatchDecision {
  FusedQuantGemm::DownProjOperator op{
      FusedQuantGemm::DownProjOperator::kFallback};
  const char *reason{nullptr};
};

InferfluxCudaFfnDispatchProfile BuildInferfluxCudaFfnDispatchProfile(
    InferfluxCudaDispatchPhase phase, int quant_type0, int quant_type1,
    const FusedDispatchGeometry &geometry, bool q81_ready, bool packed_ready);

InferfluxCudaFfnDispatchDecision SelectInferfluxCudaFfnDispatchDecision(
    const InferfluxCudaFfnDispatchProfile &profile,
    const NativeExecutionPolicy &policy);

std::string DescribeInferfluxCudaFfnDispatchDecision(
    const InferfluxCudaFfnDispatchProfile &profile, std::string_view reason);

InferfluxCudaDownProjDispatchProfile BuildInferfluxCudaDownProjDispatchProfile(
    InferfluxCudaDispatchPhase phase, int quant_type,
    const FusedDispatchGeometry &geometry, bool q81_ready, bool packed_ready,
    bool mmq_ready);

InferfluxCudaDownProjDispatchDecision
SelectInferfluxCudaDownProjDispatchDecision(
    const InferfluxCudaDownProjDispatchProfile &profile,
    const NativeExecutionPolicy &policy);

std::string DescribeInferfluxCudaDownProjDispatchDecision(
    const InferfluxCudaDownProjDispatchProfile &profile,
    std::string_view reason);

} // namespace inferflux
