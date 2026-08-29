#include "runtime/backends/cuda/native/native_dispatch_registry.h"

#include "runtime/backends/cuda/native/dispatch_operator_health.h"
#include "runtime/backends/cuda/native/gguf_util.h"

#include <array>

namespace inferflux {

namespace {

struct InferfluxCudaFfnDispatchRule {
  FusedQuantGemm::FfnProjOperator op{
      FusedQuantGemm::FfnProjOperator::kFallback};
  bool requires_q81{false};
  bool requires_packed{false};
  bool (*match)(const InferfluxCudaFfnDispatchProfile &,
                const NativeExecutionPolicy &){nullptr};
  const char *reason{nullptr};
};

struct InferfluxCudaDownProjDispatchRule {
  FusedQuantGemm::DownProjOperator op{
      FusedQuantGemm::DownProjOperator::kFallback};
  bool requires_q81{false};
  bool requires_packed{false};
  bool requires_mmq{false};
  bool (*match)(const InferfluxCudaDownProjDispatchProfile &,
                const NativeExecutionPolicy &){nullptr};
  const char *reason{nullptr};
};

bool MatchDecodeGroupedHotQ4K(const InferfluxCudaFfnDispatchProfile &profile,
                              const NativeExecutionPolicy &) {
  return profile.phase == InferfluxCudaDispatchPhase::kDecode &&
         profile.same_quant &&
         FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
             profile.quant_type0, profile.geometry);
}

bool MatchDecodeGroupedRowPairW4(const InferfluxCudaFfnDispatchProfile &profile,
                                 const NativeExecutionPolicy &policy) {
  return profile.phase == InferfluxCudaDispatchPhase::kDecode &&
         policy.enable_experimental_q81_grouped_rowpair_w4 &&
         profile.same_quant &&
         profile.quant_type0 ==
             static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K) &&
         profile.geometry.grouped_outputs == 2 &&
         profile.m_bucket == InferfluxCudaDispatchBucket::k2 &&
         profile.geometry.N >= 8192 && profile.geometry.K == 2048;
}

bool MatchDecodeGroupedRowQuadM4(const InferfluxCudaFfnDispatchProfile &profile,
                                 const NativeExecutionPolicy &policy) {
  return profile.phase == InferfluxCudaDispatchPhase::kDecode &&
         policy.enable_experimental_q81_grouped_rowquad_m4 &&
         profile.same_quant &&
         profile.quant_type0 ==
             static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K) &&
         profile.geometry.grouped_outputs == 2 && profile.geometry.M == 4 &&
         profile.geometry.N >= 8192 && profile.geometry.K == 2048;
}

bool MatchDecodeGroupedMmq3(const InferfluxCudaFfnDispatchProfile &profile,
                            const NativeExecutionPolicy &policy) {
  return profile.phase == InferfluxCudaDispatchPhase::kDecode &&
         policy.enable_experimental_q81_grouped_mmq3 && profile.same_quant &&
         profile.quant_type0 ==
             static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q4_K) &&
         profile.geometry.grouped_outputs == 2 && profile.geometry.M >= 2;
}

bool MatchDecodeGenericQ81(const InferfluxCudaFfnDispatchProfile &profile,
                           const NativeExecutionPolicy &) {
  return profile.phase == InferfluxCudaDispatchPhase::kDecode &&
         profile.geometry.grouped_outputs == 2;
}

bool MatchPrefillGenericQ81(const InferfluxCudaFfnDispatchProfile &profile,
                            const NativeExecutionPolicy &) {
  return profile.phase == InferfluxCudaDispatchPhase::kPrefill &&
         profile.geometry.grouped_outputs == 2;
}

bool MatchUnknownGenericQ81(const InferfluxCudaFfnDispatchProfile &profile,
                            const NativeExecutionPolicy &) {
  return profile.phase == InferfluxCudaDispatchPhase::kUnknown &&
         profile.geometry.grouped_outputs == 2;
}

bool MatchPackedGroup(const InferfluxCudaFfnDispatchProfile &profile,
                      const NativeExecutionPolicy &) {
  return profile.geometry.grouped_outputs == 2;
}

const std::array<InferfluxCudaFfnDispatchRule, 8> &
GetInferfluxCudaFfnDispatchRules() {
  static const std::array<InferfluxCudaFfnDispatchRule, 8> rules = {{
      {FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K,
       /*requires_q81=*/true, /*requires_packed=*/false,
       MatchDecodeGroupedHotQ4K, "decode_q81_hot_q4k"},
      {FusedQuantGemm::FfnProjOperator::kQ81GroupRowQuadM4,
       /*requires_q81=*/true, /*requires_packed=*/false,
       MatchDecodeGroupedRowQuadM4, "decode_q81_rowquad_m4"},
      {FusedQuantGemm::FfnProjOperator::kQ81GroupMmq3,
       /*requires_q81=*/true, /*requires_packed=*/false, MatchDecodeGroupedMmq3,
       "decode_q81_mmq3"},
      {FusedQuantGemm::FfnProjOperator::kQ81GroupRowPairW4,
       /*requires_q81=*/true, /*requires_packed=*/false,
       MatchDecodeGroupedRowPairW4, "decode_q81_rowpair_w4"},
      {FusedQuantGemm::FfnProjOperator::kQ81Group,
       /*requires_q81=*/true, /*requires_packed=*/false, MatchDecodeGenericQ81,
       "decode_q81_generic"},
      {FusedQuantGemm::FfnProjOperator::kQ81Group,
       /*requires_q81=*/true, /*requires_packed=*/false, MatchPrefillGenericQ81,
       "prefill_q81_generic"},
      {FusedQuantGemm::FfnProjOperator::kQ81Group,
       /*requires_q81=*/true, /*requires_packed=*/false, MatchUnknownGenericQ81,
       "q81_generic"},
      {FusedQuantGemm::FfnProjOperator::kPackedGroup,
       /*requires_q81=*/false, /*requires_packed=*/true, MatchPackedGroup,
       "packed_group"},
  }};
  return rules;
}

bool MatchDownProjMmq(const InferfluxCudaDownProjDispatchProfile &profile,
                      const NativeExecutionPolicy &) {
  return profile.mmq_ready;
}

bool MatchDownProjHotFixed(const InferfluxCudaDownProjDispatchProfile &profile,
                           const NativeExecutionPolicy &) {
  return FusedQuantGemm::ShouldUseSpecializedQ8_1DownProjHotPath(
      profile.quant_type, profile.geometry);
}

bool MatchDownProjRowPairHotFixed(
    const InferfluxCudaDownProjDispatchProfile &profile,
    const NativeExecutionPolicy &) {
  return FusedQuantGemm::ShouldUseSpecializedQ8_1DownProjRowPairHotPath(
      profile.quant_type, profile.geometry);
}

bool MatchDownProjRowQuad(const InferfluxCudaDownProjDispatchProfile &profile,
                          const NativeExecutionPolicy &) {
  const auto qtype =
      static_cast<runtime::cuda::native::GGUF::TensorType>(profile.quant_type);
  return (qtype == runtime::cuda::native::GGUF::TensorType::Q4_K ||
          qtype == runtime::cuda::native::GGUF::TensorType::Q6_K) &&
         profile.geometry.M >= 4;
}

bool MatchDownProjRowPair(const InferfluxCudaDownProjDispatchProfile &profile,
                          const NativeExecutionPolicy &) {
  const auto qtype =
      static_cast<runtime::cuda::native::GGUF::TensorType>(profile.quant_type);
  return (qtype == runtime::cuda::native::GGUF::TensorType::Q4_K ||
          qtype == runtime::cuda::native::GGUF::TensorType::Q6_K) &&
         profile.geometry.M > 1;
}

bool MatchDownProjGenericQ81(const InferfluxCudaDownProjDispatchProfile &,
                             const NativeExecutionPolicy &) {
  return true;
}

bool MatchDownProjPacked(const InferfluxCudaDownProjDispatchProfile &,
                         const NativeExecutionPolicy &) {
  return true;
}

const std::array<InferfluxCudaDownProjDispatchRule, 7> &
GetInferfluxCudaDownProjDispatchRules() {
  static const std::array<InferfluxCudaDownProjDispatchRule, 7> rules = {{
      {FusedQuantGemm::DownProjOperator::kMmq,
       /*requires_q81=*/false, /*requires_packed=*/false,
       /*requires_mmq=*/true, MatchDownProjMmq, "mmq"},
      {FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed,
       /*requires_q81=*/true, /*requires_packed=*/false,
       /*requires_mmq=*/false, MatchDownProjHotFixed, "q81_hot_fixed"},
      {FusedQuantGemm::DownProjOperator::kQ81GemvRowPairHotFixed,
       /*requires_q81=*/true, /*requires_packed=*/false,
       /*requires_mmq=*/false, MatchDownProjRowPairHotFixed,
       "q81_rowpair_hot_fixed"},
      {FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad,
       /*requires_q81=*/true, /*requires_packed=*/false,
       /*requires_mmq=*/false, MatchDownProjRowQuad, "q81_rowquad"},
      {FusedQuantGemm::DownProjOperator::kQ81GemvRowPair,
       /*requires_q81=*/true, /*requires_packed=*/false,
       /*requires_mmq=*/false, MatchDownProjRowPair, "q81_rowpair"},
      {FusedQuantGemm::DownProjOperator::kQ81Gemv,
       /*requires_q81=*/true, /*requires_packed=*/false,
       /*requires_mmq=*/false, MatchDownProjGenericQ81, "q81_generic"},
      {FusedQuantGemm::DownProjOperator::kPackedGemv,
       /*requires_q81=*/false, /*requires_packed=*/true,
       /*requires_mmq=*/false, MatchDownProjPacked, "packed_gemv"},
  }};
  return rules;
}

} // namespace

const char *InferfluxCudaDispatchPhaseName(InferfluxCudaDispatchPhase phase) {
  switch (phase) {
  case InferfluxCudaDispatchPhase::kDecode:
    return "decode";
  case InferfluxCudaDispatchPhase::kPrefill:
    return "prefill";
  case InferfluxCudaDispatchPhase::kUnknown:
  default:
    return "unknown";
  }
}

InferfluxCudaDispatchPhase
ParseInferfluxCudaDispatchPhase(std::string_view phase) {
  if (phase == "decode") {
    return InferfluxCudaDispatchPhase::kDecode;
  }
  if (phase == "prefill") {
    return InferfluxCudaDispatchPhase::kPrefill;
  }
  return InferfluxCudaDispatchPhase::kUnknown;
}

InferfluxCudaDispatchBucket BucketBatchRows(int rows) {
  if (rows <= 1) {
    return InferfluxCudaDispatchBucket::k1;
  }
  if (rows == 2) {
    return InferfluxCudaDispatchBucket::k2;
  }
  if (rows <= 4) {
    return InferfluxCudaDispatchBucket::k3_4;
  }
  if (rows <= 8) {
    return InferfluxCudaDispatchBucket::k5_8;
  }
  if (rows <= 16) {
    return InferfluxCudaDispatchBucket::k9_16;
  }
  return InferfluxCudaDispatchBucket::k17Plus;
}

InferfluxCudaDispatchBucket BucketExtent(int extent) {
  return BucketBatchRows(extent);
}

const char *
InferfluxCudaDispatchBucketName(InferfluxCudaDispatchBucket bucket) {
  switch (bucket) {
  case InferfluxCudaDispatchBucket::k1:
    return "1";
  case InferfluxCudaDispatchBucket::k2:
    return "2";
  case InferfluxCudaDispatchBucket::k3_4:
    return "3_4";
  case InferfluxCudaDispatchBucket::k5_8:
    return "5_8";
  case InferfluxCudaDispatchBucket::k9_16:
    return "9_16";
  case InferfluxCudaDispatchBucket::k17Plus:
  default:
    return "17_plus";
  }
}

InferfluxCudaFfnDispatchProfile BuildInferfluxCudaFfnDispatchProfile(
    InferfluxCudaDispatchPhase phase, int quant_type0, int quant_type1,
    const FusedDispatchGeometry &geometry, bool q81_ready, bool packed_ready) {
  InferfluxCudaFfnDispatchProfile profile;
  profile.phase = phase;
  profile.quant_type0 = quant_type0;
  profile.quant_type1 = quant_type1;
  profile.geometry = geometry;
  profile.same_quant = quant_type0 == quant_type1;
  profile.q81_ready = q81_ready;
  profile.packed_ready = packed_ready;
  profile.m_bucket = BucketBatchRows(geometry.M);
  profile.n_bucket = BucketExtent(geometry.N);
  profile.k_bucket = BucketExtent(geometry.K);
  return profile;
}

InferfluxCudaFfnDispatchDecision SelectInferfluxCudaFfnDispatchDecision(
    const InferfluxCudaFfnDispatchProfile &profile,
    const NativeExecutionPolicy &policy) {
  for (const auto &rule : GetInferfluxCudaFfnDispatchRules()) {
    if (InferfluxCudaOperatorHealth::Instance().IsUnhealthy(rule.op)) {
      continue; // down-ranked by the probe / divergence telemetry
    }
    if (rule.requires_q81 && !profile.q81_ready) {
      continue;
    }
    if (rule.requires_packed && !profile.packed_ready) {
      continue;
    }
    if (!rule.match || !rule.match(profile, policy)) {
      continue;
    }
    return {rule.op, rule.reason};
  }
  return {};
}

std::string DescribeInferfluxCudaFfnDispatchDecision(
    const InferfluxCudaFfnDispatchProfile &profile, std::string_view reason) {
  return "phase=" + std::string(InferfluxCudaDispatchPhaseName(profile.phase)) +
         ", quant0=" +
         runtime::cuda::native::TensorTypeToString(
             static_cast<runtime::cuda::native::GGUF::TensorType>(
                 profile.quant_type0)) +
         ", quant1=" +
         runtime::cuda::native::TensorTypeToString(
             static_cast<runtime::cuda::native::GGUF::TensorType>(
                 profile.quant_type1)) +
         ", q81_ready=" + std::string(profile.q81_ready ? "true" : "false") +
         ", packed_ready=" +
         std::string(profile.packed_ready ? "true" : "false") +
         ", m_bucket=" + InferfluxCudaDispatchBucketName(profile.m_bucket) +
         ", n_bucket=" + InferfluxCudaDispatchBucketName(profile.n_bucket) +
         ", k_bucket=" + InferfluxCudaDispatchBucketName(profile.k_bucket) +
         (reason.empty() ? std::string() : ", rule=" + std::string(reason));
}

InferfluxCudaDownProjDispatchProfile BuildInferfluxCudaDownProjDispatchProfile(
    InferfluxCudaDispatchPhase phase, int quant_type,
    const FusedDispatchGeometry &geometry, bool q81_ready, bool packed_ready,
    bool mmq_ready) {
  InferfluxCudaDownProjDispatchProfile profile;
  profile.phase = phase;
  profile.quant_type = quant_type;
  profile.geometry = geometry;
  profile.q81_ready = q81_ready;
  profile.packed_ready = packed_ready;
  profile.mmq_ready = mmq_ready;
  profile.m_bucket = BucketBatchRows(geometry.M);
  profile.n_bucket = BucketExtent(geometry.N);
  profile.k_bucket = BucketExtent(geometry.K);
  return profile;
}

InferfluxCudaDownProjDispatchDecision
SelectInferfluxCudaDownProjDispatchDecision(
    const InferfluxCudaDownProjDispatchProfile &profile,
    const NativeExecutionPolicy &policy) {
  for (const auto &rule : GetInferfluxCudaDownProjDispatchRules()) {
    if (InferfluxCudaOperatorHealth::Instance().IsUnhealthy(rule.op)) {
      continue; // down-ranked by the probe / divergence telemetry
    }
    if (rule.requires_q81 && !profile.q81_ready) {
      continue;
    }
    if (rule.requires_packed && !profile.packed_ready) {
      continue;
    }
    if (rule.requires_mmq && !profile.mmq_ready) {
      continue;
    }
    if (!rule.match || !rule.match(profile, policy)) {
      continue;
    }
    return {rule.op, rule.reason};
  }
  return {};
}

std::string DescribeInferfluxCudaDownProjDispatchDecision(
    const InferfluxCudaDownProjDispatchProfile &profile,
    std::string_view reason) {
  return "phase=" + std::string(InferfluxCudaDispatchPhaseName(profile.phase)) +
         ", quant=" +
         runtime::cuda::native::TensorTypeToString(
             static_cast<runtime::cuda::native::GGUF::TensorType>(
                 profile.quant_type)) +
         ", q81_ready=" + std::string(profile.q81_ready ? "true" : "false") +
         ", packed_ready=" +
         std::string(profile.packed_ready ? "true" : "false") +
         ", mmq_ready=" + std::string(profile.mmq_ready ? "true" : "false") +
         ", m_bucket=" + InferfluxCudaDispatchBucketName(profile.m_bucket) +
         ", n_bucket=" + InferfluxCudaDispatchBucketName(profile.n_bucket) +
         ", k_bucket=" + InferfluxCudaDispatchBucketName(profile.k_bucket) +
         (reason.empty() ? std::string() : ", rule=" + std::string(reason));
}

} // namespace inferflux
