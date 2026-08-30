#include "runtime/backends/cuda/native/native_dispatch_policy.h"

#include "runtime/backends/cuda/native/gguf_util.h"

namespace inferflux {

const NativeExecutionPolicy &
ResolveInferfluxCudaExecutionPolicy(const NativeExecutionPolicy *policy) {
  if (policy) {
    return *policy;
  }
  static thread_local NativeExecutionPolicy env_policy;
  env_policy = NativeExecutionPolicy::FromEnv();
  return env_policy;
}

std::string ProjectionQuantLabel(const QuantizedWeightInfo &weight) {
  if (!weight.data) {
    return "unknown";
  }
  return runtime::cuda::native::TensorTypeToString(
      static_cast<runtime::cuda::native::GGUF::TensorType>(weight.quant_type));
}

std::string ProjectionGroupQuantLabel(const QuantizedWeightInfo &first,
                                      const QuantizedWeightInfo &second) {
  if (!first.data || !second.data) {
    return "unknown";
  }
  if (first.quant_type == second.quant_type) {
    return ProjectionQuantLabel(first);
  }
  return "mixed";
}

bool ForceCublasRequested(const NativeExecutionPolicy &policy) {
  return policy.force_cublas;
}

bool PackedActivationsDisabled(const NativeExecutionPolicy &policy) {
  return policy.disable_prepacked_activations;
}

bool Q81ActivationsDisabled(const NativeExecutionPolicy &policy) {
  return policy.disable_q81_activations;
}

bool ProjectionHasGraphSafeKernel(const QuantizedWeightInfo &raw,
                                  const FusedDispatchGeometry &geometry,
                                  bool allow_fused_quantized_matmul,
                                  const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul ||
      !raw.data || raw.quant_type < 0 || geometry.M <= 0 || geometry.N <= 0 ||
      geometry.K <= 0) {
    return false;
  }

  if (!FusedQuantGemm::ShouldUseFusedPath(raw.quant_type, geometry)) {
    return false;
  }

  if (!Q81ActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type)) {
    return true;
  }
  if (!PackedActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsPackedActivations(raw.quant_type)) {
    return true;
  }

  // Both activation tiers are policy-disabled: no fused kernel can run,
  // so the projection is not graph-safe (it would hit the cuBLAS capture
  // abort). The old unconditional `return true` started doomed captures.
  return !Q81ActivationsDisabled(policy) || !PackedActivationsDisabled(policy);
}

FusedQuantGemm::DownProjOperator SelectInferfluxCudaDownProjOperator(
    const QuantizedWeightInfo &raw, const MmqWeightInfo &mmq_weight,
    InferfluxCudaDispatchPhase phase, const FusedDispatchGeometry &geometry,
    bool allow_fused_quantized_matmul, const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul) {
    return FusedQuantGemm::DownProjOperator::kFallback;
  }

  const FusedDispatchGeometry single_geometry{
      geometry.M, geometry.N, geometry.K, 1, true, false};
  const bool q81_ready =
      !Q81ActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) &&
      FusedQuantGemm::ShouldUseFusedPath(raw.quant_type, single_geometry);
  const bool packed_ready =
      !PackedActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsPackedActivations(raw.quant_type) &&
      FusedQuantGemm::ShouldUseFusedPath(raw.quant_type, single_geometry);
  // MMA tensor-core path (S7): ready on its own policy for Q6_K — it reads
  // the raw row-major layout, so it needs neither the transformed MMQ
  // layout nor the dp4a MMQ threshold. When it wins, the executor's
  // try_mmq callback runs the MMA kernel first (transformer_forward.cu).
  const bool mma_ready =
      policy.enable_mmq_mma && raw.data != nullptr &&
      raw.quant_type ==
          static_cast<int>(runtime::cuda::native::GGUF::TensorType::Q6_K) &&
      geometry.M >= policy.mmq_mma_min_batch &&
      geometry.M <= policy.mmq_mma_max_batch && geometry.K % 256 == 0;
  const bool mmq_ready =
      mma_ready ||
      (mmq_weight.data != nullptr &&
       FusedQuantGemm::IsDownProjMmqEnabled(&policy) &&
       FusedQuantGemm::SupportsDownProjMmq(raw.quant_type) &&
       geometry.M >= FusedQuantGemm::GetDownProjMmqThreshold(
                         raw.quant_type, geometry.M, geometry.N, geometry.K));
  const auto profile = BuildInferfluxCudaDownProjDispatchProfile(
      phase, raw.quant_type, geometry, q81_ready, packed_ready, mmq_ready);
  return SelectInferfluxCudaDownProjDispatchDecision(profile, policy).op;
}

FusedQuantGemm::FfnProjOperator SelectInferfluxCudaFfnProjOperator(
    const QuantizedWeightInfo &gate_raw, const QuantizedWeightInfo &up_raw,
    InferfluxCudaDispatchPhase phase, const FusedDispatchGeometry &geometry,
    bool allow_fused_quantized_matmul, const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul) {
    return FusedQuantGemm::FfnProjOperator::kFallback;
  }
  const FusedDispatchGeometry single_geometry{geometry.M,
                                              geometry.N,
                                              geometry.K,
                                              1,
                                              geometry.packed_activation,
                                              geometry.includes_rmsnorm};
  const bool q81_ready =
      !Q81ActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsQ8_1Activations(gate_raw.quant_type) &&
      FusedQuantGemm::SupportsQ8_1Activations(up_raw.quant_type) &&
      FusedQuantGemm::ShouldUseFusedPath(gate_raw.quant_type,
                                         single_geometry) &&
      FusedQuantGemm::ShouldUseFusedPath(up_raw.quant_type, single_geometry);
  const bool packed_ready =
      !PackedActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsPackedActivations(gate_raw.quant_type) &&
      FusedQuantGemm::SupportsPackedActivations(up_raw.quant_type) &&
      FusedQuantGemm::ShouldUseFusedPath(gate_raw.quant_type,
                                         single_geometry) &&
      FusedQuantGemm::ShouldUseFusedPath(up_raw.quant_type, single_geometry);
  const auto profile = BuildInferfluxCudaFfnDispatchProfile(
      phase, gate_raw.quant_type, up_raw.quant_type, geometry, q81_ready,
      packed_ready);
  return SelectInferfluxCudaFfnDispatchDecision(profile, policy).op;
}

bool DecodeGraphCaptureSafe(const WeightMap *weights, int num_layers, int M,
                            int hidden_size, int num_heads, int num_kv_heads,
                            int head_dim, int intermediate_size, int vocab_size,
                            bool allow_fused_quantized_matmul,
                            const NativeExecutionPolicy &policy) {
  if (!weights || !weights->HasQuantizedWeights()) {
    return false;
  }

  for (int layer = 0; layer < num_layers; ++layer) {
    if (!ProjectionHasGraphSafeKernel(
            weights->LayerQProjRaw(layer),
            FusedDispatchGeometry{M, num_heads * head_dim, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerKProjRaw(layer),
            FusedDispatchGeometry{M, num_kv_heads * head_dim, hidden_size, 1,
                                  true, true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerVProjRaw(layer),
            FusedDispatchGeometry{M, num_kv_heads * head_dim, hidden_size, 1,
                                  true, true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerOProjRaw(layer),
            FusedDispatchGeometry{M, hidden_size, num_heads * head_dim, 1, true,
                                  false},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerGateProjRaw(layer),
            FusedDispatchGeometry{M, intermediate_size, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerUpProjRaw(layer),
            FusedDispatchGeometry{M, intermediate_size, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(weights->LayerDownProjRaw(layer),
                                      FusedDispatchGeometry{M, hidden_size,
                                                            intermediate_size,
                                                            1, true, false},
                                      allow_fused_quantized_matmul, policy)) {
      return false;
    }
  }

  return ProjectionHasGraphSafeKernel(
      weights->LmHeadRaw(),
      FusedDispatchGeometry{M, vocab_size, hidden_size, 1, true, true},
      allow_fused_quantized_matmul, policy);
}

} // namespace inferflux
