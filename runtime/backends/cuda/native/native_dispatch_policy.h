#pragma once

#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/native_dispatch_registry.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "runtime/backends/cuda/native/weight_map.h"

#include <string>

namespace inferflux {

const NativeExecutionPolicy &
ResolveInferfluxCudaExecutionPolicy(const NativeExecutionPolicy *policy);

std::string ProjectionQuantLabel(const QuantizedWeightInfo &weight);

std::string ProjectionGroupQuantLabel(const QuantizedWeightInfo &first,
                                      const QuantizedWeightInfo &second);

bool ForceCublasRequested(const NativeExecutionPolicy &policy);

bool PackedActivationsDisabled(const NativeExecutionPolicy &policy);

bool Q81ActivationsDisabled(const NativeExecutionPolicy &policy);

bool ProjectionHasGraphSafeKernel(const QuantizedWeightInfo &raw,
                                  const FusedDispatchGeometry &geometry,
                                  bool allow_fused_quantized_matmul,
                                  const NativeExecutionPolicy &policy);

FusedQuantGemm::DownProjOperator SelectInferfluxCudaDownProjOperator(
    const QuantizedWeightInfo &raw, const MmqWeightInfo &mmq_weight,
    InferfluxCudaDispatchPhase phase, const FusedDispatchGeometry &geometry,
    bool allow_fused_quantized_matmul, const NativeExecutionPolicy &policy);

FusedQuantGemm::FfnProjOperator SelectInferfluxCudaFfnProjOperator(
    const QuantizedWeightInfo &gate_raw, const QuantizedWeightInfo &up_raw,
    InferfluxCudaDispatchPhase phase, const FusedDispatchGeometry &geometry,
    bool allow_fused_quantized_matmul, const NativeExecutionPolicy &policy);

bool DecodeGraphCaptureSafe(const WeightMap *weights, int num_layers, int M,
                            int hidden_size, int num_heads, int num_kv_heads,
                            int head_dim, int intermediate_size, int vocab_size,
                            bool allow_fused_quantized_matmul,
                            const NativeExecutionPolicy &policy);

} // namespace inferflux
