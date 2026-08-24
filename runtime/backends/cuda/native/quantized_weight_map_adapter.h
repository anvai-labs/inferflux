#pragma once

#include "runtime/backends/cuda/native/quantized_weight_map.h"
#include "runtime/backends/cuda/native/weight_map.h"

namespace inferflux {

/**
 * Adapter that presents a QuantizedWeightMap (GGUF) as a WeightMapTyped<half>,
 * allowing LlamaForwardTyped<half> to consume GGUF models directly.
 *
 * Both weight sources return identical const half* pointers via identical
 * method signatures — this adapter simply delegates each call.
 */
class QuantizedWeightMapAdapter : public WeightMapTyped<half> {
public:
  explicit QuantizedWeightMapAdapter(QuantizedWeightMap *qwm) : qwm_(qwm) {}

  const half *LayerQProj(int layer) const override {
    return qwm_->LayerQProj(layer);
  }
  const half *LayerKProj(int layer) const override {
    return qwm_->LayerKProj(layer);
  }
  const half *LayerVProj(int layer) const override {
    return qwm_->LayerVProj(layer);
  }
  const half *LayerOProj(int layer) const override {
    return qwm_->LayerOProj(layer);
  }
  const half *LayerInputNorm(int layer) const override {
    return qwm_->LayerInputNorm(layer);
  }
  const half *LayerPostAttnNorm(int layer) const override {
    return qwm_->LayerPostAttnNorm(layer);
  }
  const half *LayerGateProj(int layer) const override {
    return qwm_->LayerGateProj(layer);
  }
  const half *LayerUpProj(int layer) const override {
    return qwm_->LayerUpProj(layer);
  }
  const half *LayerDownProj(int layer) const override {
    return qwm_->LayerDownProj(layer);
  }

  const half *LayerQProjBias(int layer) const override {
    return qwm_->LayerQProjBias(layer);
  }
  const half *LayerKProjBias(int layer) const override {
    return qwm_->LayerKProjBias(layer);
  }
  const half *LayerVProjBias(int layer) const override {
    return qwm_->LayerVProjBias(layer);
  }

  const half *EmbedTokens() const override { return qwm_->EmbedTokens(); }
  const half *FinalNorm() const override { return qwm_->FinalNorm(); }
  const half *LmHead() const override { return qwm_->LmHead(); }

  int NumLayers() const override { return qwm_->NumLayers(); }

  // --- Raw quantized weight accessors (fused dequant-GEMV) ---
  QuantizedWeightInfo LayerQProjRaw(int layer) const override {
    return qwm_->GetRawLayerQProj(layer);
  }
  QuantizedWeightInfo LayerKProjRaw(int layer) const override {
    return qwm_->GetRawLayerKProj(layer);
  }
  QuantizedWeightInfo LayerVProjRaw(int layer) const override {
    return qwm_->GetRawLayerVProj(layer);
  }
  QuantizedWeightInfo LayerOProjRaw(int layer) const override {
    return qwm_->GetRawLayerOProj(layer);
  }
  QuantizedWeightInfo LayerGateProjRaw(int layer) const override {
    return qwm_->GetRawLayerGateProj(layer);
  }
  QuantizedWeightInfo LayerUpProjRaw(int layer) const override {
    return qwm_->GetRawLayerUpProj(layer);
  }
  QuantizedWeightInfo LayerDownProjRaw(int layer) const override {
    return qwm_->GetRawLayerDownProj(layer);
  }
  MmqWeightInfo LayerDownProjMmq(int layer) const override {
    return qwm_->GetMmqLayerDownProj(layer);
  }
  QuantizedWeightInfo LmHeadRaw() const override {
    return qwm_->GetRawLmHead();
  }
  bool HasQuantizedWeights() const override { return qwm_->IsQuantized(); }
  bool AllowFusedQuantizedMatmul() const override {
    return qwm_->AllowFusedQuantizedMatmul();
  }

private:
  QuantizedWeightMap *qwm_; // Non-owning
};

} // namespace inferflux
