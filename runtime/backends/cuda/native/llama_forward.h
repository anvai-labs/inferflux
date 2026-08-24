#pragma once

#include "runtime/backends/cuda/native/model_forward.h"
#include <algorithm>
#include <array>

#include <cstdint>

namespace inferflux {

inline int PackedActivationWidth(int hidden_size, int intermediate_size) {
  return std::max(hidden_size, intermediate_size);
}

inline bool ShouldUseDecodeGraph(bool graph_enabled, bool graph_disabled,
                                 bool phase_timing_enabled,
                                 bool capture_safe = true) {
  return graph_enabled && !graph_disabled && !phase_timing_enabled &&
         capture_safe;
}

template <size_t GroupSize> struct SharedActivationGroupingChoice {
  int grouped_count{0};
  std::array<int, GroupSize> indices{};
};

template <size_t GroupSize>
inline SharedActivationGroupingChoice<GroupSize> SelectSharedActivationGrouping(
    const std::array<int, GroupSize> &quant_types,
    const std::array<int, GroupSize> &output_cols,
    const std::array<bool, GroupSize> &pair_ready,
    const std::array<bool, GroupSize> &triple_ready) {
  SharedActivationGroupingChoice<GroupSize> choice;
  choice.indices.fill(-1);

  if constexpr (GroupSize == 2) {
    if (quant_types[0] == quant_types[1] && pair_ready[0] && pair_ready[1]) {
      choice.grouped_count = 2;
      choice.indices[0] = 0;
      choice.indices[1] = 1;
    }
    return choice;
  }

  if constexpr (GroupSize == 3) {
    if (quant_types[0] == quant_types[1] && quant_types[1] == quant_types[2] &&
        triple_ready[0] && triple_ready[1] && triple_ready[2]) {
      choice.grouped_count = 3;
      choice.indices[0] = 0;
      choice.indices[1] = 1;
      choice.indices[2] = 2;
      return choice;
    }

    int best_i = -1;
    int best_j = -1;
    int best_cols = -1;
    for (size_t i = 0; i < GroupSize; ++i) {
      for (size_t j = i + 1; j < GroupSize; ++j) {
        if (quant_types[i] != quant_types[j] || !pair_ready[i] ||
            !pair_ready[j]) {
          continue;
        }
        const int cols = output_cols[i] + output_cols[j];
        if (cols > best_cols) {
          best_cols = cols;
          best_i = static_cast<int>(i);
          best_j = static_cast<int>(j);
        }
      }
    }
    if (best_i >= 0) {
      choice.grouped_count = 2;
      choice.indices[0] = best_i;
      choice.indices[1] = best_j;
    }
  }

  return choice;
}

/**
 * LlamaForwardTyped<T>: Llama/Qwen decoder-only transformer forward pass.
 *
 * Implements the standard pipeline:
 *   embed -> (rmsnorm -> attn -> residual -> rmsnorm -> ffn -> residual) x L
 *   -> rmsnorm -> lm_head -> logits
 */
template <typename T> class LlamaForwardTyped : public ModelForward {
public:
  LlamaForwardTyped() = default;
  ~LlamaForwardTyped() override;

  bool Initialize(const SafetensorsLoader::ModelConfig &config,
                  const WeightMap &weights, IKvCacheGpu *kv_cache,
                  CublasGemm *gemm, cudaStream_t stream) override;

  bool Forward(const std::vector<int> &token_ids, int n_past, int sequence_id,
               float *d_logits) override;

  bool BatchForward(const std::vector<int> &token_ids,
                    const std::vector<int> &n_past,
                    const std::vector<int> &sequence_ids, float *d_logits,
                    int batch_size) override;

  BatchMetaDevice BatchMetaDevicePointers() const override {
    return {d_batch_token_ids_, d_batch_n_past_, d_batch_seq_ids_,
            d_batch_kv_lens_};
  }

  bool DecodeGraphReady(int batch_size) const override {
    return decode_graph_exec_ != nullptr && graph_batch_size_ == batch_size &&
           graph_enabled_;
  }

  bool BatchForwardDevice(int batch_size, float *d_logits) override;

  void WarmWeightCaches() override;
  std::string ProbeDispatchPaths() override;
  void SetStream(cudaStream_t stream) override;
  void SetExecutionPolicy(const NativeExecutionPolicy &policy) override;

  void FreeScratchBuffers() override;
  std::size_t DeviceWorkspaceBytes() const override {
    return device_workspace_bytes_;
  }
  std::size_t HostWorkspaceBytes() const override {
    return host_workspace_bytes_;
  }

  std::string ModelType() const override { return "llama"; }

  int VocabSize() const override { return vocab_size_; }
  int HiddenSize() const override { return hidden_size_; }
  bool EmbedForward(const std::vector<int> &token_ids, int sequence_id,
                    float *d_output) override;

private:
  // Model config
  int hidden_size_{0};
  int num_layers_{0};
  int num_heads_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int intermediate_size_{0};
  int vocab_size_{0};
  int max_seq_len_{0};
  int max_batch_size_{32};
  float rope_freq_base_{10000.0f};
  float rms_norm_eps_{1e-5f};
  int rope_type_{0}; // 0 = kNorm (consecutive), 2 = kNeox (split-half)

  // External references (not owned)
  const WeightMap *weights_{nullptr};
  IKvCacheGpu *kv_cache_{nullptr};
  CublasGemm *gemm_{nullptr};
  cudaStream_t stream_{nullptr};

  // Scratch buffers on GPU (pre-allocated for max_seq_len)
  T *d_hidden_{nullptr};
  T *d_residual_{nullptr};
  T *d_norm_out_{nullptr};
  T *d_q_{nullptr};
  T *d_k_new_{nullptr};
  T *d_v_new_{nullptr};
  T *d_attn_out_{nullptr};
  // Split-attention merge partials (see FlashDecodeMultiSeqStridedSplit).
  static constexpr int kMaxAttnKSplits = 8;
  float *d_attn_partials_{nullptr};
  T *d_ffn_gate_{nullptr};
  T *d_ffn_up_{nullptr};
  T *d_ffn_down_{nullptr};
  int8_t *d_packed_activation_{nullptr};
  float *d_packed_activation_scales_{nullptr};
  void *d_act_q8_1_{nullptr}; // Pre-quantized Q8_1 activation buffer
  int *d_token_ids_{nullptr};
  T *d_logits_typed_{nullptr};

  // Batch metadata buffers (pre-allocated for max_batch_size)
  int *d_batch_meta_{nullptr};
  int *d_batch_token_ids_{nullptr};
  int *d_batch_n_past_{nullptr};
  int *d_batch_seq_ids_{nullptr};
  int *d_batch_kv_lens_{nullptr};
  int *h_batch_meta_{nullptr};
  int *h_batch_token_ids_{nullptr};
  int *h_batch_n_past_{nullptr};
  int *h_batch_seq_ids_{nullptr};
  int *h_batch_kv_lens_{nullptr};

  // CUDA graph state for batched decode.
  // graph_warmup_remaining_ skips capture for the first N calls to let lazy
  // allocations and weight dequantizations settle before attempting capture.
  cudaGraph_t decode_graph_{nullptr};
  cudaGraphExec_t decode_graph_exec_{nullptr};
  int graph_batch_size_{0};
  bool graph_enabled_{true};
  int graph_warmup_remaining_{4};
  NativeExecutionPolicy execution_policy_{};
  std::size_t device_workspace_bytes_{0};
  std::size_t host_workspace_bytes_{0};

  bool AllocateScratch();
};

// Backward-compatible alias
using LlamaForward = LlamaForwardTyped<half>;

} // namespace inferflux
