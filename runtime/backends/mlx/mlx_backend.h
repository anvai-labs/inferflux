#pragma once

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/backends/mlx/mlx_execution.h"
#include "runtime/backends/mlx/mlx_loader.h"
#include "runtime/backends/mlx/mlx_tokenizer.h"

namespace inferflux {

class MlxBackend : public LlamaCppBackend {
public:
  MlxBackend();
  ~MlxBackend() override;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

  // Evaluate all prompt tokens for sequence_id and populate the KV cache.
  PrefillResult Prefill(const std::string &prompt, int sequence_id) override;

  // Partial prefill that evaluates only the suffix of prompt starting at
  // n_past_start.
  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override;

  // Autoregressive decode starting from n_past (returned by Prefill) for
  // sequence_id.
  std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1,
         const std::vector<std::string> &stop_seqs = {}) override;

  // Release KV cache slots for the given sequence_id.
  void FreeSequence(int sequence_id) override;

  // Copy KV cache entries for positions [0, n_tokens) from src_seq to dst_seq.
  void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) override;

  // Execute a mixed batch of prefill and decode sequences.
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {}) override;

  void SetupSampler(const std::string &grammar, const std::string &root,
                    const SamplingParams &sp) override;
  void TeardownSampler() override;

  bool IsReady() const override;
  int TokenCount(const std::string &text) const override;

  PerfSnapshot TakePerf() override;

  ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) override;

  std::vector<int> TokenizeForCache(const std::string &prompt) const override;

private:
  MlxExecutionEngine engine_;
  MlxTokenizer tokenizer_;
  MlxWeightLoader loader_;
  MlxModelDescriptor descriptor_;
  MlxWeightStore weight_store_;
  bool engine_ready_{false};

  // Sampling params stored by SetupSampler(), used during Generate().
  SamplingParams mlx_sp_{};
  std::string mlx_grammar_{};

  // Perf snapshot populated at the end of Generate().
  mutable PerfSnapshot mlx_perf_{};
};

} // namespace inferflux
