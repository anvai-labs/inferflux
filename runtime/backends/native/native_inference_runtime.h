#pragma once

// NativeInferenceRuntime: device-neutral interface for InferFlux native
// runtimes.  Extracted from InferfluxCudaRuntime so that the same backend base
// class (NativeGpuBackend) can own any native runtime regardless of hardware.

#include "runtime/backends/common/backend_config.h"
#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/common/backend_types.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace inferflux {

class ITokenizer;

class NativeInferenceRuntime {
public:
  using UnifiedBatchHandle = ::inferflux::UnifiedBatchHandle;
  using UnifiedBatchInput = ::inferflux::UnifiedBatchInput;
  using UnifiedBatchLane = ::inferflux::UnifiedBatchLane;
  using UnifiedBatchOutput = ::inferflux::UnifiedBatchOutput;

  virtual ~NativeInferenceRuntime() = default;

  virtual std::string Name() const = 0;
  virtual bool IsFallback() const = 0;
  virtual const std::string &FallbackReason() const = 0;

  virtual bool LoadModel(const std::filesystem::path &model_path,
                         const LlamaBackendConfig &config) = 0;
  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) = 0;
  // Burst-pipelined decode (optional). Default: unsupported.
  virtual bool NativeSupportsUnifiedBatchBurst() const { return false; }
  virtual UnifiedBurstResult
  NativeExecuteUnifiedBatchBurst(const std::vector<UnifiedBatchInput> &inputs,
                                 const UnifiedBurstOptions &options,
                                 const BurstTokenSink &sink) {
    (void)inputs;
    (void)options;
    (void)sink;
    return UnifiedBurstResult{};
  }
  virtual bool SupportsAsyncUnifiedBatch() const = 0;
  virtual UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) = 0;
  virtual bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs) = 0;
  virtual std::shared_ptr<BackendInterface> BackendHandle() const = 0;

  struct PerfSnapshot {
    double prefill_ms{0.0};
    double decode_ms{0.0};
    int32_t prompt_tokens{0};
    int32_t generated_tokens{0};
  };
  virtual PerfSnapshot NativeTakePerf() { return {}; }

  virtual std::vector<int> NativeTokenize(const std::string &prompt) const {
    auto backend = BackendHandle();
    return backend ? backend->TokenizeForCache(prompt) : std::vector<int>{};
  }

  virtual int NativeTokenCount(const std::string &text) const {
    auto backend = BackendHandle();
    return backend ? backend->TokenCount(text) : 0;
  }

  virtual bool NativeIsReady() const {
    auto backend = BackendHandle();
    return backend && backend->IsReady();
  }

  virtual void NativeFreeSequence(int sequence_id) {
    auto backend = BackendHandle();
    if (backend) {
      backend->FreeSequence(sequence_id);
    }
  }

  virtual SequenceReleaseFence NativeBeginFreeSequence(int sequence_id) {
    NativeFreeSequence(sequence_id);
    (void)sequence_id;
    return {};
  }

  virtual bool NativePollFreeSequence(const SequenceReleaseFence &fence) {
    (void)fence;
    return true;
  }

  virtual void NativeCopySequencePrefix(int src_seq, int dst_seq,
                                        int n_tokens) {
    auto backend = BackendHandle();
    if (backend) {
      backend->CopySequencePrefix(src_seq, dst_seq, n_tokens);
    }
  }

  virtual std::vector<uint8_t> NativeSerializeSequence(int sequence_id) const {
    auto backend = BackendHandle();
    return backend ? backend->SerializeSequence(sequence_id)
                   : std::vector<uint8_t>{};
  }

  virtual bool NativeHydrateSequence(int dest_sequence_id,
                                     const std::vector<uint8_t> &blob) {
    auto backend = BackendHandle();
    return backend ? backend->HydrateSequence(dest_sequence_id, blob) : false;
  }

  struct ChatResult {
    std::string prompt;
    bool valid{false};
  };

  virtual ChatResult NativeFormatChat(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const {
    (void)messages;
    (void)add_assistant_prefix;
    return {};
  }

  virtual const ITokenizer *NativeGetTokenizer() const { return nullptr; }

  virtual int CopyLastLogitsToHost(float *host_buf, int buf_size) {
    (void)host_buf;
    (void)buf_size;
    return 0;
  }

  virtual int NativeVocabSize() const { return 0; }

  virtual std::vector<float> NativeEmbed(const std::string &text) {
    (void)text;
    return {};
  }

  virtual int NativeEmbedDims() const { return 0; }
};

} // namespace inferflux
