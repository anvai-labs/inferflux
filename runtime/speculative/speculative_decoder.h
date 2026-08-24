#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/cpu/cpu_backend.h"

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace inferflux {

struct SpeculativeConfig {
  bool enabled{false};
  int max_prefill_tokens{256};
  int chunk_size{4};
  std::string draft_model;
};

struct SpeculativeChunk {
  std::size_t start{0};
  std::size_t end{0};
};

struct SpeculativeDraft {
  std::vector<int> completion_tokens;
  std::vector<SpeculativeChunk> chunks;
};

struct SpeculativeMetrics {
  std::size_t total_chunks{0};
  std::size_t accepted_chunks{0};
  std::size_t reused_tokens{0};
};

struct SpeculativeValidationResult {
  std::vector<int> completion_tokens;
  SpeculativeMetrics metrics;
};

class SpeculativeDecoder {
public:
  SpeculativeDecoder(SpeculativeConfig config,
                     std::shared_ptr<CPUDeviceContext> device,
                     SimpleTokenizer *tokenizer,
                     std::shared_ptr<BackendInterface> draft_backend);

  bool Enabled() const { return config_.enabled; }
  SpeculativeConfig Config() const { return config_; }

  SpeculativeDraft Draft(const std::vector<int> &prompt_tokens,
                         int max_new_tokens);
  SpeculativeValidationResult
  Validate(const std::vector<int> &prompt_tokens, const SpeculativeDraft &draft,
           int max_new_tokens,
           std::shared_ptr<BackendInterface> target_backend);

  using ValidationOverride = std::function<std::vector<int>(
      const std::vector<int> &prompt_tokens, int max_new_tokens)>;

  void SetValidationOverride(ValidationOverride cb);
  void ClearValidationOverride();
  std::string DraftModel() const { return config_.draft_model; }

private:
  SpeculativeConfig config_;
  std::shared_ptr<CPUDeviceContext> device_;
  SimpleTokenizer *tokenizer_;
  std::shared_ptr<BackendInterface> draft_backend_;
  ValidationOverride validation_override_;
  mutable std::mutex override_mutex_;
};

} // namespace inferflux
