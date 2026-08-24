#include "runtime/speculative/speculative_decoder.h"

#include <algorithm>
#include <cstddef>
#include <iostream>

namespace inferflux {

SpeculativeDecoder::SpeculativeDecoder(
    SpeculativeConfig config, std::shared_ptr<CPUDeviceContext> device,
    SimpleTokenizer *tokenizer, std::shared_ptr<BackendInterface> draft_backend)
    : config_(std::move(config)), device_(std::move(device)),
      tokenizer_(tokenizer), draft_backend_(std::move(draft_backend)) {}

SpeculativeDraft
SpeculativeDecoder::Draft(const std::vector<int> &prompt_tokens,
                          int max_new_tokens) {
  SpeculativeDraft draft;
  if (!config_.enabled || max_new_tokens <= 0) {
    return draft;
  }

  std::vector<int> prompt = prompt_tokens;
  if (config_.max_prefill_tokens > 0 &&
      static_cast<int>(prompt.size()) > config_.max_prefill_tokens) {
    prompt.resize(config_.max_prefill_tokens);
  }

  std::vector<int> new_tokens;
  if (draft_backend_ && draft_backend_->IsReady()) {
    auto draft_text = draft_backend_->Generate(
        tokenizer_->Decode(prompt_tokens), max_new_tokens);
    new_tokens = tokenizer_->Encode(draft_text);
  } else {
    auto greedy = device_->RunGreedyDecode(prompt);
    if (greedy.size() > prompt.size()) {
      new_tokens.assign(greedy.begin() +
                            static_cast<std::ptrdiff_t>(prompt.size()),
                        greedy.end());
    } else {
      new_tokens = greedy;
    }
  }
  if (max_new_tokens > 0 &&
      static_cast<int>(new_tokens.size()) > max_new_tokens) {
    new_tokens.resize(max_new_tokens);
  }

  if (new_tokens.empty()) {
    return draft;
  }

  int chunk_size = std::max(1, config_.chunk_size);
  draft.completion_tokens.reserve(new_tokens.size());
  std::size_t cursor = 0;
  while (cursor < new_tokens.size()) {
    SpeculativeChunk chunk;
    chunk.start = draft.completion_tokens.size();
    for (int i = 0; i < chunk_size && cursor < new_tokens.size();
         ++i, ++cursor) {
      draft.completion_tokens.push_back(new_tokens[cursor]);
    }
    chunk.end = draft.completion_tokens.size();
    draft.chunks.push_back(chunk);
  }

  std::cout << "[speculative] draft produced " << draft.completion_tokens.size()
            << " tokens across " << draft.chunks.size()
            << " chunks (chunk_size=" << chunk_size << ")\n";
  return draft;
}

SpeculativeValidationResult
SpeculativeDecoder::Validate(const std::vector<int> &prompt_tokens,
                             const SpeculativeDraft &draft, int max_new_tokens,
                             std::shared_ptr<BackendInterface> target_backend) {
  SpeculativeValidationResult result;
  result.metrics.total_chunks = draft.chunks.size();
  if (max_new_tokens <= 0) {
    return result;
  }

  std::vector<int> target_tokens;
  ValidationOverride override_cb;
  {
    std::lock_guard<std::mutex> lock(override_mutex_);
    override_cb = validation_override_;
  }

  if (override_cb) {
    target_tokens = override_cb(prompt_tokens, max_new_tokens);
  } else if (target_backend && target_backend->IsReady()) {
    auto prompt_text = tokenizer_->Decode(prompt_tokens);
    auto validated = target_backend->Generate(prompt_text, max_new_tokens);
    target_tokens = tokenizer_->Encode(validated);
  } else {
    target_tokens = draft.completion_tokens;
  }

  if (target_tokens.empty()) {
    return result;
  }

  std::size_t target_index = 0;
  int limit = max_new_tokens;

  for (const auto &chunk : draft.chunks) {
    if (static_cast<int>(result.completion_tokens.size()) >= limit) {
      break;
    }
    bool chunk_ok = true;
    std::vector<int> chunk_tokens;
    for (std::size_t idx = chunk.start; idx < chunk.end; ++idx) {
      if (static_cast<int>(result.completion_tokens.size() +
                           chunk_tokens.size()) >= limit) {
        chunk_ok = false;
        break;
      }
      if (target_index >= target_tokens.size() ||
          draft.completion_tokens[idx] != target_tokens[target_index]) {
        chunk_ok = false;
        break;
      }
      chunk_tokens.push_back(draft.completion_tokens[idx]);
      ++target_index;
    }
    if (chunk_tokens.empty()) {
      break;
    }
    result.metrics.reused_tokens += chunk_tokens.size();
    result.completion_tokens.insert(result.completion_tokens.end(),
                                    chunk_tokens.begin(), chunk_tokens.end());
    if (chunk_ok && chunk_tokens.size() == (chunk.end - chunk.start)) {
      result.metrics.accepted_chunks++;
    } else {
      break;
    }
  }

  while (target_index < target_tokens.size() &&
         static_cast<int>(result.completion_tokens.size()) < limit) {
    result.completion_tokens.push_back(target_tokens[target_index++]);
  }
  return result;
}

void SpeculativeDecoder::SetValidationOverride(ValidationOverride cb) {
  std::lock_guard<std::mutex> lock(override_mutex_);
  validation_override_ = std::move(cb);
}

void SpeculativeDecoder::ClearValidationOverride() {
  std::lock_guard<std::mutex> lock(override_mutex_);
  validation_override_ = nullptr;
}

} // namespace inferflux
