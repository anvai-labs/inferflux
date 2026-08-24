#pragma once

// Shared utilities used by multiple backends (LlamaCppBackend, MlxBackend).
// Keep this header dependency-free: no llama.h, no mlx headers.

#include "runtime/logprob.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace inferflux {

// Check whether output ends with any stop sequence.
// If matched: output is trimmed to remove the stop suffix, and *emit_piece is
// set to the portion of piece that precedes the stop (may be empty if the
// entire piece was the stop sequence or part of it).
// Returns true when a stop sequence was matched.
inline bool ApplyStop(const std::string &piece, std::string &output,
                      const std::vector<std::string> &stops,
                      std::string *emit_piece) {
  *emit_piece = piece;
  for (const auto &s : stops) {
    if (s.empty())
      continue;
    if (output.size() >= s.size() &&
        output.compare(output.size() - s.size(), s.size(), s) == 0) {
      size_t pre_piece_len = output.size() - piece.size();
      size_t stop_start = output.size() - s.size();
      output.resize(stop_start);
      *emit_piece = (stop_start > pre_piece_len)
                        ? piece.substr(0, stop_start - pre_piece_len)
                        : "";
      return true;
    }
  }
  return false;
}

// Compute log-softmax + collect token logprob + up to top_n alternatives.
// `logits` must point to `vocab_size` floats (last-position logits).
// `id_to_token` maps token id → display string.
inline TokenLogprob
ComputeLogprob(const float *logits, int vocab_size, int32_t token_id,
               const std::string &token_str, int top_n,
               const std::function<std::string(int32_t)> &id_to_token) {
  TokenLogprob tlp;
  tlp.token = token_str;

  if (!logits || vocab_size <= 0) {
    return tlp;
  }
  if (token_id < 0 || token_id >= vocab_size) {
    return tlp;
  }

  // Numerically stable log-softmax: subtract max before exp.
  float max_l = logits[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (logits[i] > max_l)
      max_l = logits[i];
  }
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    sum_exp += std::exp(static_cast<double>(logits[i] - max_l));
  }
  float log_denom = static_cast<float>(std::log(sum_exp)) + max_l;

  tlp.logprob = logits[token_id] - log_denom;

  // UTF-8 bytes of the token string.
  tlp.bytes.reserve(token_str.size());
  for (unsigned char c : token_str) {
    tlp.bytes.push_back(static_cast<int>(c));
  }

  if (top_n > 0) {
    int k = std::min(top_n, vocab_size);
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    tlp.top_logprobs.reserve(k);
    for (int i = 0; i < k; ++i) {
      int alt_id = indices[i];
      tlp.top_logprobs.push_back(
          {id_to_token(alt_id), logits[alt_id] - log_denom});
    }
  }
  return tlp;
}

// Collect the top-k raw logits without applying softmax.
inline std::vector<TopLogitEntry>
ComputeTopLogits(const float *logits, int vocab_size, int top_n,
                 const std::function<std::string(int32_t)> &id_to_token) {
  if (!logits || vocab_size <= 0 || top_n <= 0) {
    return {};
  }

  const int k = std::min(top_n, vocab_size);
  std::vector<int> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });

  std::vector<TopLogitEntry> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int token_id = indices[i];
    out.push_back({id_to_token(token_id), logits[token_id]});
  }
  return out;
}

} // namespace inferflux
