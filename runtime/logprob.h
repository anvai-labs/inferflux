#pragma once

#include <string>
#include <utility>
#include <vector>

namespace inferflux {

// Per-token logprob entry returned by Generate()/Decode() and surfaced in the
// OpenAI-compatible API response (choices[].logprobs).
struct TokenLogprob {
  std::string token;      // Token string as decoded from vocab.
  float logprob{0.0f};    // Log-probability of the sampled token (log-softmax).
  std::vector<int> bytes; // UTF-8 byte values of `token` (OpenAI spec field).
  // Top-N alternative tokens at this position; empty when top_logprob_n == 0.
  std::vector<std::pair<std::string, float>> top_logprobs;
};

// Raw top-logit entry used by parity/debug probes.
struct TopLogitEntry {
  std::string token;
  float logit{0.0f};
};

} // namespace inferflux
