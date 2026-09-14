#pragma once

#include <string>

namespace inferflux {

// Splits model output into reasoning vs user-facing content for models that
// emit <think>...</think> blocks (Qwen3, LFM2.5, and other reasoning models).
//
// One instance per request. Feed pieces in generation order — Feed works for
// both whole-string and per-token streaming input and handles tags that
// straddle piece boundaries. Classify() reports the running split; Finish()
// flushes any held-back bytes. reasoning_tokens are counted by classifying
// each fed piece (never by re-tokenizing the final string).
//
// Not thread-safe: use one instance per request, fed from one thread.
class ReasoningSplitter {
public:
  struct Parts {
    std::string reasoning;
    std::string content;
  };

  ReasoningSplitter() = default;

  // Feed one generation piece (a token, a chunk, or the whole output).
  // Appends text to the running reasoning or content per the current state.
  void Feed(std::string_view piece);

  // Flush held-back bytes and return the accumulated split.
  Parts Finish();

  // True while inside a <think> block (reasoning still streaming).
  bool in_reasoning() const { return state_ == State::InThink; }

  std::size_t reasoning_bytes() const { return reasoning_bytes_; }
  std::size_t content_bytes() const { return content_bytes_; }

  // Whole-string convenience path (the test oracle): one-shot split that
  // never holds bytes back.
  static Parts Split(std::string_view output);

private:
  enum class State { Normal, InThink };
  State state_{State::Normal};
  std::string pending_; // bytes that may be a partial tag — held back
  std::string reasoning_;
  std::string content_;
  std::size_t reasoning_bytes_{0};
  std::size_t content_bytes_{0};
};

} // namespace inferflux
