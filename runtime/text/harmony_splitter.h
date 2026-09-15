#pragma once

#include <string>
#include <string_view>

namespace inferflux {

// Splits model output produced under gpt-oss's harmony chat format into
// reasoning vs user-facing content. Harmony structures output into named
// channels — <|channel|>analysis<|message|>...<|end|> (reasoning),
// <|channel|>final<|message|>...<|end|or|return|> (user-facing), and
// <|channel|>commentary...<|message|>...<|end|> (tool-call chatter, not
// implemented here — silently dropped, never surfaced) — closed by either
// <|end|> or <|return|>, and turns are separated by a <|start|> role
// marker that is always stripped regardless of channel state.
//
// One instance per request, mirroring ReasoningSplitter's shape exactly
// (same Feed/Finish/Drain/Split API, same Parts{reasoning, content} return
// type) so callers can pick either splitter based on the detected chat
// template family without changing call-site shape. Not thread-safe: one
// instance per request, fed from one thread.
class HarmonySplitter {
public:
  struct Parts {
    std::string reasoning;
    std::string content;
  };

  HarmonySplitter() = default;

  // Feed one generation piece (a token, a chunk, or the whole output).
  void Feed(std::string_view piece);

  // Flush held-back bytes and return the accumulated split. Any text still
  // pending inside an unclosed commentary channel is dropped, not flushed —
  // consistent with commentary never being surfaced. Text pending outside
  // any channel (a malformed/non-harmony completion) is flushed to content,
  // the safe default.
  Parts Finish();

  // Return the bytes accumulated since the last Drain() (or since start).
  Parts Drain();

  // True while inside the analysis channel (reasoning still streaming).
  bool in_reasoning() const;

  // Whole-string convenience path (the test oracle): one-shot split that
  // never holds bytes back.
  static Parts Split(std::string_view output);

private:
  enum class State {
    Normal,         // outside any channel; scanning for a channel opener
    Analysis,       // inside the analysis channel -> reasoning
    Final,          // inside the final channel -> content
    CommentaryScan, // saw "<|channel|>commentary", scanning for <|message|>
    Commentary,     // inside the commentary channel -> dropped
  };
  State state_{State::Normal};
  std::string pending_; // bytes that may be a partial marker — held back
  std::string reasoning_;
  std::string content_;
  std::size_t drained_reasoning_bytes_{0};
  std::size_t drained_content_bytes_{0};
};

} // namespace inferflux
