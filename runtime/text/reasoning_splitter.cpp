#include "runtime/text/reasoning_splitter.h"

#include <string_view>

namespace inferflux {

namespace {

constexpr std::string_view kOpenTag = "<think>";
constexpr std::string_view kCloseTag = "</think>";
constexpr std::size_t kMaxTagLen = kCloseTag.size(); // 8

// True when text starts with tag.
bool StartsWithTag(std::string_view text, std::string_view tag) {
  return text.size() >= tag.size() && text.compare(0, tag.size(), tag) == 0;
}

// True when text could become `tag` if more bytes arrived
// (text is a proper prefix of tag).
bool MaybePartialTag(std::string_view text, std::string_view tag) {
  return text.size() < tag.size() && tag.compare(0, text.size(), text) == 0;
}

} // namespace

// Feed appends the piece, then emits everything that cannot be part of a
// partial tag: in Normal state we hold back at most 7 bytes (a potential
// "<think>" or "</think>" start); in InThink we emit reasoning and only
// hold back a potential "</think>" close.
void ReasoningSplitter::Feed(std::string_view piece) {
  pending_ += piece;
  bool progressed = true;
  while (progressed && !pending_.empty()) {
    progressed = false;
    if (state_ == State::InThink) {
      const std::size_t close = pending_.find(kCloseTag);
      if (close != std::string_view::npos) {
        reasoning_.append(pending_, 0, close);
        pending_.erase(0, close + kCloseTag.size());
        state_ = State::Normal;
        progressed = true;
        continue;
      }
      // No close tag: everything except a possible partial close-tag
      // suffix is reasoning.
      std::size_t keep = 0;
      for (std::size_t k = kCloseTag.size() - 1; k >= 1; --k) {
        if (pending_.size() >= k &&
            pending_.compare(pending_.size() - k, k, kCloseTag.data(), k) ==
                0) {
          keep = k;
          break;
        }
      }
      const std::size_t emit = pending_.size() - keep;
      if (emit == 0) {
        break;
      }
      reasoning_.append(pending_, 0, emit);
      pending_.erase(0, emit);
      progressed = emit > 0;
      continue;
    }
    // Normal state.
    if (StartsWithTag(pending_, kOpenTag)) {
      pending_.erase(0, kOpenTag.size());
      state_ = State::InThink;
      progressed = true;
      continue;
    }
    if (StartsWithTag(pending_, kCloseTag)) {
      pending_.erase(0, kCloseTag.size());
      progressed = true;
      continue;
    }
    // Hold back the longest suffix that could complete a tag.
    std::size_t hold = 0;
    for (std::size_t k = kMaxTagLen; k >= 1; --k) {
      if (pending_.size() >= k) {
        const std::string_view tail =
            std::string_view(pending_).substr(pending_.size() - k);
        if (MaybePartialTag(tail, kOpenTag) ||
            MaybePartialTag(tail, kCloseTag)) {
          hold = k;
          break;
        }
      }
    }
    const std::size_t emit = pending_.size() - hold;
    if (emit > 0) {
      content_.append(pending_, 0, emit);
      pending_.erase(0, emit);
      progressed = true;
    }
    if (emit == 0) {
      break; // waiting for more bytes to disambiguate a tag start
    }
  }
}

ReasoningSplitter::Parts ReasoningSplitter::Finish() {
  // Flush held-back bytes into the accumulator they belong to.
  if (!pending_.empty()) {
    if (state_ == State::InThink) {
      reasoning_ += pending_;
    } else {
      content_ += pending_;
    }
    pending_.clear();
  }
  Parts parts;
  parts.reasoning = std::move(reasoning_);
  parts.content = std::move(content_);
  reasoning_bytes_ = parts.reasoning.size();
  content_bytes_ = parts.content.size();
  return parts;
}

ReasoningSplitter::Parts ReasoningSplitter::Drain() {
  Parts parts;
  parts.reasoning = reasoning_.substr(drained_reasoning_bytes_);
  parts.content = content_.substr(drained_content_bytes_);
  drained_reasoning_bytes_ = reasoning_.size();
  drained_content_bytes_ = content_.size();
  return parts;
}

ReasoningSplitter::Parts ReasoningSplitter::Split(std::string_view output) {
  ReasoningSplitter splitter;
  splitter.Feed(output);
  return splitter.Finish();
}

} // namespace inferflux
