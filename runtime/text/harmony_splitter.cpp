#include "runtime/text/harmony_splitter.h"

#include <algorithm>
#include <string_view>
#include <vector>

namespace inferflux {

namespace {

constexpr std::string_view kAnalysisOpen = "<|channel|>analysis<|message|>";
constexpr std::string_view kFinalOpen = "<|channel|>final<|message|>";
constexpr std::string_view kCommentaryOpen = "<|channel|>commentary";
constexpr std::string_view kMessage = "<|message|>";
constexpr std::string_view kEnd = "<|end|>";
constexpr std::string_view kReturn = "<|return|>";
// The only role gpt-oss's own generated tokens re-emit mid-completion is
// its own ("assistant"), switching channels within one turn — user/system/
// developer only ever appear in the prompt we construct, never in the
// model's output. Treated as a no-op: consumed, no state change.
constexpr std::string_view kStartAssistant = "<|start|>assistant";

// Finds the earliest occurrence of any candidate in `text`. Ties (equal
// start position) never happen for this marker set — every prefix
// relationship between candidates (e.g. kStartAssistant + kAnalysisOpen)
// necessarily starts the longer/combined form at an earlier or equal
// position, and earliest-position selection alone is sufficient because
// each caller's candidate list only ever contains mutually non-nesting
// markers.
struct MarkerHit {
  std::size_t pos = std::string_view::npos;
  std::string_view marker;
};

MarkerHit FindEarliestMarker(std::string_view text,
                             const std::vector<std::string_view> &candidates) {
  MarkerHit best;
  for (auto cand : candidates) {
    std::size_t pos = text.find(cand);
    if (pos != std::string_view::npos && pos < best.pos) {
      best.pos = pos;
      best.marker = cand;
    }
  }
  return best;
}

// Longest suffix of `text` that could still become a prefix of one of the
// candidates if more bytes arrive — held back so a marker straddling a
// Feed() boundary is never split.
std::size_t HoldbackSuffixLen(std::string_view text,
                              const std::vector<std::string_view> &candidates) {
  std::size_t max_len = 0;
  for (auto cand : candidates) {
    max_len = std::max(max_len, cand.size() - 1);
  }
  for (std::size_t k = std::min(max_len, text.size()); k >= 1; --k) {
    std::string_view tail = text.substr(text.size() - k);
    for (auto cand : candidates) {
      if (k < cand.size() && cand.compare(0, k, tail) == 0) {
        return k;
      }
    }
  }
  return 0;
}

} // namespace

bool HarmonySplitter::in_reasoning() const { return state_ == State::Analysis; }

namespace {

// Emits the safe (cannot-become-part-of-a-marker) prefix of `pending` to
// `sink` (nullptr to discard, for commentary/scan states), leaving only the
// holdback tail behind. Returns true if anything was emitted/discarded, so
// the caller can decide whether to keep looping (more bytes might still
// resolve — no, see below) or stop.
bool EmitSafePrefix(std::string *pending, std::string *sink,
                    const std::vector<std::string_view> &candidates) {
  std::size_t hold = HoldbackSuffixLen(*pending, candidates);
  std::size_t emit = pending->size() - hold;
  if (emit == 0) {
    return false;
  }
  if (sink) {
    sink->append(*pending, 0, emit);
  }
  pending->erase(0, emit);
  return true;
}

} // namespace

void HarmonySplitter::Feed(std::string_view piece) {
  pending_ += piece;
  bool progressed = true;
  while (progressed && !pending_.empty()) {
    progressed = false;
    std::string_view text(pending_);

    switch (state_) {
    case State::Normal: {
      // Outside any channel: the only things that can legally appear are
      // channel openers or the no-op start-of-turn marker. Everything else
      // is unexpected (malformed/non-harmony output) — flushed to content
      // as the safe default.
      const std::vector<std::string_view> candidates = {
          kAnalysisOpen, kFinalOpen, kCommentaryOpen, kStartAssistant};
      auto hit = FindEarliestMarker(text, candidates);
      if (hit.pos == std::string_view::npos) {
        progressed = EmitSafePrefix(&pending_, &content_, candidates);
        break;
      }
      content_.append(pending_, 0, hit.pos);
      pending_.erase(0, hit.pos + hit.marker.size());
      if (hit.marker == kAnalysisOpen) {
        state_ = State::Analysis;
      } else if (hit.marker == kFinalOpen) {
        state_ = State::Final;
      } else if (hit.marker == kCommentaryOpen) {
        state_ = State::CommentaryScan;
      } // kStartAssistant: no-op, stay Normal.
      progressed = true;
      continue;
    }
    case State::Analysis:
    case State::Final: {
      std::string &sink = (state_ == State::Analysis) ? reasoning_ : content_;
      const std::vector<std::string_view> candidates = {kEnd, kReturn};
      auto hit = FindEarliestMarker(text, candidates);
      if (hit.pos == std::string_view::npos) {
        progressed = EmitSafePrefix(&pending_, &sink, candidates);
        break;
      }
      sink.append(pending_, 0, hit.pos);
      pending_.erase(0, hit.pos + hit.marker.size());
      state_ = State::Normal;
      progressed = true;
      continue;
    }
    case State::CommentaryScan: {
      // gpt-oss occasionally emits a stray, vestigial "<|channel|>commentary"
      // with no <|message|> of its own immediately before the real next
      // channel opens (a documented upstream quirk) — watch for that escape
      // hatch alongside the normal <|message|> transition into the
      // commentary body. Everything scanned here is discarded regardless
      // of which candidate resolves it.
      const std::vector<std::string_view> candidates = {kMessage, kAnalysisOpen,
                                                        kFinalOpen};
      auto hit = FindEarliestMarker(text, candidates);
      if (hit.pos == std::string_view::npos) {
        progressed = EmitSafePrefix(&pending_, nullptr, candidates);
        break;
      }
      pending_.erase(0, hit.pos + hit.marker.size());
      if (hit.marker == kMessage) {
        state_ = State::Commentary;
      } else if (hit.marker == kAnalysisOpen) {
        state_ = State::Analysis;
      } else { // kFinalOpen
        state_ = State::Final;
      }
      progressed = true;
      continue;
    }
    case State::Commentary: {
      // Tool-call chatter — never surfaced. Discard through the closer.
      const std::vector<std::string_view> candidates = {kEnd, kReturn};
      auto hit = FindEarliestMarker(text, candidates);
      if (hit.pos == std::string_view::npos) {
        progressed = EmitSafePrefix(&pending_, nullptr, candidates);
        break;
      }
      pending_.erase(0, hit.pos + hit.marker.size());
      state_ = State::Normal;
      progressed = true;
      continue;
    }
    }
  }
}

HarmonySplitter::Parts HarmonySplitter::Finish() {
  if (!pending_.empty()) {
    switch (state_) {
    case State::Analysis:
      reasoning_ += pending_;
      break;
    case State::Final:
    case State::Normal:
      content_ += pending_;
      break;
    case State::CommentaryScan:
    case State::Commentary:
      // Unterminated commentary — drop, consistent with commentary never
      // being surfaced.
      break;
    }
    pending_.clear();
  }
  Parts parts;
  parts.reasoning = std::move(reasoning_);
  parts.content = std::move(content_);
  return parts;
}

HarmonySplitter::Parts HarmonySplitter::Drain() {
  Parts parts;
  parts.reasoning = reasoning_.substr(drained_reasoning_bytes_);
  parts.content = content_.substr(drained_content_bytes_);
  drained_reasoning_bytes_ = reasoning_.size();
  drained_content_bytes_ = content_.size();
  return parts;
}

HarmonySplitter::Parts HarmonySplitter::Split(std::string_view output) {
  HarmonySplitter splitter;
  splitter.Feed(output);
  return splitter.Finish();
}

} // namespace inferflux
