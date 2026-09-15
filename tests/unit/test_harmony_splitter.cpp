#include <catch2/catch_amalgamated.hpp>

#include "runtime/text/harmony_splitter.h"

using inferflux::HarmonySplitter;

namespace {

// Feed the string in per-token-sized pieces (simulates streaming).
HarmonySplitter::Parts FeedByPieces(const std::string &output, std::size_t n) {
  HarmonySplitter splitter;
  for (std::size_t i = 0; i < output.size(); i += n) {
    splitter.Feed(output.substr(i, n));
  }
  return splitter.Finish();
}

} // namespace

TEST_CASE("HarmonySplitter: plain non-harmony output is all content",
          "[harmony]") {
  auto parts = HarmonySplitter::Split("Just a plain answer.");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "Just a plain answer.");
}

TEST_CASE("HarmonySplitter: analysis then final splits reasoning/content",
          "[harmony]") {
  auto parts =
      HarmonySplitter::Split("<|channel|>analysis<|message|>step by step<|end|>"
                             "<|start|>assistant<|channel|>final<|message|>The "
                             "answer is 4.<|return|>");
  REQUIRE(parts.reasoning == "step by step");
  REQUIRE(parts.content == "The answer is 4.");
}

TEST_CASE("HarmonySplitter: final channel closed with <|end|> instead of "
          "<|return|>",
          "[harmony]") {
  auto parts =
      HarmonySplitter::Split("<|channel|>analysis<|message|>thinking<|end|>"
                             "<|channel|>final<|message|>done<|end|>");
  REQUIRE(parts.reasoning == "thinking");
  REQUIRE(parts.content == "done");
}

TEST_CASE("HarmonySplitter: unterminated analysis flushes to reasoning",
          "[harmony]") {
  auto parts =
      HarmonySplitter::Split("<|channel|>analysis<|message|>cut off mid");
  REQUIRE(parts.content.empty());
  REQUIRE(parts.reasoning == "cut off mid");
}

TEST_CASE("HarmonySplitter: final-only output (no reasoning) is content-only",
          "[harmony]") {
  auto parts =
      HarmonySplitter::Split("<|channel|>final<|message|>Paris.<|return|>");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "Paris.");
}

TEST_CASE("HarmonySplitter: commentary channel is dropped, never surfaced",
          "[harmony]") {
  auto parts = HarmonySplitter::Split(
      "<|channel|>commentary<|message|>tool chatter<|end|>"
      "<|channel|>final<|message|>answer<|return|>");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "answer");
}

TEST_CASE("HarmonySplitter: stray commentary prefix with no message of its "
          "own is skipped",
          "[harmony]") {
  // Documented upstream quirk: gpt-oss-20b occasionally prefixes the real
  // analysis channel with a vestigial "<|channel|>commentary" that never
  // gets its own <|message|>.
  auto parts = HarmonySplitter::Split(
      "<|channel|>commentary<|channel|>analysis<|message|>real thinking"
      "<|end|><|channel|>final<|message|>real answer<|return|>");
  REQUIRE(parts.reasoning == "real thinking");
  REQUIRE(parts.content == "real answer");
}

TEST_CASE("HarmonySplitter: stray commentary with a 'to=' recipient suffix "
          "is skipped",
          "[harmony]") {
  auto parts = HarmonySplitter::Split(
      "<|channel|>commentary to=assistant<|channel|>final<|message|>hi"
      "<|return|>");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "hi");
}

TEST_CASE("HarmonySplitter: <|start|>assistant between channels is a no-op",
          "[harmony]") {
  auto parts = HarmonySplitter::Split(
      "<|channel|>analysis<|message|>think<|end|>"
      "<|start|>assistant<|channel|>final<|message|>say<|end|>");
  REQUIRE(parts.reasoning == "think");
  REQUIRE(parts.content == "say");
  // The role marker and channel-name control tokens never leak into either
  // output field.
  REQUIRE(parts.reasoning.find("<|") == std::string::npos);
  REQUIRE(parts.content.find("<|") == std::string::npos);
}

TEST_CASE("HarmonySplitter: control tokens never leak into visible text "
          "regardless of channel",
          "[harmony]") {
  auto parts = HarmonySplitter::Split("<|channel|>analysis<|message|>a<|end|>"
                                      "<|channel|>final<|message|>b<|return|>");
  for (const auto &tok :
       {"<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>"}) {
    REQUIRE(parts.reasoning.find(tok) == std::string::npos);
    REQUIRE(parts.content.find(tok) == std::string::npos);
  }
}

TEST_CASE("HarmonySplitter: per-token streaming matches one-shot oracle",
          "[harmony]") {
  const std::string output =
      "<|channel|>analysis<|message|>\nLet me work this out.\n<|end|>"
      "<|start|>assistant<|channel|>final<|message|>\n\n42.<|return|>";
  for (std::size_t n :
       {std::size_t(1), std::size_t(3), std::size_t(7), std::size_t(16)}) {
    auto parts = FeedByPieces(output, n);
    auto oracle = HarmonySplitter::Split(output);
    CAPTURE(n);
    REQUIRE(parts.reasoning == oracle.reasoning);
    REQUIRE(parts.content == oracle.content);
  }
}

TEST_CASE("HarmonySplitter::Drain yields incremental content deltas",
          "[harmony]") {
  HarmonySplitter splitter;
  splitter.Feed("<|channel|>final<|message|>hel");
  auto d1 = splitter.Drain();
  REQUIRE(d1.content == "hel");
  REQUIRE(d1.reasoning.empty());

  splitter.Feed("lo wor");
  auto d2 = splitter.Drain();
  REQUIRE(d2.content == "lo wor");

  splitter.Feed("ld<|return|>");
  auto d3 = splitter.Drain();
  REQUIRE(d3.content == "ld");

  auto tail = splitter.Finish();
  REQUIRE(tail.content == "hello world");
  REQUIRE(tail.reasoning.empty());
}

TEST_CASE("HarmonySplitter::Drain holds bytes that straddle a channel "
          "marker",
          "[harmony]") {
  HarmonySplitter splitter;
  splitter.Feed("<|channel|>analysis<|mess");
  auto d1 = splitter.Drain();
  REQUIRE(d1.reasoning.empty());
  REQUIRE(d1.content.empty());

  splitter.Feed("age|>secret<|e");
  auto d2 = splitter.Drain();
  REQUIRE(d2.reasoning == "secret");

  splitter.Feed("nd|>");
  auto d3 = splitter.Drain();
  REQUIRE(d3.reasoning.empty());
}

TEST_CASE("HarmonySplitter: in_reasoning reflects the current channel",
          "[harmony]") {
  HarmonySplitter splitter;
  REQUIRE_FALSE(splitter.in_reasoning());
  splitter.Feed("<|channel|>analysis<|message|>x");
  REQUIRE(splitter.in_reasoning());
  splitter.Feed("<|end|>");
  REQUIRE_FALSE(splitter.in_reasoning());
  splitter.Feed("<|channel|>final<|message|>y");
  REQUIRE_FALSE(splitter.in_reasoning());
}

TEST_CASE("HarmonySplitter: angle-bracket text is not mistaken for markers",
          "[harmony]") {
  const std::string output =
      "<|channel|>final<|message|>Use < compares and > compares. <notatag> "
      "stays.<|return|>";
  auto parts = HarmonySplitter::Split(output);
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "Use < compares and > compares. <notatag> stays.");
}
