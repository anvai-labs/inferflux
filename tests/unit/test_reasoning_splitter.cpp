#include <catch2/catch_amalgamated.hpp>

#include "runtime/text/reasoning_splitter.h"

using inferflux::ReasoningSplitter;

namespace {

// Feed the string in per-token-sized pieces (simulates streaming).
ReasoningSplitter::Parts FeedByPieces(const std::string &output,
                                      std::size_t n) {
  ReasoningSplitter splitter;
  for (std::size_t i = 0; i < output.size(); i += n) {
    splitter.Feed(output.substr(i, n));
  }
  return splitter.Finish();
}

} // namespace

TEST_CASE("ReasoningSplitter: plain output is all content", "[reasoning]") {
  auto parts = ReasoningSplitter::Split("Just a plain answer.");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "Just a plain answer.");
}

TEST_CASE("ReasoningSplitter: think block moves to reasoning", "[reasoning]") {
  auto parts =
      ReasoningSplitter::Split("<think>step by step</think>The answer is 4.");
  REQUIRE(parts.reasoning == "step by step");
  REQUIRE(parts.content == "The answer is 4.");
}

TEST_CASE("ReasoningSplitter: empty think block", "[reasoning]") {
  auto parts = ReasoningSplitter::Split("<think></think>Answer.");
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == "Answer.");
}

TEST_CASE("ReasoningSplitter: unterminated think block flushes to reasoning",
          "[reasoning]") {
  auto parts = ReasoningSplitter::Split("<think>cut off mid thought");
  REQUIRE(parts.content.empty());
  REQUIRE(parts.reasoning == "cut off mid thought");
}

TEST_CASE("ReasoningSplitter: split across piece boundaries (streaming)",
          "[reasoning]") {
  ReasoningSplitter splitter;
  std::string reasoning, content;
  for (char c : std::string("<thi")) {
    splitter.Feed(std::string(1, c));
  }
  splitter.Feed("nk>reason");
  for (char c : std::string("</thin")) {
    splitter.Feed(std::string(1, c));
  }
  splitter.Feed("k>answer text");
  auto parts = splitter.Finish();
  // Byte-level equality with the one-shot oracle.
  auto oracle = ReasoningSplitter::Split("<think>reason</think>answer text");
  REQUIRE(parts.reasoning == oracle.reasoning);
  REQUIRE(parts.content == oracle.content);
  (void)reasoning;
  (void)content;
}

TEST_CASE("ReasoningSplitter: per-token streaming matches one-shot oracle",
          "[reasoning]") {
  const std::string output = "<think>\nLet me work this out.\n</think>\n\n42.";
  for (std::size_t n :
       {std::size_t(1), std::size_t(3), std::size_t(7), std::size_t(16)}) {
    auto parts = FeedByPieces(output, n);
    auto oracle = ReasoningSplitter::Split(output);
    CAPTURE(n);
    REQUIRE(parts.reasoning == oracle.reasoning);
    REQUIRE(parts.content == oracle.content);
  }
}

TEST_CASE("ReasoningSplitter: angle-bracket text is not mistaken for tags",
          "[reasoning]") {
  const std::string output = "Use < compares and > compares. <notatag> stays.";
  auto parts = ReasoningSplitter::Split(output);
  REQUIRE(parts.reasoning.empty());
  REQUIRE(parts.content == output);
}

TEST_CASE("ReasoningSplitter::Drain yields incremental deltas", "[reasoning]") {
  ReasoningSplitter splitter;
  splitter.Feed("<think>rea");
  auto d1 = splitter.Drain();
  REQUIRE(d1.reasoning == "rea");
  REQUIRE(d1.content.empty());

  splitter.Feed("soning</thi");
  auto d2 = splitter.Drain();
  // "soning" is final; "</thi" may still be a partial close tag.
  REQUIRE(d2.reasoning == "soning");

  splitter.Feed("nk>answer");
  auto d3 = splitter.Drain();
  REQUIRE(d3.reasoning.empty());
  REQUIRE(d3.content == "answer");

  auto tail = splitter.Finish();
  REQUIRE(tail.reasoning == "reasoning");
  REQUIRE(tail.content == "answer");
}

TEST_CASE("ReasoningSplitter::Drain holds bytes that straddle the open tag",
          "[reasoning]") {
  ReasoningSplitter splitter;
  splitter.Feed("hello <th");
  auto d1 = splitter.Drain();
  REQUIRE(d1.content == "hello ");
  REQUIRE(d1.reasoning.empty());

  splitter.Feed("ink>secret");
  auto d2 = splitter.Drain();
  REQUIRE(d2.content.empty());
  REQUIRE(d2.reasoning == "secret");
}

TEST_CASE("ReasoningSplitter::Drain splits a boundary-straddling chunk",
          "[reasoning]") {
  ReasoningSplitter splitter;
  splitter.Feed("abc</thi");
  auto d1 = splitter.Drain();
  REQUIRE(d1.content == "abc");

  splitter.Feed("nk>xyz");
  auto d2 = splitter.Drain();
  // The close tag completed: nothing new is reasoning, "xyz" is content.
  REQUIRE(d2.reasoning.empty());
  REQUIRE(d2.content == "xyz");
}
