#include <catch2/catch_amalgamated.hpp>

#include "runtime/text/incremental_utf8.h"

using inferflux::IncrementalUtf8Assembler;

TEST_CASE("Incremental UTF-8 assembler buffers split code points", "[utf8]") {
  SECTION("two-byte sequence") {
    IncrementalUtf8Assembler assembler;
    REQUIRE(assembler.Append("\xC2").text.empty());
    REQUIRE(assembler.Append("\xA2").text == "\xC2\xA2");
    REQUIRE(assembler.Finish().text.empty());
  }

  SECTION("three-byte sequence") {
    IncrementalUtf8Assembler assembler;
    REQUIRE(assembler.Append("\xE2").text.empty());
    REQUIRE(assembler.Append("\x82").text.empty());
    REQUIRE(assembler.Append("\xAC").text == "\xE2\x82\xAC");
  }

  SECTION("four-byte sequence") {
    IncrementalUtf8Assembler assembler;
    REQUIRE(assembler.Append("\xF0").text.empty());
    REQUIRE(assembler.Append("\x9F\x92").text.empty());
    REQUIRE(assembler.Append("\xA9").text == "\xF0\x9F\x92\xA9");
  }
}

TEST_CASE("Incremental UTF-8 assembler replaces malformed input", "[utf8]") {
  IncrementalUtf8Assembler assembler;
  const auto result = assembler.Append("\x9B");
  REQUIRE(result.text == "\xEF\xBF\xBD");
  REQUIRE(result.replacements == 1);
}

TEST_CASE("Incremental UTF-8 assembler replaces incomplete terminal input",
          "[utf8]") {
  IncrementalUtf8Assembler assembler;
  REQUIRE(assembler.Append("ok\xE2\x82").text == "ok");
  REQUIRE(assembler.HasPendingBytes());

  const auto result = assembler.Finish();
  REQUIRE(result.text == "\xEF\xBF\xBD");
  REQUIRE(result.replacements == 1);
  REQUIRE_FALSE(assembler.HasPendingBytes());
}

TEST_CASE("Incremental UTF-8 assembler rejects invalid scalar encodings",
          "[utf8]") {
  IncrementalUtf8Assembler assembler;
  const auto result = assembler.Append("\xE0\x80\x80\xED\xA0\x80"
                                       "\xF4\x90\x80\x80");
  REQUIRE(result.replacements > 0);
  REQUIRE_FALSE(result.text.empty());
  REQUIRE_FALSE(assembler.HasPendingBytes());
}
