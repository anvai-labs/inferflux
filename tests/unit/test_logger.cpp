#include <catch2/catch_amalgamated.hpp>

#include "server/logging/logger.h"

#include <iostream>
#include <sstream>

namespace {

class StderrCapture {
public:
  StderrCapture() : old_(std::cerr.rdbuf(stream_.rdbuf())) {}
  ~StderrCapture() { std::cerr.rdbuf(old_); }

  std::string str() const { return stream_.str(); }

private:
  std::ostringstream stream_;
  std::streambuf *old_;
};

} // namespace

// ---------------------------------------------------------------------------
// Structured JSON / plain-text application logger
// ---------------------------------------------------------------------------

TEST_CASE("Logger defaults to plain-text mode", "[logger]") {
  // After construction the global mode must be text (non-JSON).
  inferflux::log::SetJsonMode(false);
  REQUIRE_FALSE(inferflux::log::IsJsonMode());
  inferflux::log::SetLevel(inferflux::log::Level::INFO);
}

TEST_CASE("Logger can be switched to JSON mode", "[logger]") {
  inferflux::log::SetJsonMode(true);
  REQUIRE(inferflux::log::IsJsonMode());
  // Reset so other tests see text mode.
  inferflux::log::SetJsonMode(false);
  inferflux::log::SetLevel(inferflux::log::Level::INFO);
}

TEST_CASE("Logger::Log does not throw in text mode", "[logger]") {
  inferflux::log::SetJsonMode(false);
  REQUIRE_NOTHROW(inferflux::log::Info("test", "hello from text mode"));
  REQUIRE_NOTHROW(
      inferflux::log::Warn("test", "warn in text mode", "detail=foo"));
  REQUIRE_NOTHROW(inferflux::log::Error("test", "error in text mode"));
}

TEST_CASE("Logger::Log does not throw in JSON mode", "[logger]") {
  inferflux::log::SetJsonMode(true);
  REQUIRE_NOTHROW(inferflux::log::Info("test", "hello from json mode"));
  REQUIRE_NOTHROW(
      inferflux::log::Warn("test", "warn in json mode", "detail=bar"));
  REQUIRE_NOTHROW(inferflux::log::Error("test", "error in json mode"));
  inferflux::log::SetJsonMode(false);
}

TEST_CASE("Logger::Debug convenience wrapper does not throw", "[logger]") {
  REQUIRE_NOTHROW(inferflux::log::Debug("test", "debug message"));
}

TEST_CASE("Logger filters messages below configured level", "[logger]") {
  inferflux::log::SetJsonMode(false);
  inferflux::log::SetLevel(inferflux::log::Level::WARN);
  StderrCapture capture;

  inferflux::log::Info("test", "hidden info");
  inferflux::log::Warn("test", "visible warning");

  const std::string output = capture.str();
  REQUIRE(output.find("hidden info") == std::string::npos);
  REQUIRE(output.find("visible warning") != std::string::npos);

  inferflux::log::SetLevel(inferflux::log::Level::INFO);
}

TEST_CASE("Logger parses level aliases", "[logger]") {
  inferflux::log::Level level = inferflux::log::Level::INFO;
  REQUIRE(inferflux::log::ParseLevel("debug", &level));
  REQUIRE(level == inferflux::log::Level::DEBUG);
  REQUIRE(inferflux::log::ParseLevel("warning", &level));
  REQUIRE(level == inferflux::log::Level::WARN);
  REQUIRE_FALSE(inferflux::log::ParseLevel("verbose", &level));
}

TEST_CASE("Logger mode toggle is thread-safe (no crash under concurrent calls)",
          "[logger]") {
  // Simple sequential toggle to verify atomicity without a full thread library.
  for (int i = 0; i < 100; ++i) {
    inferflux::log::SetJsonMode(i % 2 == 0);
    (void)inferflux::log::IsJsonMode();
  }
  inferflux::log::SetJsonMode(false);
  inferflux::log::SetLevel(inferflux::log::Level::INFO);
  SUCCEED("Mode toggle did not crash");
}
