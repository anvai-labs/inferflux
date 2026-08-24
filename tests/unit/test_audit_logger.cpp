#include <catch2/catch_amalgamated.hpp>

#include "server/logging/audit_logger.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>

using json = nlohmann::json;

TEST_CASE("AuditLogger disabled without path", "[audit]") {
  inferflux::AuditLogger logger;
  REQUIRE(!logger.Enabled());
  // Should be a no-op, not crash.
  logger.Log("user", "model", "ok", "hello");
}

TEST_CASE("AuditLogger writes valid JSON lines", "[audit]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_audit_test.jsonl";
  std::filesystem::remove(tmp_path);  // clean up any leftover from prior run
  {
    inferflux::AuditLogger logger(tmp_path.string());
    REQUIRE(logger.Enabled());
    logger.Log("alice", "llama-3", "success", "test message");
    logger.Log("bob", "gpt-4", "error", "something failed");
  }

  {
    std::ifstream in(tmp_path);
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
      auto j = json::parse(line);
      REQUIRE(j.contains("timestamp"));
      REQUIRE(j.contains("subject"));
      REQUIRE(j.contains("model"));
      REQUIRE(j.contains("status"));
      REQUIRE(j.contains("message"));
      count++;
    }
    REQUIRE(count == 2);
  }

  std::filesystem::remove(tmp_path);
}

TEST_CASE("AuditLogger LogRequest hashes prompt and response by default",
          "[audit]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_audit_hash.jsonl";
  {
    inferflux::AuditLogger logger(tmp_path.string()); // debug_mode=false
    logger.LogRequest("alice", "llama-3", "Hello world", "Hi there!", 2, 3);
  }
  {
    std::ifstream in(tmp_path);
    std::string line;
    REQUIRE(std::getline(in, line));
    auto j = json::parse(line);
    REQUIRE(j.contains("prompt_sha256"));
    REQUIRE(j.contains("response_sha256"));
    REQUIRE(!j.contains("prompt"));
    REQUIRE(!j.contains("response"));
    // SHA-256 is 64 hex chars.
    REQUIRE(j["prompt_sha256"].get<std::string>().size() == 64);
    REQUIRE(j["prompt_tokens"] == 2);
    REQUIRE(j["completion_tokens"] == 3);
  }
  std::filesystem::remove(tmp_path);
}

TEST_CASE("AuditLogger LogRequest writes raw text in debug mode", "[audit]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_audit_debug.jsonl";
  {
    inferflux::AuditLogger logger(tmp_path.string(), /*debug_mode=*/true);
    logger.LogRequest("alice", "llama-3", "Hello world", "Hi there!", 2, 3);
  }
  {
    std::ifstream in(tmp_path);
    std::string line;
    REQUIRE(std::getline(in, line));
    auto j = json::parse(line);
    REQUIRE(j["prompt"] == "Hello world");
    REQUIRE(j["response"] == "Hi there!");
    REQUIRE(!j.contains("prompt_sha256"));
  }
  std::filesystem::remove(tmp_path);
}

TEST_CASE("AuditLogger HashContent is stable SHA-256", "[audit]") {
  auto h1 = inferflux::AuditLogger::HashContent("hello");
  auto h2 = inferflux::AuditLogger::HashContent("hello");
  REQUIRE(h1 == h2);
  REQUIRE(h1.size() == 64);
  REQUIRE(h1 != inferflux::AuditLogger::HashContent("world"));
}

TEST_CASE("AuditLogger JSON-escapes special characters", "[audit]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_audit_escape.jsonl";
  {
    inferflux::AuditLogger logger(tmp_path.string());
    logger.Log("user\"evil", "model", "ok", "line\nnewline");
  }

  {
    std::ifstream in(tmp_path);
    std::string line;
    REQUIRE(std::getline(in, line));
    auto j = json::parse(line);
    REQUIRE(j["subject"] == "user\"evil");
    REQUIRE(j["message"] == "line\nnewline");
  }

  std::filesystem::remove(tmp_path);
}
