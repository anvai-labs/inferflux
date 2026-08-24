#include "server/logging/logger.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <chrono>
#include <iostream>
#include <mutex>

using json = nlohmann::json;

namespace inferflux {
namespace log {

namespace {

std::atomic<bool> g_json_mode{false};
std::atomic<int> g_min_level{static_cast<int>(Level::INFO)};
std::mutex g_mutex;

const char *LevelString(Level level) {
  switch (level) {
  case Level::DEBUG:
    return "DEBUG";
  case Level::INFO:
    return "INFO";
  case Level::WARN:
    return "WARN";
  case Level::ERROR:
    return "ERROR";
  }
  return "UNKNOWN";
}

std::string NormalizeLevelString(const std::string &text) {
  std::string lowered = text;
  for (auto &ch : lowered) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return lowered;
}

} // namespace

void SetJsonMode(bool enabled) { g_json_mode.store(enabled); }
bool IsJsonMode() { return g_json_mode.load(); }
void SetLevel(Level level) { g_min_level.store(static_cast<int>(level)); }
Level GetLevel() { return static_cast<Level>(g_min_level.load()); }

bool ParseLevel(const std::string &text, Level *level) {
  if (!level) {
    return false;
  }
  const std::string lowered = NormalizeLevelString(text);
  if (lowered == "debug") {
    *level = Level::DEBUG;
    return true;
  }
  if (lowered == "info") {
    *level = Level::INFO;
    return true;
  }
  if (lowered == "warn" || lowered == "warning") {
    *level = Level::WARN;
    return true;
  }
  if (lowered == "error") {
    *level = Level::ERROR;
    return true;
  }
  return false;
}

void Log(Level level, const std::string &component, const std::string &message,
         const std::string &extra) {
  if (static_cast<int>(level) < g_min_level.load()) {
    return;
  }
  auto now = std::chrono::system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();

  std::string line;
  if (g_json_mode.load()) {
    json j;
    j["ts"] = ts;
    j["level"] = LevelString(level);
    j["component"] = component;
    j["message"] = message;
    if (!extra.empty()) {
      j["extra"] = extra;
    }
    line = j.dump();
  } else {
    line = std::string("[") + LevelString(level) + "] " + component + ": " +
           message;
    if (!extra.empty()) {
      line += " | " + extra;
    }
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  std::cerr << line << "\n";
}

} // namespace log
} // namespace inferflux
