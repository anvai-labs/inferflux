#pragma once

#include <string>

// Windows headers define ERROR as 0 — undefine to avoid collision.
#ifdef ERROR
#undef ERROR
#endif

namespace inferflux {
namespace log {

enum class Level { DEBUG, INFO, WARN, ERROR };

// Enable JSON-structured output (one JSON object per line to stderr).
// Default mode is plain text: "[LEVEL] component: message".
// Call from main() based on INFERFLUX_LOG_FORMAT=json before any logging.
void SetJsonMode(bool enabled);
bool IsJsonMode();
void SetLevel(Level level);
Level GetLevel();
bool ParseLevel(const std::string &text, Level *level);

// Emit a log entry at the given level.  `component` identifies the subsystem
// (e.g. "server", "scheduler", "backend").  `extra` is an optional key=value
// string appended to the JSON object or the text line (ignored when empty).
void Log(Level level, const std::string &component, const std::string &message,
         const std::string &extra = {});

// Convenience wrappers.
inline void Debug(const std::string &component, const std::string &message,
                  const std::string &extra = {}) {
  Log(Level::DEBUG, component, message, extra);
}
inline void Info(const std::string &component, const std::string &message,
                 const std::string &extra = {}) {
  Log(Level::INFO, component, message, extra);
}
inline void Warn(const std::string &component, const std::string &message,
                 const std::string &extra = {}) {
  Log(Level::WARN, component, message, extra);
}
inline void Error(const std::string &component, const std::string &message,
                  const std::string &extra = {}) {
  Log(Level::ERROR, component, message, extra);
}

} // namespace log
} // namespace inferflux
