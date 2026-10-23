#pragma once

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <limits>
#include <string>

namespace inferflux {

/// Convert a string to lowercase (ASCII only).
inline std::string ToLower(const std::string &s) {
  std::string result = s;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

/// Trim leading and trailing whitespace.
inline std::string Trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

/// True for the C strings ParseBool/ParseBoolEnv accept as true ("true",
/// "1", "yes", "on"; ASCII case-insensitive). Allocation-free, for code that
/// must not allocate (e.g. inside an interposed pthread_create).
inline bool IsTruthyCStr(const char *val) noexcept {
  if (val == nullptr) {
    return false;
  }
  for (const char *yes : {"true", "1", "yes", "on"}) {
    const char *a = val;
    const char *b = yes;
    while (*a != '\0' && *b != '\0' &&
           std::tolower(static_cast<unsigned char>(*a)) == *b) {
      ++a;
      ++b;
    }
    if (*a == '\0' && *b == '\0') {
      return true;
    }
  }
  return false;
}

/// Parse a boolean from string.
/// Recognises "true"/"1"/"yes"/"on" (case-insensitive) as true,
/// "false"/"0"/"no"/"off" as false.  Any other value returns false.
inline bool ParseBool(const std::string &val) {
  std::string lower = ToLower(val);
  return lower == "true" || lower == "1" || lower == "yes" || lower == "on";
}

/// Parse a boolean from environment variable with fallback.
/// Recognises "true"/"1"/"yes"/"on" as true, "false"/"0"/"no"/"off" as false.
/// Unset or empty returns @p fallback.
inline bool ParseBoolEnv(const char *env_name, bool fallback) {
  const char *val = std::getenv(env_name);
  if (!val || *val == '\0')
    return fallback;
  if (IsTruthyCStr(val))
    return true;
  std::string lower = ToLower(val);
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off")
    return false;
  return fallback;
}

/// Parse an integer from environment variable with fallback and bounds.
/// Returns @p default_value if unset, empty, unparseable, or out of
/// [min_value, max_value].
inline int ParseIntEnv(const char *env_name, int default_value,
                       int min_value = 0,
                       int max_value = std::numeric_limits<int>::max()) {
  const char *raw = std::getenv(env_name);
  if (!raw || *raw == '\0')
    return default_value;
  char *end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end && *end != '\0'))
    return default_value;
  if (parsed < min_value || parsed > max_value)
    return default_value;
  return static_cast<int>(parsed);
}

// Common size constants
constexpr std::size_t kKiB = 1024;
constexpr std::size_t kMiB = 1024 * kKiB;
constexpr std::size_t kGiB = 1024 * kMiB;

} // namespace inferflux
