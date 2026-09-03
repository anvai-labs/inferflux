#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace inferflux {

/// Case-insensitive HTTP header value lookup (RFC 9110).
/// Scans raw header block for @p name and returns the trimmed value.
/// Returns empty string if the header is not found.
inline std::string GetHeaderValue(const std::string &headers,
                                  const std::string &name) {
  auto lower_name = name;
  std::transform(
      lower_name.begin(), lower_name.end(), lower_name.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  std::size_t line_start = 0;
  while (line_start < headers.size()) {
    const std::size_t line_end = headers.find("\r\n", line_start);
    const std::size_t current_end =
        line_end == std::string::npos ? headers.size() : line_end;
    const std::size_t colon = headers.find(':', line_start);
    if (colon != std::string::npos && colon < current_end) {
      std::string header_name = headers.substr(line_start, colon - line_start);
      std::transform(
          header_name.begin(), header_name.end(), header_name.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (header_name == lower_name) {
        std::string value =
            headers.substr(colon + 1, current_end - (colon + 1));
        const auto s = value.find_first_not_of(" \t");
        const auto e = value.find_last_not_of(" \t\r\n");
        return (s == std::string::npos) ? "" : value.substr(s, e - s + 1);
      }
    }
    if (line_end == std::string::npos) {
      break;
    }
    line_start = line_end + 2;
  }
  return {};
}

/// Extracts the bearer token from a raw header block. Field names are
/// case-insensitive (RFC 9110 §5.1) and so is the auth-scheme (§11.1), so
/// "AUTHORIZATION: bEaReR <key>" parses the same as the canonical spelling.
/// Returns empty string when no bearer credentials are present.
inline std::string ExtractBearerToken(const std::string &headers) {
  const std::string value = GetHeaderValue(headers, "authorization");
  const std::size_t scheme_start = value.find_first_not_of(" \t");
  if (scheme_start == std::string::npos) {
    return {};
  }
  const std::size_t scheme_end = value.find_first_of(" \t", scheme_start);
  if (scheme_end == std::string::npos) {
    return {}; // Scheme without credentials.
  }
  std::string scheme = value.substr(scheme_start, scheme_end - scheme_start);
  std::transform(
      scheme.begin(), scheme.end(), scheme.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (scheme != "bearer") {
    return {};
  }
  const std::size_t token_start = value.find_first_not_of(" \t", scheme_end);
  if (token_start == std::string::npos) {
    return {};
  }
  return value.substr(token_start);
}

} // namespace inferflux
