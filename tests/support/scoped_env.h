#pragma once

/// @file scoped_env.h
/// @brief RAII environment variable helper for test isolation.
///
/// Consolidates 5 duplicate ScopedEnvVar implementations scattered across
/// test files.  Include this from any test that needs to temporarily set
/// environment variables.

#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <cstdlib> // _putenv_s
#endif

namespace inferflux {
namespace test {

/// RAII guard that sets an environment variable for the current scope
/// and restores the original value (or unsets it) on destruction.
class ScopedEnvVar {
public:
  ScopedEnvVar(std::string name, std::string value) : name_(std::move(name)) {
    SaveOriginal();
    Set(value);
  }

  // A null value means "temporarily unset", matching ScopedEnvVar call sites.
  ScopedEnvVar(std::string name, const char *value) : name_(std::move(name)) {
    SaveOriginal();
    if (value) {
      Set(value);
    } else {
      Unset();
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      Set(original_value_);
    } else {
      Unset();
    }
  }

  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  void SaveOriginal() {
    const char *existing = std::getenv(name_.c_str());
    if (existing != nullptr) {
      had_original_ = true;
      original_value_ = existing;
    }
  }

  void Set(const std::string &value) {
#ifdef _WIN32
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }

  void Unset() {
#ifdef _WIN32
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  std::string name_;
  std::string original_value_;
  bool had_original_{false};
};

/// Portable setenv/unsetenv for tests that don't use the RAII pattern.
inline void portable_setenv(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline void portable_unsetenv(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

} // namespace test
} // namespace inferflux
