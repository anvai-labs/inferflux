#include "runtime/backends/cuda/native/native_bootstrap_config.h"

#include "runtime/string_utils.h"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace inferflux {

namespace {

bool ParsePositiveIntSetting(const char *raw, int *out) {
  if (!raw || !out) {
    return false;
  }
  char *end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

bool ParsePositiveDoubleSetting(const char *raw, double *out) {
  if (!raw || !out) {
    return false;
  }
  char *end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) {
    return false;
  }
  *out = parsed;
  return true;
}

bool ParseBoolSetting(const char *raw, bool fallback) {
  if (!raw) {
    return fallback;
  }
  return inferflux::ParseBool(raw);
}

} // namespace

NativeBootstrapConfig
NativeBootstrapConfig::FromEnv(const std::string &kv_precision_hint) {
  NativeBootstrapConfig config;

  if (const char *dtype_override = std::getenv("INFERFLUX_CUDA_DTYPE")) {
    config.dtype_override = inferflux::ToLower(dtype_override);
  }

  config.kv_precision_choice = inferflux::ToLower(kv_precision_hint);
  if (config.kv_precision_choice.empty()) {
    config.kv_precision_choice = "auto";
  }
  if (const char *env_kv_precision = std::getenv("INFERFLUX_CUDA_KV_DTYPE")) {
    config.kv_precision_choice = inferflux::ToLower(env_kv_precision);
    if (config.kv_precision_choice.empty()) {
      config.kv_precision_choice = "auto";
    }
  }

  if (const char *env = std::getenv("INFERFLUX_CUDA_KV_MAX_BATCH")) {
    int val = 0;
    if (ParsePositiveIntSetting(env, &val) && val <= 128) {
      config.kv_max_batch = val;
    } else {
      config.invalid_kv_max_batch = env;
    }
  }

  if (const char *env = std::getenv("INFERFLUX_CUDA_KV_MAX_SEQ")) {
    int val = 0;
    if (ParsePositiveIntSetting(env, &val) && val <= 131072) {
      config.kv_max_seq = val;
      config.kv_max_seq_overridden = true;
    } else {
      config.invalid_kv_max_seq = env;
    }
  }

  if (const char *env = std::getenv("INFERFLUX_CUDA_KV_AUTO_TUNE")) {
    config.kv_auto_tune = ParseBoolSetting(env, true);
  }

  if (const char *env = std::getenv("INFERFLUX_CUDA_KV_BUDGET_MB")) {
    int budget_mb = 0;
    if (ParsePositiveIntSetting(env, &budget_mb)) {
      config.kv_budget_bytes =
          static_cast<std::size_t>(budget_mb) * 1024U * 1024U;
    } else {
      config.invalid_kv_budget_mb = env;
    }
  }

  if (const char *env = std::getenv("INFERFLUX_CUDA_KV_FREE_MEM_RATIO")) {
    double parsed = 0.0;
    if (ParsePositiveDoubleSetting(env, &parsed) && parsed <= 1.0) {
      config.kv_budget_ratio = parsed;
    } else {
      config.invalid_kv_free_mem_ratio = env;
    }
  }

  return config;
}

} // namespace inferflux
