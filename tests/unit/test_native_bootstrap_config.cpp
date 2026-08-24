#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/native_bootstrap_config.h"

#include <cstdlib>
#include <string>

namespace inferflux {
namespace {

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

class ScopedEnvVar {
public:
  ScopedEnvVar(std::string name, const char *value) : name_(std::move(name)) {
    const char *existing = std::getenv(name_.c_str());
    if (existing) {
      had_original_ = true;
      original_ = existing;
    }
    if (value) {
      portable_setenv(name_.c_str(), value);
    } else {
      portable_unsetenv(name_.c_str());
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      portable_setenv(name_.c_str(), original_.c_str());
    } else {
      portable_unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::string original_;
  bool had_original_{false};
};

} // namespace

TEST_CASE("NativeBootstrapConfig: uses hinted KV precision and default sizing",
          "[native_bootstrap]") {
  ScopedEnvVar dtype("INFERFLUX_CUDA_DTYPE", nullptr);
  ScopedEnvVar kv_dtype("INFERFLUX_CUDA_KV_DTYPE", nullptr);
  ScopedEnvVar kv_batch("INFERFLUX_CUDA_KV_MAX_BATCH", nullptr);
  ScopedEnvVar kv_seq("INFERFLUX_CUDA_KV_MAX_SEQ", nullptr);
  ScopedEnvVar kv_auto("INFERFLUX_CUDA_KV_AUTO_TUNE", nullptr);
  ScopedEnvVar kv_budget("INFERFLUX_CUDA_KV_BUDGET_MB", nullptr);
  ScopedEnvVar kv_ratio("INFERFLUX_CUDA_KV_FREE_MEM_RATIO", nullptr);

  const auto config = NativeBootstrapConfig::FromEnv("bf16");
  REQUIRE(config.dtype_override.empty());
  REQUIRE(config.kv_precision_choice == "bf16");
  REQUIRE(config.kv_max_batch == 32);
  REQUIRE(config.kv_max_seq == 2048);
  REQUIRE_FALSE(config.kv_max_seq_overridden);
  REQUIRE(config.kv_auto_tune);
  REQUIRE(config.kv_budget_bytes == 0);
  REQUIRE(config.kv_budget_ratio == Catch::Approx(0.30));
}

TEST_CASE("NativeBootstrapConfig: records valid overrides and invalid raw values",
          "[native_bootstrap]") {
  ScopedEnvVar dtype("INFERFLUX_CUDA_DTYPE", "FP16");
  ScopedEnvVar kv_dtype("INFERFLUX_CUDA_KV_DTYPE", "bf16");
  ScopedEnvVar kv_batch("INFERFLUX_CUDA_KV_MAX_BATCH", "bad");
  ScopedEnvVar kv_seq("INFERFLUX_CUDA_KV_MAX_SEQ", "8192");
  ScopedEnvVar kv_auto("INFERFLUX_CUDA_KV_AUTO_TUNE", "0");
  ScopedEnvVar kv_budget("INFERFLUX_CUDA_KV_BUDGET_MB", "256");
  ScopedEnvVar kv_ratio("INFERFLUX_CUDA_KV_FREE_MEM_RATIO", "bad");

  const auto config = NativeBootstrapConfig::FromEnv("auto");
  REQUIRE(config.dtype_override == "fp16");
  REQUIRE(config.ForceFp16());
  REQUIRE(config.kv_precision_choice == "bf16");
  REQUIRE(config.kv_max_batch == 32);
  REQUIRE(config.invalid_kv_max_batch == "bad");
  REQUIRE(config.kv_max_seq == 8192);
  REQUIRE(config.kv_max_seq_overridden);
  REQUIRE_FALSE(config.kv_auto_tune);
  REQUIRE(config.kv_budget_bytes == static_cast<std::size_t>(256) * 1024U *
                                         1024U);
  REQUIRE(config.kv_budget_ratio == Catch::Approx(0.30));
  REQUIRE(config.invalid_kv_free_mem_ratio == "bad");
}

} // namespace inferflux
