#pragma once

#include <cstddef>
#include <string>

namespace inferflux {

struct NativeBootstrapConfig {
  std::string dtype_override;
  std::string kv_precision_choice{"auto"};
  int kv_max_batch{32};
  int kv_max_seq{2048};
  bool kv_max_seq_overridden{false};
  bool kv_auto_tune{true};
  std::size_t kv_budget_bytes{0};
  double kv_budget_ratio{0.30};
  std::string invalid_kv_max_batch;
  std::string invalid_kv_max_seq;
  std::string invalid_kv_budget_mb;
  std::string invalid_kv_free_mem_ratio;

  static NativeBootstrapConfig
  FromEnv(const std::string &kv_precision_hint = "auto");

  bool ForceFp16() const { return dtype_override == "fp16"; }
  bool ForceBf16() const { return dtype_override == "bf16"; }
};

} // namespace inferflux
