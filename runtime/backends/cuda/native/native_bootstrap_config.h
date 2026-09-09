#pragma once

#include <cstddef>
#include <string>

namespace inferflux {

struct NativeBootstrapConfig {
  std::string dtype_override;
  std::string kv_precision_choice{"auto"};
  int kv_max_batch{16};
  // Per-slot context default: 1,024 tokens x 16 slots = 604 MB KV (was
  // 2,048 x 16 = 1,208 MB). Longer prompts reject at admission; restore
  // via INFERFLUX_CUDA_KV_MAX_SEQ (user decision, Sep 8 memory campaign).
  int kv_max_seq{1024};
  bool kv_max_seq_overridden{false};
  bool kv_max_batch_overridden{false};
  bool kv_auto_tune{true};
  std::size_t kv_budget_bytes{0};
  double kv_budget_ratio{0.20};
  std::string invalid_kv_max_batch;
  std::string invalid_kv_max_seq;
  std::string invalid_kv_budget_mb;
  int kv_base_slots{0}; // 0 = all dense (backward compat), >0 = hybrid
  std::string invalid_kv_base_slots;
  std::string invalid_kv_free_mem_ratio;

  static NativeBootstrapConfig
  FromEnv(const std::string &kv_precision_hint = "auto");

  bool ForceFp16() const { return dtype_override == "fp16"; }
  bool ForceBf16() const { return dtype_override == "bf16"; }
};

} // namespace inferflux
