#pragma once

#include <string>

namespace inferflux {

struct CudaConfigExtension {
  std::string attention_kernel{"auto"};
  bool phase_overlap_scaffold{false};
  int phase_overlap_min_prefill_tokens{256};
  bool phase_overlap_prefill_replica{false};
  std::string kv_cache_dtype{"auto"};
  std::string dequantized_cache_policy{"none"};
  bool require_fused_quantized_matmul{false};
};

} // namespace inferflux
