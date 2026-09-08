#pragma once

#include "runtime/backends/gpu/backend_config_extensions.h"

#include <cstdint>
#include <string>

namespace inferflux {

struct LlamaBackendConfig {
  int32_t ctx_size = 2048;
  int32_t batch_size = 512;
  int gpu_layers = 0;
  bool use_flash_attention = false;
  int flash_attention_tile = 128;
  // CUDA attention kernel policy:
  // auto | fa3 | fa2 | standard
  std::string cuda_attention_kernel{"auto"};
  // CUDA phase-overlap scaffold (foundation for native async overlap).
  // When enabled, CUDA backend can split mixed unified batches into decode-
  // first and prefill lanes to reduce decode head-of-line blocking.
  bool cuda_phase_overlap_scaffold{false};
  // Minimum count of prefill tokens in a mixed batch before split kicks in.
  int cuda_phase_overlap_min_prefill_tokens{256};
  // Optional dual-context overlap mode: run prefill lane on a separate CUDA
  // context and hand off KV to decode lane via
  // SerializeSequence/HydrateSequence. Disabled by default because it increases
  // memory footprint.
  bool cuda_phase_overlap_prefill_replica{false};
  // KV cache precision policy for InferFlux CUDA runtime:
  // auto | fp16 | bf16 | int8 | fp8
  // `auto` keeps current behavior (match inference dtype).
  std::string inferflux_cuda_kv_cache_dtype{"auto"};
  // Dequantized GGUF weight cache lifecycle in InferFlux CUDA runtime:
  // none (no caching) | batch (batch-boundary cleanup) |
  // model (persist for model lifetime; highest VRAM use).
  std::string inferflux_cuda_dequantized_cache_policy{"none"};
  // When true, quantized GGUF model-load fails unless fused dequant-tile GEMM
  // strategy is selected for this GPU/runtime capability set.
  bool inferflux_cuda_require_fused_quantized_matmul{false};
  std::string
      mmproj_path; // Path to multimodal projector; empty = vision disabled.
  // Widest single prefill call the scheduler will issue (tokens). Mirrors
  // the scheduler's chunked_prefill_tokens so the native forward can size
  // activation scratch to the actual working set instead of the context
  // window.
  int32_t prefill_chunk_tokens{512};
  // Maximum number of KV-cache sequences that can be live simultaneously.
  // Increased from 16 to 128 for production concurrent workloads.
  // Managed by SequenceSlotManager for timeout-based eviction.
  // NOTE: this value sizes the llama.cpp KV pool for llama.cpp-backed
  // backends (n_seq_max) — 128 reserves substantial VRAM; right-size it to
  // the intended admission concurrency when tuning memory.
  int max_parallel_sequences{128};
  // KV-cache element type for llama.cpp-backed backends: f16 (default,
  // llama.cpp stock) | q8_0 | q4_0. Applies to both K and V caches.
  std::string llama_kv_cache_type{"f16"};

  // Distributed Parallelism Degrees.
  int tp_degree{1}; // Tensor Parallel degree
  int pp_degree{1}; // Pipeline Parallel degree

  // Structured CUDA config extension. Populated by TuneLlamaBackendConfig
  // when target is kCuda. Mirrors the flat cuda_* fields above for
  // forward compatibility; new code should prefer this over flat fields.
  CudaConfigExtension cuda_ext;
};

// Future-facing alias.
using BackendConfig = LlamaBackendConfig;

} // namespace inferflux
