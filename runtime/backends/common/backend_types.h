#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "scheduler/request_batch.h" // For SamplingParams

namespace inferflux {

// ============================================================================
// Unified Batch Types
// ============================================================================

/// Input for one sequence in a unified batch execution.
/// A single ExecuteUnifiedBatch() call can mix prefill (multiple tokens,
/// n_past=0) and decode (one token, n_past>0) sequences in the same forward
/// pass.
struct UnifiedBatchInput {
  int sequence_id{0};
  int n_past{0};
  std::vector<int> tokens;
  bool request_logits{true};
  SamplingParams sampling; // Per-request sampling parameters
  int64_t request_id{-1};
  std::string client_request_id;
  uint64_t sequence_generation{0};
};

/// Output for one sequence in a unified batch execution.
struct UnifiedBatchOutput {
  int token{-1};     // Next sampled token; -1 = EOS or error
  std::string piece; // Text of token; empty when token == -1
  bool ok{false};    // true if token was successfully sampled
};

// ============================================================================
// Decode Burst Types
// ============================================================================

/// Tuning knobs for a burst-pipelined decode call. The executor enqueues up to
/// max_tokens_per_seq decode steps per sequence back-to-back on the device
/// (forward + sample + on-device token feed) and streams tokens to the sink
/// from a pinned ring while the GPU runs ahead. Stop decisions made by the
/// sink take effect at chunk boundaries; at most max_tokens_per_seq extra
/// tokens are computed past a stop.
struct UnifiedBurstOptions {
  int max_tokens_per_seq{8}; ///< Steps enqueued per sequence per chunk
  int max_batch_tokens{256}; ///< Cap on chunk_tokens * batch_size
  std::chrono::milliseconds max_wall_ms{40}; ///< Wall-clock cap per burst call
};

/// One sampled token surfaced by a burst. input_idx maps back to the position
/// in the ExecuteUnifiedBatchBurst() input vector. token_id is the sampled
/// token; piece is its detokenized text (may be empty for non-emitting
/// tokens). The sink returns false to stop that sequence (stop string, decode
/// limit, cancellation); emission-side stops freeze the sequence at the next
/// chunk boundary.
struct BurstTokenEvent {
  std::size_t input_idx{0};
  int token_id{-1};
  std::string_view piece;
};

using BurstTokenSink = std::function<bool(const BurstTokenEvent &)>;

/// Result of a burst call. last_tokens[b] is the most recent token sampled for
/// input b (the token that feeds the next decode step); finished[b] is true
/// when the sequence ended inside the burst (device EOS or sink stop). ok is
/// false only on hard execution errors, in which case no tokens were emitted.
struct UnifiedBurstResult {
  bool ok{false};
  std::vector<int> last_tokens;
  std::vector<bool> finished;
};

/// Execution lane hint for async unified-batch submission.
/// kDecode should be favored for lower token latency.
enum class UnifiedBatchLane {
  kAuto,    // Let backend decide (default)
  kDecode,  // Decode lane (lower latency priority)
  kPrefill, // Prefill lane (higher throughput priority)
};

/// Handle for tracking async batch submissions.
/// Used with SubmitUnifiedBatchAsync/TryCollectUnifiedBatchAsync.
using UnifiedBatchHandle = uint64_t;

/// Result of a phased prefill pass (for backends that support phased
/// execution).
struct PrefillResult {
  int n_past{0};           ///< KV position after prompt evaluation
  bool ok{false};          ///< true on success
  int first_token{-1};     ///< First output token sampled from prefill logits
  std::string first_piece; ///< Text of first_token
};

} // namespace inferflux
