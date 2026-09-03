#pragma once

#include <cstdint>
#include <vector>

namespace inferflux {
namespace cuda {

// Identity fingerprint for the device-side decode relay.
//
// The relay lets the next decode call replay the captured CUDA graph without
// an H2D metadata upload: DeviceTokenRelay advances token/n_past in the
// device metadata in place, and BatchForwardReplay re-runs the graph against
// that mutated metadata. That is only valid when the next batch is THE SAME
// batch of sequences at the positions the relay advanced them to.
//
// Guarding on batch size alone (the historical behavior) lets a different
// batch with the same size inherit the previous batch's device metadata — a
// new sequence then decodes on the departed sequence's KV slot and emits its
// continuation (deterministic repro: tests/tools/batched_isolation_probe.cpp
// op W). The fingerprint pins row identity: sequence id, slot generation
// (slot reuse recycles sequence ids), and the n_past each row is expected at
// on the next call (relay advances it by one).
struct DecodeRelayFingerprint {
  std::vector<int> seq_ids;
  std::vector<uint64_t> generations;
  std::vector<int> next_n_past; // n_past each row will carry next call
  // The token each row was last fed. The relay kernel overwrites the device
  // metadata's token with the SAMPLED token and replay ignores host
  // batch_tokens entirely — so identity must include the token, or any
  // caller presenting a different token at the same (seq, generation,
  // n_past+1) would silently decode the stale device token.
  std::vector<int> tokens;

  bool empty() const { return seq_ids.empty(); }
  size_t size() const { return seq_ids.size(); }
};

// True when the next decode batch is exactly the batch the relay is armed
// for: `count` rows, same sequence ids, same slot generations, and each row at
// the n_past the relay advanced it to (armed value = current n_past + 1).
// The input vectors may be longer than `count` (callers reuse preallocated
// max-capacity buffers); only the first `count` entries participate.
inline bool DecodeRelayIdentityMatches(const DecodeRelayFingerprint &armed,
                                       const std::vector<int> &seq_ids,
                                       const std::vector<uint64_t> &generations,
                                       const std::vector<int> &n_past,
                                       const std::vector<int> &tokens,
                                       size_t count) {
  if (armed.size() != count || seq_ids.size() < count ||
      generations.size() < count || n_past.size() < count) {
    return false;
  }
  // Empty `tokens` = caller cannot supply them (skip the check, S20-era
  // behavior); non-empty = full identity including the fed token.
  const bool check_tokens = !tokens.empty();
  if (check_tokens && (tokens.size() < count || armed.tokens.size() < count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (armed.seq_ids[i] != seq_ids[i] ||
        armed.generations[i] != generations[i] ||
        armed.next_n_past[i] != n_past[i] + 1 ||
        (check_tokens && armed.tokens[i] != tokens[i])) {
      return false;
    }
  }
  return true;
}

// Backward-compatible wrapper: identity without token verification.
inline bool DecodeRelayIdentityMatches(const DecodeRelayFingerprint &armed,
                                       const std::vector<int> &seq_ids,
                                       const std::vector<uint64_t> &generations,
                                       const std::vector<int> &n_past,
                                       size_t count) {
  return DecodeRelayIdentityMatches(armed, seq_ids, generations, n_past,
                                    /*tokens=*/{}, count);
}

} // namespace cuda
} // namespace inferflux
