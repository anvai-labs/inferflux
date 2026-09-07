#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/decode_relay_fingerprint.h"

using namespace inferflux::cuda;

namespace {

DecodeRelayFingerprint Armed(std::vector<int> seq_ids,
                             std::vector<uint64_t> generations,
                             std::vector<int> next_n_past,
                             std::vector<int> tokens = {}) {
  DecodeRelayFingerprint f;
  f.seq_ids = std::move(seq_ids);
  f.generations = std::move(generations);
  f.next_n_past = std::move(next_n_past);
  f.tokens = std::move(tokens);
  return f;
}

} // namespace

// Semantics (matching DeviceTokenRelayKernel): after the arming step the
// relay kernel writes token_ids[b] = sampled_tokens[b] and n_past[b] += 1
// into device metadata, and the host advances its own n_past record by 1.
// The armed fingerprint therefore stores the sampled tokens and the
// incremented n_past, and the check compares the next step's presented
// (fed-token, host-n_past) directly against them -- equality, no offset.

TEST_CASE("Decode relay fingerprint matches identical batch",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Relay advanced device n_past to 30/31; the host's own record also
  // advanced to 30/31 for the next tick -- direct equality must match.
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 31}, 2));
}

TEST_CASE("Decode relay fingerprint rejects different sequence ids",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Same size, different membership: request finished + another admitted.
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 7}, {0, 0}, {30, 31}, 2));
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {6, 5}, {0, 0}, {30, 31}, 2));
}

TEST_CASE("Decode relay fingerprint rejects slot-reuse generation change",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Same sequence id recycled onto a new generation of the same slot.
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6}, {0, 1}, {30, 31}, 2));
}

TEST_CASE("Decode relay fingerprint rejects n_past mismatch",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // One row not at the position the relay advanced it to.
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 31}, 2));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 30}, 2));
  // The old contract compared against presented + 1, which accepted a row
  // one past the relayed position -- under the corrected semantics that is
  // a mismatch (e.g. the host advanced a row the relay never touched).
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {31, 31}, 2));
}

TEST_CASE("Decode relay fingerprint rejects size change", "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6, 7}, {0, 0, 0}, {30, 31, 32}, 3));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5}, {0}, {30}, 1));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {}, {}, {}, 0));
}

TEST_CASE("Decode relay fingerprint rejects a substituted token",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31}, {11, 12});
  // Armed tokens are the sampled tokens the relay wrote to device metadata;
  // the next step must present exactly those as its fed tokens. A different
  // token (regenerated/forced) would mean the device metadata is stale and
  // replay would silently decode it.
  REQUIRE(
      DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 31}, {11, 12}, 2));
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 31}, {11, 13}, 2));
  // Empty tokens vector = caller cannot supply them (S20-era behavior).
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 31}, {}, 2));
}

TEST_CASE("Decode relay fingerprint uses only the first count rows",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Executor passes preallocated max-capacity vectors; rows past count are
  // stale and must not participate.
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6, 99, 99}, {0, 0, 7, 7},
                                     {30, 31, 5, 5}, 2));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6, 99}, {0, 0, 7},
                                           {30, 31, 5}, 3));
}
