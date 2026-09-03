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

TEST_CASE("Decode relay fingerprint matches identical batch",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Relay advanced n_past to 30/31; next call carries 29/30 (match adds 1).
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 30}, 2));
}

TEST_CASE("Decode relay fingerprint rejects different sequence ids",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Same size, different membership: request finished + another admitted.
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 7}, {0, 0}, {29, 30}, 2));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {6, 5}, {0, 0}, {29, 30}, 2));
}

TEST_CASE("Decode relay fingerprint rejects slot-reuse generation change",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Same sequence id recycled onto a new generation of the same slot.
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 1}, {29, 30}, 2));
}

TEST_CASE("Decode relay fingerprint rejects n_past mismatch",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // One row not at the position the relay advanced it to.
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 31}, 2));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {30, 30}, 2));
}

TEST_CASE("Decode relay fingerprint rejects size change", "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6, 7}, {0, 0, 0}, {29, 30, 31}, 3));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {5}, {0}, {29}, 1));
  REQUIRE_FALSE(DecodeRelayIdentityMatches(armed, {}, {}, {}, 0));
}

TEST_CASE("Decode relay fingerprint rejects a substituted token",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31}, {11, 12});
  // Same identity, one row fed a different token than the armed row's last
  // token (regenerated/forced) — replay would silently decode the stale
  // device token.
  REQUIRE(
      DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 30}, {11, 12}, 2));
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 30}, {11, 13}, 2));
  // Empty tokens vector = caller cannot supply them (S20-era behavior).
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6}, {0, 0}, {29, 30}, {}, 2));
}

TEST_CASE("Decode relay fingerprint uses only the first count rows",
          "[decode_relay]") {
  const auto armed = Armed({5, 6}, {0, 0}, {30, 31});
  // Executor passes preallocated max-capacity vectors; rows past count are
  // stale and must not participate.
  REQUIRE(DecodeRelayIdentityMatches(armed, {5, 6, 99, 99}, {0, 0, 7, 7},
                                     {29, 30, 5, 5}, 2));
  REQUIRE_FALSE(
      DecodeRelayIdentityMatches(armed, {5, 6, 99}, {0, 0, 7}, {29, 30, 5}, 3));
}
