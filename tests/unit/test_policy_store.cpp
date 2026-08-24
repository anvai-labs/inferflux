#include <catch2/catch_amalgamated.hpp>

#include "policy/policy_store.h"
#include "server/auth/api_key_auth.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

TEST_CASE("PolicyStore round-trip without encryption", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_test.conf";

  {
    inferflux::PolicyStore store(tmp_path.string());
    store.SetApiKey("key1", {"generate", "admin"});
    store.SetApiKey("key2", {"read"});
    store.SetGuardrailBlocklist({"bad", "evil"});
    store.SetRateLimitPerMinute(100);
    inferflux::RoutingPolicyEntry routing;
    routing.allow_default_fallback = false;
    routing.require_ready_backend = true;
    routing.fallback_scope = "same_path_only";
    store.SetRoutingPolicy(routing);
    REQUIRE(store.Save());
  }

  {
    inferflux::PolicyStore store(tmp_path.string());
    REQUIRE(store.Load());

    auto keys = store.ApiKeys();
    REQUIRE(keys.size() == 2);

    auto bl = store.GuardrailBlocklist();
    REQUIRE(bl.size() == 2);

    REQUIRE(store.RateLimitPerMinute() == 100);
    auto routing = store.RoutingPolicy();
    REQUIRE(routing.has_value());
    REQUIRE(!routing->allow_default_fallback);
    REQUIRE(routing->require_ready_backend);
    REQUIRE(routing->fallback_scope == "same_path_only");
  }

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore round-trip with AES-GCM encryption", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_enc.conf";

  {
    inferflux::PolicyStore store(
        inferflux::PolicyStoreConfig{tmp_path.string(), "my-secret-pass"});
    store.SetApiKey("encrypted-key", {"generate"});
    store.SetGuardrailBlocklist({"secret"});
    store.SetRateLimitPerMinute(50);
    inferflux::RoutingPolicyEntry routing;
    routing.allow_default_fallback = true;
    routing.require_ready_backend = false;
    routing.fallback_scope = "any_compatible";
    store.SetRoutingPolicy(routing);
    REQUIRE(store.Save());
  }

  // Re-open with same passphrase: should decrypt.
  {
    inferflux::PolicyStore store(
        inferflux::PolicyStoreConfig{tmp_path.string(), "my-secret-pass"});
    REQUIRE(store.Load());
    REQUIRE(store.ApiKeys().size() == 1);
    REQUIRE(store.GuardrailBlocklist().size() == 1);
    REQUIRE(store.RateLimitPerMinute() == 50);
    auto routing = store.RoutingPolicy();
    REQUIRE(routing.has_value());
    REQUIRE(routing->allow_default_fallback);
    REQUIRE(!routing->require_ready_backend);
    REQUIRE(routing->fallback_scope == "any_compatible");
  }

  // Wrong passphrase: should fail.
  {
    inferflux::PolicyStore store(
        inferflux::PolicyStoreConfig{tmp_path.string(), "wrong-pass"});
    REQUIRE(!store.Load());
  }

  // No passphrase on encrypted file: should fail.
  {
    inferflux::PolicyStore store(tmp_path.string());
    REQUIRE(!store.Load());
  }

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore loads legacy files without routing policy section",
          "[policy]") {
  auto tmp_path = std::filesystem::temp_directory_path() /
                  "inferflux_policy_legacy_no_routing.conf";

  {
    std::ofstream out(tmp_path.string(), std::ios::trunc);
    REQUIRE(out.good());
    out << "[api_keys]\n";
    out << "abc=read\n";
    out << "\n[guardrail]\n";
    out << "words=secret\n";
    out << "\n[rate_limit]\n";
    out << "tokens=10\n";
  }

  inferflux::PolicyStore store(tmp_path.string());
  REQUIRE(store.Load());
  REQUIRE(!store.RoutingPolicy().has_value());
  REQUIRE(store.RateLimitPerMinute() == 10);

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore Load falls back to backup when primary is corrupt",
          "[policy]") {
  auto tmp_path = std::filesystem::temp_directory_path() /
                  "inferflux_policy_backup_fallback.conf";
  auto backup_path = tmp_path;
  backup_path += ".bak";

  {
    inferflux::PolicyStore store(
        inferflux::PolicyStoreConfig{tmp_path.string(), "my-secret-pass"});
    store.SetRateLimitPerMinute(77);
    inferflux::RoutingPolicyEntry routing;
    routing.allow_default_fallback = false;
    routing.require_ready_backend = true;
    routing.fallback_scope = "same_path_only";
    store.SetRoutingPolicy(routing);
    REQUIRE(store.Save());
  }

  std::filesystem::copy_file(tmp_path, backup_path,
                             std::filesystem::copy_options::overwrite_existing);
  {
    std::ofstream out(tmp_path.string(), std::ios::trunc);
    REQUIRE(out.good());
    out << "ENC\nnonce=broken\ntag=broken\ndata=broken\n";
  }

  inferflux::PolicyStore restored(
      inferflux::PolicyStoreConfig{tmp_path.string(), "my-secret-pass"});
  REQUIRE(restored.Load());
  REQUIRE(restored.RateLimitPerMinute() == 77);
  auto routing = restored.RoutingPolicy();
  REQUIRE(routing.has_value());
  REQUIRE(!routing->allow_default_fallback);
  REQUIRE(routing->require_ready_backend);
  REQUIRE(routing->fallback_scope == "same_path_only");

  std::filesystem::remove(tmp_path);
  std::filesystem::remove(backup_path);
}

TEST_CASE("PolicyStore Save fails on unwritable path", "[policy]") {
#ifdef _WIN32
  // Windows has no /proc; use a path under a nonexistent root directory.
  inferflux::PolicyStore store("Z:\\nonexistent_volume\\inferflux_policy_unwritable.conf");
#else
  inferflux::PolicyStore store("/proc/inferflux_policy_unwritable.conf");
#endif
  store.SetRateLimitPerMinute(5);
  REQUIRE(!store.Save());
}

TEST_CASE("PolicyStore RemoveApiKey", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_rm.conf";
  inferflux::PolicyStore store(tmp_path.string());
  store.SetApiKey("to-remove", {"read"});
  REQUIRE(store.ApiKeys().size() == 1);

  REQUIRE(store.RemoveApiKey("to-remove"));
  REQUIRE(store.ApiKeys().empty());

  // Removing non-existent key returns false.
  REQUIRE(!store.RemoveApiKey("nope"));

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore Load from nonexistent file", "[policy]") {
  inferflux::PolicyStore store("/tmp/inferflux_no_such_file_xyz.conf");
  REQUIRE(!store.Load());
}

TEST_CASE(
    "PolicyStore stores hashed keys; AddKeyHashed round-trip with ApiKeyAuth",
    "[policy][auth]") {
  // Keys stored in PolicyStore are SHA-256 hashed — never raw plaintext.
  inferflux::PolicyStore store("/tmp/inferflux_hash_test_xyz.conf");
  store.SetApiKey("plaintext-key", {"generate"});

  auto keys = store.ApiKeys();
  REQUIRE(keys.size() == 1);
  // The stored value should be a 64-character hex string (SHA-256), not the
  // original.
  REQUIRE(keys[0].key != "plaintext-key");
  REQUIRE(keys[0].key.size() == 64); // SHA-256 hex = 32 bytes × 2

  // AddKeyHashed must let a pre-hashed key be authenticated.
  inferflux::ApiKeyAuth auth;
  auth.AddKeyHashed(keys[0].key, keys[0].scopes);

  // The original plaintext is authenticated via normal IsAllowed (which also
  // hashes).
  REQUIRE(auth.IsAllowed("plaintext-key"));

  // A different plaintext is rejected.
  REQUIRE(!auth.IsAllowed("wrong-key"));
}
