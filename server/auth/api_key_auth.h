#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inferflux {

class ApiKeyAuth {
public:
  // Add a plaintext key — hashes it internally before storing.
  void AddKey(const std::string &key,
              const std::vector<std::string> &scopes = {});

  // Add a key that is already SHA-256 hex-hashed (e.g., loaded from
  // PolicyStore). Stores directly without re-hashing.
  void AddKeyHashed(const std::string &hash,
                    const std::vector<std::string> &scopes = {});
  bool IsAllowed(const std::string &key) const;
  bool IsAllowedByHash(const std::string &hash) const;
  bool HasKeys() const;
  std::vector<std::string> Scopes(const std::string &key) const;
  std::vector<std::string> ScopesByHash(const std::string &hash) const;
  void RemoveKey(const std::string &key);

  // Hash a plaintext key to its storage representation.
  static std::string HashKey(const std::string &key);

  // Nonsecret identity for an already-derived storage hash; never a bearer key.
  static std::string SubjectFromHash(const std::string &hash);

private:
  // Keys are stored by their SHA-256 hex hash.
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unordered_set<std::string>> keys_;
};

} // namespace inferflux
