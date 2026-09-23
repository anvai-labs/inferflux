#include "server/auth/api_key_auth.h"

#include <openssl/sha.h>

#include <iomanip>
#include <mutex>
#include <sstream>

namespace inferflux {

namespace {
std::unordered_set<std::string>
MakeScopes(const std::vector<std::string> &scopes) {
  std::unordered_set<std::string> set;
  if (scopes.empty()) {
    set.insert("generate");
    set.insert("read");
    return set;
  }
  for (const auto &scope : scopes) {
    if (!scope.empty()) {
      set.insert(scope);
    }
  }
  return set;
}
} // namespace

std::string ApiKeyAuth::HashKey(const std::string &key) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(key.data()), key.size(), hash);
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    hex << std::setw(2) << static_cast<int>(hash[i]);
  }
  return hex.str();
}

std::string ApiKeyAuth::SubjectFromHash(const std::string &hash) {
  return "api-key:" + hash;
}

void ApiKeyAuth::AddKey(const std::string &key,
                        const std::vector<std::string> &scopes) {
  AddKeyHashed(HashKey(key), scopes);
}

void ApiKeyAuth::AddKeyHashed(const std::string &hash,
                              const std::vector<std::string> &scopes) {
  std::unique_lock lock(mutex_);
  keys_[hash] = MakeScopes(scopes);
  if (keys_[hash].find("read") == keys_[hash].end()) {
    keys_[hash].insert("read");
  }
}

bool ApiKeyAuth::IsAllowed(const std::string &key) const {
  return IsAllowedByHash(HashKey(key));
}

bool ApiKeyAuth::IsAllowedByHash(const std::string &hash) const {
  std::shared_lock lock(mutex_);
  return keys_.find(hash) != keys_.end();
}

bool ApiKeyAuth::HasKeys() const {
  std::shared_lock lock(mutex_);
  return !keys_.empty();
}

std::vector<std::string> ApiKeyAuth::Scopes(const std::string &key) const {
  return ScopesByHash(HashKey(key));
}

std::vector<std::string>
ApiKeyAuth::ScopesByHash(const std::string &hash) const {
  std::shared_lock lock(mutex_);
  auto it = keys_.find(hash);
  if (it == keys_.end()) {
    return {};
  }
  return std::vector<std::string>(it->second.begin(), it->second.end());
}

void ApiKeyAuth::RemoveKey(const std::string &key) {
  std::unique_lock lock(mutex_);
  keys_.erase(HashKey(key));
}

} // namespace inferflux
