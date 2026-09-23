#include "server/auth/oidc_validator.h"

#include <nlohmann/json.hpp>

#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>
#include <openssl/rsa.h>

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace inferflux {

OIDCValidator::OIDCValidator(std::string issuer, std::string audience)
    : issuer_(std::move(issuer)), audience_(std::move(audience)) {}

std::string OIDCValidator::Base64UrlDecode(const std::string &input) {
  std::string normalized = input;
  for (char &c : normalized) {
    if (c == '-')
      c = '+';
    if (c == '_')
      c = '/';
  }
  while (normalized.size() % 4 != 0) {
    normalized.push_back('=');
  }
  std::string output;
  output.reserve(normalized.size() * 3 / 4);
  auto decode_char = [](char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    return -1;
  };
  for (std::size_t i = 0; i < normalized.size(); i += 4) {
    int b0 = decode_char(normalized[i]);
    int b1 = decode_char(normalized[i + 1]);
    int b2 = normalized[i + 2] == '=' ? -1 : decode_char(normalized[i + 2]);
    int b3 = normalized[i + 3] == '=' ? -1 : decode_char(normalized[i + 3]);
    if (b0 < 0 || b1 < 0 || (normalized[i + 2] != '=' && b2 < 0) ||
        (normalized[i + 3] != '=' && b3 < 0)) {
      return {};
    }
    output.push_back(static_cast<char>((b0 << 2) | (b1 >> 4)));
    if (b2 >= 0) {
      output.push_back(static_cast<char>(((b1 & 0xF) << 4) | (b2 >> 2)));
    }
    if (b3 >= 0) {
      output.push_back(static_cast<char>(((b2 & 0x3) << 6) | b3));
    }
  }
  return output;
}

bool OIDCValidator::VerifyRS256(const std::string &header_payload,
                                const std::string &signature_b64url,
                                const std::string &n_b64url,
                                const std::string &e_b64url) {
  std::string sig_bytes = Base64UrlDecode(signature_b64url);
  std::string n_bytes = Base64UrlDecode(n_b64url);
  std::string e_bytes = Base64UrlDecode(e_b64url);
  if (sig_bytes.empty() || n_bytes.empty() || e_bytes.empty()) {
    return false;
  }

  std::unique_ptr<BIGNUM, decltype(&BN_free)> bn_n(
      BN_bin2bn(reinterpret_cast<const unsigned char *>(n_bytes.data()),
                static_cast<int>(n_bytes.size()), nullptr),
      &BN_free);
  std::unique_ptr<BIGNUM, decltype(&BN_free)> bn_e(
      BN_bin2bn(reinterpret_cast<const unsigned char *>(e_bytes.data()),
                static_cast<int>(e_bytes.size()), nullptr),
      &BN_free);
  if (!bn_n || !bn_e) {
    return false;
  }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  // RAII wrapper for RSA — note: EVP_PKEY_assign_RSA transfers ownership,
  // so we release the RSA unique_ptr after successful assignment.
  std::unique_ptr<RSA, decltype(&RSA_free)> rsa(RSA_new(), &RSA_free);
  if (!rsa) {
    return false;
  }
  if (RSA_set0_key(rsa.get(), bn_n.release(), bn_e.release(), nullptr) != 1) {
    return false;
  }
  std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> pkey(EVP_PKEY_new(),
                                                           &EVP_PKEY_free);
  if (!pkey) {
    return false;
  }
  // EVP_PKEY_assign_RSA takes ownership of rsa on success.
  if (EVP_PKEY_assign_RSA(pkey.get(), rsa.get()) != 1) {
    return false;
  }
  rsa.release(); // pkey now owns the RSA object
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  bool ok = false;
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> md_ctx(
      EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (md_ctx) {
    EVP_PKEY_CTX *pctx = nullptr;
    if (EVP_DigestVerifyInit(md_ctx.get(), &pctx, EVP_sha256(), nullptr,
                             pkey.get()) == 1) {
#if defined(RSA_PKCS1_PADDING)
      if (pctx) {
        EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PADDING);
        EVP_PKEY_CTX_set_signature_md(pctx, EVP_sha256());
      }
#endif
      if (EVP_DigestVerifyUpdate(md_ctx.get(), header_payload.data(),
                                 header_payload.size()) == 1 &&
          EVP_DigestVerifyFinal(
              md_ctx.get(),
              reinterpret_cast<const unsigned char *>(sig_bytes.data()),
              sig_bytes.size()) == 1) {
        ok = true;
      }
    }
  }
  return ok;
}

bool OIDCValidator::AudienceMatches(const json &payload) const {
  if (!payload.contains("aud")) {
    return false;
  }
  const auto &aud_field = payload["aud"];
  if (aud_field.is_string()) {
    return aud_field.get<std::string>() == audience_;
  }
  if (aud_field.is_array()) {
    bool matches = false;
    for (const auto &entry : aud_field) {
      if (!entry.is_string()) {
        return false;
      }
      matches = matches || entry.get<std::string>() == audience_;
    }
    return matches;
  }
  return false;
}

bool OIDCValidator::FetchJwksFromNetwork(
    std::map<std::string, JwkKey> *out) const {
  try {
    HttpResponse resp = http_client_.Get(JwksUrl());
    if (resp.status != 200) {
      return false;
    }
    return LoadJwksJson(resp.body, out);
  } catch (const std::exception &) {
    return false;
  }
}

bool OIDCValidator::LoadJwksJson(const std::string &body,
                                 std::map<std::string, JwkKey> *out) const {
  try {
    json doc = json::parse(body);
    if (!doc.contains("keys") || !doc["keys"].is_array()) {
      return false;
    }
    std::map<std::string, JwkKey> parsed;
    for (const auto &key : doc["keys"]) {
      if (!key.contains("kty") || key["kty"] != "RSA")
        continue;
      if (key.contains("use") && key["use"].is_string() && key["use"] != "sig")
        continue;
      if (!key.contains("n") || !key.contains("e"))
        continue;
      JwkKey jwk;
      jwk.kid = key.value("kid", "");
      jwk.n = key["n"].get<std::string>();
      jwk.e = key["e"].get<std::string>();
      parsed[jwk.kid] = jwk;
    }
    if (parsed.empty()) {
      return false;
    }
    *out = std::move(parsed);
    return true;
  } catch (const json::exception &) {
    return false;
  }
}

bool OIDCValidator::EnsureJwks(const std::string &desired_kid) const {
  std::unique_lock<std::mutex> lock(jwks_mutex_);
  auto now = std::chrono::steady_clock::now();
  if (jwks_loaded_ && now < jwks_expiry_ &&
      (desired_kid.empty() || jwks_.find(desired_kid) != jwks_.end())) {
    return true;
  }
  lock.unlock();

  std::map<std::string, JwkKey> fresh;
  if (!FetchJwksFromNetwork(&fresh)) {
    return false;
  }

  lock.lock();
  jwks_ = std::move(fresh);
  jwks_loaded_ = true;
  jwks_expiry_ = std::chrono::steady_clock::now() + std::chrono::minutes(5);
  return desired_kid.empty() || jwks_.find(desired_kid) != jwks_.end();
}

bool OIDCValidator::ResolveKey(const std::string &kid, JwkKey *key) const {
  std::lock_guard<std::mutex> lock(jwks_mutex_);
  if (!jwks_loaded_) {
    return false;
  }
  if (!kid.empty()) {
    auto it = jwks_.find(kid);
    if (it == jwks_.end()) {
      return false;
    }
    *key = it->second;
    return true;
  }
  if (jwks_.size() != 1) {
    return false;
  }
  *key = jwks_.begin()->second;
  return true;
}

std::string OIDCValidator::JwksUrl() const {
  if (issuer_.empty()) {
    return "";
  }
  if (issuer_.back() == '/') {
    return issuer_ + ".well-known/jwks.json";
  }
  return issuer_ + "/.well-known/jwks.json";
}

bool OIDCValidator::Validate(const std::string &token,
                             std::string *subject_out) const {
  if (subject_out) {
    subject_out->clear();
  }
  if (!Enabled() || token.size() > 16384) {
    return false;
  }
  auto first_dot = token.find('.');
  auto second_dot =
      token.find('.', first_dot == std::string::npos ? 0 : first_dot + 1);
  if (first_dot == std::string::npos || second_dot == std::string::npos ||
      token.find('.', second_dot + 1) != std::string::npos) {
    return false;
  }
  std::string header_str = Base64UrlDecode(token.substr(0, first_dot));
  std::string payload_str =
      Base64UrlDecode(token.substr(first_dot + 1, second_dot - first_dot - 1));
  if (header_str.empty() || payload_str.empty()) {
    return false;
  }

  json header;
  json payload;
  try {
    header = json::parse(header_str);
    payload = json::parse(payload_str);
  } catch (const json::exception &) {
    return false;
  }

  if (!header.is_object() || !payload.is_object() || !header.contains("alg") ||
      !header["alg"].is_string() || header["alg"] != "RS256" ||
      (header.contains("kid") && !header["kid"].is_string()) ||
      !payload.contains("iss") || !payload["iss"].is_string() ||
      !payload.contains("sub") || !payload["sub"].is_string() ||
      payload["sub"].get_ref<const std::string &>().empty()) {
    return false;
  }

  // Validate issuer and audience claims.
  std::string iss = payload.value("iss", "");
  if (iss != issuer_ || !AudienceMatches(payload)) {
    return false;
  }

  // Validate temporal claims (exp, nbf).
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  const auto valid_timestamp = [](const json &value) {
    return value.is_number_integer() &&
           (!value.is_number_unsigned() ||
            value.get<uint64_t>() <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  };
  if (!payload.contains("exp") || !valid_timestamp(payload["exp"]) ||
      payload["exp"].get<int64_t>() <= now) {
    return false;
  }
  if (payload.contains("nbf") && (!valid_timestamp(payload["nbf"]) ||
                                  payload["nbf"].get<int64_t>() > now)) {
    return false;
  }

  std::string kid = header.value("kid", "");
  if (!EnsureJwks(kid)) {
    return false;
  }
  JwkKey jwk;
  if (!ResolveKey(kid, &jwk)) {
    return false;
  }

  std::string header_payload = token.substr(0, second_dot);
  std::string signature_b64 = token.substr(second_dot + 1);
  bool signature_ok = false;
  if (signature_verifier_override_) {
    signature_ok =
        signature_verifier_override_(header_payload, signature_b64, jwk);
  } else {
    signature_ok = VerifyRS256(header_payload, signature_b64, jwk.n, jwk.e);
  }
  if (!signature_ok) {
    return false;
  }

  if (subject_out) {
    *subject_out = payload["sub"].get<std::string>();
  }
  return true;
}

void OIDCValidator::LoadJwksForTesting(const std::string &jwks_json) const {
  std::map<std::string, JwkKey> parsed;
  if (!LoadJwksJson(jwks_json, &parsed)) {
    throw std::runtime_error("invalid JWKS json for testing");
  }
  std::lock_guard<std::mutex> lock(jwks_mutex_);
  jwks_ = std::move(parsed);
  jwks_loaded_ = true;
  jwks_expiry_ = std::chrono::steady_clock::now() + std::chrono::hours(1);
}

bool OIDCValidator::HasCachedKeyForTesting(const std::string &kid) const {
  std::lock_guard<std::mutex> lock(jwks_mutex_);
  return jwks_.find(kid) != jwks_.end();
}

bool OIDCValidator::EnsureJwksForTesting(const std::string &kid) const {
  return EnsureJwks(kid);
}

bool OIDCValidator::VerifySignatureForTesting(
    const std::string &header_payload, const std::string &signature_b64url,
    const std::string &kid) const {
  JwkKey jwk;
  if (!ResolveKey(kid, &jwk)) {
    return false;
  }
  return VerifyRS256(header_payload, signature_b64url, jwk.n, jwk.e);
}

void OIDCValidator::SetSignatureVerifierForTesting(
    std::function<bool(const std::string &, const std::string &,
                       const JwkKey &)>
        verifier) const {
  signature_verifier_override_ = std::move(verifier);
}

} // namespace inferflux
