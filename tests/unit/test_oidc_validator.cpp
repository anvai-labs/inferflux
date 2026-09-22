#include <catch2/catch_amalgamated.hpp>

#include "server/auth/oidc_validator.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <limits>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

std::string Base64UrlEncode(const std::string &input) {
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  int val = 0;
  int valb = -6;
  for (unsigned char c : input) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) {
    out.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
  }
  for (char &c : out) {
    if (c == '+')
      c = '-';
    if (c == '/')
      c = '_';
  }
  while (!out.empty() && out.back() == '=') {
    out.pop_back();
  }
  return out;
}

const std::string &TestJwksJson() {
  static std::string jwks = R"({
    "keys": [
      {
        "alg": "RS256",
        "e": "AQAB",
        "kid": "test-key",
        "kty": "RSA",
        "n": "qnhNDv34eShaxVcVLlkr2sM4dAYNuPvaDf_ZFZ6crTv1QjXfyzdYIHElCjJ4OD6lYrlCWiMmvX-kKznBl4A2YZIMn1spdIJJFeXJjNykSxM5w-m6Fq4ikSrQYNPf4hQHyfSOTC5MD5-C9z6U4XWC7bAW4D31AYB1E8H9HcMWSL4n8FWM4jDaxq0294iux131cWjKECA4oyO41a9Y8BiXpt9S8BBKzz4eNHU15hdKN50i2OBtmQgm8x36ywMQ2QuRUXdHgZcM8t8oftJ5e0IzUQoNQk67WpuUOr2K-pDawDo0GBmWIRxeXAxKKlsgAhiwF-Z1w9He3SSZGGpGCdGANw",
        "use": "sig"
      }
    ]
  })";
  return jwks;
}

const std::string &TestSignatureB64() {
  static std::string sig = Base64UrlEncode("signature-ok");
  return sig;
}

std::string
MakeSignedJWT(const json &payload,
              const std::string &signature_b64 = TestSignatureB64()) {
  json header = {{"alg", "RS256"}, {"kid", "test-key"}, {"typ", "JWT"}};
  return Base64UrlEncode(header.dump()) + "." +
         Base64UrlEncode(payload.dump()) + "." + signature_b64;
}

void ConfigureValidator(inferflux::OIDCValidator *validator) {
  validator->LoadJwksForTesting(TestJwksJson());
  validator->SetSignatureVerifierForTesting(
      [](const std::string &header_payload, const std::string &signature,
         const inferflux::OIDCValidator::JwkKey &jwk) {
        (void)header_payload;
        return jwk.kid == "test-key" && signature == TestSignatureB64();
      });
}

} // namespace

TEST_CASE("OIDCValidator disabled without config", "[oidc]") {
  inferflux::OIDCValidator validator;
  REQUIRE(!validator.Enabled());
  REQUIRE(!validator.Validate("any-token", nullptr));
}

TEST_CASE("OIDCValidator validates issuer and audience", "[oidc]") {
  inferflux::OIDCValidator validator("https://issuer.example.com",
                                     "my-audience");
  ConfigureValidator(&validator);

  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://issuer.example.com"},
      {"aud", "my-audience"},
      {"sub", "user-123"},
      {"exp", now + 3600},
  };
  std::string subject;
  REQUIRE(validator.Validate(MakeSignedJWT(payload), &subject));
  REQUIRE(subject == "user-123");
}

TEST_CASE("OIDCValidator rejects wrong issuer", "[oidc]") {
  inferflux::OIDCValidator validator("https://correct.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://wrong.example.com"},
      {"aud", "aud"},
      {"sub", "user-123"},
      {"exp", now + 3600},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects wrong audience", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "correct-aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "wrong-aud"},
      {"sub", "user-123"},
      {"exp", now + 3600},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects expired token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"sub", "user-123"},
      {"exp", 1000},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects not-yet-valid token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"sub", "user-123"},
      {"exp", now + 3600},
      {"nbf", now + 7200},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects missing or malformed identity claims",
          "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  const json valid = {{"iss", "https://iss.example.com"},
                      {"aud", "aud"},
                      {"sub", "user-123"},
                      {"exp", now + 3600}};
  for (const auto &field : {"sub", "exp"}) {
    CAPTURE(field);
    auto payload = valid;
    payload.erase(field);
    std::string subject = "previous-user";
    REQUIRE_FALSE(validator.Validate(MakeSignedJWT(payload), &subject));
    REQUIRE(subject.empty());
  }
  for (const auto &change : std::vector<std::pair<std::string, json>>{
           {"sub", ""},
           {"sub", 42},
           {"iss", 42},
           {"exp", nullptr},
           {"exp", "later"},
           {"exp", now},
           {"exp", now + 0.5},
           {"exp", std::numeric_limits<uint64_t>::max()},
           {"nbf", "later"},
           {"nbf", nullptr},
           {"aud", json::array({"aud", 42})}}) {
    CAPTURE(change.first, change.second);
    auto payload = valid;
    payload[change.first] = change.second;
    std::string subject = "previous-user";
    REQUIRE_FALSE(validator.Validate(MakeSignedJWT(payload), &subject));
    REQUIRE(subject.empty());
  }
}

TEST_CASE("OIDCValidator rejects invalid signature via override", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"sub", "user-123"},
      {"exp", now + 3600},
  };
  auto bad_signature = Base64UrlEncode("different-signature");
  REQUIRE(!validator.Validate(MakeSignedJWT(payload, bad_signature), nullptr));
}

TEST_CASE("OIDCValidator rejects malformed token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  REQUIRE(!validator.Validate("not-a-jwt", nullptr));
  REQUIRE(!validator.Validate("only.one.dot", nullptr));
  for (const auto &header :
       std::vector<json>{json::array(),
                         nullptr,
                         {{"alg", 42}},
                         {{"alg", "RS256"}, {"kid", 42}}}) {
    const auto jwt = Base64UrlEncode(header.dump()) + "." +
                     Base64UrlEncode(json::object().dump()) + "." +
                     TestSignatureB64();
    REQUIRE_FALSE(validator.Validate(jwt, nullptr));
  }
  REQUIRE_FALSE(validator.Validate(std::string(16385, 'a'), nullptr));
}
