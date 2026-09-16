#include "server/policy/guardrail.h"

#include <algorithm>
#include <cctype>

namespace inferflux {
namespace {
std::vector<std::string> NormalizeWords(const std::vector<std::string> &words) {
  std::vector<std::string> normalized;
  normalized.reserve(words.size());
  for (auto word : words) {
    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (!word.empty()) {
      normalized.push_back(word);
    }
  }
  return normalized;
}
} // namespace

void Guardrail::SetBlocklist(const std::vector<std::string> &words) {
  UpdateBlocklist(words);
}

void Guardrail::UpdateBlocklist(const std::vector<std::string> &words) {
  std::lock_guard<std::mutex> lock(mutex_);
  blocklist_ = NormalizeWords(words);
}

std::vector<std::string> Guardrail::Blocklist() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return blocklist_;
}

bool Guardrail::Check(const std::string &text, std::string *reason) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string local_reason;
  bool allowed = true;
  if (!blocklist_.empty()) {
    std::string lower = text;
    std::transform(
        lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto &word : blocklist_) {
      if (word.empty()) {
        continue;
      }
      if (lower.find(word) != std::string::npos) {
        local_reason = "Blocked content keyword: " + word;
        allowed = false;
        break;
      }
    }
  }
  if (allowed && !opa_endpoint_.empty()) {
    OPAResult opa_result;
    auto evaluated = opa_client_.Evaluate(text, &opa_result);
    if (!opa_result.allow) {
      local_reason =
          opa_result.reason.empty() ? "OPA denied prompt" : opa_result.reason;
      allowed = false;
    } else if (!evaluated && local_reason.empty()) {
      local_reason = opa_result.reason;
    }
  }
  if (!allowed && reason) {
    *reason = local_reason;
  }
  return allowed;
}

bool Guardrail::Enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !blocklist_.empty() || !opa_endpoint_.empty();
}

void Guardrail::SetOPAEndpoint(const std::string &endpoint) {
  std::lock_guard<std::mutex> lock(mutex_);
  opa_endpoint_ = endpoint;
  opa_client_ = OPAClient(endpoint);
}

std::string Guardrail::OPAEndpoint() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return opa_endpoint_;
}

} // namespace inferflux
