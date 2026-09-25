#include "server/auth/rate_limit_config.h"

#include <set>
#include <stdexcept>
#include <vector>

namespace inferflux {
namespace {
RequestBucketPolicy ParseBucket(const YAML::Node &node) {
  if (!node.IsMap() || node.size() != 2) {
    throw std::invalid_argument(
        "endpoint limit requires requests_per_minute and burst");
  }
  std::set<std::string> fields;
  for (const auto &item : node) {
    const auto key = item.first.as<std::string>();
    if ((key != "requests_per_minute" && key != "burst") ||
        !fields.insert(key).second) {
      throw std::invalid_argument("unknown or duplicate endpoint limit field");
    }
  }
  return {node["requests_per_minute"].as<int>(), node["burst"].as<int>()};
}

EndpointRateLimits ParseEndpointMap(const YAML::Node &node) {
  EndpointRateLimits limits;
  if (!node.IsDefined()) {
    return limits;
  }
  if (!node.IsMap() || node.size() == 0) {
    throw std::invalid_argument("endpoint_limits must be a nonempty mapping");
  }
  try {
    std::set<std::string> classes;
    for (const auto &item : node) {
      const auto key = item.first.as<std::string>();
      if (!classes.insert(key).second) {
        throw std::invalid_argument("duplicate endpoint rate class");
      }
      if (key == "generation") {
        limits.generation = ParseBucket(item.second);
      } else if (key == "embeddings") {
        limits.embeddings = ParseBucket(item.second);
      } else {
        throw std::invalid_argument("unknown endpoint rate class");
      }
    }
  } catch (const YAML::Exception &) {
    // Do not echo configuration values (which may include accidental secrets).
    throw std::invalid_argument(
        "endpoint limits must contain integer rates and bursts");
  }
  ValidateEndpointRateLimits(limits);
  return limits;
}
} // namespace

EndpointRateLimits ParseEndpointRateLimits(const YAML::Node &config) {
  if (!config.IsMap()) {
    return {};
  }
  std::vector<YAML::Node> auth_nodes;
  bool has_endpoint_policy = false;
  for (const auto &item : config) {
    if (item.first.IsScalar() && item.first.Scalar() == "auth") {
      auth_nodes.push_back(item.second);
      if (item.second.IsMap()) {
        for (const auto &field : item.second) {
          if (field.first.IsScalar() &&
              field.first.Scalar() == "endpoint_limits") {
            has_endpoint_policy = true;
          }
        }
      }
    }
  }
  if (!has_endpoint_policy) {
    return {}; // Keep legacy unconfigured parsing behavior.
  }
  if (auth_nodes.size() != 1) {
    throw std::invalid_argument(
        "duplicate auth mapping with endpoint rate policy");
  }
  std::set<std::string> policy_fields;
  for (const auto &field : auth_nodes.front()) {
    if (field.first.IsScalar()) {
      const auto key = field.first.Scalar();
      if ((key == "endpoint_limits" || key == "rate_limit_per_minute") &&
          !policy_fields.insert(key).second) {
        throw std::invalid_argument("duplicate auth request rate policy field");
      }
    }
  }
  return ParseEndpointMap(auth_nodes.front()["endpoint_limits"]);
}
} // namespace inferflux
