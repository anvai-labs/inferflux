#pragma once
#include "server/auth/rate_limiter.h"
#include <yaml-cpp/yaml.h>
namespace inferflux {
// Read the full document so duplicate enclosing keys cannot hide opt-in policy.
// An absent policy preserves the legacy shared-bucket behavior.
EndpointRateLimits ParseEndpointRateLimits(const YAML::Node &config);
} // namespace inferflux
