#pragma once

#include <nlohmann/json.hpp>
#include <vector>

namespace inferflux {
// The base64 payload is IEEE754 float32, little endian on every host.
nlohmann::json BuildEmbeddingValue(const std::vector<float> &values,
                                   bool base64);
} // namespace inferflux
