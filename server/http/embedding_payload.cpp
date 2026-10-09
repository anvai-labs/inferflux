#include "server/http/embedding_payload.h"

#include <openssl/evp.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace inferflux {
nlohmann::json BuildEmbeddingValue(const std::vector<float> &values,
                                   bool base64) {
  if (!base64) {
    return values;
  }
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  // EVP's input and output lengths are int; leave room for base64 expansion.
  if (values.size() >
      static_cast<std::size_t>((std::numeric_limits<int>::max() - 4) / 16)) {
    throw std::length_error("embedding exceeds base64 encoding bound");
  }
  if (values.empty()) {
    return "";
  }
  std::string bytes;
  bytes.reserve(values.size() * 4);
  for (float value : values) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<char>((bits >> shift) & 0xff));
    }
  }
  std::string encoded(4 * ((bytes.size() + 2) / 3) + 1, '\0');
  const auto length =
      EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),
                      reinterpret_cast<const unsigned char *>(bytes.data()),
                      static_cast<int>(bytes.size()));
  encoded.resize(static_cast<std::size_t>(length));
  return encoded;
}
} // namespace inferflux
