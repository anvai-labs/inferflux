#include "server/http/completion_payload.h"

namespace inferflux {

std::string SerializeJsonUtf8Safe(const nlohmann::json &payload) {
  return payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

} // namespace inferflux
