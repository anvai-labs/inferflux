#include "model/model_load_spec.h"
#include "model/model_format.h"
#include <charconv>
#include <tuple>
#include <yaml-cpp/yaml.h>

namespace inferflux {
std::optional<DeviceSelector> ParseDeviceSelector(const std::string &value) {
  const auto colon = value.find(':');
  if (colon == std::string::npos)
    return std::nullopt;
  DeviceSelector out{value.substr(0, colon), 0};
  if (out.vendor != "cuda" && out.vendor != "rocm")
    return std::nullopt;
  const auto digits = value.substr(colon + 1);
  if (digits.empty() ||
      digits.find_first_not_of("0123456789") != std::string::npos)
    return std::nullopt;
  const auto parsed = std::from_chars(
      digits.data(), digits.data() + digits.size(), out.ordinal);
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size())
    return std::nullopt;
  return out;
}

bool ModelLoadSpec::HasOverrides() const {
  return device || context_size || gpu_layers || max_parallel_sequences ||
         kv_cache_type;
}
bool ModelLoadSpec::operator==(const ModelLoadSpec &o) const {
  return std::tie(id, path, format, backend, make_default, device, context_size,
                  gpu_layers, max_parallel_sequences, kv_cache_type) ==
         std::tie(o.id, o.path, o.format, o.backend, o.make_default, o.device,
                  o.context_size, o.gpu_layers, o.max_parallel_sequences,
                  o.kv_cache_type);
}
std::string ModelLoadSpec::Validate() const {
  if (device && !ParseDeviceSelector(*device))
    return "device must be cuda:N or rocm:N";
  if (context_size && *context_size <= 0)
    return "context_size must be positive";
  if (gpu_layers && *gpu_layers < -1)
    return "gpu_layers must be -1 or nonnegative";
  if (device && gpu_layers && *gpu_layers == 0)
    return "explicit GPU placement requires GPU layers";
  if (max_parallel_sequences &&
      (*max_parallel_sequences < 1 || *max_parallel_sequences > 256))
    return "max_parallel_sequences must be between 1 and 256";
  if (context_size && max_parallel_sequences &&
      *context_size < *max_parallel_sequences)
    return "context_size must cover every sequence";
  if (kv_cache_type && *kv_cache_type != "f16" && *kv_cache_type != "q8_0" &&
      *kv_cache_type != "q4_0")
    return "kv_cache_type must be f16, q8_0 or q4_0";
  return "";
}
LlamaBackendConfig
ModelLoadSpec::Apply(const LlamaBackendConfig &defaults) const {
  auto config = defaults;
  if (device) {
    config.device = *device;
    config.device_explicit = true;
  }
  if (context_size)
    config.ctx_size = *context_size;
  if (gpu_layers) {
    config.gpu_layers = *gpu_layers;
    config.gpu_layers_explicit = true;
  }
  if (max_parallel_sequences)
    config.max_parallel_sequences = *max_parallel_sequences;
  if (kv_cache_type) {
    config.llama_kv_cache_type = *kv_cache_type;
    config.kv_cache_type_explicit = true;
  }
  return config;
}
ModelLoadSpec ParseModelLoadSpec(const YAML::Node &node) {
  ModelLoadSpec out;
  if (!node.IsMap())
    throw YAML::RepresentationException(node.Mark(), "model must be a mapping");
  if (node["id"])
    out.id = node["id"].as<std::string>();
  if (node["path"])
    out.path = node["path"].as<std::string>();
  if (node["backend"])
    out.backend = node["backend"].as<std::string>();
  if (node["format"]) {
    const auto normalized =
        NormalizeModelFormat(node["format"].as<std::string>());
    if (!normalized.empty())
      out.format = normalized;
  }
  if (node["default"])
    out.make_default = node["default"].as<bool>();
  if (node["device"])
    out.device = node["device"].as<std::string>();
  if (node["device_id"])
    throw YAML::RepresentationException(
        node.Mark(), "use vendor-qualified device, not model device_id");
  if (node["context_size"])
    out.context_size = node["context_size"].as<int>();
  if (node["gpu_layers"])
    out.gpu_layers = node["gpu_layers"].as<int>();
  if (node["max_parallel_sequences"])
    out.max_parallel_sequences = node["max_parallel_sequences"].as<int>();
  if (node["kv_cache_type"])
    out.kv_cache_type = node["kv_cache_type"].as<std::string>();
  const auto error = out.Validate();
  if (!error.empty())
    throw YAML::RepresentationException(node.Mark(), error);
  return out;
}
} // namespace inferflux
