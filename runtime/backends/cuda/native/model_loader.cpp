#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/safetensors_adapter.h"
#include "runtime/string_utils.h"
#include "server/logging/logger.h"
#include <filesystem>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

std::string DequantizedCachePolicyToString(DequantizedCachePolicy policy) {
  switch (policy) {
  case DequantizedCachePolicy::kNone:
    return "none";
  case DequantizedCachePolicy::kModelLifetime:
    return "model";
  case DequantizedCachePolicy::kBatchLifetime:
    return "batch";
  }
  return "batch";
}

bool ParseDequantizedCachePolicy(const std::string &raw,
                                 DequantizedCachePolicy *out) {
  if (!out) {
    return false;
  }
  const std::string lowered = inferflux::ToLower(raw);
  if (lowered == "none" || lowered == "off" || lowered == "disabled") {
    *out = DequantizedCachePolicy::kNone;
    return true;
  }
  if (lowered == "model" || lowered == "model_lifetime") {
    *out = DequantizedCachePolicy::kModelLifetime;
    return true;
  }
  if (lowered == "batch" || lowered == "batch_lifetime") {
    *out = DequantizedCachePolicy::kBatchLifetime;
    return true;
  }
  return false;
}

//==============================================================================
// Factory Functions
//==============================================================================

std::unique_ptr<IModelLoader>
CreateModelLoader(const std::filesystem::path &model_path) {
  log::Info("model_loader_factory",
            "Detecting model format for: " + model_path.string());

  // Check if path is a directory or file
  bool is_directory = std::filesystem::is_directory(model_path);

  // Case 1: Directory with config.json → safetensors
  if (is_directory) {
    auto config_path = (model_path / "config.json").string();
    if (std::filesystem::exists(config_path)) {
      log::Info("model_loader_factory", "Detected safetensors format");
      return std::make_unique<SafetensorsLoaderAdapter>();
    }

    // Check for model.safetensors.index.json
    auto index_path = (model_path / "model.safetensors.index.json").string();
    if (std::filesystem::exists(index_path)) {
      log::Info("model_loader_factory", "Detected safetensors format (index)");
      return std::make_unique<SafetensorsLoaderAdapter>();
    }

    // Check for single model.safetensors file
    auto single_path = (model_path / "model.safetensors").string();
    if (std::filesystem::exists(single_path)) {
      log::Info("model_loader_factory", "Detected safetensors format (single)");
      return std::make_unique<SafetensorsLoaderAdapter>();
    }
  }

  // Case 2: GGUF file (has .gguf extension)
  if (!is_directory) {
    std::string ext = model_path.extension().string();
    if (ext == ".gguf") {
      log::Info("model_loader_factory", "Detected GGUF format");
      return std::make_unique<GGUFModelLoader>();
    }
  }

  log::Error("model_loader_factory",
             "Unable to detect model format for: " + model_path.string());
  return nullptr;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
