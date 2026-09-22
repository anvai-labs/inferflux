#include "runtime/backends/llama/llama_device_placement.h"
#include "model/model_load_spec.h"
// Pinned-revision adapter: public llama API has no weight-buffer inventory.
// Keep the internal layout dependency confined to this translation unit.
#include "external/llama.cpp/src/llama-model.h"
#include <set>

namespace inferflux {
ggml_backend_dev_t ResolveLlamaDevice(const std::string &value,
                                      std::string *error) {
  const auto selector = ParseDeviceSelector(value);
  if (!selector) {
    *error = "placement_invalid: device must be cuda:N or rocm:N";
    return nullptr;
  }
  const char *registry_name = selector->vendor == "cuda" ? "CUDA" : "ROCm";
  auto registry = ggml_backend_reg_by_name(registry_name);
  if (!registry || static_cast<size_t>(selector->ordinal) >=
                       ggml_backend_reg_dev_count(registry)) {
    *error = "placement_unavailable: " + value;
    return nullptr;
  }
  auto device = ggml_backend_reg_dev_get(registry, selector->ordinal);
  if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
    *error = "placement_mismatch: selected device is not a GPU: " + value;
    return nullptr;
  }
  return device;
}

bool ObserveLlamaWeightPlacement(const llama_model *model,
                                 ggml_backend_dev_t expected,
                                 DevicePlacement *placement,
                                 std::string *error) {
  std::set<int> gpu_layers;
  std::set<const ggml_tensor *> seen;
  for (const auto &[name, tensor] : model->tensors_by_name) {
    if (!tensor || !tensor->buffer || !seen.insert(tensor).second)
      continue;
    auto buffer = tensor->buffer;
    auto device =
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
    const bool gpu =
        device && !ggml_backend_buffer_is_host(buffer) &&
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (gpu) {
      if (expected && device != expected) {
        *error =
            "placement_mismatch: weight buffer resides on an unrequested GPU";
        return false;
      }
      placement->gpu_weight_bytes += ggml_nbytes(tensor);
      if (name.rfind("blk.", 0) == 0) {
        const auto end = name.find('.', 4);
        if (end != std::string::npos) {
          try {
            gpu_layers.insert(std::stoi(name.substr(4, end - 4)));
          } catch (const std::exception &) {
          }
        }
      }
    } else {
      placement->cpu_weight_bytes += ggml_nbytes(tensor);
    }
  }
  placement->gpu_layer_count = static_cast<int>(gpu_layers.size());
  if (expected) {
    if (!placement->gpu_weight_bytes) {
      *error = "placement_mismatch: no weight buffers reside on requested GPU";
      return false;
    }
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(expected, &props);
    placement->name = props.description ? props.description : "";
    placement->stable_id = props.device_id ? props.device_id : "";
    const auto selector = ParseDeviceSelector(placement->requested);
    placement->vendor = selector->vendor;
    placement->effective =
        selector->vendor + ":" + std::to_string(selector->ordinal);
    placement->state = "verified_weights";
  }
  return true;
}
} // namespace inferflux
