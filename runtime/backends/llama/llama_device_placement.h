#pragma once
#include "runtime/backends/common/device_placement.h"
#include <ggml-backend.h>
#include <string>
struct llama_model;
namespace inferflux {
ggml_backend_dev_t ResolveLlamaDevice(const std::string &selector,
                                      std::string *error);
bool ObserveLlamaWeightPlacement(const llama_model *model,
                                 ggml_backend_dev_t expected,
                                 DevicePlacement *placement,
                                 std::string *error);
} // namespace inferflux
