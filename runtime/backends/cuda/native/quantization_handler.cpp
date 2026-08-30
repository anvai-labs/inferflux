#include "runtime/backends/cuda/native/quantization_handler.h"

// sizeof_half on CPU builds: the guarded header typedef makes __half
// opaque, so use the standard IEEE binary16 size directly.
#ifndef INFERFLUX_HAS_CUDA
#define sizeof_half 2
#else
#define sizeof_half sizeof(half)
#endif
#include "runtime/backends/cuda/native/safetensors_adapter.h"
#include "server/logging/logger.h"
#include <algorithm>

#ifdef INFERFLUX_HAS_CUDA
#include "runtime/backends/cuda/native/q4_k_m_handler.h"
#include "runtime/backends/cuda/native/q5_k_m_handler.h"
#include "runtime/backends/cuda/native/q6_k_handler.h"
#include "runtime/backends/cuda/native/q8_0_handler.h"
#include "runtime/backends/cuda/native/q8_k_handler.h"
#endif

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

namespace {

#ifdef INFERFLUX_HAS_CUDA
void EnsureCudaQuantizationHandlersRegistered(
    QuantizationHandlerRegistry *registry) {
  if (!registry)
    return;
  // Explicit registration — do not rely on static initializers in handler
  // translation units, which the linker may strip from the static library.
  registry->Register("q4_k_m",
                     []() { return std::make_shared<Q4_K_M_Handler>(); });
  registry->Register("q4_k",
                     []() { return std::make_shared<Q4_K_M_Handler>(); });
  registry->Register("q5_k_m",
                     []() { return std::make_shared<Q5_K_M_Handler>(); });
  registry->Register("q5_k",
                     []() { return std::make_shared<Q5_K_M_Handler>(); });
  registry->Register("q6_k", []() { return std::make_shared<Q6_K_Handler>(); });
  registry->Register("q6_k_m",
                     []() { return std::make_shared<Q6_K_Handler>(); });
  registry->Register("q8_0", []() { return std::make_shared<Q8_0_Handler>(); });
  registry->Register("q8_k", []() { return std::make_shared<Q8_K_Handler>(); });
  registry->Register("q8_k_m",
                     []() { return std::make_shared<Q8_K_Handler>(); });
}
#endif

#ifndef INFERFLUX_HAS_CUDA
class StaticQuantizationHandler final : public IQuantizationHandler {
public:
  StaticQuantizationHandler(std::string type, double bits_per_value,
                            size_t block_size, size_t block_bytes)
      : type_(std::move(type)), bits_per_value_(bits_per_value),
        block_size_(block_size), block_bytes_(block_bytes) {}

  void DequantizeGpuToGpu(const void *, half *, size_t, cudaStream_t) override {
    // CPU-only build: no CUDA kernels available.
  }

  std::string GetType() const override { return type_; }

  size_t GetDequantizedSize(size_t quantized_size) const override {
    if (block_bytes_ == 0) {
      return 0;
    }
    const size_t blocks = quantized_size / block_bytes_;
    return blocks * block_size_ * sizeof_half;
  }

  double GetBitsPerValue() const override { return bits_per_value_; }

private:
  std::string type_;
  double bits_per_value_{0.0};
  size_t block_size_{0};
  size_t block_bytes_{0};
};

void EnsureCpuQuantizationHandlersRegistered(
    QuantizationHandlerRegistry *registry) {
  if (!registry) {
    return;
  }

  registry->Register("q4_k_m", []() {
    return std::make_shared<StaticQuantizationHandler>("q4_k_m", 4.5, 256, 144);
  });
  registry->Register("q4_k", []() {
    return std::make_shared<StaticQuantizationHandler>("q4_k_m", 4.5, 256, 144);
  });
  registry->Register("q5_k_m", []() {
    return std::make_shared<StaticQuantizationHandler>("q5_k_m", 5.5, 256, 176);
  });
  registry->Register("q5_k", []() {
    return std::make_shared<StaticQuantizationHandler>("q5_k_m", 5.5, 256, 176);
  });
  registry->Register("q6_k", []() {
    return std::make_shared<StaticQuantizationHandler>("q6_k", 6.5625, 256,
                                                       210);
  });
  registry->Register("q6_k_m", []() {
    return std::make_shared<StaticQuantizationHandler>("q6_k", 6.5625, 256,
                                                       210);
  });
  registry->Register("q8_0", []() {
    return std::make_shared<StaticQuantizationHandler>("q8_0", 8.5, 32, 34);
  });
  registry->Register("q8_k", []() {
    return std::make_shared<StaticQuantizationHandler>("q8_k", 9.125, 256, 292);
  });
  registry->Register("q8_k_m", []() {
    return std::make_shared<StaticQuantizationHandler>("q8_k", 9.125, 256, 292);
  });
}
#endif

} // namespace

//==============================================================================
// QuantizationHandlerRegistry Implementation
//==============================================================================

QuantizationHandlerRegistry &QuantizationHandlerRegistry::Instance() {
  static QuantizationHandlerRegistry instance;
  static const bool kRegistered = []() {
#ifdef INFERFLUX_HAS_CUDA
    EnsureCudaQuantizationHandlersRegistered(&instance);
#else
    EnsureCpuQuantizationHandlersRegistered(&instance);
#endif
    return true;
  }();
  (void)kRegistered;
  return instance;
}

void QuantizationHandlerRegistry::Register(const std::string &type,
                                           FactoryFunc factory) {
  factories_[type] = std::move(factory);
  log::Debug("quantization_registry",
             "Registered quantization handler: " + type);
}

std::shared_ptr<IQuantizationHandler>
QuantizationHandlerRegistry::Create(const std::string &type) const {
  auto it = factories_.find(type);
  if (it == factories_.end()) {
    log::Warn("quantization_registry", "Unknown quantization type: " + type);
    return nullptr;
  }

  return it->second();
}

bool QuantizationHandlerRegistry::IsRegistered(const std::string &type) const {
  return factories_.find(type) != factories_.end();
}

std::vector<std::string>
QuantizationHandlerRegistry::GetRegisteredTypes() const {
  std::vector<std::string> types;
  types.reserve(factories_.size());

  for (const auto &pair : factories_) {
    types.push_back(pair.first);
  }

  std::sort(types.begin(), types.end());
  return types;
}

//==============================================================================
// Factory Functions
//==============================================================================

std::shared_ptr<IQuantizationHandler>
CreateQuantizationHandler(const std::string &quantization_type) {
  // Non-quantized models use the safetensors/null handler.
  if (quantization_type.empty() || quantization_type == "none") {
    return std::make_shared<SafetensorsQuantizationHandler>();
  }

  // Check registry first
  auto handler =
      QuantizationHandlerRegistry::Instance().Create(quantization_type);
  if (handler) {
    return handler;
  }

  log::Error("quantization_factory",
             "Unsupported quantization type: " + quantization_type);
  return nullptr;
}

//==============================================================================
// BaseQuantizationHandler Implementation
//==============================================================================

size_t BaseQuantizationHandler::GetBlockSize(const std::string &type) {
  // Block sizes for different quantization types (from ggml-common.h)
  static const std::unordered_map<std::string, size_t> block_sizes = {
      {"q4_0", 32},  {"q4_1", 32},    {"q5_0", 32},  {"q5_1", 32},
      {"q8_0", 32},  {"q8_1", 32},    {"q2_k", 256}, {"q3_k", 256},
      {"q4_k", 256}, {"q4_k_m", 256}, {"q5_k", 256}, {"q5_k_m", 256},
      {"q6_k", 256}, {"q6_k_m", 256}, {"q8_k", 256}, {"q8_k_m", 256},
  };

  auto it = block_sizes.find(type);
  if (it != block_sizes.end()) {
    return it->second;
  }

  log::Warn("quantization_handler",
            "Unknown block size for type: " + type + ", using 256");
  return 256;
}

size_t BaseQuantizationHandler::GetQuantizedSize(size_t num_elements,
                                                 const std::string &type) {
  size_t block_size = GetBlockSize(type);
  size_t num_blocks = (num_elements + block_size - 1) / block_size;

  // Block sizes in bytes (from ggml-common.h)
  static const std::unordered_map<std::string, size_t> block_bytes = {
      {"q4_0", 18}, // sizeof_half + 32/2 = 2 + 16 = 18
      {"q4_1", 20}, // 2*sizeof_half + 32/2 = 4 + 16 = 20
      {"q5_0", 22}, // sizeof_half + 4 + 32/2 = 2 + 4 + 16 = 22
      {"q5_1", 24}, // 2*sizeof_half + 4 + 32/2 = 4 + 4 + 16 = 24
      {"q8_0", 34}, // sizeof_half + 32 = 2 + 32 = 34
      {"q8_1", 36}, // 2*sizeof_half + 32 = 4 + 32 = 36
      {"q2_k", 96}, // 2*sizeof_half + 256/16 + 256/4 = 4 + 16 + 64 = 84
      {"q3_k",
       100}, // sizeof_half + 256/4 + 256/8 + 12 = 2 + 64 + 32 + 12 = 110
      {"q4_k", 144}, // 2*sizeof_half + 12 + 256/2 = 4 + 12 + 128 = 144
      {"q4_k_m", 144},
      {"q5_k",
       176}, // 2*sizeof_half + 12 + 256/2 + 256/8 = 4 + 12 + 128 + 32 = 176
      {"q5_k_m", 176},
      {"q6_k", 210}, // sizeof_half + 256/16 + 3*256/4 = 2 + 16 + 192 = 210
      {"q6_k_m", 210},
      {"q8_k", 292},   // sizeof(float) + 256 + 256/16*sizeof(int16_t)
      {"q8_k_m", 292}, // alias
  };

  auto it = block_bytes.find(type);
  if (it == block_bytes.end()) {
    log::Warn("quantization_handler", "Unknown quantization type: " + type);
    return num_elements * sizeof_half; // Assume FP16
  }

  return num_blocks * it->second;
}

bool BaseQuantizationHandler::ValidateInputs(const void *quantized,
                                             half *dequantized,
                                             size_t num_elements) const {
  if (!quantized) {
    log::Error("quantization_handler", "Null quantized input");
    return false;
  }

  if (!dequantized) {
    log::Error("quantization_handler", "Null dequantized output");
    return false;
  }

  if (num_elements == 0) {
    log::Error("quantization_handler", "Zero elements");
    return false;
  }

  return true;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
