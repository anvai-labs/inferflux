#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/native/llama_forward.h"
#include "server/logging/logger.h"

#include <algorithm>
// CUDA headers when available; opaque typedefs otherwise (mirrors
// model_loader.h) so CPU-only CI builds compile this header.
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime_api.h>) && \
     __has_include(<cuda_fp16.h>))
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct __half;
typedef __half half;
#endif

namespace inferflux {

namespace {

bool IsLlamaFamily(const std::string &type) {
  return type == "llama" || type == "qwen2" || type == "qwen3" ||
         type == "mistral" || type == "gemma" || type == "gemma2" ||
         type == "phi3" || type == "internlm2" || type == "codellama" ||
         type == "yi";
}

std::string NormalizeType(const std::string &model_type) {
  std::string type = model_type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return type;
}

} // namespace

std::unique_ptr<ModelForward>
CreateModelForward(const std::string &model_type) {
  return CreateModelForwardTyped<half>(model_type);
}

template <typename T>
std::unique_ptr<ModelForward>
CreateModelForwardTyped(const std::string &model_type) {
  std::string type = NormalizeType(model_type);

  if (IsLlamaFamily(type)) {
    log::Info("model_forward_factory", "Creating LlamaForwardTyped<" +
                                           std::string(DtypeTraits<T>::name) +
                                           "> for model_type=" + model_type);
    return std::make_unique<LlamaForwardTyped<T>>();
  }

  log::Error("model_forward_factory", "Unsupported model_type: " + model_type);
  return nullptr;
}

// Explicit instantiations
template std::unique_ptr<ModelForward>
CreateModelForwardTyped<half>(const std::string &);
template std::unique_ptr<ModelForward>
CreateModelForwardTyped<__nv_bfloat16>(const std::string &);

} // namespace inferflux
