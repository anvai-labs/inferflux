// kernel_count_probe: Runs a single forward pass on a given backend and reports
// the number of CUDA kernel launches observed via the CUPTI activity API.
//
// Usage:
//   kernel_count_probe --backend <inferflux_cuda|llama_cpp_cuda>
//                      --model <path> --prompt <text>
//                      [--max-tokens N] [--ctx-size N] [--gpu-layers N]
//
// Output: JSON with kernel_launches count and per-phase timing (when available).

#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/cuda/inferflux_cuda_backend.h"
#include "runtime/backends/llama/llama_cpp_backend.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime.h>
#ifdef INFERFLUX_HAS_CUPTI
#include <cupti.h>
#endif
#endif

namespace inferflux {
namespace {

using json = nlohmann::json;

struct Options {
  std::string backend;
  std::filesystem::path model_path;
  std::optional<std::string> prompt;
  int max_tokens{4};
  int ctx_size{4096};
  int batch_size{512};
  int gpu_layers{99};
  bool use_flash_attention{true};
  int max_parallel_sequences{16};
};

void PrintUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --backend <inferflux_cuda|llama_cpp_cuda>"
               " --model <path> --prompt <text>"
               " [--max-tokens N] [--ctx-size N] [--gpu-layers N]\n";
}

bool ParseArgs(int argc, char **argv, Options *out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--backend") {
      const char *v = require_value("--backend");
      if (!v) return false;
      out->backend = v;
    } else if (arg == "--model") {
      const char *v = require_value("--model");
      if (!v) return false;
      out->model_path = v;
    } else if (arg == "--prompt") {
      const char *v = require_value("--prompt");
      if (!v) return false;
      out->prompt = std::string(v);
    } else if (arg == "--max-tokens") {
      const char *v = require_value("--max-tokens");
      if (!v) return false;
      out->max_tokens = std::stoi(v);
    } else if (arg == "--ctx-size") {
      const char *v = require_value("--ctx-size");
      if (!v) return false;
      out->ctx_size = std::stoi(v);
    } else if (arg == "--gpu-layers") {
      const char *v = require_value("--gpu-layers");
      if (!v) return false;
      out->gpu_layers = std::stoi(v);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  if (out->backend.empty() || out->model_path.empty() || !out->prompt) {
    return false;
  }
  return true;
}

#if defined(INFERFLUX_HAS_CUDA) && defined(INFERFLUX_HAS_CUPTI)

// CUPTI activity-based kernel launch counter.
struct KernelCounter {
  std::atomic<int> count{0};
  CUpti_SubscriberHandle subscriber{nullptr};

  static void CUPTIAPI Callback(void *userdata, CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid,
                                const void *cbdata) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    auto *data = static_cast<const CUpti_CallbackData *>(cbdata);
    // Count on entry to cudaLaunchKernel
    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 &&
        data->callbackSite == CUPTI_API_ENTER) {
      auto *self = static_cast<KernelCounter *>(userdata);
      self->count.fetch_add(1, std::memory_order_relaxed);
    }
  }

  bool Start() {
    CUptiResult res = cuptiSubscribe(&subscriber, Callback, this);
    if (res != CUPTI_SUCCESS) {
      std::cerr << "cuptiSubscribe failed: " << res << "\n";
      return false;
    }
    res = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                              CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
    if (res != CUPTI_SUCCESS) {
      std::cerr << "cuptiEnableCallback failed: " << res << "\n";
      cuptiUnsubscribe(subscriber);
      return false;
    }
    count.store(0, std::memory_order_relaxed);
    return true;
  }

  int Stop() {
    if (subscriber) {
      cuptiUnsubscribe(subscriber);
      subscriber = nullptr;
    }
    return count.load(std::memory_order_relaxed);
  }
};

#else // !INFERFLUX_HAS_CUPTI

struct KernelCounter {
  bool Start() {
    std::cerr << "CUPTI not available\n";
    return false;
  }
  int Stop() { return -1; }
};

#endif

std::unique_ptr<LlamaCppBackend> CreateBackend(const std::string &backend) {
  if (backend == "inferflux_cuda") {
    return std::make_unique<InferfluxCudaBackend>();
  }
  if (backend == "llama_cpp_cuda") {
    return std::make_unique<CudaBackend>();
  }
  return nullptr;
}

int Run(const Options &options) {
  auto backend = CreateBackend(options.backend);
  if (!backend) {
    std::cout << json{{"ok", false},
                      {"error", "unsupported backend: " + options.backend}}
                     .dump(2)
              << std::endl;
    return 2;
  }

  // Enable phase timing for native backends
#ifdef _WIN32
  _putenv_s("INFERFLUX_CUDA_PHASE_TIMING", "1");
#else
  setenv("INFERFLUX_CUDA_PHASE_TIMING", "1", 1);
#endif
  if (options.backend == "inferflux_cuda") {
#ifdef _WIN32
    _putenv_s("INFERFLUX_CUDA_STRICT", "1");
    _putenv_s("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1");
#else
    setenv("INFERFLUX_CUDA_STRICT", "1", 1);
    setenv("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1", 1);
#endif
  }

  LlamaBackendConfig config;
  config.ctx_size = options.ctx_size;
  config.batch_size = options.batch_size;
  config.gpu_layers = options.gpu_layers;
  config.use_flash_attention = options.use_flash_attention;
  config.max_parallel_sequences = options.max_parallel_sequences;

  if (!backend->LoadModel(options.model_path, config)) {
    std::cout << json{{"ok", false}, {"error", "LoadModel failed"}}.dump(2)
              << std::endl;
    return 3;
  }

  SamplingParams sampling;
  sampling.temperature = 0.0f;
  sampling.top_k = 1;
  sampling.seed = 0;
  backend->SetupSampler("", "", sampling);

  // Start CUPTI kernel counting
  KernelCounter counter;
  bool cupti_ok = counter.Start();

  std::vector<TokenLogprob> logprobs;
  const std::string output = backend->Generate(
      *options.prompt, options.max_tokens, {}, {}, 0, &logprobs, {});

  int kernel_launches = counter.Stop();

  backend->TeardownSampler();

  json payload = {
      {"ok", true},
      {"backend", options.backend},
      {"model_path", options.model_path.string()},
      {"prompt_chars", options.prompt->size()},
      {"output", output},
      {"token_count", static_cast<int>(logprobs.size())},
      {"kernel_launches", kernel_launches},
      {"cupti_available", cupti_ok},
  };

  if (auto *native =
          dynamic_cast<InferfluxCudaBackend *>(backend.get())) {
    payload["executor_kind"] = native->ExecutorKind();
    payload["is_fallback"] = native->IsFallbackExecutor();
  }

  std::cout << payload.dump(2) << std::endl;
  return 0;
}

} // namespace
} // namespace inferflux

int main(int argc, char **argv) {
  inferflux::Options options;
  if (!inferflux::ParseArgs(argc, argv, &options)) {
    inferflux::PrintUsage(argv[0]);
    return 2;
  }
  try {
    return inferflux::Run(options);
  } catch (const std::exception &ex) {
    std::cout << nlohmann::json{{"ok", false}, {"error", ex.what()}}.dump(2)
              << std::endl;
    return 1;
  } catch (...) {
    std::cout << nlohmann::json{{"ok", false}, {"error", "unknown exception"}}
                     .dump(2)
              << std::endl;
    return 1;
  }
}
