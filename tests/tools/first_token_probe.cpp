#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/cuda/inferflux_cuda_backend.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace inferflux {
namespace {

using json = nlohmann::json;

struct Options {
  std::string backend;
  std::filesystem::path model_path;
  std::optional<std::filesystem::path> prompt_file;
  std::optional<std::string> prompt;
  int max_tokens{1};
  int top_n{8};
  int ctx_size{4096};
  int batch_size{512};
  int gpu_layers{99};
  bool use_flash_attention{true};
  int max_parallel_sequences{16};
};

void PrintUsage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --backend <inferflux_cuda|llama_cpp_cuda>"
         " --model <path> [--prompt <text> | --prompt-file <path>]"
         " [--top-n <int>] [--max-tokens <int>]\n";
}

bool ParseBoolFlag(const std::string &value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "on";
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
      const char *value = require_value("--backend");
      if (!value) {
        return false;
      }
      out->backend = value;
    } else if (arg == "--model") {
      const char *value = require_value("--model");
      if (!value) {
        return false;
      }
      out->model_path = value;
    } else if (arg == "--prompt-file") {
      const char *value = require_value("--prompt-file");
      if (!value) {
        return false;
      }
      out->prompt_file = std::filesystem::path(value);
    } else if (arg == "--prompt") {
      const char *value = require_value("--prompt");
      if (!value) {
        return false;
      }
      out->prompt = std::string(value);
    } else if (arg == "--top-n") {
      const char *value = require_value("--top-n");
      if (!value) {
        return false;
      }
      out->top_n = std::stoi(value);
    } else if (arg == "--max-tokens") {
      const char *value = require_value("--max-tokens");
      if (!value) {
        return false;
      }
      out->max_tokens = std::stoi(value);
    } else if (arg == "--ctx-size") {
      const char *value = require_value("--ctx-size");
      if (!value) {
        return false;
      }
      out->ctx_size = std::stoi(value);
    } else if (arg == "--batch-size") {
      const char *value = require_value("--batch-size");
      if (!value) {
        return false;
      }
      out->batch_size = std::stoi(value);
    } else if (arg == "--gpu-layers") {
      const char *value = require_value("--gpu-layers");
      if (!value) {
        return false;
      }
      out->gpu_layers = std::stoi(value);
    } else if (arg == "--flash-attention") {
      const char *value = require_value("--flash-attention");
      if (!value) {
        return false;
      }
      out->use_flash_attention = ParseBoolFlag(value);
    } else if (arg == "--max-parallel-sequences") {
      const char *value = require_value("--max-parallel-sequences");
      if (!value) {
        return false;
      }
      out->max_parallel_sequences = std::stoi(value);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (out->backend.empty() || out->model_path.empty() ||
      (!out->prompt && !out->prompt_file) ||
      (out->prompt && out->prompt_file)) {
    return false;
  }
  return true;
}

bool ReadPrompt(const Options &options, std::string *prompt) {
  if (options.prompt) {
    *prompt = *options.prompt;
    return true;
  }
  std::ifstream in(*options.prompt_file);
  if (!in.is_open()) {
    return false;
  }
  prompt->assign(std::istreambuf_iterator<char>(in),
                 std::istreambuf_iterator<char>());
  return true;
}

json SerializeTokenLogprob(const TokenLogprob &entry) {
  json top = json::array();
  for (const auto &[token, logprob] : entry.top_logprobs) {
    top.push_back({{"token", token}, {"logprob", logprob}});
  }
  return {
      {"token", entry.token},
      {"logprob", entry.logprob},
      {"bytes", entry.bytes},
      {"top_logprobs", std::move(top)},
  };
}

json SerializeTopLogit(const TopLogitEntry &entry) {
  return {
      {"token", entry.token},
      {"logit", entry.logit},
  };
}

std::unique_ptr<LlamaCppBackend> CreateBackend(const std::string &backend) {
  if (backend == "inferflux_cuda") {
    return std::make_unique<InferfluxCudaBackend>();
  }
  if (backend == "llama_cpp_cuda") {
    return std::make_unique<CudaBackend>();
  }
  return nullptr;
}

json BuildError(const std::string &message) {
  return {{"ok", false}, {"error", message}};
}

int Run(const Options &options) {
  std::string prompt;
  if (!ReadPrompt(options, &prompt)) {
    std::cout << BuildError("failed to read prompt").dump(2) << std::endl;
    return 2;
  }

  auto backend = CreateBackend(options.backend);
  if (!backend) {
    std::cout << BuildError("unsupported backend: " + options.backend).dump(2)
              << std::endl;
    return 2;
  }

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

  SamplingParams sampling;
  sampling.temperature = 0.0f;
  sampling.top_k = 1;
  sampling.seed = 0;

  if (!backend->LoadModel(options.model_path, config)) {
    std::cout << BuildError("LoadModel failed").dump(2) << std::endl;
    return 3;
  }
  if (!backend->IsReady()) {
    std::cout << BuildError("backend not ready").dump(2) << std::endl;
    return 3;
  }

  backend->SetupSampler("", "", sampling);
  std::vector<TokenLogprob> logprobs;
  const std::string output =
      backend->Generate(prompt, options.max_tokens, {}, {}, options.top_n,
                        &logprobs, {});
  const std::vector<TopLogitEntry> top_logits =
      backend->TopLogitsForParity(options.top_n);
  backend->TeardownSampler();

  json payload = {
      {"ok", true},
      {"backend", options.backend},
      {"model_path", options.model_path.string()},
      {"prompt_chars", prompt.size()},
      {"output", output},
      {"token_count", static_cast<int>(logprobs.size())},
      {"logprobs", json::array()},
      {"top_logits", json::array()},
  };

  if (auto *inferflux_backend =
          dynamic_cast<InferfluxCudaBackend *>(backend.get())) {
    payload["executor_kind"] = inferflux_backend->ExecutorKind();
    payload["is_fallback"] = inferflux_backend->IsFallbackExecutor();
    payload["fallback_reason"] = inferflux_backend->FallbackReason();
  }

  for (const auto &entry : logprobs) {
    payload["logprobs"].push_back(SerializeTokenLogprob(entry));
  }
  for (const auto &entry : top_logits) {
    payload["top_logits"].push_back(SerializeTopLogit(entry));
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
    std::cout << inferflux::BuildError(ex.what()).dump(2) << std::endl;
    return 1;
  } catch (...) {
    std::cout << inferflux::BuildError("unknown exception").dump(2)
              << std::endl;
    return 1;
  }
}
