#pragma once

#include "runtime/backends/backend_capabilities.h"

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

class BackendInterface;
class LlamaCppBackend;

// ModelInfo describes a loaded model for routing and management.
struct ModelInfo {
  std::string id; // Unique model identifier (e.g., "llama3-8b-q4").
  // Requested source path from config/admin API/registry.
  // Kept as legacy field for compatibility with existing callers.
  std::string path;
  // Explicit source path alias (same semantics as path).
  std::string source_path;
  // Effective path used by the backend loader (may differ from source_path
  // when resolving hf:// references or GGUF sidecar fallbacks).
  std::string effective_load_path;
  std::string format{"unknown"}; // Resolved model format (gguf/safetensors/hf).
  std::string requested_format{"auto"}; // Requested format hint (or auto).
  std::string backend;                  // Exposed backend id ("inferflux_cuda",
                                        // "llama_cpp_cuda", "cpu", ...).
  std::string requested_backend;        // Requested backend hint ("cuda",
                                        // "inferflux_cuda", "llama_cpp_cuda",
                                        // "auto"...).
  std::string backend_provider{"llama_cpp"}; // "inferflux" or "llama_cpp".
  bool backend_fallback{false};        // True when requested backend fell back.
  std::string backend_fallback_reason; // Optional fallback explanation.
  bool ready{false}; // True when the model is loaded and serving.
  BackendCapabilities capabilities{};
  // Legacy alias kept for compatibility with older checks/callers.
  bool supports_structured_output{false};

  // MoE metadata (§2.6).  Populated from GGUF metadata after model load.
  // is_moe is true when the GGUF "llm.expert_count" key is present and > 0.
  bool is_moe{false};
  int n_experts{0};        // Total expert count (llm.expert_count).
  int n_active_experts{0}; // Active experts per token (llm.expert_used_count).
};

// ModelRouter is the plugin interface for multi-model serving.
// It decouples the HTTP layer from the physical model topology, enabling
// future features like model sharding, A/B serving, and per-tenant routing.
//
// The default implementation wraps a single LlamaCppBackend. Future backends
// (CUDA, ROCm, disaggregated prefill/decode) implement this same interface.
//
// Thread safety: all methods must be safe to call concurrently.
class ModelRouter {
public:
  virtual ~ModelRouter() = default;

  // List all models known to the router (loaded and unloaded).
  virtual std::vector<ModelInfo> ListModels() const = 0;

  // Load a model from the given path, returning its assigned ID.
  // The optional requested_id is treated as a hint; routers may adjust it to
  // ensure uniqueness. Returns empty string on failure.
  virtual std::string LoadModel(const std::string &path,
                                const std::string &backend_hint = "",
                                const std::string &requested_id = "",
                                const std::string &model_format = "auto") = 0;

  // Optional diagnostics for the last LoadModel failure. Empty on success or
  // when unavailable in a specific router implementation.
  virtual std::string LastLoadError() const { return ""; }

  // Unload a model by ID. Returns false if the model was not found.
  virtual bool UnloadModel(const std::string &id) = 0;

  // Resolve which model should handle a request. The caller provides
  // the model ID from the API request (may be empty for default routing).
  // Returns nullptr if no suitable model is available.
  virtual ModelInfo *Resolve(const std::string &requested_model) = 0;

  // Resolve an explicit model ID without applying default fallback behavior.
  // Returns nullptr when the exact model ID is unknown.
  virtual ModelInfo *ResolveExact(const std::string &model_id) = 0;

  // Retrieve the backend instance associated with a resolved model ID.
  virtual std::shared_ptr<BackendInterface>
  GetBackend(const std::string &model_id) = 0;

  // Set or query the default routing target.
  virtual bool SetDefaultModel(const std::string &model_id) = 0;
  virtual std::string DefaultModelId() const = 0;

  // Identity — useful for logging and diagnostics.
  virtual std::string Name() const = 0;
};

} // namespace inferflux
