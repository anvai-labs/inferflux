#pragma once

#include "model/model_load_spec.h"
#include "runtime/backends/backend_capabilities.h"
#include "runtime/backends/common/device_placement.h"

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
  DevicePlacement placement;
  // Legacy alias kept for compatibility with older checks/callers.
  bool supports_structured_output{false};

  // MoE metadata (§2.6).  Populated from GGUF metadata after model load.
  // is_moe is true when the GGUF "llm.expert_count" key is present and > 0.
  bool is_moe{false};
  int n_experts{0};        // Total expert count (llm.expert_count).
  int n_active_experts{0}; // Active experts per token (llm.expert_used_count).

  // GGUF model metadata (populated from GGUF KV during load).
  // Enables Ollama-style `show` with architecture, quantization, and context.
  struct GgufMetadata {
    std::string architecture;   // "qwen2", "llama", "gemma", etc.
    std::string quantization;   // "Q4_K_M", "Q6_K", "F16", etc.
    int64_t parameter_count{0}; // Total parameters (approx).
    int context_length{0};      // max_position_embeddings.
    int embedding_length{0};    // hidden_size.
    int num_layers{0};          // num_hidden_layers.
    int num_heads{0};           // num_attention_heads.
    int num_kv_heads{0};        // num_key_value_heads (GQA).
    std::string chat_template;  // Jinja2 template from metadata.
  };
  GgufMetadata gguf{};
};

// --- Segregated model router interfaces (Phase D2) ---
// ModelResolver provides read-only query operations (used by Scheduler).
// ModelLifecycle adds mutation operations (used by admin endpoints).
// ModelRouter = ModelLifecycle for backward compatibility.

/// Read-only model resolution and query interface.
/// Scheduler and inference paths depend only on this narrow interface,
/// enabling simpler test stubs.
class ModelResolver {
public:
  virtual ~ModelResolver() = default;

  virtual std::vector<ModelInfo> ListModels() const = 0;
  virtual ModelInfo *Resolve(const std::string &requested_model) = 0;
  virtual ModelInfo *ResolveExact(const std::string &model_id) = 0;
  virtual std::shared_ptr<BackendInterface>
  GetBackend(const std::string &model_id) = 0;
  virtual std::string DefaultModelId() const = 0;
  virtual std::string Name() const = 0;
};

/// Model lifecycle management (load/unload/set-default).
/// Extends ModelResolver with mutation operations.
class ModelLifecycle : public ModelResolver {
public:
  ~ModelLifecycle() override = default;

  virtual std::string LoadModel(const std::string &path,
                                const std::string &backend_hint = "",
                                const std::string &requested_id = "",
                                const std::string &model_format = "auto") = 0;
  virtual std::string LoadModel(const ModelLoadSpec &spec) {
    // Legacy/test implementations must never silently discard new fields.
    if (spec.HasOverrides())
      return "";
    return LoadModel(spec.path, spec.backend, spec.id, spec.format);
  }
  virtual void SetMinimumSequenceCapacity(int) {}
  virtual std::string LastLoadError() const { return ""; }
  virtual bool UnloadModel(const std::string &id) = 0;
  virtual bool SetDefaultModel(const std::string &model_id) = 0;
};

/// Full model router — backward compatibility alias for ModelLifecycle.
/// Existing implementations (SingleModelRouter) inherit ModelRouter unchanged.
using ModelRouter = ModelLifecycle;

} // namespace inferflux
