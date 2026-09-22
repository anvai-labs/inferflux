#include "server/http/model_json.h"

#include "runtime/backends/backend_factory.h"
#include "runtime/backends/common/backend_interface.h"

using json = nlohmann::json;

namespace inferflux {

json BuildModelRuntimeJson(const ModelInfo &info,
                           const BackendInterface *backend,
                           bool prefix_cache_enabled,
                           bool session_handles_enabled) {
  const bool ready = backend && backend->IsReady();
  const bool reusable = ready && info.capabilities.supports_kv_prefix_transfer;
  auto capacity = [](int value) -> json {
    return value > 0 ? json(value) : json(nullptr);
  };
  return {
      {"sequence_capacity", capacity(ready ? backend->SequenceCapacity() : 0)},
      {"context_tokens_per_sequence",
       capacity(ready ? backend->SequenceContextCapacity() : 0)},
      {"session_handles_enabled", session_handles_enabled},
      {"cache_reuse",
       {{"phased",
         {{"prefix", reusable && prefix_cache_enabled},
          {"session", reusable && session_handles_enabled}}},
        {"full_generate",
         {{"prefix", false},
          {"session", false},
          {"reason", "sampler_grammar_continuity"}}}}}};
}

json BuildCapabilitiesJson(const BackendCapabilities &capabilities) {
  return json{
      {"generation", capabilities.supports_generation},
      {"streaming", capabilities.supports_streaming},
      {"logprobs", capabilities.supports_logprobs},
      {"structured_output", capabilities.supports_structured_output},
      {"embeddings", capabilities.supports_embeddings},
      {"vision", capabilities.supports_vision},
      {"speculative_decoding", capabilities.supports_speculative_decoding},
      {"fairness_preemption", capabilities.supports_fairness_preemption},
      {"kv_prefix_transfer", capabilities.supports_kv_prefix_transfer},
  };
}

std::string ModelSourcePath(const ModelInfo &info) {
  return info.source_path.empty() ? info.path : info.source_path;
}

std::string ModelEffectiveLoadPath(const ModelInfo &info) {
  const std::string source = ModelSourcePath(info);
  return info.effective_load_path.empty() ? source : info.effective_load_path;
}

json BuildBackendExposureJson(const ModelInfo &info) {
  const std::string requested =
      info.requested_backend.empty() ? info.backend : info.requested_backend;
  const std::string provider =
      info.backend_provider.empty()
          ? BackendFactory::ProviderLabel(BackendProvider::kLlamaCpp)
          : BackendFactory::ProviderLabel(
                BackendFactory::ParseProviderLabel(info.backend_provider));
  return json{
      {"requested_backend", requested},
      {"exposed_backend", info.backend},
      {"provider", provider},
      {"fallback", info.backend_fallback},
      {"fallback_reason", info.backend_fallback_reason},
  };
}

json BuildModelIdentityJson(const ModelInfo &info) {
  json model = {
      {"id", info.id},
      {"path", info.path},
      {"source_path", ModelSourcePath(info)},
      {"effective_load_path", ModelEffectiveLoadPath(info)},
      {"format", info.format},
      {"requested_format", info.requested_format},
      {"backend", info.backend},
      {"backend_exposure", BuildBackendExposureJson(info)},
      {"ready", info.ready},
      {"capabilities", BuildCapabilitiesJson(info.capabilities)},
  };
  const auto &p = info.placement;
  model["placement"] = {
      {"requested", p.requested},
      {"effective", p.effective},
      {"vendor", p.vendor},
      {"device_name", p.name},
      {"stable_id", p.stable_id.empty() ? json(nullptr) : json(p.stable_id)},
      {"state", p.state},
      {"gpu_weight_bytes", p.gpu_weight_bytes},
      {"cpu_weight_bytes", p.cpu_weight_bytes},
      {"gpu_layer_count", p.gpu_layer_count},
      {"requested_gpu_layers", p.requested_gpu_layers},
      {"context_size", p.context_size},
      {"max_parallel_sequences", p.max_parallel_sequences},
      {"kv_cache_type", p.kv_cache_type}};
  // GGUF metadata (Ollama-style model details).
  const auto &g = info.gguf;
  if (!g.architecture.empty() || g.parameter_count > 0) {
    model["details"] = {
        {"architecture", g.architecture},
        {"quantization", g.quantization},
        {"parameter_count", g.parameter_count},
        {"context_length", g.context_length},
        {"embedding_length", g.embedding_length},
        {"num_layers", g.num_layers},
        {"num_heads", g.num_heads},
        {"num_kv_heads", g.num_kv_heads},
    };
    if (!g.chat_template.empty()) {
      model["details"]["chat_template"] =
          g.chat_template.substr(0, 200); // Truncate for display
    }
  }
  return model;
}

json BuildOpenAIModelJson(const ModelInfo &info, int64_t created_ts) {
  json model = BuildModelIdentityJson(info);
  model["object"] = "model";
  model["created"] = created_ts;
  model["owned_by"] = "inferflux";
  return model;
}

json BuildAdminModelJson(const ModelInfo &info, const std::string &default_id) {
  json model = BuildModelIdentityJson(info);
  model["requested_backend"] =
      info.requested_backend.empty() ? info.backend : info.requested_backend;
  model["backend_provider"] =
      info.backend_provider.empty()
          ? BackendFactory::ProviderLabel(BackendProvider::kLlamaCpp)
          : BackendFactory::ProviderLabel(
                BackendFactory::ParseProviderLabel(info.backend_provider));
  model["default"] = (info.id == default_id);
  return model;
}

} // namespace inferflux
