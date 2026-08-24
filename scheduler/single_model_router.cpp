#include "scheduler/single_model_router.h"

#include "model/model_format.h"
#include "runtime/backends/backend_factory.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <algorithm>
#include <filesystem>
#include <vector>

namespace inferflux {

namespace {

bool IsMlxBackendLabel(const std::string &label) {
  return BackendFactory::NormalizeHint(label) == "mlx";
}

bool BackendSupportsModelFormat(const BackendFactoryResult &selection,
                                const std::string &format) {
  const std::string normalized = NormalizeModelFormat(format);
  if (normalized == "gguf") {
    return true;
  }
  if (normalized == "hf" || normalized == "safetensors") {
    // MLX backend supports HF/safetensors natively
    if (IsMlxBackendLabel(selection.backend_label)) {
      return true;
    }
    // InferFlux CUDA backend supports safetensors
    if (selection.provider == BackendProvider::kNative) {
      return normalized == "safetensors";
    }
  }
  return false;
}

BackendCapabilities
BuildModelCapabilities(const BackendCapabilities &base,
                       BackendProvider provider,
                       const std::shared_ptr<BackendInterface> &backend) {
  BackendCapabilities caps = base;
  caps.supports_vision = backend && backend->SupportsVision();

  // When the backend reports its own capabilities (native or any future
  // standalone backend), prefer its self-reported surface over the trait
  // defaults. Detect native backends by checking whether Name() starts
  // with "inferflux_" — a vanilla LlamaCppBackend registered under kNative
  // (test/mock scenario) gets conservative defaults instead.
  if (provider == BackendProvider::kNative) {
    if (backend) {
      const std::string name = backend->Name();
      bool is_genuine_native =
          name.size() >= 10 && name.compare(0, 10, "inferflux_") == 0;
      if (is_genuine_native) {
        auto reported = backend->ReportCapabilities();
        caps.supports_logprobs = reported.supports_logprobs;
        caps.supports_structured_output = reported.supports_structured_output;
        caps.supports_embeddings = reported.supports_embeddings;
        caps.supports_speculative_decoding =
            reported.supports_speculative_decoding;
        caps.supports_kv_prefix_transfer = reported.supports_kv_prefix_transfer;
        caps.supports_vision = reported.supports_vision;
      } else {
        // Conservative default for test/mock registrations under kNative.
        caps.supports_logprobs = false;
        caps.supports_structured_output = false;
        caps.supports_embeddings = false;
        caps.supports_speculative_decoding = false;
        caps.supports_kv_prefix_transfer = true;
      }
    } else {
      caps.supports_logprobs = false;
      caps.supports_structured_output = false;
      caps.supports_embeddings = false;
      caps.supports_speculative_decoding = false;
      caps.supports_kv_prefix_transfer = true;
    }
  }

  return caps;
}

void AppendUniqueHint(std::vector<std::string> *candidates,
                      const std::string &hint) {
  if (!candidates) {
    return;
  }
  const std::string normalized = BackendFactory::NormalizeHint(hint);
  if (normalized.empty()) {
    return;
  }
  if (std::find(candidates->begin(), candidates->end(), normalized) ==
      candidates->end()) {
    candidates->push_back(normalized);
  }
}

std::vector<std::string> BuildCudaFallbackCandidates() {
  std::vector<std::string> candidates;
  candidates.reserve(6);
  // Requested chain: InferFlux CUDA -> llama.cpp CUDA -> llama.cpp ROCm -> MLX
  // -> llama.cpp MPS -> CPU.
  AppendUniqueHint(&candidates, "cuda");
  AppendUniqueHint(&candidates, "llama_cpp_cuda");
#ifdef INFERFLUX_HAS_ROCM
  AppendUniqueHint(&candidates, "rocm");
#endif
#ifdef INFERFLUX_HAS_MLX
  AppendUniqueHint(&candidates, "mlx");
#endif
#ifdef INFERFLUX_HAS_METAL
  AppendUniqueHint(&candidates, "mps");
#endif
  AppendUniqueHint(&candidates, "cpu");
  return candidates;
}

void MaybePrependMlxCandidate(std::vector<std::string> *candidates,
                              const std::string &backend_hint,
                              const std::string &resolved_format) {
#if INFERFLUX_HAS_MLX
  const std::string normalized_format = NormalizeModelFormat(resolved_format);
  if (normalized_format != "hf" && normalized_format != "safetensors") {
    return;
  }
  std::string normalized_hint =
      backend_hint.empty() ? "auto"
                           : BackendFactory::NormalizeHint(backend_hint);
  if (normalized_hint != "auto" && normalized_hint != "mps" &&
      normalized_hint != "mlx") {
    return;
  }
  if (std::find(candidates->begin(), candidates->end(), "mlx") ==
      candidates->end()) {
    candidates->insert(candidates->begin(), "mlx");
  }
#else
  (void)candidates;
  (void)backend_hint;
  (void)resolved_format;
#endif
}

} // namespace

SingleModelRouter::SingleModelRouter(std::shared_ptr<BackendInterface> backend,
                                     ModelInfo info) {
  RegisterModel(info, std::move(backend));
}

SingleModelRouter::SingleModelRouter() = default;

SingleModelRouter::SingleModelRouter(
    const LlamaBackendConfig &default_backend_config,
    const std::string &default_backend_hint,
    const std::vector<std::string> &backend_priority)
    : default_backend_config_(default_backend_config),
      default_backend_hint_(BackendFactory::NormalizeHint(
          default_backend_hint.empty() ? "cpu" : default_backend_hint)),
      backend_priority_(BackendFactory::NormalizeHintList(
          backend_priority, default_backend_hint_)) {}

bool SingleModelRouter::RegisterModel(
    const ModelInfo &info, std::shared_ptr<BackendInterface> backend) {
  if (!backend) {
    return false;
  }
  if (info.path.empty() && info.id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Entry entry;
  entry.info = info;
  entry.info.path = info.path;
  entry.info.source_path =
      info.source_path.empty() ? info.path : info.source_path;
  if (entry.info.source_path.empty()) {
    entry.info.source_path = entry.info.path;
  }
  entry.info.path = entry.info.source_path;
  entry.info.effective_load_path = info.effective_load_path.empty()
                                       ? entry.info.path
                                       : info.effective_load_path;
  entry.info.requested_format = NormalizeModelFormat(info.requested_format);
  if (entry.info.requested_format.empty()) {
    entry.info.requested_format = "auto";
  }
  entry.info.format = NormalizeModelFormat(info.format);
  if (entry.info.format.empty() || entry.info.format == "auto") {
    entry.info.format =
        ResolveModelFormat(entry.info.source_path, entry.info.requested_format);
  }
  entry.info.backend = BackendFactory::NormalizeHint(
      info.backend.empty() ? "cpu" : info.backend);
  entry.info.requested_backend =
      info.requested_backend.empty()
          ? entry.info.backend
          : BackendFactory::NormalizeHint(info.requested_backend);
  entry.info.backend_provider =
      info.backend_provider.empty()
          ? BackendFactory::ProviderLabel(BackendProvider::kLlamaCpp)
          : BackendFactory::ProviderLabel(
                BackendFactory::ParseProviderLabel(info.backend_provider));
  entry.info.backend_fallback = info.backend_fallback;
  entry.info.backend_fallback_reason = info.backend_fallback_reason;
  auto register_target = ParseLlamaBackendTarget(entry.info.backend);
  const BackendProvider register_provider =
      BackendFactory::ParseProviderLabel(entry.info.backend_provider);
  entry.info.capabilities = BuildModelCapabilities(
      DescribeLlamaBackendTarget(register_target).capabilities,
      register_provider, backend);
  entry.info.supports_structured_output =
      entry.info.capabilities.supports_structured_output;
  entry.info.is_moe = backend->IsMoE();
  entry.info.n_experts = backend->ExpertCount();
  entry.info.n_active_experts = backend->ActiveExperts();
  entry.info.id = info.id.empty()
                      ? GenerateModelIdLocked(entry.info.source_path.empty()
                                                  ? "model"
                                                  : entry.info.source_path)
                      : EnsureUniqueIdLocked(info.id);
  entry.info.ready = backend->IsReady();
  entry.backend = std::move(backend);
  entry.load_time = std::chrono::steady_clock::now();
  models_[entry.info.id] = entry;
  if (default_model_id_.empty()) {
    default_model_id_ = entry.info.id;
  }
  last_load_error_.clear();
  GlobalMetrics().RecordModelLoad(entry.info.id, entry.info.backend, 0.0);
  GlobalMetrics().RecordBackendExposure(
      entry.info.requested_backend, entry.info.backend,
      entry.info.backend_provider, entry.info.backend_fallback);
  RecordModelReadyLocked(entry, entry.info.ready);
  return true;
}

std::vector<ModelInfo> SingleModelRouter::ListModels() const {
  std::vector<ModelInfo> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(models_.size());
  for (const auto &[id, entry] : models_) {
    ModelInfo copy = entry.info;
    copy.ready = entry.backend && entry.backend->IsReady();
    out.push_back(copy);
  }
  return out;
}

std::string SingleModelRouter::LoadModel(const std::string &path,
                                         const std::string &backend_hint,
                                         const std::string &requested_id,
                                         const std::string &model_format) {
  const auto set_last_load_error = [&](const std::string &error) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_load_error_ = error;
  };
  set_last_load_error("");

  std::string normalized_requested_format = NormalizeModelFormat(model_format);
  if (normalized_requested_format.empty()) {
    normalized_requested_format = model_format.empty() ? "auto" : "";
  }
  if (normalized_requested_format.empty()) {
    const std::string error =
        "invalid model format '" + model_format + "' for path '" + path + "'";
    log::Error("single_model_router", error);
    set_last_load_error(error);
    return "";
  }
  const std::string resolved_format =
      ResolveModelFormat(path, normalized_requested_format);
  if (resolved_format == "unknown") {
    const std::string error =
        "failed to resolve model format for path '" + path + "'";
    log::Error("single_model_router", error);
    set_last_load_error(error);
    return "";
  }

  auto backend_candidates = BuildBackendCandidates(backend_hint);
  MaybePrependMlxCandidate(&backend_candidates, backend_hint, resolved_format);
  BackendFactoryResult selection;
  std::string selected_requested_backend;
  std::string selected_path = path;
  std::string selected_format = resolved_format;
  bool selected_native_executor_fallback = false;
  std::string selected_native_executor_fallback_reason;
  std::string failure_reason;

  auto start = std::chrono::steady_clock::now();
  for (const auto &candidate : backend_candidates) {
    auto candidate_selection = BackendFactory::Create(candidate);
    selected_requested_backend = candidate;
    if (!candidate_selection.backend) {
      if (!candidate_selection.fallback_reason.empty()) {
        failure_reason = candidate_selection.fallback_reason;
      } else {
        failure_reason =
            "backend candidate '" + candidate + "' was not available";
      }
      continue;
    }

    std::string candidate_path = path;
    std::string candidate_format = resolved_format;

    struct LoadAttempt {
      std::string path;
      std::string format;
    };
    std::vector<LoadAttempt> load_attempts;
    load_attempts.push_back({candidate_path, candidate_format});

    if (candidate_format == "hf" || candidate_format == "safetensors") {
      if (IsMlxBackendLabel(candidate_selection.backend_label)) {
        load_attempts[0].path = ResolveMlxLoadPath(path, candidate_format);
        const auto gguf_fallback = ResolveLlamaLoadPath(path, candidate_format);
        if (!gguf_fallback.empty() && gguf_fallback != load_attempts[0].path) {
          load_attempts.push_back({gguf_fallback, "gguf"});
        }
      } else if (candidate_selection.provider == BackendProvider::kNative &&
                 candidate_format == "safetensors") {
        // InferFlux CUDA backend uses safetensors directly
        load_attempts[0].path = ResolveHfReferenceToCachePath(path);
      } else {
        const auto gguf_fallback = ResolveLlamaLoadPath(path, candidate_format);
        if (!gguf_fallback.empty()) {
          load_attempts.clear();
          load_attempts.push_back({gguf_fallback, "gguf"});
        }
      }
    }

    auto cfg = MergeBackendConfig(default_backend_config_, candidate_selection);
    bool loaded = false;
    for (const auto &attempt : load_attempts) {
      if (!BackendSupportsModelFormat(candidate_selection, attempt.format)) {
        failure_reason = "backend candidate '" + candidate +
                         "' does not support "
                         "model format '" +
                         attempt.format + "' for path '" + path +
                         "'. Configure backend=mlx or provide a GGUF artifact.";
        continue;
      }
      if (!candidate_selection.backend->LoadModel(attempt.path, cfg)) {
        if (candidate_selection.require_strict_inferflux_execution &&
            candidate_selection.provider == BackendProvider::kNative) {
          if (candidate_selection.backend->IsFallback()) {
            failure_reason =
                "backend_policy_violation: strict_inferflux_request enabled; "
                "inferflux backend rejected fallback runtime";
            if (!candidate_selection.backend->FallbackReason().empty()) {
              failure_reason +=
                  " (" + candidate_selection.backend->FallbackReason() + ")";
            }
            continue;
          }
        }
        failure_reason = "backend candidate '" + candidate +
                         "' failed to load model from path '" + attempt.path +
                         "'";
        continue;
      }
      selected_path = attempt.path;
      selected_format = attempt.format;
      loaded = true;
      break;
    }
    if (!loaded) {
      continue;
    }
    bool native_executor_fallback = false;
    std::string native_executor_fallback_reason;
    if (candidate_selection.provider == BackendProvider::kNative) {
      if (candidate_selection.backend->IsFallback()) {
        native_executor_fallback = true;
        native_executor_fallback_reason =
            candidate_selection.backend->FallbackReason();
      }
    }
    if (candidate_selection.require_strict_inferflux_execution &&
        (candidate_selection.used_fallback || native_executor_fallback)) {
      failure_reason = "backend policy violation: strict inferflux execution "
                       "required for '" +
                       candidate + "'";
      if (!candidate_selection.fallback_reason.empty()) {
        failure_reason += " (" + candidate_selection.fallback_reason + ")";
      } else if (!native_executor_fallback_reason.empty()) {
        failure_reason += " (" + native_executor_fallback_reason + ")";
      }
      continue;
    }
    selected_native_executor_fallback = native_executor_fallback;
    selected_native_executor_fallback_reason = native_executor_fallback_reason;
    selection = std::move(candidate_selection);
    break;
  }

  if (!selection.backend) {
    if (!failure_reason.empty()) {
      log::Error("single_model_router", failure_reason);
    }
    set_last_load_error(failure_reason.empty() ? "model load failed"
                                               : failure_reason);
    return "";
  }

  auto load_finished = std::chrono::steady_clock::now();
  double load_seconds =
      std::chrono::duration<double>(load_finished - start).count();

  ModelInfo info;
  info.path = path;
  info.source_path = path;
  info.effective_load_path = selected_path;
  info.requested_format = normalized_requested_format;
  info.format = selected_format;
  info.backend = selection.backend_label;
  info.requested_backend = selected_requested_backend.empty()
                               ? selection.backend_label
                               : selected_requested_backend;
  info.backend_provider = BackendFactory::ProviderLabel(selection.provider);
  info.backend_fallback =
      selection.used_fallback || selected_native_executor_fallback;
  info.backend_fallback_reason = selection.fallback_reason;
  if (info.backend_fallback_reason.empty() &&
      !selected_native_executor_fallback_reason.empty()) {
    info.backend_fallback_reason = selected_native_executor_fallback_reason;
  }
  info.ready = selection.backend->IsReady();
  info.capabilities = BuildModelCapabilities(
      selection.capabilities, selection.provider, selection.backend);
  info.supports_structured_output =
      info.capabilities.supports_structured_output;
  info.is_moe = selection.backend->IsMoE();
  info.n_experts = selection.backend->ExpertCount();
  info.n_active_experts = selection.backend->ActiveExperts();
  if (selected_path != path || selected_format != resolved_format) {
    log::Info("single_model_router", "Resolved model source for '" + path +
                                         "': effective_path='" + selected_path +
                                         "', effective_format='" +
                                         selected_format + "'");
  }
  if (info.backend_fallback) {
    log::Warn("single_model_router",
              "Backend fallback for model '" + path + "': requested=" +
                  info.requested_backend + ", selected=" + info.backend +
                  (info.backend_fallback_reason.empty()
                       ? ""
                       : ", reason=" + info.backend_fallback_reason));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::string preferred = !requested_id.empty()
                              ? requested_id
                              : std::filesystem::path(path).stem().string();
  info.id = EnsureUniqueIdLocked(preferred);

  Entry entry;
  entry.info = info;
  entry.backend = std::move(selection.backend);
  entry.load_time = load_finished;
  models_[info.id] = entry;
  if (default_model_id_.empty()) {
    default_model_id_ = info.id;
  }
  last_load_error_.clear();
  GlobalMetrics().RecordModelLoad(info.id, info.backend, load_seconds);
  GlobalMetrics().RecordBackendExposure(info.requested_backend, info.backend,
                                        info.backend_provider,
                                        info.backend_fallback);
  RecordModelReadyLocked(entry, entry.info.ready);
  return info.id;
}

bool SingleModelRouter::UnloadModel(const std::string &id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(id);
  if (it == models_.end()) {
    return false;
  }
  RecordModelReadyLocked(it->second, false);
  models_.erase(it);
  if (default_model_id_ == id) {
    default_model_id_.clear();
    UpdateDefaultModelLocked();
  }
  return true;
}

ModelInfo *SingleModelRouter::Resolve(const std::string &requested_model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.empty()) {
    return nullptr;
  }
  std::string target =
      requested_model.empty() ? default_model_id_ : requested_model;
  auto it = models_.find(target);
  if (it == models_.end()) {
    if (!requested_model.empty() && !default_model_id_.empty()) {
      it = models_.find(default_model_id_);
      if (it == models_.end()) {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }
  bool ready = it->second.backend && it->second.backend->IsReady();
  it->second.info.ready = ready;
  RecordModelReadyLocked(it->second, ready);
  return &it->second.info;
}

ModelInfo *SingleModelRouter::ResolveExact(const std::string &model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (model_id.empty()) {
    return nullptr;
  }
  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return nullptr;
  }
  bool ready = it->second.backend && it->second.backend->IsReady();
  it->second.info.ready = ready;
  RecordModelReadyLocked(it->second, ready);
  return &it->second.info;
}

std::shared_ptr<BackendInterface>
SingleModelRouter::GetBackend(const std::string &model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.backend;
}

bool SingleModelRouter::SetDefaultModel(const std::string &model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.find(model_id) == models_.end()) {
    return false;
  }
  default_model_id_ = model_id;
  return true;
}

std::string SingleModelRouter::LastLoadError() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_load_error_;
}

std::string SingleModelRouter::DefaultModelId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return default_model_id_;
}

std::shared_ptr<BackendInterface> SingleModelRouter::Backend() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (default_model_id_.empty()) {
    return nullptr;
  }
  auto it = models_.find(default_model_id_);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.backend;
}

std::string
SingleModelRouter::GenerateModelIdLocked(const std::string &path) const {
  std::string base = std::filesystem::path(path).stem().string();
  if (base.empty()) {
    base = "model";
  }
  return EnsureUniqueIdLocked(base);
}

std::string
SingleModelRouter::EnsureUniqueIdLocked(const std::string &preferred) const {
  std::string base = preferred.empty() ? "model" : preferred;
  std::string candidate = base;
  int suffix = 1;
  while (models_.find(candidate) != models_.end()) {
    candidate = base + "-" + std::to_string(suffix++);
  }
  return candidate;
}

void SingleModelRouter::RecordModelReadyLocked(const Entry &entry,
                                               bool ready) const {
  GlobalMetrics().RecordModelReady(entry.info.id, entry.info.backend, ready);
}

void SingleModelRouter::UpdateDefaultModelLocked() {
  if (!default_model_id_.empty()) {
    return;
  }
  if (models_.empty()) {
    return;
  }
  default_model_id_ = models_.begin()->first;
}

std::vector<std::string> SingleModelRouter::BuildBackendCandidates(
    const std::string &backend_hint) const {
  if (!backend_hint.empty()) {
    const std::string normalized = BackendFactory::NormalizeHint(backend_hint);
    if (normalized != "auto") {
      if (normalized == "cuda") {
        return BuildCudaFallbackCandidates();
      }
      return {normalized};
    }
  }
  if (!backend_priority_.empty()) {
    return backend_priority_;
  }
  return {default_backend_hint_};
}

} // namespace inferflux
