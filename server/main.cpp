#include "model/model_format.h"
#include "model/tokenizer/simple_tokenizer.h"
#include "policy/policy_store.h"
#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_manager.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/disaggregated/kv_channel.h"
#include "runtime/disaggregated/shm_kv_transport.h"
#include "runtime/execution/parallel_context.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/prefix_cache/prefix_cache.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "runtime/string_utils.h"
#include "scheduler/model_registry.h"
#include "scheduler/model_router.h"
#include "scheduler/model_selection.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/http/http_server.h"
#include "server/diagnostics/crash_handler.h"
#include "server/logging/audit_logger.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"
#include "server/startup_advisor.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>

struct ModelConfig {
  std::string id;
  std::string path;
  std::string format{"auto"};
  std::string backend;
  bool make_default{false};
};

namespace {
using inferflux::ParseBool;
using inferflux::ToLower;
using inferflux::Trim;

std::size_t ParsePositiveSize(const std::string &text) {
  try {
    long long value = std::stoll(text);
    if (value > 0) {
      return static_cast<std::size_t>(value);
    }
  } catch (...) {
  }
  return 0;
}

std::vector<ModelConfig> ParseModelsEnv(const std::string &raw) {
  std::vector<ModelConfig> entries;
  std::stringstream ss(raw);
  std::string segment;
  while (std::getline(ss, segment, ';')) {
    ModelConfig cfg;
    std::stringstream kv_stream(segment);
    std::string pair;
    while (std::getline(kv_stream, pair, ',')) {
      auto eq = pair.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      auto key = Trim(pair.substr(0, eq));
      auto value = Trim(pair.substr(eq + 1));
      if (key == "id") {
        cfg.id = value;
      } else if (key == "path") {
        cfg.path = value;
      } else if (key == "format") {
        auto normalized = inferflux::NormalizeModelFormat(value);
        if (!normalized.empty()) {
          cfg.format = normalized;
        } else {
          cfg.format = "auto";
          inferflux::log::Warn("server",
                               "Invalid model format in INFERFLUX_MODELS "
                               "entry; defaulting to auto",
                               value);
        }
      } else if (key == "backend") {
        cfg.backend = ToLower(value);
      } else if (key == "default") {
        cfg.make_default = ParseBool(value);
      }
    }
    if (!cfg.path.empty()) {
      entries.push_back(cfg);
    }
  }
  return entries;
}

std::vector<std::string> ParseBackendPriorityList(const std::string &raw) {
  std::vector<std::string> hints;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = ToLower(Trim(token));
    if (!token.empty()) {
      hints.push_back(token);
    }
  }
  return hints;
}

std::string JoinList(const std::vector<std::string> &items) {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      out.append(",");
    }
    out.append(items[i]);
  }
  return out;
}
} // namespace

static std::atomic<bool> g_running{true};

void SignalHandler(int) { g_running = false; }

int main(int argc, char **argv) {
  // Structured JSON logging: set before any log output.
  const bool log_format_from_env = std::getenv("INFERFLUX_LOG_FORMAT") != nullptr;
  if (const char *fmt = std::getenv("INFERFLUX_LOG_FORMAT")) {
    std::string s = fmt;
    inferflux::log::SetJsonMode(s == "json" || s == "JSON");
  }
  std::string configured_log_level = "info";
  bool log_level_from_env = false;
  if (const char *level = std::getenv("INFERFLUX_LOG_LEVEL")) {
    inferflux::log::Level parsed_level;
    if (inferflux::log::ParseLevel(level, &parsed_level)) {
      inferflux::log::SetLevel(parsed_level);
      configured_log_level = level;
      log_level_from_env = true;
    } else {
      std::cerr << "[WARN] server: Ignoring invalid INFERFLUX_LOG_LEVEL | "
                << level << "\n";
    }
  }

  // Install crash signal handlers (SIGSEGV, SIGABRT, SIGBUS, SIGFPE).
  // Writes diagnostic breadcrumbs to stderr before re-raising for core dump.
  inferflux::diagnostics::InstallCrashHandler();

  // §P1e: Initialize distributed parallel environment.
  int dist_rank = 0;
  int dist_world_size = 1;
  std::string dist_backend = "stub";
  if (const char *env_rank = std::getenv("INFERFLUX_RANK")) {
    dist_rank = std::stoi(env_rank);
  }
  if (const char *env_ws = std::getenv("INFERFLUX_WORLD_SIZE")) {
    dist_world_size = std::stoi(env_ws);
  }
  if (const char *env_be = std::getenv("INFERFLUX_DIST_BACKEND")) {
    dist_backend = env_be;
  }

  if (dist_world_size > 1) {
    inferflux::ParallelContext::Get().Initialize(dist_rank, dist_world_size,
                                                 dist_backend);
    std::cout << "[distributed] initialized rank=" << dist_rank
              << " world_size=" << dist_world_size
              << " backend=" << dist_backend << "\n";
  }

  std::string config_path = "config/server.yaml";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--ui") {
      // WebUI is enabled at compile time via INFERFLUX_ENABLE_WEBUI.
      // This flag is accepted for CLI/Docker compatibility but is a no-op at
      // runtime; the UI is always served at /ui when the feature is built in.
    }
  }

  std::string host = "0.0.0.0";
  int port = 8080;
  auto auth = std::make_shared<inferflux::ApiKeyAuth>();
  std::string model_path;
  std::string legacy_model_format{"auto"};
  int mps_layers = 0;
  int rate_limit_per_minute = 0;
  std::string audit_log_path;
  std::vector<std::string> guard_blocklist;
  std::string oidc_issuer;
  std::string oidc_audience;
  bool tls_enabled = false;
  std::string tls_cert_path;
  std::string tls_key_path;
  bool cuda_enabled = false;
  bool cuda_flash_attention_enabled = false;
  int cuda_flash_attention_tile = 128;
  std::string cuda_attention_kernel = "auto";
  std::string inferflux_cuda_kv_cache_dtype = "auto";
  std::string inferflux_cuda_dequantized_cache_policy = "none";
  bool inferflux_cuda_require_fused_quantized_matmul = false;
  bool cuda_phase_overlap_scaffold = false;
  int cuda_phase_overlap_min_prefill_tokens = 256;
  bool cuda_phase_overlap_prefill_replica = false;
  bool backend_prefer_inferflux = true;
  bool backend_allow_llama_cpp_fallback = true;
  bool backend_strict_inferflux_request = false;
  std::vector<std::string> backend_priority;
  bool speculative_enabled = false;
  std::string speculative_draft_model;
  int speculative_max_prefill_tokens = 256;
  int speculative_chunk_size = 4;
  std::string nvme_offload_path;
  std::string opa_endpoint;
  std::size_t paged_kv_pages = 32;
  std::string paged_kv_policy = "lru";
  std::size_t nvme_writer_workers = 1;
  std::size_t nvme_writer_queue_depth = 256;
  std::size_t prefix_cache_capacity = 256;
  int prefill_pool_size = 1;
  int decode_pool_size =
      1; // 0 = decode on WorkerLoop thread (legacy); 1 = single dedicated decode lane (current default)
  std::size_t kv_channel_capacity = 64;
  std::string kv_transport_type = "channel"; // "channel" | "shm"
  int kv_enqueue_max_retries = 3;
  inferflux::Scheduler::Config scheduler_config;
  std::vector<ModelConfig> configured_models;
  std::string default_model_override;
  inferflux::ModelSelectionOptions routing_selection_options{};
  routing_selection_options.allow_capability_fallback_for_default = true;
  routing_selection_options.require_ready_backend = true;
  routing_selection_options.capability_fallback_scope =
      inferflux::CapabilityFallbackScope::kAnyCompatible;
  std::string
      mmproj_path; // §2.2: path to multimodal projector (empty = disabled).
  std::string registry_path; // CQ-8: hot-reload registry.yaml (empty = off)
  int registry_poll_ms = 5000;

  struct ApiKeyEntry {
    std::string key;
    std::vector<std::string> scopes;
  };
  std::vector<ApiKeyEntry> api_key_entries;

  if (std::filesystem::exists(config_path)) {
    try {
      YAML::Node config = YAML::LoadFile(config_path);

      // Server config
      if (config["server"]) {
        if (config["server"]["host"])
          host = config["server"]["host"].as<std::string>();
        if (config["server"]["http_port"])
          port = config["server"]["http_port"].as<int>();
      }

      // Legacy model config
      if (config["model"] && config["model"]["path"]) {
        model_path = config["model"]["path"].as<std::string>();
      }
      if (config["model"] && config["model"]["format"]) {
        auto normalized = inferflux::NormalizeModelFormat(
            config["model"]["format"].as<std::string>());
        if (!normalized.empty()) {
          legacy_model_format = normalized;
        }
      }
      if (config["model"] && config["model"]["mmproj_path"]) {
        mmproj_path = config["model"]["mmproj_path"].as<std::string>();
      }

      // Models config
      if (config["models"] && config["models"].IsSequence()) {
        for (const auto &model_node : config["models"]) {
          ModelConfig mc;
          if (model_node["id"])
            mc.id = model_node["id"].as<std::string>();
          if (model_node["path"])
            mc.path = model_node["path"].as<std::string>();
          if (model_node["format"]) {
            auto normalized = inferflux::NormalizeModelFormat(
                model_node["format"].as<std::string>());
            if (!normalized.empty()) {
              mc.format = normalized;
            } else {
              inferflux::log::Warn("server",
                                   "Invalid model format; defaulting to auto",
                                   model_node["format"].as<std::string>());
            }
          }
          if (model_node["backend"])
            mc.backend = ToLower(model_node["backend"].as<std::string>());
          if (model_node["default"])
            mc.make_default = model_node["default"].as<bool>();
          if (!mc.path.empty()) {
            configured_models.push_back(mc);
          }
        }
      }

      // Registry config (CQ-8: hot-reload model registry)
      if (config["registry"]) {
        if (config["registry"]["path"])
          registry_path = config["registry"]["path"].as<std::string>();
        if (config["registry"]["poll_interval_ms"])
          registry_poll_ms = config["registry"]["poll_interval_ms"].as<int>();
      }

      // Runtime config
      if (config["runtime"]) {
        if (config["runtime"]["backend_priority"] &&
            config["runtime"]["backend_priority"].IsSequence()) {
          backend_priority.clear();
          for (const auto &hint : config["runtime"]["backend_priority"]) {
            if (hint.IsScalar()) {
              backend_priority.push_back(ToLower(hint.as<std::string>()));
            }
          }
        }
        if (config["runtime"]["mps_layers"])
          mps_layers = config["runtime"]["mps_layers"].as<int>();
        if (config["runtime"]["prefix_cache_capacity"])
          prefix_cache_capacity =
              config["runtime"]["prefix_cache_capacity"].as<std::size_t>();
        if (config["runtime"]["cuda"] && config["runtime"]["cuda"]["enabled"]) {
          cuda_enabled = config["runtime"]["cuda"]["enabled"].as<bool>();
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["flash_attention"] &&
            config["runtime"]["cuda"]["flash_attention"]["enabled"]) {
          cuda_flash_attention_enabled =
              config["runtime"]["cuda"]["flash_attention"]["enabled"]
                  .as<bool>();
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["flash_attention"] &&
            config["runtime"]["cuda"]["flash_attention"]["tile_size"]) {
          cuda_flash_attention_tile =
              config["runtime"]["cuda"]["flash_attention"]["tile_size"]
                  .as<int>();
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["attention"] &&
            config["runtime"]["cuda"]["attention"]["kernel"]) {
          cuda_attention_kernel =
              ToLower(config["runtime"]["cuda"]["attention"]["kernel"]
                          .as<std::string>());
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["kv_cache_dtype"] &&
            config["runtime"]["cuda"]["kv_cache_dtype"].IsScalar()) {
          inferflux_cuda_kv_cache_dtype = ToLower(
              config["runtime"]["cuda"]["kv_cache_dtype"].as<std::string>());
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["dequantized_cache_policy"] &&
            config["runtime"]["cuda"]["dequantized_cache_policy"].IsScalar()) {
          inferflux_cuda_dequantized_cache_policy =
              ToLower(config["runtime"]["cuda"]["dequantized_cache_policy"]
                          .as<std::string>());
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["quantized_runtime"] &&
            config["runtime"]["cuda"]["quantized_runtime"]
                  ["require_fused_matmul"]) {
          inferflux_cuda_require_fused_quantized_matmul =
              config["runtime"]["cuda"]["quantized_runtime"]
                    ["require_fused_matmul"]
                        .as<bool>();
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["phase_overlap"] &&
            config["runtime"]["cuda"]["phase_overlap"]["enabled"]) {
          cuda_phase_overlap_scaffold =
              config["runtime"]["cuda"]["phase_overlap"]["enabled"].as<bool>();
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["phase_overlap"] &&
            config["runtime"]["cuda"]["phase_overlap"]["min_prefill_tokens"]) {
          cuda_phase_overlap_min_prefill_tokens = std::max(
              1,
              config["runtime"]["cuda"]["phase_overlap"]["min_prefill_tokens"]
                  .as<int>());
        }
        if (config["runtime"]["cuda"] &&
            config["runtime"]["cuda"]["phase_overlap"] &&
            config["runtime"]["cuda"]["phase_overlap"]["prefill_replica"]) {
          cuda_phase_overlap_prefill_replica =
              config["runtime"]["cuda"]["phase_overlap"]["prefill_replica"]
                  .as<bool>();
        }
        if (config["runtime"]["backend_exposure"]) {
          const auto &exposure = config["runtime"]["backend_exposure"];
          if (exposure["prefer_inferflux"]) {
            backend_prefer_inferflux = exposure["prefer_inferflux"].as<bool>();
          }
          if (exposure["allow_llama_cpp_fallback"]) {
            backend_allow_llama_cpp_fallback =
                exposure["allow_llama_cpp_fallback"].as<bool>();
          }
          if (exposure["strict_inferflux_request"]) {
            backend_strict_inferflux_request =
                exposure["strict_inferflux_request"].as<bool>();
          }
        }
        if (config["runtime"]["capability_routing"]) {
          const auto &routing = config["runtime"]["capability_routing"];
          if (routing["allow_default_fallback"]) {
            routing_selection_options.allow_capability_fallback_for_default =
                routing["allow_default_fallback"].as<bool>();
          }
          if (routing["require_ready_backend"]) {
            routing_selection_options.require_ready_backend =
                routing["require_ready_backend"].as<bool>();
          }
          if (routing["fallback_scope"] &&
              routing["fallback_scope"].IsScalar()) {
            routing_selection_options.capability_fallback_scope =
                inferflux::ParseCapabilityFallbackScope(
                    routing["fallback_scope"].as<std::string>(),
                    routing_selection_options.capability_fallback_scope);
          }
        }
        if (config["runtime"]["speculative_decoding"]) {
          if (config["runtime"]["speculative_decoding"]["enabled"])
            speculative_enabled =
                config["runtime"]["speculative_decoding"]["enabled"].as<bool>();
          if (config["runtime"]["speculative_decoding"]["draft_model"])
            speculative_draft_model =
                config["runtime"]["speculative_decoding"]["draft_model"]
                    .as<std::string>();
          if (config["runtime"]["speculative_decoding"]["max_prefill_tokens"])
            speculative_max_prefill_tokens =
                config["runtime"]["speculative_decoding"]["max_prefill_tokens"]
                    .as<int>();
          if (config["runtime"]["speculative_decoding"]["chunk_size"])
            speculative_chunk_size =
                config["runtime"]["speculative_decoding"]["chunk_size"]
                    .as<int>();
        }
        if (config["runtime"]["nvme_offload"]) {
          if (config["runtime"]["nvme_offload"]["path"])
            nvme_offload_path =
                config["runtime"]["nvme_offload"]["path"].as<std::string>();
          if (config["runtime"]["nvme_offload"]["workers"])
            nvme_writer_workers =
                config["runtime"]["nvme_offload"]["workers"].as<std::size_t>();
          if (config["runtime"]["nvme_offload"]["queue_depth"])
            nvme_writer_queue_depth =
                config["runtime"]["nvme_offload"]["queue_depth"]
                    .as<std::size_t>();
        }
        if (config["runtime"]["paged_kv"]) {
          if (config["runtime"]["paged_kv"]["cpu_pages"])
            paged_kv_pages =
                config["runtime"]["paged_kv"]["cpu_pages"].as<std::size_t>();
          if (config["runtime"]["paged_kv"]["eviction"])
            paged_kv_policy =
                config["runtime"]["paged_kv"]["eviction"].as<std::string>();
        }
        if (config["runtime"]["disaggregated"]) {
          const auto &disagg = config["runtime"]["disaggregated"];
          if (disagg["prefill_pool_size"])
            prefill_pool_size =
                std::max(0, disagg["prefill_pool_size"].as<int>());
          if (disagg["decode_pool_size"])
            decode_pool_size =
                std::max(0, disagg["decode_pool_size"].as<int>());
          if (disagg["kv_channel_capacity"])
            kv_channel_capacity =
                disagg["kv_channel_capacity"].as<std::size_t>();
          if (disagg["kv_enqueue_max_retries"])
            kv_enqueue_max_retries =
                std::max(0, disagg["kv_enqueue_max_retries"].as<int>());
        }
        if (config["runtime"]["scheduler"]) {
          const auto &scheduler_node = config["runtime"]["scheduler"];
          if (scheduler_node["max_batch_size"]) {
            scheduler_config.max_batch_size =
                std::max(1, scheduler_node["max_batch_size"].as<int>());
          }
          if (scheduler_node["max_batch_tokens"]) {
            scheduler_config.max_batch_tokens =
                std::max(1, scheduler_node["max_batch_tokens"].as<int>());
          }
          if (scheduler_node["min_batch_size"]) {
            scheduler_config.min_batch_size =
                std::max(1, scheduler_node["min_batch_size"].as<int>());
          }
          if (scheduler_node["batch_accumulation_ms"]) {
            scheduler_config.batch_accumulation_ms =
                std::max(0, scheduler_node["batch_accumulation_ms"].as<int>());
          }
          if (scheduler_node["decode_burst_tokens"]) {
            scheduler_config.decode_burst_tokens = std::max(
                0, scheduler_node["decode_burst_tokens"].as<int>());
          }
          if (scheduler_node["chunked_prefill_tokens"]) {
            scheduler_config.chunked_prefill_tokens =
                std::max(1, scheduler_node["chunked_prefill_tokens"].as<int>());
          }
          if (scheduler_node["mixed_prefill_budget_ratio"]) {
            scheduler_config.mixed_prefill_budget_ratio = std::clamp(
                scheduler_node["mixed_prefill_budget_ratio"].as<double>(), 0.0,
                1.0);
          }
          if (scheduler_node["policy"]) {
            const std::string policy_raw =
                scheduler_node["policy"].as<std::string>();
            if (inferflux::IsSchedulerBatchPolicyValue(policy_raw)) {
              scheduler_config.batch_policy =
                  inferflux::ParseSchedulerBatchPolicy(
                      policy_raw, scheduler_config.batch_policy);
            } else {
              inferflux::log::Warn("server",
                                   "Invalid runtime.scheduler.policy '" +
                                       policy_raw + "'; keeping " +
                                       inferflux::SchedulerBatchPolicyToString(
                                           scheduler_config.batch_policy));
            }
          }
          if (scheduler_node["session_handles"]) {
            const auto &session_node = scheduler_node["session_handles"];
            if (session_node["enabled"]) {
              scheduler_config.session_handles.enabled =
                  session_node["enabled"].as<bool>();
            }
            if (session_node["ttl_ms"]) {
              scheduler_config.session_handles.ttl_ms =
                  std::max(1, session_node["ttl_ms"].as<int>());
            }
            if (session_node["max_sessions"]) {
              scheduler_config.session_handles.max_sessions =
                  std::max(1, session_node["max_sessions"].as<int>());
            }
          }
        }
      }

      // Auth config
      if (config["auth"]) {
        if (config["auth"]["rate_limit_per_minute"])
          rate_limit_per_minute =
              config["auth"]["rate_limit_per_minute"].as<int>();
        if (config["auth"]["oidc_issuer"])
          oidc_issuer = config["auth"]["oidc_issuer"].as<std::string>();
        if (config["auth"]["oidc_audience"])
          oidc_audience = config["auth"]["oidc_audience"].as<std::string>();
        if (config["auth"]["api_keys"] &&
            config["auth"]["api_keys"].IsSequence()) {
          for (const auto &key_node : config["auth"]["api_keys"]) {
            ApiKeyEntry entry;
            if (key_node.IsScalar()) { // Simple key string
              entry.key = key_node.as<std::string>();
            } else if (key_node.IsMap() && key_node["key"]) {
              entry.key = key_node["key"].as<std::string>();
              if (key_node["scopes"] && key_node["scopes"].IsSequence()) {
                for (const auto &scope_node : key_node["scopes"]) {
                  entry.scopes.push_back(scope_node.as<std::string>());
                }
              }
            }
            if (!entry.key.empty()) {
              api_key_entries.push_back(entry);
            }
          }
        }
      }

      // Guardrails config
      if (config["guardrails"]) {
        if (config["guardrails"]["blocklist"] &&
            config["guardrails"]["blocklist"].IsSequence()) {
          for (const auto &item : config["guardrails"]["blocklist"]) {
            guard_blocklist.push_back(item.as<std::string>());
          }
        }
        if (config["guardrails"]["opa_endpoint"])
          opa_endpoint = config["guardrails"]["opa_endpoint"].as<std::string>();
      }

      // Logging config
      if (config["logging"]) {
        if (config["logging"]["audit_log"]) {
          audit_log_path = config["logging"]["audit_log"].as<std::string>();
        }
        if (!log_format_from_env && config["logging"]["format"]) {
          const auto format = config["logging"]["format"].as<std::string>();
          inferflux::log::SetJsonMode(format == "json" || format == "JSON");
        }
        if (config["logging"]["level"]) {
          configured_log_level = config["logging"]["level"].as<std::string>();
        }
      }

      // TLS config
      if (config["tls"]) {
        if (config["tls"]["enabled"])
          tls_enabled = config["tls"]["enabled"].as<bool>();
        if (config["tls"]["cert_path"])
          tls_cert_path = config["tls"]["cert_path"].as<std::string>();
        if (config["tls"]["key_path"])
          tls_key_path = config["tls"]["key_path"].as<std::string>();
      }

    } catch (const YAML::Exception &e) {
      inferflux::log::Error("server", "Error parsing config file",
                            config_path + ": " + e.what());
    }
  }

  for (const auto &entry : api_key_entries) {
    auth->AddKey(entry.key, entry.scopes);
  }

  if (const char *env_keys = std::getenv("INFERFLUX_API_KEYS")) {
    std::stringstream ss(env_keys);
    std::string key;
    while (std::getline(ss, key, ',')) {
      auto trimmed = Trim(key);
      if (!trimmed.empty()) {
        auth->AddKey(trimmed, {});
      }
    }
  }
  // INFERCTL_API_KEY: convenience single-key override (used in dev/test
  // environments).
  if (const char *env_key = std::getenv("INFERCTL_API_KEY")) {
    std::string key = Trim(env_key);
    if (!key.empty()) {
      auth->AddKey(key, {"generate", "read", "admin"});
    }
  }

  bool has_env_model_path = false;
  if (const char *env_model = std::getenv("INFERFLUX_MODEL_PATH")) {
    model_path = env_model;
    has_env_model_path = true;
  }
  if (const char *env_model_format = std::getenv("INFERFLUX_MODEL_FORMAT")) {
    auto normalized = inferflux::NormalizeModelFormat(env_model_format);
    if (!normalized.empty()) {
      legacy_model_format = normalized;
    } else {
      inferflux::log::Warn("server",
                           "Ignoring invalid INFERFLUX_MODEL_FORMAT value",
                           env_model_format);
    }
  }
  if (const char *env_reg = std::getenv("INFERFLUX_REGISTRY_PATH")) {
    registry_path = env_reg;
  }
  if (const char *env_mmproj = std::getenv("INFERFLUX_MMPROJ_PATH")) {
    mmproj_path = env_mmproj;
  }
  if (const char *env_mps = std::getenv("INFERFLUX_MPS_LAYERS")) {
    mps_layers = std::stoi(env_mps);
  }
  if (const char *env_rate = std::getenv("INFERFLUX_RATE_LIMIT_PER_MINUTE")) {
    rate_limit_per_minute = std::stoi(env_rate);
  }
  if (const char *env_audit = std::getenv("INFERFLUX_AUDIT_LOG")) {
    audit_log_path = env_audit;
  }
  if (!log_level_from_env) {
    inferflux::log::Level parsed_level;
    if (inferflux::log::ParseLevel(configured_log_level, &parsed_level)) {
      inferflux::log::SetLevel(parsed_level);
    } else {
      inferflux::log::Warn("server", "Ignoring invalid logging.level setting",
                           configured_log_level);
    }
  }
  if (const char *env_blk = std::getenv("INFERFLUX_GUARDRAIL_BLOCKLIST")) {
    std::stringstream ss(env_blk);
    std::string item;
    while (std::getline(ss, item, ',')) {
      auto word = Trim(item);
      if (!word.empty()) {
        guard_blocklist.push_back(word);
      }
    }
  }
  if (const char *env_oidc_iss = std::getenv("INFERFLUX_OIDC_ISSUER")) {
    oidc_issuer = env_oidc_iss;
  }
  if (const char *env_oidc_aud = std::getenv("INFERFLUX_OIDC_AUDIENCE")) {
    oidc_audience = env_oidc_aud;
  }
  if (const char *env_spec = std::getenv("INFERFLUX_SPECULATIVE_ENABLED")) {
    speculative_enabled = std::string(env_spec) == "true";
  }
  if (const char *env_spec_draft = std::getenv("INFERFLUX_SPEC_DRAFT_MODEL")) {
    speculative_draft_model = env_spec_draft;
  }
  if (const char *env_spec_prefill =
          std::getenv("INFERFLUX_SPEC_MAX_PREFILL")) {
    speculative_max_prefill_tokens = std::stoi(env_spec_prefill);
  }
  if (const char *env_spec_chunk = std::getenv("INFERFLUX_SPEC_CHUNK_SIZE")) {
    auto chunk = ParsePositiveSize(env_spec_chunk);
    if (chunk > 0) {
      speculative_chunk_size = static_cast<int>(chunk);
    }
  }
  if (const char *env_nvme = std::getenv("INFERFLUX_NVME_OFFLOAD_PATH")) {
    nvme_offload_path = env_nvme;
  }
  if (const char *env_kv_pages = std::getenv("INFERFLUX_KV_CPU_PAGES")) {
    auto pages_value = ParsePositiveSize(env_kv_pages);
    if (pages_value > 0) {
      paged_kv_pages = pages_value;
    }
  }
  if (const char *env_kv_policy = std::getenv("INFERFLUX_KV_EVICTION")) {
    paged_kv_policy = env_kv_policy;
  }
  if (const char *env_nvme_workers = std::getenv("INFERFLUX_NVME_WORKERS")) {
    auto workers = ParsePositiveSize(env_nvme_workers);
    if (workers > 0) {
      nvme_writer_workers = workers;
    }
  }
  if (const char *env_nvme_depth = std::getenv("INFERFLUX_NVME_QUEUE_DEPTH")) {
    auto depth = ParsePositiveSize(env_nvme_depth);
    if (depth > 0) {
      nvme_writer_queue_depth = depth;
    }
  }
  if (const char *env_opa = std::getenv("INFERFLUX_OPA_ENDPOINT")) {
    opa_endpoint = env_opa;
  }
  if (const char *env_port = std::getenv("INFERFLUX_PORT_OVERRIDE")) {
    port = std::stoi(env_port);
  }
  if (const char *env_host = std::getenv("INFERFLUX_HOST_OVERRIDE")) {
    host = env_host;
  }
  if (const char *env_tls = std::getenv("INFERFLUX_TLS_ENABLED")) {
    std::string value = env_tls;
    tls_enabled = (value == "true" || value == "1");
  }
  if (const char *env_tls_cert = std::getenv("INFERFLUX_TLS_CERT_PATH")) {
    tls_cert_path = env_tls_cert;
  }
  if (const char *env_tls_key = std::getenv("INFERFLUX_TLS_KEY_PATH")) {
    tls_key_path = env_tls_key;
  }
  if (const char *env_prefix_cap =
          std::getenv("INFERFLUX_PREFIX_CACHE_CAPACITY")) {
    auto cap = ParsePositiveSize(env_prefix_cap);
    if (cap > 0) {
      prefix_cache_capacity = cap;
    }
  }
  if (const char *env_cuda = std::getenv("INFERFLUX_CUDA_ENABLED")) {
    std::string value = env_cuda;
    cuda_enabled = (value == "true" || value == "1");
  }
  if (const char *env_cuda_fa = std::getenv("INFERFLUX_CUDA_FLASH_ATTENTION")) {
    std::string value = env_cuda_fa;
    cuda_flash_attention_enabled = (value == "true" || value == "1");
  }
  if (const char *env_cuda_tile = std::getenv("INFERFLUX_CUDA_FLASH_TILE")) {
    auto tile = ParsePositiveSize(env_cuda_tile);
    if (tile > 0) {
      cuda_flash_attention_tile = static_cast<int>(tile);
    }
  }
  if (const char *env_cuda_attention_kernel =
          std::getenv("INFERFLUX_CUDA_ATTENTION_KERNEL")) {
    cuda_attention_kernel = ToLower(env_cuda_attention_kernel);
  }
  if (const char *env_inferflux_cuda_kv_precision =
          std::getenv("INFERFLUX_CUDA_KV_DTYPE")) {
    inferflux_cuda_kv_cache_dtype = ToLower(env_inferflux_cuda_kv_precision);
  }
  if (const char *env_inferflux_cuda_dequant_policy =
          std::getenv("INFERFLUX_CUDA_DEQUANT_CACHE_POLICY")) {
    inferflux_cuda_dequantized_cache_policy =
        ToLower(env_inferflux_cuda_dequant_policy);
  }
  if (const char *env_inferflux_cuda_require_fused =
          std::getenv("INFERFLUX_CUDA_REQUIRE_FUSED_MATMUL")) {
    inferflux_cuda_require_fused_quantized_matmul =
        ParseBool(env_inferflux_cuda_require_fused);
  }
  if (const char *env_cuda_overlap =
          std::getenv("INFERFLUX_CUDA_PHASE_OVERLAP")) {
    cuda_phase_overlap_scaffold = ParseBool(env_cuda_overlap);
  }
  if (const char *env_cuda_overlap_prefill =
          std::getenv("INFERFLUX_CUDA_PHASE_OVERLAP_MIN_PREFILL_TOKENS")) {
    auto v = ParsePositiveSize(env_cuda_overlap_prefill);
    if (v > 0) {
      cuda_phase_overlap_min_prefill_tokens = static_cast<int>(v);
    }
  }
  if (const char *env_cuda_overlap_prefill_replica =
          std::getenv("INFERFLUX_CUDA_PHASE_OVERLAP_PREFILL_REPLICA")) {
    cuda_phase_overlap_prefill_replica =
        ParseBool(env_cuda_overlap_prefill_replica);
  }
  if (const char *env_prefer_inferflux =
          std::getenv("INFERFLUX_BACKEND_PREFER_INFERFLUX")) {
    backend_prefer_inferflux = ParseBool(env_prefer_inferflux);
  }
  if (const char *env_allow_fallback =
          std::getenv("INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK")) {
    backend_allow_llama_cpp_fallback = ParseBool(env_allow_fallback);
  }
  if (const char *env_strict_inferflux =
          std::getenv("INFERFLUX_BACKEND_STRICT_INFERFLUX_REQUEST")) {
    backend_strict_inferflux_request = ParseBool(env_strict_inferflux);
  }
  if (const char *env_backend_priority =
          std::getenv("INFERFLUX_BACKEND_PRIORITY")) {
    auto parsed_priority = ParseBackendPriorityList(env_backend_priority);
    if (!parsed_priority.empty()) {
      backend_priority = std::move(parsed_priority);
    }
  }
  if (const char *env_allow_default_cap_fallback =
          std::getenv("INFERFLUX_ROUTING_ALLOW_DEFAULT_CAPABILITY_FALLBACK")) {
    routing_selection_options.allow_capability_fallback_for_default =
        ParseBool(env_allow_default_cap_fallback);
  }
  if (const char *env_routing_require_ready =
          std::getenv("INFERFLUX_ROUTING_REQUIRE_READY_BACKEND")) {
    routing_selection_options.require_ready_backend =
        ParseBool(env_routing_require_ready);
  }
  if (const char *env_routing_scope =
          std::getenv("INFERFLUX_ROUTING_FALLBACK_SCOPE")) {
    routing_selection_options.capability_fallback_scope =
        inferflux::ParseCapabilityFallbackScope(
            env_routing_scope,
            routing_selection_options.capability_fallback_scope);
  }
  inferflux::FairnessConfig fairness_config;
  fairness_config.enable_preemption = false;
  fairness_config.high_priority_threshold = 5;
  fairness_config.max_timeslice_tokens = 0;
  if (const char *env_preempt =
          std::getenv("INFERFLUX_FAIRNESS_ENABLE_PREEMPTION")) {
    fairness_config.enable_preemption =
        std::string(env_preempt) == "true" || std::string(env_preempt) == "1";
  }
  if (const char *env_threshold =
          std::getenv("INFERFLUX_FAIRNESS_PRIORITY_THRESHOLD")) {
    fairness_config.high_priority_threshold = std::stoi(env_threshold);
  }
  if (const char *env_timeslice =
          std::getenv("INFERFLUX_FAIRNESS_MAX_TIMESLICE")) {
    fairness_config.max_timeslice_tokens = std::stoi(env_timeslice);
  }
  if (const char *env_prefill_pool =
          std::getenv("INFERFLUX_PREFILL_POOL_SIZE")) {
    // Allow "0" explicitly so a decode-only node can disable all prefill
    // workers.
    try {
      int v = std::stoi(env_prefill_pool);
      if (v >= 0)
        prefill_pool_size = v;
    } catch (...) {
    }
  }
  if (const char *env_decode_pool = std::getenv("INFERFLUX_DECODE_POOL_SIZE")) {
    auto pool = ParsePositiveSize(env_decode_pool);
    if (pool > 0) {
      decode_pool_size = static_cast<int>(pool);
    }
  }
  if (const char *env_batch_size =
          std::getenv("INFERFLUX_SCHED_MAX_BATCH_SIZE")) {
    auto batch_size = ParsePositiveSize(env_batch_size);
    if (batch_size > 0) {
      scheduler_config.max_batch_size = static_cast<int>(batch_size);
    }
  }
  if (const char *env_batch_tokens =
          std::getenv("INFERFLUX_SCHED_MAX_BATCH_TOKENS")) {
    auto batch_tokens = ParsePositiveSize(env_batch_tokens);
    if (batch_tokens > 0) {
      scheduler_config.max_batch_tokens = static_cast<int>(batch_tokens);
    }
  }
  if (const char *env_min_batch =
          std::getenv("INFERFLUX_SCHED_MIN_BATCH_SIZE")) {
    auto min_batch = ParsePositiveSize(env_min_batch);
    if (min_batch > 0) {
      scheduler_config.min_batch_size = static_cast<int>(min_batch);
    }
  }
  if (const char *env_accumulation =
          std::getenv("INFERFLUX_SCHED_BATCH_ACCUMULATION_MS")) {
    auto accumulation = ParsePositiveSize(env_accumulation);
    if (accumulation >= 0) {
      scheduler_config.batch_accumulation_ms = static_cast<int>(accumulation);
    }
  }
  if (const char *env_decode_burst_tokens =
          std::getenv("INFERFLUX_SCHED_DECODE_BURST_TOKENS")) {
    try {
      int v = std::stoi(env_decode_burst_tokens);
      if (v >= 0) {
        scheduler_config.decode_burst_tokens = v;
      }
    } catch (...) {
    }
  }
  if (const char *env_chunked_prefill_tokens =
          std::getenv("INFERFLUX_SCHED_CHUNKED_PREFILL_TOKENS")) {
    auto chunk_tokens = ParsePositiveSize(env_chunked_prefill_tokens);
    if (chunk_tokens > 0) {
      scheduler_config.chunked_prefill_tokens = static_cast<int>(chunk_tokens);
    }
  }
  if (const char *env_prefill_budget_ratio =
          std::getenv("INFERFLUX_SCHED_MIXED_PREFILL_BUDGET_RATIO")) {
    try {
      const double ratio = std::stod(std::string(env_prefill_budget_ratio));
      scheduler_config.mixed_prefill_budget_ratio = std::clamp(ratio, 0.0, 1.0);
    } catch (...) {
    }
  }
  if (const char *env_sched_policy = std::getenv("INFERFLUX_SCHED_POLICY")) {
    const std::string policy_raw = env_sched_policy;
    if (inferflux::IsSchedulerBatchPolicyValue(policy_raw)) {
      scheduler_config.batch_policy = inferflux::ParseSchedulerBatchPolicy(
          policy_raw, scheduler_config.batch_policy);
    } else {
      inferflux::log::Warn("server",
                           "Invalid INFERFLUX_SCHED_POLICY '" + policy_raw +
                               "'; keeping " +
                               inferflux::SchedulerBatchPolicyToString(
                                   scheduler_config.batch_policy));
    }
  }
  if (const char *env_session_handles =
          std::getenv("INFERFLUX_SESSION_HANDLES_ENABLED")) {
    scheduler_config.session_handles.enabled = ParseBool(env_session_handles);
  }
  if (const char *env_session_ttl = std::getenv("INFERFLUX_SESSION_TTL_MS")) {
    auto ttl = ParsePositiveSize(env_session_ttl);
    if (ttl > 0) {
      scheduler_config.session_handles.ttl_ms = static_cast<int>(ttl);
    }
  }
  if (const char *env_session_max = std::getenv("INFERFLUX_SESSION_MAX")) {
    auto max_sessions = ParsePositiveSize(env_session_max);
    if (max_sessions > 0) {
      scheduler_config.session_handles.max_sessions =
          static_cast<int>(max_sessions);
    }
  }
  if (const char *env_kv_channel_cap =
          std::getenv("INFERFLUX_KV_CHANNEL_CAPACITY")) {
    auto cap = ParsePositiveSize(env_kv_channel_cap);
    if (cap > 0) {
      kv_channel_capacity = cap;
    }
  }
  if (const char *env_kv_transport = std::getenv("INFERFLUX_KV_TRANSPORT")) {
    kv_transport_type = env_kv_transport; // "channel" or "shm"
  }
  if (const char *env_kv_enqueue_retries =
          std::getenv("INFERFLUX_KV_ENQUEUE_MAX_RETRIES")) {
    try {
      int retries = std::stoi(env_kv_enqueue_retries);
      if (retries >= 0) {
        kv_enqueue_max_retries = retries;
      }
    } catch (...) {
    }
  }
  if (const char *env_models = std::getenv("INFERFLUX_MODELS")) {
    auto parsed = ParseModelsEnv(env_models);
    if (!parsed.empty()) {
      configured_models = parsed;
    }
  }
  if (const char *env_default_model =
          std::getenv("INFERFLUX_DEFAULT_MODEL_ID")) {
    default_model_override = env_default_model;
  }

  // INFERFLUX_MODEL_PATH overrides config file models (for testing/flexibility)
  if (has_env_model_path && !model_path.empty()) {
    inferflux::log::Info(
        "server",
        "INFERFLUX_MODEL_PATH is set, overriding config file model path: " +
            model_path + " (original config will be ignored)");

    ModelConfig cfg;
    cfg.path = model_path;
    cfg.format = legacy_model_format;
    cfg.make_default = true;
    configured_models.clear(); // Clear any models from config file
    configured_models.push_back(cfg);
  } else if (configured_models.empty() && !model_path.empty()) {
    ModelConfig cfg;
    cfg.path = model_path;
    cfg.format = legacy_model_format;
    cfg.make_default = true;
    configured_models.push_back(cfg);
  }

  inferflux::HttpServer::TlsConfig tls_config;
  tls_config.enabled = tls_enabled;
  tls_config.cert_path = tls_cert_path;
  tls_config.key_path = tls_key_path;
  if (tls_config.enabled &&
      (tls_config.cert_path.empty() || tls_config.key_path.empty())) {
    inferflux::log::Warn(
        "server",
        "TLS enabled without cert/key paths; disabling in-process TLS");
    tls_config.enabled = false;
  }

  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto normalized_policy = ToLower(paged_kv_policy);
  inferflux::PagedKVCache::EvictionPolicy policy =
      inferflux::PagedKVCache::EvictionPolicy::kLRU;
  if (normalized_policy == "clock") {
    policy = inferflux::PagedKVCache::EvictionPolicy::kClock;
  }
  std::size_t cache_pages = paged_kv_pages == 0 ? 32 : paged_kv_pages;
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      cache_pages, /*page_size_bytes=*/16384, policy);
  cache->ConfigureAsyncWriter(nvme_writer_workers, nvme_writer_queue_depth);
  if (!nvme_offload_path.empty()) {
    cache->SetOffloadPath(nvme_offload_path);
  }
  std::cout << "Paged KV cache pages: " << cache_pages
            << " eviction=" << normalized_policy << std::endl;
  inferflux::BackendFactory::SetExposurePolicy(
      {backend_prefer_inferflux, backend_allow_llama_cpp_fallback,
       backend_strict_inferflux_request});
  inferflux::log::Info(
      "server",
      "Backend exposure policy: prefer_inferflux=" +
          std::string(backend_prefer_inferflux ? "true" : "false") +
          ", allow_llama_cpp_fallback=" +
          std::string(backend_allow_llama_cpp_fallback ? "true" : "false") +
          ", strict_inferflux_request=" +
          std::string(backend_strict_inferflux_request ? "true" : "false"));
  // When allow_llama_cpp_fallback=true with prefer_inferflux=true and the
  // parity delegate is explicitly enabled, the delegate loads a second copy
  // of the model weights via llama.cpp — lazily, on the first request that
  // needs it (grammar-constrained output), after a free-VRAM pre-check.
  // The delegate is OFF by default (INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE
  // defaults true); disable fallback via allow_llama_cpp_fallback=false to
  // avoid the duplicate weights entirely. Long-term fix: share the GGUF
  // weight data between the two backends.
  if (backend_prefer_inferflux && backend_allow_llama_cpp_fallback) {
    inferflux::log::Warn(
        "server",
        "allow_llama_cpp_fallback=true with prefer_inferflux=true: parity "
        "delegate may load duplicate model weights. Set "
        "allow_llama_cpp_fallback=false or "
        "INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE=1 to reduce GPU memory.");
  }
  inferflux::log::Info(
      "server",
      "Capability routing policy: allow_default_fallback=" +
          std::string(
              routing_selection_options.allow_capability_fallback_for_default
                  ? "true"
                  : "false") +
          ", require_ready_backend=" +
          std::string(routing_selection_options.require_ready_backend
                          ? "true"
                          : "false") +
          ", fallback_scope=" +
          inferflux::CapabilityFallbackScopeToString(
              routing_selection_options.capability_fallback_scope));
  inferflux::log::Info(
      "server",
      "Scheduler batch policy: max_batch_size=" +
          std::to_string(scheduler_config.max_batch_size) +
          ", max_batch_tokens=" +
          std::to_string(scheduler_config.max_batch_tokens) +
          ", min_batch_size=" +
          std::to_string(scheduler_config.min_batch_size) +
          ", batch_accumulation_ms=" +
          std::to_string(scheduler_config.batch_accumulation_ms) + ", policy=" +
          inferflux::SchedulerBatchPolicyToString(
              scheduler_config.batch_policy) +
          ", decode_burst_tokens=" +
          std::to_string(scheduler_config.decode_burst_tokens) +
          ", chunked_prefill_tokens=" +
          std::to_string(scheduler_config.chunked_prefill_tokens) +
          ", mixed_prefill_budget_ratio=" +
          std::to_string(scheduler_config.mixed_prefill_budget_ratio) +
          ", session_handles.enabled=" +
          std::string(scheduler_config.session_handles.enabled ? "true"
                                                               : "false") +
          ", session_handles.ttl_ms=" +
          std::to_string(scheduler_config.session_handles.ttl_ms) +
          ", session_handles.max_sessions=" +
          std::to_string(scheduler_config.session_handles.max_sessions));
  inferflux::BackendManager backend_manager;
  std::string backend_label = "stub";
  std::string primary_model_id;
  inferflux::LlamaBackendConfig primary_cfg;
  primary_cfg.gpu_layers = mps_layers;
  primary_cfg.use_flash_attention =
      cuda_enabled && cuda_flash_attention_enabled;
  primary_cfg.flash_attention_tile = cuda_flash_attention_tile;
  primary_cfg.cuda_attention_kernel = cuda_attention_kernel;
  primary_cfg.inferflux_cuda_kv_cache_dtype = inferflux_cuda_kv_cache_dtype;
  primary_cfg.inferflux_cuda_dequantized_cache_policy =
      inferflux_cuda_dequantized_cache_policy;
  primary_cfg.inferflux_cuda_require_fused_quantized_matmul =
      inferflux_cuda_require_fused_quantized_matmul;
  primary_cfg.cuda_phase_overlap_scaffold =
      cuda_enabled && cuda_phase_overlap_scaffold;
  primary_cfg.cuda_phase_overlap_min_prefill_tokens =
      cuda_phase_overlap_min_prefill_tokens;
  primary_cfg.cuda_phase_overlap_prefill_replica =
      cuda_enabled && cuda_phase_overlap_prefill_replica;

#ifndef INFERFLUX_HAS_CUDA
  if (cuda_enabled) {
    inferflux::log::Warn(
        "server",
        "CUDA requested but binary was built without CUDA support. "
        "Set ENABLE_CUDA=ON with a CUDA toolkit installed to enable GPU mode.");
    cuda_enabled = false;
  }
  if (cuda_flash_attention_enabled) {
    inferflux::log::Warn("server",
                         "FlashAttention requested but CUDA support is "
                         "unavailable. Disabling flash_attention.");
    cuda_flash_attention_enabled = false;
  }
#else
  if (cuda_enabled) {
    std::cout << "[server] CUDA support enabled; FlashAttention executor will "
                 "be wired in upcoming releases.\n";
  }
#endif
  if (cuda_flash_attention_enabled && !cuda_enabled) {
    inferflux::log::Warn(
        "server",
        "FlashAttention requires CUDA. Disable or build with CUDA support.");
    cuda_flash_attention_enabled = false;
  }
  if (cuda_phase_overlap_scaffold && !cuda_enabled) {
    inferflux::log::Warn(
        "server",
        "CUDA phase-overlap scaffold requires CUDA. Disabling overlap mode.");
    cuda_phase_overlap_scaffold = false;
  }
  if (cuda_phase_overlap_prefill_replica && !cuda_phase_overlap_scaffold) {
    inferflux::log::Warn(
        "server",
        "CUDA prefill replica requires phase-overlap scaffold. Disabling "
        "prefill replica mode.");
    cuda_phase_overlap_prefill_replica = false;
  }
#ifdef INFERFLUX_HAS_CUDA
  if (cuda_enabled) {
    std::cout << "[server] InferFlux CUDA KV cache precision policy: "
              << inferflux_cuda_kv_cache_dtype << ".\n";
    std::cout << "[server] InferFlux CUDA dequant cache policy: "
              << inferflux_cuda_dequantized_cache_policy
              << " (none default, memory-first).\n";
    std::cout << "[server] InferFlux CUDA strict fused quantized matmul: "
              << (inferflux_cuda_require_fused_quantized_matmul ? "enabled"
                                                        : "disabled")
              << ".\n";
  }
  if (cuda_flash_attention_enabled) {
    std::cout << "[server] FlashAttention toggled on (kernel="
              << cuda_attention_kernel
              << ", tile_size=" << cuda_flash_attention_tile << ").\n";
  }
  if (cuda_phase_overlap_scaffold) {
    std::cout << "[server] CUDA phase-overlap scaffold enabled "
                 "(min_prefill_tokens="
              << cuda_phase_overlap_min_prefill_tokens << ").\n";
    if (cuda_phase_overlap_prefill_replica) {
      std::cout << "[server] CUDA prefill replica enabled (dual-context "
                   "overlap).\n";
    }
  }
#endif

  // Sync primary_cfg with effective post-guard FA state; record Prometheus
  // gauge (§2.7).
  primary_cfg.use_flash_attention = cuda_flash_attention_enabled;
  primary_cfg.cuda_phase_overlap_scaffold = cuda_phase_overlap_scaffold;
  primary_cfg.cuda_phase_overlap_prefill_replica =
      cuda_phase_overlap_prefill_replica;
  inferflux::GlobalMetrics().SetFlashAttentionEnabled(
      primary_cfg.use_flash_attention);

  inferflux::LlamaBackendConfig router_backend_config = primary_cfg;
  if (cuda_enabled && router_backend_config.gpu_layers <= 0) {
    router_backend_config.gpu_layers = 99;
  }
  std::string default_backend_hint = "cpu";
  if (cuda_enabled) {
    default_backend_hint = "cuda";
  } else if (mps_layers > 0) {
    default_backend_hint = "mps";
  }
  auto normalized_backend_priority =
      inferflux::BackendFactory::NormalizeHintList(backend_priority,
                                                   default_backend_hint);
  if (!normalized_backend_priority.empty()) {
    default_backend_hint = normalized_backend_priority.front();
  }
  inferflux::log::Info("server", "Backend priority order: " +
                                     JoinList(normalized_backend_priority));
  auto router = std::make_shared<inferflux::SingleModelRouter>(
      router_backend_config, default_backend_hint, normalized_backend_priority);
  std::string resolved_default_model_id;
  std::string resolved_default_path = model_path;
  if (!configured_models.empty()) {
    for (auto &cfg : configured_models) {
      auto assigned_id =
          router->LoadModel(cfg.path, cfg.backend, cfg.id, cfg.format);
      if (assigned_id.empty()) {
        const std::string load_error = router->LastLoadError();
        if (!load_error.empty()) {
          inferflux::log::Error("server", "Failed to load model: " + load_error,
                                cfg.path);
        } else {
          inferflux::log::Error("server", "Failed to load model", cfg.path);
        }
        continue;
      }
      cfg.id = assigned_id;
      std::string assigned_backend = cfg.backend;
      std::string assigned_format = cfg.format;
      std::string assigned_effective_path = cfg.path;
      for (const auto &loaded : router->ListModels()) {
        if (loaded.id == assigned_id) {
          assigned_backend = loaded.backend;
          assigned_format = loaded.format;
          if (!loaded.effective_load_path.empty()) {
            assigned_effective_path = loaded.effective_load_path;
          }
          break;
        }
      }
      if (assigned_backend.empty()) {
        assigned_backend = default_backend_hint;
      }
      std::cout << "InferFlux loaded model: " << cfg.path << " (format="
                << (assigned_format.empty() ? "auto" : assigned_format)
                << ", backend="
                << (assigned_backend.empty() ? "cpu" : assigned_backend)
                << ", effective_path=" << assigned_effective_path
                << ", id=" << assigned_id << ")" << std::endl;
      if (resolved_default_model_id.empty()) {
        resolved_default_model_id = assigned_id;
        resolved_default_path = cfg.path;
      }
      if (cfg.make_default) {
        resolved_default_model_id = assigned_id;
        resolved_default_path = cfg.path;
      }
    }
  }
  if (!default_model_override.empty()) {
    if (router->SetDefaultModel(default_model_override)) {
      resolved_default_model_id = default_model_override;
    } else {
      inferflux::log::Warn(
          "server", "INFERFLUX_DEFAULT_MODEL_ID did not match a loaded model",
          default_model_override);
    }
  } else if (!resolved_default_model_id.empty()) {
    router->SetDefaultModel(resolved_default_model_id);
  }
  auto llama_backend = resolved_default_model_id.empty()
                           ? nullptr
                           : router->GetBackend(resolved_default_model_id);
  primary_model_id = resolved_default_model_id;
  if (!resolved_default_path.empty()) {
    model_path = resolved_default_path;
  }
  if (!primary_model_id.empty()) {
    auto models = router->ListModels();
    for (const auto &info : models) {
      if (info.id == primary_model_id) {
        backend_label = info.backend.empty() ? backend_label : info.backend;
        break;
      }
    }
  }
  if (llama_backend && backend_label == "stub") {
    backend_label = cuda_enabled ? "cuda" : (mps_layers > 0 ? "mps" : "cpu");
  }

  // CQ-8: Start the hot-reload model registry (if configured).
  std::unique_ptr<inferflux::ModelRegistry> model_registry;
  if (!registry_path.empty()) {
    model_registry = std::make_unique<inferflux::ModelRegistry>(router);
    int n = model_registry->LoadAndWatch(registry_path, registry_poll_ms);
    inferflux::log::Info(
        "server", "Model registry loaded " + std::to_string(n) +
                      " model(s) from " + registry_path +
                      " (poll_ms=" + std::to_string(registry_poll_ms) + ")");
  }

  // §2.2: Load multimodal projector if specified.
  if (llama_backend && !mmproj_path.empty()) {
    auto llama_backend_cast =
        std::static_pointer_cast<inferflux::LlamaCppBackend>(llama_backend);
    if (llama_backend_cast->LoadMmproj(mmproj_path)) {
      std::cout << "[server] Multimodal projector loaded: " << mmproj_path
                << "\n";
    } else {
      inferflux::log::Warn("server",
                           "Failed to load mmproj; vision features disabled",
                           mmproj_path);
    }
  } else if (!mmproj_path.empty()) {
    inferflux::log::Warn("server", "INFERFLUX_MMPROJ_PATH set but no model "
                                   "loaded; mmproj will not be applied");
  }
  std::string policy_store_path = "config/policy_store.conf";
  if (const char *env_policy = std::getenv("INFERFLUX_POLICY_STORE")) {
    policy_store_path = env_policy;
  }
  std::string policy_passphrase;
  if (const char *env_pass = std::getenv("INFERFLUX_POLICY_PASSPHRASE")) {
    policy_passphrase = env_pass;
  }
  inferflux::PolicyStore policy_store(
      inferflux::PolicyStoreConfig{policy_store_path, policy_passphrase});
  policy_store.Load();

  // Keys from PolicyStore are already SHA-256 hashed on disk — load them
  // directly into ApiKeyAuth without re-hashing.
  auto stored_keys = policy_store.ApiKeys();
  for (const auto &entry : stored_keys) {
    auth->AddKeyHashed(entry.key, entry.scopes);
  }

  // Fresh keys from config/env are plaintext — hash them into both auth and
  // the policy store (PolicyStore::SetApiKey hashes before storing).
  if (stored_keys.empty()) {
    for (const auto &entry : api_key_entries) {
      auth->AddKey(entry.key, entry.scopes);
      policy_store.SetApiKey(entry.key, entry.scopes);
    }
  }
  int store_limit = policy_store.RateLimitPerMinute();
  if (store_limit > 0) {
    rate_limit_per_minute = store_limit;
  } else if (rate_limit_per_minute > 0) {
    policy_store.SetRateLimitPerMinute(rate_limit_per_minute);
  }
  auto store_guardrail = policy_store.GuardrailBlocklist();
  if (!store_guardrail.empty()) {
    guard_blocklist = store_guardrail;
  } else if (!guard_blocklist.empty()) {
    policy_store.SetGuardrailBlocklist(guard_blocklist);
  }
  auto store_routing = policy_store.RoutingPolicy();
  if (store_routing.has_value()) {
    routing_selection_options.allow_capability_fallback_for_default =
        store_routing->allow_default_fallback;
    routing_selection_options.require_ready_backend =
        store_routing->require_ready_backend;
    routing_selection_options.capability_fallback_scope =
        inferflux::ParseCapabilityFallbackScope(
            store_routing->fallback_scope,
            routing_selection_options.capability_fallback_scope);
  } else {
    inferflux::RoutingPolicyEntry initial_routing_policy;
    initial_routing_policy.allow_default_fallback =
        routing_selection_options.allow_capability_fallback_for_default;
    initial_routing_policy.require_ready_backend =
        routing_selection_options.require_ready_backend;
    initial_routing_policy.fallback_scope =
        inferflux::CapabilityFallbackScopeToString(
            routing_selection_options.capability_fallback_scope);
    policy_store.SetRoutingPolicy(initial_routing_policy);
  }
  if (!policy_store.Save()) {
    inferflux::log::Warn(
        "server",
        "Failed to persist policy store; runtime policy updates may be "
        "non-durable");
  }
  inferflux::log::Info(
      "server",
      "Effective capability routing policy: allow_default_fallback=" +
          std::string(
              routing_selection_options.allow_capability_fallback_for_default
                  ? "true"
                  : "false") +
          ", require_ready_backend=" +
          std::string(routing_selection_options.require_ready_backend
                          ? "true"
                          : "false") +
          ", fallback_scope=" +
          inferflux::CapabilityFallbackScopeToString(
              routing_selection_options.capability_fallback_scope));

  if (speculative_chunk_size <= 0) {
    speculative_chunk_size = 1;
  }
  inferflux::SpeculativeConfig spec_config;
  spec_config.enabled = speculative_enabled;
  spec_config.max_prefill_tokens = speculative_max_prefill_tokens;
  spec_config.draft_model = speculative_draft_model;
  spec_config.chunk_size = speculative_chunk_size;
  std::shared_ptr<inferflux::BackendInterface> draft_backend = llama_backend;
  if (speculative_enabled && !speculative_draft_model.empty() &&
      speculative_draft_model != model_path) {
    auto draft_cfg = primary_cfg;
    auto backend = backend_manager.LoadBackend("draft", speculative_draft_model,
                                               draft_cfg, cuda_enabled);
    if (backend) {
      draft_backend = backend;
    }
  }
  std::shared_ptr<inferflux::SpeculativeDecoder> speculative_decoder;
  if (speculative_enabled) {
    speculative_decoder = std::make_shared<inferflux::SpeculativeDecoder>(
        spec_config, device, &tokenizer, draft_backend);
  }

  // Build ModelRouter (SingleModelRouter is the default; future multi-model
  // routers can be substituted here without touching Scheduler or HttpServer).
  // The prefix cache needs a callback to free scheduler sequence slots on
  // eviction. Using a pointer-to-scheduler to break the circular initialization
  // dependency.
  inferflux::Scheduler *sched_ptr = nullptr;
  inferflux::RadixPrefixCacheLimits prefix_cache_limits;
  prefix_cache_limits.capacity =
      static_cast<std::size_t>(prefix_cache_capacity);
  auto prefix_cache = std::make_shared<inferflux::RadixPrefixCache>(
      cache,
      [&sched_ptr](int seq_id) {
        if (sched_ptr)
          sched_ptr->FreeSeqSlot(seq_id);
      },
      prefix_cache_limits);
  std::shared_ptr<inferflux::disaggregated::IKVTransport> kv_transport;
  if (kv_transport_type == "shm") {
    kv_transport = std::make_shared<inferflux::disaggregated::ShmKVTransport>(
        kv_channel_capacity);
    std::cout << "[server] KV transport: POSIX SHM (capacity="
              << kv_channel_capacity << ")\n";
  } else {
    kv_transport = std::make_shared<inferflux::disaggregated::KVChannel>(
        kv_channel_capacity);
    std::cout << "[server] KV transport: in-process channel (capacity="
              << kv_channel_capacity << ")\n";
  }
  inferflux::DisaggregatedConfig disagg_config;
  disagg_config.prefill_pool_size = std::max(0, prefill_pool_size);
  disagg_config.decode_pool_size = std::max(0, decode_pool_size);
  disagg_config.kv_transport = kv_transport;
  disagg_config.kv_enqueue_max_retries = std::max(0, kv_enqueue_max_retries);
  inferflux::Scheduler scheduler(tokenizer, device, cache, router,
                                 speculative_decoder, prefix_cache,
                                 fairness_config, disagg_config,
                                 routing_selection_options, scheduler_config);
  sched_ptr = &scheduler;
  auto &metrics = inferflux::GlobalMetrics();
  metrics.SetBackend(backend_label);
  inferflux::OIDCValidator oidc_validator(oidc_issuer, oidc_audience);
  inferflux::RateLimiter rate_limiter(rate_limit_per_minute);
  inferflux::Guardrail guardrail;
  guardrail.SetBlocklist(guard_blocklist);
  guardrail.SetOPAEndpoint(opa_endpoint);
  bool audit_debug_mode = false;
  if (const char *env_audit_debug = std::getenv("INFERFLUX_AUDIT_DEBUG")) {
    audit_debug_mode = std::string(env_audit_debug) == "true";
  }
  inferflux::AuditLogger audit_logger(audit_log_path, audit_debug_mode);
  std::cout << "Speculative decoding: "
            << (speculative_enabled ? "enabled" : "disabled") << " draft_model="
            << (speculative_draft_model.empty() ? "<none>"
                                                : speculative_draft_model)
            << " max_prefill_tokens=" << speculative_max_prefill_tokens
            << " chunk_size=" << speculative_chunk_size << std::endl;
  if (!nvme_offload_path.empty()) {
    std::cout << "NVMe KV offload path: " << nvme_offload_path
              << " (workers=" << nvme_writer_workers
              << ", queue_depth=" << nvme_writer_queue_depth << ")"
              << std::endl;
  }
  std::cout << "Scheduler pools (prefill/decode): "
            << disagg_config.prefill_pool_size << "/"
            << disagg_config.decode_pool_size
            << " kv_transport=" << kv_transport_type
            << " capacity=" << kv_channel_capacity << " kv_enqueue_max_retries="
            << disagg_config.kv_enqueue_max_retries
            << " max_batch_size=" << scheduler_config.max_batch_size
            << " max_batch_tokens=" << scheduler_config.max_batch_tokens
            << std::endl;
  if (!opa_endpoint.empty()) {
    std::cout << "OPA guardrail endpoint: " << opa_endpoint << std::endl;
  }

  // --- Startup Advisor: log-only recommendations ---
  {
    inferflux::StartupAdvisorContext advisor_ctx;
    // Snapshot loaded models.
    for (const auto &mi : router->ListModels()) {
      inferflux::AdvisorModelInfo am;
      am.id = mi.id;
      am.path = mi.path;
      am.format = mi.format;
      am.backend = mi.backend;
      am.backend_provider = mi.backend_provider;
      am.backend_fallback = mi.backend_fallback;
      am.backend_fallback_reason = mi.backend_fallback_reason;
      am.is_moe = mi.is_moe;
      am.n_experts = mi.n_experts;
      std::error_code ec;
      auto fsize = std::filesystem::file_size(mi.path, ec);
      if (!ec)
        am.file_size_bytes = static_cast<std::uint64_t>(fsize);
      // Detect quantization from filename for accurate memory calculations
      am.quantization = inferflux::DetectQuantization(am.path, am.format);
      advisor_ctx.models.push_back(am);
    }
    // Probe GPU (post-model-load to get real free VRAM).
    advisor_ctx.gpu = inferflux::ProbeCudaGpu();
    if (!advisor_ctx.gpu.available) {
      advisor_ctx.gpu = inferflux::ProbeRocmGpu();
    }
    // Snapshot config.
    advisor_ctx.config.cuda_enabled = cuda_enabled;
    advisor_ctx.config.flash_attention_enabled = cuda_flash_attention_enabled;
    advisor_ctx.config.cuda_attention_kernel = cuda_attention_kernel;
    advisor_ctx.config.phase_overlap_enabled = cuda_phase_overlap_scaffold;
    advisor_ctx.config.max_batch_size = scheduler_config.max_batch_size;
    advisor_ctx.config.max_batch_tokens = scheduler_config.max_batch_tokens;
    advisor_ctx.config.kv_cpu_pages = paged_kv_pages;
    advisor_ctx.config.prefer_inferflux = backend_prefer_inferflux;
    advisor_ctx.config.allow_llama_cpp_fallback =
        backend_allow_llama_cpp_fallback;
    advisor_ctx.config.strict_inferflux_request =
        backend_strict_inferflux_request;
    advisor_ctx.config.backend_priority =
        normalized_backend_priority.empty()
            ? ""
            : normalized_backend_priority.front();
    advisor_ctx.config.tp_degree = dist_world_size;
    advisor_ctx.config.speculative_enabled = speculative_enabled;
    inferflux::RunStartupAdvisor(advisor_ctx);
  }

  int http_workers = 16; // Increased from 4 for better concurrent throughput
  if (const char *env_workers = std::getenv("INFERFLUX_HTTP_WORKERS")) {
    try {
      http_workers = std::stoi(env_workers);
    } catch (...) {
    }
  }
  if (inferflux::ParallelContext::Get().IsInitialized() &&
      !inferflux::ParallelContext::Get().IsMaster()) {
    std::cout << "[distributed] rank=" << dist_rank
              << " entering shard worker mode\n";
    while (g_running) {
      // In a full implementation, shard workers wait for a broadcasted batch
      // from the master rank and execute it.
      // For this foundation, we just sleep.
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
  }

  inferflux::HttpServer server(
      host, port, &scheduler, auth, &metrics, &oidc_validator,
      rate_limit_per_minute > 0 ? &rate_limiter : nullptr,
      guardrail.Enabled() ? &guardrail : nullptr,
      audit_logger.Enabled() ? &audit_logger : nullptr, &policy_store,
      speculative_decoder, tls_config, http_workers, routing_selection_options);

  // §2.5 item 12: disaggregated deployment role.
  inferflux::HttpServer::PoolRole server_role =
      inferflux::HttpServer::PoolRole::kUnified;
  if (const char *env_role = std::getenv("INFERFLUX_ROLE")) {
    std::string role_str = env_role;
    if (role_str == "prefill") {
      server_role = inferflux::HttpServer::PoolRole::kPrefill;
    } else if (role_str == "decode") {
      server_role = inferflux::HttpServer::PoolRole::kDecode;
    }
  }
  server.SetRole(server_role);
  // decode_pool_ready_ is no longer the primary gate; /readyz uses
  // Scheduler::LiveDecodeWorkers() so worker thread exits are reflected live.

  if (!router->DefaultModelId().empty()) {
    server.SetModelReady(true);
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  server.Start();
  std::cout << "InferFlux listening on " << host << ":" << port
            << (tls_config.enabled ? " (TLS enabled)" : "")
            << " prefix_cache_capacity=" << prefix_cache_capacity << std::endl;

  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  server.Stop();
  std::cout << "InferFlux shutting down" << std::endl;
  return 0;
}
