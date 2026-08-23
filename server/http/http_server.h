#pragma once

#include "policy/policy_backend.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/model_selection.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/http/http_utils.h"
#include "server/logging/audit_logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"

#if INFERFLUX_ENABLE_WEBUI
#include "webui/ui_renderer.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <openssl/ssl.h>

namespace inferflux {

// Backward-compatible alias — delegates to GetHeaderValue in http_utils.h.
inline std::string LookupHeaderValueForTest(const std::string &headers,
                                            const std::string &name) {
  return GetHeaderValue(headers, name);
}

std::string BuildStreamChunkForTest(const std::string &content,
                                    const TokenLogprob *logprob);

class HttpServer {
public:
  struct TlsConfig {
    bool enabled{false};
    std::string cert_path;
    std::string key_path;
  };

  struct ReadyStatus {
    bool ready{false};
    bool model_loaded{false};
    bool decode_pool_warm{false};
    bool disagg_transport_degraded{false};
    uint64_t disagg_timeout_debt{0};
    uint64_t disagg_timeout_debt_threshold{0};
    uint64_t disagg_timeout_streak{0};
    uint64_t disagg_timeout_streak_threshold{0};
    std::string role{"unified"};
    std::string reason;
  };

  struct AdminPoolsStatus {
    struct DistributedKVStatus {
      std::optional<int64_t> enqueue_rejections_total;
      std::optional<int64_t> enqueue_exhausted_total;
      std::optional<int64_t> tickets_enqueued_total;
      std::optional<int64_t> tickets_acknowledged_total;
      std::optional<int64_t> tickets_committed_total;
      std::optional<int64_t> tickets_timed_out_total;
    };

    ReadyStatus pool_health;
    std::optional<int64_t> queue_depth;
    std::optional<int64_t> prefill_queue_depth;
    std::optional<int64_t> decode_queue_depth;
    std::optional<int64_t> batch_limit_size;
    std::optional<int64_t> batch_limit_tokens;
    DistributedKVStatus distributed_kv;
  };

  struct AdmissionDecision {
    bool allowed{true};
    int http_status{200};
    std::string error;
    std::string reason;
  };

  // Disaggregated deployment role (§2.5 item 12).
  // kUnified: combined prefill+decode (default; existing behaviour).
  // kPrefill: this instance only runs prefill; /readyz gates on model load.
  // kDecode:  this instance only runs decode; /readyz gates on decode pool.
  enum class PoolRole { kUnified, kPrefill, kDecode };

  HttpServer(std::string host, int port, Scheduler *scheduler,
             std::shared_ptr<ApiKeyAuth> auth, MetricsRegistry *metrics,
             OIDCValidator *oidc, RateLimiter *rate_limiter,
             Guardrail *guardrail, AuditLogger *audit_logger,
             PolicyBackend *policy_store,
             std::shared_ptr<SpeculativeDecoder> speculative_decoder,
             const TlsConfig &tls_config, int num_workers = 4,
             const ModelSelectionOptions &model_selection_options = {});
  ~HttpServer();

  void Start();
  void Stop();
  void SetModelReady(bool ready) { model_ready_.store(ready); }
  void SetRole(PoolRole role) { role_.store(role, std::memory_order_relaxed); }
  void SetDecodePoolReady(bool ready) {
    decode_pool_ready_.store(ready, std::memory_order_relaxed);
  }
  ReadyStatus EvaluateReadyStatus() const;
  AdminPoolsStatus EvaluateAdminPoolsStatus() const;
  AdmissionDecision EvaluateGenerationAdmissionDecision() const;

private:
  struct ClientSession {
    int fd{-1};
    SSL *ssl{nullptr};
    bool keep_alive{
        false}; // Set to true after successful non-streaming response
  };

  void Run();
  void HandleClient(ClientSession &session);

  struct AuthContext {
    std::string subject{"anonymous"};
    std::unordered_set<std::string> scopes;
  };

  bool ResolveSubject(const std::string &header_blob, AuthContext *ctx) const;
  bool RequireScope(const AuthContext &ctx, const std::string &scope,
                    ClientSession &session, const std::string &error_message);

  bool SendAll(ClientSession &session, const std::string &payload);
  ssize_t Receive(ClientSession &session, char *buffer, std::size_t length);
  void CloseSession(ClientSession &session);

  std::string host_;
  int port_;
  Scheduler *scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry *metrics_;
  OIDCValidator *oidc_;
  RateLimiter *rate_limiter_;
  Guardrail *guardrail_;
  AuditLogger *audit_logger_;
  PolicyBackend *policy_store_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  mutable std::mutex policy_update_mutex_;
  mutable std::mutex model_selection_mutex_;
  ModelSelectionOptions model_selection_options_;
  bool tls_enabled_{false};
  SSL_CTX *ssl_ctx_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> model_ready_{false};
  std::atomic<PoolRole> role_{PoolRole::kUnified};
  std::atomic<bool> decode_pool_ready_{false};
  bool admission_fail_closed_on_disagg_degraded_{false};
  int readyz_disagg_timeout_debt_threshold_{0};
  int readyz_disagg_timeout_streak_threshold_{3};
#if INFERFLUX_ENABLE_WEBUI
  std::unique_ptr<WebUiRenderer> webui_renderer_;
#endif
  std::atomic<int> server_fd_{-1};
  int num_workers_;
  std::thread accept_thread_;
  std::vector<std::thread> workers_;
  std::queue<ClientSession> client_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  void WorkerLoop();
};

} // namespace inferflux
