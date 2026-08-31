#include <catch2/catch_amalgamated.hpp>

#include "runtime/disaggregated/kv_channel.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/scheduler.h"
#include "server/http/completion_payload.h"
#include "server/http/http_server.h"
#include "server/metrics/metrics.h"
#include "support/scoped_env.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>

using namespace inferflux;

namespace {

// ScopedEnvVar is provided by support/scoped_env.h.
using inferflux::test::ScopedEnvVar;

bool WaitForCondition(
    const std::function<bool()> &predicate,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

std::unique_ptr<Scheduler> MakeScheduler(SimpleTokenizer &tokenizer,
                                         bool with_transport) {
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  if (with_transport) {
    disagg_config.kv_transport = std::make_shared<disaggregated::KVChannel>(8);
  }
  return std::make_unique<Scheduler>(tokenizer, device, cache, nullptr, nullptr,
                                     nullptr, FairnessConfig{}, disagg_config);
}

std::unique_ptr<HttpServer> MakeServer(Scheduler *scheduler,
                                       MetricsRegistry *metrics) {
  return std::make_unique<HttpServer>(
      "127.0.0.1", 0, scheduler, nullptr, metrics, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, HttpServer::TlsConfig{}, 1);
}

} // namespace

TEST_CASE("Completion JSON replaces malformed model UTF-8",
          "[http_server][utf8]") {
  const std::string malformed(1, static_cast<char>(0x9B));
  const nlohmann::json payload = {
      {"choices", {{{"message", {{"content", malformed}}}}}}};

  const auto encoded = SerializeJsonUtf8Safe(payload);
  const auto decoded = nlohmann::json::parse(encoded);

  REQUIRE(decoded["choices"][0]["message"]["content"] == "\xEF\xBF\xBD");
}

TEST_CASE("Streaming logprobs preserve raw token bytes",
          "[http_server][utf8]") {
  TokenLogprob logprob;
  logprob.token = std::string(1, static_cast<char>(0x9B));
  logprob.logprob = -0.5f;
  logprob.bytes = {0x9B};

  const auto event = BuildStreamChunkForTest("\xEF\xBF\xBD", &logprob);
  REQUIRE(event.rfind("data: ", 0) == 0);
  const auto payload = event.substr(6, event.size() - 8);
  const auto decoded = nlohmann::json::parse(payload);
  const auto &entry = decoded["choices"][0]["logprobs"]["content"][0];

  REQUIRE(entry["token"] == "\xEF\xBF\xBD");
  REQUIRE(entry["bytes"] == nlohmann::json::array({0x9B}));
  REQUIRE(decoded["choices"][0]["delta"]["content"] == "\xEF\xBF\xBD");
}

TEST_CASE("LookupHeaderValueForTest matches header names case-insensitively",
          "[http_server]") {
  const std::string headers =
      "POST /v1/completions HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "X-InferFlux-Client-Request-Id: bench-5\r\n"
      "traceparent: 00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01\r\n"
      "\r\n";

  REQUIRE(LookupHeaderValueForTest(headers, "x-inferflux-client-request-id") ==
          "bench-5");
  REQUIRE(LookupHeaderValueForTest(headers, "X-INFERFLUX-CLIENT-REQUEST-ID") ==
          "bench-5");
  REQUIRE(LookupHeaderValueForTest(headers, "TraceParent") ==
          "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01");
}

TEST_CASE("ExtractBearerToken parses any header or scheme casing",
          "[http_server]") {
  const std::string token = "dev-key-123";

  // RFC 9110 §5.1 (field names) and §11.1 (auth-scheme) are case-insensitive.
  REQUIRE(ExtractBearerToken("Authorization: Bearer " + token) == token);
  REQUIRE(ExtractBearerToken("authorization: Bearer " + token) == token);
  REQUIRE(ExtractBearerToken("AUTHORIZATION: bearer " + token) == token);
  REQUIRE(ExtractBearerToken("Authorization: bEaReR   " + token) == token);

  // Header sits among other headers, not first.
  REQUIRE(ExtractBearerToken("Host: 127.0.0.1\r\n"
                             "Content-Type: application/json\r\n"
                             "authorization: Bearer " +
                             token + "\r\n") == token);

  // Non-bearer schemes, missing headers, and bare schemes yield no token.
  REQUIRE(ExtractBearerToken("Authorization: Basic dXNlcjpwYXNz").empty());
  REQUIRE(ExtractBearerToken("Content-Type: application/json").empty());
  REQUIRE(ExtractBearerToken("Authorization: Bearer").empty());
  REQUIRE(ExtractBearerToken("").empty());
}

TEST_CASE("HttpServer ready status reports healthy distributed decode pool",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE(status.model_loaded);
  REQUIRE(status.decode_pool_warm);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 0);
  REQUIRE(status.disagg_timeout_debt_threshold == 6);
  REQUIRE(status.disagg_timeout_streak == 0);
  REQUIRE(status.disagg_timeout_streak_threshold == 3);
  REQUIRE(status.role == "decode");
}

TEST_CASE(
    "HttpServer ready status fails when distributed KV timeout streak hits "
    "threshold",
    "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(status.ready);
  REQUIRE(status.model_loaded);
  REQUIRE(status.decode_pool_warm);
  REQUIRE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 3);
  REQUIRE(status.disagg_timeout_debt_threshold == 6);
  REQUIRE(status.disagg_timeout_streak == 3);
  REQUIRE(status.reason == "distributed kv transport degraded");
}

TEST_CASE("HttpServer ready status fails when distributed KV timeout debt hits "
          "threshold despite intervening commit",
          "[http_server]") {
  ScopedEnvVar streak_threshold(
      "INFERFLUX_READYZ_DISAGG_TIMEOUT_STREAK_THRESHOLD", "5");
  ScopedEnvVar debt_threshold("INFERFLUX_READYZ_DISAGG_TIMEOUT_DEBT_THRESHOLD",
                              "2");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("committed");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(status.ready);
  REQUIRE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 2);
  REQUIRE(status.disagg_timeout_debt_threshold == 2);
  REQUIRE(status.disagg_timeout_streak == 2);
  REQUIRE(status.disagg_timeout_streak_threshold == 5);
  REQUIRE(status.reason == "distributed kv transport degraded");
}

TEST_CASE("HttpServer ready status recovers after a committed KV handoff",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("committed");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 2);
  REQUIRE(status.disagg_timeout_streak == 0);
}

TEST_CASE("HttpServer generation admission stays open by default when "
          "distributed transport is degraded",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE(decision.allowed);
  REQUIRE(decision.error.empty());
}

TEST_CASE("HttpServer generation admission fails closed when configured and "
          "distributed transport is degraded",
          "[http_server]") {
  ScopedEnvVar fail_closed("INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED",
                           "true");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE_FALSE(decision.allowed);
  REQUIRE(decision.http_status == 503);
  REQUIRE(decision.error == "distributed_kv_transport_degraded");
  REQUIRE(decision.reason == "distributed kv transport degraded");
}

TEST_CASE("HttpServer generation admission ignores fail-closed policy without "
          "KV transport",
          "[http_server]") {
  ScopedEnvVar fail_closed("INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED",
                           "true");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, false);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE(decision.allowed);
  REQUIRE(decision.error.empty());
}

TEST_CASE("HttpServer ignores distributed timeout streak without KV transport",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, false);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_streak == 0);
}

TEST_CASE(
    "HttpServer admin pools status mirrors readiness and scheduler gauges",
    "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  metrics.SetQueueDepth(7);
  metrics.SetPrefillQueueDepth(2);
  metrics.SetDecodeQueueDepth(5);
  metrics.SetSchedulerBatchLimits(4, 8192);
  metrics.RecordDisaggKVEnqueueRejected(false);
  metrics.RecordDisaggKVEnqueueRejected(true);

  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("enqueued");
  metrics.RecordDisaggKVTicketStage("enqueued");
  metrics.RecordDisaggKVTicketStage("acknowledged");
  metrics.RecordDisaggKVTicketStage("committed");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  REQUIRE(metrics.GetDisaggKVTimeoutStreak() == 3);
  REQUIRE(metrics.GetDisaggKVTimeoutDebt() == 3);

  const auto ready_status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(ready_status.ready);
  REQUIRE(ready_status.disagg_transport_degraded);
  REQUIRE(ready_status.disagg_timeout_debt == 3);
  REQUIRE(ready_status.disagg_timeout_debt_threshold == 6);

  const auto status = server->EvaluateAdminPoolsStatus();
  REQUIRE_FALSE(status.pool_health.ready);
  REQUIRE(status.pool_health.disagg_transport_degraded);
  REQUIRE(status.pool_health.disagg_timeout_debt == 3);
  REQUIRE(status.queue_depth == 7);
  REQUIRE(status.prefill_queue_depth == 2);
  REQUIRE(status.decode_queue_depth == 5);
  REQUIRE(status.batch_limit_size == 4);
  REQUIRE(status.batch_limit_tokens == 8192);
  REQUIRE(status.distributed_kv.enqueue_rejections_total == 2);
  REQUIRE(status.distributed_kv.enqueue_exhausted_total == 1);
  REQUIRE(status.distributed_kv.tickets_enqueued_total == 2);
  REQUIRE(status.distributed_kv.tickets_acknowledged_total == 1);
  REQUIRE(status.distributed_kv.tickets_committed_total == 1);
  REQUIRE(status.distributed_kv.tickets_timed_out_total == 3);
}

TEST_CASE("HttpServer admin pools status tolerates missing metrics registry",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  auto server = MakeServer(scheduler.get(), nullptr);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(
      WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  const auto status = server->EvaluateAdminPoolsStatus();
  REQUIRE(status.pool_health.ready);
  REQUIRE(status.pool_health.disagg_timeout_debt == 0);
  REQUIRE_FALSE(status.queue_depth.has_value());
  REQUIRE_FALSE(status.prefill_queue_depth.has_value());
  REQUIRE_FALSE(status.decode_queue_depth.has_value());
  REQUIRE_FALSE(status.batch_limit_size.has_value());
  REQUIRE_FALSE(status.batch_limit_tokens.has_value());
  REQUIRE_FALSE(status.distributed_kv.enqueue_rejections_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.enqueue_exhausted_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_enqueued_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_acknowledged_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_committed_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_timed_out_total.has_value());
}
