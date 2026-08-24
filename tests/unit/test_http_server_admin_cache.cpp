#include <catch2/catch_amalgamated.hpp>

#include <any>
#include <memory>
#include <string>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifndef _MSC_VER
#define private public
#endif
#include "server/http/http_server.h"
#ifndef _MSC_VER
#undef private
#endif

#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/metrics/metrics.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace inferflux {
namespace {

std::unique_ptr<Scheduler> MakeSchedulerWithPrefix(SimpleTokenizer &tokenizer,
                                                   std::shared_ptr<PagedKVCache> cache,
                                                   std::shared_ptr<RadixPrefixCache> prefix_cache) {
  auto device = std::make_shared<CPUDeviceContext>();
  DisaggregatedConfig disagg_config;
  return std::make_unique<Scheduler>(tokenizer, device, std::move(cache),
                                     nullptr, nullptr,
                                     std::move(prefix_cache),
                                     FairnessConfig{}, disagg_config);
}

} // namespace

#ifndef _WIN32  // socketpair() not available on Windows
namespace {

std::string ReadAll(int fd) {
  std::string out;
  char buffer[4096];
  while (true) {
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n <= 0) {
      break;
    }
    out.append(buffer, static_cast<std::size_t>(n));
  }
  return out;
}

TEST_CASE("HttpServer admin cache endpoint includes memory payload",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto paged_kv = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      paged_kv, [](int) {}, RadixPrefixCacheLimits{64, 8});
  auto scheduler =
      MakeSchedulerWithPrefix(tokenizer, paged_kv, prefix_cache);

  const auto retained_blocks = paged_kv->ReserveBlocks(2);
  prefix_cache->Insert(std::vector<int>(32, 7), retained_blocks,
                       /*sequence_id=*/11, {});

  MetricsRegistry metrics;
  MetricsRegistry::MemoryUsageMetrics total{};
  total.reserved_bytes = 4096;
  total.in_use_bytes = 2048;
  total.high_water_bytes = 4096;
  total.evictable_bytes = 512;
  metrics.SetInferfluxCudaModelMemorySnapshot(
      "qwen2.5-3b", total,
      {{"kv_cache", {1024, 1024, 1024, 0}},
       {"batch_ephemeral", {0, 0, 512, 0}}});
  metrics.SetInferfluxCudaKvMemoryBytes(/*total_bytes=*/2048,
                                        /*active_bytes=*/1024,
                                        /*prefix_retained_bytes=*/512,
                                        /*free_bytes=*/512,
                                        /*active_sequences=*/2,
                                        /*prefix_retained_sequences=*/1,
                                        /*free_sequences=*/1,
                                        /*max_sequences=*/4);

  auto auth = std::make_shared<ApiKeyAuth>();
  auth->AddKey("admin-key", {"admin", "read", "generate"});

  HttpServer server("127.0.0.1", 0, scheduler.get(), auth, &metrics, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr,
                    HttpServer::TlsConfig{}, 1);

  int fds[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

  const std::string request =
      "GET /v1/admin/cache HTTP/1.1\r\n"
      "Host: localhost\r\n"
      "Authorization: Bearer admin-key\r\n"
      "\r\n";
  REQUIRE(::write(fds[0], request.data(),
                  static_cast<unsigned long>(request.size())) ==
          static_cast<ssize_t>(request.size()));
  REQUIRE(::shutdown(fds[0], SHUT_WR) == 0);

  HttpServer::ClientSession session;
  session.fd = fds[1];
  server.HandleClient(session);
  ::close(fds[1]);

  const std::string response = ReadAll(fds[0]);
  ::close(fds[0]);

  REQUIRE(response.find("HTTP/1.1 200 OK") != std::string::npos);
  const auto body_pos = response.find("\r\n\r\n");
  REQUIRE(body_pos != std::string::npos);

  const json body = json::parse(response.substr(body_pos + 4));
  REQUIRE(body["memory"].contains("inferflux_cuda_model"));
  REQUIRE(body["memory"].contains("inferflux_cuda_kv"));
  REQUIRE(body["memory"].contains("paged_kv"));

  REQUIRE(body["memory"]["inferflux_cuda_model"]["model"] == "qwen2.5-3b");
  REQUIRE(body["memory"]["inferflux_cuda_model"]["reserved_bytes"] == 4096);
  REQUIRE(
      body["memory"]["inferflux_cuda_model"]["domains"]["kv_cache"]
          ["reserved_bytes"] == 1024);

  REQUIRE(body["memory"]["inferflux_cuda_kv"]["total_bytes"] == 2048);
  REQUIRE(
      body["memory"]["inferflux_cuda_kv"]["prefix_retained_bytes"] == 512);
  REQUIRE(body["memory"]["inferflux_cuda_kv"]["prefix_retained_sequences"] ==
          1);

  REQUIRE(body["memory"]["paged_kv"]["total_blocks"] == 16);
  REQUIRE(body["memory"]["paged_kv"]["used_blocks"] == 2);
  REQUIRE(body["memory"]["paged_kv"]["free_blocks"] == 14);
  REQUIRE(body["memory"]["paged_kv"]["page_size_bytes"] == 1024);
  REQUIRE(body["memory"]["paged_kv"]["prefix_retained_blocks"] == 2);
  REQUIRE(body["memory"]["paged_kv"]["prefix_retained_bytes"] == 2048);
  REQUIRE(body["memory"]["paged_kv"]["prefix_live_sequences"] == 1);
}

} // namespace
#endif  // _WIN32

} // namespace inferflux
