#include <catch2/catch_amalgamated.hpp>

#include <any>
#include <memory>
#include <string>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#define private public
#include "server/http/http_server.h"
#undef private

#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"
#include "server/auth/api_key_auth.h"
#include "server/metrics/metrics.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace inferflux {
namespace {

std::unique_ptr<Scheduler>
MakeSchedulerWithPrefix(SimpleTokenizer &tokenizer,
                        std::shared_ptr<PagedKVCache> cache,
                        std::shared_ptr<RadixPrefixCache> prefix_cache) {
  auto device = std::make_shared<CPUDeviceContext>();
  DisaggregatedConfig disagg_config;
  return std::make_unique<Scheduler>(tokenizer, device, std::move(cache),
                                     nullptr, nullptr, std::move(prefix_cache),
                                     FairnessConfig{}, disagg_config);
}

} // namespace

#ifndef _WIN32 // socketpair() not available on Windows
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

class ToolStreamBackend final : public BackendInterface {
public:
  bool LoadModel(const std::filesystem::path &,
                 const LlamaBackendConfig &) override {
    return true;
  }
  bool IsReady() const override { return true; }
  std::string Name() const override { return "tool_stream_test"; }
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &) override {
    return {};
  }
  PrefillResult Prefill(const std::string &, int) override {
    PrefillResult result;
    result.ok = true;
    result.n_past = 1;
    return result;
  }
  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }
  std::string
  Generate(const std::string &, int,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk,
           const std::function<bool()> &, int, std::vector<TokenLogprob> *,
           const std::vector<std::string> &) override {
    const std::string text =
        R"(<think>plan</think>Before <tool_call>{"name":"read","arguments":{}}</tool_call> After)";
    if (on_chunk)
      on_chunk(text, nullptr);
    return text;
  }
  std::string Decode(int, int, int max_tokens,
                     const std::function<bool(const std::string &,
                                              const TokenLogprob *)> &on_chunk,
                     const std::function<bool()> &stop, int top_n,
                     std::vector<TokenLogprob> *logprobs, int,
                     const std::vector<std::string> &stop_sequences) override {
    return Generate("", max_tokens, on_chunk, stop, top_n, logprobs,
                    stop_sequences);
  }
};

TEST_CASE(
    "HttpServer live tool stream retains reasoning prose and terminal usage",
    "[http_server][tool_calls][reasoning]") {
  SimpleTokenizer tokenizer;
  auto backend = std::make_shared<ToolStreamBackend>();
  ModelInfo info;
  info.id = "tool-stream-model";
  info.ready = true;
  auto router = std::make_shared<SingleModelRouter>(backend, info);
  MetricsRegistry metrics;
  Scheduler::Config config;
  config.metrics = &metrics;
  Scheduler scheduler(tokenizer, std::make_shared<CPUDeviceContext>(), nullptr,
                      router, nullptr, nullptr, {}, {}, ModelSelectionOptions{},
                      config);
  auto auth = std::make_shared<ApiKeyAuth>();
  auth->AddKey("test-key", {"generate"});
  HttpServer server("127.0.0.1", 0, &scheduler, auth, &metrics, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr,
                    HttpServer::TlsConfig{}, 1);
  server.SetModelReady(true);
  server.Start();
  sockaddr_in address{};
  socklen_t length = sizeof(address);
  bool listening = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    const int fd = server.server_fd_.load();
    if (fd >= 0 &&
        ::getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) ==
            0 &&
        address.sin_port != 0) {
      listening = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(listening);
  struct Socket {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ~Socket() {
      if (fd >= 0)
        ::close(fd);
    }
  } client;
  REQUIRE(client.fd >= 0);
  timeval timeout{5, 0};
  REQUIRE(::setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       sizeof(timeout)) == 0);
  REQUIRE(::connect(client.fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0);
  const std::string body =
      R"({"model":"tool-stream-model","messages":[{"role":"user","content":"read"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":256})";
  const std::string request =
      "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Authorization: Bearer test-key\r\nContent-Type: application/json\r\n"
      "Connection: close\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  REQUIRE(::write(client.fd, request.data(), request.size()) ==
          static_cast<ssize_t>(request.size()));
  const auto response = ReadAll(client.fd);
  server.Stop();
  REQUIRE(response.find("HTTP/1.1 200 OK") != std::string::npos);
  REQUIRE(response.find("data: [DONE]") != std::string::npos);
  std::string reasoning, content, tool_name;
  int usage_frames = 0, finish_frames = 0;
  for (std::size_t pos = response.find("data: "); pos != std::string::npos;) {
    const auto end = response.find("\n\n", pos);
    REQUIRE(end != std::string::npos);
    const auto data = response.substr(pos + 6, end - pos - 6);
    if (data != "[DONE]") {
      const auto frame = json::parse(data);
      if (frame.contains("usage")) {
        ++usage_frames;
        REQUIRE(frame["usage"]["completion_tokens_details"]["reasoning_tokens"]
                    .get<int>() > 0);
        REQUIRE(frame["usage"]["completion_tokens"].get<int>() > 0);
      }
      for (const auto &choice : frame["choices"]) {
        const auto &delta = choice["delta"];
        reasoning += delta.value("reasoning_content", std::string{});
        if (delta.contains("content") && delta["content"].is_string())
          content += delta["content"].get<std::string>();
        if (delta.contains("tool_calls")) {
          for (const auto &call : delta["tool_calls"])
            tool_name += call["function"].value("name", std::string{});
        }
        if (!choice["finish_reason"].is_null()) {
          REQUIRE(choice["finish_reason"] == "tool_calls");
          ++finish_frames;
        }
      }
    }
    pos = response.find("data: ", end + 2);
  }
  REQUIRE(reasoning == "plan");
  REQUIRE(content == "Before  After");
  REQUIRE(tool_name == "read");
  REQUIRE(usage_frames == 1);
  REQUIRE(finish_frames == 1);
}

TEST_CASE("HttpServer admin cache endpoint includes memory payload",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto paged_kv = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      paged_kv, [](int, std::shared_ptr<inferflux::BackendInterface>) {},
      RadixPrefixCacheLimits{64, 8});
  auto scheduler = MakeSchedulerWithPrefix(tokenizer, paged_kv, prefix_cache);

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

  const std::string request = "GET /v1/admin/cache HTTP/1.1\r\n"
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
  REQUIRE(body["memory"]["inferflux_cuda_model"]["domains"]["kv_cache"]
              ["reserved_bytes"] == 1024);

  REQUIRE(body["memory"]["inferflux_cuda_kv"]["total_bytes"] == 2048);
  REQUIRE(body["memory"]["inferflux_cuda_kv"]["prefix_retained_bytes"] == 512);
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
#endif // _WIN32

} // namespace inferflux
