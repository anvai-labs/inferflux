// Dual-listener TLS tests: plain main listener + dedicated HTTPS listener.
// Runs the real acceptors on loopback with a disposable self-signed
// certificate; skipped entirely when the openssl CLI is unavailable (and on
// Windows, matching the POSIX socket harness below).
#include <catch2/catch_amalgamated.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#ifdef _WIN32
#define INFERFLUX_TLS_TEST_NO_POSIX 1
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifndef INFERFLUX_TLS_TEST_NO_POSIX
#include <openssl/err.h>
#include <openssl/ssl.h>

#define private public
#include "server/http/http_server.h"
#undef private

#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/metrics/metrics.h"

namespace inferflux {
namespace {

std::string ShellOut(const std::string &cmd) {
  std::string out;
  FILE *pipe = ::popen(cmd.c_str(), "r");
  if (!pipe) {
    return out;
  }
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    out.append(buffer);
  }
  ::pclose(pipe);
  return out;
}

struct CertFixture {
  std::filesystem::path dir;
  std::string cert;
  std::string key;
  bool ok{false};

  CertFixture() {
    dir = std::filesystem::temp_directory_path() /
          ("inferflux-tls-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    cert = (dir / "server.crt").string();
    key = (dir / "server.key").string();
    const std::string cmd =
        "openssl req -x509 -newkey rsa:2048 -keyout '" + key + "' -out '" +
        cert + "' -days 2 -nodes -subj /CN=localhost >/dev/null 2>&1";
    ok = std::system(cmd.c_str()) == 0 && std::filesystem::exists(cert);
  }
  ~CertFixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

// Grab a free TCP port from the kernel and release it. Small TOCTOU window,
// standard practice for listener tests.
int ReserveEphemeralPort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
  socklen_t len = sizeof(addr);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

int GetListenerPort(const HttpServer::Listener &listener) {
  sockaddr_in addr{};
  socklen_t len = sizeof(addr);
  if (listener.fd.load() < 0 ||
      ::getsockname(listener.fd.load(), reinterpret_cast<sockaddr *>(&addr),
                    &len) != 0) {
    return -1;
  }
  return ntohs(addr.sin_port);
}

std::string HttpGet(int port, const std::string &request) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  REQUIRE(::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
          0);
  REQUIRE(::send(fd, request.c_str(), request.size(), 0) >= 0);
  std::string out;
  char buffer[1024];
  while (true) {
    const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
    if (n <= 0) {
      break;
    }
    out.append(buffer, static_cast<std::size_t>(n));
  }
  ::close(fd);
  return out;
}

// Minimal TLS client: handshake without verification, one request, full read.
std::string HttpsGet(int port, const std::string &request) {
  SSL_load_error_strings();
  OpenSSL_add_ssl_algorithms();
  SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
  REQUIRE(ctx != nullptr);
  SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
  SSL *ssl = SSL_new(ctx);
  REQUIRE(ssl != nullptr);
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  REQUIRE(::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
          0);
  SSL_set_fd(ssl, fd);
  REQUIRE(SSL_connect(ssl) == 1);
  REQUIRE(SSL_write(ssl, request.c_str(), static_cast<int>(request.size())) >
          0);
  std::string out;
  char buffer[1024];
  while (true) {
    const int n = SSL_read(ssl, buffer, sizeof(buffer));
    if (n <= 0) {
      break;
    }
    out.append(buffer, static_cast<std::size_t>(n));
  }
  SSL_free(ssl);
  ::close(fd);
  SSL_CTX_free(ctx);
  return out;
}

struct ServerFixture {
  SimpleTokenizer tokenizer;
  MetricsRegistry metrics;
  std::unique_ptr<Scheduler> scheduler;
  std::shared_ptr<ApiKeyAuth> auth;

  ServerFixture() {
    auto device = std::make_shared<CPUDeviceContext>();
    scheduler = std::make_unique<Scheduler>(tokenizer, device, nullptr, nullptr,
                                            nullptr, nullptr, FairnessConfig{},
                                            DisaggregatedConfig{});
    auth = std::make_shared<ApiKeyAuth>();
  }
};

// Connection: close is REQUIRED — the server enables keep-alive for any
// request that does not ask for close (regardless of HTTP version), and an
// EOF-reading client would block forever on the held-open connection.
const char *kHealthRequest =
    "GET /healthz HTTP/1.0\r\nConnection: close\r\n\r\n";

} // namespace

TEST_CASE("Dual listener serves plain HTTP and HTTPS concurrently",
          "[http_server][tls]") {
  CertFixture certs;
  if (!certs.ok) {
    SKIP("openssl CLI unavailable; cannot build the certificate fixture");
  }
  ServerFixture fx;
  const int tls_port = ReserveEphemeralPort();

  HttpServer::TlsConfig tls;
  tls.port = tls_port;
  tls.bind_host = "127.0.0.1";
  tls.cert_path = certs.cert;
  tls.key_path = certs.key;
  HttpServer server("127.0.0.1", 0, fx.scheduler.get(), fx.auth, &fx.metrics,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, tls,
                    1);
  REQUIRE(server.Start());
  REQUIRE(server.listener_count_ == 2);

  const int plain_port = GetListenerPort(server.listeners_[0]);
  const int https_port = GetListenerPort(server.listeners_[1]);
  REQUIRE(plain_port > 0);
  REQUIRE(https_port == tls_port);

  // Plain listener: HTTP works (the zero-overhead loopback contract).
  const std::string plain = HttpGet(plain_port, kHealthRequest);
  REQUIRE(plain.find("200") != std::string::npos);

  // Dedicated listener: TLS works and answers the same route table.
  const std::string secured = HttpsGet(https_port, kHealthRequest);
  REQUIRE(secured.find("200") != std::string::npos);

  // ...and rejects plaintext: an HTTP probe to the TLS port gets no HTTP
  // response (the server tears the connection down at handshake).
  const std::string plaintext_probe = HttpGet(https_port, kHealthRequest);
  REQUIRE(plaintext_probe.find("200") == std::string::npos);

  server.Stop();
}

TEST_CASE("Dedicated TLS listener fails closed on a bad certificate",
          "[http_server][tls]") {
  CertFixture certs;
  if (!certs.ok) {
    SKIP("openssl CLI unavailable; cannot build the certificate fixture");
  }
  ServerFixture fx;
  const int tls_port = ReserveEphemeralPort();

  HttpServer::TlsConfig tls;
  tls.port = tls_port;
  tls.bind_host = "127.0.0.1";
  tls.cert_path = certs.cert + ".missing";
  tls.key_path = certs.key;
  HttpServer server("127.0.0.1", 0, fx.scheduler.get(), fx.auth, &fx.metrics,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, tls,
                    1);
  REQUIRE_FALSE(server.Start());
  REQUIRE_FALSE(server.startup_error_.empty());
  server.Stop();
}

TEST_CASE("Dedicated TLS listener fails closed on a bound TLS port",
          "[http_server][tls]") {
  CertFixture certs;
  if (!certs.ok) {
    SKIP("openssl CLI unavailable; cannot build the certificate fixture");
  }
  ServerFixture fx;
  const int tls_port = ReserveEphemeralPort();

  // Occupy the port before the server starts.
  int blocker = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(blocker >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(tls_port));
  REQUIRE(::bind(blocker, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
          0);
  REQUIRE(::listen(blocker, 4) == 0);

  HttpServer::TlsConfig tls;
  tls.port = tls_port;
  tls.bind_host = "127.0.0.1";
  tls.cert_path = certs.cert;
  tls.key_path = certs.key;
  HttpServer server("127.0.0.1", 0, fx.scheduler.get(), fx.auth, &fx.metrics,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, tls,
                    1);
  REQUIRE_FALSE(server.Start());
  ::close(blocker);
  server.Stop();
}

TEST_CASE("Legacy same-port TLS mode keeps one wrapped listener",
          "[http_server][tls]") {
  CertFixture certs;
  if (!certs.ok) {
    SKIP("openssl CLI unavailable; cannot build the certificate fixture");
  }
  ServerFixture fx;

  HttpServer::TlsConfig tls;
  tls.enabled = true;
  tls.cert_path = certs.cert;
  tls.key_path = certs.key;
  HttpServer server("127.0.0.1", 0, fx.scheduler.get(), fx.auth, &fx.metrics,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, tls,
                    1);
  REQUIRE(server.Start());
  REQUIRE(server.listener_count_ == 1);
  REQUIRE(server.listeners_[0].ssl_ctx != nullptr);

  // The legacy listener requires TLS (plaintext gets no HTTP response).
  const int port = GetListenerPort(server.listeners_[0]);
  REQUIRE(port > 0);
  const std::string plaintext = HttpGet(port, kHealthRequest);
  REQUIRE(plaintext.find("200") == std::string::npos);
  const std::string secured = HttpsGet(port, kHealthRequest);
  REQUIRE(secured.find("200") != std::string::npos);
  server.Stop();
}

} // namespace inferflux
#endif // !INFERFLUX_TLS_TEST_NO_POSIX
