#include <catch2/catch_amalgamated.hpp>

#include <any>
#include <memory>
#include <string>

#define private public
#include "server/http/http_server.h"
#undef private

#ifndef _WIN32
#include <csignal>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace inferflux {
namespace {

// Run with the operating system's default SIGPIPE disposition, independently of
// Catch2 or a parent shell that might ignore it. A regression kills only this
// child; waitpid lets the test report the actual terminating signal.
int ExerciseClosedPeer(bool tls, bool pending_signal) {
  std::signal(SIGPIPE, SIG_DFL);
  sigset_t pipe_set;
  sigemptyset(&pipe_set);
  sigaddset(&pipe_set, SIGPIPE);
  if (pthread_sigmask(pending_signal ? SIG_BLOCK : SIG_UNBLOCK, &pipe_set,
                      nullptr) != 0) {
    return 10;
  }
  if (pending_signal && std::raise(SIGPIPE) != 0) {
    return 11;
  }

  HttpServer server("127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, HttpServer::TlsConfig{},
                    1);
  int closed[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, closed) != 0) {
    return 12;
  }
  ::close(closed[1]);
  HttpServer::ClientSession session;
  session.fd = closed[0];
  SSL_CTX *context = nullptr;
  if (tls) {
    context = SSL_CTX_new(TLS_method());
    if (!context) {
      return 13;
    }
    session.ssl = SSL_new(context);
    if (!session.ssl || SSL_set_fd(session.ssl, session.fd) != 1) {
      return 14;
    }
    // SSL_write must write the handshake to its socket BIO before application
    // data. A closed peer exercises OpenSSL's underlying write, too.
    SSL_set_connect_state(session.ssl);
  }
  if (server.SendAll(session, "data: first token\n\n")) {
    return 15;
  }
  if (session.ssl) {
    SSL_free(session.ssl);
    SSL_CTX_free(context);
  }
  ::close(closed[0]);

  sigset_t current;
  if (pthread_sigmask(SIG_BLOCK, nullptr, &current) != 0 ||
      sigismember(&current, SIGPIPE) != static_cast<int>(pending_signal)) {
    return 16;
  }
  struct sigaction action{};
  if (sigaction(SIGPIPE, nullptr, &action) != 0 ||
      action.sa_handler != SIG_DFL) {
    return 17;
  }
  if (sigpending(&current) != 0 ||
      sigismember(&current, SIGPIPE) != static_cast<int>(pending_signal)) {
    return 18;
  }
  if (pending_signal) {
    int received = 0;
    if (sigwait(&pipe_set, &received) != 0 || received != SIGPIPE ||
        pthread_sigmask(SIG_UNBLOCK, &pipe_set, nullptr) != 0) {
      return 19;
    }
  }

  // A failed response must not prevent the same server from writing a later
  // healthy connection. No model, listener port or scheduler threads required.
  int healthy[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, healthy) != 0) {
    return 20;
  }
  session.fd = healthy[0];
  session.ssl = nullptr;
  const std::string payload = "healthy";
  if (!server.SendAll(session, payload)) {
    return 21;
  }
  char buffer[16];
  const ssize_t size = ::read(healthy[1], buffer, sizeof(buffer));
  ::close(healthy[0]);
  ::close(healthy[1]);
  return size == static_cast<ssize_t>(payload.size()) &&
                 std::string(buffer, static_cast<std::size_t>(size)) == payload
             ? 0
             : 22;
}

} // namespace

TEST_CASE("HTTP closed peers return write failure without terminating server",
          "[http_server][closed_peer]") {
  const bool tls = GENERATE(false, true);
  const bool pending_signal = GENERATE(false, true);
  CAPTURE(tls, pending_signal);
  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    // Bound a broken handshake or accidentally blocking signal wait.
    ::alarm(5);
    ::_exit(ExerciseClosedPeer(tls, pending_signal));
  }
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  if (WIFSIGNALED(status)) {
    INFO("Child terminated by signal " << WTERMSIG(status));
    REQUIRE_FALSE(WIFSIGNALED(status));
  }
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

} // namespace inferflux
#endif
