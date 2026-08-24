#include "net/http_client.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <cerrno>

#include <openssl/err.h>
#include <openssl/x509v3.h>

#include <sstream>
#include <stdexcept>

namespace inferflux {
namespace {
struct ParsedUrl {
  std::string scheme{"http"};
  std::string host;
  std::string path{"/"};
  int port{80};
  bool use_tls{false};
};

ParsedUrl ParseUrl(const std::string &url) {
  ParsedUrl parsed;
  std::string remainder = url;
  auto scheme_pos = url.find("://");
  if (scheme_pos != std::string::npos) {
    parsed.scheme = url.substr(0, scheme_pos);
    remainder = url.substr(scheme_pos + 3);
  }
  parsed.use_tls = (parsed.scheme == "https");
  parsed.port = parsed.use_tls ? 443 : 80;

  auto slash = remainder.find('/');
  std::string host_port =
      slash == std::string::npos ? remainder : remainder.substr(0, slash);
  parsed.path = slash == std::string::npos ? "/" : remainder.substr(slash);

  auto colon = host_port.find(':');
  if (colon == std::string::npos) {
    parsed.host = host_port;
  } else {
    parsed.host = host_port.substr(0, colon);
    parsed.port = std::stoi(host_port.substr(colon + 1));
  }
  if (parsed.host.empty()) {
    throw std::runtime_error("invalid URL host");
  }
  return parsed;
}

int CreateSocket(const ParsedUrl &parsed) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *result = nullptr;
  if (getaddrinfo(parsed.host.c_str(), std::to_string(parsed.port).c_str(),
                  &hints, &result) != 0) {
    throw std::runtime_error("failed to resolve host");
  }
  int sock = -1;
  for (addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
    sock = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock == -1)
      continue;
    if (::connect(sock, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0)
      break;
#ifdef _WIN32
    ::closesocket(sock);
#else
    ::close(sock);
#endif
    sock = -1;
  }
  freeaddrinfo(result);
  if (sock == -1)
    throw std::runtime_error("failed to connect");
#ifdef _WIN32
  DWORD timeout_ms = 30000;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
             reinterpret_cast<const char *>(&timeout_ms), sizeof(timeout_ms));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
             reinterpret_cast<const char *>(&timeout_ms), sizeof(timeout_ms));
#else
  struct timeval tv;
  tv.tv_sec = 30;
  tv.tv_usec = 0;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
  return sock;
}

std::string BuildRequest(const ParsedUrl &parsed, const std::string &method,
                         const std::string &body,
                         const std::map<std::string, std::string> &headers) {
  std::ostringstream request;
  request << method << " " << parsed.path << " HTTP/1.1\r\n";
  request << "Host: " << parsed.host << "\r\n";
  request << "Content-Length: " << body.size() << "\r\n";
  request << "Content-Type: application/json\r\n";
  for (const auto &[key, value] : headers) {
    request << key << ": " << value << "\r\n";
  }
  request << "Connection: close\r\n\r\n";
  request << body;
  return request.str();
}
} // namespace

HttpClient::HttpClient() {
  SSL_load_error_strings();
  OpenSSL_add_ssl_algorithms();
  ssl_ctx_ = SSL_CTX_new(TLS_client_method());
  if (ssl_ctx_) {
    SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_default_verify_paths(ssl_ctx_);
    tls_ready_ = true;
  }
}

HttpClient::~HttpClient() {
  if (ssl_ctx_) {
    SSL_CTX_free(ssl_ctx_);
    ssl_ctx_ = nullptr;
  }
}

HttpResponse
HttpClient::Get(const std::string &url,
                const std::map<std::string, std::string> &headers) const {
  return Send("GET", url, "", headers);
}

HttpResponse
HttpClient::Post(const std::string &url, const std::string &body,
                 const std::map<std::string, std::string> &headers) const {
  return Send("POST", url, body, headers);
}

HttpResponse
HttpClient::Put(const std::string &url, const std::string &body,
                const std::map<std::string, std::string> &headers) const {
  return Send("PUT", url, body, headers);
}

HttpResponse
HttpClient::Delete(const std::string &url, const std::string &body,
                   const std::map<std::string, std::string> &headers) const {
  return Send("DELETE", url, body, headers);
}

HttpClient::RawConnection
HttpClient::SendRaw(const std::string &method, const std::string &url,
                    const std::string &body,
                    const std::map<std::string, std::string> &headers) const {
  auto parsed = ParseUrl(url);
  RawConnection conn;
  conn.sock = CreateSocket(parsed);
  auto payload = BuildRequest(parsed, method, body, headers);
  const char *send_ptr = payload.c_str();
  std::size_t send_remaining = payload.size();

  auto close_connection = [&](const char *message) {
    CloseRaw(conn);
    throw std::runtime_error(message);
  };

  if (parsed.use_tls) {
    if (!tls_ready_) {
      close_connection("TLS not available in HttpClient");
    }
    conn.ssl = SSL_new(ssl_ctx_);
    if (!conn.ssl) {
      close_connection("failed to allocate TLS context");
    }
    SSL_set_tlsext_host_name(conn.ssl, parsed.host.c_str());
#if defined(SSL_set1_host)
    SSL_set1_host(conn.ssl, parsed.host.c_str());
#endif
    SSL_set_fd(conn.ssl, conn.sock);
    if (SSL_connect(conn.ssl) != 1) {
      close_connection("TLS handshake failed");
    }
    if (SSL_get_verify_result(conn.ssl) != X509_V_OK) {
      close_connection("TLS certificate verification failed");
    }
    while (send_remaining > 0) {
      int sent =
          SSL_write(conn.ssl, send_ptr, static_cast<int>(send_remaining));
      if (sent <= 0) {
        close_connection("failed to send TLS request");
      }
      send_ptr += sent;
      send_remaining -= static_cast<std::size_t>(sent);
    }
  } else {
    while (send_remaining > 0) {
      ssize_t sent =
          ::send(conn.sock, send_ptr, static_cast<int>(send_remaining), 0);
      if (sent <= 0) {
        close_connection("failed to send request");
      }
      send_ptr += sent;
      send_remaining -= static_cast<std::size_t>(sent);
    }
  }

  return conn;
}

ssize_t HttpClient::RecvRaw(RawConnection &conn, char *buffer,
                            std::size_t length) const {
  if (conn.ssl) {
    while (true) {
      int received = SSL_read(conn.ssl, buffer, static_cast<int>(length));
      if (received > 0) {
        return received;
      }
      int err = SSL_get_error(conn.ssl, received);
      if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
        continue;
      }
      return -1;
    }
  }
  while (true) {
    ssize_t received = ::recv(conn.sock, buffer, static_cast<int>(length), 0);
#ifdef _WIN32
    if (received < 0 && WSAGetLastError() == WSAEINTR) {
      continue;
    }
#else
    if (received < 0 && errno == EINTR) {
      continue;
    }
#endif
    return received;
  }
}

void HttpClient::CloseRaw(RawConnection &conn) const {
  if (conn.ssl) {
    SSL_shutdown(conn.ssl);
    SSL_free(conn.ssl);
    conn.ssl = nullptr;
  }
  if (conn.sock >= 0) {
#ifdef _WIN32
    ::closesocket(conn.sock);
#else
    ::close(conn.sock);
#endif
    conn.sock = -1;
  }
}

HttpResponse
HttpClient::Send(const std::string &method, const std::string &url,
                 const std::string &body,
                 const std::map<std::string, std::string> &headers) const {
  auto parsed = ParseUrl(url);
  int sock = CreateSocket(parsed);
  auto payload = BuildRequest(parsed, method, body, headers);
  std::string response;

  auto close_socket = [&]() {
    if (sock != -1) {
#ifdef _WIN32
      ::closesocket(sock);
#else
      ::close(sock);
#endif
      sock = -1;
    }
  };

  if (parsed.use_tls) {
    if (!tls_ready_) {
      close_socket();
      throw std::runtime_error("TLS not available in HttpClient");
    }
    SSL *ssl = SSL_new(ssl_ctx_);
    if (!ssl) {
      close_socket();
      throw std::runtime_error("failed to allocate TLS context");
    }
    SSL_set_tlsext_host_name(ssl, parsed.host.c_str());
#if defined(SSL_set1_host)
    SSL_set1_host(ssl, parsed.host.c_str());
#endif
    SSL_set_fd(ssl, sock);
    if (SSL_connect(ssl) != 1) {
      SSL_free(ssl);
      close_socket();
      throw std::runtime_error("TLS handshake failed");
    }
    if (SSL_get_verify_result(ssl) != X509_V_OK) {
      SSL_shutdown(ssl);
      SSL_free(ssl);
      close_socket();
      throw std::runtime_error("TLS certificate verification failed");
    }
    const char *send_ptr = payload.c_str();
    std::size_t send_remaining = payload.size();
    while (send_remaining > 0) {
      int sent = SSL_write(ssl, send_ptr, static_cast<int>(send_remaining));
      if (sent <= 0) {
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close_socket();
        throw std::runtime_error("failed to send TLS request");
      }
      send_ptr += sent;
      send_remaining -= static_cast<std::size_t>(sent);
    }
    char buffer[4096];
    int read_bytes = 0;
    while ((read_bytes = SSL_read(ssl, buffer, sizeof(buffer))) > 0) {
      response.append(buffer, buffer + read_bytes);
    }
    SSL_shutdown(ssl);
    SSL_free(ssl);
  } else {
    const char *send_ptr = payload.c_str();
    std::size_t send_remaining = payload.size();
    while (send_remaining > 0) {
      ssize_t sent =
          ::send(sock, send_ptr, static_cast<int>(send_remaining), 0);
      if (sent <= 0) {
        close_socket();
        throw std::runtime_error("failed to send request");
      }
      send_ptr += sent;
      send_remaining -= static_cast<std::size_t>(sent);
    }
    char buffer[4096];
    ssize_t read_bytes = 0;
    while ((read_bytes = ::recv(sock, buffer, static_cast<int>(sizeof(buffer)),
                                0)) > 0) {
      response.append(buffer, buffer + read_bytes);
    }
  }

  close_socket();

  HttpResponse http_response;
  auto header_end = response.find("\r\n\r\n");
  std::string header = header_end == std::string::npos
                           ? response
                           : response.substr(0, header_end);
  std::string body_str = header_end == std::string::npos
                             ? std::string()
                             : response.substr(header_end + 4);
  auto status_pos = header.find(' ');
  if (status_pos != std::string::npos) {
    try {
      http_response.status = std::stoi(header.substr(status_pos + 1));
    } catch (...) {
      http_response.status = 0;
    }
  }
  http_response.body = body_str;
  return http_response;
}

} // namespace inferflux
