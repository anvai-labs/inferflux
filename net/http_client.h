#pragma once

#include <cstddef>
#include <map>
#include <string>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <openssl/ssl.h>

namespace inferflux {

struct HttpResponse {
  int status{0};
  std::string body;
};

class HttpClient {
public:
  struct RawConnection {
    int sock{-1};
    SSL *ssl{nullptr};
  };

  HttpClient();
  ~HttpClient();
  HttpClient(const HttpClient &) = delete;
  HttpClient &operator=(const HttpClient &) = delete;

  HttpResponse
  Get(const std::string &url,
      const std::map<std::string, std::string> &headers = {}) const;
  HttpResponse
  Post(const std::string &url, const std::string &body,
       const std::map<std::string, std::string> &headers = {}) const;
  HttpResponse
  Put(const std::string &url, const std::string &body,
      const std::map<std::string, std::string> &headers = {}) const;
  HttpResponse
  Delete(const std::string &url, const std::string &body,
         const std::map<std::string, std::string> &headers = {}) const;

  // Returns the raw socket for streaming reads. Caller owns the socket.
  RawConnection
  SendRaw(const std::string &method, const std::string &url,
          const std::string &body,
          const std::map<std::string, std::string> &headers) const;
  ssize_t RecvRaw(RawConnection &conn, char *buffer, std::size_t length) const;
  void CloseRaw(RawConnection &conn) const;

private:
  HttpResponse Send(const std::string &method, const std::string &url,
                    const std::string &body,
                    const std::map<std::string, std::string> &headers) const;

  SSL_CTX *ssl_ctx_{nullptr};
  bool tls_ready_{false};
};

} // namespace inferflux
