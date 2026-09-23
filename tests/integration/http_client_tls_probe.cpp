#ifdef _WIN32
#include <winsock2.h>
#endif
#include "net/http_client.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
#ifdef _WIN32
  WSADATA winsock;
  if (WSAStartup(MAKEWORD(2, 2), &winsock) != 0)
    return 2;
#endif
  try {
    inferflux::HttpClient client;
    if (std::string(argv[1]) == "raw") {
      auto connection = client.SendRaw("GET", argv[2], "", {});
      char buffer[1024];
      const auto count = client.RecvRaw(connection, buffer, sizeof(buffer));
      client.CloseRaw(connection);
      return count > 0 ? 0 : 3;
    }
    return client.Get(argv[2]).status == 200 ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
