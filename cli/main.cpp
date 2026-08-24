#include "net/http_client.h"
#include "runtime/string_utils.h"

#include <nlohmann/json.hpp>

#include <errno.h>
#include <signal.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

// PATH_MAX is not defined on some systems by default
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

using inferflux::ToLower;
using inferflux::Trim;

struct ChatMessage {
  std::string role;
  std::string content;
};

std::string BuildUrl(const std::string &host, int port,
                     const std::string &path) {
  return "http://" + host + ":" + std::to_string(port) + path;
}

std::map<std::string, std::string> AuthHeaders(const std::string &api_key) {
  std::map<std::string, std::string> headers;
  if (!api_key.empty()) {
    headers["Authorization"] = "Bearer " + api_key;
  }
  return headers;
}

bool IsHttpSuccess(int status) { return status >= 200 && status < 300; }

bool ParseRequiredFlagValue(int argc, char **argv, int *index,
                            const std::string &command_context,
                            const std::string &flag,
                            const std::string &value_name,
                            std::string *value_out) {
  if (*index + 1 >= argc) {
    std::cerr << "inferctl " << command_context << ": " << flag << " requires "
              << value_name << std::endl;
    return false;
  }
  std::string candidate = argv[*index + 1];
  if (!candidate.empty() && candidate.front() == '-') {
    std::cerr << "inferctl " << command_context << ": " << flag << " requires "
              << value_name << std::endl;
    return false;
  }
  *value_out = candidate;
  ++(*index);
  return true;
}

bool ParseStrictBoolValue(const std::string &raw, bool *value_out) {
  if (raw == "true") {
    *value_out = true;
    return true;
  }
  if (raw == "false") {
    *value_out = false;
    return true;
  }
  return false;
}

bool ParsePrometheusMetricValue(const std::string &metrics_body,
                                const std::string &metric_name,
                                double *value_out) {
  if (!value_out || metric_name.empty()) {
    return false;
  }
  std::stringstream ss(metrics_body);
  std::string line;
  while (std::getline(ss, line)) {
    line = Trim(line);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    if (line.rfind(metric_name, 0) != 0) {
      continue;
    }
    if (line.size() > metric_name.size()) {
      char next = line[metric_name.size()];
      if (next != ' ' && next != '\t' && next != '{') {
        continue;
      }
    }
    std::size_t sep = line.find_last_of(" \t");
    if (sep == std::string::npos || sep + 1 >= line.size()) {
      continue;
    }
    std::string value_str = Trim(line.substr(sep + 1));
    try {
      *value_out = std::stod(value_str);
      return true;
    } catch (...) {
      continue;
    }
  }
  return false;
}

bool ParsePrometheusLabeledMetricValue(
    const std::string &metrics_body, const std::string &metric_name,
    const std::vector<std::pair<std::string, std::string>> &labels,
    double *value_out) {
  if (!value_out || metric_name.empty()) {
    return false;
  }
  std::stringstream ss(metrics_body);
  std::string line;
  while (std::getline(ss, line)) {
    line = Trim(line);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    if (line.rfind(metric_name, 0) != 0) {
      continue;
    }
    if (line.size() <= metric_name.size() || line[metric_name.size()] != '{') {
      continue;
    }
    const std::size_t labels_end = line.find('}', metric_name.size() + 1);
    if (labels_end == std::string::npos) {
      continue;
    }
    const std::string label_blob = line.substr(
        metric_name.size() + 1, labels_end - metric_name.size() - 1);
    bool labels_match = true;
    for (const auto &[key, value] : labels) {
      const std::string needle = key + "=\"" + value + "\"";
      if (label_blob.find(needle) == std::string::npos) {
        labels_match = false;
        break;
      }
    }
    if (!labels_match) {
      continue;
    }
    const std::size_t sep = line.find_last_of(" \t");
    if (sep == std::string::npos || sep + 1 >= line.size()) {
      continue;
    }
    const std::string value_str = Trim(line.substr(sep + 1));
    try {
      *value_out = std::stod(value_str);
      return true;
    } catch (...) {
      continue;
    }
  }
  return false;
}

int PrintJsonResponseAndReturn(const inferflux::HttpResponse &resp,
                               const std::string &auth_command_hint = "") {
  if (resp.status == 401 || resp.status == 403) {
    if (!auth_command_hint.empty()) {
      std::cerr << auth_command_hint
                << ": authentication required (set --api-key or "
                   "INFERCTL_API_KEY)\n";
    } else {
      std::cout << resp.body << std::endl;
    }
    return 1;
  }
  std::cout << resp.body << std::endl;
  return IsHttpSuccess(resp.status) ? 0 : 1;
}

std::filesystem::path InferfluxHome() {
  if (const char *env = std::getenv("INFERFLUX_HOME")) {
    return std::filesystem::path(env);
  }
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / ".inferflux";
  }
  return std::filesystem::current_path() / ".inferflux";
}

std::filesystem::path DefaultConfigPath() {
  return InferfluxHome() / "config.yaml";
}

std::string SelectBestGguf(const std::vector<std::string> &files);
std::filesystem::path ModelsDir();

json BuildChatPayload(const std::vector<ChatMessage> &messages,
                      const std::string &model, int max_tokens, bool stream) {
  json j;
  j["model"] = model.empty() ? "unknown" : model;
  j["max_tokens"] = max_tokens;
  j["stream"] = stream;
  j["messages"] = json::array();
  for (const auto &msg : messages) {
    j["messages"].push_back({{"role", msg.role.empty() ? "user" : msg.role},
                             {"content", msg.content}});
  }
  return j;
}

json BuildCompletionPayload(const std::string &prompt, const std::string &model,
                            int max_tokens, bool stream) {
  json j;
  j["prompt"] = prompt;
  j["max_tokens"] = max_tokens;
  j["stream"] = stream;
  if (!model.empty()) {
    j["model"] = model;
  }
  return j;
}

std::string TruncateCell(const std::string &value, std::size_t max_width) {
  if (value.size() <= max_width) {
    return value;
  }
  if (max_width <= 3) {
    return value.substr(0, max_width);
  }
  return value.substr(0, max_width - 3) + "...";
}

struct BackendExposureView {
  std::string requested_backend;
  std::string exposed_backend;
  std::string provider;
  bool fallback{false};
  std::string fallback_reason;
};

BackendExposureView ParseBackendExposure(const json &model) {
  BackendExposureView out;
  out.exposed_backend = model.value("backend", "");
  out.requested_backend = model.value("requested_backend", out.exposed_backend);
  out.provider = model.value("backend_provider", "unknown");
  out.fallback = model.value("backend_fallback", false);
  out.fallback_reason = model.value("backend_fallback_reason", "");

  if (model.contains("backend_exposure") &&
      model["backend_exposure"].is_object()) {
    const auto &exposure = model["backend_exposure"];
    out.exposed_backend =
        exposure.value("exposed_backend", out.exposed_backend);
    out.requested_backend =
        exposure.value("requested_backend", out.requested_backend);
    out.provider = exposure.value("provider", out.provider);
    out.fallback = exposure.value("fallback", out.fallback);
    out.fallback_reason =
        exposure.value("fallback_reason", out.fallback_reason);
  }

  if (out.requested_backend.empty()) {
    out.requested_backend = out.exposed_backend;
  }
  if (out.provider.empty()) {
    out.provider = "unknown";
  }
  return out;
}

void PrintModelsTable(const json &payload) {
  if (!payload.contains("data") || !payload["data"].is_array()) {
    std::cout << payload.dump() << std::endl;
    return;
  }
  const auto &data = payload["data"];
  if (data.empty()) {
    std::cout << "(no models loaded)\n";
    return;
  }

  constexpr int kIdW = 28;
  constexpr int kExposedW = 11;
  constexpr int kRequestedW = 11;
  constexpr int kProviderW = 11;
  constexpr int kFallbackW = 10;
  constexpr int kReadyW = 8;
  constexpr int kOwnerW = 14;

  std::cout << std::left << std::setw(kIdW) << "ID" << std::setw(kExposedW)
            << "EXPOSED-BE" << std::setw(kRequestedW) << "REQ-BE"
            << std::setw(kProviderW) << "PROVIDER" << std::setw(kFallbackW)
            << "FALLBACK" << std::setw(kReadyW) << "READY" << std::setw(kOwnerW)
            << "OWNED-BY" << "CREATED\n";
  std::cout << std::string(kIdW + kExposedW + kRequestedW + kProviderW +
                               kFallbackW + kReadyW + kOwnerW + 12,
                           '-')
            << "\n";

  for (const auto &m : data) {
    const std::string id = m.value("id", "");
    const auto exposure = ParseBackendExposure(m);
    const bool ready = m.value("ready", false);
    const std::string owned_by = m.value("owned_by", "");
    const int64_t created = m.value("created", int64_t{0});

    std::cout << std::left << std::setw(kIdW) << TruncateCell(id, kIdW - 1)
              << std::setw(kExposedW)
              << TruncateCell(exposure.exposed_backend, kExposedW - 1)
              << std::setw(kRequestedW)
              << TruncateCell(exposure.requested_backend, kRequestedW - 1)
              << std::setw(kProviderW)
              << TruncateCell(exposure.provider, kProviderW - 1)
              << std::setw(kFallbackW) << (exposure.fallback ? "yes" : "no")
              << std::setw(kReadyW) << (ready ? "yes" : "no")
              << std::setw(kOwnerW) << TruncateCell(owned_by, kOwnerW - 1)
              << created << "\n";
  }
}

void PrintAdminModelsTable(const json &payload) {
  if (!payload.contains("models") || !payload["models"].is_array()) {
    std::cout << payload.dump() << std::endl;
    return;
  }
  const auto &models = payload["models"];
  if (models.empty()) {
    std::cout << "(no models loaded)\n";
    return;
  }

  const std::string default_id = payload.value("default_model", "");

  constexpr int kDefaultW = 4;
  constexpr int kIdW = 22;
  constexpr int kBackendW = 11;
  constexpr int kRequestedW = 11;
  constexpr int kProviderW = 11;
  constexpr int kFallbackW = 10;
  constexpr int kFormatW = 14;
  constexpr int kReadyW = 8;
  constexpr int kSourceW = 34;
  constexpr int kEffectiveW = 34;

  std::cout << std::left << std::setw(kDefaultW) << "DEF" << std::setw(kIdW)
            << "ID" << std::setw(kBackendW) << "EXPOSED-BE"
            << std::setw(kRequestedW) << "REQ-BE" << std::setw(kProviderW)
            << "PROVIDER" << std::setw(kFallbackW) << "FALLBACK"
            << std::setw(kFormatW) << "FORMAT" << std::setw(kReadyW) << "READY"
            << std::setw(kSourceW) << "SOURCE-PATH" << "EFFECTIVE-LOAD-PATH\n";
  std::cout << std::string(kDefaultW + kIdW + kBackendW + kFormatW + kReadyW +
                               kRequestedW + kProviderW + kFallbackW +
                               kSourceW + kEffectiveW,
                           '-')
            << "\n";

  for (const auto &model : models) {
    const std::string id = model.value("id", "");
    const auto exposure = ParseBackendExposure(model);
    const std::string format = model.value("format", "");
    const std::string source_path =
        model.value("source_path", model.value("path", ""));
    const std::string effective_load_path =
        model.value("effective_load_path", source_path);
    const bool ready = model.value("ready", false);
    bool is_default = model.value("default", false);
    if (!is_default && !default_id.empty()) {
      is_default = (id == default_id);
    }

    std::cout << std::left << std::setw(kDefaultW) << (is_default ? "*" : "")
              << std::setw(kIdW) << TruncateCell(id, kIdW - 1)
              << std::setw(kBackendW)
              << TruncateCell(exposure.exposed_backend, kBackendW - 1)
              << std::setw(kRequestedW)
              << TruncateCell(exposure.requested_backend, kRequestedW - 1)
              << std::setw(kProviderW)
              << TruncateCell(exposure.provider, kProviderW - 1)
              << std::setw(kFallbackW) << (exposure.fallback ? "yes" : "no")
              << std::setw(kFormatW) << TruncateCell(format, kFormatW - 1)
              << std::setw(kReadyW) << (ready ? "yes" : "no")
              << std::setw(kSourceW) << TruncateCell(source_path, kSourceW - 1)
              << TruncateCell(effective_load_path, kEffectiveW - 1) << "\n";
  }
}

std::string JsonValueToText(const json &value) {
  if (value.is_null()) {
    return "n/a";
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? "yes" : "no";
  }
  if (value.is_string()) {
    return value.get<std::string>();
  }
  return value.dump();
}

std::string JsonThresholdPair(const json &value, const json &threshold) {
  if (value.is_null() && threshold.is_null()) {
    return "n/a";
  }
  if (threshold.is_null()) {
    return JsonValueToText(value);
  }
  return JsonValueToText(value) + " / " + JsonValueToText(threshold);
}

void PrintAdminPoolsSectionHeader(const std::string &title) {
  std::cout << title << "\n" << std::string(title.size(), '-') << "\n";
}

void PrintAdminPoolsRow(const std::string &label, const std::string &value) {
  constexpr int kLabelW = 27;
  std::cout << std::left << std::setw(kLabelW) << label << value << "\n";
}

void PrintAdminPoolsTable(const json &payload) {
  if (!payload.is_object() || !payload.contains("pool_health") ||
      !payload.contains("scheduler") || !payload.contains("distributed_kv")) {
    std::cout << payload.dump() << std::endl;
    return;
  }

  const json &pool = payload["pool_health"];
  const json &scheduler = payload["scheduler"];
  const json &disagg = payload["distributed_kv"];

  PrintAdminPoolsSectionHeader("POOL HEALTH");
  PrintAdminPoolsRow("Ready", JsonValueToText(pool.value("ready", false)));
  PrintAdminPoolsRow("HTTP status",
                     JsonValueToText(pool.value("http_status", json(nullptr))));
  PrintAdminPoolsRow("Role",
                     JsonValueToText(pool.value("role", json(nullptr))));
  PrintAdminPoolsRow("Reason",
                     JsonValueToText(pool.value("reason", json(nullptr))));
  PrintAdminPoolsRow("Model loaded", JsonValueToText(pool.value(
                                         "model_loaded", json(nullptr))));
  PrintAdminPoolsRow(
      "Decode pool warm",
      JsonValueToText(pool.value("decode_pool_warm", json(nullptr))));
  PrintAdminPoolsRow(
      "Transport degraded",
      JsonValueToText(pool.value("disagg_transport_degraded", json(nullptr))));
  PrintAdminPoolsRow(
      "Timeout debt",
      JsonThresholdPair(
          pool.value("disagg_timeout_debt", json(nullptr)),
          pool.value("disagg_timeout_debt_threshold", json(nullptr))));
  PrintAdminPoolsRow(
      "Timeout streak",
      JsonThresholdPair(
          pool.value("disagg_timeout_streak", json(nullptr)),
          pool.value("disagg_timeout_streak_threshold", json(nullptr))));

  std::cout << "\n";
  PrintAdminPoolsSectionHeader("SCHEDULER");
  PrintAdminPoolsRow("Queue depth", JsonValueToText(scheduler.value(
                                        "queue_depth", json(nullptr))));
  PrintAdminPoolsRow(
      "Prefill queue depth",
      JsonValueToText(scheduler.value("prefill_queue_depth", json(nullptr))));
  PrintAdminPoolsRow(
      "Decode queue depth",
      JsonValueToText(scheduler.value("decode_queue_depth", json(nullptr))));
  PrintAdminPoolsRow(
      "Batch limit size",
      JsonValueToText(scheduler.value("batch_limit_size", json(nullptr))));
  PrintAdminPoolsRow(
      "Batch limit tokens",
      JsonValueToText(scheduler.value("batch_limit_tokens", json(nullptr))));

  std::cout << "\n";
  PrintAdminPoolsSectionHeader("DISTRIBUTED KV");
  PrintAdminPoolsRow(
      "Enqueue rejections",
      JsonValueToText(disagg.value("enqueue_rejections_total", json(nullptr))));
  PrintAdminPoolsRow(
      "Enqueue exhausted",
      JsonValueToText(disagg.value("enqueue_exhausted_total", json(nullptr))));
  PrintAdminPoolsRow(
      "Tickets enqueued",
      JsonValueToText(disagg.value("tickets_enqueued_total", json(nullptr))));
  PrintAdminPoolsRow("Tickets acknowledged",
                     JsonValueToText(disagg.value("tickets_acknowledged_total",
                                                  json(nullptr))));
  PrintAdminPoolsRow(
      "Tickets committed",
      JsonValueToText(disagg.value("tickets_committed_total", json(nullptr))));
  PrintAdminPoolsRow(
      "Tickets timed out",
      JsonValueToText(disagg.value("tickets_timed_out_total", json(nullptr))));
}

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  inferctl server start [--config PATH] [--no-wait]\n"
      << "      Start InferFlux server in background with PID tracking.\n"
      << "      --config: Server config file (default: "
         "~/.inferflux/config.yaml)\n"
      << "      --no-wait: Don't wait for server to be ready\n"
      << "  inferctl server stop [--force]\n"
      << "      Stop running InferFlux server gracefully (or with SIGKILL if "
         "--force).\n"
      << "  inferctl server status [--verbose]\n"
      << "      Show server status, PID, and health information.\n"
      << "  inferctl server restart [--config PATH] [--no-wait]\n"
      << "      Restart the server (stop + start).\n"
      << "  inferctl server logs [--tail N]\n"
      << "      Show server logs (default: tail -f with last 100 lines).\n"
      << "      --tail N: Show last N lines without following.\n"
      << "  inferctl pull <owner/model-name>\n"
      << "      Download the best GGUF from HuggingFace Hub "
         "(~/.cache/inferflux/models/).\n"
      << "      Prints the local path on success.\n"
      << "  inferctl quickstart <model-id> [--profile cpu-laptop] [--backend "
         "cpu|mps|cuda|mlx]\n"
      << "      Writes a starter config (~/.inferflux/config.yaml) with the "
         "desired backend and prints "
         "next steps.\n"
      << "  inferctl serve [--config ~/.inferflux/config.yaml] [--no-ui]\n"
      << "      Launch inferfluxd using the specified config (defaults to "
         "~/.inferflux/config.yaml).\n"
      << "  inferctl status [--host 127.0.0.1] [--port 8080] [--api-key KEY]\n"
      << "  inferctl completion --prompt 'Hello' [--model MODEL] [--max-tokens "
         "N] [--stream]\n"
         "                      [--host 127.0.0.1] [--port 8080] [--api-key "
         "KEY]\n"
      << "  inferctl chat --message 'user:Hello' [--message 'assistant:Hi'] "
         "[--model MODEL]\n"
         "               [--max-tokens N] [--stream] [--interactive]\n"
      << "  inferctl admin guardrails [--list | --set word1,word2] [--host ... "
         "--port ... --api-key KEY]\n"
      << "  inferctl admin rate-limit [--get | --set N] [--host ... --port ... "
         "--api-key KEY]\n"
      << "  inferctl admin routing [--get | --set [--allow-default-fallback "
         "true|false]\n"
         "                         [--require-ready-backend true|false]\n"
         "                         [--fallback-scope any_compatible|"
         "same_path_only]]\n"
         "                        [--host ... --port ... --api-key KEY]\n"
      << "  inferctl admin pools --get [--json|--table] [--host ... --port "
         "... --api-key KEY]\n"
      << "  inferctl admin models --list | --load PATH [--backend TYPE] [--id "
         "NAME] [--default] [--json]\n"
         "                       | --unload ID | --set-default ID [--host ... "
         "--port ... --api-key KEY]\n"
      << "  inferctl admin cache [--status] [--host ... --port ... --api-key "
         "KEY]\n"
      << "  inferctl admin cache --warm --tokens ID,ID,... --completion TEXT\n"
         "                       [--completion-tokens N] [--host ... --port "
         "... --api-key KEY]\n"
      << "  inferctl admin api-keys [--list | --add KEY --scopes read,admin | "
         "--remove KEY]\n"
      << "  inferctl models [--id MODEL_ID] [--host 127.0.0.1] [--port 8080] "
         "[--api-key KEY] [--json]\n"
         "      List loaded models (GET /v1/models) or fetch one model "
         "(GET /v1/models/{id}); requires 'read' scope.\n";
}

bool ProcessStreamChunks(std::string &buffer, std::string *accumulated) {
  bool done = false;
  while (true) {
    auto data_pos = buffer.find("data:");
    if (data_pos == std::string::npos) {
      break;
    }
    auto end_pos = buffer.find("\n\n", data_pos);
    if (end_pos == std::string::npos) {
      buffer.erase(0, data_pos);
      break;
    }
    std::string chunk =
        Trim(buffer.substr(data_pos + 5, end_pos - (data_pos + 5)));
    buffer.erase(0, end_pos + 2);
    if (chunk == "[DONE]") {
      done = true;
      break;
    }
    try {
      auto j = json::parse(chunk);
      if (j.contains("choices") && j["choices"].is_array() &&
          !j["choices"].empty()) {
        auto &delta = j["choices"][0];
        if (delta.contains("delta") && delta["delta"].contains("content")) {
          std::string content = delta["delta"]["content"].get<std::string>();
          std::cout << content << std::flush;
          if (accumulated) {
            accumulated->append(content);
          }
        }
      }
    } catch (const json::exception &) {
    }
  }
  return done;
}

bool ReceiveStream(inferflux::HttpClient &client,
                   inferflux::HttpClient::RawConnection &conn,
                   std::string *accumulated) {
  std::string buffer;
  char temp[4096];
  ssize_t read_bytes = 0;
  bool headers_done = false;
  while ((read_bytes = client.RecvRaw(conn, temp, sizeof(temp))) > 0) {
    buffer.append(temp, temp + read_bytes);
    if (!headers_done) {
      auto header_end = buffer.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        headers_done = true;
        buffer.erase(0, header_end + 4);
      } else {
        continue;
      }
    }
    if (ProcessStreamChunks(buffer, accumulated)) {
      break;
    }
  }
  std::cout << std::endl;
  client.CloseRaw(conn);
  return true;
}

void InteractiveChat(const std::string &host, int port,
                     const std::string &api_key, const std::string &model,
                     int max_tokens, bool stream) {
  inferflux::HttpClient client;
  std::vector<ChatMessage> history;
  std::cout << "Interactive chat session. Type /exit to quit." << std::endl;
  std::string line;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, line)) {
      break;
    }
    if (line == "/exit") {
      break;
    }
    if (line.empty()) {
      continue;
    }
    history.push_back({"user", line});
    auto payload = BuildChatPayload(history, model, max_tokens, stream);
    std::string url = BuildUrl(host, port, "/v1/chat/completions");
    std::string assistant;
    if (stream) {
      auto conn =
          client.SendRaw("POST", url, payload.dump(), AuthHeaders(api_key));
      ReceiveStream(client, conn, &assistant);
    } else {
      auto resp = client.Post(url, payload.dump(), AuthHeaders(api_key));
      try {
        auto j = json::parse(resp.body);
        if (j.contains("choices") && j["choices"].is_array() &&
            !j["choices"].empty()) {
          auto &choice = j["choices"][0];
          if (choice.contains("message") &&
              choice["message"].contains("content")) {
            assistant = choice["message"]["content"].get<std::string>();
          }
        }
      } catch (const json::exception &) {
        assistant = resp.body;
      }
      std::cout << assistant << std::endl;
    }
    history.push_back({"assistant", assistant});
  }
}

bool ResolveBestGguf(inferflux::HttpClient &client, const std::string &repo,
                     std::string *selected_out) {
  if (repo.find('/') == std::string::npos) {
    std::cerr << "Repository must be in owner/model-name format (e.g. "
                 "TheBloke/Llama-2-7B-GGUF)\n";
    return false;
  }
  std::string api_url = "https://huggingface.co/api/models/" + repo;
  std::cerr << "Fetching model info for " << repo << "...\n";
  auto resp = client.Get(api_url, {{"Accept", "application/json"}});
  if (resp.status != 200) {
    std::cerr << "HuggingFace API error " << resp.status << " for " << repo
              << "\n";
    return false;
  }
  std::vector<std::string> gguf_files;
  try {
    auto j = json::parse(resp.body);
    if (j.contains("siblings") && j["siblings"].is_array()) {
      for (const auto &s : j["siblings"]) {
        if (!s.contains("rfilename"))
          continue;
        std::string fname = s["rfilename"].get<std::string>();
        std::string lower = ToLower(fname);
        if (lower.find(".gguf") != std::string::npos) {
          gguf_files.push_back(fname);
        }
      }
    }
  } catch (const json::exception &e) {
    std::cerr << "Failed to parse HuggingFace API response: " << e.what()
              << "\n";
    return false;
  }
  if (gguf_files.empty()) {
    std::cerr << "No GGUF files found in " << repo << "\n";
    return false;
  }
  *selected_out = SelectBestGguf(gguf_files);
  std::cerr << "Selected: " << *selected_out << "\n";
  return true;
}

int CmdQuickstart(const std::string &repo, const std::string &profile,
                  const std::string &backend_hint) {
  inferflux::HttpClient client;
  std::string selected;
  if (!ResolveBestGguf(client, repo, &selected)) {
    return 1;
  }
  namespace fs = std::filesystem;
  fs::path base_dir = InferfluxHome();
  fs::create_directories(base_dir);
  fs::path config_path = base_dir / "config.yaml";
  fs::path model_dir = ModelsDir() / repo;
  fs::create_directories(model_dir);
  fs::path model_path = model_dir / selected;

  std::ofstream cfg(config_path);
  if (!cfg) {
    std::cerr << "Failed to write config at " << config_path << std::endl;
    return 1;
  }
  std::string model_id = repo;
  std::replace(model_id.begin(), model_id.end(), '/', '_');
  std::string backend = ToLower(backend_hint.empty() ? "cpu" : backend_hint);
  cfg << "# InferFlux quickstart config (profile: " << profile << ")\n"
      << "server:\n"
      << "  host: 0.0.0.0\n"
      << "  http_port: 8080\n"
      << "auth:\n"
      << "  api_keys:\n"
      << "    - key: dev-key-123\n"
      << "      scopes: [generate, read, admin]\n"
      << "models:\n"
      << "  - id: " << model_id << "\n"
      << "    path: \"" << model_path.string() << "\"\n"
      << "    backend: " << backend << "\n"
      << "    default: true\n";
  cfg.close();
  std::cout << "Wrote config to " << config_path << "\n";
  std::cout << "Next steps:\n"
            << "  1. inferctl pull " << repo << "\n"
            << "  2. inferctl serve --config " << config_path << "\n";
  return 0;
}

int CmdServe(const std::string &config_path, bool enable_ui) {
  namespace fs = std::filesystem;
  fs::path cfg =
      config_path.empty() ? DefaultConfigPath() : fs::path(config_path);
  if (!fs::exists(cfg)) {
    std::cerr << "Config not found: " << cfg << std::endl;
    return 1;
  }
  std::string cmd = "inferfluxd --config \"" + cfg.string() + "\"";
  if (enable_ui) {
    cmd += " --ui";
  }
  std::cout << "Launching: " << cmd << std::endl;
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::cerr << "inferfluxd exited with code " << rc << std::endl;
  }
  return rc;
}

// ---- inferctl pull helpers ----

std::filesystem::path ModelsDir() { return InferfluxHome() / "models"; }

// Prefer quantization quality order: Q4_K_M > Q4_K_S > Q4_K > Q4_0 > Q4 >
// Q5_K_M > Q5 > Q8 > any .gguf
std::string SelectBestGguf(const std::vector<std::string> &files) {
  auto score = [](const std::string &name) -> int {
    std::string lower = ToLower(name);
    if (lower.find("q4_k_m") != std::string::npos)
      return 100;
    if (lower.find("q4_k_s") != std::string::npos)
      return 90;
    if (lower.find("q4_k") != std::string::npos)
      return 85;
    if (lower.find("q4_0") != std::string::npos)
      return 80;
    if (lower.find("q4") != std::string::npos)
      return 75;
    if (lower.find("q5_k_m") != std::string::npos)
      return 70;
    if (lower.find("q5") != std::string::npos)
      return 65;
    if (lower.find("q8_0") != std::string::npos)
      return 50;
    if (lower.find(".gguf") != std::string::npos)
      return 10;
    return -1;
  };
  std::string best;
  int best_score = -1;
  for (const auto &f : files) {
    int s = score(f);
    if (s > best_score) {
      best_score = s;
      best = f;
    }
  }
  return best;
}

void PrintProgress(long long downloaded, long long total) {
  if (total <= 0) {
    std::cerr << "\r  " << (downloaded / (1024 * 1024)) << " MB downloaded"
              << std::flush;
    return;
  }
  int pct = static_cast<int>(downloaded * 100 / total);
  const int kBarWidth = 40;
  int filled = pct * kBarWidth / 100;
  std::cerr << "\r[";
  for (int i = 0; i < filled; ++i)
    std::cerr << '=';
  for (int i = filled; i < kBarWidth; ++i)
    std::cerr << ' ';
  std::cerr << "] " << pct << "%  " << (downloaded / (1024 * 1024)) << "/"
            << (total / (1024 * 1024)) << " MB" << std::flush;
}

struct DownloadHeaders {
  int status{0};
  std::string location;
  long long content_length{-1};
};

DownloadHeaders ParseDownloadHeaders(const std::string &header_block) {
  DownloadHeaders h;
  std::istringstream iss(header_block);
  std::string line;
  bool first = true;
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (first) {
      first = false;
      auto sp = line.find(' ');
      if (sp != std::string::npos) {
        try {
          h.status = std::stoi(line.substr(sp + 1));
        } catch (...) {
        }
      }
      continue;
    }
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    std::string key = Trim(line.substr(0, colon));
    std::string val = Trim(line.substr(colon + 1));
    std::string key_lower = ToLower(key);
    if (key_lower == "location")
      h.location = val;
    else if (key_lower == "content-length") {
      try {
        h.content_length = std::stoll(val);
      } catch (...) {
      }
    }
  }
  return h;
}

// Streams a GET response body to dest_path, following up to max_redirects
// redirects. Returns 0 on success, 1 on error.
int DownloadToFile(inferflux::HttpClient &client, const std::string &url,
                   const std::string &dest_path) {
  std::string current_url = url;
  const int kMaxRedirects = 5;
  for (int attempt = 0; attempt <= kMaxRedirects; ++attempt) {
    auto conn = client.SendRaw("GET", current_url, "", {});

    // Extend recv timeout to 5 minutes for large file downloads.
    if (conn.sock >= 0) {
      struct timeval tv {
        300, 0
      };
      setsockopt(conn.sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char *>(&tv), sizeof(tv));
    }

    // Read until we have the full response headers.
    char buf[65536];
    std::string header_buf;
    std::string leftover;
    bool found_end = false;
    while (!found_end) {
      ssize_t n = client.RecvRaw(conn, buf, sizeof(buf));
      if (n <= 0)
        break;
      header_buf.append(buf, static_cast<std::size_t>(n));
      auto pos = header_buf.find("\r\n\r\n");
      if (pos != std::string::npos) {
        leftover = header_buf.substr(pos + 4);
        header_buf.erase(pos);
        found_end = true;
      }
    }
    if (!found_end) {
      client.CloseRaw(conn);
      std::cerr << "Failed to read response headers from " << current_url
                << "\n";
      return 1;
    }

    auto hdrs = ParseDownloadHeaders(header_buf);

    if (hdrs.status == 301 || hdrs.status == 302 || hdrs.status == 307 ||
        hdrs.status == 308) {
      client.CloseRaw(conn);
      if (hdrs.location.empty()) {
        std::cerr << "Redirect with no Location header\n";
        return 1;
      }
      current_url = hdrs.location;
      std::cerr << "Following redirect...\n";
      continue;
    }

    if (hdrs.status != 200) {
      client.CloseRaw(conn);
      std::cerr << "HTTP error " << hdrs.status << " downloading "
                << current_url << "\n";
      return 1;
    }

    // Stream body to a .tmp file, then atomically rename.
    std::string tmp_path = dest_path + ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out) {
      client.CloseRaw(conn);
      std::cerr << "Cannot create file: " << tmp_path << "\n";
      return 1;
    }

    long long downloaded = 0;
    long long total = hdrs.content_length;

    if (!leftover.empty()) {
      out.write(leftover.data(), static_cast<std::streamsize>(leftover.size()));
      downloaded += static_cast<long long>(leftover.size());
      PrintProgress(downloaded, total);
    }

    while (true) {
      ssize_t n = client.RecvRaw(conn, buf, sizeof(buf));
      if (n <= 0)
        break;
      out.write(buf, n);
      downloaded += n;
      PrintProgress(downloaded, total);
    }

    client.CloseRaw(conn);
    out.close();
    std::cerr << "\n";

    if (total > 0 && downloaded < total) {
      std::filesystem::remove(tmp_path);
      std::cerr << "Download incomplete (" << downloaded << "/" << total
                << " bytes)\n";
      return 1;
    }

    std::filesystem::rename(tmp_path, dest_path);
    return 0;
  }

  std::cerr << "Too many redirects for " << url << "\n";
  return 1;
}

int CmdPull(const std::string &repo) {
  inferflux::HttpClient client;
  std::string selected;
  if (!ResolveBestGguf(client, repo, &selected)) {
    return 1;
  }

  // Step 4: Resolve destination path.
  auto cache_dir = ModelsDir();
  std::filesystem::path dest_dir = cache_dir / repo;
  std::filesystem::create_directories(dest_dir);
  std::filesystem::path dest_path = dest_dir / selected;

  if (std::filesystem::exists(dest_path)) {
    std::cerr << "Already cached: " << dest_path << "\n";
    std::cout << dest_path << "\n";
    return 0;
  }

  // Step 5: Stream-download with progress.
  // HuggingFace resolve URL:
  // https://huggingface.co/{repo}/resolve/main/{filename}
  std::string download_url =
      "https://huggingface.co/" + repo + "/resolve/main/" + selected;
  std::cerr << "Downloading " << selected << "...\n";

  int rc = DownloadToFile(client, download_url, dest_path.string());
  if (rc != 0)
    return rc;

  std::cerr << "Saved to: " << dest_path << "\n";
  std::cout << dest_path << "\n";
  return 0;
}

// -----------------------------------------------------------------------------
// Server Management Commands
// -----------------------------------------------------------------------------

std::filesystem::path ServerPidFile() { return InferfluxHome() / "server.pid"; }

std::filesystem::path ServerLogFile() {
  return InferfluxHome() / "logs" / "server.log";
}

#ifdef _WIN32
// Windows helper: check if a process with the given PID is still running.
bool IsProcessRunning(DWORD pid) {
  HANDLE h = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid);
  if (h == nullptr)
    return false;
  DWORD exit_code = 0;
  bool running = GetExitCodeProcess(h, &exit_code) && exit_code == STILL_ACTIVE;
  CloseHandle(h);
  return running;
}
#endif

bool IsServerRunning() {
  auto pid_file = ServerPidFile();
  if (!std::filesystem::exists(pid_file)) {
    return false;
  }

  std::ifstream pid_file_stream(pid_file);
  if (!pid_file_stream.is_open()) {
    return false;
  }

#ifdef _WIN32
  DWORD pid;
  pid_file_stream >> pid;
  return IsProcessRunning(pid);
#else
  pid_t pid;
  pid_file_stream >> pid;

  // Check if process is running by sending signal 0
  if (kill(pid, 0) == 0) {
    return true; // Process exists
  }
  return false; // Process doesn't exist
#endif
}

int CmdServerStart(const std::string &config_path, bool wait_for_ready = true,
                   int timeout_sec = 30) {
  if (IsServerRunning()) {
    std::cerr << "Error: Server is already running (PID file: "
              << ServerPidFile() << ")\n";
    std::cerr << "Use 'inferctl server stop' to stop it first.\n";
    return 1;
  }

  // Create logs directory
  auto log_file = ServerLogFile();
  std::filesystem::create_directories(log_file.parent_path());

  // Build server binary path
#ifdef _WIN32
  std::string server_bin = ".\\build\\inferfluxd.exe";
  if (!std::filesystem::exists(server_bin)) {
    // Try relative path from inferctl
    char exe_path[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
      std::filesystem::path exe_dir =
          std::filesystem::path(exe_path).parent_path();
      server_bin = (exe_dir / "inferfluxd.exe").string();
      if (!std::filesystem::exists(server_bin)) {
        server_bin = (exe_dir / "..\\inferfluxd.exe").string();
      }
    }
  }
#else
  std::string server_bin = "./build/inferfluxd";
  if (!std::filesystem::exists(server_bin)) {
    // Try relative path from inferctl
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
      exe_path[len] = '\0';
      std::filesystem::path exe_dir =
          std::filesystem::path(exe_path).parent_path();
      server_bin = (exe_dir / "inferfluxd").string();
      if (!std::filesystem::exists(server_bin)) {
        server_bin = (exe_dir / "../inferfluxd").string();
      }
    }
  }
#endif

  if (!std::filesystem::exists(server_bin)) {
    std::cerr << "Error: Server binary not found at " << server_bin << "\n";
    std::cerr
        << "Build the server with: cmake --build build --target inferfluxd\n";
    return 1;
  }

#ifdef _WIN32
  // Start server as a detached process with output redirected to log file
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;

  HANDLE log_handle = CreateFileA(log_file.string().c_str(), FILE_APPEND_DATA,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (log_handle == INVALID_HANDLE_VALUE) {
    std::cerr << "Error: Failed to open log file: " << log_file << "\n";
    return 1;
  }

  STARTUPINFOA si = {};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = INVALID_HANDLE_VALUE;
  si.hStdOutput = log_handle;
  si.hStdError = log_handle;

  PROCESS_INFORMATION pi = {};
  std::string cmd_line = server_bin + " --config " + config_path;

  if (!CreateProcessA(nullptr, const_cast<char *>(cmd_line.c_str()), nullptr,
                      nullptr, TRUE, DETACHED_PROCESS, nullptr, nullptr, &si,
                      &pi)) {
    std::cerr << "Error: Failed to start server process (error "
              << GetLastError() << ")\n";
    CloseHandle(log_handle);
    return 1;
  }

  DWORD pid = pi.dwProcessId;
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(log_handle);

  // Write PID file
  {
    std::ofstream pid_file_stream(ServerPidFile());
    pid_file_stream << pid << "\n";
  }

  std::cout << "Starting InferFlux server (PID: " << pid << ")...\n";
  std::cout << "Config: " << config_path << "\n";
  std::cout << "Logs: " << log_file << "\n";

  if (wait_for_ready) {
    std::cout << "Waiting for server to be ready...";
    std::cout.flush();

    auto start = std::chrono::steady_clock::now();
    bool ready = false;

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
               .count() < timeout_sec) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      std::cout << ".";
      std::cout.flush();

      // Check if process is still running
      if (!IsProcessRunning(pid)) {
        std::cout << "\n";
        std::cerr << "Error: Server process died unexpectedly\n";
        std::filesystem::remove(ServerPidFile());
        return 1;
      }

      // Try health endpoint
      try {
        inferflux::HttpClient client;
        auto headers = AuthHeaders("");
        auto resp =
            client.Get(BuildUrl("127.0.0.1", 8080, "/healthz"), headers);
        if (resp.status == 200) {
          ready = true;
          break;
        }
      } catch (...) {
        // Server not ready yet
      }
    }

    std::cout << "\n";

    if (ready) {
      std::cout << "Server is ready!\n";
      std::cout << "  API: http://127.0.0.1:8080\n";
      std::cout << "  Health: http://127.0.0.1:8080/healthz\n";
      std::cout << "  Metrics: http://127.0.0.1:8080/metrics\n";
      return 0;
    } else {
      std::cerr << "Error: Server failed to become ready after " << timeout_sec
                << " seconds\n";
      std::cerr << "Check logs at: " << log_file << "\n";
      return 1;
    }
  }

  return 0;
}
#else
  // Start server in background
  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "Error: Failed to fork server process: " << strerror(errno)
              << "\n";
    return 1;
  }

  if (pid == 0) {
    // Child process - start server
    // Close stdin/stdout/stderr
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // Redirect output to log file
    int log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log_fd < 0) {
      _exit(1);
    }
    dup2(log_fd, STDOUT_FILENO);
    dup2(log_fd, STDERR_FILENO);
    close(log_fd);

    // Start server
    execl(server_bin.c_str(), server_bin.c_str(), "--config",
          config_path.c_str(), (char *)nullptr);
    _exit(1); // If exec returns
  }

  // Parent process - write PID file
  {
    std::ofstream pid_file_stream(ServerPidFile());
    pid_file_stream << pid << "\n";
  }

  std::cout << "Starting InferFlux server (PID: " << pid << ")...\n";
  std::cout << "Config: " << config_path << "\n";
  std::cout << "Logs: " << log_file << "\n";

  if (wait_for_ready) {
    std::cout << "Waiting for server to be ready...";
    std::cout.flush();

    auto start = std::chrono::steady_clock::now();
    bool ready = false;

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
               .count() < timeout_sec) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      std::cout << ".";
      std::cout.flush();

      // Check if process is still running
      if (kill(pid, 0) != 0) {
        std::cout << "\n";
        std::cerr << "Error: Server process died unexpectedly\n";
        std::filesystem::remove(ServerPidFile());
        return 1;
      }

      // Try health endpoint
      try {
        inferflux::HttpClient client;
        auto headers = AuthHeaders(""); // Health endpoint doesn't need auth
        auto resp =
            client.Get(BuildUrl("127.0.0.1", 8080, "/healthz"), headers);
        if (resp.status == 200) {
          ready = true;
          break;
        }
      } catch (...) {
        // Server not ready yet
      }
    }

    std::cout << "\n";

    if (ready) {
      std::cout << "✓ Server is ready!\n";
      std::cout << "  API: http://127.0.0.1:8080\n";
      std::cout << "  Health: http://127.0.0.1:8080/healthz\n";
      std::cout << "  Metrics: http://127.0.0.1:8080/metrics\n";
      return 0;
    } else {
      std::cerr << "Error: Server failed to become ready after " << timeout_sec
                << " seconds\n";
      std::cerr << "Check logs at: " << log_file << "\n";
      return 1;
    }
  }

  return 0;
}
#endif

int CmdServerStop(bool wait = true, int timeout_sec = 10) {
  auto pid_file = ServerPidFile();
  if (!std::filesystem::exists(pid_file)) {
    std::cerr << "Error: Server is not running (no PID file found)\n";
    return 1;
  }

  std::ifstream pid_file_stream(pid_file);
  if (!pid_file_stream.is_open()) {
    std::cerr << "Error: Cannot read PID file\n";
    return 1;
  }

#ifdef _WIN32
  DWORD pid;
  pid_file_stream >> pid;

  // Check if process is running
  if (!IsProcessRunning(pid)) {
    std::cerr << "Server process (PID " << pid << ") is not running\n";
    std::filesystem::remove(pid_file);
    return 0;
  }

  std::cout << "Stopping server (PID: " << pid << ")...";
  std::cout.flush();

  // Terminate the process
  HANDLE proc = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
  if (proc == nullptr) {
    std::cerr << "\nError: Failed to open server process (error "
              << GetLastError() << ")\n";
    return 1;
  }

  if (!TerminateProcess(proc, 0)) {
    std::cerr << "\nError: Failed to terminate server process (error "
              << GetLastError() << ")\n";
    CloseHandle(proc);
    return 1;
  }

  if (wait) {
    WaitForSingleObject(proc, timeout_sec * 1000);
  }

  CloseHandle(proc);

  std::cout << "\nServer stopped\n";
#else
  pid_t pid;
  pid_file_stream >> pid;

  // Check if process is running
  if (kill(pid, 0) != 0) {
    std::cerr << "Server process (PID " << pid << ") is not running\n";
    std::filesystem::remove(pid_file);
    return 0;
  }

  std::cout << "Stopping server (PID: " << pid << ")...";
  std::cout.flush();

  // Send SIGTERM
  if (kill(pid, SIGTERM) != 0) {
    std::cerr << "\nError: Failed to send SIGTERM: " << strerror(errno) << "\n";
    return 1;
  }

  if (wait) {
    auto start = std::chrono::steady_clock::now();
    bool stopped = false;

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
               .count() < timeout_sec) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      std::cout << ".";
      std::cout.flush();

      if (kill(pid, 0) != 0) {
        stopped = true;
        break;
      }
    }

    if (!stopped) {
      std::cout << "\nServer did not stop gracefully, forcing...";
      std::cout.flush();
      kill(pid, SIGKILL);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  std::cout << "\n✓ Server stopped\n";
#endif

  // Clean up PID file
  std::filesystem::remove(pid_file);

  return 0;
}

int CmdServerStatus(bool verbose = false) {
  auto pid_file = ServerPidFile();
  bool pid_exists = std::filesystem::exists(pid_file);

  if (!pid_exists) {
    std::cout << "Server: " << "\033[31m" << "STOPPED" << "\033[0m" << "\n";
    return 1;
  }

  std::ifstream pid_file_stream(pid_file);
  if (!pid_file_stream.is_open()) {
    std::cerr << "Error: Cannot read PID file\n";
    return 1;
  }

#ifdef _WIN32
  DWORD pid;
#else
  pid_t pid;
#endif
  pid_file_stream >> pid;

#ifdef _WIN32
  if (!IsProcessRunning(pid)) {
#else
  if (kill(pid, 0) != 0) {
#endif
    std::cout << "Server: " << "\033[31m" << "CRASHED" << "\033[0m"
              << " (PID file exists but process not running)\n";
    std::cout << "Cleaning up stale PID file...\n";
    std::filesystem::remove(pid_file);
    return 1;
  }

  std::cout << "Server: " << "\033[32m" << "RUNNING" << "\033[0m" << "\n";
  std::cout << "PID: " << pid << "\n";
  std::cout << "Config: "
            << (std::filesystem::exists(DefaultConfigPath())
                    ? DefaultConfigPath().string()
                    : "not found")
            << "\n";
  std::cout << "Logs: " << ServerLogFile() << "\n";

  if (verbose) {
    std::cout << "\nChecking server health...\n";
    try {
      inferflux::HttpClient client;
      auto headers = AuthHeaders("");

      auto health_resp =
          client.Get(BuildUrl("127.0.0.1", 8080, "/healthz"), headers);
      if (health_resp.status == 200) {
        auto health = json::parse(health_resp.body);
        bool model_ready = health.value("model_ready", false);
        std::cout << "Health: " << "\033[32m" << "OK" << "\033[0m" << "\n";
        std::cout << "Model Ready: " << (model_ready ? "yes" : "no") << "\n";
      }

      auto models_resp =
          client.Get(BuildUrl("127.0.0.1", 8080, "/v1/models"), headers);
      if (models_resp.status == 200) {
        auto models = json::parse(models_resp.body);
        if (models.contains("data") && models["data"].is_array()) {
          std::cout << "\nLoaded Models:\n";
          for (const auto &model : models["data"]) {
            std::string id = model.value("id", "unknown");
            std::string backend = model.value("backend", "unknown");
            bool ready = model.value("ready", false);
            std::cout << "  - " << id << " [" << backend << "] "
                      << (ready ? "✓" : "loading...") << "\n";
          }
        }
      }
    } catch (const std::exception &e) {
      std::cout << "Warning: Could not fetch server status: " << e.what()
                << "\n";
    }
  }

  return 0;
}

int CmdServerRestart(const std::string &config_path,
                     bool wait_for_ready = true) {
  std::cout << "Restarting server...\n";
  CmdServerStop(true, 10);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  return CmdServerStart(config_path, wait_for_ready);
}

int CmdServerLogs(int tail_lines = 100) {
  auto log_file = ServerLogFile();
  if (!std::filesystem::exists(log_file)) {
    std::cerr << "Error: Log file not found at " << log_file << "\n";
    std::cerr << "Server may not have been started yet.\n";
    return 1;
  }

#ifdef _WIN32
  std::string command = "powershell -Command \"Get-Content -Path '" +
                        log_file.string() + "' -Tail " +
                        std::to_string(tail_lines) + " -Wait\"";
#else
  std::string command =
      "tail -n " + std::to_string(tail_lines) + " -f " + log_file.string();
#endif
  int result = std::system(command.c_str());

  return (result == 0) ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }
  std::string command = argv[1];
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string prompt;
  std::string api_key;
  std::string model;
  int max_tokens = 256;
  bool stream_mode = false;
  bool interactive_mode = false;
  std::vector<ChatMessage> chat_messages;
  std::string quickstart_profile = "default";
  std::string quickstart_backend = "cpu";
  std::string serve_config_override;
  bool serve_no_ui = false;
  if (const char *env_key = std::getenv("INFERCTL_API_KEY")) {
    api_key = env_key;
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--host" || arg == "-H") && i + 1 < argc) {
      host = argv[++i];
    } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--api-key" && i + 1 < argc) {
      api_key = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if ((arg == "--max-tokens" || arg == "--max_tokens") &&
               i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--stream") {
      stream_mode = true;
    } else if (arg == "--interactive") {
      interactive_mode = true;
    } else if (arg == "--profile" && i + 1 < argc) {
      quickstart_profile = argv[++i];
    } else if (arg == "--backend" && i + 1 < argc) {
      quickstart_backend = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      serve_config_override = argv[++i];
    } else if (arg == "--no-ui") {
      serve_no_ui = true;
    } else if ((arg == "--message" || arg == "-m") && i + 1 < argc) {
      std::string raw = argv[++i];
      auto colon = raw.find(':');
      ChatMessage msg;
      if (colon == std::string::npos) {
        msg.role = "user";
        msg.content = raw;
      } else {
        msg.role = raw.substr(0, colon);
        msg.content = raw.substr(colon + 1);
        if (msg.role.empty()) {
          msg.role = "user";
        }
      }
      chat_messages.push_back(std::move(msg));
    }
  }

  // pull does not require a running server — handle before try/client block.
  if (command == "pull") {
    if (argc < 3) {
      std::cerr << "Usage: inferctl pull <owner/model-name>\n";
      return 1;
    }
    return CmdPull(argv[2]);
  }
  if (command == "quickstart") {
    if (argc < 3) {
      std::cerr << "Usage: inferctl quickstart <model-id> [--profile "
                   "cpu-laptop] [--backend cpu|mps|cuda|mlx]\n";
      return 1;
    }
    return CmdQuickstart(argv[2], quickstart_profile, quickstart_backend);
  }
  if (command == "serve") {
    return CmdServe(serve_config_override.empty() ? DefaultConfigPath().string()
                                                  : serve_config_override,
                    !serve_no_ui);
  }

  // Server management commands (don't require running server)
  if (command == "server") {
    if (argc < 3) {
      std::cerr
          << "Usage: inferctl server {start|stop|status|restart|logs} [...]\n";
      return 1;
    }
    std::string server_cmd = argv[2];

    if (server_cmd == "start") {
      std::string config = serve_config_override.empty()
                               ? DefaultConfigPath().string()
                               : serve_config_override;
      bool wait_for_ready = true;
      for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
          config = argv[++i];
        } else if (arg == "--no-wait") {
          wait_for_ready = false;
        }
      }
      return CmdServerStart(config, wait_for_ready);
    }

    if (server_cmd == "stop") {
      bool force = false;
      for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--force") {
          force = true;
        }
      }
      return CmdServerStop(!force, force ? 2 : 10);
    }

    if (server_cmd == "status") {
      bool verbose = false;
      for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
          verbose = true;
        }
      }
      return CmdServerStatus(verbose);
    }

    if (server_cmd == "restart") {
      std::string config = serve_config_override.empty()
                               ? DefaultConfigPath().string()
                               : serve_config_override;
      bool wait_for_ready = true;
      for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
          config = argv[++i];
        } else if (arg == "--no-wait") {
          wait_for_ready = false;
        }
      }
      return CmdServerRestart(config, wait_for_ready);
    }

    if (server_cmd == "logs") {
      int tail_lines = 0; // 0 means follow mode
      for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tail" && i + 1 < argc) {
          tail_lines = std::stoi(argv[++i]);
        }
      }
      return CmdServerLogs(tail_lines);
    }

    std::cerr << "Unknown server command: " << server_cmd << "\n";
    std::cerr << "Valid commands: start, stop, status, restart, logs\n";
    return 1;
  }

  try {
    inferflux::HttpClient client;
    auto headers = AuthHeaders(api_key);

    if (command == "admin") {
      if (argc < 3) {
        PrintUsage();
        return 1;
      }
      if (api_key.empty()) {
        std::cerr << "--api-key is required for admin commands" << std::endl;
        return 1;
      }
      std::string target = argv[2];
      if (target == "guardrails") {
        bool list = false;
        bool set = false;
        std::string set_values;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list") {
            list = true;
          } else if (arg == "--set") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin guardrails",
                                        "--set", "word1,word2", &set_values)) {
              return 1;
            }
            set = true;
          }
        }
        const int op_count = (list ? 1 : 0) + (set ? 1 : 0);
        if (op_count == 0) {
          PrintUsage();
          return 1;
        }
        if (op_count > 1) {
          std::cerr << "inferctl admin guardrails: choose exactly one of "
                       "--list, --set"
                    << std::endl;
          return 1;
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/guardrails"), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin guardrails --list");
        }
        if (set) {
          std::vector<std::string> words;
          std::stringstream ss(set_values);
          std::string token;
          while (std::getline(ss, token, ',')) {
            auto trimmed = Trim(token);
            if (!trimmed.empty())
              words.push_back(trimmed);
          }
          auto resp = client.Put(BuildUrl(host, port, "/v1/admin/guardrails"),
                                 json({{"blocklist", words}}).dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin guardrails --set");
        }
        PrintUsage();
        return 1;
      }
      if (target == "rate-limit") {
        bool get = false;
        bool set_flag = false;
        std::string set_value_raw;
        int new_limit = 0;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--get") {
            get = true;
          } else if (arg == "--set") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin rate-limit",
                                        "--set", "N", &set_value_raw)) {
              return 1;
            }
            set_flag = true;
          }
        }
        const int op_count = (get ? 1 : 0) + (set_flag ? 1 : 0);
        if (op_count == 0) {
          PrintUsage();
          return 1;
        }
        if (op_count > 1) {
          std::cerr << "inferctl admin rate-limit: choose exactly one of "
                       "--get, --set"
                    << std::endl;
          return 1;
        }
        if (get) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/rate_limit"), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin rate-limit --get");
        }
        if (set_flag) {
          try {
            std::size_t consumed = 0;
            new_limit = std::stoi(set_value_raw, &consumed);
            if (consumed != set_value_raw.size()) {
              throw std::invalid_argument("trailing characters");
            }
          } catch (...) {
            std::cerr << "inferctl admin rate-limit: --set must be an integer"
                      << std::endl;
            return 1;
          }
          auto resp = client.Put(
              BuildUrl(host, port, "/v1/admin/rate_limit"),
              json({{"tokens_per_minute", new_limit}}).dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin rate-limit --set");
        }
        PrintUsage();
        return 1;
      }
      if (target == "routing") {
        bool get = false;
        bool set = false;
        bool allow_default_set = false;
        bool require_ready_set = false;
        bool fallback_scope_set = false;
        std::string allow_default_raw;
        std::string require_ready_raw;
        std::string fallback_scope_value;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--get") {
            get = true;
          } else if (arg == "--set") {
            set = true;
          } else if (arg == "--allow-default-fallback") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin routing",
                                        "--allow-default-fallback",
                                        "true|false", &allow_default_raw)) {
              return 1;
            }
            allow_default_set = true;
          } else if (arg == "--require-ready-backend") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin routing",
                                        "--require-ready-backend", "true|false",
                                        &require_ready_raw)) {
              return 1;
            }
            require_ready_set = true;
          } else if (arg == "--fallback-scope") {
            if (!ParseRequiredFlagValue(
                    argc, argv, &i, "admin routing", "--fallback-scope",
                    "any_compatible|same_path_only", &fallback_scope_value)) {
              return 1;
            }
            fallback_scope_set = true;
          }
        }
        const int op_count = (get ? 1 : 0) + (set ? 1 : 0);
        if (op_count == 0) {
          if (allow_default_set || require_ready_set || fallback_scope_set) {
            if (allow_default_set) {
              std::cerr << "inferctl admin routing: --allow-default-fallback "
                           "requires --set"
                        << std::endl;
            } else if (require_ready_set) {
              std::cerr << "inferctl admin routing: --require-ready-backend "
                           "requires --set"
                        << std::endl;
            } else {
              std::cerr << "inferctl admin routing: --fallback-scope requires "
                           "--set"
                        << std::endl;
            }
            return 1;
          }
          PrintUsage();
          return 1;
        }
        if (op_count > 1) {
          std::cerr << "inferctl admin routing: choose exactly one of --get, "
                       "--set"
                    << std::endl;
          return 1;
        }
        if (!set &&
            (allow_default_set || require_ready_set || fallback_scope_set)) {
          if (allow_default_set) {
            std::cerr << "inferctl admin routing: --allow-default-fallback "
                         "requires --set"
                      << std::endl;
          } else if (require_ready_set) {
            std::cerr << "inferctl admin routing: --require-ready-backend "
                         "requires --set"
                      << std::endl;
          } else {
            std::cerr << "inferctl admin routing: --fallback-scope requires "
                         "--set"
                      << std::endl;
          }
          return 1;
        }
        if (get) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/routing"), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin routing --get");
        }

        if (!allow_default_set && !require_ready_set && !fallback_scope_set) {
          std::cerr << "inferctl admin routing: --set requires at least one of "
                       "--allow-default-fallback, --require-ready-backend, "
                       "--fallback-scope"
                    << std::endl;
          return 1;
        }

        json body_j;
        if (allow_default_set) {
          bool parsed_value = false;
          if (!ParseStrictBoolValue(allow_default_raw, &parsed_value)) {
            std::cerr << "inferctl admin routing: --allow-default-fallback "
                         "must be true or false"
                      << std::endl;
            return 1;
          }
          body_j["allow_default_fallback"] = parsed_value;
        }
        if (require_ready_set) {
          bool parsed_value = false;
          if (!ParseStrictBoolValue(require_ready_raw, &parsed_value)) {
            std::cerr << "inferctl admin routing: --require-ready-backend must "
                         "be true or false"
                      << std::endl;
            return 1;
          }
          body_j["require_ready_backend"] = parsed_value;
        }
        if (fallback_scope_set) {
          if (fallback_scope_value != "any_compatible" &&
              fallback_scope_value != "same_path_only") {
            std::cerr << "inferctl admin routing: --fallback-scope must be "
                         "any_compatible or same_path_only"
                      << std::endl;
            return 1;
          }
          body_j["fallback_scope"] = fallback_scope_value;
        }
        auto resp = client.Put(BuildUrl(host, port, "/v1/admin/routing"),
                               body_j.dump(), headers);
        return PrintJsonResponseAndReturn(resp, "inferctl admin routing --set");
      }
      if (target == "pools") {
        bool get = false;
        bool output_json = false;
        bool output_table = false;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--get") {
            get = true;
          } else if (arg == "--json") {
            output_json = true;
          } else if (arg == "--table") {
            output_table = true;
          }
        }
        if (!get) {
          std::cerr << "inferctl admin pools: --get is required" << std::endl;
          return 1;
        }
        if (output_json && output_table) {
          std::cerr << "inferctl admin pools: choose at most one of --json, "
                       "--table"
                    << std::endl;
          return 1;
        }
        const bool render_table = output_table;

        auto pools_resp =
            client.Get(BuildUrl(host, port, "/v1/admin/pools"), headers);
        if (pools_resp.status == 401 || pools_resp.status == 403) {
          std::cerr << "inferctl admin pools --get: authentication required "
                       "(set --api-key or INFERCTL_API_KEY)\n";
          return 1;
        }
        if (IsHttpSuccess(pools_resp.status)) {
          if (render_table) {
            try {
              PrintAdminPoolsTable(json::parse(pools_resp.body));
            } catch (const json::exception &) {
              std::cout << pools_resp.body << std::endl;
            }
          } else {
            std::cout << pools_resp.body << std::endl;
          }
          return 0;
        }
        if (pools_resp.status != 404) {
          std::cout << pools_resp.body << std::endl;
          return 1;
        }

        auto ready_resp = client.Get(BuildUrl(host, port, "/readyz"), headers);
        if (ready_resp.status == 401 || ready_resp.status == 403) {
          std::cerr << "inferctl admin pools --get: authentication required "
                       "(set --api-key or INFERCTL_API_KEY)\n";
          return 1;
        }
        if (!(ready_resp.status == 200 || ready_resp.status == 503)) {
          std::cout << ready_resp.body << std::endl;
          return 1;
        }

        auto metrics_resp =
            client.Get(BuildUrl(host, port, "/metrics"), headers);
        if (metrics_resp.status == 401 || metrics_resp.status == 403) {
          std::cerr << "inferctl admin pools --get: authentication required "
                       "(set --api-key or INFERCTL_API_KEY)\n";
          return 1;
        }
        if (!IsHttpSuccess(metrics_resp.status)) {
          std::cout << metrics_resp.body << std::endl;
          return 1;
        }

        json ready_payload;
        try {
          ready_payload = json::parse(ready_resp.body);
        } catch (const json::exception &) {
          std::cout << ready_resp.body << std::endl;
          return 1;
        }

        auto metric_or_null = [&](const std::string &name) -> json {
          double value = 0.0;
          if (!ParsePrometheusMetricValue(metrics_resp.body, name, &value)) {
            return json(nullptr);
          }
          double rounded = std::round(value);
          if (std::fabs(value - rounded) < 1e-9) {
            return json(static_cast<int64_t>(rounded));
          }
          return json(value);
        };
        auto labeled_metric_or_null =
            [&](const std::string &name,
                const std::vector<std::pair<std::string, std::string>> &labels)
            -> json {
          double value = 0.0;
          if (!ParsePrometheusLabeledMetricValue(metrics_resp.body, name,
                                                 labels, &value)) {
            return json(nullptr);
          }
          double rounded = std::round(value);
          if (std::fabs(value - rounded) < 1e-9) {
            return json(static_cast<int64_t>(rounded));
          }
          return json(value);
        };

        json scheduler_metrics{
            {"queue_depth", metric_or_null("inferflux_scheduler_queue_depth")},
            {"prefill_queue_depth",
             metric_or_null("inferflux_prefill_queue_depth")},
            {"decode_queue_depth",
             metric_or_null("inferflux_decode_queue_depth")},
            {"batch_limit_size",
             metric_or_null("inferflux_scheduler_batch_limit_size")},
            {"batch_limit_tokens",
             metric_or_null("inferflux_scheduler_batch_limit_tokens")},
        };
        json distributed_kv{
            {"enqueue_rejections_total",
             metric_or_null("inferflux_disagg_kv_enqueue_rejections_total")},
            {"enqueue_exhausted_total",
             metric_or_null("inferflux_disagg_kv_enqueue_exhausted_total")},
            {"tickets_enqueued_total",
             labeled_metric_or_null("inferflux_disagg_kv_tickets_total",
                                    {{"stage", "enqueued"}})},
            {"tickets_acknowledged_total",
             labeled_metric_or_null("inferflux_disagg_kv_tickets_total",
                                    {{"stage", "acknowledged"}})},
            {"tickets_committed_total",
             labeled_metric_or_null("inferflux_disagg_kv_tickets_total",
                                    {{"stage", "committed"}})},
            {"tickets_timed_out_total",
             labeled_metric_or_null("inferflux_disagg_kv_tickets_total",
                                    {{"stage", "timed_out"}})},
        };

        json payload{
            {"status", "ok"},
            {"pool_health",
             {{"ready", ready_payload.value("status", "") == "ready"},
              {"http_status", ready_resp.status},
              {"role", ready_payload.value("role", "unknown")},
              {"reason", ready_payload.value("reason", "")},
              {"model_loaded", ready_payload.value("model_loaded", false)},
              {"decode_pool_warm",
               ready_payload.value("decode_pool_warm", false)},
              {"disagg_transport_degraded",
               ready_payload.value("disagg_transport_degraded", false)},
              {"disagg_timeout_debt",
               ready_payload.contains("disagg_timeout_debt")
                   ? ready_payload["disagg_timeout_debt"]
                   : metric_or_null("inferflux_disagg_kv_timeout_debt")},
              {"disagg_timeout_debt_threshold",
               ready_payload.contains("disagg_timeout_debt_threshold")
                   ? ready_payload["disagg_timeout_debt_threshold"]
                   : json(nullptr)},
              {"disagg_timeout_streak",
               ready_payload.contains("disagg_timeout_streak")
                   ? ready_payload["disagg_timeout_streak"]
                   : json(nullptr)},
              {"disagg_timeout_streak_threshold",
               ready_payload.contains("disagg_timeout_streak_threshold")
                   ? ready_payload["disagg_timeout_streak_threshold"]
                   : json(nullptr)}}},
            {"scheduler", scheduler_metrics},
            {"distributed_kv", distributed_kv},
        };
        if (render_table) {
          PrintAdminPoolsTable(payload);
        } else {
          std::cout << payload.dump() << std::endl;
        }
        return 0;
      }
      if (target == "models") {
        bool list = false;
        bool list_json = false;
        std::string load_path;
        std::string load_backend;
        std::string load_id;
        bool load_default = false;
        std::string unload_id;
        std::string set_default_id;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list") {
            list = true;
          } else if (arg == "--json") {
            list_json = true;
          } else if (arg == "--load") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin models",
                                        "--load", "PATH", &load_path)) {
              return 1;
            }
          } else if (arg == "--backend") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin models",
                                        "--backend", "TYPE", &load_backend)) {
              return 1;
            }
          } else if (arg == "--id") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin models", "--id",
                                        "NAME", &load_id)) {
              return 1;
            }
          } else if (arg == "--default") {
            load_default = true;
          } else if (arg == "--unload") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin models",
                                        "--unload", "ID", &unload_id)) {
              return 1;
            }
          } else if (arg == "--set-default") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin models",
                                        "--set-default", "ID",
                                        &set_default_id)) {
              return 1;
            }
          }
        }
        const int op_count = (list ? 1 : 0) + (!load_path.empty() ? 1 : 0) +
                             (!unload_id.empty() ? 1 : 0) +
                             (!set_default_id.empty() ? 1 : 0);
        if (op_count == 0) {
          if (list_json) {
            std::cerr << "inferctl admin models: --json requires --list"
                      << std::endl;
            return 1;
          }
          if (load_default) {
            std::cerr << "inferctl admin models: --default requires --load "
                         "PATH"
                      << std::endl;
            return 1;
          }
          if (!load_backend.empty()) {
            std::cerr << "inferctl admin models: --backend requires --load "
                         "PATH"
                      << std::endl;
            return 1;
          }
          if (!load_id.empty()) {
            std::cerr << "inferctl admin models: --id requires --load PATH"
                      << std::endl;
            return 1;
          }
          PrintUsage();
          return 1;
        }
        if (op_count > 1) {
          std::cerr << "inferctl admin models: choose exactly one of "
                       "--list, --load, --unload, --set-default"
                    << std::endl;
          return 1;
        }
        if (!list && list_json) {
          std::cerr << "inferctl admin models: --json requires --list"
                    << std::endl;
          return 1;
        }
        if (load_path.empty()) {
          if (load_default) {
            std::cerr << "inferctl admin models: --default requires --load "
                         "PATH"
                      << std::endl;
            return 1;
          }
          if (!load_backend.empty()) {
            std::cerr << "inferctl admin models: --backend requires --load "
                         "PATH"
                      << std::endl;
            return 1;
          }
          if (!load_id.empty()) {
            std::cerr << "inferctl admin models: --id requires --load PATH"
                      << std::endl;
            return 1;
          }
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/models"), headers);
          if (resp.status == 401 || resp.status == 403) {
            std::cerr << "inferctl admin models --list: authentication "
                         "required (set --api-key or INFERCTL_API_KEY)\n";
            return 1;
          }
          if (!IsHttpSuccess(resp.status)) {
            std::cout << resp.body << std::endl;
            return 1;
          }
          if (list_json) {
            std::cout << resp.body << std::endl;
            return 0;
          }
          try {
            auto j = json::parse(resp.body);
            PrintAdminModelsTable(j);
          } catch (const json::exception &) {
            std::cout << resp.body << std::endl;
          }
          return 0;
        }
        if (!load_path.empty()) {
          json payload;
          payload["path"] = load_path;
          if (!load_backend.empty()) {
            payload["backend"] = load_backend;
          }
          if (!load_id.empty()) {
            payload["id"] = load_id;
          }
          if (load_default) {
            payload["default"] = true;
          }
          auto resp = client.Post(BuildUrl(host, port, "/v1/admin/models"),
                                  payload.dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin models --load");
        }
        if (!unload_id.empty()) {
          auto resp = client.Delete(BuildUrl(host, port, "/v1/admin/models"),
                                    json({{"id", unload_id}}).dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin models --unload");
        }
        if (!set_default_id.empty()) {
          auto resp =
              client.Put(BuildUrl(host, port, "/v1/admin/models/default"),
                         json({{"id", set_default_id}}).dump(), headers);
          return PrintJsonResponseAndReturn(
              resp, "inferctl admin models --set-default");
        }
        PrintUsage();
        return 1;
      }
      if (target == "cache") {
        bool status = false;
        bool do_warm = false;
        std::string warm_tokens_str;
        std::string warm_completion;
        bool warm_completion_tokens_set = false;
        int warm_completion_tokens = 0;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--status") {
            status = true;
          } else if (arg == "--warm") {
            do_warm = true;
          } else if (arg == "--tokens") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin cache",
                                        "--tokens", "ID,ID,...",
                                        &warm_tokens_str)) {
              return 1;
            }
          } else if (arg == "--completion") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin cache",
                                        "--completion", "TEXT",
                                        &warm_completion)) {
              return 1;
            }
          } else if (arg == "--completion-tokens") {
            std::string completion_tokens_value;
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin cache",
                                        "--completion-tokens", "N",
                                        &completion_tokens_value)) {
              return 1;
            }
            try {
              warm_completion_tokens = std::stoi(completion_tokens_value);
            } catch (...) {
              std::cerr
                  << "inferctl admin cache: --completion-tokens must be an "
                     "integer"
                  << std::endl;
              return 1;
            }
            warm_completion_tokens_set = true;
          }
        }
        if (status && do_warm) {
          std::cerr << "inferctl admin cache: choose exactly one of --status "
                       "or --warm"
                    << std::endl;
          return 1;
        }
        if (!status && !do_warm) {
          status = true;
        }
        if (!do_warm && (!warm_tokens_str.empty() || !warm_completion.empty() ||
                         warm_completion_tokens_set)) {
          if (!warm_tokens_str.empty()) {
            std::cerr << "inferctl admin cache: --tokens requires --warm"
                      << std::endl;
          } else if (!warm_completion.empty()) {
            std::cerr << "inferctl admin cache: --completion requires --warm"
                      << std::endl;
          } else {
            std::cerr << "inferctl admin cache: --completion-tokens requires "
                         "--warm"
                      << std::endl;
          }
          return 1;
        }
        if (do_warm) {
          if (warm_tokens_str.empty()) {
            std::cerr << "inferctl admin cache: --tokens is required with "
                         "--warm"
                      << std::endl;
            return 1;
          }
          if (warm_completion.empty()) {
            std::cerr << "inferctl admin cache: --completion is required with "
                         "--warm"
                      << std::endl;
            return 1;
          }
          json tokens_arr = json::array();
          std::stringstream tok_ss(warm_tokens_str);
          std::string tok;
          while (std::getline(tok_ss, tok, ',')) {
            try {
              tokens_arr.push_back(std::stoi(Trim(tok)));
            } catch (...) {
              std::cerr << "Invalid token id: " << tok << "\n";
              return 1;
            }
          }
          json body_j{{"tokens", tokens_arr},
                      {"completion", warm_completion},
                      {"completion_tokens", warm_completion_tokens}};
          auto resp = client.Post(BuildUrl(host, port, "/v1/admin/cache/warm"),
                                  body_j.dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin cache --warm");
        }
        auto resp =
            client.Get(BuildUrl(host, port, "/v1/admin/cache"), headers);
        return PrintJsonResponseAndReturn(resp,
                                          "inferctl admin cache --status");
      }
      if (target == "api-keys" || target == "api-key") {
        bool list = false, remove = false, add = false;
        std::string add_key, remove_key, scopes_csv;
        bool scopes_set = false;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list")
            list = true;
          else if (arg == "--add") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin api-keys",
                                        "--add", "KEY", &add_key)) {
              return 1;
            }
            add = true;
          } else if (arg == "--scopes") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin api-keys",
                                        "--scopes", "read,admin",
                                        &scopes_csv)) {
              return 1;
            }
            scopes_set = true;
          } else if (arg == "--remove") {
            if (!ParseRequiredFlagValue(argc, argv, &i, "admin api-keys",
                                        "--remove", "KEY", &remove_key)) {
              return 1;
            }
            remove = true;
          }
        }
        const int op_count = (list ? 1 : 0) + (add ? 1 : 0) + (remove ? 1 : 0);
        if (op_count == 0) {
          if (scopes_set) {
            std::cerr << "inferctl admin api-keys: --scopes requires --add KEY"
                      << std::endl;
            return 1;
          }
          PrintUsage();
          return 1;
        }
        if (op_count > 1) {
          std::cerr << "inferctl admin api-keys: choose exactly one of --list, "
                       "--add, --remove"
                    << std::endl;
          return 1;
        }
        if (!add && scopes_set) {
          std::cerr << "inferctl admin api-keys: --scopes requires --add KEY"
                    << std::endl;
          return 1;
        }
        if (add && !scopes_set) {
          std::cerr
              << "inferctl admin api-keys: --add requires --scopes read,admin"
              << std::endl;
          return 1;
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/api_keys"), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin api-keys --list");
        }
        if (add) {
          std::vector<std::string> scopes;
          std::stringstream ss(scopes_csv);
          std::string token;
          while (std::getline(ss, token, ',')) {
            auto trimmed = Trim(token);
            if (!trimmed.empty())
              scopes.push_back(trimmed);
          }
          if (scopes.empty()) {
            std::cerr << "inferctl admin api-keys: --scopes must include at "
                         "least one scope"
                      << std::endl;
            return 1;
          }
          auto resp = client.Post(
              BuildUrl(host, port, "/v1/admin/api_keys"),
              json({{"key", add_key}, {"scopes", scopes}}).dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin api-keys --add");
        }
        if (remove) {
          auto resp =
              client.Delete(BuildUrl(host, port, "/v1/admin/api_keys"),
                            json({{"key", remove_key}}).dump(), headers);
          return PrintJsonResponseAndReturn(resp,
                                            "inferctl admin api-keys --remove");
        }
        PrintUsage();
        return 1;
      }
      PrintUsage();
      return 1;
    }
    if (command == "status") {
      auto resp = client.Get(BuildUrl(host, port, "/healthz"), headers);
      std::cout << resp.body << std::endl;
      return 0;
    }
    if (command == "models") {
      bool output_json = false;
      std::string model_id;
      for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
          output_json = true;
        } else if (arg == "--id") {
          if (!ParseRequiredFlagValue(argc, argv, &i, "models", "--id",
                                      "MODEL_ID", &model_id)) {
            return 1;
          }
        }
      }
      const std::string path =
          model_id.empty() ? "/v1/models" : "/v1/models/" + model_id;
      auto resp = client.Get(BuildUrl(host, port, path), headers);
      if (resp.status == 401 || resp.status == 403) {
        std::cerr << "inferctl models: authentication required (set --api-key "
                     "or INFERCTL_API_KEY)\n";
        return 1;
      }
      if (!IsHttpSuccess(resp.status)) {
        std::cout << resp.body << std::endl;
        return 1;
      }
      if (output_json) {
        std::cout << resp.body << std::endl;
        return 0;
      }
      try {
        auto j = json::parse(resp.body);
        if (!model_id.empty()) {
          // Reuse list-style table formatting for single-model lookups by
          // wrapping the object into a synthetic data array.
          j = json{{"data", json::array({j})}};
        }
        if (!j.contains("data") || !j["data"].is_array()) {
          std::cout << resp.body << std::endl;
          return 0;
        }
        PrintModelsTable(j);
      } catch (const json::exception &) {
        std::cout << resp.body << std::endl;
      }
      return 0;
    }
    if (command == "completion") {
      if (prompt.empty()) {
        std::cerr << "--prompt is required" << std::endl;
        return 1;
      }
      // Use /v1/chat/completions (the legacy /v1/completions is deprecated).
      std::vector<ChatMessage> msgs{{"user", prompt}};
      auto payload = BuildChatPayload(msgs, model, max_tokens, stream_mode);
      std::string url = BuildUrl(host, port, "/v1/chat/completions");
      if (stream_mode) {
        auto conn = client.SendRaw("POST", url, payload.dump(), headers);
        ReceiveStream(client, conn, nullptr);
      } else {
        auto resp = client.Post(url, payload.dump(), headers);
        std::cout << resp.body << std::endl;
      }
      return 0;
    }
    if (command == "chat") {
      if (interactive_mode) {
        InteractiveChat(host, port, api_key, model, max_tokens, stream_mode);
        return 0;
      }
      if (chat_messages.empty()) {
        if (prompt.empty()) {
          std::cerr << "Provide at least one --message role:text or --prompt"
                    << std::endl;
          return 1;
        }
        chat_messages.push_back(ChatMessage{"user", prompt});
      }
      auto payload =
          BuildChatPayload(chat_messages, model, max_tokens, stream_mode);
      std::string url = BuildUrl(host, port, "/v1/chat/completions");
      if (stream_mode) {
        auto conn = client.SendRaw("POST", url, payload.dump(), headers);
        ReceiveStream(client, conn, nullptr);
      } else {
        auto resp = client.Post(url, payload.dump(), headers);
        std::cout << resp.body << std::endl;
      }
      return 0;
    }
  } catch (const std::exception &ex) {
    std::cerr << "inferctl error: " << ex.what() << std::endl;
    return 1;
  }

  PrintUsage();
  return 1;
}
