#include "model/model_format.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace inferflux {

namespace {

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string InferfluxHomePath() {
  if (const char *env_home = std::getenv("INFERFLUX_HOME")) {
    if (*env_home != '\0') {
      return env_home;
    }
  }
  if (const char *env_user_home = std::getenv("HOME")) {
    if (*env_user_home != '\0') {
      return (std::filesystem::path(env_user_home) / ".inferflux").string();
    }
  }
  return (std::filesystem::current_path() / ".inferflux").string();
}

std::string ExtractHfRepoId(const std::string &path) {
  const std::string lowered = ToLower(path);
  std::size_t prefix_len = 0;
  if (lowered.rfind("hf://", 0) == 0) {
    prefix_len = 5;
  } else if (lowered.rfind("huggingface://", 0) == 0) {
    prefix_len = std::string("huggingface://").size();
  } else {
    return "";
  }
  std::string repo = path.substr(prefix_len);
  while (!repo.empty() && repo.front() == '/') {
    repo.erase(repo.begin());
  }
  while (!repo.empty() && repo.back() == '/') {
    repo.pop_back();
  }
  return repo;
}

int ScoreGgufFile(const std::string &file_name) {
  const std::string lower = ToLower(file_name);
  if (lower.find("q4_k_m") != std::string::npos) {
    return 100;
  }
  if (lower.find("q4_k_s") != std::string::npos) {
    return 90;
  }
  if (lower.find("q4_k") != std::string::npos) {
    return 85;
  }
  if (lower.find("q4_0") != std::string::npos) {
    return 80;
  }
  if (lower.find("q4") != std::string::npos) {
    return 75;
  }
  if (lower.find("q5_k_m") != std::string::npos) {
    return 70;
  }
  if (lower.find("q5") != std::string::npos) {
    return 65;
  }
  if (lower.find("q8_0") != std::string::npos) {
    return 50;
  }
  if (lower.find(".gguf") != std::string::npos) {
    return 10;
  }
  return -1;
}

std::filesystem::path
SelectBestGgufInDirectory(const std::filesystem::path &dir) {
  std::error_code ec;
  std::filesystem::directory_iterator it(dir, ec);
  if (ec) {
    return {};
  }

  std::filesystem::path best_path;
  int best_score = -1;
  std::string best_name;

  for (const auto &entry : it) {
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }
    const auto ext = ToLower(entry.path().extension().string());
    if (ext != ".gguf") {
      continue;
    }
    const auto name = entry.path().filename().string();
    const int score = ScoreGgufFile(name);
    if (score > best_score || (score == best_score && name < best_name)) {
      best_score = score;
      best_name = name;
      best_path = entry.path();
    }
  }
  return best_path;
}

bool DirectoryContainsExtension(const std::filesystem::path &dir,
                                const std::string &ext) {
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      return false;
    }
    if (!entry.is_regular_file(ec) || ec) {
      continue;
    }
    const auto entry_ext = ToLower(entry.path().extension().string());
    if (entry_ext == ext) {
      return true;
    }
  }
  return false;
}

} // namespace

std::string NormalizeModelFormat(const std::string &format) {
  auto normalized = ToLower(format);
  if (normalized.empty()) {
    return "";
  }
  if (normalized == "auto" || normalized == "gguf") {
    return normalized;
  }
  if (normalized == "safetensors" || normalized == "safetensor" ||
      normalized == "safe_tensors") {
    return "safetensors";
  }
  if (normalized == "hf" || normalized == "huggingface") {
    return "hf";
  }
  return "";
}

bool IsModelFormatValue(const std::string &format) {
  return !NormalizeModelFormat(format).empty();
}

std::string DetectModelFormat(const std::string &path) {
  if (path.empty()) {
    return "unknown";
  }

  const auto lowered_path = ToLower(path);
  if (lowered_path.rfind("hf://", 0) == 0 ||
      lowered_path.rfind("huggingface://", 0) == 0) {
    return "hf";
  }

  const std::filesystem::path fs_path(path);
  std::error_code ec;
  const auto status = std::filesystem::status(fs_path, ec);
  if (ec) {
    // Path may not exist yet (for example mounted later); rely on extension.
    const auto ext = ToLower(fs_path.extension().string());
    if (ext == ".gguf") {
      return "gguf";
    }
    if (ext == ".safetensors") {
      return "safetensors";
    }
    return "unknown";
  }

  if (std::filesystem::is_regular_file(status)) {
    const auto ext = ToLower(fs_path.extension().string());
    if (ext == ".gguf") {
      return "gguf";
    }
    if (ext == ".safetensors") {
      return "safetensors";
    }
    return "unknown";
  }

  if (!std::filesystem::is_directory(status)) {
    return "unknown";
  }

  if (DirectoryContainsExtension(fs_path, ".gguf")) {
    return "gguf";
  }

  if (std::filesystem::exists(fs_path / "model.safetensors") ||
      std::filesystem::exists(fs_path / "model.safetensors.index.json") ||
      DirectoryContainsExtension(fs_path, ".safetensors")) {
    return "safetensors";
  }

  if (std::filesystem::exists(fs_path / "config.json")) {
    return "hf";
  }

  return "unknown";
}

std::string ResolveModelFormat(const std::string &path,
                               const std::string &requested_format) {
  std::string normalized_requested = NormalizeModelFormat(requested_format);
  if (normalized_requested.empty()) {
    normalized_requested = requested_format.empty() ? "auto" : "";
  }
  if (normalized_requested.empty()) {
    return "unknown";
  }

  if (normalized_requested != "auto") {
    return normalized_requested;
  }

  const std::string detected = DetectModelFormat(path);
  if (detected != "unknown") {
    return detected;
  }
  // Backward-compatible default for paths without extension.
  return "gguf";
}

std::string ResolveHfReferenceToCachePath(const std::string &path) {
  const std::string repo = ExtractHfRepoId(path);
  if (repo.empty()) {
    return path;
  }
  auto cache_path =
      std::filesystem::path(InferfluxHomePath()) / "models" / repo;
  return cache_path.make_preferred().string();
}

std::string ResolveMlxLoadPath(const std::string &path,
                               const std::string &resolved_format) {
  const std::string normalized_format = NormalizeModelFormat(resolved_format);
  if (normalized_format != "hf" && normalized_format != "safetensors") {
    return path;
  }

  std::string mlx_path = ResolveHfReferenceToCachePath(path);
  const std::filesystem::path fs_path(mlx_path);
  if (ToLower(fs_path.extension().string()) == ".safetensors") {
    return fs_path.parent_path().string();
  }
  return mlx_path;
}

std::string ResolveLlamaLoadPath(const std::string &path,
                                 const std::string &resolved_format) {
  const std::string normalized_format = NormalizeModelFormat(resolved_format);
  if (normalized_format.empty()) {
    return "";
  }
  if (normalized_format == "gguf") {
    return path;
  }
  if (normalized_format != "hf" && normalized_format != "safetensors") {
    return "";
  }

  if (ToLower(std::filesystem::path(path).extension().string()) == ".gguf") {
    return path;
  }

  // hf://org/repo → ~/.inferflux/models/org/repo/*.gguf
  {
    const std::string hf_cache_path = ResolveHfReferenceToCachePath(path);
    if (hf_cache_path != path) {
      const auto best = SelectBestGgufInDirectory(hf_cache_path);
      if (!best.empty()) {
        return best.string();
      }
    }
  }

  // Local model directory with a GGUF sidecar.
  {
    std::error_code ec;
    std::filesystem::path local_path(path);
    const auto status = std::filesystem::status(local_path, ec);
    if (!ec && std::filesystem::is_directory(status)) {
      const auto best = SelectBestGgufInDirectory(local_path);
      if (!best.empty()) {
        return best.string();
      }
    }
  }

  return "";
}

bool IsLoadableByLlamaBackend(const std::string &resolved_format) {
  return NormalizeModelFormat(resolved_format) == "gguf";
}

} // namespace inferflux
