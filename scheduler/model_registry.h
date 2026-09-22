#pragma once

#include "scheduler/model_router.h"

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace inferflux {

// ── Registry entry ──────────────────────────────────────────────────────────
// Represents one model entry in registry.yaml.
using RegistryEntry = ModelLoadSpec;

// ── ModelRegistry ────────────────────────────────────────────────────────────
// Loads a YAML registry file and hot-reloads it when the file changes on disk.
//
// Format of registry.yaml:
//
//   models:
//     - id: llama3-8b
//       path: /models/llama3-8b-q4.gguf
//       format: gguf
//       backend: cpu
//     - path: /models/mistral-7b.gguf   # id derived from filename
//
// Lifecycle:
//   ModelRegistry reg(router);
//   reg.LoadAndWatch("/etc/inferflux/registry.yaml", /*poll_ms=*/5000);
//   // ... server runs ...
//   reg.Stop();   // (or let destructor call it)
//
// Thread safety: all public methods are thread-safe.
class ModelRegistry {
public:
  explicit ModelRegistry(std::shared_ptr<ModelRouter> router);
  ~ModelRegistry();

  // Parse the registry file and start a background polling thread.
  // Returns the count of models successfully loaded from the initial read.
  // Calling this a second time is a no-op (returns 0).
  int LoadAndWatch(const std::filesystem::path &path,
                   int poll_interval_ms = 5000);

  // Force an immediate re-read and diff against the current state.
  // Called automatically by the watcher loop; may also be called externally.
  // Returns the number of models loaded/unloaded in this cycle (+ for loads,
  // - for unloads, net count).
  int Reload();

  // Stop the background watcher thread. Safe to call more than once.
  void Stop();

  // True when a registry file has been loaded and the watcher is running.
  bool IsWatching() const { return running_.load(); }

  // Current set of model IDs managed by this registry (for diagnostics).
  std::set<std::string> ManagedIds() const;

private:
  // Background loop: polls file mtime every poll_interval_ms_.
  void WatchLoop();

  // Parse YAML content into entries. Returns false on parse error.
  bool ParseYaml(const std::string &yaml_text,
                 std::vector<RegistryEntry> &out) const;

  // Apply a new set of entries: load newcomers, unload removed models.
  // Returns net change count.
  int ApplyEntries(const std::vector<RegistryEntry> &entries);

  std::shared_ptr<ModelRouter> router_;
  std::filesystem::path path_;
  int poll_interval_ms_{5000};

  std::atomic<bool> running_{false};
  std::thread watch_thread_;

  // Last observed file modification time (used to skip unchanged reads).
  std::filesystem::file_time_type last_mtime_{};

  // Model IDs currently managed by this registry instance.
  // Keyed by the path (to detect path→id mapping changes on reload).
  mutable std::mutex managed_mutex_;
  // path → assigned_id (as returned by router_->LoadModel())
  std::map<std::string, std::string> path_to_id_;
  std::map<std::string, RegistryEntry> specifications_;
};

} // namespace inferflux
