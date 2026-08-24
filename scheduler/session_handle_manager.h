#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {
namespace scheduler {

struct SessionHandleState {
  std::string model_id;
  int sequence_id{-1};
  uint64_t sequence_generation{0};
  std::vector<int> prompt_tokens;
  std::vector<int> block_table;
};

class SessionHandleManager {
public:
  struct Config {
    std::size_t max_sessions{1024};
    std::chrono::milliseconds ttl{std::chrono::minutes(5)};
  };

  struct LeaseResult {
    enum class Status {
      kAcquired,
      kBusy,
    };

    Status status{Status::kBusy};
    bool has_state{false};
    SessionHandleState state;
    std::vector<SessionHandleState> cleanup_states;
  };

  SessionHandleManager();
  explicit SessionHandleManager(const Config &config);

  LeaseResult AcquireLease(const std::string &session_id);
  void CommitLease(const std::string &session_id,
                   const SessionHandleState &state);
  void ReleaseLease(const std::string &session_id);
  void DiscardLeasedState(const std::string &session_id);
  std::vector<SessionHandleState> CollectExpired();
  std::vector<SessionHandleState> DrainAll();
  std::size_t SessionCount() const;
  Config CurrentConfig() const { return config_; }

private:
  struct Entry {
    SessionHandleState state;
    bool has_state{false};
    bool in_use{false};
    std::chrono::steady_clock::time_point last_access;
    std::chrono::steady_clock::time_point expires_at;
  };

  using EntryMap = std::unordered_map<std::string, Entry>;
  EntryMap::iterator FindEvictionCandidateLocked();

  Config config_;
  mutable std::mutex mutex_;
  EntryMap entries_;
};

} // namespace scheduler
} // namespace inferflux
