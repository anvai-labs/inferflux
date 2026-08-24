#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace inferflux {
namespace disaggregated {

enum class KVTicketStage {
  kNone,
  kEnqueued,
  kAcknowledged,
  kCommitted,
  kTimedOut,
};

inline std::string_view KVTicketStageToString(KVTicketStage stage) {
  switch (stage) {
  case KVTicketStage::kEnqueued:
    return "enqueued";
  case KVTicketStage::kAcknowledged:
    return "acknowledged";
  case KVTicketStage::kCommitted:
    return "committed";
  case KVTicketStage::kTimedOut:
    return "timed_out";
  case KVTicketStage::kNone:
  default:
    return "none";
  }
}

struct KVPacket {
  uint64_t request_id{0};
  uint64_t ticket_id{0};
  KVTicketStage ticket_stage{KVTicketStage::kNone};
  std::vector<uint8_t> prompt_tokens;
  std::vector<uint8_t> kv_blob;
  int kv_page{-1};
  int n_past{-1}; // Filled after phased Prefill(); -1 if not yet prefilled.
  int sequence_id{
      -1}; // KV cache sequence slot used during Prefill(); -1 = unassigned.
  std::string metadata;
  std::chrono::steady_clock::time_point enqueue_time{
      std::chrono::steady_clock::now()};
};

// Common interface for KV-state transports (in-process queue and POSIX SHM).
// Both KVChannel and ShmKVTransport implement this so they are substitutable
// in DisaggregatedConfig without touching Scheduler or BatchExecutor.
class IKVTransport {
public:
  virtual ~IKVTransport() = default;
  virtual bool Enqueue(KVPacket packet) = 0;
  virtual std::optional<KVPacket> TryDequeue() = 0;
  virtual std::size_t Size() const = 0;
  virtual std::size_t Capacity() const = 0;
  virtual bool IsProcessLocal() const { return false; }
  virtual bool UpdateTicketStage(uint64_t ticket_id, KVTicketStage stage) {
    (void)ticket_id;
    (void)stage;
    return false;
  }
  virtual KVTicketStage GetTicketStage(uint64_t ticket_id) const {
    (void)ticket_id;
    return KVTicketStage::kNone;
  }
};

// Thread-safe in-process queue.  Default transport when INFERFLUX_KV_TRANSPORT
// is unset or set to "channel".
class KVChannel : public IKVTransport {
public:
  explicit KVChannel(std::size_t capacity = 64);

  void SetCapacity(std::size_t capacity);
  std::size_t Capacity() const override;
  bool IsProcessLocal() const override { return true; }

  bool Enqueue(KVPacket packet) override;
  std::optional<KVPacket> TryDequeue() override;
  std::size_t Size() const override;
  bool UpdateTicketStage(uint64_t ticket_id, KVTicketStage stage) override;
  KVTicketStage GetTicketStage(uint64_t ticket_id) const override;
  void Clear();

private:
  mutable std::mutex mutex_;
  std::deque<KVPacket> queue_;
  std::unordered_map<uint64_t, KVTicketStage> ticket_stages_;
  std::size_t capacity_;
};

} // namespace disaggregated
} // namespace inferflux
