#pragma once

#include "runtime/disaggregated/kv_channel.h"

#include <cstdint>
#include <optional>
#include <string>

namespace inferflux {
namespace disaggregated {

// POSIX shared-memory backed KV transport (§2.5 item 11).
//
// Each Enqueue() stores packet.kv_blob in a uniquely-named SHM segment
// (shm_open + mmap + memcpy) and enqueues a lightweight control packet
// that carries only the segment name and blob size.  TryDequeue() reads the
// control packet, maps the SHM segment, copies the blob back into the
// packet, then unlinks the segment.
//
// Because the OS shares physical pages between writer and reader, the
// actual data-transfer cost is near-zero for in-process and sub-millisecond
// for cross-process use on the same host — meeting the <5 ms SLA target.
//
// Cross-process use: both writer and reader must use the same capacity value
// and call shm_open with the segment name embedded in packet.metadata.
// Leaked segments (e.g. after a crash) can be cleaned with shm_unlink(3).
class ShmKVTransport : public IKVTransport {
public:
  explicit ShmKVTransport(std::size_t capacity = 64);
  ~ShmKVTransport() override;

  // Store packet.kv_blob in a named SHM segment, enqueue control metadata.
  // For empty kv_blob, enqueues a control-only packet with no SHM segment.
  // Returns false if the control queue is full or SHM creation fails.
  bool Enqueue(KVPacket packet) override;

  // Dequeue control metadata, map and read the SHM blob, unlink the segment,
  // return the reassembled packet.  Returns nullopt when the queue is empty.
  std::optional<KVPacket> TryDequeue() override;

  std::size_t Capacity() const override;
  std::size_t Size() const override;
  bool UpdateTicketStage(uint64_t ticket_id, KVTicketStage stage) override;
  KVTicketStage GetTicketStage(uint64_t ticket_id) const override;

private:
  // Generate a unique SHM segment name embedding request_id and a process-
  // level counter to avoid collisions on rapid reuse of the same request_id.
  static std::string MakeShmName(uint64_t request_id);

  // Extract "shm=<name>" from the pipe-delimited metadata string.
  static std::string ExtractShmName(const std::string &metadata);

  // Extract "|size=<N>" from the pipe-delimited metadata string.
  static std::size_t ExtractShmSize(const std::string &metadata);

  KVChannel control_queue_;
};

} // namespace disaggregated
} // namespace inferflux
