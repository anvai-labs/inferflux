#include "runtime/disaggregated/kv_channel.h"

#include <algorithm>

namespace inferflux {
namespace disaggregated {

namespace {

void AppendControlMetadata(KVPacket *packet) {
  if (!packet) {
    return;
  }
  if (packet->ticket_id > 0) {
    packet->metadata += "|ticket=" + std::to_string(packet->ticket_id);
    packet->metadata +=
        "|ticket_stage=" +
        std::string(KVTicketStageToString(packet->ticket_stage));
  }
  if (packet->kv_page >= 0) {
    packet->metadata += "|kv_page=" + std::to_string(packet->kv_page);
  }
}

} // namespace

KVChannel::KVChannel(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void KVChannel::SetCapacity(std::size_t capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  capacity_ = std::max<std::size_t>(1, capacity);
  while (queue_.size() > capacity_) {
    queue_.pop_front();
  }
}

std::size_t KVChannel::Capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return capacity_;
}

bool KVChannel::Enqueue(KVPacket packet) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.size() >= capacity_) {
    return false;
  }
  if (packet.ticket_id > 0) {
    ticket_stages_[packet.ticket_id] = packet.ticket_stage;
  }
  AppendControlMetadata(&packet);
  queue_.push_back(std::move(packet));
  return true;
}

std::optional<KVPacket> KVChannel::TryDequeue() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  KVPacket packet = std::move(queue_.front());
  queue_.pop_front();
  return packet;
}

std::size_t KVChannel::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool KVChannel::UpdateTicketStage(uint64_t ticket_id, KVTicketStage stage) {
  if (ticket_id == 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = ticket_stages_.find(ticket_id);
  if (it == ticket_stages_.end()) {
    return false;
  }
  it->second = stage;
  return true;
}

KVTicketStage KVChannel::GetTicketStage(uint64_t ticket_id) const {
  if (ticket_id == 0) {
    return KVTicketStage::kNone;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = ticket_stages_.find(ticket_id);
  if (it == ticket_stages_.end()) {
    return KVTicketStage::kNone;
  }
  return it->second;
}

void KVChannel::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.clear();
  ticket_stages_.clear();
}

} // namespace disaggregated
} // namespace inferflux
