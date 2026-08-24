#include "runtime/disaggregated/shm_kv_transport.h"
#include "server/logging/logger.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <sstream>

#ifdef _WIN32
// Stub: POSIX SHM not available on Windows.
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace inferflux {
namespace disaggregated {

namespace {
// Process-level counter to make SHM segment names unique even when the same
// request_id is reused rapidly (e.g. during fairness-yield loops).
std::atomic<uint64_t> g_shm_seq{0};
} // namespace

ShmKVTransport::ShmKVTransport(std::size_t capacity)
    : control_queue_(capacity) {}

ShmKVTransport::~ShmKVTransport() = default;

// static
std::string ShmKVTransport::MakeShmName(uint64_t request_id) {
  std::ostringstream oss;
  oss << "/ifx_kv_" << request_id << "_"
      << g_shm_seq.fetch_add(1, std::memory_order_relaxed);
  return oss.str();
}

// static
std::string ShmKVTransport::ExtractShmName(const std::string &metadata) {
  auto pos = metadata.find("|shm=");
  if (pos == std::string::npos)
    return {};
  pos += 5; // skip "|shm="
  auto end = metadata.find('|', pos);
  return metadata.substr(pos, end == std::string::npos ? std::string::npos
                                                       : end - pos);
}

// static
std::size_t ShmKVTransport::ExtractShmSize(const std::string &metadata) {
  auto pos = metadata.find("|size=");
  if (pos == std::string::npos)
    return 0;
  pos += 6; // skip "|size="
  try {
    return static_cast<std::size_t>(std::stoull(metadata.substr(pos)));
  } catch (...) {
    return 0;
  }
}

bool ShmKVTransport::Enqueue(KVPacket packet) {
  if (packet.kv_blob.empty()) {
    // No KV state: pass through as a control-only packet.
    return control_queue_.Enqueue(std::move(packet));
  }

#ifdef _WIN32
  // Windows stub: fall back to in-process enqueue (kv_blob stays inline).
  return control_queue_.Enqueue(std::move(packet));
#else
  const std::string shm_name = MakeShmName(packet.request_id);
  const std::size_t blob_size = packet.kv_blob.size();

  int fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd < 0) {
    log::Warn("shm_transport",
              "shm_open(write) failed for " + shm_name + ": " +
                  std::strerror(errno) +
                  "; falling back to inline control queue payload");
    return control_queue_.Enqueue(std::move(packet));
  }

  if (::ftruncate(fd, static_cast<off_t>(blob_size)) != 0) {
    log::Warn("shm_transport",
              std::string("ftruncate failed: ") + std::strerror(errno) +
                  "; falling back to inline control queue payload");
    ::close(fd);
    ::shm_unlink(shm_name.c_str());
    return control_queue_.Enqueue(std::move(packet));
  }

  void *ptr =
      ::mmap(nullptr, blob_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (ptr == MAP_FAILED) {
    log::Warn("shm_transport",
              std::string("mmap(write) failed: ") + std::strerror(errno) +
                  "; falling back to inline control queue payload");
    ::shm_unlink(shm_name.c_str());
    return control_queue_.Enqueue(std::move(packet));
  }

  std::memcpy(ptr, packet.kv_blob.data(), blob_size);
  ::munmap(ptr, blob_size);

  // Encode SHM name + size in metadata; clear the inline blob to avoid
  // duplicating up to hundreds of MB in the control queue.
  packet.metadata += "|shm=" + shm_name + "|size=" + std::to_string(blob_size);
  packet.kv_blob.clear();
  packet.kv_blob.shrink_to_fit();

  bool ok = control_queue_.Enqueue(std::move(packet));
  if (!ok) {
    // Queue full; clean up the orphaned SHM segment immediately.
    ::shm_unlink(shm_name.c_str());
  }
  return ok;
#endif
}

std::optional<KVPacket> ShmKVTransport::TryDequeue() {
  auto maybe = control_queue_.TryDequeue();
  if (!maybe)
    return std::nullopt;

#ifdef _WIN32
  return maybe; // Windows: kv_blob already inline.
#else
  KVPacket &pkt = *maybe;
  std::string shm_name = ExtractShmName(pkt.metadata);
  std::size_t blob_size = ExtractShmSize(pkt.metadata);

  if (!shm_name.empty() && blob_size > 0) {
    int fd = ::shm_open(shm_name.c_str(), O_RDWR, 0);
    if (fd >= 0) {
      void *ptr = ::mmap(nullptr, blob_size, PROT_READ, MAP_SHARED, fd, 0);
      ::close(fd);
      if (ptr != MAP_FAILED) {
        pkt.kv_blob.resize(blob_size);
        std::memcpy(pkt.kv_blob.data(), ptr, blob_size);
        ::munmap(ptr, blob_size);
      } else {
        log::Error("shm_transport",
                   std::string("mmap(read) failed: ") + std::strerror(errno));
      }
      ::shm_unlink(shm_name.c_str());
    } else {
      log::Error("shm_transport", "shm_open(read) failed for " + shm_name +
                                      ": " + std::strerror(errno));
    }
  }

  return maybe;
#endif
}

std::size_t ShmKVTransport::Capacity() const {
  return control_queue_.Capacity();
}
std::size_t ShmKVTransport::Size() const { return control_queue_.Size(); }

bool ShmKVTransport::UpdateTicketStage(uint64_t ticket_id,
                                       KVTicketStage stage) {
  return control_queue_.UpdateTicketStage(ticket_id, stage);
}

KVTicketStage ShmKVTransport::GetTicketStage(uint64_t ticket_id) const {
  return control_queue_.GetTicketStage(ticket_id);
}

} // namespace disaggregated
} // namespace inferflux
