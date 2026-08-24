#include "runtime/kv_cache/paged_kv_cache.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace inferflux {

PagedKVCache::PagedKVCache(std::size_t pages, std::size_t page_size_bytes,
                           EvictionPolicy policy)
    : page_size_bytes_(page_size_bytes), pages_(pages),
      host_blocks_(pages * 2), // Default host pool is 2x the primary cache.
      eviction_policy_(policy) {
  clock_ref_bits_.assign(pages, false);
}

int PagedKVCache::ReservePage() {
  std::lock_guard<std::mutex> lock(mutex_);
  int candidate = SelectEvictionCandidate();
  if (candidate < 0) {
    throw std::runtime_error("KV cache exhausted");
  }
  auto &page = pages_[candidate];
  page.last_used = usage_counter_++;
  page.in_use = true;
  page.ref_count = 1; // Initial reference (§P1b)
  page.dirty = false;
  if (page.data.size() != page_size_bytes_ / sizeof(float)) {
    page.data.assign(page_size_bytes_ / sizeof(float), 0.0f);
  }
  if (static_cast<std::size_t>(candidate) < clock_ref_bits_.size()) {
    clock_ref_bits_[candidate] = true;
  }
  return candidate;
}

std::vector<int> PagedKVCache::ReserveBlocks(std::size_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<int> reserved;
  reserved.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    int candidate = SelectEvictionCandidate();
    if (candidate < 0) {
      for (int id : reserved) {
        pages_[id].in_use = false;
      }
      throw std::runtime_error(
          "KV cache exhausted during multi-block allocation");
    }
    auto &page = pages_[candidate];
    page.last_used = usage_counter_++;
    page.in_use = true;
    page.ref_count = 1; // Initial reference (§P1b)
    page.dirty = false;
    if (page.data.size() != page_size_bytes_ / sizeof(float)) {
      page.data.assign(page_size_bytes_ / sizeof(float), 0.0f);
    }
    if (static_cast<std::size_t>(candidate) < clock_ref_bits_.size()) {
      clock_ref_bits_[candidate] = true;
    }
    reserved.push_back(candidate);
  }
  return reserved;
}

void PagedKVCache::ReleasePage(int page_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    return;
  }

  if (--pages_[page_id].ref_count > 0) {
    return; // Still in use by other owners (e.g. prefix cache)
  }

  if (!offload_path_.empty() && !pages_[page_id].data.empty() &&
      pages_[page_id].dirty) {
    PersistPage(page_id, pages_[page_id].data);
  }
  pages_[page_id].in_use = false;
  pages_[page_id].data.clear();
  pages_[page_id].dirty = false;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = false;
  }
}

void PagedKVCache::ReleaseBlocks(const std::vector<int> &blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (int page_id : blocks) {
    if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
      continue;
    }

    if (--pages_[page_id].ref_count > 0) {
      continue; // Still in use
    }

    if (!offload_path_.empty() && !pages_[page_id].data.empty() &&
        pages_[page_id].dirty) {
      PersistPage(page_id, pages_[page_id].data);
    }
    pages_[page_id].in_use = false;
    pages_[page_id].data.clear();
    pages_[page_id].dirty = false;
    if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
      clock_ref_bits_[page_id] = false;
    }
  }
}

void PagedKVCache::Write(int page_id, const std::vector<float> &values) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  pages_[page_id].data = values;
  pages_[page_id].dirty = true;
  pages_[page_id].last_used = usage_counter_++;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = true;
  }
  if (!offload_path_.empty()) {
    PersistPage(page_id, values);
  }
}

std::vector<float> PagedKVCache::Read(int page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  if (pages_[page_id].data.empty() && !offload_path_.empty()) {
    auto data = LoadPage(page_id);
    pages_[page_id].data = data;
  }
  pages_[page_id].last_used = usage_counter_++;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = true;
  }
  return pages_[page_id].data;
}

void PagedKVCache::SetOffloadPath(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex_);
  offload_path_ = path;
  if (!offload_path_.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(offload_path_, ec);
    if (!writer_) {
      writer_ = std::make_shared<AsyncFileWriter>(writer_queue_depth_);
    }
    writer_->Configure(writer_workers_, writer_queue_depth_);
    writer_->Start(writer_workers_);
  } else if (writer_) {
    writer_->Stop();
    writer_.reset();
  }
}

void PagedKVCache::SetEvictionPolicy(EvictionPolicy policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  eviction_policy_ = policy;
}

void PagedKVCache::ConfigureAsyncWriter(std::size_t workers,
                                        std::size_t queue_depth) {
  std::lock_guard<std::mutex> lock(mutex_);
  writer_workers_ = workers == 0 ? 1 : workers;
  writer_queue_depth_ = queue_depth == 0 ? 64 : queue_depth;
  if (writer_) {
    writer_->Configure(writer_workers_, writer_queue_depth_);
    writer_->Start(writer_workers_);
  }
}

void PagedKVCache::PersistPage(int page_id,
                               const std::vector<float> &values) const {
  if (offload_path_.empty()) {
    return;
  }
  std::filesystem::path file = std::filesystem::path(offload_path_) /
                               ("page_" + std::to_string(page_id) + ".bin");
  std::vector<char> buffer(reinterpret_cast<const char *>(values.data()),
                           reinterpret_cast<const char *>(values.data()) +
                               values.size() * sizeof(float));
  if (writer_) {
    writer_->Enqueue(AsyncWriteTask{file, std::move(buffer)});
  } else {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      return;
    }
    out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  }
}

std::vector<float> PagedKVCache::LoadPage(int page_id) const {
  std::vector<float> data;
  if (offload_path_.empty()) {
    return data;
  }
  std::filesystem::path file = std::filesystem::path(offload_path_) /
                               ("page_" + std::to_string(page_id) + ".bin");
  if (!std::filesystem::exists(file)) {
    return data;
  }
  auto size = std::filesystem::file_size(file);
  if (size % sizeof(float) != 0) {
    return data;
  }
  data.resize(size / sizeof(float));
  std::ifstream in(file, std::ios::binary);
  if (!in.good()) {
    data.clear();
    return data;
  }
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(size));
  return data;
}

int PagedKVCache::SelectEvictionCandidate() const {
  if (pages_.empty()) {
    return -1;
  }
  if (eviction_policy_ == EvictionPolicy::kClock) {
    for (std::size_t scanned = 0; scanned < pages_.size() * 2; ++scanned) {
      if (clock_hand_ >= pages_.size()) {
        clock_hand_ = 0;
      }
      std::size_t idx = clock_hand_;
      clock_hand_ = (clock_hand_ + 1) % pages_.size();
      if (pages_[idx].in_use) {
        continue;
      }
      if (!clock_ref_bits_[idx]) {
        return static_cast<int>(idx);
      }
      clock_ref_bits_[idx] = false;
    }
    for (std::size_t i = 0; i < pages_.size(); ++i) {
      if (!pages_[i].in_use) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
  std::size_t candidate = pages_.size();
  std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    if (!pages_[i].in_use && pages_[i].last_used <= oldest) {
      oldest = pages_[i].last_used;
      candidate = i;
    }
  }
  return candidate == pages_.size() ? -1 : static_cast<int>(candidate);
}

std::vector<int> PagedKVCache::SwapOut(const std::vector<int> &blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<int> host_handles;
  host_handles.reserve(blocks.size());

  for (int page_id : blocks) {
    if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
      continue;
    }

    // Find a free host block.
    int host_idx = -1;
    for (std::size_t i = 0; i < host_blocks_.size(); ++i) {
      if (!host_blocks_[i].in_use) {
        host_idx = static_cast<int>(i);
        break;
      }
    }

    if (host_idx == -1) {
      // Host RAM exhausted — this is a critical failure in swapping.
      // In a real system we'd swap to NVMe here.
      throw std::runtime_error("Host KV block pool exhausted during SwapOut");
    }

    // Migrate data from primary cache to host.
    auto &src = pages_[page_id];
    auto &dst = host_blocks_[host_idx];
    dst.data = std::move(src.data); // O(1) move.
    dst.in_use = true;
    dst.last_used = usage_counter_++;

    // Release the primary block.
    src.in_use = false;
    src.dirty = false;
    if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
      clock_ref_bits_[page_id] = false;
    }

    host_handles.push_back(host_idx);
  }
  return host_handles;
}

void PagedKVCache::SwapIn(const std::vector<int> &host_handles,
                          const std::vector<int> &target_blocks) {
  if (host_handles.size() != target_blocks.size()) {
    throw std::invalid_argument("SwapIn: handles and targets size mismatch");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (std::size_t i = 0; i < host_handles.size(); ++i) {
    int host_idx = host_handles[i];
    int page_id = target_blocks[i];

    if (host_idx < 0 ||
        static_cast<std::size_t>(host_idx) >= host_blocks_.size() ||
        page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
      continue;
    }

    auto &src = host_blocks_[host_idx];
    auto &dst = pages_[page_id];

    // Migrate data back to primary cache.
    dst.data = std::move(src.data);
    dst.in_use = true;
    dst.last_used = usage_counter_++;
    dst.dirty = true; // Mark dirty so it's persisted if offload is enabled.

    // Release host block.
    src.in_use = false;
    src.data.clear();
  }
}

std::size_t PagedKVCache::NumAvailableHostBlocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t free_count = 0;
  for (const auto &block : host_blocks_) {
    if (!block.in_use) {
      free_count++;
    }
  }
  return free_count;
}

std::size_t PagedKVCache::NumFreeBlocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t free_count = 0;
  for (const auto &page : pages_) {
    if (!page.in_use) {
      free_count++;
    }
  }
  return free_count;
}

PagedKVCache::UsageSnapshot PagedKVCache::GetUsageSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  UsageSnapshot snapshot;
  snapshot.total_blocks = pages_.size();
  snapshot.page_size_bytes = page_size_bytes_;
  for (const auto &page : pages_) {
    if (page.in_use) {
      ++snapshot.used_blocks;
    }
  }
  snapshot.free_blocks = snapshot.total_blocks - snapshot.used_blocks;
  return snapshot;
}

void PagedKVCache::AcquireBlocks(const std::vector<int> &blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (int page_id : blocks) {
    if (page_id >= 0 && static_cast<std::size_t>(page_id) < pages_.size()) {
      pages_[page_id].ref_count++;
    }
  }
}

void PagedKVCache::ReleaseBlocksRef(const std::vector<int> &blocks) {
  ReleaseBlocks(blocks);
}

} // namespace inferflux
