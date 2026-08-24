#pragma once

#include "io/async_file_writer.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <memory>
#include <vector>

namespace inferflux {

struct KVPage {
  std::vector<float> data;
  bool in_use{false};
  int ref_count{0}; // Reference count (§P1b)
  std::uint64_t last_used{0};
  bool dirty{false};
};

class PagedKVCache {
public:
  enum class EvictionPolicy { kLRU, kClock };

  struct UsageSnapshot {
    std::size_t total_blocks{0};
    std::size_t used_blocks{0};
    std::size_t free_blocks{0};
    std::size_t page_size_bytes{0};

    std::size_t total_bytes() const { return total_blocks * page_size_bytes; }
    std::size_t used_bytes() const { return used_blocks * page_size_bytes; }
    std::size_t free_bytes() const { return free_blocks * page_size_bytes; }
  };

  explicit PagedKVCache(std::size_t pages, std::size_t page_size_bytes,
                        EvictionPolicy policy = EvictionPolicy::kLRU);

  int ReservePage();
  std::vector<int> ReserveBlocks(std::size_t n);
  void ReleasePage(int page_id);
  void ReleaseBlocks(const std::vector<int> &blocks);

  // Reference counting for KV blocks (§P1b).
  void AcquireBlocks(const std::vector<int> &blocks);
  void ReleaseBlocksRef(const std::vector<int> &blocks);

  // Swap blocks to/from Host RAM (§P1c).
  // Returns a vector of 'host_handle' IDs representing the blocks in Host RAM.
  std::vector<int> SwapOut(const std::vector<int> &blocks);
  // Restores blocks from Host RAM handles back to physical cache blocks.
  void SwapIn(const std::vector<int> &host_handles,
              const std::vector<int> &target_blocks);

  void Write(int page_id, const std::vector<float> &values);
  std::vector<float> Read(int page_id) const;

  std::size_t NumFreeBlocks() const;
  std::size_t NumAvailableHostBlocks() const;
  UsageSnapshot GetUsageSnapshot() const;
  void SetOffloadPath(const std::string &path);
  void SetEvictionPolicy(EvictionPolicy policy);
  void ConfigureAsyncWriter(std::size_t workers, std::size_t queue_depth);

  std::size_t PageSizeBytes() const { return page_size_bytes_; }

private:
  std::size_t page_size_bytes_;
  mutable std::vector<KVPage> pages_;
  // Secondary storage tier in Host RAM for swapped blocks (§P1c).
  mutable std::vector<KVPage> host_blocks_;
  mutable std::mutex mutex_;
  std::string offload_path_;
  std::shared_ptr<AsyncFileWriter> writer_;
  mutable std::uint64_t usage_counter_{0};
  EvictionPolicy eviction_policy_;
  mutable std::vector<bool> clock_ref_bits_;
  mutable std::size_t clock_hand_{0};
  std::size_t writer_workers_{1};
  std::size_t writer_queue_depth_{256};

  void PersistPage(int page_id, const std::vector<float> &values) const;
  std::vector<float> LoadPage(int page_id) const;
  int SelectEvictionCandidate() const;
};

} // namespace inferflux
