#pragma once

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

class PagedKVCache;
class BackendInterface;

struct RadixNode {
  std::vector<int> edge;
  std::unordered_map<int, std::unique_ptr<RadixNode>> children;
  RadixNode *parent{nullptr};

  // KV block IDs assigned to this prefix segment.
  std::vector<int> block_table;
  // The sequence_id where these blocks were originally computed (§P1b).
  int sequence_id{-1};
  // The backend that owns the sequence_id.
  std::weak_ptr<BackendInterface> backend;

  // Total BPE tokens covered by the prefix at this node (including edges up to
  // here).
  int n_tokens{0};

  uint64_t last_used{0};

  int NumTokens() const { return static_cast<int>(edge.size()); }
};

struct RadixPrefixCacheLimits {
  std::size_t capacity{1024};
  std::size_t max_sequences{12};
};

struct RadixLookupResult {
  std::vector<int> block_table;
  int sequence_id{-1};
  int matched_tokens{0};
};

struct RadixPrefixMemorySnapshot {
  std::size_t unique_retained_blocks{0};
  std::size_t retained_bytes{0};
  std::size_t live_sequences{0};
};

class RadixPrefixCache {
public:
  using EvictCallback = std::function<void(int)>;

  explicit RadixPrefixCache(std::shared_ptr<PagedKVCache> kv_cache,
                            EvictCallback on_evict_seq,
                            const RadixPrefixCacheLimits &limits = {});

  // Returns true if a full node match was found (allowing CopySequencePrefix).
  // matched_tokens is always filled with the longest common prefix length
  // including partial edge matches for metrics (§ Item 3).
  bool Lookup(const std::vector<int> &tokens, BackendInterface *backend,
              RadixLookupResult *result);

  // Insert a sequence of tokens and its corresponding block_table.
  void Insert(const std::vector<int> &tokens,
              const std::vector<int> &block_table, int sequence_id,
              const std::shared_ptr<BackendInterface> &backend);

  std::size_t Capacity() const { return capacity_; }
  std::size_t Size() const; // total nodes in tree
  std::size_t LiveSequences() const;
  RadixPrefixMemorySnapshot MemorySnapshot() const;

private:
  // Walk tree following tokens; return (deepest node reached, tokens matched).
  std::pair<RadixNode *, std::size_t>
  FindLongestPrefix(const std::vector<int> &tokens) const;

  struct SplitEdgeSpec {
    int first_token{0};
    std::size_t split_at{0};
  };

  // Split the child edge of parent keyed by first_token at split_at tokens in.
  void SplitEdge(RadixNode *parent, const SplitEdgeSpec &spec);

  // Evict the leaf node with the smallest last_used timestamp.
  void EvictOne();
  // Evict the sequence-holding node with the smallest last_used timestamp (§
  // Item 1).
  void EvictOneSequence();

  // DFS: collect nodes matching a criteria.
  void CollectNodes(RadixNode *node,
                    const std::function<bool(const RadixNode *)> &criteria,
                    std::vector<std::pair<uint64_t, RadixNode *>> &out) const;

  std::shared_ptr<PagedKVCache> kv_cache_;
  EvictCallback on_evict_seq_;
  std::size_t capacity_;
  std::size_t max_sequences_;
  std::size_t size_{0};
  std::size_t live_sequences_{0};
  uint64_t clock_{0};
  std::unique_ptr<RadixNode> root_;
  mutable std::shared_mutex mutex_;
};

} // namespace inferflux
