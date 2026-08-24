#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/backends/common/backend_interface.h"
#include "runtime/kv_cache/paged_kv_cache.h"

#include <algorithm>
#include <cassert>
#include <unordered_set>

namespace inferflux {

namespace {

// Return the length of the longest common prefix between two token spans.
std::size_t CommonPrefixLength(const std::vector<int> &edge,
                               const std::vector<int> &tokens,
                               std::size_t tokens_offset) {
  std::size_t len = 0;
  std::size_t edge_len = edge.size();
  std::size_t tokens_remaining = tokens.size() - tokens_offset;
  std::size_t limit = std::min(edge_len, tokens_remaining);
  while (len < limit && edge[len] == tokens[tokens_offset + len]) {
    ++len;
  }
  return len;
}

} // namespace

RadixPrefixCache::RadixPrefixCache(std::shared_ptr<PagedKVCache> kv_cache,
                                   EvictCallback on_evict_seq,
                                   const RadixPrefixCacheLimits &limits)
    : kv_cache_(std::move(kv_cache)), on_evict_seq_(std::move(on_evict_seq)),
      capacity_(limits.capacity), max_sequences_(limits.max_sequences),
      root_(std::make_unique<RadixNode>()) {}

std::size_t RadixPrefixCache::Size() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return size_;
}

std::size_t RadixPrefixCache::LiveSequences() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return live_sequences_;
}

RadixPrefixMemorySnapshot RadixPrefixCache::MemorySnapshot() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::pair<uint64_t, RadixNode *>> seq_nodes;
  CollectNodes(
      root_.get(), [](const RadixNode *n) { return n->sequence_id >= 0; },
      seq_nodes);

  std::unordered_set<int> blocks;
  for (const auto &[_, node] : seq_nodes) {
    for (int block_id : node->block_table) {
      if (block_id >= 0) {
        blocks.insert(block_id);
      }
    }
  }

  RadixPrefixMemorySnapshot snapshot;
  snapshot.unique_retained_blocks = blocks.size();
  snapshot.live_sequences = live_sequences_;
  if (kv_cache_) {
    snapshot.retained_bytes =
        snapshot.unique_retained_blocks * kv_cache_->PageSizeBytes();
  }
  return snapshot;
}

bool RadixPrefixCache::Lookup(const std::vector<int> &tokens,
                              BackendInterface *backend,
                              RadixLookupResult *result) {
  if (result) {
    result->block_table.clear();
    result->sequence_id = -1;
    result->matched_tokens = 0;
  }
  if (capacity_ == 0 || tokens.empty()) {
    return false;
  }

  std::shared_lock<std::shared_mutex> lock(mutex_);
  RadixNode *node = root_.get();
  std::size_t offset = 0;
  std::vector<int> blocks;
  int last_seq_id = -1;
  bool node_hit = false;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);
    if (it == node->children.end()) {
      break;
    }
    RadixNode *child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      // Mid-edge divergence (§ Item 3): record partial match length but hit is
      // false.
      offset += common;
      node_hit = false;
      break;
    }

    // Backend validation logic for full node reuse.
    if (child->sequence_id >= 0) {
      auto locked_be = child->backend.lock();
      if (locked_be.get() != backend) {
        // Mismatch — we still count these tokens but can't reuse the node.
        offset += common;
        node_hit = false;
        break;
      }
    }

    offset += common;
    node = child;
    node_hit = true;
    if (!node->block_table.empty()) {
      blocks.insert(blocks.end(), node->block_table.begin(),
                    node->block_table.end());
    }
    if (node->sequence_id >= 0) {
      last_seq_id = node->sequence_id;
    }
    node->last_used = ++clock_;
  }

  if (result) {
    result->matched_tokens = static_cast<int>(offset);
  }

  if (node_hit && node != root_.get()) {
    if (result) {
      result->block_table = std::move(blocks);
      result->sequence_id = last_seq_id;
    }
    return true;
  }

  return false;
}

void RadixPrefixCache::SplitEdge(RadixNode *parent, const SplitEdgeSpec &spec) {
  auto original = std::move(parent->children[spec.first_token]);

  auto intermediate = std::make_unique<RadixNode>();
  intermediate->parent = parent;
  intermediate->edge.assign(original->edge.begin(),
                            original->edge.begin() +
                                static_cast<std::ptrdiff_t>(spec.split_at));

  intermediate->block_table = {};
  intermediate->sequence_id = -1;

  int original_first = original->edge[spec.split_at];
  original->parent = intermediate.get();
  original->edge.erase(original->edge.begin(),
                       original->edge.begin() +
                           static_cast<std::ptrdiff_t>(spec.split_at));

  intermediate->children[original_first] = std::move(original);
  parent->children[spec.first_token] = std::move(intermediate);
  size_++;
}

void RadixPrefixCache::Insert(
    const std::vector<int> &tokens, const std::vector<int> &block_table,
    int sequence_id, const std::shared_ptr<BackendInterface> &backend) {
  if (capacity_ == 0 || tokens.empty() || block_table.empty()) {
    return;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  RadixNode *node = root_.get();
  std::size_t offset = 0;
  std::size_t node_start_offset = 0;
  const int kTokensPerBlock = 16;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);

    if (it == node->children.end()) {
      auto leaf = std::make_unique<RadixNode>();
      leaf->parent = node;
      leaf->edge.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                        tokens.end());

      std::size_t block_offset = offset / kTokensPerBlock;
      if (block_offset < block_table.size()) {
        leaf->block_table.assign(block_table.begin() +
                                     static_cast<std::ptrdiff_t>(block_offset),
                                 block_table.end());
      }

      leaf->sequence_id = sequence_id;
      leaf->backend = backend;
      if (sequence_id >= 0)
        live_sequences_++;

      leaf->last_used = ++clock_;
      node->children[first] = std::move(leaf);
      size_++;

      while (live_sequences_ > max_sequences_) {
        EvictOneSequence();
      }
      while (size_ > capacity_) {
        EvictOne();
      }
      return;
    }

    RadixNode *child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      SplitEdge(node, {first, common});
      child = node->children[first].get();
    }

    node_start_offset = offset;
    offset += common;
    node = child;
  }

  if (node->sequence_id < 0 && sequence_id >= 0)
    live_sequences_++;

  // Update the block table for this node only (prefix blocks are owned by
  // ancestor nodes and are concatenated during lookup).
  std::vector<int> node_blocks;
  std::size_t block_offset = node_start_offset / kTokensPerBlock;
  if (block_offset < block_table.size()) {
    node_blocks.assign(block_table.begin() +
                           static_cast<std::ptrdiff_t>(block_offset),
                       block_table.end());
  }

  if (kv_cache_ && !node->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(node->block_table);
  }
  node->block_table = std::move(node_blocks);

  node->sequence_id = sequence_id;
  node->backend = backend;
  node->last_used = ++clock_;
}

void RadixPrefixCache::CollectNodes(
    RadixNode *node, const std::function<bool(const RadixNode *)> &criteria,
    std::vector<std::pair<uint64_t, RadixNode *>> &out) const {
  if (criteria(node)) {
    out.emplace_back(node->last_used, node);
  }
  for (const auto &[key, child] : node->children) {
    CollectNodes(child.get(), criteria, out);
  }
}

void RadixPrefixCache::EvictOne() {
  std::vector<std::pair<uint64_t, RadixNode *>> leaves;
  leaves.reserve(size_);
  CollectNodes(
      root_.get(),
      [](const RadixNode *n) {
        return n->children.empty() && n->parent != nullptr;
      },
      leaves);

  if (leaves.empty())
    return;

  auto victim_it = std::min_element(
      leaves.begin(), leaves.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  RadixNode *victim = victim_it->second;
  if (victim->sequence_id >= 0) {
    if (auto locked_be = victim->backend.lock()) {
      locked_be->FreeSequence(victim->sequence_id);
    }
    if (on_evict_seq_) {
      on_evict_seq_(victim->sequence_id);
    }
    live_sequences_--;
  }

  if (kv_cache_ && !victim->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(victim->block_table);
  }

  // Prune from trie.
  int first_token = victim->edge[0];
  victim->parent->children.erase(first_token);
  size_--;
}

void RadixPrefixCache::EvictOneSequence() {
  std::vector<std::pair<uint64_t, RadixNode *>> seq_nodes;
  CollectNodes(
      root_.get(), [](const RadixNode *n) { return n->sequence_id >= 0; },
      seq_nodes);

  if (seq_nodes.empty())
    return;

  auto victim_it = std::min_element(
      seq_nodes.begin(), seq_nodes.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  RadixNode *victim = victim_it->second;
  if (auto locked_be = victim->backend.lock()) {
    locked_be->FreeSequence(victim->sequence_id);
  }
  if (on_evict_seq_) {
    on_evict_seq_(victim->sequence_id);
  }

  // Correctness Fix (§ Item 2): release block references back to the cache!
  if (kv_cache_ && !victim->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(victim->block_table);
  }
  victim->block_table.clear();

  victim->sequence_id = -1;
  live_sequences_--;
}

} // namespace inferflux
