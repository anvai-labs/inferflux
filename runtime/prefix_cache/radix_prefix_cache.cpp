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
  int last_seq_tokens = 0;
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
    if (node->sequence_id >= 0) {
      last_seq_id = node->sequence_id;
      last_seq_tokens = static_cast<int>(offset);
      // Each donor owns a complete sequence table. Ancestors may reference
      // different physical prefixes; concatenating tables duplicates blocks.
      blocks = node->block_table;
    }
    // Relaxed atomics: Lookup runs under the SHARED lock, so the touch must
    // not tear (previously ++clock_ and last_used were plain uint64_t — a
    // data race with any second Lookup caller). Approximate ordering is all
    // LRU eviction needs.
    node->last_used.store(clock_.fetch_add(1, std::memory_order_relaxed) + 1,
                          std::memory_order_relaxed);
  }

  if (result) {
    result->matched_tokens = static_cast<int>(offset);
  }

  if (node_hit && node != root_.get() && last_seq_id >= 0 && !blocks.empty()) {
    if (result) {
      result->block_table = std::move(blocks);
      result->sequence_id = last_seq_id;
      result->matched_tokens = last_seq_tokens;
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

bool RadixPrefixCache::Insert(
    const std::vector<int> &tokens, const std::vector<int> &block_table,
    int sequence_id, const std::shared_ptr<BackendInterface> &backend) {
  if (capacity_ == 0 || tokens.empty() || block_table.empty()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  RadixNode *node = root_.get();
  std::size_t offset = 0;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);

    if (it == node->children.end()) {
      auto leaf = std::make_unique<RadixNode>();
      leaf->parent = node;
      leaf->edge.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                        tokens.end());

      // The scheduler transfers one reference for EVERY donated block.
      // Keeping only suffix blocks loses ownership of the prefix references
      // when an edge splits, eventually exhausting the paged cache.
      leaf->block_table = block_table;

      leaf->sequence_id = sequence_id;
      leaf->backend = backend;
      if (sequence_id >= 0)
        live_sequences_++;

      leaf->last_used.store(clock_.fetch_add(1, std::memory_order_relaxed) + 1,
                            std::memory_order_relaxed);
      node->children[first] = std::move(leaf);
      size_++;

      while (live_sequences_ > max_sequences_) {
        EvictOneSequenceLocked();
      }
      while (size_ > capacity_) {
        EvictOne();
      }
      return true;
    }

    RadixNode *child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      SplitEdge(node, {first, common});
      child = node->children[first].get();
    }

    offset += common;
    node = child;
  }

  // Live-sequence accounting for the (re)donation happens at the assignment
  // below, together with old-donor cleanup.

  if (kv_cache_ && !node->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(node->block_table);
  }
  node->block_table = block_table;

  // Re-donation overwrites the previous donor's claim on this node. Free the
  // old donor's backend KV and release its slot first — otherwise the old
  // slot parks occupied forever, referenced by no trie node (issue #161
  // review, blocker 2).
  const int old_donor = node->sequence_id;
  if (old_donor >= 0 && old_donor != sequence_id) {
    auto old_be = node->backend.lock();
    if (old_be) {
      old_be->FreeSequence(old_donor);
    }
    if (on_evict_seq_) {
      on_evict_seq_(old_donor, old_be);
    }
    live_sequences_--;
  }

  node->sequence_id = sequence_id;
  node->backend = backend;
  if (sequence_id >= 0 && sequence_id != old_donor) {
    live_sequences_++;
  }
  node->last_used.store(clock_.fetch_add(1, std::memory_order_relaxed) + 1,
                        std::memory_order_relaxed);
  return true;
}

void RadixPrefixCache::CollectNodes(
    RadixNode *node, const std::function<bool(const RadixNode *)> &criteria,
    std::vector<std::pair<uint64_t, RadixNode *>> &out) const {
  if (criteria(node)) {
    out.emplace_back(node->last_used.load(std::memory_order_relaxed), node);
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
    auto locked_be = victim->backend.lock();
    if (locked_be) {
      locked_be->FreeSequence(victim->sequence_id);
    }
    if (on_evict_seq_) {
      on_evict_seq_(victim->sequence_id, locked_be);
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

bool RadixPrefixCache::EvictOneSequence() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return EvictOneSequenceLocked();
}

// Must hold mutex_ exclusively.
bool RadixPrefixCache::EvictOneSequenceLocked() {
  std::vector<std::pair<uint64_t, RadixNode *>> seq_nodes;
  CollectNodes(
      root_.get(), [](const RadixNode *n) { return n->sequence_id >= 0; },
      seq_nodes);

  if (seq_nodes.empty())
    return false;

  auto victim_it = std::min_element(
      seq_nodes.begin(), seq_nodes.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  RadixNode *victim = victim_it->second;
  auto locked_be = victim->backend.lock();
  if (locked_be) {
    locked_be->FreeSequence(victim->sequence_id);
  }
  if (on_evict_seq_) {
    on_evict_seq_(victim->sequence_id, locked_be);
  }

  // Correctness Fix (§ Item 2): release block references back to the cache!
  if (kv_cache_ && !victim->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(victim->block_table);
  }
  victim->block_table.clear();

  victim->sequence_id = -1;
  live_sequences_--;

  // Prune childless husks so Lookup does not walk (and overclaim matched
  // tokens across) evicted edges. Internal nodes stay: their edges route to
  // live descendants.
  if (victim->children.empty() && victim->parent != nullptr) {
    int first_token = victim->edge[0];
    victim->parent->children.erase(first_token);
    size_--;
  }
  return true;
}

} // namespace inferflux
