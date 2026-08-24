#pragma once

// §2.6 Expert Parallelism (EP) dispatch interface — stub.
//
// This header defines the interface that an EP-aware router would implement to
// partition MoE expert layers across multiple devices or processes.  The full
// implementation requires multi-GPU topology discovery and a mechanism to split
// llama.cpp expert layers by rank (e.g., via NCCL or a shared-memory ring).
//
// Current status: stub.  The MoE metadata (ModelInfo::is_moe, n_experts,
// n_active_experts) is now populated by SingleModelRouter from GGUF metadata,
// and RecordMoERequest() increments the Prometheus counter on each MoE
// dispatch.  Actual expert partitioning is tracked in TechDebt §2.6.

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

class LlamaCppBackend;

// A rank in an EP group: identifies which experts [expert_start, expert_end)
// this worker is responsible for.
struct EPRank {
  int rank{0};
  int world_size{1};
  int expert_start{0}; // inclusive
  int expert_end{0};   // exclusive
};

// EPDispatch routes MoE forward passes across an EP group.
// Full implementation shards expert layers and performs all-to-all
// communication (§P1f).
class EPDispatch {
public:
  virtual ~EPDispatch() = default;

  // Returns the local rank configuration for this worker.
  virtual EPRank LocalRank() const = 0;

  // Returns true if this worker owns expert `expert_id`.
  virtual bool OwnsExpert(int expert_id) const = 0;

  // Routes hidden states to the ranks owning the required experts and
  // gathers the results back (§P1f).
  // `hidden_states` are the inputs to the MoE layer.
  // `expert_indices` are the indices of the experts to be invoked for each
  // token.
  virtual std::vector<float> Route(const std::vector<float> &hidden_states,
                                   const std::vector<int> &expert_indices) = 0;

  // Stub: returns the name of this dispatch strategy (e.g., "local", "dist").
  virtual std::string Name() const = 0;
};

// LocalEPDispatch: single-process stub — owns all experts, world_size=1.
class LocalEPDispatch : public EPDispatch {
public:
  explicit LocalEPDispatch(int n_experts) : rank_{0, 1, 0, n_experts} {}

  EPRank LocalRank() const override { return rank_; }
  bool OwnsExpert(int expert_id) const override {
    return expert_id >= rank_.expert_start && expert_id < rank_.expert_end;
  }

  std::vector<float> Route(const std::vector<float> &hidden_states,
                           const std::vector<int> &) override {
    // In local mode, just return the states (expert computation is local).
    return hidden_states;
  }

  std::string Name() const override { return "local"; }

private:
  EPRank rank_;
};

// DistributedEPDispatch: routes experts across multiple ranks using CommBackend
// (§P1f).
class DistributedEPDispatch : public EPDispatch {
public:
  DistributedEPDispatch(int rank, int world_size, int n_experts);

  EPRank LocalRank() const override { return rank_; }
  bool OwnsExpert(int expert_id) const override {
    return expert_id >= rank_.expert_start && expert_id < rank_.expert_end;
  }

  std::vector<float> Route(const std::vector<float> &hidden_states,
                           const std::vector<int> &expert_indices) override;

  std::string Name() const override { return "distributed"; }

private:
  EPRank rank_;
};

} // namespace inferflux
