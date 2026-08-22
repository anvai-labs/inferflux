#pragma once

#include "runtime/backends/common/backend_types.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"

#include <cstddef>

// CUDA headers when available; opaque typedefs otherwise so this header
// parses in CPU-only builds (the controller member itself only exists under
// INFERFLUX_NATIVE_KERNELS_READY).
#if defined(INFERFLUX_HAS_CUDA) ||                                             \
    (defined(__has_include) && __has_include(<cuda_runtime.h>))
#include <cuda_runtime.h>
#else
struct cudaStream_t__;
typedef cudaStream_t__ *cudaStream_t;
struct CUevent_st;
typedef CUevent_st *cudaEvent_t;
#endif

namespace inferflux {
class ModelForward;
class GpuSampler;

namespace runtime {
namespace cuda {
namespace native {

// DecodeBurstController owns the device-side state for burst-pipelined
// decode: the token-feed kernel's scratch (done / steps_left / eos ids), the
// pinned host ring of sampled tokens, and one CUDA event per ring slot.
//
// The executor drives it:
//   BeginChunk()          — (re)initialize per-sequence step budgets
//   EnqueueStep(i)        — forward + argmax + token feed + slot D2H + event
//   PollSlot(i)           — non-blocking check for slot i readiness
//   ReadSlot(i, b)        — token for slot i, sequence b (host pinned read)
//   FreezeSequence(b)     — stop feeding sequence b from the next chunk on
//   FinishChunk()         — traced chunk-boundary drain
//
// The freeze-idempotence property is what makes launch-ahead safe: once
// done[b] is set (or steps_left[b] hits 0), the feed kernel stops updating
// token_ids[b]/n_past[b], so any already-enqueued replays re-forward the SAME
// token at the SAME position — RoPE output is identical, KvAppend overwrites
// the same slot byte-identically, FlashDecode sees the same kv_len. Wasted
// compute only; no state corruption. Sequences are always re-initialized via
// the full metadata upload at BeginChunk(), so stale device state from a
// previous burst can never leak into a new one.
class DecodeBurstController {
public:
  static constexpr int kMaxRingSlots = 32;
  static constexpr int kMaxBurstBatch = 64; // matches GpuSampler batch cap
  static constexpr int kMaxEosIds = 8;

  DecodeBurstController() = default;
  ~DecodeBurstController();
  DecodeBurstController(const DecodeBurstController &) = delete;
  DecodeBurstController &operator=(const DecodeBurstController &) = delete;

  // Lazily allocate ring/events/device scratch. Idempotent; returns false on
  // CUDA allocation failure (caller falls back to per-step decode).
  bool EnsureResources(cudaStream_t stream);

  // Reset per-sequence budgets and upload EOS ids. step_budget[b] is the
  // number of decode steps sequence b may still take this burst (already
  // clamped by the caller against decode limits and KV headroom).
  bool BeginChunk(const int *step_budget_host, int batch_size,
                  const int *eos_ids_host, int num_eos);

  // Enqueue one decode step into ring slot slot_idx: device-fed forward
  // (metadata already current on device), batched argmax, token feed, D2H
  // copy of the sampled ids, and the slot event. All async.
  bool EnqueueStep(int slot_idx, ModelForward *model_forward,
                   GpuSampler *sampler, float *d_logits, int batch_size);

  // Enqueue only the post-forward epilogue for slot_idx (argmax + token
  // feed + slot D2H + event). Used for the first step of a chunk, whose
  // forward is driven by the host-side BatchForward (metadata upload +
  // graph capture).
  bool EnqueueStepEpilogue(int slot_idx, ModelForward *model_forward,
                           GpuSampler *sampler, float *d_logits,
                           int batch_size);

  // Non-blocking readiness check for a slot enqueued by EnqueueStep.
  bool PollSlot(int slot_idx);

  // Sampled token for slot slot_idx, sequence b. Only valid after PollSlot.
  int ReadSlot(int slot_idx, int b) const;

  // Mark sequence b finished from the host side (stop string, decode limit,
  // cancellation). Takes effect at the next BeginChunk(); within the current
  // chunk the host simply ignores that sequence's slots.
  void FreezeSequence(int b);

  // Drain the last enqueued slot (traced as burst.chunk_ready). Call once at
  // the end of the burst before reading the final slot.
  bool FinishChunk(int last_slot_idx);

  int max_slots() const { return kMaxRingSlots; }

private:
  bool resources_ready_{false};
  cudaStream_t stream_{nullptr};

  // Pinned ring: slot i holds the sampled token ids for the i-th step of the
  // current chunk. 64 ids + a done mask per slot.
  struct RingSlot {
    int token_ids[kMaxBurstBatch];
    unsigned int done_mask;
  };
  RingSlot *ring_{nullptr};
  cudaEvent_t slot_events_[kMaxRingSlots]{};

  // Device scratch for the feed kernel.
  int *d_done_{nullptr};       // [batch] sequence finished flags
  int *d_steps_left_{nullptr}; // [batch] remaining step budget
  int *d_eos_ids_{nullptr};    // [kMaxEosIds] terminal token ids
  int eos_count_{0};
  int active_batch_{0};
};

// Launch the token-feed kernel: reads the sampler's device result buffer and
// advances the batch metadata (token id, n_past, kv_len) for the next decode
// step without a host round trip. Exposed for unit testing.
bool LaunchDecodeTokenFeed(const int *d_sampled, int *d_token_ids,
                           int *d_n_past, int *d_kv_lens, const int *d_eos_ids,
                           int num_eos, int *d_done, int *d_steps_left,
                           int batch_size, cudaStream_t stream);

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
