#include "runtime/backends/cuda/native/decode_burst.h"

#include "runtime/backends/cuda/native/cuda_copy_trace.h"
#include "runtime/backends/cuda/native/cuda_sync_trace.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/model_forward.h"

#include <cuda_runtime.h>

#include <thread>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

namespace {

// Advance batch metadata for the next decode step from the sampler's device
// result buffer — no host round trip. One block per sequence; thread 0 only.
//
// Freeze semantics (load-bearing): once done[b] is set or steps_left[b]
// reaches 0, the kernel returns without touching state, so any replayed
// steps re-forward the same token at the same position: RoPE output is
// identical, KvAppendStrided overwrites the same slot byte-identically, and
// FlashDecode sees the same kv_len. Launch-ahead is therefore safe without
// cancellable graphs — already-enqueued work past a stop is wasted compute
// only, never corruption.
__global__ void DecodeTokenFeedKernel(const int *__restrict__ sampled,
                                      int *__restrict__ token_ids,
                                      int *__restrict__ n_past,
                                      int *__restrict__ kv_lens,
                                      const int *__restrict__ eos_ids,
                                      int num_eos, int *__restrict__ done,
                                      int *__restrict__ steps_left) {
  const int b = blockIdx.x;
  if (done[b] || steps_left[b] <= 0) {
    return;
  }
  const int t = sampled[b];
  for (int e = 0; e < num_eos; ++e) {
    if (t == eos_ids[e]) {
      done[b] = 1;
      return;
    }
  }
  token_ids[b] = t;
  n_past[b] += 1;
  // Match the host-side metadata invariant: kv_lens = n_past + 1 AFTER the
  // increment (the next forward appends one more entry than n_past counts).
  kv_lens[b] = n_past[b] + 1;
  steps_left[b] -= 1;
}

} // namespace

bool LaunchDecodeTokenFeed(const int *d_sampled, int *d_token_ids,
                           int *d_n_past, int *d_kv_lens, const int *d_eos_ids,
                           int num_eos, int *d_done, int *d_steps_left,
                           int batch_size, cudaStream_t stream) {
  if (batch_size <= 0) {
    return false;
  }
  DecodeTokenFeedKernel<<<batch_size, 32, 0, stream>>>(
      d_sampled, d_token_ids, d_n_past, d_kv_lens, d_eos_ids, num_eos, d_done,
      d_steps_left);
  return cudaGetLastError() == cudaSuccess;
}

DecodeBurstController::~DecodeBurstController() {
  if (ring_) {
    cudaFreeHost(ring_);
  }
  for (cudaEvent_t &e : slot_events_) {
    if (e) {
      cudaEventDestroy(e);
    }
  }
  if (d_done_) {
    cudaFree(d_done_);
  }
  if (d_steps_left_) {
    cudaFree(d_steps_left_);
  }
  if (d_eos_ids_) {
    cudaFree(d_eos_ids_);
  }
}

bool DecodeBurstController::EnsureResources(cudaStream_t stream) {
  if (resources_ready_) {
    stream_ = stream;
    return true;
  }
  stream_ = stream;
  if (cudaMallocHost(&ring_, sizeof(RingSlot) * kMaxRingSlots) != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&d_done_, sizeof(int) * kMaxBurstBatch) != cudaSuccess ||
      cudaMalloc(&d_steps_left_, sizeof(int) * kMaxBurstBatch) != cudaSuccess ||
      cudaMalloc(&d_eos_ids_, sizeof(int) * kMaxEosIds) != cudaSuccess) {
    return false;
  }
  for (int i = 0; i < kMaxRingSlots; ++i) {
    if (cudaEventCreateWithFlags(&slot_events_[i], cudaEventDisableTiming) !=
        cudaSuccess) {
      return false;
    }
  }
  resources_ready_ = true;
  return true;
}

bool DecodeBurstController::BeginChunk(const int *step_budget_host,
                                       int batch_size, const int *eos_ids_host,
                                       int num_eos) {
  if (!resources_ready_ || batch_size <= 0 || batch_size > kMaxBurstBatch) {
    return false;
  }
  active_batch_ = batch_size;
  eos_count_ = std::min(num_eos, kMaxEosIds);
  if (cudaMemsetAsync(d_done_, 0, sizeof(int) * batch_size, stream_) !=
      cudaSuccess) {
    return false;
  }
  if (cudaMemcpyAsync(d_steps_left_, step_budget_host, sizeof(int) * batch_size,
                      cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    return false;
  }
  if (eos_count_ > 0 &&
      cudaMemcpyAsync(d_eos_ids_, eos_ids_host, sizeof(int) * eos_count_,
                      cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    return false;
  }
  return true;
}

bool DecodeBurstController::EnqueueStep(int slot_idx,
                                        ModelForward *model_forward,
                                        GpuSampler *sampler, float *d_logits,
                                        int batch_size) {
  if (!model_forward->BatchForwardDevice(batch_size, d_logits)) {
    return false;
  }
  return EnqueueStepEpilogue(slot_idx, model_forward, sampler, d_logits,
                             batch_size);
}

bool DecodeBurstController::EnqueueStepEpilogue(int slot_idx,
                                                ModelForward *model_forward,
                                                GpuSampler *sampler,
                                                float *d_logits,
                                                int batch_size) {
  if (!resources_ready_ || slot_idx < 0 || slot_idx >= kMaxRingSlots) {
    return false;
  }
  if (!sampler->EnqueueBatchedArgmax(d_logits, batch_size)) {
    return false;
  }
  const ModelForward::BatchMetaDevice meta =
      model_forward->BatchMetaDevicePointers();
  if (!meta.token_ids || !meta.n_past || !meta.kv_lens) {
    return false;
  }
  if (!LaunchDecodeTokenFeed(sampler->batch_result_device(), meta.token_ids,
                             meta.n_past, meta.kv_lens, d_eos_ids_, eos_count_,
                             d_done_, d_steps_left_, batch_size, stream_)) {
    return false;
  }
  RingSlot &slot = ring_[slot_idx];
  if (TracedCudaMemcpyAsync(CopyTraceSite::kBurstSlotD2H, slot.token_ids,
                            sampler->batch_result_device(),
                            sizeof(int) * batch_size, cudaMemcpyDeviceToHost,
                            stream_) != cudaSuccess) {
    return false;
  }
  return cudaEventRecord(slot_events_[slot_idx], stream_) == cudaSuccess;
}

bool DecodeBurstController::PollSlot(int slot_idx) {
  if (!resources_ready_ || slot_idx < 0 || slot_idx >= kMaxRingSlots) {
    return false;
  }
  return cudaEventQuery(slot_events_[slot_idx]) == cudaSuccess;
}

int DecodeBurstController::ReadSlot(int slot_idx, int b) const {
  if (!resources_ready_ || slot_idx < 0 || slot_idx >= kMaxRingSlots || b < 0 ||
      b >= active_batch_) {
    return -1;
  }
  return ring_[slot_idx].token_ids[b];
}

void DecodeBurstController::FreezeSequence(int b) {
  if (!resources_ready_ || b < 0 || b >= kMaxBurstBatch) {
    return;
  }
  // Memset (not memcpy from a stack local) so the async stream op has no
  // host-lifetime dependency.
  cudaMemsetAsync(d_steps_left_ + b, 0, sizeof(int), stream_);
}

bool DecodeBurstController::FinishChunk(int last_slot_idx) {
  if (!resources_ready_ || last_slot_idx < 0 ||
      last_slot_idx >= kMaxRingSlots) {
    return false;
  }
  TracedCudaEventSynchronize(SyncTraceSite::kBurstChunkReady,
                             slot_events_[last_slot_idx]);
  return true;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
