#pragma once

#include "scheduler/request_batch.h"

#include <chrono>

namespace inferflux {

inline void
PrepareDecodeRequeue(InferenceRequest *inference,
                     std::chrono::steady_clock::time_point enqueue_time) {
  if (!inference) {
    return;
  }
  inference->enqueue_time = enqueue_time;
  inference->phase = RequestPhase::kDecode;
  // Resumed decode keeps its original prompt/tokenization. Continuity is
  // carried by sequence state, n_past, first_token, and accumulated_output.
}

inline void PrepareFairnessDecodeRequeue(
    InferenceRequest *inference,
    std::chrono::steady_clock::time_point enqueue_time) {
  PrepareDecodeRequeue(inference, enqueue_time);
}

} // namespace inferflux
