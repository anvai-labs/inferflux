#include <catch2/catch_amalgamated.hpp>

#include "runtime/execution/unified_batch_lane_dispatcher.h"

#include <chrono>
#include <future>
#include <mutex>
#include <thread>

namespace {

inferflux::LlamaCppBackend::UnifiedBatchInput
MakeInput(int sequence_id, int n_past, std::vector<int> tokens) {
  inferflux::LlamaCppBackend::UnifiedBatchInput input;
  input.sequence_id = sequence_id;
  input.n_past = n_past;
  input.tokens = std::move(tokens);
  input.request_logits = true;
  return input;
}

bool CollectWithTimeout(
    inferflux::UnifiedBatchLaneDispatcher *dispatcher,
    inferflux::LlamaCppBackend::UnifiedBatchHandle handle,
    std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> *outputs,
    bool *decode_lane,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    std::string error;
    auto status = dispatcher->TryCollect(handle, outputs, decode_lane, &error);
    if (status ==
        inferflux::UnifiedBatchLaneDispatcher::CollectStatus::kSuccess) {
      return true;
    }
    if (status ==
            inferflux::UnifiedBatchLaneDispatcher::CollectStatus::kFailed ||
        status ==
            inferflux::UnifiedBatchLaneDispatcher::CollectStatus::kMissing) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

} // namespace

TEST_CASE("UnifiedBatchLaneDispatcher executes lane work on persistent workers",
          "[lane_dispatcher]") {
  inferflux::UnifiedBatchLaneDispatcher dispatcher;

  std::mutex id_mutex;
  std::thread::id decode_thread_id;
  std::thread::id prefill_thread_id;

  REQUIRE(dispatcher.Start(
      [&](const std::vector<inferflux::LlamaCppBackend::UnifiedBatchInput>
              &inputs,
          bool decode_lane)
          -> inferflux::UnifiedBatchLaneDispatcher::ExecutionResult {
        std::lock_guard<std::mutex> lock(id_mutex);
        if (decode_lane) {
          decode_thread_id = std::this_thread::get_id();
        } else {
          prefill_thread_id = std::this_thread::get_id();
        }

        inferflux::LlamaCppBackend::UnifiedBatchOutput output;
        output.ok = true;
        output.token = decode_lane ? 101 : 202;
        output.piece = decode_lane ? "d" : "p";
        output.piece += std::to_string(inputs.size());
        inferflux::UnifiedBatchLaneDispatcher::ExecutionResult result;
        result.success = true;
        result.outputs.push_back(std::move(output));
        return result;
      }));

  const auto decode_handle = dispatcher.Submit({MakeInput(1, 4, {42})}, true);
  const auto prefill_handle =
      dispatcher.Submit({MakeInput(2, 0, {11, 12, 13})}, false);
  REQUIRE(decode_handle != 0);
  REQUIRE(prefill_handle != 0);

  std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> decode_outputs;
  bool decode_lane = false;
  REQUIRE(CollectWithTimeout(&dispatcher, decode_handle, &decode_outputs,
                             &decode_lane));
  REQUIRE(decode_lane);
  REQUIRE(decode_outputs.size() == 1);
  REQUIRE(decode_outputs[0].ok);
  REQUIRE(decode_outputs[0].token == 101);

  std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> prefill_outputs;
  bool prefill_lane = true;
  REQUIRE(CollectWithTimeout(&dispatcher, prefill_handle, &prefill_outputs,
                             &prefill_lane));
  REQUIRE_FALSE(prefill_lane);
  REQUIRE(prefill_outputs.size() == 1);
  REQUIRE(prefill_outputs[0].ok);
  REQUIRE(prefill_outputs[0].token == 202);

  REQUIRE(decode_thread_id != std::thread::id{});
  REQUIRE(prefill_thread_id != std::thread::id{});
  REQUIRE(decode_thread_id != prefill_thread_id);
}

TEST_CASE("UnifiedBatchLaneDispatcher enforces per-lane pending limits",
          "[lane_dispatcher]") {
  inferflux::UnifiedBatchLaneDispatcher::Config config;
  config.max_pending_per_lane = 1;
  inferflux::UnifiedBatchLaneDispatcher dispatcher(config);

  std::promise<void> release_gate_promise;
  std::shared_future<void> release_gate = release_gate_promise.get_future();
  REQUIRE(dispatcher.Start(
      [&](const std::vector<inferflux::LlamaCppBackend::UnifiedBatchInput>
              &inputs,
          bool decode_lane)
          -> inferflux::UnifiedBatchLaneDispatcher::ExecutionResult {
        (void)inputs;
        (void)decode_lane;
        release_gate.wait();
        inferflux::LlamaCppBackend::UnifiedBatchOutput output;
        output.ok = true;
        output.token = 7;
        output.piece = "ok";
        inferflux::UnifiedBatchLaneDispatcher::ExecutionResult result;
        result.success = true;
        result.outputs.push_back(std::move(output));
        return result;
      }));

  const auto decode_handle_a = dispatcher.Submit({MakeInput(1, 3, {9})}, true);
  REQUIRE(decode_handle_a != 0);
  REQUIRE(dispatcher.PendingCount(/*decode_lane=*/true) == 1);

  const auto decode_handle_b = dispatcher.Submit({MakeInput(2, 4, {10})}, true);
  REQUIRE(decode_handle_b == 0);

  const auto prefill_handle =
      dispatcher.Submit({MakeInput(3, 0, {1, 2, 3})}, false);
  REQUIRE(prefill_handle != 0);
  REQUIRE(dispatcher.PendingCount(/*decode_lane=*/false) == 1);

  release_gate_promise.set_value();

  std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> outputs_a;
  bool lane_a = false;
  REQUIRE(
      CollectWithTimeout(&dispatcher, decode_handle_a, &outputs_a, &lane_a));
  REQUIRE(lane_a);

  std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> outputs_prefill;
  bool lane_prefill = true;
  REQUIRE(CollectWithTimeout(&dispatcher, prefill_handle, &outputs_prefill,
                             &lane_prefill));
  REQUIRE_FALSE(lane_prefill);

  REQUIRE(dispatcher.PendingCount(/*decode_lane=*/true) == 0);
  REQUIRE(dispatcher.PendingCount(/*decode_lane=*/false) == 0);

  const auto decode_handle_c = dispatcher.Submit({MakeInput(4, 5, {11})}, true);
  REQUIRE(decode_handle_c != 0);

  std::vector<inferflux::LlamaCppBackend::UnifiedBatchOutput> outputs_c;
  bool lane_c = false;
  REQUIRE(
      CollectWithTimeout(&dispatcher, decode_handle_c, &outputs_c, &lane_c));
  REQUIRE(lane_c);
}
