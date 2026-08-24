#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/disaggregated/kv_channel.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/request_requeue.h"
// MSVC includes access specifiers in name mangling, so the
// "#define private public" trick causes linker errors on MSVC.
// Tests that access private members are skipped on MSVC.
#ifndef _MSC_VER
#define private public
#endif
#include "scheduler/scheduler.h"
#ifndef _MSC_VER
#undef private
#endif
#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

using namespace inferflux;

namespace {

int64_t ReadBatchTokenBudgetSkipsTotal() {
  const std::string output = GlobalMetrics().RenderPrometheus();
  const std::string key =
      "inferflux_scheduler_batch_token_budget_skips_total{backend=\"cpu\"} ";
  auto pos = output.find(key);
  if (pos == std::string::npos) {
    return 0;
  }
  pos += key.size();
  auto line_end = output.find('\n', pos);
  const std::string value = output.substr(
      pos, line_end == std::string::npos ? std::string::npos : line_end - pos);
  try {
    return std::stoll(value);
  } catch (const std::exception &) {
    return 0;
  }
}

int64_t ReadDisaggTicketStageTotal(const std::string &stage) {
  const std::string output = GlobalMetrics().RenderPrometheus();
  const std::string key = "inferflux_disagg_kv_tickets_total{backend=\"cpu\","
                          "stage=\"" +
                          stage + "\"} ";
  auto pos = output.find(key);
  if (pos == std::string::npos) {
    return 0;
  }
  pos += key.size();
  auto line_end = output.find('\n', pos);
  const std::string value = output.substr(
      pos, line_end == std::string::npos ? std::string::npos : line_end - pos);
  try {
    return std::stoll(value);
  } catch (const std::exception &) {
    return 0;
  }
}

int64_t ReadScalarMetric(const std::string &key) {
  const std::string output = GlobalMetrics().RenderPrometheus();
  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind(key, 0) != 0) {
      continue;
    }
    const std::string value = line.substr(key.size());
    try {
      return std::stoll(value);
    } catch (const std::exception &) {
      return 0;
    }
  }
  return 0;
}

bool WaitForCondition(
    const std::function<bool()> &predicate,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

class ReadyStubBackend : public LlamaCppBackend {
public:
  explicit ReadyStubBackend(std::string output) : output_(std::move(output)) {}

  bool LoadModel(const std::filesystem::path &,
                 const LlamaBackendConfig &) override {
    return true;
  }

  bool IsReady() const override { return true; }
  bool SupportsSplitPrefillDecodeHandoff() const override { return true; }

  PrefillResult Prefill(const std::string &, int) override {
    PrefillResult out;
    out.n_past = 1;
    out.ok = true;
    return out;
  }

  PrefillResult PrefillPartial(const std::string &, int,
                               int n_past_start) override {
    PrefillResult out;
    out.n_past = n_past_start + 1;
    out.ok = true;
    return out;
  }

  std::string Decode(int, int, int,
                     const std::function<bool(const std::string &,
                                              const TokenLogprob *)> &on_chunk,
                     const std::function<bool()> &, int,
                     std::vector<TokenLogprob> *out_logprobs, int,
                     const std::vector<std::string> &) override {
    if (out_logprobs) {
      TokenLogprob lp;
      lp.token = output_;
      lp.logprob = -0.1f;
      out_logprobs->push_back(lp);
    }
    if (on_chunk && !on_chunk(output_, nullptr)) {
      return {};
    }
    return output_;
  }

  std::string
  Generate(const std::string &, int,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk,
           const std::function<bool()> &, int,
           std::vector<TokenLogprob> *out_logprobs,
           const std::vector<std::string> &) override {
    if (out_logprobs) {
      TokenLogprob lp;
      lp.token = output_;
      lp.logprob = -0.1f;
      out_logprobs->push_back(lp);
    }
    if (on_chunk && !on_chunk(output_, nullptr)) {
      return {};
    }
    return output_;
  }

  void FreeSequence(int) override {}

  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }

  std::vector<int> TokenizeForCache(const std::string &) const override {
    return {1, 2, 3};
  }

private:
  std::string output_;
};

class ProcessLocalSplitStubBackend final : public ReadyStubBackend {
public:
  explicit ProcessLocalSplitStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  bool SupportsProcessLocalSequenceTransfer() const override { return true; }

  std::vector<uint8_t> SerializeSequence(int sequence_id) override {
    ++serialize_calls_;
    return {static_cast<uint8_t>(sequence_id & 0xff), 0x42};
  }

  bool HydrateSequence(int, const std::vector<uint8_t> &blob) override {
    ++hydrate_calls_;
    return !blob.empty();
  }

  int SerializeCalls() const { return serialize_calls_.load(); }
  int HydrateCalls() const { return hydrate_calls_.load(); }

private:
  std::atomic<int> serialize_calls_{0};
  std::atomic<int> hydrate_calls_{0};
};

class AsyncLaneStubBackend final : public ReadyStubBackend {
public:
  explicit AsyncLaneStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  bool SupportsAsyncUnifiedBatch() const override { return true; }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    const int submission_ticket =
        global_submission_ticket_.fetch_add(1, std::memory_order_relaxed) + 1;
    int expected_ticket = 0;
    first_submission_ticket_.compare_exchange_strong(
        expected_ticket, submission_ticket, std::memory_order_relaxed,
        std::memory_order_relaxed);

    if (lane == UnifiedBatchLane::kPrefill) {
      prefill_submissions_.fetch_add(1, std::memory_order_relaxed);
    } else if (lane == UnifiedBatchLane::kDecode) {
      decode_submissions_.fetch_add(1, std::memory_order_relaxed);
    } else {
      auto_submissions_.fetch_add(1, std::memory_order_relaxed);
    }

    const auto handle =
        next_handle_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_results_[handle] = BuildOutputs(inputs, lane);
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    if (!outputs || handle == 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(async_mutex_);
    auto it = async_results_.find(handle);
    if (it == async_results_.end()) {
      return false;
    }
    *outputs = std::move(it->second);
    async_results_.erase(it);
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    return BuildOutputs(inputs, UnifiedBatchLane::kAuto);
  }

  PrefillResult Prefill(const std::string &prompt, int sequence_id) override {
    prefill_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    return ReadyStubBackend::Prefill(prompt, sequence_id);
  }

  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override {
    prefill_partial_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    return ReadyStubBackend::PrefillPartial(prompt, sequence_id, n_past_start);
  }

  std::vector<int> TokenizeForCache(const std::string &) const override {
    return {1, 2, 3, 4};
  }

  int PrefillSubmissions() const {
    return prefill_submissions_.load(std::memory_order_relaxed);
  }
  int DecodeSubmissions() const {
    return decode_submissions_.load(std::memory_order_relaxed);
  }
  int PrefillFallbackCalls() const {
    return prefill_fallback_calls_.load(std::memory_order_relaxed);
  }
  int PrefillPartialFallbackCalls() const {
    return prefill_partial_fallback_calls_.load(std::memory_order_relaxed);
  }
  int FirstSubmissionTicket() const {
    return first_submission_ticket_.load(std::memory_order_relaxed);
  }
  std::vector<int> PrefillBatchSizes() const {
    std::lock_guard<std::mutex> lock(batch_size_mutex_);
    return prefill_batch_sizes_;
  }
  std::vector<int> DecodeBatchSizes() const {
    std::lock_guard<std::mutex> lock(batch_size_mutex_);
    return decode_batch_sizes_;
  }

private:
  std::vector<UnifiedBatchOutput>
  BuildOutputs(const std::vector<UnifiedBatchInput> &inputs,
               UnifiedBatchLane lane) const {
    {
      std::lock_guard<std::mutex> lock(batch_size_mutex_);
      if (lane == UnifiedBatchLane::kPrefill) {
        prefill_batch_sizes_.push_back(static_cast<int>(inputs.size()));
      } else if (lane == UnifiedBatchLane::kDecode) {
        decode_batch_sizes_.push_back(static_cast<int>(inputs.size()));
      } else {
        auto_batch_sizes_.push_back(static_cast<int>(inputs.size()));
      }
    }
    std::vector<UnifiedBatchOutput> outputs(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      outputs[i].ok = true;
      if (!inputs[i].request_logits) {
        continue;
      }
      const bool prefill_lane =
          lane == UnifiedBatchLane::kPrefill || inputs[i].tokens.size() > 1;
      outputs[i].token = prefill_lane ? 100 : 101;
      outputs[i].piece = prefill_lane ? "p" : "d";
    }
    return outputs;
  }

  std::atomic<uint64_t> next_handle_{0};
  mutable std::mutex async_mutex_;
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      async_results_;
  std::atomic<int> prefill_submissions_{0};
  std::atomic<int> decode_submissions_{0};
  std::atomic<int> auto_submissions_{0};
  std::atomic<int> prefill_fallback_calls_{0};
  std::atomic<int> prefill_partial_fallback_calls_{0};
  std::atomic<int> first_submission_ticket_{0};
  mutable std::mutex batch_size_mutex_;
  mutable std::vector<int> prefill_batch_sizes_;
  mutable std::vector<int> decode_batch_sizes_;
  mutable std::vector<int> auto_batch_sizes_;

  static std::atomic<int> global_submission_ticket_;
};

class PromptRecordingSliceBackend final : public LlamaCppBackend {
public:
  explicit PromptRecordingSliceBackend(std::vector<std::string> outputs)
      : outputs_(std::move(outputs)) {}

  bool IsReady() const override { return true; }

  std::string
  Generate(const std::string &prompt, int,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &,
           const std::function<bool()> &, int,
           std::vector<TokenLogprob> *,
           const std::vector<std::string> &) override {
    seen_prompts_.push_back(prompt);
    if (next_output_ >= outputs_.size()) {
      return {};
    }
    return outputs_[next_output_++];
  }

  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }

  const std::vector<std::string> &SeenPrompts() const { return seen_prompts_; }

private:
  std::vector<std::string> outputs_;
  std::vector<std::string> seen_prompts_;
  std::size_t next_output_{0};
};

std::atomic<int> AsyncLaneStubBackend::global_submission_ticket_{0};

class SessionLeaseStubBackend final : public ReadyStubBackend {
public:
  explicit SessionLeaseStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  void FreeSequence(int) override {
    free_sequence_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  int FreeSequenceCalls() const {
    return free_sequence_calls_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> free_sequence_calls_{0};
};

class NonSplitHandoffStubBackend final : public ReadyStubBackend {
public:
  explicit NonSplitHandoffStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  bool SupportsSplitPrefillDecodeHandoff() const override { return false; }
};

class DeferredFreeStubBackend final : public ReadyStubBackend {
public:
  explicit DeferredFreeStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  SequenceReleaseFence BeginFreeSequence(int sequence_id) override {
    last_sequence_id_.store(sequence_id, std::memory_order_relaxed);
    begin_calls_.fetch_add(1, std::memory_order_relaxed);
    SequenceReleaseFence fence;
    fence.token = 1;
    fence.pending = true;
    return fence;
  }

  bool PollFreeSequence(const SequenceReleaseFence &) override {
    poll_calls_.fetch_add(1, std::memory_order_relaxed);
    return ready_.load(std::memory_order_relaxed);
  }

  void SetReady(bool ready) { ready_.store(ready, std::memory_order_relaxed); }
  int BeginCalls() const { return begin_calls_.load(std::memory_order_relaxed); }
  int PollCalls() const { return poll_calls_.load(std::memory_order_relaxed); }
  int LastSequenceId() const {
    return last_sequence_id_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<bool> ready_{false};
  std::atomic<int> begin_calls_{0};
  std::atomic<int> poll_calls_{0};
  std::atomic<int> last_sequence_id_{-1};
};

class RejectingKVTransport final : public disaggregated::IKVTransport {
public:
  bool Enqueue(disaggregated::KVPacket packet) override {
    (void)packet;
    enqueue_calls_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  std::optional<disaggregated::KVPacket> TryDequeue() override {
    return std::nullopt;
  }

  std::size_t Size() const override { return 0; }
  std::size_t Capacity() const override { return 0; }

  int EnqueueCalls() const {
    return enqueue_calls_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> enqueue_calls_{0};
};

class CapturingKVTransport final : public disaggregated::IKVTransport {
public:
  explicit CapturingKVTransport(bool process_local = false)
      : process_local_(process_local) {}

  bool Enqueue(disaggregated::KVPacket packet) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (packet.ticket_id > 0) {
      ticket_stages_[packet.ticket_id] = packet.ticket_stage;
    }
    packets_.push_back(packet);
    queue_.push_back(std::move(packet));
    return true;
  }

  std::optional<disaggregated::KVPacket> TryDequeue() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    auto pkt = std::move(queue_.front());
    queue_.erase(queue_.begin());
    if (rewrite_dequeued_request_id_) {
      pkt.request_id = rewritten_request_id_;
    }
    return pkt;
  }

  std::size_t Size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  std::size_t Capacity() const override { return 1024; }
  bool IsProcessLocal() const override { return process_local_; }

  bool UpdateTicketStage(uint64_t ticket_id,
                         disaggregated::KVTicketStage stage) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ticket_stages_.find(ticket_id);
    if (it == ticket_stages_.end()) {
      return false;
    }
    it->second = stage;
    return true;
  }

  disaggregated::KVTicketStage
  GetTicketStage(uint64_t ticket_id) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ticket_stages_.find(ticket_id);
    if (it == ticket_stages_.end()) {
      return disaggregated::KVTicketStage::kNone;
    }
    return it->second;
  }

  std::vector<disaggregated::KVPacket> Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return packets_;
  }

  void RewriteDequeuedRequestId(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    rewrite_dequeued_request_id_ = true;
    rewritten_request_id_ = request_id;
  }

  void InjectPacket(disaggregated::KVPacket packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (packet.ticket_id > 0) {
      ticket_stages_[packet.ticket_id] = packet.ticket_stage;
    }
    queue_.push_back(std::move(packet));
  }

private:
  mutable std::mutex mutex_;
  std::vector<disaggregated::KVPacket> packets_;
  std::vector<disaggregated::KVPacket> queue_;
  std::unordered_map<uint64_t, disaggregated::KVTicketStage> ticket_stages_;
  bool rewrite_dequeued_request_id_{false};
  uint64_t rewritten_request_id_{0};
  bool process_local_{false};
};

class CountingRouter final : public ModelRouter {
public:
  void AddModel(const ModelInfo &info,
                std::shared_ptr<LlamaCppBackend> backend) {
    models_[info.id] = info;
    backends_[info.id] = std::move(backend);
    if (default_model_id_.empty()) {
      default_model_id_ = info.id;
    }
  }

  std::vector<ModelInfo> ListModels() const override {
    std::vector<ModelInfo> out;
    out.reserve(models_.size());
    for (const auto &entry : models_) {
      out.push_back(entry.second);
    }
    return out;
  }

  std::string LoadModel(const std::string &, const std::string &,
                        const std::string &, const std::string &) override {
    return "";
  }

  bool UnloadModel(const std::string &id) override {
    backends_.erase(id);
    return models_.erase(id) > 0;
  }

  ModelInfo *Resolve(const std::string &requested_model) override {
    resolve_calls_.fetch_add(1, std::memory_order_relaxed);
    if (models_.empty()) {
      return nullptr;
    }
    const std::string &key =
        requested_model.empty() ? default_model_id_ : requested_model;
    auto it = models_.find(key);
    return it == models_.end() ? nullptr : &it->second;
  }

  ModelInfo *ResolveExact(const std::string &model_id) override {
    resolve_calls_.fetch_add(1, std::memory_order_relaxed);
    auto it = models_.find(model_id);
    return it == models_.end() ? nullptr : &it->second;
  }

  std::shared_ptr<BackendInterface>
  GetBackend(const std::string &model_id) override {
    auto it = backends_.find(model_id);
    return it == backends_.end() ? nullptr : it->second;
  }

  bool SetDefaultModel(const std::string &model_id) override {
    if (models_.find(model_id) == models_.end()) {
      return false;
    }
    default_model_id_ = model_id;
    return true;
  }

  std::string DefaultModelId() const override { return default_model_id_; }
  std::string Name() const override { return "counting_router"; }

  int ResolveCalls() const {
    return resolve_calls_.load(std::memory_order_relaxed);
  }

private:
  std::unordered_map<std::string, ModelInfo> models_;
  std::unordered_map<std::string, std::shared_ptr<LlamaCppBackend>> backends_;
  std::string default_model_id_;
  std::atomic<int> resolve_calls_{0};
};

} // namespace

TEST_CASE("Scheduler stub response with no backend", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  // No router → no backend → should return no_backend flag.
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  InferenceRequest req;
  req.prompt = "Hello world";
  req.max_tokens = 10;
  auto fut = scheduler.Generate(req);
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
}

TEST_CASE("Scheduler with empty SingleModelRouter returns no_backend",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  auto router = std::make_shared<SingleModelRouter>();
  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "Hello";
  req.max_tokens = 5;
  auto fut = scheduler.Generate(req);
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
}

TEST_CASE("on_token callback fires on prefix cache hit", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      cache, [](int) {}, RadixPrefixCacheLimits{1024, 12});

  const std::string prompt = "cached prompt";
  auto prompt_tokens = tokenizer.Encode(prompt);
  prefix_cache->Insert(prompt_tokens, {100}, 0, nullptr);

  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, prefix_cache);

  std::vector<std::string> tokens_received;
  InferenceRequest req;
  req.prompt = prompt;
  req.max_tokens = 8;
  req.stream = true;
  req.on_token = [&](const std::string &tok, const TokenLogprob *) {
    tokens_received.push_back(tok);
  };

  auto fut = scheduler.Generate(std::move(req));
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
}

TEST_CASE("Scheduler clamps max_tokens=0 to 1", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  InferenceRequest req;
  req.prompt = "test";
  req.max_tokens = 0;
  auto fut = scheduler.Generate(std::move(req));
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
}

TEST_CASE("Scheduler slot manager mirrors alloc/free lifecycle",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  auto *slot_manager = scheduler.SlotManager();
  REQUIRE(slot_manager != nullptr);
  REQUIRE(slot_manager->GetUsedSlotCount() == 0);

  uint64_t generation = 0;
  const int slot = scheduler.AllocSeqSlot(4242, &generation);
  REQUIRE(slot >= 0);
  REQUIRE(generation == 1);
  REQUIRE(slot_manager->GetUsedSlotCount() == 1);

  bool found = false;
  for (const auto &status : slot_manager->GetSlotStatus()) {
    if (status.slot_id == slot) {
      REQUIRE(status.request_id == 4242);
      REQUIRE(status.generation == generation);
      REQUIRE(status.state == scheduler::SequenceState::kPrefilling);
      found = true;
      break;
    }
  }
  REQUIRE(found);

  scheduler.FreeSeqSlot(slot, generation);
  REQUIRE(slot_manager->GetUsedSlotCount() == 0);
}

TEST_CASE("Scheduler rejects stale sequence lease release after slot reuse",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  auto *slot_manager = scheduler.SlotManager();
  REQUIRE(slot_manager != nullptr);

  uint64_t first_generation = 0;
  const int slot = scheduler.AllocSeqSlot(6001, &first_generation);
  REQUIRE(slot >= 0);
  REQUIRE(first_generation == 1);
  scheduler.FreeSeqSlot(slot, first_generation);

  uint64_t second_generation = 0;
  const int reused_slot = scheduler.AllocSeqSlot(6002, &second_generation);
  REQUIRE(reused_slot == slot);
  REQUIRE(second_generation > first_generation);
  REQUIRE(slot_manager->GetUsedSlotCount() == 1);

  scheduler.FreeSeqSlot(slot, first_generation);
  REQUIRE(slot_manager->GetUsedSlotCount() == 1);

  bool found = false;
  for (const auto &status : slot_manager->GetSlotStatus()) {
    if (status.slot_id == slot) {
      REQUIRE(status.request_id == 6002);
      REQUIRE(status.generation == second_generation);
      REQUIRE(status.state == scheduler::SequenceState::kPrefilling);
      found = true;
      break;
    }
  }
  REQUIRE(found);

  scheduler.FreeSeqSlot(reused_slot, second_generation);
  REQUIRE(slot_manager->GetUsedSlotCount() == 0);
}

TEST_CASE("Scheduler defers slot reuse until backend free fence is ready",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  auto backend = std::make_shared<DeferredFreeStubBackend>("deferred");
  auto *slot_manager = scheduler.SlotManager();
  REQUIRE(slot_manager != nullptr);
  GlobalMetrics().SetBackend("cpu");
  const int64_t deferred_completed_before = ReadScalarMetric(
      "inferflux_scheduler_deferred_sequence_retirements_completed_total{"
      "backend=\"cpu\"} ");

  uint64_t generation = 0;
  const int slot = scheduler.AllocSeqSlot(7001, &generation);
  REQUIRE(slot >= 0);
  REQUIRE(generation == 1);

  scheduler.FreeSeqSlot(slot, generation, backend);
  REQUIRE(slot_manager->GetRetiringSlotCount() == 1);
  REQUIRE(slot_manager->GetUsedSlotCount() == 0);
  // Free = max_slots - retiring(1).  Retiring slots are not counted as free.
  const int remaining_free =
      static_cast<int>(slot_manager->GetMaxSlots()) - 1;
  REQUIRE(slot_manager->GetFreeSlotCount() ==
          static_cast<size_t>(remaining_free));
  REQUIRE(GlobalMetrics().GetSchedulerDeferredSequenceRetirements() == 1);
  REQUIRE(backend->BeginCalls() == 1);
  REQUIRE(backend->LastSequenceId() == slot);

  std::vector<std::pair<int, uint64_t>> held_slots;
  held_slots.reserve(remaining_free);
  for (int i = 0; i < remaining_free; ++i) {
    uint64_t held_generation = 0;
    const int held_slot = scheduler.AllocSeqSlot(7100 + i, &held_generation);
    REQUIRE(held_slot >= 0);
    REQUIRE(held_slot != slot);
    held_slots.push_back({held_slot, held_generation});
  }

  uint64_t blocked_generation = 0;
  REQUIRE(scheduler.AllocSeqSlot(7002, &blocked_generation) == -1);
  REQUIRE(slot_manager->GetRetiringSlotCount() == 1);
  REQUIRE(backend->PollCalls() >= 1);

  backend->SetReady(true);
  uint64_t reacquired_generation = 0;
  const int reacquired_slot =
      scheduler.AllocSeqSlot(7003, &reacquired_generation);
  REQUIRE(reacquired_slot == slot);
  REQUIRE(reacquired_generation > generation);
  REQUIRE(slot_manager->GetRetiringSlotCount() == 0);
  REQUIRE(GlobalMetrics().GetSchedulerDeferredSequenceRetirements() == 0);
  REQUIRE(ReadScalarMetric(
              "inferflux_scheduler_deferred_sequence_retirements_completed_"
              "total{backend=\"cpu\"} ") ==
          deferred_completed_before + 1);

  scheduler.FreeSeqSlot(reacquired_slot, reacquired_generation);
  for (const auto &[held_slot, held_generation] : held_slots) {
    scheduler.FreeSeqSlot(held_slot, held_generation);
  }
}

TEST_CASE("Scheduler prefill uses async unified prefill lane when available",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<AsyncLaneStubBackend>("ok");

  ModelInfo info;
  info.id = "lane-model";
  info.path = "/tmp/lane.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "lane activity";
  req.max_tokens = 2;
  auto resp = scheduler.Generate(std::move(req)).get();

  REQUIRE_FALSE(resp.no_backend);
  REQUIRE_FALSE(resp.completion.empty());
  REQUIRE(backend->PrefillSubmissions() > 0);
  REQUIRE(backend->DecodeSubmissions() > 0);
  REQUIRE(backend->PrefillFallbackCalls() == 0);
  REQUIRE(backend->PrefillPartialFallbackCalls() == 0);
}

TEST_CASE("Scheduler routes decode-ready prefill requests directly to decode",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<AsyncLaneStubBackend>("ok");

  ModelInfo info;
  info.id = "decode-ready-model";
  info.path = "/tmp/decode-ready.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler scheduler(tokenizer, device, cache, router);

  auto pending = std::make_shared<Scheduler::PendingRequest>();
  pending->inference.id = 7;
  pending->inference.model = info.id;
  pending->inference.resolved_model = info.id;
  pending->inference.prompt = "decode ready";
  pending->inference.prompt_tokens = tokenizer.Encode(pending->inference.prompt);
  pending->inference.bpe_prompt_tokens = {1, 2, 3, 4};
  pending->inference.max_tokens = 2;
  pending->inference.remaining_decode_tokens = 2;
  pending->inference.phase = RequestPhase::kPrefill;
  pending->enqueue_time = std::chrono::steady_clock::now();
  pending->inference.enqueue_time = pending->enqueue_time;
  pending->resolved_backend = backend;
  pending->priority = 0;
  pending->priority_level = 0;
  pending->sequence = 7;

  uint64_t generation = 0;
  const int seq_id = scheduler.AllocSeqSlot(pending->inference.id, &generation);
  REQUIRE(seq_id >= 0);
  pending->inference.sequence_id = seq_id;
  pending->inference.sequence_generation = generation;
  pending->inference.n_past = 4;
  pending->inference.first_token = 100;
  pending->inference.first_piece = "p";

  Scheduler::BatchSelection selection;
  selection.pending.push_back(pending);
  selection.batch.requests.push_back(&pending->inference);
  selection.total_tokens = pending->inference.prompt_tokens.size();

  auto future = pending->promise.get_future();
  scheduler.ProcessBatch(std::move(selection));
  auto result = future.get();

  REQUIRE_FALSE(result.no_backend);
  REQUIRE(result.completion == "pd");
  REQUIRE(backend->PrefillSubmissions() == 0);
  REQUIRE(backend->PrefillFallbackCalls() == 0);
  REQUIRE(backend->PrefillPartialFallbackCalls() == 0);
  REQUIRE(backend->DecodeSubmissions() > 0);
}

TEST_CASE("Scheduler opportunistically accumulates decode batches when "
          "min_batch_size is one",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<AsyncLaneStubBackend>("ok");

  ModelInfo info;
  info.id = "decode-accumulate-model";
  info.path = "/tmp/decode-accumulate.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler::Config scheduler_config;
  scheduler_config.max_batch_size = 4;
  scheduler_config.min_batch_size = 1;
  scheduler_config.batch_accumulation_ms = 20;

  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config, ModelSelectionOptions{},
                      scheduler_config);

  auto make_request = [&]() {
    InferenceRequest req;
    req.model = info.id;
    req.prompt = "decode accumulation check";
    req.max_tokens = 2;
    return req;
  };

  auto first = scheduler.Generate(make_request());
  auto second = scheduler.Generate(make_request());

  auto first_resp = first.get();
  auto second_resp = second.get();

  REQUIRE_FALSE(first_resp.no_backend);
  REQUIRE_FALSE(second_resp.no_backend);
  REQUIRE(backend->DecodeSubmissions() > 0);

  const auto decode_batch_sizes = backend->DecodeBatchSizes();
  REQUIRE(std::any_of(decode_batch_sizes.begin(), decode_batch_sizes.end(),
                      [](int size) { return size >= 2; }));
}

TEST_CASE("Scheduler decode worker rebuilds stepwise decode cohorts",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<AsyncLaneStubBackend>("ok");

  ModelInfo info;
  info.id = "stepwise-decode-model";
  info.path = "/tmp/stepwise-decode.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler::Config scheduler_config;
  scheduler_config.max_batch_size = 4;
  scheduler_config.min_batch_size = 2;
  scheduler_config.batch_accumulation_ms = 10;
  scheduler_config.decode_burst_tokens = 0;

  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;

  const int64_t bucket2_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_batch_size_total{bucket=\"2\"} ");
  const int64_t bucket1_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_batch_size_total{bucket=\"1\"} ");
  const int64_t direct_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_execution_path_total{path=\"direct_stepwise\"} ");
  const int64_t general_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_execution_path_total{path=\"general\"} ");

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config, ModelSelectionOptions{},
                      scheduler_config);

  auto make_request = [&]() {
    InferenceRequest req;
    req.model = info.id;
    req.prompt = "stepwise decode lane";
    req.max_tokens = 3;
    return req;
  };

  auto first = scheduler.Generate(make_request());
  auto second = scheduler.Generate(make_request());

  auto first_resp = first.get();
  auto second_resp = second.get();

  REQUIRE_FALSE(first_resp.no_backend);
  REQUIRE_FALSE(second_resp.no_backend);
  REQUIRE(first_resp.completion == "pdd");
  REQUIRE(second_resp.completion == "pdd");

  const int64_t bucket2_after = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_batch_size_total{bucket=\"2\"} ");
  const int64_t bucket1_after = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_batch_size_total{bucket=\"1\"} ");
  const int64_t direct_after = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_execution_path_total{path=\"direct_stepwise\"} ");
  const int64_t general_after = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_execution_path_total{path=\"general\"} ");
  REQUIRE(bucket2_after - bucket2_before == 2);
  REQUIRE(bucket1_after - bucket1_before == 0);
  REQUIRE(direct_after - direct_before == 2);
  REQUIRE(general_after - general_before == 0);
}

TEST_CASE("Scheduler sticky decode fill skips incompatible queue head",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  Scheduler scheduler(tokenizer, device, cache, router);

  auto backend_a = std::make_shared<AsyncLaneStubBackend>("a");
  auto backend_b = std::make_shared<AsyncLaneStubBackend>("b");

  auto make_pending = [](int id, int sequence_id,
                         const std::shared_ptr<LlamaCppBackend> &backend) {
    auto pending = std::make_shared<Scheduler::PendingRequest>();
    pending->inference.id = id;
    pending->inference.model = "sticky-model";
    pending->inference.resolved_model = "sticky-model";
    pending->inference.prompt = "sticky decode";
    pending->inference.phase = RequestPhase::kDecode;
    pending->inference.sequence_id = sequence_id;
    pending->inference.sequence_generation = 1;
    pending->inference.n_past = 4;
    pending->inference.first_token = 101;
    pending->inference.response_format_supported = true;
    pending->resolved_backend = backend;
    return pending;
  };

  auto active = make_pending(1, 11, backend_a);
  auto incompatible = make_pending(2, 12, backend_b);
  auto compatible_one = make_pending(3, 13, backend_a);
  auto compatible_two = make_pending(4, 14, backend_a);

  std::vector<std::shared_ptr<Scheduler::PendingRequest>> batch{active};
  const int64_t sticky_merge_events_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_sticky_merge_total{merged=\"2\"} ");
  const int64_t sticky_merged_requests_before = ReadScalarMetric(
      "inferflux_scheduler_decode_worker_sticky_merged_requests_total ");

  {
    std::lock_guard<std::mutex> lock(scheduler.queue_mutex_);
    scheduler.pending_decode_ = {incompatible, compatible_one, compatible_two};
    const std::size_t merged =
        scheduler.AppendCompatiblePendingDecodeLocked(&batch, backend_a, 3);
    REQUIRE(merged == 2);
  }

  REQUIRE(batch.size() == 3);
  REQUIRE(batch[0] == active);
  REQUIRE(batch[1] == compatible_one);
  REQUIRE(batch[2] == compatible_two);
  REQUIRE(scheduler.pending_decode_.size() == 1);
  REQUIRE(scheduler.pending_decode_[0] == incompatible);
  REQUIRE(ReadScalarMetric(
              "inferflux_scheduler_decode_worker_sticky_merge_total{merged=\"2\"} ") -
              sticky_merge_events_before ==
          1);
  REQUIRE(ReadScalarMetric(
              "inferflux_scheduler_decode_worker_sticky_merged_requests_total ") -
              sticky_merged_requests_before ==
          2);
}

TEST_CASE("Scheduler preserves bound backend for in-flight decode requests",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<CountingRouter>();
  auto backend = std::make_shared<ReadyStubBackend>("ok");

  ModelInfo info;
  info.id = "bound-decode-model";
  info.path = "/tmp/bound-decode.gguf";
  info.backend = "cuda";
  router->AddModel(info, backend);
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler scheduler(tokenizer, device, cache, router);

  auto decode_pending = std::make_shared<Scheduler::PendingRequest>();
  decode_pending->inference.id = 7;
  decode_pending->inference.model = info.id;
  decode_pending->inference.resolved_model = info.id;
  decode_pending->inference.prompt = "decode binding";
  decode_pending->inference.phase = RequestPhase::kDecode;
  decode_pending->inference.sequence_id = 3;
  decode_pending->inference.sequence_generation = 1;
  decode_pending->inference.n_past = 4;
  decode_pending->inference.first_token = 42;
  decode_pending->resolved_backend = backend;

  std::vector<std::shared_ptr<Scheduler::PendingRequest>> decode_batch{
      decode_pending};
  scheduler.ResolveBackends(decode_batch);

  REQUIRE(router->ResolveCalls() == 0);
  REQUIRE(decode_pending->resolved_backend.get() == backend.get());
  REQUIRE(decode_pending->inference.resolved_model == info.id);

  auto fresh_pending = std::make_shared<Scheduler::PendingRequest>();
  fresh_pending->inference.id = 8;
  fresh_pending->inference.model = info.id;
  fresh_pending->inference.prompt = "fresh routing";
  fresh_pending->inference.phase = RequestPhase::kPending;

  std::vector<std::shared_ptr<Scheduler::PendingRequest>> fresh_batch{
      fresh_pending};
  scheduler.ResolveBackends(fresh_batch);

  REQUIRE(router->ResolveCalls() == 1);
  REQUIRE(fresh_pending->resolved_backend.get() == backend.get());
  REQUIRE(fresh_pending->inference.resolved_model == info.id);
}

TEST_CASE("Scheduler batch selection deduplicates one pending request leaked "
          "into both queues",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  Scheduler scheduler(tokenizer, device, cache, router);

  auto pending = std::make_shared<Scheduler::PendingRequest>();
  pending->inference.id = 42;
  pending->inference.prompt = "duplicate queue entry";
  pending->inference.prompt_tokens = tokenizer.Encode(pending->inference.prompt);
  pending->inference.phase = RequestPhase::kPending;
  pending->priority = 0;
  pending->priority_level = 0;
  pending->sequence = 42;
  pending->enqueue_time = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(scheduler.queue_mutex_);
    scheduler.pending_prefill_.push_back(pending);
    scheduler.pending_decode_.push_back(pending);
    auto selection = scheduler.BuildBatchLocked();

    REQUIRE(selection.pending.size() == 1);
    REQUIRE(selection.pending.front().get() == pending.get());
    REQUIRE(scheduler.pending_prefill_.empty());
    REQUIRE(scheduler.pending_decode_.empty());
  }
}

TEST_CASE("Scheduler token budget accounts decode slices, not full prompts",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<ReadyStubBackend>(" next");

  ModelInfo info;
  info.id = "decode-budget-model";
  info.path = "/tmp/decode-budget.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  FairnessConfig fairness_config;
  fairness_config.max_timeslice_tokens = 1;
  fairness_config.high_priority_threshold = 5;

  Scheduler::Config scheduler_config;
  scheduler_config.max_batch_size = 2;
  scheduler_config.max_batch_tokens = 12;
  scheduler_config.min_batch_size = 2;
  scheduler_config.batch_accumulation_ms = 20;

  Scheduler scheduler(
      tokenizer, device, cache, router, nullptr, nullptr, fairness_config,
      DisaggregatedConfig{},
      ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                            /*require_ready_backend=*/true},
      scheduler_config);

  GlobalMetrics().SetBackend("cpu");
  const int64_t skips_before = ReadBatchTokenBudgetSkipsTotal();

  auto make_request = []() {
    InferenceRequest req;
    req.prompt = "alpha beta gamma delta epsilon";
    req.max_tokens = 20;
    req.priority = 0;
    return req;
  };

  auto fut1 = scheduler.Generate(make_request());
  auto fut2 = scheduler.Generate(make_request());
  auto resp1 = fut1.get();
  auto resp2 = fut2.get();

  REQUIRE_FALSE(resp1.no_backend);
  REQUIRE_FALSE(resp2.no_backend);
  REQUIRE_FALSE(resp1.completion.empty());
  REQUIRE_FALSE(resp2.completion.empty());

  const int64_t skips_after = ReadBatchTokenBudgetSkipsTotal();
  REQUIRE(skips_after == skips_before);
}

TEST_CASE("Scheduler unified mode records decode assembly selection metrics",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  Scheduler scheduler(tokenizer, device, cache, router);

  auto make_pending = [&](int id) {
    auto pending = std::make_shared<Scheduler::PendingRequest>();
    pending->inference.id = id;
    pending->inference.prompt = "decode assembly";
    pending->inference.prompt_tokens = tokenizer.Encode(pending->inference.prompt);
    pending->inference.phase = RequestPhase::kDecode;
    pending->inference.n_past = 4;
    pending->priority = 0;
    pending->priority_level = 0;
    pending->sequence = static_cast<uint64_t>(id);
    pending->enqueue_time = std::chrono::steady_clock::now();
    return pending;
  };

  const int64_t ready_before = ReadScalarMetric(
      "inferflux_scheduler_decode_assembly_ready_total{mode=\"unified\",bucket=\"2\"} ");
  const int64_t selected_before = ReadScalarMetric(
      "inferflux_scheduler_decode_assembly_selected_total{mode=\"unified\",bucket=\"2\"} ");

  {
    std::lock_guard<std::mutex> lock(scheduler.queue_mutex_);
    scheduler.pending_decode_.push_back(make_pending(1));
    scheduler.pending_decode_.push_back(make_pending(2));
    auto selection = scheduler.BuildBatchLocked();
    REQUIRE(selection.pending.size() == 2);
  }

  REQUIRE(ReadScalarMetric(
              "inferflux_scheduler_decode_assembly_ready_total{mode=\"unified\",bucket=\"2\"} ") -
              ready_before ==
          1);
  REQUIRE(ReadScalarMetric(
              "inferflux_scheduler_decode_assembly_selected_total{mode=\"unified\",bucket=\"2\"} ") -
              selected_before ==
          1);
}

TEST_CASE("Scheduler session handles preserve sequence until lease release",
          "[scheduler]") {
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  {
    SimpleTokenizer tokenizer;
    auto device = std::make_shared<CPUDeviceContext>();
    auto cache = std::make_shared<PagedKVCache>(
        16, 1024, PagedKVCache::EvictionPolicy::kLRU);
    auto router = std::make_shared<SingleModelRouter>();

    ModelInfo info;
    info.id = "session-model";
    info.path = "/tmp/session.gguf";
    info.backend = "cpu";
    REQUIRE(router->RegisterModel(info, backend));
    REQUIRE(router->SetDefaultModel(info.id));

    Scheduler::Config cfg;
    cfg.session_handles.enabled = true;
    cfg.session_handles.ttl_ms = 60000;
    cfg.session_handles.max_sessions = 64;

    Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                        FairnessConfig{}, DisaggregatedConfig{},
                        ModelSelectionOptions{}, cfg);

    InferenceRequest req1;
    req1.model = info.id;
    req1.session_id = "session-a";
    req1.prompt = "hello session";
    req1.max_tokens = 2;
    auto resp1 = scheduler.Generate(std::move(req1)).get();
    REQUIRE_FALSE(resp1.no_backend);

    InferenceRequest req2;
    req2.model = info.id;
    req2.session_id = "session-a";
    req2.prompt = "hello session";
    req2.max_tokens = 2;
    auto resp2 = scheduler.Generate(std::move(req2)).get();
    REQUIRE_FALSE(resp2.no_backend);

    // Session-owned sequences should reduce per-request sequence teardown.
    REQUIRE(backend->FreeSequenceCalls() < 2);
  }

  // Scheduler teardown drains session handles and frees retained sequence
  // state.
  REQUIRE(backend->FreeSequenceCalls() >= 1);
}

TEST_CASE("PrepareFairnessDecodeRequeue preserves original prompt state",
          "[scheduler]") {
  InferenceRequest req;
  req.id = 42;
  req.prompt = "seed prompt";
  req.phase = RequestPhase::kPending;
  req.sequence_id = 7;
  req.sequence_generation = 3;
  req.n_past = 11;
  req.remaining_decode_tokens = 2;
  req.accumulated_output = "AB";

  const auto now = std::chrono::steady_clock::now();
  PrepareFairnessDecodeRequeue(&req, now);

  REQUIRE(req.prompt == "seed prompt");
  REQUIRE(req.accumulated_output == "AB");
  REQUIRE(req.phase == RequestPhase::kDecode);
  REQUIRE(req.enqueue_time == now);
  REQUIRE(req.sequence_id == 7);
  REQUIRE(req.sequence_generation == 3);
}

TEST_CASE("Scheduler fairness requeue does not mutate prompt between slices",
          "[scheduler]") {
  auto backend = std::make_shared<PromptRecordingSliceBackend>(
      std::vector<std::string>{"A", "B", "C"});

  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo info;
  info.id = "fairness-prompt";
  info.path = "/tmp/fairness-prompt.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  FairnessConfig fairness_config;
  fairness_config.max_timeslice_tokens = 1;

  Scheduler::Config scheduler_config;
  scheduler_config.max_batch_size = 1;
  scheduler_config.max_batch_tokens = 8;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      fairness_config, DisaggregatedConfig{},
                      ModelSelectionOptions{}, scheduler_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "seed prompt";
  req.max_tokens = 3;

  auto result = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(result.no_backend);
  REQUIRE(result.completion == "ABC");
  REQUIRE(backend->SeenPrompts() ==
          std::vector<std::string>{"seed prompt", "seed prompt",
                                   "seed prompt"});
}

TEST_CASE("Scheduler fails fast when distributed enqueue retries are exhausted",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  ModelInfo info;
  info.id = "dist-retry-model";
  info.path = "/tmp/dist-retry.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto rejecting_transport = std::make_shared<RejectingKVTransport>();
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = rejecting_transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "distributed retry test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.completion.find("distributed_overloaded") != std::string::npos);
  REQUIRE(rejecting_transport->EnqueueCalls() >= 2);
  REQUIRE(backend->FreeSequenceCalls() >= 1);
}

TEST_CASE("Scheduler stamps distributed KV packets with transport ticket ids",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  ModelInfo info;
  info.id = "dist-ticket-model";
  info.path = "/tmp/dist-ticket.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto transport = std::make_shared<CapturingKVTransport>();
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "distributed ticket test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(resp.no_backend);

  const auto packets = transport->Snapshot();
  REQUIRE_FALSE(packets.empty());
  REQUIRE(packets.front().ticket_id > 0);
  REQUIRE(packets.front().ticket_stage ==
          disaggregated::KVTicketStage::kEnqueued);
  REQUIRE(transport->GetTicketStage(packets.front().ticket_id) ==
          disaggregated::KVTicketStage::kCommitted);
}

TEST_CASE("Scheduler keeps non-split backends on local decode lane",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<NonSplitHandoffStubBackend>("local-safe");

  ModelInfo info;
  info.id = "local-safe-model";
  info.path = "/tmp/local-safe.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto transport = std::make_shared<CapturingKVTransport>();
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;

  GlobalMetrics().SetBackend("cpu");
  const int64_t enqueued_before = ReadDisaggTicketStageTotal("enqueued");

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "split lane safety test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.completion == "local-safe");
  REQUIRE(transport->Snapshot().empty());
  REQUIRE(ReadDisaggTicketStageTotal("enqueued") - enqueued_before == 0);
}

TEST_CASE("Scheduler skips KV blob serialize/hydrate for process-local native "
          "split handoff",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<ProcessLocalSplitStubBackend>("ok");

  ModelInfo info;
  info.id = "process-local-split-model";
  info.path = "/tmp/process-local-split.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto transport = std::make_shared<CapturingKVTransport>(true);
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "process local split handoff test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.completion == "ok");

  const auto packets = transport->Snapshot();
  REQUIRE_FALSE(packets.empty());
  REQUIRE(packets.front().sequence_id >= 0);
  REQUIRE(packets.front().kv_blob.empty());
  REQUIRE(backend->SerializeCalls() == 0);
  REQUIRE(backend->HydrateCalls() == 0);
}

TEST_CASE("Scheduler advances distributed KV ticket to committed on dequeue",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  ModelInfo info;
  info.id = "dist-ticket-stage-model";
  info.path = "/tmp/dist-ticket-stage.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto transport = std::make_shared<CapturingKVTransport>();
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  GlobalMetrics().SetBackend("cpu");
  const int64_t enqueued_before = ReadDisaggTicketStageTotal("enqueued");
  const int64_t acknowledged_before =
      ReadDisaggTicketStageTotal("acknowledged");
  const int64_t committed_before = ReadDisaggTicketStageTotal("committed");

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "distributed ticket stage test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(resp.no_backend);

  const auto packets = transport->Snapshot();
  REQUIRE_FALSE(packets.empty());
  REQUIRE(transport->GetTicketStage(packets.front().ticket_id) ==
          disaggregated::KVTicketStage::kCommitted);
  REQUIRE(ReadDisaggTicketStageTotal("enqueued") - enqueued_before == 1);
  REQUIRE(ReadDisaggTicketStageTotal("acknowledged") - acknowledged_before ==
          1);
  REQUIRE(ReadDisaggTicketStageTotal("committed") - committed_before == 1);
}

TEST_CASE("Scheduler marks unmatched distributed KV tickets as timed out",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");
  auto transport = std::make_shared<CapturingKVTransport>();

  ModelInfo info;
  info.id = "dist-ticket-timeout-model";
  info.path = "/tmp/dist-ticket-timeout.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  GlobalMetrics().SetBackend("cpu");
  const int64_t enqueued_before = ReadDisaggTicketStageTotal("enqueued");
  const int64_t acknowledged_before =
      ReadDisaggTicketStageTotal("acknowledged");
  const int64_t timed_out_before = ReadDisaggTicketStageTotal("timed_out");
  transport->RewriteDequeuedRequestId(999999);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "distributed ticket timeout test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE_FALSE(resp.no_backend);

  const auto packets = transport->Snapshot();
  REQUIRE_FALSE(packets.empty());
  REQUIRE(transport->GetTicketStage(packets.front().ticket_id) ==
          disaggregated::KVTicketStage::kTimedOut);
  REQUIRE(ReadDisaggTicketStageTotal("enqueued") - enqueued_before == 1);
  REQUIRE(ReadDisaggTicketStageTotal("acknowledged") - acknowledged_before ==
          1);
  REQUIRE(ReadDisaggTicketStageTotal("timed_out") - timed_out_before == 1);
}

TEST_CASE(
    "Scheduler drains transport-only KV tickets without decode queue work",
    "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto transport = std::make_shared<CapturingKVTransport>();

  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = transport;

  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  GlobalMetrics().SetBackend("cpu");
  const int64_t acknowledged_before =
      ReadDisaggTicketStageTotal("acknowledged");
  const int64_t timed_out_before = ReadDisaggTicketStageTotal("timed_out");

  disaggregated::KVPacket packet;
  packet.request_id = 424242;
  packet.ticket_id = 888;
  packet.ticket_stage = disaggregated::KVTicketStage::kEnqueued;
  packet.n_past = 3;
  transport->InjectPacket(std::move(packet));

  REQUIRE(WaitForCondition([&]() {
    return transport->GetTicketStage(888) ==
           disaggregated::KVTicketStage::kTimedOut;
  }));
  REQUIRE(WaitForCondition([&]() {
    return ReadDisaggTicketStageTotal("acknowledged") - acknowledged_before ==
               1 &&
           ReadDisaggTicketStageTotal("timed_out") - timed_out_before == 1;
  }));
}

TEST_CASE("FairnessController evaluation", "[fairness]") {
  FairnessConfig cfg;
  cfg.enable_preemption = true;
  cfg.high_priority_threshold = 5;
  FairnessController controller(cfg);

  InferenceRequest low, high;
  low.priority_level = 1;
  high.priority_level = 10;

  std::vector<FairnessEntry> batch{{&low, 1, 0}};
  std::vector<FairnessEntry> queue{{&high, 10, 0}};

  auto decision = controller.Evaluate(batch, queue);
  REQUIRE(decision.swap);
}

TEST_CASE("Scheduler fairness preemption keeps decode requests on decode queue",
          "[scheduler][fairness]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  FairnessConfig cfg;
  cfg.enable_preemption = true;
  cfg.high_priority_threshold = 5;

  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr, cfg);

  auto decode_pending = std::make_shared<Scheduler::PendingRequest>();
  decode_pending->priority = 1;
  decode_pending->inference.priority = 1;
  decode_pending->inference.priority_level = 1;
  decode_pending->inference.id = 101;
  decode_pending->inference.phase = RequestPhase::kDecode;
  decode_pending->inference.sequence_id = 6;
  decode_pending->inference.sequence_generation = 7;
  decode_pending->inference.n_past = 12;
  decode_pending->inference.remaining_decode_tokens = 16;

  auto high_pending = std::make_shared<Scheduler::PendingRequest>();
  high_pending->priority = 10;
  high_pending->inference.priority = 10;
  high_pending->inference.priority_level = 10;
  high_pending->inference.id = 202;
  high_pending->inference.phase = RequestPhase::kPending;

  scheduler.pending_prefill_.push_back(high_pending);

  Scheduler::BatchSelection selection;
  selection.pending.push_back(decode_pending);

  scheduler.ApplyFairness(&selection);

  REQUIRE(selection.pending.size() == 1);
  REQUIRE(selection.pending[0] == high_pending);
  REQUIRE(scheduler.pending_prefill_.empty());
  REQUIRE(scheduler.pending_decode_.size() == 1);
  REQUIRE(scheduler.pending_decode_[0] == decode_pending);
  REQUIRE(decode_pending->inference.phase == RequestPhase::kDecode);
  REQUIRE(decode_pending->inference.sequence_id == 6);
  REQUIRE(decode_pending->inference.sequence_generation == 7);
  REQUIRE(decode_pending->inference.n_past == 12);
  REQUIRE(decode_pending->inference.remaining_decode_tokens == 16);
}

TEST_CASE(
    "Scheduler falls back to compatible backend for default model routing",
    "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/shared.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_logprobs = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.model_id == "shared-cpu");
  REQUIRE_FALSE(resp.completion.empty());
}

TEST_CASE("Scheduler routes streaming requests to streaming-capable backend",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/other.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_streaming = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.stream = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.model_id == "shared-cpu");
  REQUIRE_FALSE(resp.completion.empty());
}

TEST_CASE("Scheduler respects same-path fallback routing scope",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo primary;
  primary.id = "default-cuda";
  primary.path = "/tmp/shared.gguf";
  primary.backend = "cuda";
  REQUIRE(router->RegisterModel(
      primary, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cross_path_fallback;
  cross_path_fallback.id = "other-cpu";
  cross_path_fallback.path = "/tmp/other.gguf";
  cross_path_fallback.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cross_path_fallback, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel(primary.id));

  auto *resolved_primary = router->Resolve(primary.id);
  REQUIRE(resolved_primary != nullptr);
  resolved_primary->capabilities.supports_logprobs = false;

  ModelSelectionOptions selection_options;
  selection_options.allow_capability_fallback_for_default = true;
  selection_options.require_ready_backend = true;
  selection_options.capability_fallback_scope =
      CapabilityFallbackScope::kSamePathOnly;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      selection_options);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.model_id.empty());
  REQUIRE(resp.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler applies runtime routing policy updates", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo primary;
  primary.id = "default-cuda";
  primary.path = "/tmp/shared.gguf";
  primary.backend = "cuda";
  REQUIRE(router->RegisterModel(
      primary, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cross_path_fallback;
  cross_path_fallback.id = "other-cpu";
  cross_path_fallback.path = "/tmp/other.gguf";
  cross_path_fallback.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cross_path_fallback, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel(primary.id));

  auto *resolved_primary = router->Resolve(primary.id);
  REQUIRE(resolved_primary != nullptr);
  resolved_primary->capabilities.supports_logprobs = false;

  ModelSelectionOptions selection_options;
  selection_options.allow_capability_fallback_for_default = true;
  selection_options.require_ready_backend = true;
  selection_options.capability_fallback_scope =
      CapabilityFallbackScope::kAnyCompatible;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      selection_options);

  InferenceRequest req_before;
  req_before.prompt = "hello";
  req_before.max_tokens = 4;
  req_before.collect_logprobs = true;
  auto resp_before = scheduler.Generate(req_before).get();
  REQUIRE_FALSE(resp_before.no_backend);
  REQUIRE(resp_before.model_id == "other-cpu");

  ModelSelectionOptions tightened = selection_options;
  tightened.capability_fallback_scope = CapabilityFallbackScope::kSamePathOnly;
  scheduler.UpdateModelSelectionOptions(tightened);

  auto snapshot = scheduler.ModelSelectionOptionsSnapshot();
  REQUIRE(snapshot.capability_fallback_scope ==
          CapabilityFallbackScope::kSamePathOnly);

  InferenceRequest req_after;
  req_after.prompt = "hello";
  req_after.max_tokens = 4;
  req_after.collect_logprobs = true;
  auto resp_after = scheduler.Generate(req_after).get();
  REQUIRE(resp_after.no_backend);
  REQUIRE(resp_after.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler does not auto-fallback for explicit model requests",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/shared.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_logprobs = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.model = "shared-cuda";
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.model_id.empty());
  REQUIRE(resp.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler batch policy parse and stringify are stable",
          "[scheduler]") {
  REQUIRE(SchedulerBatchPolicyToString(SchedulerBatchPolicy::kPriorityAge) ==
          "priority_age");
  REQUIRE(SchedulerBatchPolicyToString(SchedulerBatchPolicy::kLpmPriority) ==
          "lpm_priority");
  REQUIRE(
      SchedulerBatchPolicyToString(SchedulerBatchPolicy::kThroughputBalanced) ==
      "throughput_balanced");

  REQUIRE(IsSchedulerBatchPolicyValue("priority_age"));
  REQUIRE(IsSchedulerBatchPolicyValue("LPM_PRIORITY"));
  REQUIRE(IsSchedulerBatchPolicyValue("throughput_balanced"));
  REQUIRE_FALSE(IsSchedulerBatchPolicyValue("unknown_policy"));

  REQUIRE(ParseSchedulerBatchPolicy("priority_age") ==
          SchedulerBatchPolicy::kPriorityAge);
  REQUIRE(ParseSchedulerBatchPolicy("lpm_priority") ==
          SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(ParseSchedulerBatchPolicy("THROUGHPUT_BALANCED") ==
          SchedulerBatchPolicy::kThroughputBalanced);
  REQUIRE(ParseSchedulerBatchPolicy(
              "invalid", SchedulerBatchPolicy::kThroughputBalanced) ==
          SchedulerBatchPolicy::kThroughputBalanced);
}

TEST_CASE("Scheduler preserves configured batch policy", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  Scheduler::Config cfg;
  cfg.batch_policy = SchedulerBatchPolicy::kLpmPriority;
  cfg.decode_burst_tokens = 3;
  cfg.chunked_prefill_tokens = 96;
  cfg.mixed_prefill_budget_ratio = 0.4;
  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  REQUIRE(scheduler.BatchPolicy() == SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(scheduler.DecodeBurstTokens() == 3);
  REQUIRE(scheduler.ChunkedPrefillTokens() == 96);
  REQUIRE(scheduler.MixedPrefillBudgetRatio() ==
          Catch::Approx(0.4).epsilon(1e-6));
}

TEST_CASE("Scheduler normalizes mixed-step tuning bounds", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  Scheduler::Config cfg;
  cfg.decode_burst_tokens = -7;
  cfg.chunked_prefill_tokens = 0;
  cfg.mixed_prefill_budget_ratio = 2.5;
  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  REQUIRE(scheduler.DecodeBurstTokens() == 0);
  REQUIRE(scheduler.ChunkedPrefillTokens() == 1);
  REQUIRE(scheduler.MixedPrefillBudgetRatio() ==
          Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("Scheduler lpm policy prioritizes prefix-affinity request",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      8, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      cache, [](int) {}, RadixPrefixCacheLimits{1024, 32});

  auto cold_backend = std::make_shared<AsyncLaneStubBackend>("cold");
  auto hot_backend = std::make_shared<AsyncLaneStubBackend>("hot");

  ModelInfo cold_info;
  cold_info.id = "cold-model";
  cold_info.path = "/tmp/cold.gguf";
  cold_info.backend = "cpu";
  REQUIRE(router->RegisterModel(cold_info, cold_backend));

  ModelInfo hot_info;
  hot_info.id = "hot-model";
  hot_info.path = "/tmp/hot.gguf";
  hot_info.backend = "cpu";
  REQUIRE(router->RegisterModel(hot_info, hot_backend));
  REQUIRE(router->SetDefaultModel(cold_info.id));

  const std::string hot_prompt = "prefix hot prompt";
  auto hot_tokens = tokenizer.Encode(hot_prompt);
  prefix_cache->Insert(hot_tokens, {101}, 7, hot_backend);

  Scheduler::Config cfg;
  cfg.max_batch_size = 1;
  cfg.min_batch_size = 2;
  cfg.batch_accumulation_ms = 20;
  cfg.batch_policy = SchedulerBatchPolicy::kLpmPriority;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, prefix_cache,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  InferenceRequest cold_req;
  cold_req.model = cold_info.id;
  cold_req.prompt = "cold request prompt";
  cold_req.max_tokens = 2;

  InferenceRequest hot_req;
  hot_req.model = hot_info.id;
  hot_req.prompt = hot_prompt;
  hot_req.max_tokens = 2;

  auto cold_future = scheduler.Generate(std::move(cold_req));
  auto hot_future = scheduler.Generate(std::move(hot_req));

  auto cold_resp = cold_future.get();
  auto hot_resp = hot_future.get();
  REQUIRE_FALSE(cold_resp.no_backend);
  REQUIRE_FALSE(hot_resp.no_backend);

  REQUIRE(hot_backend->FirstSubmissionTicket() > 0);
  REQUIRE(cold_backend->FirstSubmissionTicket() > 0);
  REQUIRE(hot_backend->FirstSubmissionTicket() <
          cold_backend->FirstSubmissionTicket());
}
