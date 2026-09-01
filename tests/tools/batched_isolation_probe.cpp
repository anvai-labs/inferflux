/**
 * Batched-decode sequence isolation probe.
 *
 * Drives InferfluxCudaBackend::ExecuteUnifiedBatch directly with N sequences
 * — no HTTP server, no scheduler — and checks that decoding sequences
 * TOGETHER produces the same tokens as decoding each one ALONE (greedy, so
 * outputs must match exactly).
 *
 * This is the regression oracle for the cross-sequence contamination bug:
 * concurrent greedy requests with identical prompts produced 7 distinct
 * outputs (greedy + identical input must produce one), with one response
 * containing another's continuation verbatim. Server-level harnesses are
 * timing-noisy; this probe is single-threaded with fully controlled batch
 * metadata, so a failure here is reproducible and attributable.
 *
 * Phases:
 *   A (reference): each sequence prefilled and decoded ALONE (B=1 calls).
 *   B (batched):   all sequences prefilled in ONE call, then decoded
 *                  together in B=N calls.
 *   C (staggered): mixed batches — new sequences prefilled in the SAME call
 *                  while already-admitted sequences decode, mimicking real
 *                  concurrent arrival.
 *
 * Exit 0 = all phases isolated; exit 1 = at least one sequence diverged.
 */
#include "runtime/backends/cuda/inferflux_cuda_backend.h"
#include "runtime/backends/common/backend_types.h"
#include "runtime/backends/llama/llama_cpp_backend.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace inferflux {
namespace {

constexpr int kNumSequences = 8;
constexpr int kDecodeSteps = 32;
int g_num_sequences = kNumSequences; // argv[2] override

std::vector<std::string> MakePrompts() {
  // Distinctive per-sequence tags; the correct greedy continuation echoes
  // the tag, so losing it is an unambiguous corruption signal.
  std::vector<std::string> prompts;
  for (int i = 0; i < kNumSequences; ++i) {
    prompts.push_back("First repeat the code PR" + std::to_string(917 * (i + 3)) +
                      "X verbatim, then name the capital of France in one word.");
  }
  return prompts;
}

SamplingParams GreedyParams() {
  SamplingParams sp;
  sp.temperature = 0.0f;
  sp.top_k = 1;
  sp.seed = 0;
  if (std::getenv("PROBE_NO_PENALTY")) {
    sp.repetition_penalty = 1.000001f; // disarm implicit greedy 1.15
  }
  return sp;
}

UnifiedBatchInput MakeDecodeInput(int seq, int n_past, int token,
                                  uint64_t generation = 0) {
  UnifiedBatchInput input;
  input.sequence_id = seq;
  input.n_past = n_past;
  input.tokens = {token};
  input.request_logits = true;
  input.sampling = GreedyParams();
  input.request_id = seq;
  input.sequence_generation = generation;
  return input;
}

// Dump attention tensors captured during the most recent forward(s).
void DumpCapture(LlamaCppBackend *backend, const std::string &tag) {
  auto data = backend->CaptureAttentionTensors();
  std::printf("    [capture %s] ok=%d snapshots=%zu\n", tag.c_str(),
              data.ok ? 1 : 0, data.snapshots.size());
  if (!data.ok)
    return;
  const std::string dir = "/tmp/s20_capture_" + tag;
  std::string mkdir = "mkdir -p " + dir;
  if (system(mkdir.c_str()) != 0) {
    return;
  }
  for (const auto &snap : data.snapshots) {
    if (snap.layer_idx != 0)
      continue; // layer 0 only — first divergence localizes the bug
    const std::string path =
        dir + "/" + snap.operation + "_" + std::to_string(snap.layer_idx) +
        ".f32";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
      continue;
    std::fwrite(snap.data.data(), sizeof(float), snap.data.size(), f);
    std::fclose(f);
    std::printf("      %s shape=[%d,%d,%d]\n", path.c_str(),
                snap.shape.size() > 0 ? snap.shape[0] : 0,
                snap.shape.size() > 1 ? snap.shape[1] : 0,
                snap.shape.size() > 2 ? snap.shape[2] : 0);
  }
}

// Decode one sequence alone: single prefill call, then B=1 decode steps.
bool RunSingle(BackendInterface *backend,
               const std::vector<int> &prompt_tokens, int seq,
               std::vector<int> *out_tokens) {
  UnifiedBatchInput prefill;
  prefill.sequence_id = seq;
  prefill.n_past = 0;
  prefill.tokens = prompt_tokens;
  prefill.request_logits = true;
  prefill.sampling = GreedyParams();
  prefill.request_id = seq;
  auto outputs = backend->ExecuteUnifiedBatch({prefill});
  if (outputs.size() != 1 || !outputs[0].ok || outputs[0].token < 0) {
    std::fprintf(stderr, "single prefill seq %d failed\n", seq);
    return false;
  }
  int token = outputs[0].token;
  std::printf("    [dbg] single prefill seq %d -> token %d\n", seq, token);
  int n_past = static_cast<int>(prompt_tokens.size());
  out_tokens->clear();
  bool hit_cap = true;
  for (int step = 0; step < kDecodeSteps; ++step) {
    out_tokens->push_back(token);
    auto o = backend->ExecuteUnifiedBatch({MakeDecodeInput(seq, n_past, token)});
    if (o.size() != 1 || !o[0].ok || o[0].token < 0) {
      hit_cap = false;
      break; // EOS
    }
    token = o[0].token;
    n_past += 1;
  }
  if (hit_cap) {
    out_tokens->push_back(token); // match RunTogether's final flush
  }
  return true;
}

// All sequences prefilled in ONE call, then B=N decode steps.
bool RunTogether(BackendInterface *backend,
                 const std::vector<std::vector<int>> &tokenized,
                 std::vector<std::vector<int>> *out, int slot_offset = 0) {
  const int n = static_cast<int>(tokenized.size());
  std::vector<UnifiedBatchInput> prefills;
  for (int i = 0; i < n; ++i) {
    UnifiedBatchInput prefill;
    prefill.sequence_id = i + slot_offset;
    prefill.n_past = 0;
    prefill.tokens = tokenized[i];
    prefill.request_logits = true;
    prefill.sampling = GreedyParams();
    prefill.request_id = i;
    prefills.push_back(prefill);
  }
  auto outputs = backend->ExecuteUnifiedBatch(prefills);
  if (outputs.size() != static_cast<size_t>(n)) {
    std::fprintf(stderr, "batched prefill returned %zu outputs\n", outputs.size());
    return false;
  }
  std::vector<int> last(n), n_past(n);
  out->assign(n, {});
  for (int i = 0; i < n; ++i) {
    if (!outputs[i].ok || outputs[i].token < 0) {
      std::fprintf(stderr, "batched prefill seq %d failed\n", i);
      return false;
    }
    last[i] = outputs[i].token;
    n_past[i] = static_cast<int>(tokenized[i].size());
    std::printf("    [dbg] together prefill seq %d -> token %d\n", i,
                outputs[i].token);
  }
  for (int step = 0; step < kDecodeSteps; ++step) {
    std::vector<UnifiedBatchInput> inputs;
    std::vector<int> rows;
    for (int i = 0; i < n; ++i) {
      if (last[i] < 0) {
        continue;
      }
      inputs.push_back(MakeDecodeInput(i + slot_offset, n_past[i], last[i]));
      rows.push_back(i);
    }
    if (inputs.empty()) {
      break;
    }
    auto o = backend->ExecuteUnifiedBatch(inputs);
    if (o.size() != inputs.size()) {
      std::fprintf(stderr, "decode step %d: %zu outputs for %zu inputs\n", step,
                   o.size(), inputs.size());
      return false;
    }
    for (size_t k = 0; k < inputs.size(); ++k) {
      const int i = rows[k];
      // `last` was itself sampled as a visible token by the previous step, so
      // record it before checking this step's output — this matches RunSingle,
      // which pushes before decoding (otherwise batched runs end one token
      // short of the reference whenever EOS arrives).
      (*out)[i].push_back(last[i]);
      if (!o[k].ok || o[k].token < 0) {
        last[i] = -1;
        continue;
      }
      last[i] = o[k].token;
      n_past[i] += 1;
    }
  }
  for (int i = 0; i < n; ++i) {
    if (last[i] >= 0) {
      (*out)[i].push_back(last[i]);
    }
  }
  return true;
}

// Mixed batches: per step, one call containing decode rows for admitted
// sequences AND prefill rows for up to `admit_per_step` newcomers.
bool RunStaggered(BackendInterface *backend,
                  const std::vector<std::vector<int>> &tokenized,
                  int admit_per_step, std::vector<std::vector<int>> *out,
                  int slot_offset = 0) {
  const int n = static_cast<int>(tokenized.size());
  std::vector<int> n_past(n, 0);
  std::vector<int> last(n, -1);
  std::vector<int> steps(n, 0);
  std::vector<bool> admitted(n, false);
  out->assign(n, {});
  int admitted_count = 0;

  // NOTE: keep looping while any sequence is still decoding. The previous
  // condition (admitted_count < n) exited the moment the last sequence was
  // admitted, before the late joiners decoded a single step.
  for (int step = 0; step < kDecodeSteps + n + 4; ++step) {
    std::vector<UnifiedBatchInput> inputs;
    std::vector<int> decode_rows; // row k of `inputs` -> tokenized index
    for (int i = 0; i < n; ++i) {
      if (admitted[i] && last[i] >= 0 && steps[i] < kDecodeSteps) {
        inputs.push_back(MakeDecodeInput(i + slot_offset, n_past[i], last[i]));
        decode_rows.push_back(i);
      }
    }
    std::vector<int> prefill_rows; // appended after decode rows
    int admitting = 0;
    while (admitted_count < n && admitting < admit_per_step) {
      const int i = admitted_count;
      UnifiedBatchInput prefill;
      prefill.sequence_id = i + slot_offset;
      prefill.n_past = 0;
      prefill.tokens = tokenized[i];
      prefill.request_logits = true;
      prefill.sampling = GreedyParams();
      prefill.request_id = i;
      inputs.push_back(prefill);
      prefill_rows.push_back(i);
      ++admitting;
      ++admitted_count;
    }
    if (inputs.empty()) {
      break;
    }
    auto outputs = backend->ExecuteUnifiedBatch(inputs);
    if (outputs.size() != inputs.size()) {
      std::fprintf(stderr, "stagger step %d: %zu outputs for %zu inputs\n", step,
                   outputs.size(), inputs.size());
      return false;
    }
    // Outputs are index-aligned with inputs: decode rows first, then prefills.
    for (size_t k = 0; k < decode_rows.size(); ++k) {
      const int i = decode_rows[k];
      (*out)[i].push_back(last[i]); // visible token; matches RunSingle
      if (!outputs[k].ok || outputs[k].token < 0) {
        last[i] = -1; // EOS
        continue;
      }
      last[i] = outputs[k].token;
      n_past[i] += 1;
      steps[i] += 1;
    }
    for (size_t k = 0; k < prefill_rows.size(); ++k) {
      const int i = prefill_rows[k];
      const size_t idx = decode_rows.size() + k;
      if (!outputs[idx].ok || outputs[idx].token < 0) {
        continue;
      }
      admitted[i] = true;
      last[i] = outputs[idx].token;
      n_past[i] = static_cast<int>(tokenized[i].size());
      steps[i] = 0;
      std::printf("    [dbg] stagger prefill seq %d -> token %d\n", i,
                  outputs[idx].token);
    }
  }
  // Flush the final token of every still-active sequence, like RunTogether.
  for (int i = 0; i < n; ++i) {
    if (admitted[i] && last[i] >= 0) {
      (*out)[i].push_back(last[i]);
    }
  }
  return true;
}

// Membership-swap decode: all n prefilled together, then decoded together; at
// step `swap_step` sequence 0 departs and a newcomer with the SAME prompt
// prefills in the same call, so the input count stays constant while batch
// membership changes. This is the shape the decode relay's size-only guard
// mishandled (replayed with the departed row's seq_id/n_past) and the shape
// the server hits whenever one request finishes as another is admitted.
//
// Output rows: 0..n-1 as usual (row 0 is sequence 0's PARTIAL transcript,
// truncated at the swap — not comparable to the reference), plus row n =
// newcomer, whose reference is ref[0] (same prompt, greedy).
//
// reuse_departed_slot=true (op X): the newcomer is prefilled onto the
// DEPARTED sequence's slot with no intervening release — the state an unpolled
// deferred release leaves behind. The primary prefill path scrubs reused slots
// (ClearSequenceAsync); the lane path (overlap) has no scrub, so divergence
// with overlap on but not off proves stale-KV contamination on the lane path.
bool RunSwap(BackendInterface *backend,
             const std::vector<std::vector<int>> &tokenized,
             std::vector<std::vector<int>> *out, int slot_offset = 0,
             bool reuse_departed_slot = false) {
  const int n = static_cast<int>(tokenized.size());
  const int newcomer = n; // bookkeeping row index; slot chosen below
  const int newcomer_slot =
      reuse_departed_slot ? 0 + slot_offset : n + slot_offset;
  const uint64_t newcomer_generation = reuse_departed_slot ? 1u : 0u;
  std::vector<int> n_past(n + 1, 0);
  std::vector<int> last(n + 1, -1);
  std::vector<int> steps(n + 1, 0);
  std::vector<bool> active(n + 1, false);
  out->assign(n + 1, {});

  std::vector<UnifiedBatchInput> prefills;
  for (int i = 0; i < n; ++i) {
    UnifiedBatchInput prefill;
    prefill.sequence_id = i + slot_offset;
    prefill.n_past = 0;
    prefill.tokens = tokenized[i];
    prefill.request_logits = true;
    prefill.sampling = GreedyParams();
    prefill.request_id = i;
    prefills.push_back(prefill);
  }
  auto outputs = backend->ExecuteUnifiedBatch(prefills);
  if (outputs.size() != prefills.size()) {
    std::fprintf(stderr, "swap prefill returned %zu outputs\n", outputs.size());
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (!outputs[i].ok || outputs[i].token < 0) {
      std::fprintf(stderr, "swap prefill seq %d failed\n", i);
      return false;
    }
    active[i] = true;
    last[i] = outputs[i].token;
    n_past[i] = static_cast<int>(tokenized[i].size());
    std::printf("    [dbg] swap prefill seq %d -> token %d\n", i, outputs[i].token);
  }

  const int swap_step = 6;
  for (int step = 0; step < kDecodeSteps + 2; ++step) {
    std::vector<UnifiedBatchInput> inputs;
    std::vector<int> decode_rows; // -> sequence index in 0..n
    for (int i = 0; i <= n; ++i) {
      if (i == 0 && step >= swap_step) {
        continue; // departed at swap_step
      }
      if (active[i] && last[i] >= 0 && steps[i] < kDecodeSteps) {
        inputs.push_back(MakeDecodeInput(
            i == n ? newcomer_slot : i + slot_offset, n_past[i], last[i],
            i == n ? newcomer_generation : 0u));
        decode_rows.push_back(i);
      }
    }
    std::vector<int> prefill_rows;
    if (step == swap_step) {
      UnifiedBatchInput prefill;
      prefill.sequence_id = newcomer_slot;
      prefill.sequence_generation = newcomer_generation;
      prefill.n_past = 0;
      prefill.tokens = tokenized[0]; // same prompt as the departed sequence
      prefill.request_logits = true;
      prefill.sampling = GreedyParams();
      prefill.request_id = newcomer;
      inputs.push_back(prefill);
      prefill_rows.push_back(newcomer);
      std::printf(
          "    [dbg] swap step %d: seq 0 departs, newcomer slot %d prefills "
          "(reuse=%d gen=%llu)\n",
          step, newcomer_slot, reuse_departed_slot ? 1 : 0,
          static_cast<unsigned long long>(newcomer_generation));
    }
    if (inputs.empty()) {
      break;
    }
    auto o = backend->ExecuteUnifiedBatch(inputs);
    if (o.size() != inputs.size()) {
      std::fprintf(stderr, "swap step %d: %zu outputs for %zu inputs\n", step,
                   o.size(), inputs.size());
      return false;
    }
    for (size_t k = 0; k < decode_rows.size(); ++k) {
      const int i = decode_rows[k];
      (*out)[i].push_back(last[i]); // visible token; matches RunSingle
      if (!o[k].ok || o[k].token < 0) {
        last[i] = -1; // EOS
        continue;
      }
      last[i] = o[k].token;
      n_past[i] += 1;
      steps[i] += 1;
    }
    for (size_t k = 0; k < prefill_rows.size(); ++k) {
      const size_t idx = decode_rows.size() + k;
      if (!o[idx].ok || o[idx].token < 0) {
        continue;
      }
      active[newcomer] = true;
      last[newcomer] = o[idx].token;
      n_past[newcomer] = static_cast<int>(tokenized[0].size());
      steps[newcomer] = 0;
      std::printf("    [dbg] newcomer prefill -> token %d\n", o[idx].token);
    }
  }
  for (int i = 0; i <= n; ++i) {
    if (active[i] && last[i] >= 0) {
      (*out)[i].push_back(last[i]);
    }
  }
  return true;
}

// Release every sequence the way the scheduler does between requests, so
// slot reuse mirrors the server lifecycle (KV clear + penalty-history clear).
void ReleaseAll(inferflux::LlamaCppBackend *backend, int n, int slot_offset) {
  for (int i = 0; i < n; ++i) {
    auto fence = backend->BeginFreeSequence(i + slot_offset);
    // Poll to completion like the scheduler: a not-ready fence defers the KV
    // clear, and an unpolled pending release never clears at all.
    for (int tries = 0; tries < 200; ++tries) {
      if (backend->PollFreeSequence(fence)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    // Synchronous fallback so the release ALWAYS lands in this probe (the
    // server's scheduler polls the deferred path to completion instead).
    backend->FreeSequence(i + slot_offset);
  }
}

int CompareRuns(const std::vector<std::vector<int>> &reference,
                const std::vector<std::vector<int>> &batched, const char *label) {
  int diverged = 0;
  for (int i = 0; i < g_num_sequences; ++i) {
    const auto &a = reference[i];
    const auto &b = batched[i];
    const size_t m = std::min(a.size(), b.size());
    int first_diff = -1;
    for (size_t t = 0; t < m; ++t) {
      if (a[t] != b[t]) {
        first_diff = static_cast<int>(t);
        break;
      }
    }
    if (first_diff >= 0 || a.size() != b.size()) {
      ++diverged;
      std::printf(
          "  [FAIL] %s seq %d: diverged at token %d (ref len %zu, run len %zu)\n",
          label, i, first_diff, a.size(), b.size());
    }
  }
  std::printf("  %s: %d/%d sequences diverged\n", label, diverged, kNumSequences);
  return diverged;
}

} // namespace
} // namespace inferflux

using namespace inferflux;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
#ifdef _WIN32
  _putenv_s("INFERFLUX_DEBUG_ATTENTION_TENSORS", "1");
#else
  setenv("INFERFLUX_DEBUG_ATTENTION_TENSORS", "1", 1);
#endif

  if (argc > 2) {
    g_num_sequences = std::atoi(argv[2]);
    if (g_num_sequences < 1 || g_num_sequences > kNumSequences) {
      g_num_sequences = kNumSequences;
    }
  }

  auto backend = std::make_unique<InferfluxCudaBackend>();
  if (!backend) {
    std::fprintf(stderr, "backend factory returned null\n");
    return 2;
  }
  inferflux::LlamaBackendConfig config;
  config.ctx_size = 4096;
  config.batch_size = 16;
  config.max_parallel_sequences = 16;
  // Match the server's lane-overlap configuration when PROBE_OVERLAP=1 so the
  // probe exercises the lane path (server.cuda.yaml enables it; the struct
  // default is off, which is why earlier probe runs never touched lanes).
  // min_prefill_tokens=1 makes short probe prefills qualify for lane overlap.
  if (std::getenv("PROBE_OVERLAP")) {
    config.cuda_phase_overlap_scaffold = true;
    config.cuda_phase_overlap_min_prefill_tokens = 1;
  }
  if (!backend->LoadModel(model_path, config) || !backend->IsReady()) {
    std::fprintf(stderr, "LoadModel failed\n");
    return 3;
  }

  const auto prompts = MakePrompts();
  std::vector<std::vector<int>> tokenized(g_num_sequences);
  for (int i = 0; i < g_num_sequences; ++i) {
    tokenized[i] = backend->TokenizeForCache(prompts[i]);
    if (tokenized[i].empty()) {
      std::fprintf(stderr, "tokenize failed for seq %d\n", i);
      return 3;
    }
  }

  // Transition-matrix mode: argv[3] is a sequence of ops.
  // T=together slots 0.., U=together virgin 4.., S=stagger slots 0.., V=stagger virgin 4..
  if (argc > 3) {
    const std::string script = argv[3];
    std::vector<std::vector<int>> ref(g_num_sequences);
    for (int i = 0; i < g_num_sequences; ++i) {
      if (!RunSingle(backend.get(), tokenized[i], i, &ref[i])) {
        return 4;
      }
    }
    ReleaseAll(backend.get(), g_num_sequences, 0);
    for (char op : script) {
      // Release before every op so each starts from the server-like state
      // (slot released between requests). Without this, an armed penalty run
      // of S/V after T compares against prefills biased by T's history — an
      // artifact of the script, not contamination. X's in-op reuse (departed
      // slot re-prefilled without release) is unaffected. Cover the offset
      // slots too (U/V run at slot_offset=4).
      ReleaseAll(backend.get(), g_num_sequences + 4, 0);
      std::vector<std::vector<int>> run;
      bool ok = false;
      const char *label = "?";
      if (op == 'T') {
        ok = RunTogether(backend.get(), tokenized, &run, 0);
        label = "T(together,slots0)";
      } else if (op == 'U') {
        ok = RunTogether(backend.get(), tokenized, &run, 4);
        label = "U(together,slots4)";
      } else if (op == 'S') {
        ok = RunStaggered(backend.get(), tokenized, 2, &run, 0);
        label = "S(stagger,slots0)";
      } else if (op == 'V') {
        ok = RunStaggered(backend.get(), tokenized, 2, &run, 4);
        label = "V(stagger,slots4)";
      } else if (op == 'W') {
        ok = RunSwap(backend.get(), tokenized, &run, 0);
        label = "W(swap,slots0)";
      } else if (op == 'X') {
        ok = RunSwap(backend.get(), tokenized, &run, 0, /*reuse_departed_slot=*/true);
        label = "X(swap-reuse-stale-slot0)";
      }
      if (!ok) {
        std::printf("op %c: RUN FAILED\n", op);
        continue;
      }
      int div = 0;
      const int rows = static_cast<int>(run.size());
      const bool swap_op = (op == 'W' || op == 'X');
      for (int i = 0; i < rows; ++i) {
        if (swap_op) {
          if (i == 0) {
            continue; // sequence 0's transcript is truncated at the swap
          }
          if (i == g_num_sequences) {
            std::printf("    [newcomer row, reference = ref[0]]\n");
          }
        }
        const auto &a = (swap_op && i == g_num_sequences) ? ref[0] : ref[i];
        const auto &b = run[i];
        if (a == b) {
          continue;
        }
        ++div;
        const size_t m = std::min(a.size(), b.size());
        int first_diff = -1;
        for (size_t t = 0; t < m; ++t) {
          if (a[t] != b[t]) {
            first_diff = static_cast<int>(t);
            break;
          }
        }
        std::printf("    seq %d: first_diff=%d ref_len=%zu run_len=%zu\n", i,
                    first_diff, a.size(), b.size());
      }
      std::printf("op %s: diverged %d/%d %s\n", label, div, rows,
                  div ? "FAIL" : "PASS");
    }
    return 0;
  }

  // Reference capture: prefill seq 0 once on a scratch slot, capture.
  {
    std::vector<int> scratch;
    if (!RunSingle(backend.get(), tokenized[0], /*seq=*/12, &scratch)) {
      return 4;
    }
    ReleaseAll(backend.get(), 1, /*slot_offset=*/12);
  }
  DumpCapture(backend.get(), "ref_prefill");

  // Phase A: B=1 references.
  std::vector<std::vector<int>> reference(g_num_sequences);
  for (int i = 0; i < g_num_sequences; ++i) {
    if (!RunSingle(backend.get(), tokenized[i], i, &reference[i])) {
      return 4;
    }
  }
  std::printf("phase A (B=1 references): done\n");

  ReleaseAll(backend.get(), g_num_sequences, 0);

  int failures = 0;

  {
    std::vector<std::vector<int>> shifted = tokenized;
    std::vector<std::vector<int>> together;
    // DIAGNOSTIC: fresh slots (8+i) never touched by phase A.
    if (!RunTogether(backend.get(), tokenized, &together)) {
      return 4;
    }
    failures += CompareRuns(reference, together, "phase B (slots 0..N)");
  }
  {
    std::vector<std::vector<int>> staggered;
    if (!RunStaggered(backend.get(), tokenized, /*admit_per_step=*/2, &staggered)) {
      return 4;
    }
    failures += CompareRuns(reference, staggered, "phase C (staggered x2)");
  }

  ReleaseAll(backend.get(), g_num_sequences, 0);

  {
    std::vector<std::vector<int>> fresh;
    if (!RunTogether(backend.get(), tokenized, &fresh,
                     /*slot_offset=*/kNumSequences)) {
      return 4;
    }
    failures += CompareRuns(reference, fresh, "phase B2 (FRESH slots 8..)");
  }
  {
    std::vector<std::vector<int>> fresh;
    if (!RunStaggered(backend.get(), tokenized, /*admit_per_step=*/2, &fresh,
                      /*slot_offset=*/kNumSequences)) {
      return 4;
    }
    failures += CompareRuns(reference, fresh, "phase C2 (FRESH slots 8..)");
  }
  ReleaseAll(backend.get(), g_num_sequences, kNumSequences);
  {
    // Phase C3: stagger after batched decode, on VIRGIN slots (4+i).
    std::vector<std::vector<int>> fresh;
    if (!RunStaggered(backend.get(), tokenized, /*admit_per_step=*/2, &fresh,
                      /*slot_offset=*/4)) {
      return 4;
    }
    failures += CompareRuns(reference, fresh, "phase C3 (VIRGIN slots 4..)");
  }
  ReleaseAll(backend.get(), g_num_sequences, 4);

  std::printf("%s (%d divergent sequences total)\n",
              failures ? "RESULT: FAIL" : "RESULT: PASS", failures);
  return failures ? 1 : 0;
}
