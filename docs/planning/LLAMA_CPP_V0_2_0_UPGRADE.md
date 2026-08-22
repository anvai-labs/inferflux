# llama.cpp v0.2.0 Upgrade Compatibility Delta

**Status:** Ready for isolated implementation

**Baseline:** `6c97bffd6508f4999d5bc292addd4f433a3648bc` (`b8213`)

**Target:** `bb4caa7540188872173c44d161602d9271386413` (`v0.2.0`)

## Scope and Risk

The target is 2,353 commits ahead of the baseline. Across the interfaces and
accelerator sources InferFlux consumes, it changes 277 files (+53,409/-18,510
lines). Treat this as a runtime migration, not a routine submodule bump. Keep
Catch2, nlohmann/json, MLX-C, CUDA, and ROCm upgrades out of the same change.

## Known Adaptations

| Area | Observed delta | Required change |
|---|---|---|
| Sampling | `llama_sampler_init_penalties` adds leading `n_vocab` | Pass `llama_vocab_n_tokens(vocab_)`; retain penalty behavior tests |
| Multimodal | bitmap helpers return a wrapper, add `placeholder`, and evaluation callbacks change | Stop compiling mtmd internals directly; enable `LLAMA_BUILD_MTMD` and adapt the image bridge to the public target/API |
| Common library | upstream CMake target is now `llama-common`; vendor setup moved | Keep the JSON-schema shim explicit or link `llama-common`, but never compile two copies of the converter |
| Structured output | JSON-schema grammar generation changed whitespace, pattern, reference, and numeric handling | Run schema/grammar golden tests and agent-contract cases; output differences need review, not snapshot acceptance |
| CUDA/HIP | large GGML kernel rewrite; RDNA4 MMQ configuration added; optional NCCL/RCCL appeared | Keep `sm_89` and `gfx1201` explicit, collect provider identity, memory, correctness, and throughput evidence |
| Context/KV | llama context and memory/state interfaces evolved | Complete TD-007 capacity/admission exposure before comparing load results |

## Isolated Change Sequence

1. Create a clean worktree from the validated baseline and update only the
   llama.cpp gitlink plus manifest ref/digest.
2. Configure CPU with `ENABLE_MTMD=OFF`; fix public llama API compilation and
   pass the complete model-free suite.
3. Validate structured output, chat templates, logit bias, penalties, stop/EOG,
   prefix copy, sequence serialization, and cancellation with a GGUF model.
4. Build and run CUDA `sm_89`, then ROCm `gfx1201`, on the serialized GPU runner.
5. Compare startup, output validity, peak memory, TTFT, and throughput against
   retained logs. Reject unexplained correctness/fallback regressions; record
   performance deltas rather than silently retuning floors.
6. Adapt and validate `ENABLE_MTMD=ON` separately after the text runtime is
   green. Roll back by restoring the baseline gitlink and manifest entry.

## Merge Gate

Require clean CPU/CUDA/ROCm builds, all model-free tests, 32/32 native CUDA and
16/16 ROCm behavioral requests, explicit provider identity without fallback,
and reviewed structured-output differences. Do not combine a ROCm 7.14 host
upgrade with this migration.
