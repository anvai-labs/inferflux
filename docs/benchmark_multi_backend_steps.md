# Multi-backend Benchmark Checklist

This doc captures the repeatable steps and instrumentation that keep `run_gguf_comparison_benchmark.sh` trustworthy while the native row-pair kernels run alongside `llama.cpp`.

## 1. Flags and defaults
* `INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWPAIR_W4` now defaults to `false` in `NativeExecutionPolicy`. Keep it opt-in for controlled experiments only; exact-shape isolated benchmarking on Ada RTX 4000 showed the `M=2,N=11008,K=2048` row-pair FFN kernel was numerically clean but slower than the generic grouped path.
* Keep `INFERFLUX_ENABLE_BATCHED_DECODE=1` in the benchmark so multi-row decode batches naturally occur and exercise the row-pair operator per the metrics below.
* `INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT=1` is an experimental scheduler knob only. Keep default benchmarking on `wait=0`; use `wait=1` only as an A/B comparison because the effect is workload-sensitive and not stable enough for default serving policy.

## 2. Harness contract
`run_gguf_comparison_benchmark.sh` now supports multi-concurrency sweeps from one invocation:

* default prompt set: 16 longer “real usage” prompts
* default matrix: `CONCURRENCY=1,4,8`
* default requests: `NUM_REQUESTS=16`

Per-concurrency artifacts are intentionally isolated. Expect files such as:

* `responses_inferflux_cuda_c1/`, `responses_inferflux_cuda_c4/`, `responses_inferflux_cuda_c8/`
* `stats_inferflux_cuda_c1.json`
* `metrics_inferflux_cuda_c4.txt`
* `admin_cache_inferflux_cuda_c8.json`
* `similarity_c1.json`, `similarity_c4.json`, `similarity_c8.json`

If these files are being overwritten across concurrency levels, the harness is broken and the benchmark should not be trusted.

`benchmark_multi_backend_comparison.sh` now runs each backend in an isolated child invocation. That is intentional:

1. the parent process only prepares the suite, dispatches one backend at a time, and merges results
2. each child process starts, benchmarks, and tears down exactly one backend
3. local CUDA backends get a `cudaDeviceReset()` between children

This avoids cross-backend allocator / stream / shell-job state leaking from one engine into the next.

For targeted debugging, set `INFERFLUX_BENCH_SINGLE_BACKEND=<backend_id>` and run the same script directly.

## 3. Run order and reset hook
The benchmark runs `inferflux_cuda` first, then `llama_cpp_cuda`. To avoid CUDA state leaking between the two:
1. The script already calls `stop_server inferflux_cuda` once the InferFlux CUDA run is done.
2. We added `reset_cuda_device()` which issues `cudaDeviceReset()` (via `libcudart`) immediately after native shutdown. That ensures the GPU context is fully torn down before the llama.cpp start.
3. Only then does the script launch `llama_cpp_cuda`; the 3-second sleep after the reset gives the GPU a final breathing room.
4. The script also traps exit and runs the same cleanup path so aborted runs free the active server and leave GPU state predictable for the next benchmark.

If you ever replicate the benchmark manually, follow the same order: stop native, reset the CUDA device (via `cudaDeviceReset()` or `./build-cuda/inferfluxd --reset-cuda` if available), then start the llama.cpp backend. This guarantees accurate throughput isolation for regression comparisons.

## 4. Metrics to validate operator and scheduler behavior
* Inspect `inferflux_cuda_rowpair_selection_total{phase="decode",operator="q8_1_group_row_pair_w4",bucket="2"}` and `...operator="q8_1_gemv_row_pair"` in the resulting `metrics_inferflux_cuda_c*.txt`. Successful runs record counts (>0) in bucket `2` or `3_4`, proving the specialized operators handled the multi-row batches.
* The benchmark also captures `inferflux_cuda_ffn_proj_operator_total` and `inferflux_cuda_down_proj_operator_total` summaries (written to `inferflux_cuda_ffn_proj_summary_inferflux_cuda_c*.json` and `inferflux_cuda_operator_summary_inferflux_cuda_c*.json`) so you can correlate which kernels were chosen.
* Every InferFlux backend run now also captures `/v1/admin/cache` into `admin_cache_<backend>_c*.json`. The corresponding `stats_<backend>_c*.json` embeds that data under `cache_snapshot` and `memory_snapshot`, including:
  * `memory_snapshot.inferflux_cuda_model`
  * `memory_snapshot.inferflux_cuda_kv`
  * `memory_snapshot.paged_kv`
* The multi-backend CSV export now carries the key memory fields alongside throughput so concurrency runs can be compared on both tok/s and memory state.
* The decode-worker sticky-merge counters (`inferflux_scheduler_decode_worker_sticky_merge_total`, `inferflux_scheduler_decode_worker_sticky_merged_requests_total`) are the intended validation signal for `INFERFLUX_ENABLE_STICKY_DECODE_ACCUMULATION_WAIT`, but benchmark-side metric capture still needs follow-up because those lines are visible in direct `/metrics` scrapes yet have not been reliable in the saved benchmark snapshots.

## 5. Accuracy safeguards
* The similarity report is now per concurrency (`similarity_c*.json`). Treat the whole sweep as invalid if only one concurrency level produces similarity output; that indicates the harness wiped earlier response artifacts.
* Keep `INFERFLUX_DEBUG_OPERATOR_SELECTION=0`/`INFERFLUX_DEBUG_LOGITS=0` for normal benchmarks; enable them only for debugging because they add logging noise.

## 6. Experimental sticky wait status
Prompt-heavy Qwen2.5-3B Q4_K_M benchmark matrix on Ada RTX 4000 (`16` requests, `64` max tokens, `1/4/8` concurrency):

* `wait=0`
  * `c=1`: native `81.6 tok/s`, llama.cpp `111.6 tok/s` (`0.73x`)
  * `c=4`: native `139.9 tok/s`, llama.cpp `208.2 tok/s` (`0.67x`)
  * `c=8`: native `158.7 tok/s`, llama.cpp `312.0 tok/s` (`0.51x`)
* `wait=1`
  * `c=1`: native `81.9 tok/s`, llama.cpp `108.6 tok/s` (`0.75x`)
  * `c=4`: native `142.6 tok/s`, llama.cpp `203.1 tok/s` (`0.70x`)
  * `c=8`: native `163.2 tok/s`, llama.cpp `305.0 tok/s` (`0.54x`)

Interpretation:

* `wait=1` is not universally regressive, but the gain is modest and workload-specific.
* Keep it available for benchmark matrices.
* Do not treat it as the recommended default scheduler policy.

## 7. Release-note checklist
When promoting the row-pair flag for release:
* Update client-facing docs (this file) and point to the new metric so operators can verify row-pair usage.
* Mention that `llama_cpp_cuda` now runs against a clean GPU thanks to the reset hook—this avoids the sporadic `socket: Operation not permitted` issues that plagued earlier runs.
* Leave the instrumentation (metrics_capture hooks in the benchmark) so any regression gate re-running this benchmark automatically records operator breakdown, row-pair counters, and similarity data.

Current release posture:
* Keep the proven `Q4_K M=1` grouped hot path on by default.
* Keep `q8_1_group_row_pair_w4` behind `INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWPAIR_W4=1` until a future implementation beats the generic grouped path on the exact live `M=2,N=11008,K=2048` envelope.

## 8. Local vLLM / SGLang safetensors runs

`benchmark_multi_backend_comparison.sh` can now auto-launch local `vllm` and `sglang` servers one at a time so their VRAM is released before the next backend starts.

Recommended local safetensors setup:

```bash
AUTOSTART_VLLM=true \
AUTOSTART_SGLANG=true \
VLLM_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
SGLANG_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
VLLM_LAUNCH_ARGS="--dtype half --max-model-len 2048" \
SGLANG_LAUNCH_ARGS="--dtype half --context-length 2048" \
BUILD_DIR=./build-cuda \
./scripts/benchmark.sh multi-backend \
  models/qwen2.5-3b-instruct-safetensors
```

Notes:

* The harness auto-detects the supplied model format and skips incompatible backends.
* GGUF inputs skip `vllm` / `sglang`; safetensors inputs skip `llama_cpp_cuda` / `ollama`.
* `vllm` and `sglang` should be benchmarked with safetensors/Hugging Face model directories via `VLLM_MODEL_PATH` / `SGLANG_MODEL_PATH`.
* If you already run those servers elsewhere, leave `AUTOSTART_VLLM` / `AUTOSTART_SGLANG` unset and point `VLLM_HOST` / `SGLANG_HOST` at the existing endpoints.
