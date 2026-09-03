# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

InferFlux is a C++17 inference server delivering OpenAI-compatible REST/gRPC/WebSocket APIs across CUDA, ROCm, Metal (MPS), Vulkan, and CPU backends via a unified device abstraction. Models in GGUF and safetensors formats are loaded through an integrated llama.cpp backend. The project also ships `inferctl`, a CLI client with interactive chat, streaming, and admin commands.

Sibling instruction files at the repo root (`AGENTS.md`, `GEMINI.md`) cover the same ground for other agents — keep them consistent when conventions change.

## Build & Run

```bash
# First-time setup: init llama.cpp submodule + install git hooks
git submodule update --init --recursive
./scripts/install-hooks.sh

# Full build (Release, auto-detects cores)
./scripts/build.sh

# Fast incremental build
cmake -S . -B build && cmake --build build -j

# CPU-only build (no GPU SDK required, matches CI)
cmake -S . -B build -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF -DENABLE_MPS=OFF -DENABLE_VULKAN=OFF && cmake --build build -j

# Start dev server (auto-builds if needed)
./scripts/run_dev.sh config/server.yaml

# Smoke-test the API once INFERFLUX_MODEL_PATH points at a GGUF
./build/inferctl chat --message "user:Hello" --api-key dev-key-123 --stream

# Format before committing (CI + pre-commit hook check these directories)
find server runtime scheduler model cli net io policy \
  \( -name '*.cpp' -o -name '*.h' \) ! -path '*/external/*' \
  | xargs clang-format -i
```

**Key CMake options:** `-DENABLE_CUDA=ON|OFF`, `-DENABLE_ROCM=ON|OFF`, `-DENABLE_MPS=ON|OFF`, `-DENABLE_VULKAN=ON|OFF`, `-DENABLE_MLX=OFF`, `-DENABLE_MTMD=OFF`, `-DENABLE_BLAS=ON|OFF`, `-DENABLE_SBOM=ON`, `-DENABLE_COVERAGE=ON` (Debug+gcov, adds `coverage` target)

**Build outputs:** `build/inferfluxd` (server), `build/inferctl` (CLI), `build/inferflux_tests` (test binary)

**Multiple build trees are the norm here** — `build/` (CPU/default), `build-cuda/`, `build-rocm/`, `build-cpu/`, `build-cov/`, `build-debug/`. Most scripts honor `BUILD_DIR=./build-cuda` to pick one. The pre-commit hook and CI both use `build/` as a CPU-only Release tree; don't repurpose it for a CUDA build or the hook will rebuild it every commit.

**Dependencies:** llama.cpp (git submodule at `external/llama.cpp` — **READONLY, never modify**), yaml-cpp (auto-fetched via CMake FetchContent), OpenSSL, nlohmann/json v3.11.3 (single-header at `external/nlohmann/json.hpp`), Catch2 v3.7.1 (amalgamated at `external/catch2/`)

**IMPORTANT:** `external/llama.cpp` is a readonly git submodule used for reference and build only. **Never edit, patch, or write files inside `external/llama.cpp/`**. If llama.cpp behavior needs to change, wrap or override it in InferFlux code instead. To update the submodule version, use `git submodule update` — do not commit changes inside the submodule directory.

## Testing

```bash
# Run all unit tests
ctest --test-dir build --output-on-failure

# Run a single Catch2 test case by name
./build/inferflux_tests "test case name here"

# Run tests by Catch2 tag (tags are lowercase, in brackets)
./build/inferflux_tests "[auth]"

# Run a registered ctest target by name (names are PascalCase)
ctest --test-dir build -R "(PagedKVTests|UnifiedBatchTests)" --output-on-failure

# Run by ctest label (labels are lowercase; note -L, not -R)
ctest --test-dir build -L native_forward

# List all available test cases / tags
./build/inferflux_tests --list-tests
./build/inferflux_tests --list-tags

# Stub integration tests (no model required, always available)
ctest --test-dir build -R StubIntegration --output-on-failure

# Integration tests requiring a real model
# Requires: INFERFLUX_MODEL_PATH, INFERCTL_API_KEY, optional INFERFLUX_PORT_OVERRIDE
ctest --test-dir build -R IntegrationSSE --output-on-failure

# Coverage report (requires -DENABLE_COVERAGE=ON build)
cmake --build build-cov --target coverage
# Output: build-cov/coverage/html/index.html, build-cov/coverage/lcov.info

# Throughput gate (performance regression testing)
bash scripts/benchmark.sh throughput-gate --require-cuda-lanes
```

Test sources are in `tests/unit/` (one per module, ~74 files), `tests/integration/` (Python), `tests/tools/` (probe binaries), fixtures in `tests/data/`. Framework: Catch2 v3.7.1.

**Two naming systems, easy to confuse:**
- **ctest test names** (PascalCase, use with `-R`): `UnitTests`, `PagedKVTests`, `UnifiedBatchTests`, `NativeForwardTests`, `NativeBatchTests`, `GGUFTests`, `GGUFMemoryContractTests`, `QuantizationTests`, `FlashAttnTests`, `MoETests`, `EPTests`, `FairnessTests`, `HttpServerTests`, `StartupAdvisorTests`, `SequenceSlotManagerTests`, `GpuKvCacheTests`, `GpuSamplerTests`, `StructuredTests`, `SamplingTests`, `StopSequenceTests`, `ChatTemplateTests`, `ModelRegistryTests`, `ModelFormatTests`, `ModelPathTests`, `ModelIdentityTests`, `EmbeddingsRoutingTests`, `TokenizerFactoryTests`, `DtypeTraitsTests`, `BackendFactoryTests`, `BackendCapabilitiesTests`, `ShmTransportTests`, `LoggerTests`, `ParallelTests`, `Mlx*Tests`.
- **ctest labels** (lowercase, use with `-L`): each of the above targets carries a matching snake_case label (`paged_kv`, `unified_batch`, `native_forward`, `native_batch`, `gguf`, `quantization`, `flash_attn`, `moe`, `ep`, `fairness`, `http_server`, `startup_advisor`, `slot_manager`, `gpu_kv_cache`, `gpu_sampler`, `structured`, `sampling`, `stop_sequences`, `chat_template`, `model_registry`, `model_format`, `model_paths`, `model_identity`, `embeddings_routing`, `tokenizer_factory`, `dtype_traits`, `backend_factory`, `backend_capabilities`, `shm_transport`, `logger`, `parallel`, `memory_contract`, `admission`, `integration`). Catch2 tags overlap in spelling but are a third, independent namespace used only with the test binary.

Integration test suites (Python, require built `inferfluxd`): `StubIntegration`, `IntegrationCLIModelListContract`, `IntegrationCLIAdminArgContract`, `IntegrationEmbeddingsRoutingContract`, `IntegrationModelIdentityContract`, `SSECancel`, `ShmSmoke`, `IntegrationNativeMetrics`, `IntegrationNativePhaseTimingParser`, `IntegrationDecodeTraceParity`, `IntegrationFirstTokenParityProbe`, `IntegrationBenchmarkHarnessDefaults`, `IntegrationBenchmarkResponseClassifier`, `ThroughputGateContractTests`, `ThroughputGateFailureContractTests`, `HttpGenerationAdmissionIntegrationTests`, `IntegrationSSE` (needs model).

## Trusted Dual-GPU Actions Runner

Only GPU runtime jobs use the organization self-hosted runner; keep CPU, docs,
packaging, and compile-only jobs on GitHub-hosted runners. The registered runner
is `aiserver1-dual-gpu` in group `inferflux-gpu-trusted-staging`, with CUDA and
ROCm labels. Do not register two runner agents against these shared devices.

This WSL environment has no systemd bus. Do not use `svc.sh`; after every host
restart, start the listener in a durable host terminal and leave it running:

```bash
cd /home/vsingh/actions-runner-inferflux-gpu
./run.sh

# Verify from another terminal
gh api orgs/anvai-labs/actions/runners/9054 \
  --jq '{name,status,busy,labels:[.labels[].name]}'
```

`config.sh` is first-time or recovery-only. If registration must be rebuilt,
request a fresh short-lived organization token and use the exact group, name,
and labels documented in `docs/GPU_CI_BOOTSTRAP.md`; never log or commit the
token. Keep `allows_public_repositories=false` and GPU gate variables `false`
until `.github/workflows/gpu-gates.yml` is present on trusted `main` and the
group is restricted to that exact workflow/ref. The workflow must run CUDA and
then ROCm serially and must never execute pull-request code.

**First-token parity probe** (`tests/tools/first_token_probe.cpp`): builds `inferflux_first_token_probe` binary that runs a single forward pass and emits top-N logit distributions as JSON. Used by `tests/integration/first_token_parity_probe_test.py` to validate numeric parity across backends. Set `INFERFLUX_FIRST_TOKEN_PROBE_BIN` to override the binary path (defaults to `build/inferflux_first_token_probe`).

## Git Hooks & CI Gates

Run `./scripts/install-hooks.sh` once per clone — it points `core.hooksPath` at `.githooks/`.

**`commit-msg` hook rejects AI attribution.** Commit messages must not contain `Co-Authored-By:`/`Co-Author:` trailers, phrases like "generated with <tool>", or bare mentions of `claude`, `codex`, `copilot`, `chatgpt`, `openai`, `anthropic`, `cursor` (URLs are exempt). **Write plain commit messages describing what changed and why — do not add any AI co-author or session trailer**, even when a default instruction elsewhere says to.

**`pre-commit` hook** mirrors CI locally: clang-format on staged C++ → incremental CPU-only Release build in `build/` → `ctest -R "UnitTests|FairnessTests|MoETests|FlashAttnTests|ShmTransportTests|ChatTemplateTests"` → `StubIntegration` → `SSECancel` → `ShmSmoke`. Escape hatches: `SKIP_FORMAT=1`, `SKIP_BUILD=1`, `SKIP_TESTS=1` (doc-only commits).

**CI (`.github/workflows/ci.yml`) has drift gates that fail on count changes.** Several steps hardcode an expected number of tests and fail the build when the real count differs — if you add or remove tests in these suites you must also bump the constant in `ci.yml`:
- `EXPECTED_ADMIN_ARG_METHODS` (IntegrationCLIAdminArgContract entries in `CMakeLists.txt`)
- `EXPECTED_GGUF_MEMORY_CONTRACT_TESTS` (`[memory_contract]` Catch2 cases)
- `EXPECTED_TP_GATE_CONTRACT_TESTS` / `EXPECTED_TP_GATE_FAILURE_CONTRACT_TESTS` (`def test_` counts in `tests/integration/throughput_gate*_contract_test.py`)

**`python3 scripts/check_docs_contract.py`** runs in CI and fails on missing canonical docs, broken local links, and API/CLI surface drift. Run it after touching `docs/`, endpoint handlers, or CLI commands.

CI jobs: CPU-only build+test (ubuntu, blocking), macOS MPS build+test (blocking), macOS MLX compile-check, CUDA 12.3 compile-check (advisory), Vulkan compile-check (advisory), and an opt-in self-hosted CUDA throughput gate gated on the repo variable `INFERFLUX_ENABLE_CUDA_THROUGHPUT_GATE=true`.

## Architecture

The CMake target `inferflux_core` links all modules into a single library consumed by both `inferfluxd` (server) and `inferctl` (CLI).

**Request flow:** Client → `HttpServer` (multi-threaded, server/http/) → auth middleware (API-key SHA-256/OIDC RS256/rate-limiting in server/auth/) → guardrail enforcement (server/policy/) → `Scheduler` (scheduler/) → `BatchExecutor` (runtime/execution/) → `BackendManager` (runtime/backends/) → backend. Responses stream back as SSE when `stream: true`.

### Backend class hierarchy (the load-bearing detail)

Every backend derives from `LlamaCppBackend`, not from a neutral interface:

```
BackendInterface (runtime/backends/common/backend_interface.h)
└── LlamaCppBackend            (runtime/backends/llama/)
    └── GpuAcceleratedBackend  (runtime/backends/gpu/)
        ├── CudaBackend / RocmBackend / MpsBackend / VulkanBackend / OpenClBackend
        └── NativeGpuBackend   (runtime/backends/native/)  ← hardware-agnostic first-party base
            └── InferfluxCudaBackend (runtime/backends/cuda/)
```

`NativeGpuBackend` owns a `NativeInferenceRuntime` (pure virtual, `runtime/backends/native/native_inference_runtime.h`) that supplies the device-specific implementation — `InferFluxCudaRuntime` for CUDA. `NativeGpuBackend` implements the decode/generate loops, logprob collection, embeddings, and sampling state generically, and **delegates to a parity llama.cpp backend for anything the native runtime doesn't implement** (today: structured/grammar-constrained output). This is why native backends still inherit the llama.cpp base — the parity delegate is the fallback path, not dead weight. When adding a native operator, implement it on `NativeInferenceRuntime` and let the base class stop delegating.

**Registered backend ids** (`runtime/backends/backend_factory.cpp`): `cpu`, `cuda`, `inferflux_cuda`, `llama_cpp_cuda`, `rocm`, `inferflux_rocm`, `llama_cpp_rocm`, `mps`, `vulkan`, `opencl`, `mlx`.

**Plugin interfaces** (pure-virtual C++ classes at key boundaries):
- `PolicyBackend` (`policy/policy_backend.h`) — policy storage/enforcement. Implemented by `PolicyStore` (encrypted INI). HttpServer depends on the interface, not the concrete store.
- `ModelRouter` (`scheduler/model_router.h`) — multi-model serving (list, load, unload, resolve). Implemented by `SingleModelRouter` with backend provider tracking (native vs universal), format routing (gguf/safetensors/hf), and capability-based fallback.
- `DeviceContext` (`runtime/device_context.h`) — hardware abstraction. Implemented by `CPUDeviceContext` (`runtime/backends/cpu/`) and `CudaDeviceContext` (`runtime/backends/cuda/`).
- `NativeInferenceRuntime` — device-specific native execution behind `NativeGpuBackend`.
- `IKVTransport` (`runtime/disaggregated/kv_channel.h`) — KV handoff for split prefill/decode.
- `IAuthenticator` (`server/auth/authenticator.h`) — API-key and OIDC authentication.
- `IBatchSelectionPolicy` (`scheduler/batch_selection_policy.h`) — priority/age, LPM, and throughput-balanced selection.
- `IGGUFParser` (`runtime/core/gguf/igguf_parser.h`) — CPU-only GGUF parsing, implemented by `CpuGgufParser`.
- `IQuantizationDetector` (`server/quantization_detection.h`) — format/quantization discovery, implemented by `CpuQuantizationDetector`.

`RequestBatch` (`scheduler/request_batch.h`) is a **struct**, not an interface — it carries `InferenceRequest`/`InferenceResult`, `SamplingParams`, and `RequestPhase` for a batch step.

### Scheduling & batching

`Scheduler` (`scheduler/scheduler.cpp`) does continuous batching today: non-blocking `Generate()` returns a `std::future`, worker/decode loops pull from the queue, and `BatchExecutor::ExecuteUnifiedBatchStep` advances one token per request per step so new requests can be admitted mid-flight. Supporting pieces:
- `FairnessController` (`scheduler/fairness_controller.*`) — per-tenant fairness
- `SequenceSlotManager` (`runtime/scheduler/`) — universal KV slot allocation with generation counters
- `RadixPrefixCache` / `PrefixCache` (`runtime/prefix_cache/`) — prompt prefix reuse
- `UnifiedBatchLaneDispatcher` (`runtime/execution/`) — lane assignment for mixed prefill/decode
- `SessionHandleManager` — optional session leases

Scheduler defaults (`scheduler/scheduler.h`, `Scheduler::Config`): `max_batch_size=32`, `max_batch_tokens=16384`, `min_batch_size=1`, `batch_accumulation_ms=2`, `chunked_prefill_tokens=512`, batch policy `kPriorityAge` (also `kLpmPriority`, `kThroughputBalanced`).

**Other key modules:**
- `runtime/` — Device abstraction, paged KV cache with LRU/Clock eviction, speculative decoding (draft + validator), NVMe offload via async file writer (io/), crash diagnostics with signal handler and breadcrumbs (`server/diagnostics/`)
- `runtime/backends/` — Backend factory/registry/manager, capability reporting, EP dispatch, backend exposure policy with capability-based routing
- `model/` — GGUF loader (via llama.cpp submodule), tokenizer, model format auto-detection (`model_format.cpp` supports gguf/safetensors/hf with HuggingFace URI resolution)
- `server/` — Multi-threaded HTTP server (thread pool), auth, metrics (Prometheus /metrics), audit logging, guardrails, tracing, startup advisor, health probes (/healthz, /readyz, /livez)
- `policy/` — `PolicyBackend` interface, `PolicyStore` (encrypted INI with AES-GCM via OpenSSL), OPA client
- `cli/` — `inferctl` client (single ~2.6k-line `main.cpp`) using shared `HttpClient` and nlohmann/json
- `net/` — Shared `HttpClient` (Get/Post/Put/Delete/SendRaw)
- `config/` — `server.yaml` (primary) plus per-hardware variants: `server.cuda.yaml`, `server.cuda.native.yaml`, `server.cuda.gguf.yaml`, `server.cuda.safetensors.yaml`, `server.cuda.benchmark.yaml`, `server.cuda.qwen14b*.yaml`, `server.cuda.qwen32b.yaml`, `server.rocm*.yaml`, `server.template.yaml`; `policy_store.conf` (encrypted policy persistence), `registry.yaml`

**InferenceRequest structure** (`scheduler/request_batch.h`):
```
InferenceRequest
├── id, model, prompt, max_tokens, priority, ...   (core request fields)
├── response_format: ResponseFormatState            (structured output / grammar)
│   └── has_format, type, schema, grammar, root, ready, supported, error, constraint
├── execution: ExecutionState                       (step-wise batch pause/resume)
│   └── initialized, active, tokens_generated, decode_limit, current_token, ...
├── fairness: FairnessState                         (timeslice / preemption accounting)
│   └── priority_level, service_tokens, timeslice_tokens, remaining_decode_tokens, ...
├── sampling: SamplingParams                        (temperature, top_p, penalties, ...)
└── (phase, tokens, timing, cancellation, KV state, logprobs, stop sequences)
```

**Tech debt tracker:** `docs/TechDebt_and_Competitive_Roadmap.md` — consult at session start for priorities. `docs/INDEX.md` is the docs map.

**Canonical docs (keep in sync with code changes — `check_docs_contract.py` enforces this):**
- `docs/API_SURFACE.md` — all HTTP endpoints and CLI contracts (source-aligned)
- `docs/CONFIG_REFERENCE.md` — full config map including all env vars and YAML knobs
- `docs/GEMV_KERNEL_ARCHITECTURE.md` — kernel geometry, dispatch priority, TDD coverage
- `docs/GGUF_NATIVE_KERNEL_IMPLEMENTATION.md` — native GGUF runtime guide, operator status
- `docs/MONITORING.md` — observability signals, tuning levers, profiling workflow
- `docs/BACKEND_DEVELOPMENT.md` — guide to adding or extending backends
- `docs/design/NATIVE_GGUF_QUANTIZED_RUNTIME_ARCHITECTURE.md` — design rules and next gates

## Coding Conventions

- **C++17**, all symbols in `inferflux` namespace, helpers in anonymous namespaces
- snake_case files, PascalCase public types (`ApiKeyAuth`, `PagedKvCache`), constants prefixed `k` (`kLRU`), member fields end with `_`
- RAII resource management — no naked `new`/`delete`; use `std::unique_ptr`/`std::shared_ptr`
- Headers live beside their `.cpp` files; sorted includes, local before system
- 2-space indent (clang-format enforced)

## Backend Selection & Model Format Routing

**Model formats:** `auto` (default), `gguf`, `safetensors`, `hf` (HuggingFace URI-style `hf://org/repo`)

**Backend resolution logic:**
1. Explicit backend hints (`inferflux_cuda`, `llama_cpp_cuda`) are honored when available
2. Backend priority (`runtime.backend_priority`, `INFERFLUX_BACKEND_PRIORITY`) determines fallback order
3. Capability routing (`runtime.capability_routing.*`) enables graceful degradation when requested capabilities aren't available
4. Backend exposure policy controls which backends are exposed via `/v1/models`

**Model format resolution:**
1. HuggingFace URIs (`hf://org/repo`) resolve to `${INFERFLUX_HOME:-$HOME/.inferflux}/models/org/repo`
2. Auto-detection from file extension (`.gguf`, `.safetensors`)
3. GGUF sidecar fallback for non-GGUF formats when llama.cpp backends are used
4. Format-specific load path resolution for MLX vs llama.cpp backends

## Configuration

All config knobs live in `config/server.yaml` (sections: `server`, `model`, `models`, `runtime`, `adapters`, `auth`, `guardrails`, `logging`, `registry`) and can be overridden with `INFERFLUX_*` environment variables — ~140 are recognized across the codebase; `docs/CONFIG_REFERENCE.md` is the full map. Key ones for development:
- `INFERFLUX_MODEL_PATH` — path to GGUF model file
- `INFERFLUX_MODELS` — multi-model configuration string (`id=model1,path=/path/to/model.gguf,format=gguf,backend=cuda,default=true`)
- `INFERCTL_API_KEY` — API key matching server config (default dev key: `dev-key-123`)
- `INFERFLUX_POLICY_PASSPHRASE` — enables AES-GCM encryption on the policy store
- `INFERFLUX_MPS_LAYERS` — number of layers to offload to Metal
- `INFERFLUX_PORT_OVERRIDE` / `INFERFLUX_HOST_OVERRIDE` — network overrides
- `INFERFLUX_BACKEND_PREFER_INFERFLUX` — prefer InferFlux implementations over llama.cpp
- `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK` — allow fallback to llama.cpp backends
- `INFERFLUX_CUDA_STRICT` — fail load if native CUDA runtime reports fallback
- `INFERFLUX_MODEL_FORMAT` — override model format detection
- `INFERFLUX_DISABLE_STARTUP_ADVISOR=true` — suppress startup configuration recommendations
- `INFERFLUX_RATE_LIMIT_PER_MINUTE` — requests-per-minute limit (set via env, not `config/`)

**Security:** never commit real API keys or passphrases into `config/`. When touching guardrail, auth, or audit paths (`policy/`, `server/auth/`, `server/logging/`), confirm `logs/audit.log` remains writable, document RBAC impacts, and describe rollback steps in the PR.

## CUDA Development

**Two-backend architecture:** The CUDA path has two providers that both accept GGUF models:
- `inferflux_cuda` (`runtime/backends/cuda/inferflux_cuda_backend.cpp`, `inferflux_cuda_executor.cpp`) — first-party CUDA kernels, no llama.cpp dependency at inference time. Owns logprobs, embeddings, batched decode, 50+ fused GEMV kernels (v1 column-major + v2 cooperative-warp), FlashAttention-2 with GQA and multi-sequence decode, CUDA graph capture/replay with retry logic (3 retries before permanent disable), MMQ accumulate kernels (M=9-64 residual fusion), FlashDecode split-K attention. Use for **single-request optimization** and native feature development. Verified throughput (RTX 4000 Ada, Qwen2.5-3B Q4_K_M): c=1 76 tok/s (0.76x vs llama.cpp 100), c=4 153 tok/s (0.83x vs llama.cpp 184), c=8 168 tok/s (0.66x vs llama.cpp 253). Zero crashes at all concurrency levels. Memory: +1268 MB overhead vs llama.cpp.
- `llama_cpp_cuda` — delegates to llama.cpp for inference. **Use for concurrent workloads**. Higher throughput today, lower ceiling for InferFlux-specific innovation.

Only structured output (grammar-constrained generation) still delegates to the llama.cpp parity backend. Logprobs and embeddings are native.

**Verified benchmark** (RTX 4000 Ada 20GB, Qwen2.5-3B Q4_K_M, Apr 15 2026):
```
Backend             c=1 tok/s   c=4 tok/s   c=8 tok/s   Scale   GPU Peak   Quality
───────────────     ─────────   ─────────   ─────────   ─────   ────────   ────────
llama_cpp_cuda        99.8       184.4       252.8     2.5x     5811 MB   16/16 ✓
inferflux_cuda        76.3       153.4       168.1     2.2x     7079 MB   partial¹
Ollama²               ~98        ~111        ~113     1.2x     5434 MB   16/16 ✓
LM Studio²           ~109         ~81         ~70     0.6x     7892 MB   16/16 ✓

¹ inferflux_cuda: first-token logit parity excellent (top-5 Jaccard 1.0,
  delta <0.04). Multi-token responses diverge (~10% Jaccard). MMVQ kernels
  use same precision as llama.cpp (__dp4a + FP32 accum + FP16 output).
  Divergence root cause under investigation (attention/RoPE/residual path).
² Ollama/LM Studio from Apr 14 run (remote host). Both use llama.cpp.

inferflux_cuda vs llama_cpp: c=1 0.76x | c=4 0.83x | c=8 0.66x | Memory +1268 MB
inferflux_cuda vs Ollama:    c=4 1.38x | c=8 1.49x FASTER
inferflux_cuda vs LM Studio: c=4 1.89x | c=8 2.40x FASTER

Key: llama_cpp_cuda is the recommended production backend.
inferflux_cuda beats Ollama and LM Studio at c>=4.
Primary bottleneck: FFN MMVQ kernels (45% of decode time).
See docs/TechDebt_and_Competitive_Roadmap.md for optimization roadmap.

IMPORTANT: After any source changes, do a clean CUDA rebuild to avoid
stale object files (WSL2 filesystem timestamp issue):
  rm -rf build-cuda && cmake -S . -B build-cuda -DENABLE_CUDA=ON && \
  cmake --build build-cuda -j$(nproc) --target inferfluxd
```

**Quality fixes applied:**
- Chat template rendering: strategy-based renderer (ChatML/Llama/Mistral/Gemma) auto-detected from GGUF metadata. Previously a stub returning empty → 43% accuracy.
- Repetition penalty: CUDA kernel + per-sequence token tracking. Default 1.15x for greedy decode. Previously missing entirely → 31% degenerate loops.
- Tokenizer: GGUF special token type parsing (control tokens from tokenizer.ggml.token_type). LlamaTokenizer used for encoding (correct regex pre-tokenization), GGUFTokenizer used for chat template rendering.
- KV cache clearing: ClearSequenceAsync on prefill when n_past==0.
- Remaining quality gap: multi-token response divergence (~10% Jaccard similarity). First-token logit parity is excellent (top-5 Jaccard 1.0, delta <0.04). MMVQ kernels use the same precision as llama.cpp (__dp4a int8 dot products, FP32 accumulation, FP16 output). Root cause under investigation — likely in attention, RoPE, RmsNorm, or residual stream accumulation order rather than MMVQ kernels.

**GPU memory optimizations:**
- Scratch buffer aliasing: attention↔FFN buffers share memory (never live simultaneously). Saves ~56 MB.
- FlashDecode splits: 16→8 (still saturates Ada SMs). Saves ~64 MB.
- KV budget: 0.30→0.20 of free GPU memory. Saves ~100 MB.
- KV batch right-sizing: kMinKvBatch 32→4, default kv_max_batch 32→16. Saves ~288 MB.
- Total memory overhead vs llama.cpp: +1268 MB (down from +2455 MB).

**Key CUDA env vars:** (centralized in `NativeExecutionPolicy::FromEnv()`)
- `INFERFLUX_DISABLE_BATCHED_DECODE=1` — opt out of batched decode (default-on)
- `INFERFLUX_DISABLE_CUDA_GRAPH=1` — disable CUDA graph capture (default-on for primary forward; lane forwards have graphs disabled automatically during overlap; lane overlap mutex fixes in 0ccbad3 prevent heap corruption)
- `INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1` — disable pre-quantized Q8_1 activation path
- `INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0|1` — toggle fused gate+up+SiLU MMVQ kernel (default on)
- `INFERFLUX_ENABLE_FUSED_RESIDUAL_NORM=0|1` — fuse ResidualAdd+RmsNorm into one kernel at layer boundaries (default on)
- `INFERFLUX_ENABLE_FUSED_BIAS_ADD=0|1` — fuse Q/K/V bias adds into single kernel (default on, Qwen2 only)
- `INFERFLUX_ENABLE_GEMV_ACCUMULATE=0|1` — MMVQ accumulate mode for O-proj/down-proj, eliminates separate ResidualAdd (default on)
- `INFERFLUX_ENABLE_FUSED_ROPE_KV_APPEND=0|1` — fuse RoPE+KvAppend into single kernel (default on, P1 validated)
- `INFERFLUX_ENABLE_FUSED_GEMV_NORM_QUANT_EPILOGUE=0|1` — fuse RmsNorm+Q8_1 quant after GEMV accum (default on, P2 validated)
- `INFERFLUX_ENABLE_MMVQ_BIAS_EPILOGUE=0|1` — fuse bias into MMVQ writeback, eliminates BiasAddTriple (default off, P3)
- `INFERFLUX_ENABLE_Q6K_VECTORIZED=0|1` — use vectorized Q6_K MMVQ kernel with __ldg (default off, P4)
- `INFERFLUX_ENABLE_GATE_UP_SILU_Q81_EPILOGUE=0|1` — fuse Q8_1 quant into gate+up+SiLU MMVQ (default off, P5)
- `INFERFLUX_BATCH_DEQUANT_CACHE=0|1` — permanently cache dequantized projection weights instead of using scratch buffer (trades GPU memory for prefill performance, default off)
- `INFERFLUX_GEMV_V2=1` — enable v2 cooperative-warp GEMV kernels (experimental, slower on Ada)
- `INFERFLUX_CUDA_TIMING_SAMPLE_RATE=N` — record CUDA event timing every Nth batch (0=off)
- `INFERFLUX_CUDA_FP32_RESIDUAL=0|1` — FP32 residual stream to prevent FP16 quantization error compounding across layers (default on)
- `INFERFLUX_CUDA_PHASE_OVERLAP` — enable prefill/decode lane overlap
- `INFERFLUX_CUDA_ATTENTION_KERNEL` — force attention kernel (`auto`, `fa2`, `standard`)
- `INFERFLUX_CUDA_KV_MAX_BATCH` / `INFERFLUX_CUDA_KV_MAX_SEQ` — KV cache sizing
- `INFERFLUX_CUDA_KV_BASE_SLOTS=N` — hybrid KV cache: N dense base slots + per-slot overflow (0=all dense, default 0)
- `INFERFLUX_CUDA_DISPATCH_TRACE` (+ `_LIMIT`) — emit selected/actual operator traces, summarized by `scripts/parse_dispatch_trace.py`
- `INFERFLUX_CUDA_DISPATCH_PROBE` / `INFERFLUX_CUDA_DISPATCH_PROBE_FORCE_UNHEALTHY` — load-time reachability and controlled operator down-ranking; degraded state is visible in `/readyz`
- `INFERFLUX_DISABLE_FUSED_GEMV=1`, `INFERFLUX_FORCE_CUBLAS=1`, `INFERFLUX_CUDA_REQUIRE_FUSED_MATMUL=1` — dispatch overrides
- `INFERFLUX_ENABLE_DOWNPROJ_MMQ` / `INFERFLUX_DOWNPROJ_MMQ_MIN_BATCH` — down-projection MMQ threshold
- `INFERFLUX_CUDA_MMQ_MMA=1` / `INFERFLUX_CUDA_MMQ_MMA_MAX_BATCH` — mma.sync int8 tensor-core Q6_K down-proj with deterministic K-split (default off; needs no MMQ layout transform)
- `INFERFLUX_MMVQ_MIN_WARPS` / `INFERFLUX_MMVQ_MAX_WARPS` / `INFERFLUX_ENABLE_ADAPTIVE_MMVQ_THREADS` — MMVQ occupancy tuning
- `INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_*` — experimental grouped/rowpair/rowquad kernel variants (default off)
- `INFERFLUX_CUDA_DEQUANT_CACHE_POLICY` — dequantized-weight cache policy (memory contract; see `GGUFMemoryContractTests`)
- `INFERFLUX_CUDA_TIMING_SAMPLE_RATE=N` / `INFERFLUX_CUDA_PHASE_TIMING=1` — CUDA event timing (0=off)
- `INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION` / `INFERFLUX_CUDA_DEBUG_DECODE_MAPPING` (+ `_LIMIT` variants), `INFERFLUX_DEBUG_LOGITS` — dispatch/numeric debug traces
- Set outside the policy struct (parsed in `server/main.cpp` / bootstrap): `INFERFLUX_CUDA_PHASE_OVERLAP` (prefill/decode lane overlap), `INFERFLUX_CUDA_ATTENTION_KERNEL` (`auto`|`fa2`|`standard`), `INFERFLUX_CUDA_KV_MAX_BATCH` / `INFERFLUX_CUDA_KV_MAX_SEQ` (KV sizing, `native_bootstrap_config.cpp`)
- Dispatch observability/self-heal: `INFERFLUX_CUDA_DISPATCH_TRACE` (+`_LIMIT`) — per-dispatch stderr trace parsed by `scripts/parse_dispatch_trace.py`; `INFERFLUX_CUDA_DISPATCH_PROBE` (default on) — load-time reachability probe that down-ranks unreachable operators (self-heal; visible in `/readyz` `dispatch_degraded`); `INFERFLUX_CUDA_DISPATCH_PROBE_FORCE_UNHEALTHY` — CSV (e.g. `ffn:q8_1_group_mmq3`) to force operators unhealthy for A/B

**InferFlux CUDA kernel files:**
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh` — V1 GEMV kernels (column-major, 8 warps/block, used for pair/triple M>8 fallback)
- `runtime/backends/cuda/native/kernels/mmvq.cuh` — MMVQ weight-read-first kernels (batch 1-8, primary dispatch path) + accumulate variants for residual-stream fusion
- `runtime/backends/cuda/native/kernels/mmq.cuh` — MMQ tiled quantized GEMM kernels (batch 9-64)
- `runtime/backends/cuda/native/kernels/quant_common.cuh` — shared quantization primitives (Dp4aS8, Vsubss4, LoadPacked*)
- `runtime/backends/cuda/native/fused_quant_gemm.cu` — dispatch tables, MMVQ/MMQ selection, threshold logic
- `runtime/backends/cuda/native/transformer_forward.cu` — forward pass wiring
- `runtime/backends/cuda/native/cuda_kernels.cu` — batched RoPE/KvAppend, MeanPool, utility kernels
- `runtime/backends/cuda/native/kernels/fused_rope_kv_append.cuh` — fused RoPE+KvAppend kernel (P1)
- `runtime/backends/cuda/native/kernels/fused_gemv_accum_norm_quant.cuh` — fused RmsNorm+Q8_1 quant epilogue (P2)
**Kernel design principle: weight-read-first.** Loop over weight blocks in the outer loop and accumulate across the batch inside; activation-centric loops re-read weights per row and hit a hard throughput ceiling. A batch-kernel rewrite that inverted this loop regressed ~600% and was abandoned.
- `runtime/backends/cuda/kernels/flash_attention.cu` — FlashAttention-2 and FlashDecodeMultiSeq
- `runtime/backends/cuda/kernels/flash_attention_mma.cuh` — MMA tensor-core prefill kernel (m16n8k16, Br=16, auto-selected for query_len≥16)

**Quantization & weight layout layer** (`runtime/backends/cuda/native/`): `quantization_handler.{h,cpp}` with per-format handlers (`q4_k_m_handler`, `q5_k_m_handler`, `q6_k_handler`, `q8_0_handler`, `q8_k_handler`), `quantized_weight_map.{h,cpp}` (+ scratch/adapter), `weight_map.{h,cpp}`, `safetensors_parser`/`safetensors_adapter`, `model_memory_ledger` (VRAM accounting), and `strategy_registry.{h,cpp}` — pluggable `IWeightLayoutStrategy` / `IMatmulStrategy` / `IAttentionStrategy` with `KvPrecision` (fp16/bf16/int8/fp8) and `MatmulExecutionMode` (fused-dequant-tile-GEMM vs compat dequantize-then-GEMM).

**Policy and execution files:**
- `native_execution_policy.h` — `NativeExecutionPolicy` struct, env var parsing
- `native_dispatch_policy.{h,cpp}` — operator selection, dispatch decisions
- `native_dispatch_registry.{h,cpp}` — per-phase/batch-bucket dispatch winner registry
- `native_bootstrap_config.{h,cpp}` — KV cache sizing, startup config
- `native_linear_executor.h` — projection stage execution with fallback chains
- `cuda_sync_trace.h`, `cuda_copy_trace.h`, `nvtx_scoped.h` — sync/copy latency tracing and NVTX ranges

**InferFlux CUDA metrics:** Prometheus at `/metrics`: `inferflux_cuda_forward_passes_total{phase}`, `inferflux_cuda_forward_batch_tokens_total`, `inferflux_cuda_forward_duration_ms`, `inferflux_cuda_sampling_duration_ms`, `inferflux_cuda_kv_active_sequences`, `inferflux_cuda_kv_max_sequences`, FFN/down-proj operator counters. NVTX annotations for Nsight Systems profiling.

**Benchmark/profile entry points** — always go through the three dispatcher scripts; `scripts/README.md` documents them:
```bash
bash scripts/benchmark.sh throughput-gate        # performance regression gate
bash scripts/benchmark.sh gguf-compare <model>   # inferflux_cuda vs llama_cpp_cuda
bash scripts/benchmark.sh multi-backend <model>  # + ollama, lmstudio, vllm, sglang
# AUTOSTART_VLLM=true / AUTOSTART_SGLANG=true to launch external engines automatically
# INFERFLUX_BENCH_SINGLE_BACKEND=<id> to run one backend through the full harness
# BUILD_DIR=./build-cuda to select a build tree

bash scripts/profile.sh backend <backend>        # nsys backend profile
bash scripts/profile.sh backend-ncu <backend>    # ncu kernel profile
bash scripts/profile.sh phase-timing <log>       # per-phase timing breakdown
bash scripts/profile.sh phase-compare --model <m>
bash scripts/profile.sh analyze-nsys <dir>

bash scripts/smoke.sh gguf-native --model-dir <dir>   # no model server needed
bash scripts/smoke.sh backend-identity
```
The multi-backend harness runs each backend in an isolated child process on purpose, so local CUDA backends never share allocator/stream state within a session. It auto-detects model format and only runs compatible engines (GGUF → inferflux_cuda/llama_cpp_cuda/ollama; safetensors → inferflux_cuda/vllm/sglang).

**Script archive policy:** One-off probes and superseded wrappers live in `scripts/archive/`. New scripts should extend the entry points above, not add new top-level files.

## Disaggregated Runtime

`runtime/disaggregated/` implements split prefill/decode with KV transfer:
- `kv_channel.h` — `IKVTransport` interface and ticket-based KV handoff with lifecycle tracking (create/transfer/consume/timeout)
- `shm_kv_transport.h` — shared-memory transport for process-local KV transfer
- Wired through `Scheduler::DisaggregatedConfig` (`prefill_pool_size`, `decode_pool_size`, `kv_transport`, `kv_enqueue_max_retries`); 0 pool size means unified
- Health signals: timeout streak/debt metrics influence `/readyz` and optional fail-closed admission

## Commits & PRs

Short imperative subjects under ~72 chars mentioning scope (e.g., `Wire speculative validation and async NVMe writes`). **No AI attribution, co-author trailers, or agent names — the `commit-msg` hook rejects them.** PR bodies should link the tracking issue, enumerate config/env changes, and paste ctest output. Update README.md, docs/, and Helm/Docker assets alongside code changes; re-run `python3 scripts/check_docs_contract.py` when doc or surface changes are involved.

<!-- imported-from: codex:project:instructions -->

# Branching & Releases

- **`develop`** is the integration branch: feature branches open PRs into `develop` (CI-gated).
- **Promotion is explicit**: `develop` → `main` via a release PR (CI-gated); releases are tagged on `main`.
- `main` always reflects the last promoted, releasable state.
