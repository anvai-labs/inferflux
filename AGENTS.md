# Repository Guidelines

## Project Structure & Architecture

InferFlux is a C++17 inference server. `runtime/` owns backend execution (CPU, CUDA, ROCm, MPS, Vulkan, OpenCL, and MLX), batching, caches, structured output, and distributed KV transport. `scheduler/` handles fairness, routing, and model selection; `server/` contains HTTP, auth, policy, metrics, logging, and diagnostics. Model/tokenizer code lives in `model/`, `inferctl` in `cli/`, and shared helpers in `io/` and `net/`. Tests are split across `tests/unit/`, `tests/integration/`, and `tests/tools/`. Keep deployment changes aligned across `config/`, `docker/`, `charts/`, and `deploy/`; canonical documentation lives in `docs/`. Treat `external/` and `third_party/` as pinned dependencies.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` initializes the pinned `llama.cpp` dependency.
- `./scripts/build.sh` creates a Release build in `build/`; use `--cpu-only`, `--all-backends`, or `BUILD_DIR=build-cuda` when appropriate.
- `cmake -S . -B build -DENABLE_CUDA=OFF && cmake --build build -j` is the incremental loop. Other toggles include `ENABLE_ROCM`, `ENABLE_MPS`, `ENABLE_VULKAN`, `ENABLE_MLX`, `ENABLE_MTMD`, and `ENABLE_WEBUI`.
- `INFERFLUX_MODEL_PATH=/path/model.gguf ./scripts/run_dev.sh config/server.yaml` starts `inferfluxd`. Verify it with `./build/inferctl status --api-key dev-key-123`.

## Coding Style & Naming

Use two-space indentation and run `clang-format -i` on touched `.cpp` and `.h` files. Prefer RAII and smart pointers. Files and free functions use `snake_case`, public types use `PascalCase`, constants use `kName`, and members end in `_`. Keep implementation-only helpers in anonymous namespaces and production symbols in `inferflux`. Add new sources and tests to `CMakeLists.txt`.

## Testing Guidelines

Unit tests use vendored Catch2 and descriptive `TEST_CASE` names with focused tags such as `[paged_kv]`. Run `ctest --test-dir build --output-on-failure --timeout 90`; target a suite with `ctest --test-dir build -R ModelIdentityTests -V`. Python integration and stub-contract tests are registered through CTest and need no model. `IntegrationSSE` is registered only when `INFERFLUX_MODEL_PATH` is set. For documentation or public CLI/API changes, run `python3 scripts/check_docs_contract.py`. Install local checks with `bash scripts/install-hooks.sh`.

## Commits & Pull Requests

Use short, imperative subjects (`Fix ...`, `Add ...`, `Extract ...`) and explain rationale and observable effects in the body. PRs should link the issue, summarize backend/config/API impacts, list exact test commands and results, and update `README.md` or canonical `docs/` pages for user-visible changes. Include curl/CLI transcripts for endpoint changes and benchmark evidence for performance claims.

## Security & Generated Data

Never commit production keys, model files, policy stores, logs, or benchmark/profiling output. Use `INFERCTL_API_KEY`, `INFERFLUX_POLICY_STORE`, and `INFERFLUX_POLICY_PASSPHRASE`; sample `dev-key-123` is local-only. Changes under `policy/`, `server/auth/`, or `server/logging/` must preserve least-privilege scopes, atomic policy persistence, and a writable audit-log path.
