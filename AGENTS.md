# Repository Guidelines

## Project Structure & Architecture

InferFlux is a C++17 inference server. `runtime/` owns backend execution, batching, caches, structured output, and distributed KV transport; `scheduler/` handles fairness and routing; `server/` contains HTTP, auth, policy, metrics, and diagnostics. Model/tokenizer code lives in `model/`, `inferctl` in `cli/`, and shared helpers in `io/` and `net/`. Tests are split across `tests/unit/`, `tests/integration/`, and `tests/tools/`. Keep deployment changes aligned across `config/`, `docker/`, `charts/`, and `deploy/`. Treat `external/` and `third_party/` as pinned dependencies.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` initializes pinned dependencies.
- `./scripts/build.sh` creates a release build; use `--cpu-only`, `--all-backends`, or `BUILD_DIR=build-cuda` as needed.
- `cmake -S . -B build -DENABLE_CUDA=OFF && cmake --build build -j` is the hosted-CI loop.
- `ctest --test-dir build --output-on-failure --timeout 90` runs configured tests; target one with `ctest --test-dir build -R ModelIdentityTests -V`.
- `INFERFLUX_MODEL_PATH=/path/model.gguf ./scripts/run_dev.sh config/server.yaml` starts the server. Verify with `./build/inferctl status --api-key dev-key-123`.

Install local checks with `bash scripts/install-hooks.sh`. Run `python3 scripts/check_docs_contract.py` after changing documentation, endpoints, or CLI commands. Use a clean GPU build after backend changes because WSL timestamps can preserve stale objects.

## Coding Style & Testing

Use two-space indentation, RAII, smart pointers, and the `inferflux` namespace. Files and functions use `snake_case`, public types use `PascalCase`, constants use `kName`, and members end in `_`. Run `clang-format` on touched C++/CUDA files and add new sources to `CMakeLists.txt`. Unit tests use vendored Catch2; name cases after observable behavior and keep fixtures deterministic. Never claim GPU coverage from a compile-only job.

## Trusted GPU Runner Operations

Use only `aiserver1-dual-gpu` for runtime gates; hosted runners handle non-GPU work. This WSL environment lacks a usable systemd service bus, so after a host restart run `/home/vsingh/actions-runner-inferflux-gpu/run.sh` in a durable terminal. Verify runner `9054` with `gh api orgs/anvai-labs/actions/runners/9054 --jq .status`. Do not rerun `config.sh` during normal startup or register a second agent. CUDA and ROCm jobs must run serially and never execute pull-request code. Follow `docs/GPU_CI_BOOTSTRAP.md` for recovery.

## Commits, Pull Requests, and Security

Use imperative subjects under about 72 characters. PRs must link an issue, explain rationale and backend/config/API effects, list exact validation, document rollback, and update relevant docs. Never commit production keys, models, policy stores, logs, or benchmark artifacts. Use environment variables such as `INFERCTL_API_KEY`, `INFERFLUX_POLICY_STORE`, and `INFERFLUX_POLICY_PASSPHRASE`; `dev-key-123` is local-only.
