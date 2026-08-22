# Repository Guidelines

## Project Structure & Module Organization
Production C++ lives in `runtime/` (backends, paged KV, decoding), `server/`
(HTTP, auth, metrics), `scheduler/`, `policy/`, `model/`, and `cli/` (`inferctl`).
Deployment assets are in `docker/`, `charts/`, and `scripts/`; configuration is
under `config/`, documentation under `docs/`, and tests under `tests/unit` and
`tests/integration`. Treat `external/` as pinned, read-only vendor code.

## Build, Test, and Development Commands

- `./scripts/build.sh` creates a release build in `build/`.
- `cmake -S . -B build && cmake --build build -j` is the incremental loop.
- `ctest --test-dir build --output-on-failure` runs configured tests.
- `./scripts/run_dev.sh --config config/server.yaml` starts the dev server.
- `./build/inferctl chat --message 'user:Hello' --api-key dev-key-123 --stream`
  exercises the API after a model is configured.

Set `INFERFLUX_MODEL_PATH` for model-backed tests. Prefer a clean GPU build after
backend changes because WSL timestamps can preserve stale objects.

## Coding Style & Naming Conventions
Use C++17, RAII, smart pointers, and the `inferflux` namespace. Apply
`clang-format` with 2-space indentation and sorted includes. Use snake_case for
files and functions, PascalCase for public types, `k`-prefixed constants, and
trailing underscores for members. Keep helpers in anonymous namespaces.

## Testing Guidelines
Unit tests use Catch2. Name `TEST_CASE`s after observable behavior and keep data
in `tests/data/` deterministic. Run focused tests with
`./build/inferflux_tests "case name"` or `ctest -R StubIntegration`. Attach test,
CLI, or HTTP evidence for user-visible changes; never claim GPU coverage from a
compile-only job.

## Trusted GPU Runner Operations
Use only `aiserver1-dual-gpu` for CUDA/ROCm runtime gates; hosted runners handle
non-GPU work. This WSL environment has no systemd bus, so after a host restart
run `/home/vsingh/actions-runner-inferflux-gpu/run.sh` in a durable terminal.
Verify it with `gh api orgs/anvai-labs/actions/runners/9054 --jq .status`. Do not
rerun `config.sh` during normal startup or register a second agent. Follow
`docs/GPU_CI_BOOTSTRAP.md` for recovery. Keep public-repository access and GPU
gate variables disabled until the trusted `main` workflow is promoted.

## Commit & Pull Request Guidelines
Use imperative subjects under about 72 characters and explain rationale in the
body. PRs should link an issue, describe config/environment changes, include test
output, and update relevant docs or deployment assets. Add screenshots or curl
transcripts for API changes.

## Security & Configuration Tips
Never commit keys or passphrases; use `INFERCTL_API_KEY`,
`INFERFLUX_POLICY_PASSPHRASE`, and related environment variables. For auth,
policy, or audit changes, verify `logs/audit.log`, document RBAC impact, and
include rollback instructions.
