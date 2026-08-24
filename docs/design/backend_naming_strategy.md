# Backend Naming Strategy

## Status

Phase 0 is complete. Phases 1-4 are substantially implemented. Phases 5-6 are in active migration and file/include identities are now partially aligned.

Progress snapshot:

- Phase 0: complete
- Phase 1: complete
- Phase 2: substantially complete
- Phase 3: substantially complete
- Phase 4: substantially complete
- Phase 5: in progress
- Phase 6: substantially complete
- Phase 7: in progress

Progress snapshot:

- completed: about 92%
- pending: about 8%

Latest completed slice:

- backend-exposure semantic knobs renamed from `native` to `inferflux`
- public admin/cache JSON keys renamed from `memory.native_*` to
  `memory.inferflux_cuda_*`
- public `INFERFLUX_NATIVE_*` runtime env knobs renamed to
  `INFERFLUX_CUDA_*` / `INFERFLUX_DISABLE_INFERFLUX_CUDA`
- `NativeCudaBackend` renamed to `InferfluxCudaBackend`
- `NativeKernelExecutor` renamed to `InferfluxCudaExecutor`
- first-party CUDA file identities moved to `inferflux_cuda_backend.*`,
  `inferflux_cuda_runtime.*`, and `inferflux_cuda_executor.*`
- first-party llama.cpp file identities moved to `llama_cpp_backend.*`
- `LlamaCPUBackend` renamed to `LlamaCppBackend`
- active metric families renamed from `inferflux_native_*` to `inferflux_cuda_*`
- `MetricsRegistry` recorder APIs and internal state renamed from `Native*` / `native_*` to `InferfluxCuda*` / `inferflux_cuda_*`
- active docs/scripts/config now use canonical backend ids

This document defines the canonical backend naming model for the first OSS-facing naming cleanup. It is written as a task-level execution plan for a background agent.

## Problem

Current backend naming is inconsistent and ambiguous:

- `cuda_native` mixes platform first, engine second.
- `cuda_llama_cpp` makes CUDA look like the primary identity even though the engine distinction is the real differentiator.
- `native` is relative and does not scale once more first-party runtimes exist.
- Docs, CLI, config, metrics, and code use overlapping terms like `native_cuda`, `cuda_native`, `cuda`, `llama_cpp`, and provider names inconsistently.

This hurts:

- operator understanding
- benchmark clarity
- marketing clarity
- documentation consistency
- future platform expansion to `rocm`, `mps`, and `vulkan`

## Decision

Adopt a two-axis naming model:

- `engine`: who executes the runtime
- `platform`: where it runs

Canonical concrete backend ids will use:

- `<engine>_<platform>`

Examples:

- `inferflux_cuda`
- `inferflux_rocm`
- `inferflux_mps`
- `inferflux_vulkan`
- `llama_cpp_cuda`
- `llama_cpp_rocm`
- `llama_cpp_mps`
- `llama_cpp_vulkan`
- `llama_cpp_cpu`

Short routing aliases remain separate:

- `auto`
- `cpu`
- `cuda`
- `rocm`
- `mps`
- `vulkan`

Meaning:

- `cuda` means "pick the best CUDA-capable backend according to policy"
- `inferflux_cuda` means "force the InferFlux engine on CUDA"
- `llama_cpp_cuda` means "force llama.cpp on CUDA"

## Why This Model

This model is:

- unambiguous
- readable
- stable across platforms
- easy to extend
- marketing-friendly because `inferflux_*` clearly names the differentiated engine
- developer-friendly because the same pattern works everywhere

It also aligns with how users actually reason about the system:

1. Which runtime engine am I choosing?
2. Which hardware/platform is it running on?

## Canonical Names

### User-Facing Backend Ids

Preferred concrete ids:

- `inferflux_cuda`
- `inferflux_rocm`
- `inferflux_mps`
- `inferflux_vulkan`
- `llama_cpp_cuda`
- `llama_cpp_rocm`
- `llama_cpp_mps`
- `llama_cpp_vulkan`
- `llama_cpp_cpu`

### Routing Aliases

These remain valid policy hints, not concrete engine identities:

- `auto`
- `cpu`
- `cuda`
- `rocm`
- `mps`
- `vulkan`

### Provider / Engine Strings

Canonical provider strings:

- `inferflux`
- `llama_cpp`

Deprecated terms to remove from user-visible surfaces:

- `native`
- `native_cuda`
- `cuda_native`
- `cuda_llama_cpp`

## Class and Type Naming

### Backend Classes

Rename toward:

- `InferfluxCudaBackend`
- `InferfluxRocmBackend`
- `InferfluxMpsBackend`
- `InferfluxVulkanBackend`
- `LlamaCppCudaBackend`
- `LlamaCppRocmBackend`
- `LlamaCppMpsBackend`
- `LlamaCppVulkanBackend`
- `LlamaCppCpuBackend`

### Descriptor Types

Introduce or normalize around:

- `BackendEngine`
- `BackendPlatform`
- `BackendDescriptor`

Recommended enum values:

```cpp
enum class BackendEngine {
  kInferflux,
  kLlamaCpp,
};

enum class BackendPlatform {
  kCpu,
  kCuda,
  kRocm,
  kMps,
  kVulkan,
};
```

Recommended descriptor:

```cpp
struct BackendDescriptor {
  BackendEngine engine;
  BackendPlatform platform;
};
```

## Compatibility Policy

Because this is the first OSS-facing naming cleanup, do not carry permanent compatibility debt.

Policy:

- no permanent support for old backend-exposure/config key names
- no permanent dual naming in docs
- no permanent alias sprawl in CLI help

Allowed:

- temporary normalization inside the rename branch if needed to keep tests green while migrating
- temporary warnings in a narrow migration window

Not allowed:

- long-term dual naming in user-facing docs, CLI help, or API responses
- long-term support for both old and new backend-exposure/config key names

Target end state:

- only canonical names remain in user-facing docs, CLI, config, and API responses
- legacy backend ids may still normalize internally to canonical ids if kept strictly as parser compatibility

## Scope

### In Scope

- config file backend ids
- CLI flags and output
- API backend/provider fields
- scheduler/router normalization
- benchmark script ids and labels
- docs and diagrams
- class/type names
- tests and fixtures

### Out of Scope

- changing runtime behavior or fallback policy semantics
- changing benchmark methodology
- changing model routing rules other than naming normalization
- changing metric semantics beyond label names if necessary

## Required Mappings

### Concrete Backend Id Mapping

- `cuda_native` -> `inferflux_cuda`
- `cuda_llama_cpp` -> `llama_cpp_cuda`
- `native_cuda` -> `inferflux_cuda`

Planned future platform mappings:

- `rocm_native` or `native_rocm` -> `inferflux_rocm`
- `mps_native` or `native_mps` -> `inferflux_mps`
- `vulkan_native` or `native_vulkan` -> `inferflux_vulkan`
- `rocm_llama_cpp` -> `llama_cpp_rocm`
- `mps_llama_cpp` -> `llama_cpp_mps`
- `vulkan_llama_cpp` -> `llama_cpp_vulkan`

### Provider Mapping

- `native` -> `inferflux`
- `llama_cpp` stays `llama_cpp`

### Documentation Language Mapping

- "native CUDA backend" -> "InferFlux CUDA backend" when referring to engine identity
- "llama.cpp CUDA backend" stays valid descriptive text

## Task Plan

### Phase 0: Freeze the Naming Model

Goal:

- make the naming decision explicit before mechanical edits begin

Tasks:

1. Add this design doc.
2. Reference it from relevant architecture or roadmap docs if needed.
3. Do not start partial rename edits before the mapping table is accepted.

Done when:

- the canonical naming table in this doc is the single source of truth

### Phase 1: Introduce Engine/Platform Normalization Internals

Goal:

- stop open-coded string reasoning from spreading

Tasks:

1. Add or normalize `BackendEngine`, `BackendPlatform`, and `BackendDescriptor`.
2. Add one canonical parser/normalizer:
   - string -> descriptor
   - descriptor -> canonical concrete id
3. Route scheduler/router/backend selection through this normalization layer.
4. Keep temporary legacy parsing only if needed while the branch is being migrated.

Scope files likely include:

- `scheduler/model_router.h`
- `scheduler/model_selection.*`
- `server/startup_advisor.*`
- config parsing and backend exposure code

Done when:

- there is one canonical normalization path
- string handling is not duplicated across scheduler/server/CLI

### Phase 2: Rename User-Facing Concrete Backend Ids

Goal:

- make the externally visible ids consistent

Tasks:

1. Replace `cuda_native` with `inferflux_cuda`.
2. Replace `cuda_llama_cpp` with `llama_cpp_cuda`.
3. Update config docs and examples.
4. Update CLI validation and help text.
5. Update startup advisor suggestions.
6. Update benchmark script backend ids and output labels.

Files likely include:

- `docs/CONFIG_REFERENCE.md`
- `docs/STARTUP_ADVISOR.md`
- `scripts/benchmark_multi_backend_comparison.sh`
- `scripts/run_gguf_comparison_benchmark.sh`
- `scripts/profile_backend.sh`
- `scripts/benchmark.sh`
- CLI parsing and HTTP surfaces

Done when:

- all user-facing docs and scripts use canonical ids only

### Phase 3: Rename Provider Strings

Goal:

- make provider identity explicit and brand-aligned

Tasks:

1. Replace provider string `native` with `inferflux`.
2. Keep `llama_cpp` unchanged.
3. Update API payload tests.
4. Update any metrics/admin payload assertions that check provider strings.

Done when:

- API/admin/CLI surfaces expose `inferflux` and `llama_cpp`, not `native`

### Phase 4: Rename Classes and Files Where It Improves Clarity

Goal:

- align code names with the external model

Tasks:

1. Rename backend classes to `Inferflux*Backend` / `LlamaCpp*Backend`.
2. Rename factory or registry labels if needed.
3. Avoid churn in low-value internals unless it materially improves readability.
4. Prefer semantic correctness over blanket renames.

Non-goal:

- do not rename deep kernel files unless they are actually backend identity files

Done when:

- engine/platform identity is obvious from the class names that users and developers touch most

### Phase 5: Remove Legacy Names

Goal:

- avoid permanent ambiguity

Tasks:

1. Remove old-id parsing after all tests and fixtures are migrated.
2. Remove stale docs like "native_cuda" references.
3. Remove legacy benchmark labels and archived examples that would confuse users.
4. Keep archive docs untouched only if they are clearly archived and not canonical.

Done when:

- `rg` on canonical docs/code finds only intended legacy references in archive/history materials

### Phase 6: Clarify Metrics and Kernel Naming

Goal:

- make observability and hot-path implementation names as unambiguous as the backend ids

Tasks:

1. Rename user-visible metric labels and backend label values to canonical engine/platform names.
2. Audit native CUDA metric keys that still use overloaded terms like `native` where `inferflux` is the real engine identity.
3. Rename identity-bearing GEMM/GEMV/operator and kernel entrypoint names where the current names are misleading to users or developers.
4. Do the kernel/operator rename incrementally:
   - one operator family at a time
   - with perf parity tests and grep-based verification
5. Keep purely internal math helper names unchanged unless they create active confusion.

Done when:

- metrics, operator summaries, and kernel-facing debug names use canonical engine/platform terminology
- no active dashboard or benchmark artifact uses deprecated backend ids

### Phase 7: Reduce Historical Doc Surface

Goal:

- keep OSS-facing docs focused on current product contracts, not internal rename history

Tasks:

1. Keep canonical docs current and compact.
2. Move historical rename notes, superseded investigations, and one-off postmortems under archive when they are no longer needed as active references.
3. Update `ARCHIVE_INDEX.md` or the relevant archive README whenever docs move.
4. Do not delete evidence docs that still back current grade/roadmap claims; archive them instead.

Done when:

- root docs surface only contains current canonical product/operator material
- historical rename/perf narratives are archived or removed from active indexes

## Required Tests

### Unit

Add or update tests for:

1. backend hint normalization
2. provider string mapping
3. CLI argument validation
4. scheduler/router identity preservation

Examples:

- `inferflux_cuda` normalizes to engine `inferflux`, platform `cuda`
- `llama_cpp_cuda` normalizes to engine `llama_cpp`, platform `cuda`
- `cuda` remains a routing alias, not a concrete engine id

### Integration

Update or add focused coverage for:

1. `/v1/models`
2. `/v1/models/{id}`
3. admin model load/unload/default
4. benchmark script config generation
5. startup advisor output

### Contract Gates

Required:

1. API responses expose canonical provider/backend ids
2. CLI `--json` output uses canonical ids
3. config examples use canonical ids
4. invalid old ids fail once the migration is complete

## Acceptance Criteria

The rename is complete only when all are true:

1. canonical user-facing concrete ids are `inferflux_*` and `llama_cpp_*`
2. routing aliases remain short (`cuda`, `cpu`, `rocm`, `mps`, `vulkan`, `auto`)
3. provider strings are `inferflux` and `llama_cpp`
4. no canonical doc recommends `cuda_native` or `cuda_llama_cpp`
5. benchmark scripts and profiling scripts label backends consistently
6. API, CLI, and config tests pass with canonical names only
7. no permanent compatibility parsing remains

## Risks

### Risk: Partial Rename Creates Identity Drift

Symptom:

- docs say one name, API returns another, benchmark scripts use a third

Mitigation:

- do not merge phases piecemeal without contract tests

### Risk: Over-Renaming Low-Level Files

Symptom:

- churn without user-facing value

Mitigation:

- prioritize backend identity files, docs, router, CLI, and tests

### Risk: Breaking Benchmark History Comparison

Symptom:

- historical artifact labels become harder to compare

Mitigation:

- note old-to-new mapping in benchmark docs
- keep archive artifacts archived, not canonical

## Rollback

If the rename branch becomes unstable:

1. keep the normalization layer changes if they simplify code
2. revert the user-facing rename as one unit
3. do not leave mixed concrete ids in the tree

Rollback boundary:

- the rename should be merged as a single coherent identity change, not scattered partial commits

## Suggested Implementation Order

1. add normalization layer
2. update user-facing backend ids
3. update provider strings
4. update tests and scripts
5. update docs
6. remove legacy names
7. rename metrics/operator names where needed
8. archive or retire superseded historical docs

## Background-Agent Instructions

Execute this as a bounded rename program, not as opportunistic string replacement.

Rules:

1. touch only identity-bearing names first
2. keep engine/platform semantics explicit
3. do not introduce long-term aliases
4. update tests with every phase
5. stop if benchmark/config/API identities diverge

Deliverables:

1. code rename
2. docs rename
3. test updates
4. one verification note summarizing:
   - old -> new mapping
   - files changed
   - tests passed
