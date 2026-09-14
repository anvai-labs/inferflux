# Backend Naming Strategy

## Status

Executed (2026-09-14). This document is the canonical naming reference
for backend ids, routing aliases, provider strings, and class names.
The original 8-phase execution plan, test inventory, acceptance
criteria, and risk/rollback sections are completed history in git log.
`inferflux_rocm` is implemented (`runtime/backends/backend_factory.cpp`);
`inferflux_mlx` remains future naming.

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


---

## Status Note (2026-09-14)

Executed. The naming model below is the canonical reference; the
8-phase execution task plan, test inventory, acceptance criteria, and
risk/rollback sections were removed as completed history (see git log).
`inferflux_rocm` is implemented (`runtime/backends/backend_factory.cpp`);
`inferflux_mlx` remains future naming.

