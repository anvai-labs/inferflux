#!/usr/bin/env python3
"""
Generate visual dependency graph of backend architecture.
"""

def generate_dependency_diagram():
    """Generate a text-based dependency diagram."""

    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BACKEND ARCHITECTURE DEPENDENCY GRAPH                      ║
║                         Current State (Post-Refactoring)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                            SHARED LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  common/backend_interface.h  (IBackendInterface)                            │
│  common/backend_types.h       (UnifiedBatchInput/Output, PrefillResult)     │
│  common/batching_utils.h      (BatchAnalyzer)                                │
└────────────┬────────────────────────────────────────────────────┬────────────┘
             │                                                    │
             │ (uses)                                             │ (uses)
             ▼                                                    ▼
┌────────────────────────────┐                    ┌────────────────────────────┐
│      CUDA BACKEND         │                    │      LLAMA BACKEND        │
├────────────────────────────┤                    ├────────────────────────────┤
│ cuda/cuda_backend.h       │                    │ cpu/llama_backend.h        │
│ cuda/cuda_backend.cpp     │                    │                            │
│ cuda/native_cuda_backend.h│                    │                            │
│ cuda/native_cuda_runtime.h│ ◄─────────────────╮ │                            │
│ cuda/kernels/             │                   │ │                            │
│   flash_attention.cpp     │                   │ │                            │
└──────────┬─────────────────┘                   │ └────────────────────────────┘
           │                                     │
           │ (depends on)                        │
           ▼                                     │
┌────────────────────────────┐                   │
│  ┌──────────────────────┐ │                   │
│  │ ⚠️ COUPLING POINTS  │ │                   │
│  ├──────────────────────┤ │                   │
│  │ 1. cuda_backend.h   │─┼───────────────────╯
│  │    → llama_backend.h│
│  │ 2. native_cuda_     │
│  │    runtime.h        │
│  │    → llama_backend.h│
│  │ 3. native_cuda_     │
│  │    backend.h        │
│  │    → llama_backend.h│
│  │ 4. cuda_backend.cpp │
│  │    → llama_backend_ │
│  │      traits.h       │
│  └──────────────────────┘ │
└────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                     TARGET ARCHITECTURE (After Refactoring)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                         COMMON LAYER                                 │ │
│   │  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │ │
│   │  │ IBackend       │  │ IFlashAttention  │  │ IDeviceContext      │  │ │
│   │  │ Interface      │  │ Interface        │  │ Interface           │  │ │
│   │  └────────────────┘  └──────────────────┘  └─────────────────────┘  │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                        │
│                                    │ implements                             │
│   ┌────────────────────────────────┼──────────────────────────────────────┐ │
│   │                                │                                      │ │
│   │  ┌───────────────┐    ┌──────────┴───────┐    ┌─────────────────┐  │ │
│   │  │ CUDA Backend  │    │ ROCm Backend     │    | CPU Backend     │  │ │
│   │  │ (NVIDIA)      │    │ (AMD)            │    │ (LLAMA.cpp)      │  │ │
│   │  │               │    │                  │    │                 │  │ │
│   │  │ - FA2/FA3     │    │ - ROCm FA        │    │ - CPU decode    │  │ │
│   │  │ - Ada/Ampere  │    │ - CDNA2/3        │    │ - Quantization  │  │ │
│   │  │ - Hopper      │    │                  │    │                 │  │ │
│   │  └───────────────┘    └──────────────────┘    └─────────────────┘  │ │
│   │                                                                  │ │
│   └──────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════

COUPLING ANALYSIS SUMMARY:

┌────────────────────────────────────────────────────────────────────────────┐
│ Current State                                                              │
├────────────────────────────────────────────────────────────────────────────┤
│ ✅ NO CIRCULAR DEPENDENCIES                                                │
│ ⚠️  5 CUDA → llama.cpp dependencies                                        │
│ ✅ 0 llama.cpp → CUDA dependencies                                         │
│ ✅ Common module implemented and working                                   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ Refactoring Priority                                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ PHASE 1 (2-3 days):                                                        │
│   • Extract ILlamaBackend interface to common/                            │
│   • Remove concrete type dependencies                                      │
│   • Use dependency injection                                               │
│                                                                            │
│ PHASE 2 (3-5 days):                                                        │
│   • Create IFlashAttention interface                                       │
│   • Implement kernel registry by architecture                             │
│   • Add runtime architecture detection                                     │
│                                                                            │
│ PHASE 3 (5-7 days):                                                        │
│   • Implement ROCm FlashAttention backend                                  │
│   • Add CDNA2/CDNA3 support                                                │
│   • Test on AMD hardware                                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

FLASHATTENTION MULTI-ARCHITECTURE SUPPORT:

┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Architecture │ Current      │ Target       │ Interface    │ Kernel       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ NVIDIA Ada    │ ✅ FA2      │ ✅ FA2/FA3   │ IFlashAtt    │ cuda/fa2.cu  │
│ NVIDIA Ampere │ ✅ FA2      │ ✅ FA2/FA3   │ IFlashAtt    │ cuda/fa2.cu  │
│ NVIDIA Hopper │ ⚠️  FA2     │ ✅ FA3       │ IFlashAtt    │ cuda/fa3.cu  │
│ AMD CDNA2     │ ⚠️  Basic   │ ✅ ROCm FA   │ IFlashAtt    │ rocm/fa.cpp  │
│ AMD CDNA3     │ ⚠️  Basic   │ ✅ ROCm FA   │ IFlashAtt    │ rocm/fa.cpp  │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

KEY: ✅ Implemented | ⚠️  Partial | ❌ Not Implemented

"""

    return diagram


def generate_flashattention_architecture_diagram():
    """Generate FlashAttention architecture selection diagram."""

    diagram = """

╔══════════════════════════════════════════════════════════════════════════════╗
║              FLASHATTENTION MULTI-ARCHITECTURE SELECTION                      ║
║                         Proposed Architecture                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ Runtime Selection Flow                                                       │
└──────────────────────────────────────────────────────────────────────────────┘

  1. GPU Detection
     │
     ├─► NVIDIA GPU
     │   │
     │   ├─► Compute Capability ≥ 8.0 (Ampere/Ada/Hopper)
     │   │   │
     │   │   ├─► SM ≥ 8.0 (Ampere)        → FA2 Kernel
     │   │   ├─► SM = 8.9 (Ada)           → FA2 Kernel
     │   │   └─► SM ≥ 9.0 (Hopper)        → FA3 Kernel (best)
     │   │
     │   └─► Compute Capability < 8.0 (Volta/Turing)
     │       └─► Fallback: Standard Attention
     │
     └─► AMD GPU
         │
         ├─► CDNA2 (gfx90a/940/941/942)   → ROCm FA Kernel
         ├─► CDNA3 (gfx940/941/942)       → ROCm FA3 Kernel
         └─► Other                        → Fallback: Standard Attention

┌──────────────────────────────────────────────────────────────────────────────┐
│ Kernel Registry Structure                                                    │
└──────────────────────────────────────────────────────────────────────────────┘

struct FlashAttentionKernel {
  string name;
  int min_compute_capability;
  int max_compute_capability;
  ArchType arch;  // NVIDIA_CUDA, AMD_ROCM
  void* kernel_ptr;
  bool (*is_available)();
  float (*benchmark)();  // Returns tok/s
};

class FlashAttentionRegistry {
  void register_kernel(FlashAttentionKernel kernel);
  FlashAttentionKernel* select_best_kernel(int compute_capability, ArchType arch);
  vector<FlashAttentionKernel> get_fallback_chain();
};

┌──────────────────────────────────────────────────────────────────────────────┐
│ Example: NVIDIA RTX 4000 Ada (Compute 8.9)                                   │
└──────────────────────────────────────────────────────────────────────────────┘

  Detection: GPU 0: NVIDIA RTX 4000 Ada (sm_89)
    │
    ├─► Query registry for compute_capability=8.9, arch=NVIDIA_CUDA
    │
    ├─► Available kernels:
    │   1. FA2 (8.0 ≤ sm ≤ 8.9)  ✅ MATCH
    │   2. FA3 (sm ≥ 9.0)         ❌ Not supported
    │
    ├─► Select: FA2 Kernel
    │
    └─► Execute flash_attention_fa2<sm_89>(...)

┌──────────────────────────────────────────────────────────────────────────────┐
│ Example: AMD MI250X (CDNA2, gfx90a)                                          │
└──────────────────────────────────────────────────────────────────────────────┘

  Detection: GPU 0: AMD MI250X (gfx90a)
    │
    ├─► Query registry for arch=AMD_ROCM
    │
    ├─► Available kernels:
    │   1. ROCm FA (CDNA2)       ✅ MATCH
    │   2. ROCm FA3 (CDNA3 only) ❌ Not supported
    │
    ├─► Select: ROCm FA Kernel
    │
    └─► Execute flash_attention_rocm_cdna2(...)

══════════════════════════════════════════════════════════════════════════════

IMPLEMENTATION ROADMAP:

Phase 1: Interface & Registry (3-5 days)
  └─► Create IFlashAttention interface
  └─► Implement KernelRegistry class
  └─► Add architecture detection utilities

Phase 2: CUDA Implementation (5-7 days)
  └─► Implement FA2 kernel wrapper
  └─► Add FA3 kernel support (Hopper)
  └─► Implement compute capability detection

Phase 3: ROCm Implementation (5-7 days)
  └─► Port FA2 to HIP
  └─► Implement ROCm-specific optimizations
  └─► Add CDNA2/CDNA3 detection

Phase 4: Testing & Validation (3-5 days)
  └─► Test on NVIDIA GPUs (Ada, Ampere, Hopper)
  └─► Test on AMD GPUs (CDNA2, CDNA3)
  └─► Benchmark each kernel

Total: 16-24 days (4-6 weeks)

"""
    return diagram


def main():
    print(generate_dependency_diagram())
    print(generate_flashattention_architecture_diagram())


if __name__ == "__main__":
    main()
