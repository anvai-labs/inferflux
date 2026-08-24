#!/usr/bin/env python3
"""
Circular dependency and refactoring analysis tool.

Identifies specific circular dependencies and provides refactoring roadmap
for multi-architecture FlashAttention support (NVIDIA Ada, Ampere, Hopper;
AMD ROCm CDNA 2/3).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DependencyNode:
    def __init__(self, path: str):
        self.path = path
        self.name = Path(path).name
        self.dir = Path(path).parent.name
        self.includes: Set[str] = set()
        self.included_by: Set[str] = set()


def parse_includes(file_path: str) -> List[str]:
    """Parse #include statements from a file."""
    includes = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.match(r'#include\s+[<"]([^>"]+)[>"]', line)
                if match:
                    includes.append(match.group(1))
    except Exception:
        pass
    return includes


def build_dependency_graph() -> Dict[str, DependencyNode]:
    """Build dependency graph for backend files."""

    backend_root = Path("runtime/backends")
    nodes = {}

    # Find all header and source files
    for ext in ['*.h', '*.cpp']:
        for file_path in backend_root.rglob(ext):
            abs_path = str(file_path)
            node = DependencyNode(abs_path)
            node.includes = set(parse_includes(abs_path))
            nodes[abs_path] = node

    # Build reverse dependencies (included_by)
    for path, node in nodes.items():
        for inc in node.includes:
            # Find files that match this include
            for other_path, other_node in nodes.items():
                if other_path.endswith(inc) or Path(other_path).name == inc:
                    other_node.included_by.add(path)

    return nodes


def find_circular_dependencies(nodes: Dict[str, DependencyNode]) -> List[List[str]]:
    """Find circular dependencies using DFS."""

    cycles = []

    def dfs(path: str, visited: Set[str], rec_stack: List[str]) -> bool:
        visited.add(path)
        rec_stack.append(path)

        node = nodes.get(path)
        if not node:
            rec_stack.pop()
            return False

        # Check if any of our includes lead back to us
        for inc in node.includes:
            # Find the full path for this include
            for other_path, other_node in nodes.items():
                if other_path.endswith(inc) or Path(other_path).name == inc:
                    if other_path not in visited:
                        if dfs(other_path, visited, rec_stack):
                            return True
                    elif other_path in rec_stack:
                        # Found a cycle
                        cycle_start = rec_stack.index(other_path)
                        cycle = rec_stack[cycle_start:] + [other_path]
                        cycles.append(cycle)
                        return True

        rec_stack.pop()
        return False

    visited = set()
    for path in nodes:
        if path not in visited:
            dfs(path, visited, [])

    return cycles


def analyze_cuda_llama_coupling(nodes: Dict[str, DependencyNode]) -> Dict:
    """Analyze coupling between CUDA and llama.cpp backends."""

    coupling = {
        "cuda_depends_on_llama": [],
        "llama_depends_on_cuda": [],
        "shared_types": [],
        "shared_functions": []
    }

    for path, node in nodes.items():
        if 'cuda' in node.dir.lower():
            for inc in node.includes:
                if 'llama' in inc.lower() and 'backend' in inc.lower():
                    coupling["cuda_depends_on_llama"].append({
                        "from": Path(path).name,
                        "to": inc
                    })

        if 'llama' in node.dir.lower():
            for inc in node.includes:
                if 'cuda' in inc.lower():
                    coupling["llama_depends_on_cuda"].append({
                        "from": Path(path).name,
                        "to": inc
                    })

    return coupling


def check_flashattention_readiness() -> Dict:
    """Check readiness for multi-architecture FlashAttention."""

    readiness = {
        "has_interface": False,
        "has_cuda_fa2": False,
        "has_cuda_fa3": False,
        "has_rocm_fa": False,
        "architecture_detection": False,
        "kernel_registry": False
    }

    # Check for FlashAttention interface
    fa_files = [
        "runtime/backends/cuda/kernels/flash_attention.cpp",
        "runtime/backends/cuda/kernels/flash_attention.h"
    ]

    for fa_file in fa_files:
        if Path(fa_file).exists():
            with open(fa_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "class" in content or "interface" in content.lower():
                    readiness["has_interface"] = True
                if "fa2" in content.lower() or "flash_attention_2" in content.lower():
                    readiness["has_cuda_fa2"] = True
                if "fa3" in content.lower() or "flash_attention_3" in content.lower():
                    readiness["has_cuda_fa3"] = True
                if "sm_" in content or "compute_capability" in content:
                    readiness["architecture_detection"] = True

    # Check ROCm
    rocm_files = Path("runtime/backends/rocm").rglob("*.cpp") if Path("runtime/backends/rocm").exists() else []
    for rocm_file in rocm_files:
        with open(rocm_file, 'r', encoding='utf-8', errors='ignore') as f:
            if "flash" in f.read().lower():
                readiness["has_rocm_fa"] = True
                break

    return readiness


def generate_refactoring_roadmap(coupling: Dict, cycles: List) -> Dict:
    """Generate refactoring roadmap for multi-architecture support."""

    roadmap = {
        "phase_1": {
            "name": "Decouple CUDA from llama.cpp",
            "priority": "HIGH",
            "tasks": [],
            "estimated_effort": "2-3 days"
        },
        "phase_2": {
            "name": "Create FlashAttention architecture registry",
            "priority": "HIGH",
            "tasks": [],
            "estimated_effort": "3-5 days"
        },
        "phase_3": {
            "name": "Implement ROCm FlashAttention backend",
            "priority": "MEDIUM",
            "tasks": [],
            "estimated_effort": "5-7 days"
        },
        "phase_4": {
            "name": "Add per-architecture kernel selection",
            "priority": "MEDIUM",
            "tasks": [],
            "estimated_effort": "2-3 days"
        }
    }

    # Phase 1 tasks
    if coupling["cuda_depends_on_llama"]:
        roadmap["phase_1"]["tasks"].append({
            "task": "Extract LlamaCPUBackend interface to common/",
            "files": list(set([item['to'] for item in coupling["cuda_depends_on_llama"]])),
            "action": "Move shared types to backend_types.h"
        })
        roadmap["phase_1"]["tasks"].append({
            "task": "Create ILlamaBackend interface",
            "files": ["cuda_backend.cpp", "native_cuda_runtime.cpp"],
            "action": "Use dependency injection instead of direct includes"
        })

    # Phase 2 tasks
    roadmap["phase_2"]["tasks"].append({
        "task": "Create IFlashAttention interface",
        "files": ["runtime/backends/common/flash_attention_interface.h"],
        "action": "Define architecture-independent interface"
    })
    roadmap["phase_2"]["tasks"].append({
        "task": "Create kernel registry",
        "files": ["runtime/backends/common/kernel_registry.h"],
        "action": "Register FA2/FA3 kernels by compute capability"
    })

    # Phase 3 tasks
    roadmap["phase_3"]["tasks"].append({
        "task": "Implement ROCm FlashAttention",
        "files": ["runtime/backends/rocm/rocm_flash_attention.cpp"],
        "action": "Port FA2/FA3 to HIP/ROCm"
    })

    # Phase 4 tasks
    roadmap["phase_4"]["tasks"].append({
        "task": "Add architecture detection",
        "files": ["runtime/backends/common/architecture.h"],
        "action": "Detect GPU arch (Ada, Ampere, Hopper, CDNA2, CDNA3)"
    })

    return roadmap


def main():
    print("="*80)
    print("CIRCULAR DEPENDENCY & REFACTORING ANALYSIS")
    print("="*80)
    print()

    # Build dependency graph
    print("Building dependency graph...")
    nodes = build_dependency_graph()

    # Find circular dependencies
    print("Detecting circular dependencies...")
    cycles = find_circular_dependencies(nodes)

    print()
    print("## CIRCULAR DEPENDENCIES")
    print("-"*80)
    if cycles:
        print(f"❌ Found {len(cycles)} circular dependency chains:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\nCycle {i}:")
            for j, path in enumerate(cycle):
                prefix = "  └─" if j > 0 else ""
                print(f"{prefix} {Path(path).name}")
    else:
        print("✅ No circular dependencies detected!")

    # Analyze CUDA-llama coupling
    print()
    print("## CUDA-LLAMA.CPP COUPLING")
    print("-"*80)
    coupling = analyze_cuda_llama_coupling(nodes)

    print(f"\nCUDA → llama.cpp dependencies: {len(coupling['cuda_depends_on_llama'])}")
    if coupling["cuda_depends_on_llama"]:
        print("  Files:")
        seen = set()
        for dep in coupling["cuda_depends_on_llama"]:
            if dep['from'] not in seen:
                print(f"    - {dep['from']} → {dep['to']}")
                seen.add(dep['from'])

    print(f"\nllama.cpp → CUDA dependencies: {len(coupling['llama_depends_on_cuda'])}")
    if coupling["llama_depends_on_cuda"]:
        print("  Files:")
        seen = set()
        for dep in coupling["llama_depends_on_cuda"]:
            if dep['from'] not in seen:
                print(f"    - {dep['from']} → {dep['to']}")
                seen.add(dep['from'])

    # Check FlashAttention readiness
    print()
    print("## FLASHATTENTION MULTI-ARCHITECTURE READINESS")
    print("-"*80)
    fa_readiness = check_flashattention_readiness()

    items = [
        ("Interface abstraction", fa_readiness["has_interface"]),
        ("CUDA FA2 support", fa_readiness["has_cuda_fa2"]),
        ("CUDA FA3 support", fa_readiness["has_cuda_fa3"]),
        ("ROCm FlashAttention", fa_readiness["has_rocm_fa"]),
        ("Architecture detection", fa_readiness["architecture_detection"]),
        ("Kernel registry", fa_readiness["kernel_registry"])
    ]

    for item, status in items:
        icon = "✅" if status else "❌"
        print(f"  {icon} {item}")

    # Generate roadmap
    print()
    print("## REFACTORING ROADMAP")
    print("-"*80)
    roadmap = generate_refactoring_roadmap(coupling, cycles)

    total_effort = 0
    for phase_key, phase in roadmap.items():
        print(f"\n{phase_key.upper()}: {phase['name']} ({phase['priority']} PRIORITY)")
        print(f"  Estimated effort: {phase['estimated_effort']}")
        print(f"  Tasks:")
        for task in phase['tasks']:
            print(f"    • {task['task']}")
            print(f"      Action: {task['action']}")

    # Calculate total effort
    effort_days = [2.5, 4, 6, 2.5]  # Average of ranges
    total_effort = sum(effort_days)

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Circular dependencies: {len(cycles)}")
    print(f"CUDA-llama coupling points: {len(coupling['cuda_depends_on_llama'])}")
    print(f"FlashAttention readiness: {sum(items[i][1] for i in range(len(items)))}/{len(items)}")
    print(f"Estimated refactoring effort: {total_effort} days")

    if len(coupling['cuda_depends_on_llama']) > 3:
        print("\n⚠️  HIGH COUPLING: CUDA backend tightly coupled to llama.cpp")
        print("   Action: Prioritize Phase 1 refactoring")
    else:
        print("\n✅ GOOD: Backend coupling is manageable")

    fa_score = sum(items[i][1] for i in range(len(items)))
    if fa_score < 4:
        print("\n⚠️  LOW FLASHATTENTION READINESS")
        print("   Action: Complete Phase 2 before adding ROCm support")
    else:
        print("\n✅ GOOD: FlashAttention infrastructure in place")

    print()


if __name__ == "__main__":
    main()
