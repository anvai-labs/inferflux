#!/usr/bin/env python3
"""
Backend isolation and architecture analysis tool.

Analyzes the backend code structure to identify:
1. Current isolation status between llama.cpp and native CUDA
2. Circular dependencies and cross-linking
3. Readiness for ROCm and multi-architecture FlashAttention
4. Opportunities for further optimization
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def analyze_backend_structure():
    """Analyze the backend directory structure."""

    backend_root = Path("runtime/backends")
    if not backend_root.exists():
        print(f"Error: Backend root not found at {backend_root}")
        return

    results = {
        "directories": {},
        "files": {},
        "dependencies": defaultdict(set),
        "common_patterns": [],
        "circular_risks": []
    }

    # Scan directories
    for backend_dir in backend_root.iterdir():
        if backend_dir.is_dir() and not backend_dir.name.startswith('.'):
            backend_name = backend_dir.name
            results["directories"][backend_name] = {
                "path": str(backend_dir),
                "files": []
            }

            # Scan files
            for file in backend_dir.rglob("*.cpp"):
                results["directories"][backend_name]["files"].append(file)
                results["files"][str(file)] = backend_name

            for file in backend_dir.rglob("*.h"):
                results["directories"][backend_name]["files"].append(file)
                results["files"][str(file)] = backend_name

    return results


def analyze_dependencies(results: Dict) -> Dict:
    """Analyze #include dependencies between backends."""

    include_patterns = {
        "cuda_to_llama": [],
        "llama_to_cuda": [],
        "common_usage": [],
        "external_deps": []
    }

    for file_path, backend in results["files"].items():
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                if not line.strip().startswith('#include'):
                    continue

                # Extract include target
                match = re.search(r'#include\s+[<"]([^>"]+)[>"]', line)
                if not match:
                    continue

                include_target = match.group(1)

                # Categorize includes
                if 'llama' in include_target.lower() and backend == 'cuda':
                    include_patterns["cuda_to_llama"].append({
                        "file": file_path,
                        "line": i,
                        "include": include_target
                    })
                elif 'cuda' in include_target.lower() and backend == 'llama':
                    include_patterns["llama_to_cuda"].append({
                        "file": file_path,
                        "line": i,
                        "include": include_target
                    })
                elif 'common' in include_target.lower():
                    include_patterns["common_usage"].append({
                        "file": file_path,
                        "line": i,
                        "include": include_target
                    })

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return include_patterns


def check_common_module():
    """Check status of common backend module."""

    common_dir = Path("runtime/backends/common")
    if not common_dir.exists():
        return {"status": "NOT_FOUND", "files": []}

    files = list(common_dir.glob("*.h")) + list(common_dir.glob("*.cpp"))
    return {
        "status": "EXISTS",
        "files": [f.name for f in files],
        "path": str(common_dir)
    }


def analyze_backend_factory():
    """Analyze backend factory for architecture issues."""

    factory_file = Path("runtime/backends/backend_factory.cpp")
    if not factory_file.exists():
        return {"status": "NOT_FOUND"}

    with open(factory_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find backend creation logic
    issues = []

    # Check for hardcoded backend dependencies
    if 'llama' in content.lower() and 'cuda' in content.lower():
        issues.append("Factory has dependencies on both llama.cpp and CUDA")

    # Check for ROCm support
    if 'rocm' in content.lower():
        issues.append("ROCm support detected in factory")

    return {
        "status": "EXISTS",
        "issues": issues,
        "line_count": len(content.split('\n'))
    }


def assess_rocm_readiness(results: Dict, deps: Dict) -> Dict:
    """Assess readiness for ROCm integration."""

    readiness = {
        "score": 0,
        "max_score": 100,
        "checklist": {},
        "blockers": [],
        "recommendations": []
    }

    # Check 1: Common module exists
    common = check_common_module()
    if common["status"] == "EXISTS":
        readiness["score"] += 20
        readiness["checklist"]["common_module"] = "PASS"
    else:
        readiness["checklist"]["common_module"] = "FAIL"
        readiness["blockers"].append("Common backend module not found")

    # Check 2: CUDA/llama isolation
    cross_links = len(deps["cuda_to_llama"]) + len(deps["llama_to_cuda"])
    if cross_links == 0:
        readiness["score"] += 30
        readiness["checklist"]["isolation"] = "PASS"
    else:
        readiness["checklist"]["isolation"] = "PARTIAL"
        readiness["recommendations"].append(
            f"Reduce cross-linking: {cross_links} cross-dependencies found"
        )

    # Check 3: Device abstraction
    device_contexts = results["files"].get("cuda", [])
    has_device_abstraction = any("device_context" in f.lower() for f in device_contexts)
    if has_device_abstraction:
        readiness["score"] += 20
        readiness["checklist"]["device_abstraction"] = "PASS"
    else:
        readiness["checklist"]["device_abstraction"] = "FAIL"
        readiness["blockers"].append("Device context abstraction incomplete")

    # Check 4: Backend factory flexibility
    factory = analyze_backend_factory()
    if "ROCm" not in str(factory["issues"]):
        readiness["score"] += 15
        readiness["checklist"]["factory_flexibility"] = "PASS"
    else:
        readiness["checklist"]["factory_flexibility"] = "PARTIAL"

    # Check 5: FlashAttention architecture independence
    flash_attention = Path("runtime/backends/cuda/kernels/flash_attention.cpp")
    if flash_attention.exists():
        readiness["score"] += 15
        readiness["checklist"]["flash_attention"] = "PASS"
    else:
        readiness["checklist"]["flash_attention"] = "FAIL"

    return readiness


def find_optimization_opportunities() -> List[Dict]:
    """Find further optimization opportunities."""

    opportunities = []

    # Check for batching optimization opportunities
    scheduler_file = Path("scheduler/scheduler.cpp")
    if scheduler_file.exists():
        with open(scheduler_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if "batch_accumulation" in content:
            opportunities.append({
                "area": "Batching",
                "status": "OPTIMIZED",
                "description": "Batch accumulation already implemented"
            })

    # Check for CUDA graphs
    if not any("cuda_graph" in f.lower() for f in os.listdir("runtime/backends/cuda")):
        opportunities.append({
            "area": "CUDA Graphs",
            "status": "NOT_IMPLEMENTED",
            "potential_gain": "+15-25% throughput",
            "description": "CUDA graph capture/replay for kernel launch optimization"
        })

    # Check for native kernels
    native_runtime = Path("runtime/backends/cuda/native_cuda_runtime.cpp")
    if native_runtime.exists():
        with open(native_runtime, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if "delegate" in content:
            opportunities.append({
                "area": "Native Kernels",
                "status": "IN_PROGRESS",
                "potential_gain": "+57-96% throughput",
                "description": "Native runtime still has delegate code paths"
            })

    return opportunities


def generate_report():
    """Generate comprehensive analysis report."""

    print("="*80)
    print("BACKEND ISOLATION & ARCHITECTURE ANALYSIS")
    print("="*80)
    print()

    # Analyze structure
    results = analyze_backend_structure()

    print("## 1. BACKEND STRUCTURE")
    print("-" * 80)
    for backend, info in results["directories"].items():
        file_count = len(info["files"])
        print(f"{backend:15s}: {file_count:3d} files")

    print()
    print("## 2. COMMON MODULE STATUS")
    print("-" * 80)
    common = check_common_module()
    if common["status"] == "EXISTS":
        print(f"✅ Common module EXISTS at {common['path']}")
        print(f"   Files: {', '.join(common['files'])}")
    else:
        print("❌ Common module NOT FOUND - needs creation")

    print()
    print("## 3. CROSS-DEPENDENCY ANALYSIS")
    print("-" * 80)
    deps = analyze_dependencies(results)

    print(f"CUDA → llama.cpp includes: {len(deps['cuda_to_llama'])}")
    if deps["cuda_to_llama"]:
        for item in deps["cuda_to_llama"][:5]:
            print(f"  - {Path(item['file']).name}:{item['line']} includes {item['include']}")
        if len(deps["cuda_to_llama"]) > 5:
            print(f"  ... and {len(deps['cuda_to_llama']) - 5} more")

    print(f"\nllama.cpp → CUDA includes: {len(deps['llama_to_cuda'])}")
    if deps["llama_to_cuda"]:
        for item in deps["llama_to_cuda"][:5]:
            print(f"  - {Path(item['file']).name}:{item['line']} includes {item['include']}")
        if len(deps["llama_to_cuda"]) > 5:
            print(f"  ... and {len(deps['llama_to_cuda']) - 5} more")

    print(f"\nCommon module usage: {len(deps['common_usage'])} files")

    print()
    print("## 4. ROCM READINESS ASSESSMENT")
    print("-" * 80)
    readiness = assess_rocm_readiness(results, deps)
    print(f"Readiness Score: {readiness['score']}/{readiness['max_score']}")

    print("\nChecklist:")
    for item, status in readiness["checklist"].items():
        icon = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"  {icon} {item}: {status}")

    if readiness["blockers"]:
        print("\n❌ BLOCKERS:")
        for blocker in readiness["blockers"]:
            print(f"  - {blocker}")

    if readiness["recommendations"]:
        print("\n📋 RECOMMENDATIONS:")
        for rec in readiness["recommendations"]:
            print(f"  - {rec}")

    print()
    print("## 5. OPTIMIZATION OPPORTUNITIES")
    print("-" * 80)
    opportunities = find_optimization_opportunities()
    for opp in opportunities:
        icon = "✅" if opp["status"] == "OPTIMIZED" else "🚧" if opp["status"] == "IN_PROGRESS" else "📋"
        print(f"\n{icon} {opp['area']}: {opp['status']}")
        if "potential_gain" in opp:
            print(f"   Potential: {opp['potential_gain']}")
        print(f"   {opp['description']}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    total_files = sum(len(info["files"]) for info in results["directories"].values())
    cross_links = len(deps["cuda_to_llama"]) + len(deps["llama_to_cuda"])

    print(f"Total backend files: {total_files}")
    print(f"Cross-dependencies: {cross_links}")
    print(f"Common module: {'✅' if common['status'] == 'EXISTS' else '❌'}")
    print(f"ROCm readiness: {readiness['score']}%")

    if cross_links > 10:
        print("\n⚠️  WARNING: High cross-dependency detected")
        print("   Action: Refactor to use common module interfaces")
    elif cross_links > 0:
        print("\n📋 INFO: Some cross-dependencies remain")
        print("   Action: Complete migration to common module")
    else:
        print("\n✅ EXCELLENT: No cross-dependencies detected")

    print()
    return results, deps, readiness, opportunities


if __name__ == "__main__":
    generate_report()
