#!/usr/bin/env python3
"""
Run profiling to validate optimization assumptions.

Uses Nsight Systems and Nsight Compute to:
1. Measure actual GPU utilization
2. Identify top kernels by execution time
3. Analyze memory transfer patterns
4. Validate CPU-GPU overlap opportunities
"""

import subprocess
import time
import json
import os
import sys
from pathlib import Path


def check_tools():
    """Check if profiling tools are available."""
    tools = {
        "nsys": False,
        "ncu": False
    }

    try:
        result = subprocess.run(["which", "nsys"], capture_output=True)
        if result.returncode == 0:
            tools["nsys"] = True
    except:
        pass

    try:
        result = subprocess.run(["which", "ncu"], capture_output=True)
        if result.returncode == 0:
            tools["ncu"] = True
    except:
        pass

    return tools


def start_server(config="config/server.cuda.yaml", duration=30):
    """Start the inference server."""
    print(f"Starting server with config: {config}")

    # Kill any existing server
    subprocess.run(["pkill", "-9", "-f", "inferfluxd"], stderr=subprocess.DEVNULL)

    # Start server in background
    server_proc = subprocess.Popen(
        ["./build/inferfluxd", "--config", config],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    print("Waiting for server to start...")
    time.sleep(5)

    # Check if server is running
    if server_proc.poll() is not None:
        print("ERROR: Server failed to start")
        return None

    print("✅ Server started")
    return server_proc


def generate_workload(num_requests=50, duration_sec=15):
    """Generate inference workload for profiling."""
    print(f"Generating workload: {num_requests} requests over {duration_sec}s")

    # Use inferctl to send requests
    requests_sent = 0
    start_time = time.time()
    end_time = start_time + duration_sec

    while time.time() < end_time and requests_sent < num_requests:
        try:
            # Send a completion request
            proc = subprocess.run([
                "python3", "scripts/inferctl.py",
                "complete",
                "--model", "tinyllama",
                "--prompt", "Write a short story about AI.",
                "--max-tokens", "50",
                "--no-stream"
            ], capture_output=True, timeout=10)

            requests_sent += 1

            # Small delay between requests
            time.sleep(0.3)

        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"Warning: Request failed: {e}")

    elapsed = time.time() - start_time
    print(f"✅ Sent {requests_sent} requests in {elapsed:.1f}s")

    return requests_sent


def run_nsys_profiling(duration=20, output="profile_output"):
    """Run Nsight Systems profiling."""
    print(f"\n{'='*80}")
    print("RUNNING NSIGHT SYSTEMS PROFILING")
    print(f"{'='*80}\n")

    output_file = f"/tmp/{output}"

    # Kill any existing nsys sessions
    subprocess.run(["pkill", "-9", "nsys"], stderr=subprocess.DEVNULL)

    # Build nsys command
    cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx,osrt",  # Trace CUDA, NVTX, OS runtime
        "-y",  # Overwrite output
        "-o", output_file,
        "-d", str(duration),  # Duration in seconds
        "--force-overwrite=true",
        "./build/inferfluxd",
        "--config", "config/server.cuda.yaml"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Profiling for {duration}s...")

    # Run profiling
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Generate workload during profiling
    time.sleep(3)  # Let server start
    generate_workload(num_requests=50, duration_sec=duration-5)

    # Wait for profiling to complete
    try:
        stdout, stderr = proc.communicate(timeout=duration+10)
        print(f"\n✅ Profiling complete")
        print(f"Output: {output_file}.nsys-rep")

        if stderr:
            print(f"NSYS stderr:\n{stderr}")

        return output_file
    except subprocess.TimeoutExpired:
        proc.kill()
        print("ERROR: Profiling timed out")
        return None


def run_nsys_stats(nsys_file):
    """Export Nsight Systems stats to text format."""
    print(f"\n{'='*80}")
    print("EXPORTING NSIGHT SYSTEMS STATS")
    print(f"{'='*80}\n")

    try:
        # Export to SQLite
        stats_cmd = [
            "nsys", "stats",
            "--report", "gpu_sum",
            "--format", "csv",
            "--output", "/tmp/gpu_stats.csv",
            f"{nsys_file}.nsys-rep"
        ]

        result = subprocess.run(stats_cmd, capture_output=True, text=True)
        print(f"GPU stats exported to /tmp/gpu_stats.csv")

        # Also export kernel summary
        kernel_cmd = [
            "nsys", "stats",
            "--report", "gpu_kern_sum",
            "--format", "csv",
            "--output", "/tmp/kernel_stats.csv",
            f"{nsys_file}.nsys-rep"
        ]

        result = subprocess.run(kernel_cmd, capture_output=True, text=True)
        print(f"Kernel stats exported to /tmp/kernel_stats.csv")

        return True
    except Exception as e:
        print(f"Warning: Could not export stats: {e}")
        return False


def analyze_nsys_results():
    """Analyze Nsight Systems results."""
    print(f"\n{'='*80}")
    print("ANALYZING NSIGHT SYSTEMS RESULTS")
    print(f"{'='*80}\n")

    # Try to read GPU stats
    gpu_stats = {}
    kernel_stats = []

    try:
        with open("/tmp/gpu_stats.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("GPU"):
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        gpu_id = parts[0]
                        gpu_stats[gpu_id] = parts[1]
    except Exception as e:
        print(f"Warning: Could not read GPU stats: {e}")

    try:
        with open("/tmp/kernel_stats.csv", "r") as f:
            lines = f.readlines()
            headers = lines[0].strip().split(",") if lines else []
            for line in lines[1:10]:  # Top 10 kernels
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    kernel_stats.append({
                        "name": parts[0] if len(parts) > 0 else "",
                        "total_time": parts[1] if len(parts) > 1 else "",
                        "calls": parts[2] if len(parts) > 2 else ""
                    })
    except Exception as e:
        print(f"Warning: Could not read kernel stats: {e}")

    return gpu_stats, kernel_stats


def run_ncu_sampling(duration=10):
    """Run Nsight Compute sampling."""
    print(f"\n{'='*80}")
    print("RUNNING NSIGHT COMPUTE SAMPLING")
    print(f"{'='*80}\n")

    output_file = "/tmp/ncu_output"

    # Kill any existing ncu sessions
    subprocess.run(["pkill", "-9", "ncu"], stderr=subprocess.DEVNULL)

    # Build ncu command
    cmd = [
        "ncu", "--mode=launch",
        "--set", "full",
        "-o", output_file,
        "./build/inferfluxd",
        "--config", "config/server.cuda.yaml"
    ]

    print(f"Running Nsight Compute sampling for {duration}s...")

    # Run ncu in background
    ncu_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Generate workload
    time.sleep(3)
    generate_workload(num_requests=20, duration_sec=duration-5)

    # Stop ncu
    time.sleep(2)
    ncu_proc.terminate()

    try:
        ncu_proc.wait(timeout=30)
        print(f"✅ NCU sampling complete")
        print(f"Output: {output_file}")

        return output_file
    except subprocess.TimeoutExpired:
        ncu_proc.kill()
        print("ERROR: NCU timed out")
        return None


def validate_assumptions(gpu_stats, kernel_stats):
    """Validate our optimization assumptions."""
    print(f"\n{'='*80}")
    print("VALIDATING ASSUMPTIONS")
    print(f"{'='*80}\n")

    print("## ASSUMPTION 1: GPU Utilization ~15%")
    print("-" * 80)

    # Check if we have GPU utilization data
    gpu_util_found = False
    if gpu_stats:
        for gpu_id, util in gpu_stats.items():
            if "%" in str(util):
                gpu_util_found = True
                try:
                    util_value = float(str(util).replace("%", ""))
                    print(f"Actual GPU utilization: {util_value:.1f}%")

                    if util_value < 20:
                        print("✅ ASSUMPTION VALIDATED: GPU utilization is low (~15%)")
                        print("   → GPU utilization optimization is HIGH PRIORITY")
                        print(f"   → Potential gain: +{(80-util_value)/util_value*100:.0f}% throughput")
                    else:
                        print("⚠️ ASSUMPTION INCORRECT: GPU utilization is higher than expected")
                        print(f"   → Actual: {util_value:.1f}%, Expected: ~15%")
                        print("   → May need different optimization strategy")

                except ValueError:
                    pass

    if not gpu_util_found:
        print("⚠️ Could not determine GPU utilization from profile")
        print("   → Check profile output manually")
        print("   → Use: nsys-ui {output}.nsys-rep")

    print()
    print("## ASSUMPTION 2: Kernel Time Dominates")
    print("-" * 80)

    if kernel_stats:
        print("Top kernels by execution time:")
        total_time = 0
        for i, kernel in enumerate(kernel_stats[:5], 1):
            print(f"  {i}. {kernel['name'][:60]:60s}")
            print(f"     Time: {kernel['total_time']}, Calls: {kernel['calls']}")

        print()
        print("✅ Profile data available for detailed analysis")
        print("   → Run: ncu --set full -o analysis {output}.ncu-rep")
    else:
        print("⚠️ No kernel stats available")
        print("   → Check kernel stats export")

    print()
    print("## RECOMMENDATIONS")
    print("-" * 80)

    if gpu_util_found:
        print("Based on profiling data:")
        print("1. Review full profile in Nsight Systems UI")
        print("   → nsys-ui /tmp/profile_output.nsys-rep")
        print()
        print("2. Identify top time-consuming kernels")
        print("3. Check memory transfer patterns (HtoD, DtoH)")
        print("4. Analyze CPU-GPU overlap opportunities")
        print()
        print("5. Update optimization roadmap based on findings")
    else:
        print("1. Manually review profile output")
        print("2. Check GPU utilization in timeline view")
        print("3. Identify actual bottlenecks")
        print("4. Adjust optimization strategy")


def main():
    print("="*80)
    print("PROFILING VALIDATION")
    print("="*80)
    print()

    # Check tools
    tools = check_tools()
    print("## PROFILING TOOLS")
    print("-"*80)
    for tool, available in tools.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {tool.upper()}")

    if not tools["nsys"]:
        print("\n❌ Nsight Systems not available. Cannot run profiling.")
        print("Install with: apt install nvidia-nsight-systems")
        return

    print()
    print("## PROFILING PLAN")
    print("-"*80)
    print("1. Run Nsight Systems profiling (20s)")
    print("   - Captures timeline, GPU metrics, kernel execution")
    print("   - Workload: 50 requests over 15s")
    print()
    print("2. Export stats to CSV for analysis")
    print()
    print("3. Validate assumptions:")
    print("   - GPU utilization (~15% assumed)")
    print("   - Kernel execution time")
    print("   - Memory transfer patterns")
    print()
    print("4. Generate updated recommendations")

    # Ask for confirmation
    response = input("\nContinue with profiling? (y/n): ")
    if response.lower() != 'y':
        print("Profiling cancelled")
        return

    # Run profiling
    nsys_output = run_nsys_profiling(duration=20, output="profile_output")

    if nsys_output:
        # Export stats
        run_nsys_stats(nsys_output)

        # Analyze results
        gpu_stats, kernel_stats = analyze_nsys_results()

        # Validate assumptions
        validate_assumptions(gpu_stats, kernel_stats)

        print()
        print("="*80)
        print("PROFILING COMPLETE")
        print("="*80)
        print()
        print("Results saved:")
        print(f"  - Profile: {nsys_output}.nsys-rep")
        print(f"  - GPU stats: /tmp/gpu_stats.csv")
        print(f"  - Kernel stats: /tmp/kernel_stats.csv")
        print()
        print("Next steps:")
        print("  1. Open profile: nsys-ui {nsys_output}.nsys-rep")
        print("  2. Review timeline and GPU metrics")
        print("  3. Identify top kernels and optimization opportunities")
        print()

    else:
        print("\n❌ Profiling failed. Check error messages above.")


if __name__ == "__main__":
    main()
