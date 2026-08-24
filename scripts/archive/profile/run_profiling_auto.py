#!/usr/bin/env python3
"""
Automated profiling to validate optimization assumptions.
"""

import subprocess
import time
import sys


def run_profiling():
    """Run Nsight Systems profiling."""
    print("="*80)
    print("RUNNING PROFILING VALIDATION")
    print("="*80)
    print()

    # Kill any existing processes
    subprocess.run(["pkill", "-9", "-f", "inferfluxd"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "nsys"], stderr=subprocess.DEVNULL)
    time.sleep(2)

    output_file = "/tmp/profile_validation"
    duration = 15  # seconds

    print(f"Starting Nsight Systems profiling ({duration}s)...")
    print(f"Output: {output_file}.nsys-rep")
    print()

    # Start profiling with server
    cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx",
        "-y",
        "-o", output_file,
        "-d", str(duration),
        "--force-overwrite=true",
        "./build/inferfluxd",
        "--config", "config/server.cuda.yaml"
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Start profiling
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(4)

    # Generate workload
    print("Generating inference workload...")
    requests_sent = 0
    start_time = time.time()
    end_time = start_time + (duration - 5)

    while time.time() < end_time and requests_sent < 40:
        try:
            subprocess.run([
                "./build/inferctl", "complete",
                "--model", "tinyllama",
                "--prompt", "Hello",
                "--max-tokens", "30",
                "--no-stream"
            ], capture_output=True, timeout=5)
            requests_sent += 1
            time.sleep(0.3)
        except:
            pass

    elapsed = time.time() - start_time
    print(f"Sent {requests_sent} requests in {elapsed:.1f}s")

    # Wait for profiling to complete
    print("Waiting for profiling to complete...")
    try:
        stdout, stderr = proc.communicate(timeout=20)
        print()
        print("="*80)
        print("PROFILING COMPLETE")
        print("="*80)
        print()
        print(f"Profile saved: {output_file}.nsys-rep")
        print()

        # Export stats
        print("Exporting statistics...")
        export_cmd = [
            "nsys", "stats",
            "--report", "gpu_sum",
            "--format", "csv",
            "--output", "/tmp/gpu_sum.csv",
            f"{output_file}.nsys-rep"
        ]

        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU summary: /tmp/gpu_sum.csv")
        else:
            print(f"⚠️ Export failed: {result.stderr}")

        # Export kernel stats
        kernel_cmd = [
            "nsys", "stats",
            "--report", "gpu_kern_sum",
            "--format", "csv",
            "--output", "/tmp/kernel_sum.csv",
            f"{output_file}.nsys-rep"
        ]

        result = subprocess.run(kernel_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Kernel summary: /tmp/kernel_sum.csv")

        # Try to get GPU utilization
        print()
        print("ANALYZING RESULTS...")
        print("-" * 80)

        try:
            with open("/tmp/gpu_sum.csv", "r") as f:
                content = f.read()
                lines = content.split("\n")
                for line in lines[:20]:
                    if "GPU" in line or "%" in line:
                        print(line)
        except:
            print("Could not read GPU summary")

        print()
        print("TOP KERNELS:")
        try:
            with open("/tmp/kernel_sum.csv", "r") as f:
                lines = f.readlines()
                for line in lines[:15]:
                    print(line.rstrip())
        except:
            print("Could not read kernel summary")

        print()
        print("="*80)
        print("NEXT STEPS")
        print("="*80)
        print()
        print("1. Open profile in GUI:")
        print(f"   nsys-ui {output_file}.nsys-rep")
        print()
        print("2. Review timeline view:")
        print("   - Check GPU utilization bars")
        print("   - Look for gaps in kernel execution")
        print("   - Identify memory transfer patterns")
        print()
        print("3. Check detailed stats:")
        print("   cat /tmp/gpu_sum.csv")
        print("   cat /tmp/kernel_sum.csv")
        print()
        print("4. Validate assumptions:")
        print("   - Is GPU utilization ~15%?")
        print("   - Which kernels dominate execution time?")
        print("   - Are there memory transfer bottlenecks?")
        print()

    except subprocess.TimeoutExpired:
        proc.kill()
        print("ERROR: Profiling timed out")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_profiling())
