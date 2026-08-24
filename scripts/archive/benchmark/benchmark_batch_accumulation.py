#!/usr/bin/env python3
"""
Benchmark batch accumulation delay optimization.

Tests different batch accumulation times to measure:
1. Throughput improvement (tokens/sec)
2. GPU utilization improvement
3. Latency impact (p50, p95)

Expected results from profiling analysis:
- GPU utilization: 5% → 60-80%
- Throughput: 255 → 600-800 tok/s (+135-214%)
- p50 Latency: +5-10ms (acceptable trade-off)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark batch accumulation delay"
    )
    parser.add_argument(
        "--server-bin",
        default="./build/inferfluxd",
        help="Path to inferfluxd binary",
    )
    parser.add_argument(
        "--config",
        default="config/server.cuda.yaml",
        help="Server config file",
    )
    parser.add_argument(
        "--model",
        default="tinyllama",
        help="Model identifier",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        help="Backend to use",
    )
    parser.add_argument(
        "--accumulation-times",
        default="0,5,10,20",
        help="Comma-separated list of accumulation times to test (ms)",
    )
    parser.add_argument(
        "--min-batch-sizes",
        default="1,4,8,16",
        help="Comma-separated list of min batch sizes to test",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests to send per test",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results",
    )
    return parser.parse_args()


def run_benchmark(
    server_bin: str,
    config: str,
    model: str,
    backend: str,
    accumulation_ms: int,
    min_batch_size: int,
    requests: int,
) -> Dict:
    """Run a single benchmark with given parameters."""

    print(f"\n{'='*70}")
    print(f"Testing: accumulation_ms={accumulation_ms}, min_batch_size={min_batch_size}")
    print(f"{'='*70}")

    # Set environment variables for batch accumulation
    env = os.environ.copy()
    env["INFERFLUX_SCHED_BATCH_ACCUMULATION_MS"] = str(accumulation_ms)
    env["INFERFLUX_SCHED_MIN_BATCH_SIZE"] = str(min_batch_size)

    # Start server
    print(f"Starting server with batch_accumulation_ms={accumulation_ms}, min_batch_size={min_batch_size}...")
    server_proc = subprocess.Popen(
        [server_bin, "--config", config],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to start
    time.sleep(3)

    # Check if server started successfully
    if server_proc.poll() is not None:
        stdout, stderr = server_proc.communicate()
        print(f"ERROR: Server failed to start")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return None

    # Run throughput gate
    print(f"Running throughput gate with {requests} requests...")
    try:
        result = subprocess.run(
            [
                "python3",
                "scripts/run_throughput_gate.py",
                "--server-bin", server_bin,
                "--config", config,
                "--model", model,
                "--backend", backend,
                "--num-requests", str(requests),
                "--min-completion-tok-per-sec", "100",  # Lower threshold for testing
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        output = result.stdout + result.stderr

        # Parse metrics from output
        metrics = parse_throughput_output(output)

        # Add test parameters
        metrics["accumulation_ms"] = accumulation_ms
        metrics["min_batch_size"] = min_batch_size
        metrics["requests_sent"] = requests

        print(f"Results: {metrics}")

    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out")
        metrics = None
    except Exception as e:
        print(f"ERROR: {e}")
        metrics = None
    finally:
        # Stop server
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()

    return metrics


def parse_throughput_output(output: str) -> Dict:
    """Parse metrics from throughput gate output."""

    metrics = {
        "throughput_tok_per_sec": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "tokens_processed": 0,
        "gpu_utilization_percent": 0.0,
        "batch_size_avg": 0.0,
    }

    # Parse throughput (tokens/sec)
    for line in output.split("\n"):
        if "tok/s" in line.lower() or "tokens/sec" in line.lower():
            # Look for patterns like "254.64 tok/s"
            import re
            match = re.search(r"(\d+\.?\d*)\s*tok/s", line)
            if match:
                metrics["throughput_tok_per_sec"] = float(match.group(1))

        # Parse latency
        if "p50" in line.lower() or "median" in line.lower():
            import re
            match = re.search(r"p50[:\s]+(\d+\.?\d*)\s*ms", line, re.IGNORECASE)
            if match:
                metrics["p50_latency_ms"] = float(match.group(1))

        if "p95" in line.lower():
            import re
            match = re.search(r"p95[:\s]+(\d+\.?\d*)\s*ms", line, re.IGNORECASE)
            if match:
                metrics["p95_latency_ms"] = float(match.group(1))

        # Parse GPU utilization
        if "gpu" in line.lower() and ("util" in line.lower() or "%" in line):
            import re
            match = re.search(r"(\d+\.?\d*)\s*%", line)
            if match:
                metrics["gpu_utilization_percent"] = float(match.group(1))

    return metrics


def compare_results(results: List[Dict]) -> None:
    """Compare results and print summary."""

    if not results:
        print("No results to compare")
        return

    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Find baseline (accumulation_ms=0)
    baseline = next((r for r in results if r["accumulation_ms"] == 0), None)
    if not baseline:
        baseline = results[0]

    print(f"\nBaseline (accumulation_ms={baseline['accumulation_ms']}, min_batch_size={baseline['min_batch_size']}):")
    print(f"  Throughput: {baseline['throughput_tok_per_sec']:.2f} tok/s")
    print(f"  p50 Latency: {baseline['p50_latency_ms']:.2f} ms")
    print(f"  p95 Latency: {baseline['p95_latency_ms']:.2f} ms")
    print(f"  GPU Util: {baseline['gpu_utilization_percent']:.1f}%")

    print(f"\n{'Accumulation (ms)':<15} {'Min Batch':<10} {'Throughput':<12} {'GPU Util':<10} {'p50 Latency':<12} {'Improvement':<12}")
    print("-" * 90)

    for result in results:
        acc_ms = result["accumulation_ms"]
        min_batch = result["min_batch_size"]
        throughput = result["throughput_tok_per_sec"]
        gpu_util = result["gpu_utilization_percent"]
        p50 = result["p50_latency_ms"]

        # Calculate improvement
        throughput_improvement = ((throughput - baseline['throughput_tok_per_sec']) /
                                  baseline['throughput_tok_per_sec'] * 100) if baseline['throughput_tok_per_sec'] > 0 else 0

        gpu_improvement = ((gpu_util - baseline['gpu_utilization_percent']) /
                           baseline['gpu_utilization_percent'] * 100) if baseline['gpu_utilization_percent'] > 0 else 0

        print(f"{acc_ms:<15} {min_batch:<10} {throughput:<12.2f} {gpu_util:<10.1f}% {p50:<12.2f} {throughput_improvement:+.1f}%")

    # Find best configuration
    best = max(results, key=lambda r: r["throughput_tok_per_sec"])
    print(f"\n🏆 Best configuration: accumulation_ms={best['accumulation_ms']}, min_batch_size={best['min_batch_size']}")
    print(f"   Throughput: {best['throughput_tok_per_sec']:.2f} tok/s")
    print(f"   Improvement over baseline: {((best['throughput_tok_per_sec'] - baseline['throughput_tok_per_sec']) / baseline['throughput_tok_per_sec'] * 100):.1f}%")


def main():
    args = parse_args()

    # Parse test parameters
    accumulation_times = [int(x) for x in args.accumulation_times.split(",")]
    min_batch_sizes = [int(x) for x in args.min_batch_sizes.split(",")]

    # Check server binary exists
    if not Path(args.server_bin).exists():
        print(f"ERROR: Server binary not found: {args.server_bin}")
        sys.exit(1)

    # Check config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    results = []

    # Run benchmarks for each combination
    for acc_ms in accumulation_times:
        for min_batch in min_batch_sizes:
            metrics = run_benchmark(
                args.server_bin,
                args.config,
                args.model,
                args.backend,
                acc_ms,
                min_batch,
                args.requests,
            )

            if metrics:
                results.append(metrics)

            # Cool-down period
            time.sleep(2)

    # Save results
    if results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

        # Compare results
        compare_results(results)
    else:
        print("\n❌ No valid results collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
