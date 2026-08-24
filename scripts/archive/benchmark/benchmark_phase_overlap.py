#!/usr/bin/env python3
"""
Phase Overlap Benchmark Script

This script benchmarks the phase overlap feature by sending mixed workloads
(concurrent prefill and decode requests) and measuring throughput.
"""

import argparse
import json
import statistics
import time
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

try:
    import requests
except ImportError:
    print("ERROR: requests module not found. Install with: pip3 install requests")
    sys.exit(1)


class PhaseOverlapBenchmark:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = "dev-key-123"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def check_health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok" or data.get("model_ready") == True
            return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics", headers=self.headers, timeout=5)
            metrics = {}
            for line in response.text.split('\n'):
                if 'inferflux_cuda_lane' in line and not line.startswith('#'):
                    parts = line.split(' ')
                    if len(parts) == 2:
                        try:
                            metrics[parts[0]] = float(parts[1])
                        except:
                            pass
            return metrics
        except Exception as e:
            print(f"Failed to get metrics: {e}")
            return {}

    def send_completion(self, prompt: str, max_tokens: int = 10, timeout: int = 30) -> Dict[str, Any]:
        """Send a completion request."""
        start_time = time.time()

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "elapsed": elapsed,
                    "tokens": data.get("usage", {}).get("completion_tokens", max_tokens),
                    "prompt_tokens": len(prompt.split())  # Rough estimate
                }
            else:
                return {
                    "success": False,
                    "elapsed": elapsed,
                    "error": response.status_code
                }
        except Exception as e:
            return {
                "success": False,
                "elapsed": time.time() - start_time,
                "error": str(e)
            }

    def benchmark_mixed_workload(self, num_requests: int = 48, concurrency: int = 8) -> Dict[str, Any]:
        """
        Benchmark with mixed workload (long prefill prompts + ongoing decode).

        This creates a scenario where:
        - Some requests have long prompts (prefill)
        - Other requests have short/empty prompts (decode-only)

        This mixed workload should trigger phase overlap.
        """
        print(f"\n{'='*60}")
        print(f"Mixed Workload Benchmark")
        print(f"{'='*60}")
        print(f"Requests: {num_requests}, Concurrency: {concurrency}")
        print(f"Prompt strategy: Long prompts for prefill, empty for decode")

        # Create mixed workload: half long prompts, half decode-only
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 20  # ~400 tokens
        short_prompt = ""  # Decode-only (continuation)

        requests_list = []
        for i in range(num_requests):
            if i % 2 == 0:
                # Long prompt for prefill
                prompt = long_prompt
            else:
                # Short/empty prompt for decode
                prompt = short_prompt
            requests_list.append(prompt)

        # Benchmark
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self.send_completion, prompt): i
                for i, prompt in enumerate(requests_list)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["success"]:
                    print(f"  ✓ Request {len(results)}/{num_requests}: "
                          f"{result['tokens']:.0f} tokens in {result['elapsed']:.3f}s "
                          f"({result['tokens']/result['elapsed']:.1f} tok/s)")
                else:
                    print(f"  ✗ Request {len(results)}/{num_requests} failed: {result.get('error')}")

        total_time = time.time() - start_time

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            total_tokens = sum(r["tokens"] for r in successful)
            avg_latency = statistics.mean(r["elapsed"] for r in successful)
            p50_latency = statistics.median(r["elapsed"] for r in successful)
            throughput = total_tokens / total_time

            stats = {
                "total_requests": num_requests,
                "successful": len(successful),
                "failed": len(failed),
                "total_time": total_time,
                "total_tokens": total_tokens,
                "throughput_tok_per_sec": throughput,
                "avg_latency_s": avg_latency,
                "p50_latency_s": p50_latency,
            }

            print(f"\n{'='*60}")
            print(f"Results:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Total time: {stats['total_time']:.2f}s")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Throughput: {stats['throughput_tok_per_sec']:.1f} tok/s")
            print(f"  Avg latency: {stats['avg_latency_s']:.3f}s")
            print(f"  P50 latency: {stats['p50_latency_s']:.3f}s")
            print(f"{'='*60}")

            return stats
        else:
            print(f"\n{'='*60}")
            print(f"All requests failed!")
            print(f"{'='*60}")
            return None

    def print_overlap_metrics(self, metrics_before: Dict, metrics_after: Dict):
        """Print phase overlap metrics."""
        print(f"\n{'='*60}")
        print(f"Phase Overlap Metrics")
        print(f"{'='*60}")

        # Calculate deltas
        overlap_events = metrics_after.get("inferflux_cuda_lane_overlap_events_total", 0) - \
                        metrics_before.get("inferflux_cuda_lane_overlap_events_total", 0)

        overlap_duration = metrics_after.get("inferflux_cuda_lane_overlap_duration_ms_total", 0) - \
                         metrics_before.get("inferflux_cuda_lane_overlap_duration_ms_total", 0)

        prefill_submissions = metrics_after.get("inferflux_cuda_lane_submissions_total{lane=\"prefill\"}", 0) - \
                             metrics_before.get("inferflux_cuda_lane_submissions_total{lane=\"prefill\"}", 0)

        decode_submissions = metrics_after.get("inferflux_cuda_lane_submissions_total{lane=\"decode\"}", 0) - \
                            metrics_before.get("inferflux_cuda_lane_submissions_total{lane=\"decode\"}", 0)

        prefill_completions = metrics_after.get("inferflux_cuda_lane_completions_total{lane=\"prefill\"}", 0) - \
                             metrics_before.get("inferflux_cuda_lane_completions_total{lane=\"prefill\"}", 0)

        decode_completions = metrics_after.get("inferflux_cuda_lane_completions_total{lane=\"decode\"}", 0) - \
                            metrics_before.get("inferflux_cuda_lane_completions_total{lane=\"decode\"}", 0)

        print(f"  Prefill submissions: {int(prefill_submissions)}")
        print(f"  Decode submissions: {int(decode_submissions)}")
        print(f"  Prefill completions: {int(prefill_completions)}")
        print(f"  Decode completions: {int(decode_completions)}")
        print(f"  Overlap events: {int(overlap_events)}")
        print(f"  Overlap duration: {overlap_duration:.2f}ms")

        if overlap_events > 0:
            avg_overlap_per_event = overlap_duration / overlap_events if overlap_events > 0 else 0
            print(f"  Avg overlap per event: {avg_overlap_per_event:.2f}ms")
            print(f"\n  ✓ Phase overlap is ACTIVE and working!")
        else:
            print(f"\n  ⚠ Phase overlap not triggered - need larger prefill batches")

        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark phase overlap performance")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--api-key", default="dev-key-123", help="API key")
    parser.add_argument("--requests", type=int, default=48, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")

    args = parser.parse_args()

    benchmark = PhaseOverlapBenchmark(base_url=args.url, api_key=args.api_key)

    # Check server health
    print(f"Checking server health at {args.url}...")
    if not benchmark.check_health():
        print("ERROR: Server is not healthy. Make sure the server is running.")
        sys.exit(1)

    print("✓ Server is healthy")

    # Get initial metrics
    metrics_before = benchmark.get_metrics()
    print(f"✓ Initial metrics collected")

    # Run benchmarks
    all_results = []
    for run in range(args.runs):
        print(f"\n\n{'#'*60}")
        print(f"# Run {run + 1}/{args.runs}")
        print(f"{'#'*60}")

        result = benchmark.benchmark_mixed_workload(
            num_requests=args.requests,
            concurrency=args.concurrency
        )

        if result:
            all_results.append(result)

        # Small delay between runs
        if run < args.runs - 1:
            time.sleep(2)

    # Get final metrics
    metrics_after = benchmark.get_metrics()

    # Print overlap metrics
    benchmark.print_overlap_metrics(metrics_before, metrics_after)

    # Print summary if multiple runs
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print(f"Summary Across {len(all_results)} Runs")
        print(f"{'='*60}")

        avg_throughput = statistics.mean(r["throughput_tok_per_sec"] for r in all_results)
        min_throughput = min(r["throughput_tok_per_sec"] for r in all_results)
        max_throughput = max(r["throughput_tok_per_sec"] for r in all_results)

        print(f"  Avg throughput: {avg_throughput:.1f} tok/s")
        print(f"  Min throughput: {min_throughput:.1f} tok/s")
        print(f"  Max throughput: {max_throughput:.1f} tok/s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
