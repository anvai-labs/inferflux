#!/usr/bin/env python3
"""
Scheduler Bottleneck Profiler

Analyzes InferFlux scheduler behavior to identify concurrent throughput bottlenecks.
Measures:
1. Batch size distribution (how often do we hit max_batch_size?)
2. Time spent in batch building vs GPU execution
3. Queue depth over time
4. Mutex contention (via timing patterns)

Usage:
    python3 scripts/profile_scheduler_bottleneck.py --server-url http://localhost:8080 \
        --concurrency 4 --num-requests 32 --output scheduler_profile.json
"""

import argparse
import json
import os
import statistics
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import http.client
import urllib.parse


# ============================================================================
# Configuration
# ============================================================================

API_KEY = "dev-key-123"
DEFAULT_PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France? Answer in one word.",
    "Translate 'hello world' to Spanish.",
    "List three prime numbers greater than 10.",
]


# ============================================================================
# ANSI Colors
# ============================================================================

class C:
    R = "\033[0;31m"
    G = "\033[0;32m"
    Y = "\033[1;33m"
    B = "\033[0;34m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def log(level: str, msg: str):
    colors = {"INFO": C.B, "OK": C.G, "WARN": C.Y, "ERR": C.R}
    print(f"{colors.get(level, '')}{f'[{level}]':>8}{C.NC} {msg}")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    success: bool
    tokens: int
    latency_ms: float
    time_to_first_token_ms: float = 0.0
    error: str = ""


@dataclass
class SchedulerMetrics:
    """Metrics collected from /metrics endpoint"""
    timestamp: float
    batch_size_p50: float = 0.0
    batch_size_p95: float = 0.0
    batch_size_p99: float = 0.0
    queue_depth: int = 0
    active_requests: int = 0
    forward_passes_total: int = 0
    sampled_tokens_total: int = 0


# ============================================================================
# HTTP Client
# ============================================================================

def send_completion(host: str, port: int, prompt: str,
                   max_tokens: int, model: str,
                   request_id: int) -> RequestResult:
    """Send a completion request and return timing info"""
    start = time.perf_counter()

    try:
        conn = http.client.HTTPConnection(host, port, timeout=120)
        prompt_json = json.dumps(prompt)

        payload = json.dumps({
            "model": model,
            "prompt": prompt_json,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        })

        conn.request("POST", "/v1/completions",
                    body=payload,
                    headers={"Content-Type": "application/json",
                            "Authorization": f"Bearer {API_KEY}"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            start_time=start,
            end_time=time.perf_counter(),
            success=False,
            tokens=0,
            latency_ms=0.0,
            error=str(exc)
        )

    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0

    if resp.status != 200:
        return RequestResult(
            request_id=request_id,
            start_time=start,
            end_time=end,
            success=False,
            tokens=0,
            latency_ms=latency_ms,
            error=f"HTTP {resp.status}: {body[:200]}"
        )

    try:
        data = json.loads(body)
        text = data.get("choices", [{}])[0].get("text", "")
        tokens = data.get("usage", {}).get("completion_tokens", 0)
    except Exception:
        text = body
        tokens = 0

    return RequestResult(
        request_id=request_id,
        start_time=start,
        end_time=end,
        success=True,
        tokens=tokens,
        latency_ms=latency_ms
    )


# ============================================================================
# Metrics Scraper
# ============================================================================

class MetricsScraper:
    """Background thread that scrapes Prometheus metrics"""

    def __init__(self, host: str, port: int, interval_ms: int = 100):
        self.host = host
        self.port = port
        self.interval_ms = interval_ms
        self.metrics: List[SchedulerMetrics] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _scrape_loop(self):
        while self.running:
            try:
                metrics = self._scrape_metrics()
                if metrics:
                    self.metrics.append(metrics)
            except Exception as e:
                log("WARN", f"Metrics scrape failed: {e}")
            time.sleep(self.interval_ms / 1000.0)

    def _scrape_metrics(self) -> Optional[SchedulerMetrics]:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=2)
            conn.request("GET", "/metrics",
                        headers={"Authorization": f"Bearer {API_KEY}"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()

            if resp.status != 200:
                return None

            return self._parse_metrics(body)
        except Exception:
            return None

    def _parse_metrics(self, body: str) -> Optional[SchedulerMetrics]:
        """Parse Prometheus metrics text"""
        metrics = SchedulerMetrics(timestamp=time.time())

        batch_sizes = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse metric lines
            # inferflux_queue_depth
            if "inferflux_queue_depth" in line:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics.queue_depth = int(float(parts[-1]))
                except:
                    pass

            # inferflux_native_forward_batch_size_total (histogram)
            if "inferflux_native_forward_batch_size_total{" in line and "bucket" in line:
                try:
                    # Extract bucket value
                    if 'le="1"' in line or 'le="2"' in line or 'le="4"' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            bucket_val = float(parts[-1])
                            if bucket_val > 0:  # Count only non-zero buckets
                                batch_sizes.append(bucket_val)
                except:
                    pass

            # inferflux_native_forward_passes_total
            if line.startswith("inferflux_native_forward_passes_total{") and "phase=" in line:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics.forward_passes_total += int(float(parts[-1]))
                except:
                    pass

        if batch_sizes:
            batch_sizes_sorted = sorted(batch_sizes)
            n = len(batch_sizes_sorted)
            metrics.batch_size_p50 = batch_sizes_sorted[int(n * 0.50)]
            metrics.batch_size_p95 = batch_sizes_sorted[int(n * 0.95)]
            metrics.batch_size_p99 = batch_sizes_sorted[int(n * 0.99)]

        return metrics


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_batch_size_limit(metrics_history: List[SchedulerMetrics],
                            max_batch_size: int = 4) -> Dict:
    """Analyze how often we hit the batch size limit"""
    hit_limit_count = 0
    total_samples = len(metrics_history)

    for m in metrics_history:
        if m.batch_size_p95 >= max_batch_size * 0.9:  # 90% of max
            hit_limit_count += 1

    return {
        "total_samples": total_samples,
        "hit_limit_count": hit_limit_count,
        "hit_limit_ratio": hit_limit_count / total_samples if total_samples > 0 else 0.0,
        "interpretation": (
            f"Batch size limit hit {hit_limit_count}/{total_samples} times "
            f"({100 * hit_limit_count / total_samples:.1f}%)"
        )
    }


def analyze_queue_depth_patterns(metrics_history: List[SchedulerMetrics]) -> Dict:
    """Analyze queue depth patterns"""
    queue_depths = [m.queue_depth for m in metrics_history]

    if not queue_depths:
        return {"error": "No queue depth data"}

    return {
        "min": min(queue_depths),
        "max": max(queue_depths),
        "mean": statistics.mean(queue_depths),
        "median": statistics.median(queue_depths),
        "p95": statistics.quantiles(queue_depths, n=20)[18] if len(queue_depths) >= 20 else max(queue_depths),
        "interpretation": (
            f"Queue depth averaged {statistics.mean(queue_depths):.1f}, "
            f"peaked at {max(queue_depths)}"
        )
    }


def estimate_serialization_overhead(results: List[RequestResult],
                                   concurrency: int) -> Dict:
    """Estimate scheduler serialization overhead"""
    if not results:
        return {"error": "No results"}

    # Calculate expected vs actual throughput
    successful = [r for r in results if r.success]
    if not successful:
        return {"error": "No successful requests"}

    total_time = max(r.end_time for r in successful) - min(r.start_time for r in successful)
    total_tokens = sum(r.tokens for r in successful)
    actual_tok_per_sec = total_tokens / total_time

    # Estimate ideal (linear scaling)
    avg_latency = statistics.mean(r.latency_ms for r in successful)
    ideal_tok_per_req = statistics.mean(r.tokens for r in successful)
    ideal_concurrent_tok_per_sec = (concurrency * ideal_tok_per_req) / (avg_latency / 1000.0)

    scaling_efficiency = actual_tok_per_sec / ideal_concurrent_tok_per_sec

    return {
        "actual_tok_per_sec": actual_tok_per_sec,
        "ideal_tok_per_sec": ideal_concurrent_tok_per_sec,
        "scaling_efficiency": scaling_efficiency,
        "interpretation": (
            f"Scaling efficiency: {scaling_efficiency:.1%}. "
            f"Values < 80% indicate serialization bottleneck."
        )
    }


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(host: str, port: int, concurrency: int,
                 num_requests: int, max_tokens: int) -> tuple[List[RequestResult], List[SchedulerMetrics]]:
    """Run concurrent benchmark with metrics scraping"""

    log("INFO", f"Running {num_requests} requests (concurrency={concurrency})...")

    # Start metrics scraper
    scraper = MetricsScraper(host, port, interval_ms=100)
    scraper.start()

    # Build request list
    requests = []
    for i in range(num_requests):
        prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        requests.append((prompt, i))

    # Run benchmark
    results: List[RequestResult] = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for prompt, req_id in requests:
            future = executor.submit(send_completion, host, port, prompt, max_tokens, "default", req_id)
            futures[future] = req_id

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    end_time = time.perf_counter()

    # Stop metrics scraper
    scraper.stop()
    metrics_history = scraper.metrics

    total_time = end_time - start_time
    successful = [r for r in results if r.success]
    total_tokens = sum(r.tokens for r in successful)

    log("OK", f"Completed: {len(successful)}/{num_requests} OK, "
        f"{total_tokens / total_time:.1f} tok/s, {total_time:.2f}s")

    return results, metrics_history


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: List[RequestResult],
                   metrics_history: List[SchedulerMetrics],
                   concurrency: int,
                   max_batch_size: int = 4):
    """Generate comprehensive analysis report"""

    print()
    print(f"{C.BOLD}{'=' * 72}{C.NC}")
    print(f"{C.BOLD}  Scheduler Bottleneck Analysis{C.NC}")
    print(f"{C.BOLD}{'=' * 72}{C.NC}")
    print()

    # Batch size limit analysis
    print(f"{C.BOLD}Batch Size Limit Analysis{C.NC}")
    batch_analysis = analyze_batch_size_limit(metrics_history, max_batch_size)
    if "error" not in batch_analysis:
        print(f"  {batch_analysis['interpretation']}")
        if batch_analysis['hit_limit_ratio'] > 0.5:
            print(f"  {C.Y}RECOMMENDATION: Increase max_batch_size (current={max_batch_size}){C.NC}")
        else:
            print(f"  {C.G}OK: Batch size limit not a bottleneck{C.NC}")
    print()

    # Queue depth analysis
    print(f"{C.BOLD}Queue Depth Patterns{C.NC}")
    queue_analysis = analyze_queue_depth_patterns(metrics_history)
    if "error" not in queue_analysis:
        print(f"  Min: {queue_analysis['min']}, Max: {queue_analysis['max']}, "
              f"Mean: {queue_analysis['mean']:.1f}, Median: {queue_analysis['median']:.1f}")
        print(f"  {queue_analysis['interpretation']}")
        if queue_analysis['mean'] > concurrency * 0.5:
            print(f"  {C.Y}WARN: High queue depth indicates scheduler bottleneck{C.NC}")
    print()

    # Scaling efficiency
    print(f"{C.BOLD}Scaling Efficiency{C.NC}")
    scaling_analysis = estimate_serialization_overhead(results, concurrency)
    if "error" not in scaling_analysis:
        print(f"  Actual throughput: {scaling_analysis['actual_tok_per_sec']:.1f} tok/s")
        print(f"  Ideal throughput:  {scaling_analysis['ideal_tok_per_sec']:.1f} tok/s")
        print(f"  Efficiency:        {scaling_analysis['scaling_efficiency']:.1%}")
        print(f"  {scaling_analysis['interpretation']}")
        if scaling_analysis['scaling_efficiency'] < 0.7:
            print(f"  {C.R}BOTTLENECK: Poor scaling indicates serialization issue{C.NC}")
        elif scaling_analysis['scaling_efficiency'] < 0.85:
            print(f"  {C.Y}WARN: Suboptimal scaling{C.NC}")
        else:
            print(f"  {C.G}OK: Good scaling efficiency{C.NC}")
    print()

    # Recommendations
    print(f"{C.BOLD}Recommendations{C.NC}")

    recommendations = []

    if batch_analysis.get('hit_limit_ratio', 0) > 0.5:
        recommendations.append(
            f"1. Increase max_batch_size from {max_batch_size} to 8-16 (config/server.yaml)"
        )

    if queue_analysis.get('mean', 0) > concurrency * 0.3:
        recommendations.append(
            f"2. Investigate global mutex contention in BuildBatchLocked()"
        )

    if scaling_analysis.get('scaling_efficiency', 1.0) < 0.7:
        recommendations.append(
            f"3. Profile with Nsight Systems to identify GPU kernel bottlenecks"
        )
        recommendations.append(
            f"4. Consider lock-free batch building data structures"
        )

    if not recommendations:
        recommendations.append("No bottlenecks detected - scheduler is healthy")

    for rec in recommendations:
        print(f"  {rec}")

    print(f"{'=' * 72}")
    print()

    return {
        "batch_analysis": batch_analysis,
        "queue_analysis": queue_analysis,
        "scaling_analysis": scaling_analysis,
        "recommendations": recommendations
    }


def save_results(results: List[RequestResult],
                metrics_history: List[SchedulerMetrics],
                analysis: Dict,
                output_path: str):
    """Save results to JSON file"""

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "request_id": r.request_id,
                "success": r.success,
                "tokens": r.tokens,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
        "metrics_history": [
            {
                "timestamp": m.timestamp,
                "queue_depth": m.queue_depth,
                "batch_size_p50": m.batch_size_p50,
                "batch_size_p95": m.batch_size_p95,
            }
            for m in metrics_history
        ],
        "analysis": analysis
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    log("INFO", f"Results saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Profile scheduler bottlenecks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server-url", default="http://localhost:8080",
                        help="InferFlux server URL")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent request count")
    parser.add_argument("--num-requests", type=int, default=32,
                        help="Total number of requests")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens per request")
    parser.add_argument("--max-batch-size", type=int, default=4,
                        help="Scheduler max_batch_size (for analysis)")
    parser.add_argument("--output", default="",
                        help="JSON output file")

    args = parser.parse_args()

    # Parse server URL
    parsed = urllib.parse.urlparse(args.server_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    # Run benchmark
    results, metrics_history = run_benchmark(
        host, port, args.concurrency, args.num_requests, args.max_tokens
    )

    # Generate report
    analysis = generate_report(results, metrics_history, args.concurrency, args.max_batch_size)

    # Save results
    if args.output:
        save_results(results, metrics_history, analysis, args.output)
    else:
        output_path = f"scheduler_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, metrics_history, analysis, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
