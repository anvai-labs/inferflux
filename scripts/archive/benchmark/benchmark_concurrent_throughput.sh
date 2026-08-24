#!/bin/bash
set -e

# Benchmark: Concurrent throughput with 16 workers
# This test creates measurable load to validate performance improvements

API_KEY="dev-key-123"
BASE_URL="${INFERFLUX_URL:-http://localhost:8080}"
NUM_REQUESTS=16
MAX_TOKENS=100
EXPECTED_TOTAL_TOKENS=$((NUM_REQUESTS * MAX_TOKENS))

echo "========================================"
echo "Concurrent Throughput Benchmark"
echo "========================================"
echo "Workers: 16 (default)"
echo "Requests: $NUM_REQUESTS"
echo "Max tokens per request: $MAX_TOKENS"
echo "Expected total tokens: $EXPECTED_TOTAL_TOKENS"
echo ""

# Use Python for better timing precision
cat > /tmp/benchmark_concurrent.py <<'PYEOF'
import time
import requests
import concurrent.futures
import json

API_KEY = "dev-key-123"
BASE_URL = "http://localhost:8080"
NUM_REQUESTS = 16
MAX_TOKENS = 100

def make_request(request_id):
    """Make a single request and return timing info"""
    start = time.time()

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": f"Test request {request_id}. Please respond with exactly {MAX_TOKENS} words."}
            ],
            "max_tokens": MAX_TOKENS,
            "stream": False
        },
        timeout=300
    )

    end = time.time()
    elapsed = end - start

    if response.status_code == 200:
        data = response.json()
        tokens = data["usage"]["completion_tokens"]
        return {
            "request_id": request_id,
            "success": True,
            "tokens": tokens,
            "time": elapsed,
            "tok_per_sec": tokens / elapsed if elapsed > 0 else 0
        }
    else:
        return {
            "request_id": request_id,
            "success": False,
            "status": response.status_code,
            "time": elapsed
        }

def run_benchmark(streaming=False):
    """Run concurrent benchmark"""
    print(f"\n{'=' * 40}")
    print(f"Test: {'Streaming' if streaming else 'Non-Streaming'}")
    print(f"{'=' * 40}")

    start_time = time.time()

    # Update the request function for streaming
    if streaming:
        def make_request_streaming(request_id):
            start = time.time()
            tokens = 0

            try:
                response = requests.post(
                    f"{BASE_URL}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "default",
                        "messages": [
                            {"role": "user", "content": f"Test request {request_id}. Please respond with exactly {MAX_TOKENS} words."}
                        ],
                        "max_tokens": MAX_TOKENS,
                        "stream": True
                    },
                    stream=True,
                    timeout=300
                )

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    tokens += len(content.split())
                            except:
                                pass

                end = time.time()
                elapsed = end - start

                return {
                    "request_id": request_id,
                    "success": True,
                    "tokens": tokens,
                    "time": elapsed,
                    "tok_per_sec": tokens / elapsed if elapsed > 0 else 0
                }

            except Exception as e:
                end = time.time()
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "time": end - start
                }

        request_func = make_request_streaming
    else:
        request_func = make_request

    # Run all requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        futures = [executor.submit(request_func, i) for i in range(NUM_REQUESTS)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    total_tokens = sum(r.get('tokens', 0) for r in successful)
    avg_tok_per_sec = sum(r.get('tok_per_sec', 0) for r in successful) / len(successful) if successful else 0

    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {len(successful)}/{NUM_REQUESTS}")
    print(f"Failed requests: {len(failed)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / total_time:.2f} tok/s")
    print(f"Average per-request: {avg_tok_per_sec:.2f} tok/s")

    if failed:
        print(f"\nFailed requests:")
        for r in failed:
            print(f"  Request {r['request_id']}: {r.get('status', r.get('error', 'unknown'))}")

    return {
        "total_time": total_time,
        "successful": len(successful),
        "total_tokens": total_tokens,
        "throughput": total_tokens / total_time,
        "avg_per_request": avg_tok_per_sec
    }

if __name__ == "__main__":
    # Test non-streaming first
    non_streaming = run_benchmark(streaming=False)

    # Brief pause
    time.sleep(2)

    # Test streaming
    streaming = run_benchmark(streaming=True)

    # Summary
    print(f"\n{'=' * 40}")
    print("Summary")
    print(f"{'=' * 40}")
    print(f"Non-Streaming: {non_streaming['throughput']:.2f} tok/s ({non_streaming['total_time']:.2f}s)")
    print(f"Streaming:     {streaming['throughput']:.2f} tok/s ({streaming['total_time']:.2f}s)")

    if streaming['total_time'] > 0:
        speedup = non_streaming['total_time'] / streaming['total_time']
        print(f"Streaming speedup: {speedup:.2f}x")
PYEOF

# Run the benchmark
python3 /tmp/benchmark_concurrent.py

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
