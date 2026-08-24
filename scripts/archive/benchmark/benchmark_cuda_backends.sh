#!/bin/bash
# Benchmark script for llama.cpp vs Native CUDA backends

set -e

MODEL="qwen2.5-coder-14b"
PROMPT="Write a fibonacci function in Python with proper docstring"
MAX_TOKENS=50
NUM_REQUESTS=10
SERVER_URL="http://127.0.0.1:8080"
API_KEY="dev-key-123"

echo "=========================================="
echo "CUDA Backend Benchmark"
echo "Model: $MODEL"
echo "Requests: $NUM_REQUESTS"
echo "Max tokens: $MAX_TOKENS"
echo "=========================================="
echo ""

# Function to run benchmark
run_benchmark() {
    local backend_name=$1
    local num_requests=$2

    echo "=== Benchmark: $backend_name ($num_requests concurrent requests) ==="

    local start_time=$(date +%s.%N)

    # Launch all requests in background
    for i in $(seq 1 $num_requests); do
        curl -s -X POST "$SERVER_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}" &
    done

    # Wait for all to complete
    wait

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    echo "Completed in: $duration seconds"
    echo "Throughput: $(echo "scale=2; $num_requests / $duration" | bc) req/s"
    echo ""
}

# Check server is running
echo "Checking server health..."
if ! curl -s "$SERVER_URL/healthz" > /dev/null 2>&1; then
    echo "ERROR: Server not running at $SERVER_URL"
    exit 1
fi
echo "Server is healthy"
echo ""

# Run benchmarks
run_benchmark "Single Request" 1
run_benchmark "10 Concurrent" 10
run_benchmark "50 Concurrent" 50

echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
