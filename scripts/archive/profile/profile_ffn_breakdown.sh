#!/bin/bash
# Profile FFN execution breakdown to identify optimization opportunities
#
# This script runs the server with detailed timing and extracts FFN phase timings

set -e

echo "========================================="
echo "FFN Performance Profiling"
echo "========================================="
echo ""

# Check if Nsight Systems is available
if ! command -v nsys &> /dev/null; then
    echo "⚠️  Nsight Systems not found"
    echo "   Install with: sudo apt-get install nsight-systems"
    echo "   Falling back to built-in timing metrics..."
    echo ""
    USE_NSIGHT=0
else
    echo "✅ Nsight Systems found - will create detailed trace"
    echo ""
    USE_NSIGHT=1
fi

# Configuration
MODEL="${MODEL:-tinyllama}"
CONFIG_FILE="${CONFIG_FILE:-/tmp/benchmark_native_cuda.yaml}"
SERVER_BIN="${SERVER_BIN:-./build/inferfluxd}"
DURATION="${DURATION:-60}"  # seconds
NUM_REQUESTS="${NUM_REQUESTS:-32}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Config: $CONFIG_FILE"
echo "  Duration: ${DURATION}s"
echo "  Num requests: $NUM_REQUESTS"
echo ""

# Enable detailed phase timing
export INFERFLUX_NATIVE_PHASE_TIMING=1
export INFERCTL_API_KEY=dev-key-123

# Check if server binary exists
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ Server binary not found: $SERVER_BIN"
    echo "   Build with: cmake --build build"
    exit 1
fi

# Create results directory
RESULTS_DIR="/tmp/ffn_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Profiling..."
echo "  Results dir: $RESULTS_DIR"
echo ""

# Start server in background
if [ "$USE_NSIGHT" -eq 1 ]; then
    echo "Starting server with Nsight Systems profiling..."
    nsys profile -o "$RESULTS_DIR/ffn_profile" \
        --trace=cuda,nvtx,osrt \
        --force-overwrite=true \
        --duration=$DURATION \
        $SERVER_BIN --config "$CONFIG_FILE" &
    SERVER_PID=$!
else
    echo "Starting server with phase timing enabled..."
    $SERVER_BIN --config "$CONFIG_FILE" > "$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
fi

# Wait for server to start
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    cat "$RESULTS_DIR/server.log" 2>/dev/null || true
    exit 1
fi

echo "✅ Server started (PID: $SERVER_PID)"
echo ""

# Run workload
echo "Running workload..."
python3 << EOF
import requests
import json
import time

url = "http://127.0.0.1:8668/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dev-key-123"
}

# Simple prompts to generate FFN load
prompts = [
    "Explain quantum computing in one sentence.",
    "What is machine learning?",
    "Describe a sunset.",
    "How does a computer work?",
]

for i in range($NUM_REQUESTS):
    prompt = prompts[i % len(prompts)]
    data = {
        "model": "$MODEL",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "stream": False
    }

    try:
        start = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=30)
        elapsed = time.time() - start

        if response.status_code == 200:
            print(f"  Request {i+1}/$NUM_REQUESTS: {elapsed*1000:.0f}ms")
        else:
            print(f"  Request {i+1}/$NUM_REQUESTS: ERROR {response.status_code}")
    except Exception as e:
        print(f"  Request {i+1}/$NUM_REQUESTS: {e}")

print("\nWorkload complete")
EOF

echo ""
echo "Stopping server..."
kill -TERM $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Analyze results
echo ""
echo "========================================="
echo "Analysis Results"
echo "========================================="
echo ""

if [ "$USE_NSIGHT" -eq 1 ]; then
    echo "Nsight Systems profile saved to: $RESULTS_DIR/ffn_profile.nsys-rep"
    echo ""
    echo "To view the profile:"
    echo "  nsys-ui $RESULTS_DIR/ffn_profile.nsys-rep"
    echo ""
    echo "To export statistics:"
    echo "  nsys stats $RESULTS_DIR/ffn_profile.nsys-rep --report gpumemsizesum,gpukernsum"
else
    # Parse server log for phase timings
    if [ -f "$RESULTS_DIR/server.log" ]; then
        echo "Phase Timing Breakdown:"
        echo ""
        grep "ffn_proj\|ffn_silu\|ffn_down" "$RESULTS_DIR/server.log" | \
            tail -20 || echo "  No phase timing data found in log"
    fi
fi

echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "========================================="
echo "Profiling Complete"
echo "========================================="
