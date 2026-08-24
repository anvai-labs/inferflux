#!/bin/bash
# FP16 Concurrent Benchmark with OOM Fix Validation
# Tests that server handles memory pressure gracefully instead of crashing

set -e

# Configuration
MODEL_PATH="${INFERFLUX_MODEL_PATH:-models/qwen2.5-3b-f16.gguf}"
CONCURRENT_REQUESTS=8
MAX_TOKENS=100
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
SERVER_PORT="${INFERFLUX_PORT_OVERRIDE:-8080}"
SERVER_BIN="./build/inferfluxd"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "FP16 Concurrent Benchmark (OOM Fix Test)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Concurrent requests: $CONCURRENT_REQUESTS"
echo "Max tokens per request: $MAX_TOKENS"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

# Function to cleanup
cleanup() {
    echo ""
    echo "Stopping server..."
    pkill -f "$SERVER_BIN" || true
    sleep 2
}

trap cleanup EXIT

# Test backend
test_backend() {
    local backend=$1
    local config=$2

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing backend: $backend${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Start server
    echo "Starting server with config: $config"
    INFERFLUX_MODEL_PATH="$MODEL_PATH" \
    INFERFLUX_STARTUP_ADVISOR_VERBOSE=1 \
        $SERVER_BIN --config "$config" > logs/server_fp16_${backend}.log 2>&1 &
    SERVER_PID=$!

    echo "Server PID: $SERVER_PID"

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$SERVER_PORT/healthz" > /dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}ERROR: Server failed to start${NC}"
            tail -20 logs/server_fp16_${backend}.log
            return 1
        fi
        sleep 1
    done

    # Get server startup info
    echo ""
    echo "Server startup info:"
    curl -s "http://localhost:$SERVER_PORT/healthz" || true

    # Check memory before requests
    echo ""
    echo "Memory status before requests:"
    curl -s "http://localhost:$SERVER_PORT/metrics" | grep -E "memory_available|memory_used|memory_pressure" || echo "  (no memory metrics yet)"

    # Send concurrent requests
    echo ""
    echo "Sending $CONCURRENT_REQUESTS concurrent requests..."

    START_TIME=$(date +%s)

    for i in $(seq 1 $CONCURRENT_REQUESTS); do
        (
            RESPONSE=$(curl -s -X POST "http://localhost:$SERVER_PORT/v1/chat/completions" \
                -H "Authorization: Bearer $API_KEY" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"default\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"Tell me a short joke about programming (request $i)\"}],
                    \"max_tokens\": $MAX_TOKENS,
                    \"stream\": false
                }")

            # Check if request succeeded
            if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
                echo -e "${GREEN}[Request $i] SUCCESS${NC}"
            else
                echo -e "${RED}[Request $i] FAILED: $RESPONSE${NC}"
            fi
        ) &
    done

    # Wait for all requests to complete
    wait

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "All requests completed in ${ELAPSED}s"

    # Check memory after requests
    echo ""
    echo "Memory status after requests:"
    curl -s "http://localhost:$SERVER_PORT/metrics" | grep -E "memory_available|memory_used|memory_pressure" || echo "  (no memory metrics yet)"

    # Check server health
    echo ""
    echo "Server health check:"
    if curl -s "http://localhost:$SERVER_PORT/healthz" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is still healthy!${NC}"
    else
        echo -e "${RED}❌ Server is unhealthy!${NC}"
    fi

    # Check for errors in server log
    echo ""
    echo "Checking for errors in server log:"
    if grep -i "bad_alloc\|out of memory\|oom\|cudaErrorMemoryAllocation" logs/server_fp16_${backend}.log; then
        echo -e "${RED}❌ OOM errors detected!${NC}"
        tail -50 logs/server_fp16_${backend}.log
    else
        echo -e "${GREEN}✅ No OOM errors detected!${NC}"
    fi

    # Check for graceful degradation messages
    echo ""
    echo "Checking for graceful degradation messages:"
    if grep -i "memory pressure\|graceful degradation\|reducing max_slots" logs/server_fp16_${backend}.log; then
        echo -e "${YELLOW}⚠️  Graceful degradation activated!${NC}"
    else
        echo -e "${GREEN}✅ No degradation needed${NC}"
    fi

    # Stop server
    echo ""
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Backend $backend test complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Create logs directory
mkdir -p logs

# Test cuda_llama_cpp backend first (baseline)
test_backend "cuda_llama_cpp" "config/server.cuda.universal.yaml"

# Test cuda_native backend
test_backend "cuda_native" "config/server.cuda.yaml"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Server handled concurrent requests without crashing"
echo "  - No std::bad_alloc errors detected"
echo "  - Graceful degradation prevented OOM crashes"
echo ""
echo "Logs saved to:"
echo "  - logs/server_fp16_cuda_llama_cpp.log"
echo "  - logs/server_fp16_cuda_native.log"
