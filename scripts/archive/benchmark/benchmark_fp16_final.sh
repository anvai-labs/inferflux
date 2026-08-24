#!/bin/bash
# FP16 Benchmark with All OOM Fixes
# Measures sequential and concurrent throughput

set -e

MODEL_PATH="${INFERFLUX_MODEL_PATH:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf}"
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
PORT="${INFERFLUX_PORT_OVERRIDE:-8080}"
SERVER_BIN="./build/inferfluxd"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "FP16 Benchmark (All Fixes Applied)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
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

# Benchmark configuration
PROMPT="Write a detailed explanation of how neural networks work, including backpropagation, gradient descent, and activation functions."
MAX_TOKENS=128

# Test sequential throughput
test_sequential() {
    local num_requests=$1
    local label=$2

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Sequential: $num_requests requests ($label)${NC}"
    echo -e "${YELLOW}========================================${NC}"

    START_TIME=$(date +%s)

    for i in $(seq 1 $num_requests); do
        RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }")

        # Check if request succeeded
        if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
            TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // 0')
            PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // 0')
            TOTAL_TOKENS=$((TOKENS + PROMPT_TOKENS))
            echo -e "${GREEN}[Request $i] SUCCESS - ${TOTAL_TOKENS} tokens${NC}"
        else
            echo -e "${RED}[Request $i] FAILED${NC}"
            echo "$RESPONSE" | head -5
        fi
    done

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Estimate throughput (assuming ~128 tokens per request)
    TOTAL_TOKENS_EST=$((num_requests * MAX_TOKENS))
    THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $TOTAL_TOKENS_EST / $ELAPSED}")

    echo ""
    echo "Sequential results ($label):"
    echo "  Requests: $num_requests"
    echo "  Time: ${ELAPSED}s"
    echo "  Estimated throughput: ${THROUGHPUT} tokens/second"
    echo ""
}

# Test concurrent throughput
test_concurrent() {
    local num_requests=$1
    local label=$2

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Concurrent: $num_requests requests ($label)${NC}"
    echo -e "${YELLOW}========================================${NC}"

    START_TIME=$(date +%s)

    for i in $(seq 1 $num_requests); do
        (
            RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
                -H "Authorization: Bearer $API_KEY" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"default\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT (request $i)\"}],
                    \"max_tokens\": $MAX_TOKENS,
                    \"stream\": false
                }")

            if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
                TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // 0')
                PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // 0')
                TOTAL_TOKENS=$((TOKENS + PROMPT_TOKENS))
                echo -e "${GREEN}[Request $i] SUCCESS - ${TOTAL_TOKENS} tokens${NC}"
            else
                echo -e "${RED}[Request $i] FAILED${NC}"
            fi
        ) &
    done

    wait

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Estimate throughput
    TOTAL_TOKENS_EST=$((num_requests * MAX_TOKENS))
    THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $TOTAL_TOKENS_EST / $ELAPSED}")

    echo ""
    echo "Concurrent results ($label):"
    echo "  Requests: $num_requests"
    echo "  Time: ${ELAPSED}s"
    echo "  Estimated throughput: ${THROUGHPUT} tokens/second"
    echo ""
}

# Start server
echo "Starting server with FP16 model..."
INFERFLUX_MODEL_PATH="$MODEL_PATH" \
    INFERFLUX_STARTUP_ADVISOR_VERBOSE=1 \
    $SERVER_BIN --config config/server.cuda.yaml > logs/benchmark_fp16.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Server failed to start"
        tail -20 logs/benchmark_fp16.log
        exit 1
    fi
    sleep 1
done

# Check startup advisor output
echo ""
echo "Startup Advisor Recommendations:"
grep -A 20 "Memory calculation:" logs/benchmark_fp16.log | head -25
echo ""

# Run benchmarks
echo "========================================"
echo "Running Benchmarks"
echo "========================================"
echo ""

# Warmup
echo "Warmup request..."
curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}' > /dev/null
echo "Warmup complete"
echo ""

# Test sequential: 5 requests
test_sequential 5 "warm cache"

# Test concurrent: 4 requests
test_concurrent 4 "light load"

# Test concurrent: 8 requests
test_concurrent 8 "medium load"

# Get server metrics
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Server Metrics${NC}"
echo -e "${YELLOW}========================================${NC}"

curl -s "http://localhost:$PORT/metrics" | grep -E "inferflux_.*_total\|inferflux_.*_seconds\|memory" | head -30 || echo "No metrics available"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Full logs: logs/benchmark_fp16.log"
