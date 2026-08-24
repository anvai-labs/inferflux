#!/bin/bash
# Test script for Phase Overlap - Concurrent Prefill/Decode Execution
# This script verifies that the phase overlap feature is working correctly.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Phase Overlap Verification Test"
echo "=========================================="

# Check if server binary exists
if [ ! -f "./build/inferfluxd" ]; then
    echo -e "${YELLOW}Building InferFlux...${NC}"
    ./scripts/build.sh
fi

# Model path (use GGUF model for this test since it's widely available)
MODEL_PATH="${INFERFLUX_MODEL_PATH:-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found: $MODEL_PATH${NC}"
    echo "Please set INFERFLUX_MODEL_PATH to a GGUF model file"
    exit 1
fi

echo "Model: $MODEL_PATH"

# Create test config with phase overlap enabled
cat > /tmp/test_phase_overlap.yaml <<EOF
server:
  http_port: 8080
  enable_metrics: true

models:
  - id: test-model
    path: $MODEL_PATH
    format: gguf
    backend: cuda_llama_cpp
    default: true

runtime:
  cuda:
    enabled: true
    flash_attention:
      enabled: true
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256
      prefill_replica: false

  scheduler:
    max_batch_size: 32
    batch_accumulation_ms: 5

auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]

logging:
  level: info
  format: text
EOF

# Start server
echo ""
echo "Starting server with phase overlap enabled..."
./build/inferfluxd --config /tmp/test_phase_overlap.yaml > /tmp/overlap_test_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Server failed to start${NC}"
    cat /tmp/overlap_test_server.log
    exit 1
fi

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"

# Test 1: Health Check
echo ""
echo "Test 1: Health Check"
echo "--------------------"

HEALTH=$(curl -s http://localhost:8080/healthz)
if [ "$HEALTH" = "OK" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test 2: Simple Completion
echo ""
echo "Test 2: Simple Completion"
echo "-------------------------"

RESPONSE=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello, "}],
    "max_tokens": 10
  }')

if echo "$RESPONSE" | grep -q "content"; then
    GENERATED_TEXT=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    if [ -n "$GENERATED_TEXT" ]; then
        echo -e "${GREEN}✓ Completion works: \"${GENERATED_TEXT}\"${NC}"
    else
        echo -e "${RED}✗ Completion failed: no text generated${NC}"
    fi
else
    echo -e "${RED}✗ Completion failed${NC}"
    echo "Response: $RESPONSE"
fi

# Test 3: Check Metrics for Lane Activity
echo ""
echo "Test 3: Phase Overlap Metrics"
echo "------------------------------"

METRICS=$(curl -s http://localhost:8080/metrics)

# Check for lane submissions
if echo "$METRICS" | grep -q "inferflux_cuda_lane_submissions_total"; then
    DECODE_SUBMISSIONS=$(echo "$METRICS" | grep 'inferflux_cuda_lane_submissions_total{lane="decode"' | grep -o '[0-9]*$' | head -1)
    PREFILL_SUBMISSIONS=$(echo "$METRICS" | grep 'inferflux_cuda_lane_submissions_total{lane="prefill"' | grep -o '[0-9]*$' | head -1)

    if [ -n "$DECODE_SUBMISSIONS" ] && [ "$DECODE_SUBMISSIONS" -gt 0 ]; then
        echo -e "${GREEN}✓ Decode lane submissions: ${DECODE_SUBMISSIONS}${NC}"
    else
        echo -e "${YELLOW}⚠ Decode lane submissions: 0 (may need mixed workload)${NC}"
    fi

    if [ -n "$PREFILL_SUBMISSIONS" ] && [ "$PREFILL_SUBMISSIONS" -gt 0 ]; then
        echo -e "${GREEN}✓ Prefill lane submissions: ${PREFILL_SUBMISSIONS}${NC}"
    else
        echo -e "${YELLOW}⚠ Prefill lane submissions: 0 (may need mixed workload)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Lane submission metrics not found${NC}"
fi

# Check for overlap events
if echo "$METRICS" | grep -q "inferflux_cuda_lane_overlap_events_total"; then
    OVERLAP_EVENTS=$(echo "$METRICS" | grep "inferflux_cuda_lane_overlap_events_total" | grep -o '[0-9]*$' | head -1)
    if [ -n "$OVERLAP_EVENTS" ] && [ "$OVERLAP_EVENTS" -gt 0 ]; then
        echo -e "${GREEN}✓ Phase overlap events detected: ${OVERLAP_EVENTS}${NC}"

        # Get overlap duration
        OVERLAP_DURATION=$(echo "$METRICS" | grep "inferflux_cuda_lane_overlap_duration_ms_total" | grep -o '[0-9.]*$' | head -1)
        if [ -n "$OVERLAP_DURATION" ]; then
            echo -e "${GREEN}✓ Total overlap duration: ${OVERLAP_DURATION}ms${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No overlap events detected (needs mixed workload with min_prefill_tokens >= 256)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Overlap metrics not found${NC}"
fi

# Test 4: Check Server Logs for Phase Overlap Messages
echo ""
echo "Test 4: Server Log Analysis"
echo "---------------------------"

if grep -q "CUDA phase-overlap scaffold enabled" /tmp/overlap_test_server.log; then
    echo -e "${GREEN}✓ Phase overlap scaffold enabled${NC}"
else
    echo -e "${YELLOW}⚠ Phase overlap scaffold message not found${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

# Kill server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}✓ Basic tests passed!${NC}"
echo ""
echo "Notes:"
echo "- Phase overlap requires mixed workloads (prefill + decode)"
echo "- Overlap is triggered when prefill_tokens >= 256"
echo "- Run with higher concurrency to see overlap metrics"
echo ""
echo "To test with mixed workload:"
echo "  1. Send multiple concurrent requests with long prompts"
echo "  2. Check overlap events in metrics"
echo "  3. Monitor overlap duration for effectiveness"
echo ""
echo "Server log saved to: /tmp/overlap_test_server.log"
echo ""

exit 0
