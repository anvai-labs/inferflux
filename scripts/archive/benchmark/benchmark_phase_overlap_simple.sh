#!/bin/bash
# Simple Phase Overlap Benchmark for Native Kernels
# Tests concurrent prefill/decode with safetensors + native kernels

set -e

BASE_URL="http://localhost:8080"
API_KEY="dev-key-123"
MODEL_ID="qwen2.5-3b"

echo "=========================================="
echo "Phase Overlap Benchmark - Native Kernels"
echo "=========================================="
echo "Model: $MODEL_ID (safetensors + native CUDA)"
echo ""

# Get initial metrics
echo "Getting initial metrics..."
INITIAL_METRICS=$(curl -s $BASE_URL/metrics)
INITIAL_OVERLAP=$(echo "$INITIAL_METRICS" | grep "inferflux_cuda_lane_overlap_events_total" | grep -o '[0-9]*$' | head -1)
INITIAL_OVERLAP=${INITIAL_OVERLAP:-0}
echo "Initial overlap events: $INITIAL_OVERLAP"
echo ""

# Create a simple mixed workload test
# We'll send multiple requests concurrently: some with long prompts (prefill), some with short prompts (decode)

echo "Running mixed workload test..."
echo "  - 12 requests with long prompts (prefill)"
echo "  - 12 requests with short prompts (decode)"
echo "  - Total: 24 concurrent requests"
echo ""

START_TIME=$(date +%s.%N)

# Function to send a completion request
send_request() {
    local prompt="$1"
    local max_tokens="$2"
    local request_id="$3"

    RESPONSE=$(curl -s -X POST $BASE_URL/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
            \"model\": \"$MODEL_ID\",
            \"prompt\": \"$prompt\",
            \"max_tokens\": $max_tokens
        }" --max-time 60)

    if echo "$RESPONSE" | grep -q '"choices"'; then
        TOKENS=$(echo "$RESPONSE" | jq -r '.usage.total_tokens // 0')
        echo "  ✓ Request $request_id: $TOKENS tokens"
    else
        echo "  ✗ Request $request_id failed"
    fi
}

export -f send_request
export BASE_URL API_KEY MODEL_ID

# Long prompt for prefill (repeated to create ~300 tokens)
LONG_PROMPT="The quick brown fox jumps over the lazy dog. "
LONG_PROMPT="$LONG_PROMPT""In a computer system, an algorithm is basically an instance of logic written in software by software developers, to be effective for the intended target in computer machines. "
LONG_PROMPT="$LONG_PROMPT""The quick brown fox jumps over the lazy dog. "
LONG_PROMPT="$LONG_PROMPT""In a computer system, an algorithm is basically an instance of logic written in software by software developers. "
LONG_PROMPT="$LONG_PROMPT""The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. "
LONG_PROMPT="$LONG_PROMPT""In a computer system, an algorithm is basically an instance of logic written in software by software developers, to be effective for the intended target. "
LONG_PROMPT="$LONG_PROMPT""The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. "

# Short prompt for decode (just a few tokens)
SHORT_PROMPT="Hello world"

# Send 24 concurrent requests
for i in {1..24}; do
    if [ $((i % 2)) -eq 0 ]; then
        # Even requests: long prompt (prefill)
        send_request "$LONG_PROMPT" 10 $i &
    else
        # Odd requests: short prompt (decode-like)
        send_request "$SHORT_PROMPT" 10 $i &
    fi
done

# Wait for all requests to complete
wait

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "All requests completed in ${ELAPSED}s"
echo ""

# Get final metrics
echo "Getting final metrics..."
FINAL_METRICS=$(curl -s $BASE_URL/metrics)

# Calculate overlap metrics
FINAL_OVERLAP=$(echo "$FINAL_METRICS" | grep "inferflux_cuda_lane_overlap_events_total" | grep -o '[0-9]*$' | head -1)
FINAL_OVERLAP=${FINAL_OVERLAP:-0}

OVERLAP_EVENTS=$((FINAL_OVERLAP - INITIAL_OVERLAP))

OVERLAP_DURATION=$(echo "$FINAL_METRICS" | grep "inferflux_cuda_lane_overlap_duration_ms_total" | grep -o '[0-9.]*$' | head -1)
OVERLAP_DURATION=${OVERLAP_DURATION:-0}

PREFILL_SUBMISSIONS=$(echo "$FINAL_METRICS" | grep 'inferflux_cuda_lane_submissions_total{lane="prefill"' | grep -o '[0-9]*$' | head -1)
PREFILL_SUBMISSIONS=${PREFILL_SUBMISSIONS:-0}

DECODE_SUBMISSIONS=$(echo "$FINAL_METRICS" | grep 'inferflux_cuda_lane_submissions_total{lane="decode"' | grep -o '[0-9]*$' | head -1)
DECODE_SUBMISSIONS=${DECODE_SUBMISSIONS:-0}

PREFILL_COMPLETIONS=$(echo "$FINAL_METRICS" | grep 'inferflux_cuda_lane_completions_total{lane="prefill"' | grep -o '[0-9]*$' | head -1)
PREFILL_COMPLETIONS=${PREFILL_COMPLETIONS:-0}

DECODE_COMPLETIONS=$(echo "$FINAL_METRICS" | grep 'inferflux_cuda_lane_completions_total{lane="decode"' | grep -o '[0-9]*$' | head -1)
DECODE_COMPLETIONS=${DECODE_COMPLETIONS:-0}

# Print results
echo "=========================================="
echo "Benchmark Results"
echo "=========================================="
echo "Total time: ${ELAPSED}s"
echo "Total requests: 24"
echo ""
echo "--- Lane Activity ---"
echo "Prefill submissions: $PREFILL_SUBMISSIONS"
echo "Decode submissions: $DECODE_SUBMISSIONS"
echo "Prefill completions: $PREFILL_COMPLETIONS"
echo "Decode completions: $DECODE_COMPLETIONS"
echo ""
echo "--- Phase Overlap ---"
echo "Overlap events: $OVERLAP_EVENTS"

if [ "$OVERLAP_EVENTS" -gt 0 ]; then
    echo "Overlap duration: ${OVERLAP_DURATION}ms"
    if [ "$OVERLAP_EVENTS" -gt 0 ]; then
        AVG_OVERLAP=$(echo "scale=2; $OVERLAP_DURATION / $OVERLAP_EVENTS" | bc)
        echo "Avg overlap per event: ${AVG_OVERLAP}ms"
    fi
    echo ""
    echo "✓ Phase overlap is ACTIVE and working!"
    echo ""
    OVERLAP_PERCENT=$(echo "scale=1; $OVERLAP_EVENTS * 100 / 24" | bc)
    echo "Overlap triggered on ${OVERLAP_PERCENT}% of requests"
else
    echo ""
    echo "⚠ Phase overlap not triggered"
    echo "This might mean:"
    echo "  - Requests were processed sequentially (no mixed batches)"
    echo "  - Prefill tokens were below threshold (256)"
    echo "  - Workload was all prefill or all decode"
fi
echo "=========================================="
echo ""

# Show sample metrics
echo "Sample metrics (for verification):"
echo "$FINAL_METRICS" | grep "inferflux_cuda_lane" | head -10
echo ""
