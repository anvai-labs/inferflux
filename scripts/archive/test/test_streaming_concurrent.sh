#!/bin/bash
# Quick test: Streaming vs Non-Streaming Concurrent Throughput
# Verifies that streaming mode fixes the concurrent throughput issue

set -euo pipefail

API_KEY="dev-key-123"
PORT="8080"
NUM_REQUESTS=8
MAX_TOKENS=50
TMP_PREFIX="/tmp/inferflux_stream_req_$$"

cleanup_tmp() {
    rm -f "${TMP_PREFIX}"_*.txt
}
trap cleanup_tmp EXIT

echo "========================================"
echo "Streaming vs Non-Streaming Test"
echo "========================================"
echo "Requests: $NUM_REQUESTS"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Test 1: Non-streaming (current benchmark)
echo "Test 1: Non-Streaming (stream: false)"
echo "----------------------------------------"

START=$(date +%s)

for i in $(seq 1 $NUM_REQUESTS); do
    (
        RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to 5 (request $i)\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }")

        if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
            echo "  [Request $i] SUCCESS"
        else
            echo "  [Request $i] FAILED"
        fi
    ) &
done

wait

END=$(date +%s)
NON_STREAM_TIME=$((END - START))
if [ $NON_STREAM_TIME -gt 0 ]; then
    NON_STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $NON_STREAM_TIME}")
else
    NON_STREAM_TOK_PER_SEC="N/A (too fast)"
fi

echo "  Time: ${NON_STREAM_TIME}s"
echo "  Throughput: ${NON_STREAM_TOK_PER_SEC} tok/s"
echo ""

# Wait a bit between tests
sleep 3

# Test 2: Streaming (proposed fix)
echo "Test 2: Streaming (stream: true)"
echo "----------------------------------------"

START=$(date +%s)

for i in $(seq 1 $NUM_REQUESTS); do
    (
        # Streaming request - capture all output
        curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to 5 (request $i)\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": true
            }" > "${TMP_PREFIX}_${i}.txt"

        # Count lines in streaming response (each chunk is a line)
        LINES=$(wc -l < "${TMP_PREFIX}_${i}.txt")
        if [ $LINES -gt 0 ]; then
            echo "  [Request $i] SUCCESS ($LINES chunks)"
        else
            echo "  [Request $i] FAILED"
        fi
    ) &
done

wait

END=$(date +%s)
STREAM_TIME=$((END - START))
if [ $STREAM_TIME -gt 0 ]; then
    STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $STREAM_TIME}")
else
    STREAM_TOK_PER_SEC="N/A (too fast)"
fi

echo "  Time: ${STREAM_TIME}s"
echo "  Throughput: ${STREAM_TOK_PER_SEC} tok/s"
echo ""

# Summary
echo "========================================"
echo "Results Summary"
echo "========================================"

# Calculate speedup only if both values are numeric
if [[ "$STREAM_TOK_PER_SEC" != "N/A"* ]] && [[ "$NON_STREAM_TOK_PER_SEC" != "N/A"* ]]; then
    if awk "BEGIN {exit ($STREAM_TOK_PER_SEC > $NON_STREAM_TOK_PER_SEC) ? 0 : 1}"; then
        SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $STREAM_TOK_PER_SEC / $NON_STREAM_TOK_PER_SEC}")
        echo "Streaming is ${SPEEDUP} FASTER [OK]"
    else
        SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $NON_STREAM_TOK_PER_SEC / $STREAM_TOK_PER_SEC}")
        echo "Streaming is ${SPEEDUP} SLOWER [WARN]"
    fi
else
    SPEEDUP="N/A"
    echo "Cannot calculate speedup (one or both tests too fast)"
fi

echo ""
echo "Non-Streaming: ${NON_STREAM_TOK_PER_SEC} tok/s (${NON_STREAM_TIME}s)"
echo "Streaming:     ${STREAM_TOK_PER_SEC} tok/s (${STREAM_TIME}s)"
if [[ "$SPEEDUP" != "N/A" ]]; then
    echo "Speedup:       ${SPEEDUP}"
fi
echo ""

# Expected: Streaming should be 10-20x faster
# Non-Streaming: ~2 tok/s (392s for 8 requests)
# Streaming: ~35-50 tok/s (~10-15s for 8 requests)
