#!/bin/bash
# Quick streaming test - fewer requests for faster results

set -euo pipefail

API_KEY="dev-key-123"
PORT="8080"
NUM_REQUESTS=4
MAX_TOKENS=20
TMP_PREFIX="/tmp/inferflux_stream_quick_$$"

cleanup_tmp() {
    rm -f "${TMP_PREFIX}"_*.txt
}
trap cleanup_tmp EXIT

echo "========================================"
echo "Quick Streaming Test"
echo "========================================"
echo "Requests: $NUM_REQUESTS"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Test 1: Non-streaming
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
                \"messages\": [{\"role\": \"user\", \"content\": \"Say hello $i\"}],
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
NON_STREAM_SEC=$(awk "BEGIN {printf \"%.2f\", $NON_STREAM_TIME}")

if [ $NON_STREAM_TIME -gt 0 ]; then
    NON_STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $NON_STREAM_TIME}")
else
    NON_STREAM_TOK_PER_SEC="N/A"
fi

echo "  Time: ${NON_STREAM_SEC}s"
echo "  Throughput: ${NON_STREAM_TOK_PER_SEC} tok/s"
echo ""

# Wait a bit
sleep 2

# Test 2: Streaming
echo "Test 2: Streaming (stream: true)"
echo "----------------------------------------"

START=$(date +%s)

for i in $(seq 1 $NUM_REQUESTS); do
    (
        curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Say hello $i\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": true
            }" > "${TMP_PREFIX}_${i}.txt"

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
STREAM_SEC=$(awk "BEGIN {printf \"%.2f\", $STREAM_TIME}")

if [ $STREAM_TIME -gt 0 ]; then
    STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $STREAM_TIME}")
else
    STREAM_TOK_PER_SEC="N/A"
fi

echo "  Time: ${STREAM_SEC}s"
echo "  Throughput: ${STREAM_TOK_PER_SEC} tok/s"
echo ""

# Summary
echo "========================================"
echo "Results Summary"
echo "========================================"

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
    echo "Cannot calculate speedup"
fi

echo ""
echo "Non-Streaming: ${NON_STREAM_TOK_PER_SEC} tok/s (${NON_STREAM_SEC}s)"
echo "Streaming:     ${STREAM_TOK_PER_SEC} tok/s (${STREAM_SEC}s)"
if [[ "$SPEEDUP" != "N/A" ]]; then
    echo "Speedup:       ${SPEEDUP}"
fi

# Expected: Streaming should be faster (or similar for small batches)
echo ""
echo "[OK] Test complete"
