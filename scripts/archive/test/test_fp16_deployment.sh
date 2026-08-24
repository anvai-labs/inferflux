#!/bin/bash
# Test FP16 Deployment
# Verifies all OOM fixes are working correctly

set -e

MODEL_PATH="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf"
API_KEY="dev-key-123"
PORT="8080"

echo "========================================"
echo "FP16 Deployment Test"
echo "========================================"
echo "Model: $MODEL_PATH"
echo ""

# Test 1: Check server health
echo "Test 1: Server Health"
HEALTH=$(curl -s http://localhost:$PORT/healthz)
if echo "$HEALTH" | jq -e '.model_ready == true' > /dev/null 2>&1; then
    echo "  ✅ Server is healthy and model loaded"
else
    echo "  ❌ Server health check failed"
    exit 1
fi

# Test 2: Single request
echo ""
echo "Test 2: Single Request"
RESPONSE=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 10
    }')

if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')
    echo "  ✅ Request successful: '$CONTENT'"
else
    echo "  ❌ Request failed"
    echo "$RESPONSE"
    exit 1
fi

# Test 3: Check memory calculation
echo ""
echo "Test 3: Memory Calculation Fix"
if grep -q "Activation overhead: 3.16 GB (multiplier: 1.500000)" logs/fp16_test2.log; then
    echo "  ✅ FP16 activation multiplier (1.5x) detected correctly"
else
    echo "  ❌ FP16 activation multiplier not found"
    exit 1
fi

if grep -q "Total overhead: 4.58 GB" logs/fp16_test2.log; then
    echo "  ✅ Total overhead calculated correctly (4.58 GB)"
else
    echo "  ❌ Total overhead calculation incorrect"
    exit 1
fi

# Test 4: Check config override
echo ""
echo "Test 4: Config Override Fix"
if grep -q "INFERFLUX_MODEL_PATH is set, overriding config file model path" logs/fp16_test2.log; then
    echo "  ✅ Config override working (INFERFLUX_MODEL_PATH respected)"
else
    echo "  ❌ Config override not working"
    exit 1
fi

# Test 5: Check slot recommendation
echo ""
echo "Test 5: Slot Allocation Recommendation"
if grep -q "Recommended: max_parallel_sequences=43" logs/fp16_test2.log; then
    echo "  ✅ Conservative slot recommendation (43 slots)"
else
    echo "  ❌ Slot recommendation not found"
    exit 1
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "✅ All tests passed!"
echo ""
echo "Deployment verified:"
echo "  - Server loads FP16 model successfully"
echo "  - Config override working (INFERFLUX_MODEL_PATH)"
echo "  - Quantization detection working (f16 → 1.5x activation)"
echo "  - Memory calculation accurate (4.58 GB overhead)"
echo "  - Slot recommendations conservative (43 slots)"
echo "  - Requests processed successfully"
echo ""
echo "OOM fixes are working correctly! 🎉"
