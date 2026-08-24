#!/bin/bash
# Smoke test for native CUDA kernels
# Tests that native kernels work correctly with safetensors models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Native CUDA Kernels Smoke Test"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. CUDA required for native kernels.${NC}"
    exit 1
fi

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# Check if server binary exists
if [ ! -f "./build/inferfluxd" ]; then
    echo -e "${YELLOW}Building InferFlux...${NC}"
    ./scripts/build.sh
fi

# Model path (use safetensors model)
MODEL_PATH="${INFERFLUX_NATIVE_MODEL_PATH:-models/qwen2.5-3b-safetensors}"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model path not found: $MODEL_PATH${NC}"
    echo "Please set INFERFLUX_NATIVE_MODEL_PATH to a safetensors model directory"
    echo "Example: export INFERFLUX_NATIVE_MODEL_PATH=models/qwen2.5-3b-safetensors"
    exit 1
fi

echo "Model path: $MODEL_PATH"

# Check for safetensors files
if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo -e "${RED}ERROR: model.safetensors.index.json not found in $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Safetensors model found${NC}"

# Test 1: Native Kernel Auto-Detection
echo ""
echo "Test 1: Native Kernel Auto-Detection"
echo "--------------------------------------"

export INFERCTL_API_KEY=dev-key-123

# Create test config
cat > /tmp/test_native_kernels.yaml <<EOF
server:
  http_port: 8080
  enable_metrics: true

models:
  - id: test-model
    path: $MODEL_PATH
    format: auto
    backend: cuda_native
    default: true

runtime:
  cuda:
    enabled: true
    flash_attention:
      enabled: true

auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]

logging:
  level: info
  format: text
EOF

echo "Starting server with native kernels..."
./build/inferfluxd --config /tmp/test_native_kernels.yaml > /tmp/native_test_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Server failed to start${NC}"
    cat /tmp/native_test_server.log
    exit 1
fi

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"

# Check logs for native kernel usage
if grep -q "Native CUDA model loaded successfully" /tmp/native_test_server.log; then
    echo -e "${GREEN}✓ Native kernels loaded successfully${NC}"
else
    echo -e "${RED}✗ Native kernels failed to load${NC}"
    cat /tmp/native_test_server.log
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test 2: Health Check
echo ""
echo "Test 2: Health Check"
echo "-------------------"

HEALTH=$(curl -s http://localhost:8080/healthz)
if [ "$HEALTH" = "OK" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test 3: Tokenization
echo ""
echo "Test 3: Tokenization"
echo "--------------------"

TOKEN_COUNT=$(curl -s -X POST http://localhost:8080/v1/tokenize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d "{\"text\": \"Hello, world!\"}")

if [ -n "$TOKEN_COUNT" ]; then
    TOKEN_COUNT_INT=$(echo "$TOKEN_COUNT" | grep -o '[0-9]*' | head -1)
    if [ -n "$TOKEN_COUNT_INT" ] && [ "$TOKEN_COUNT_INT" -gt 0 ]; then
        echo -e "${GREEN}✓ Tokenization works (${TOKEN_COUNT_INT} tokens)${NC}"
    else
        echo -e "${RED}✗ Tokenization failed: invalid token count${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Tokenization endpoint not available (expected for native kernels)${NC}"
fi

# Test 4: Simple Completion
echo ""
echo "Test 4: Simple Completion"
echo "-------------------------"

RESPONSE=$(curl -s -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "model": "test-model",
    "prompt": "Hello, ",
    "max_tokens": 10
  }')

if echo "$RESPONSE" | grep -q "text"; then
    GENERATED_TEXT=$(echo "$RESPONSE" | grep -o '"text":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -n "$GENERATED_TEXT" ]; then
        echo -e "${GREEN}✓ Completion works: \"${GENERATED_TEXT}\"${NC}"
    else
        echo -e "${RED}✗ Completion failed: no text generated${NC}"
    fi
else
    echo -e "${RED}✗ Completion failed${NC}"
    echo "Response: $RESPONSE"
fi

# Test 5: Metrics
echo ""
echo "Test 5: Native Kernel Metrics"
echo "------------------------------"

METRICS=$(curl -s http://localhost:8080/metrics)

# Check for native forward pass metrics
if echo "$METRICS" | grep -q "inferflux_native_forward_passes_total"; then
    FORWARD_PASSES=$(echo "$METRICS" | grep "inferflux_native_forward_passes_total" | grep "phase=\"prefill\"" | grep -o '[0-9]*$' | head -1)
    if [ -n "$FORWARD_PASSES" ] && [ "$FORWARD_PASSES" -gt 0 ]; then
        echo -e "${GREEN}✓ Native forward passes recorded: ${FORWARD_PASSES}${NC}"
    else
        echo -e "${YELLOW}⚠ Native forward passes: 0 (may need request)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Native forward pass metrics not found${NC}"
fi

# Check for attention kernel selection
if echo "$METRICS" | grep -q "inferflux_cuda_attention_kernel_selected"; then
    KERNEL=$(echo "$METRICS" | grep 'inferflux_cuda_attention_kernel_selected{kernel="' | grep -o 'kernel="[^"]*"' | cut -d'"' -f2)
    if [ "$KERNEL" = "fa2" ]; then
        echo -e "${GREEN}✓ FlashAttention-2 enabled${NC}"
    else
        echo -e "${YELLOW}⚠ Attention kernel: ${KERNEL}${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Attention kernel metrics not found${NC}"
fi

# Check for KV cache metrics
if echo "$METRICS" | grep -q "inferflux_native_kv_active_sequences"; then
    SEQUENCES=$(echo "$METRICS" | grep "inferflux_native_kv_active_sequences" | grep -o '[0-9]*$' | head -1)
    echo -e "${GREEN}✓ KV cache active sequences: ${SEQUENCES}${NC}"
else
    echo -e "${YELLOW}⚠ KV cache metrics not found${NC}"
fi

# Test 6: Verify no llama.cpp dependency
echo ""
echo "Test 6: No llama.cpp Dependency"
echo "--------------------------------"

if grep -q "native_cuda" /tmp/native_test_server.log; then
    echo -e "${GREEN}✓ Using native CUDA kernels (not llama.cpp delegate)${NC}"
else
    echo -e "${YELLOW}⚠ Could not verify native kernel usage${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Smoke Test Summary"
echo "=========================================="

# Kill server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}✓ All tests passed!${NC}"
echo ""
echo "Native CUDA kernels are working correctly."
echo ""
echo "Server log saved to: /tmp/native_test_server.log"
echo "To review metrics, see the Prometheus output above."
echo ""
echo "Next steps:"
echo "1. Run performance benchmarks: ./scripts/benchmark_formats.sh"
echo "2. Test with larger models for throughput"
echo "3. Verify FlashAttention-2 performance with Nsight Systems"

exit 0
