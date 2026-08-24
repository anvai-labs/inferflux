#!/bin/bash
# Test FlashAttention during actual inference

set -e

echo "========================================="
echo "FlashAttention Live Test"
echo "========================================="
echo ""

# Check if server is running
if ! curl -s http://localhost:8080/healthz > /dev/null 2>&1; then
    echo "Starting InferFlux server with CUDA..."
    echo ""
    echo "In a separate terminal, run:"
    echo "  INFERFLUX_LOG_LEVEL=debug ./build/inferfluxd --config config/server.cuda.yaml"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "✓ Server is running"
echo ""

# Make a test request
echo "Making test request..."
echo ""

RESPONSE=$(curl -s -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "prompt": "The quick brown fox jumps over the lazy dog. Explain what this sentence means.",
    "max_tokens": 50,
    "model": "tinyllama"
  }')

echo "Response received:"
echo "$RESPONSE" | head -5
echo ""

# Check metrics
echo "Checking FlashAttention metrics..."
echo ""

METRICS=$(curl -s http://localhost:8080/metrics)

if echo "$METRICS" | grep -q "flash_attention"; then
  echo "✓ FlashAttention metrics found:"
  echo "$METRICS" | grep -i "flash_attention" | head -5
else
  echo "⚠ No FlashAttention metrics found yet."
  echo "  Metrics will appear after inference requests are processed."
  echo ""
  echo "Current metrics:"
  echo "$METRICS" | grep -i "cuda\|backend\|tokens" | head -10
fi

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="
echo ""
echo "To monitor FlashAttention usage in real-time:"
echo "  watch -n 1 'curl -s http://localhost:8080/metrics | grep flash_attention'"
