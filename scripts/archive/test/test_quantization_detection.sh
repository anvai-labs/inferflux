#!/bin/bash
# Quick test to verify FP16 detection works

TEST_NAMES=(
    "qwen2.5-3b-instruct-f16.gguf"
    "qwen2.5-3b-instruct-fp16.gguf"
    "llama3-8b-q4_k_m.gguf"
    "model-f16.gguf"
)

echo "Testing quantization detection patterns:"
echo ""

for name in "${TEST_NAMES[@]}"; do
    echo "Testing: $name"
    # Convert to lowercase
    lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    echo "  Lowercase: $lower"

    # Check for patterns
    if [[ "$lower" == *"f16"* ]]; then
        echo "  -> Detected as FP16 ✅"
    elif [[ "$lower" == *"q4_k_m"* ]]; then
        echo "  -> Detected as Q4_K_M ✅"
    else
        echo "  -> Unknown ❌"
    fi
    echo ""
done
