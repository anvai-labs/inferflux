#!/bin/bash
# Verify all models in the models directory
# Tests loading each model and reports startup advisor recommendations

set -e

INFERFLUXD="./build/inferfluxd"
LOG_DIR="logs/model_verification"
mkdir -p "$LOG_DIR"

cleanup() {
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         MODEL VERIFICATION - ALL MODELS IN models/          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Array of models to test (format, backend, path, label)
declare -a MODELS=(
    "gguf:cuda_llama_cpp:models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf:tinyllama-1.1b-q4"
    "gguf:cuda_llama_cpp:models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf:qwen2.5-3b-q4"
    "gguf:cuda_llama_cpp:models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf:qwen2.5-3b-fp16"
    "gguf:cuda_llama_cpp:models/qwen2.5-coder-14b-instruct-q4_k_m.gguf:qwen2.5-coder-14b-q4"
    "auto:cuda_native:models/qwen2.5-3b-instruct-safetensors:qwen2.5-3b-safetensors-bf16"
)

# Check for empty files
echo "Checking for empty/placeholder files..."
echo
find models/ -name "*.gguf" -type f -size 0 | while read f; do
    echo "  ⚠️  PLACEHOLDER FILE (0 bytes): $f"
done
echo

test_model() {
    local format=$1
    local backend=$2
    local path=$3
    local label=$4

    local config_file="$LOG_DIR/${label}.yaml"
    local log_file="$LOG_DIR/${label}.log"

    echo -ne "\033[1;33mTesting: $label\033[0m "
    echo -ne "($format, $backend) "

    # Create config
    cat > "$config_file" << EOF
server:
  http_port: 8080
models:
  - id: test-model
    path: $path
    format: $format
    backend: $backend
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    flash_attention:
      enabled: true
  scheduler:
    max_batch_size: 16
  paged_kv:
    cpu_pages: 256
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
logging:
  level: warn
EOF

    # Start server
    "$INFERFLUXD" --config "$config_file" > "$log_file" 2>&1 &
    local pid=$!

    # Wait for result
    local timeout=90
    local elapsed=0
    local status="unknown"

    while [ $elapsed -lt $timeout ]; do
        if grep -q "listening on" "$log_file" 2>/dev/null; then
            status="✅ LOADED"
            break
        fi
        if grep -q "Failed to load model" "$log_file" 2>/dev/null; then
            status="❌ LOAD FAILED"
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ "$status" = "unknown" ]; then
        status="⏱️  TIMEOUT"
    fi

    # Kill server
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 1

    echo -e "$status"

    # Count recommendations
    local recs=$(grep -c "RECOMMEND" "$log_file" 2>/dev/null || echo "0")

    if [ "$status" = "✅ LOADED" ]; then
        echo "  Recommendations: $recs"
        if [ "$recs" -gt 0 ]; then
            grep "RECOMMEND" "$log_file" | sed 's/^/    /' | head -5
            if [ "$recs" -gt 5 ]; then
                echo "    ... and $((recs - 5)) more"
            fi
        fi

        # Get backend provider
        local provider=$(grep -oP 'backend_provider=\K[^,}]+' "$log_file" 2>/dev/null || echo "unknown")
        echo "  Provider: $provider"

        # Get format detected
        local fmt=$(grep -oP 'format=\K[^,}]+' "$log_file" 2>/dev/null || echo "unknown")
        echo "  Format: $fmt"
    else
        echo "  Error details:"
        grep "ERROR" "$log_file" | head -3 | sed 's/^/    /'
    fi
    echo
}

# Test all models
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r format backend path label <<< "$model_spec"
    test_model "$format" "$backend" "$path" "$label"
done

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      SUMMARY                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "Detailed logs: $LOG_DIR/"
echo
echo "Files checked:"
echo "  ✅ models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (638 MB)"
echo "  ✅ models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf (2.0 GB)"
echo "  ✅ models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf (5.8 GB)"
echo "  ✅ models/qwen2.5-coder-14b-instruct-q4_k_m.gguf (8.4 GB)"
echo "  ✅ models/qwen2.5-3b-instruct-safetensors/ (5.8 GB BF16)"
echo
echo "Placeholder files (0 bytes, not tested):"
echo "  ⚠️  models/qwen2.5-32b-instruct-q4_k_m.gguf"
echo "  ⚠️  models/qwen3-32b-instruct-q4_k_m.gguf"
echo "  ⚠️  models/qwen2.5-coder-14b-instruct-safetensors/ (no model files)"
echo
