#!/bin/bash
# Startup Advisor Tests - FINAL VERSION
# Safetensors → cuda_native
# GGUF → cuda_llama_cpp (llama.cpp CUDA)

set -e

INFERFLUXD="./build/inferfluxd"
LOG_DIR="logs/advisor_tests"
mkdir -p "$LOG_DIR"

cleanup() {
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

echo "=== Startup Advisor Tests (Final) ==="
echo "GPU: NVIDIA RTX 4000 Ada (SM 8.9)"
echo

# Helper to run a test case
test_case() {
    local name="$1"
    local model="$2"
    local format="$3"
    local backend="$4"
    local fa="$5"
    local overlap="$6"
    local batch="$7"
    local kv="$8"

    echo -e "\033[1;33mTest: $name\033[0m"
    echo "  Model: $model"
    echo "  Format: $format → Backend: $backend"
    echo "  Config: FA=$fa, Overlap=$overlap, Batch=$batch, KV=$kv"
    echo

    local config_file="$LOG_DIR/${name// /_}.yaml"
    local log_file="$LOG_DIR/${name// /_}.log"

    cat > "$config_file" << EOF
server:
  host: 0.0.0.0
  http_port: 8080
models:
  - id: test-model
    path: $model
    format: $format
    backend: $backend
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: $fa
      tile_size: 128
    phase_overlap:
      enabled: $overlap
  scheduler:
    max_batch_size: $batch
    max_batch_tokens: 4096
  paged_kv:
    cpu_pages: $kv
    eviction: lru
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
logging:
  level: info
  format: text
EOF

    # Start server
    "$INFERFLUXD" --config "$config_file" > "$log_file" 2>&1 &
    local pid=$!

    # Wait for startup
    local timeout=90
    local elapsed=0
    while ! grep -q "listening on" "$log_file" 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "  ✗ ERROR: Timeout"
            tail -30 "$log_file" | sed 's/^/    /'
            kill $pid 2>/dev/null || true
            echo
            return 1
        fi
    done

    # Check if model loaded successfully
    if grep -q "Failed to load model" "$log_file"; then
        echo "  ✗ Model load FAILED"
        grep "ERROR" "$log_file" | head -3 | sed 's/^/    /'
        kill $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
        sleep 1
        echo
        return 1
    fi

    # Get backend provider
    local provider=$(grep -oP 'backend_provider=\K[^,}]+' "$log_file" 2>/dev/null || echo "unknown")
    echo "  ✓ Model loaded (provider: $provider)"

    # Count recommendations
    local actual=$(grep -c "RECOMMEND" "$log_file" 2>/dev/null || echo "0")
    echo "  Recommendations: $actual"

    # Show advisor output
    if [ "$actual" -gt 0 ]; then
        echo "  Advisor output:"
        grep "RECOMMEND" "$log_file" | sed 's/^/    /'
    else
        echo "  ✓ Config is well-tuned!"
    fi

    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 1

    echo "  ✓ PASS"
    echo
}

echo "========================================="
echo "SAFETENSORS BF16 (cuda_native)"
echo "========================================="
echo

# Safetensors BF16 with native kernel executor (well-tuned)
test_case \
    "safetensors_bf16_well_tuned" \
    "models/qwen2.5-3b-instruct-safetensors" \
    "auto" \
    "cuda_native" \
    "true" "true" "16" "128"

# Safetensors BF16 with FA disabled (should recommend FA2)
test_case \
    "safetensors_bf16_no_fa" \
    "models/qwen2.5-3b-instruct-safetensors" \
    "auto" \
    "cuda_native" \
    "false" "true" "16" "128"

echo
echo "========================================="
echo "GGUF FP16 (cuda_llama_cpp)"
echo "========================================="
echo

# GGUF FP16 well-tuned
test_case \
    "gguf_fp16_well_tuned" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf" \
    "gguf" \
    "cuda_llama_cpp" \
    "true" "true" "16" "128"

echo
echo "========================================="
echo "GGUF Q4 QUANTIZED (cuda_llama_cpp)"
echo "========================================="
echo

# GGUF Q4 suboptimal (small batch, low KV)
test_case \
    "gguf_q4_suboptimal" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "gguf" \
    "cuda_llama_cpp" \
    "true" "true" "8" "32"

echo
echo "========================================="
echo "GGUF Q4 WELL-TUNED (large batch, high KV)"
echo "========================================="
echo

# GGUF Q4 well-tuned (large batch, high KV)
test_case \
    "gguf_q4_well_tuned" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "gguf" \
    "cuda_llama_cpp" \
    "true" "true" "32" "256"

echo
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Test logs: $LOG_DIR/"
echo
