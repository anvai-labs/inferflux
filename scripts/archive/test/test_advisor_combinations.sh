#!/bin/bash
# Startup Advisor Smoke Tests
# Tests various model format and precision combinations

set -e

INFERFLUXD="./build/inferfluxd"
INFERCTL="./build/inferctl"
LOG_DIR="logs/advisor_tests"
mkdir -p "$LOG_DIR"

echo "=== Startup Advisor Smoke Tests ==="
echo "Log directory: $LOG_DIR"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    echo "Cleaning up..."
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

# Base config template
create_config() {
    local model="$1"
    local format="$2"
    local backend="$3"
    local fa_enabled="$4"
    local phase_overlap="$5"
    local max_batch="$6"
    local kv_pages="$7"
    local config_file="$8"

    cat > "$config_file" << EOF
server:
  host: 0.0.0.0
  http_port: 8080
  max_concurrent: 1024
  enable_metrics: true
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
      enabled: $fa_enabled
      tile_size: 128
    phase_overlap:
      enabled: $phase_overlap
      min_prefill_tokens: 256
  backend_exposure:
    prefer_native: true
    allow_universal_fallback: true
  capability_routing:
    allow_default_fallback: true
    require_ready_backend: true
  scheduler:
    max_batch_size: $max_batch
    max_batch_tokens: 4096
  tensor_parallel: 1
  paged_kv:
    cpu_pages: $kv_pages
    eviction: lru
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
  rate_limit_per_minute: 120
logging:
  level: info
  format: text
  audit_log: /dev/null
EOF
}

test_case() {
    local name="$1"
    local model="$2"
    local format="$3"
    local backend="$4"
    local fa="$5"
    local overlap="$6"
    local batch="$7"
    local kv="$8"

    echo -e "${YELLOW}Testing: $name${NC}"
    echo "  Model: $model"
    echo "  Format: $format"
    echo "  Backend: $backend"
    echo "  FA: $fa, Overlap: $overlap, Batch: $batch, KV: $kv"
    echo

    local config_file="$LOG_DIR/${name// /_}.yaml"
    local log_file="$LOG_DIR/${name// /_}.log"

    create_config "$model" "$format" "$backend" "$fa" "$overlap" "$batch" "$kv" "$config_file"

    # Start server in background
    "$INFERFLUXD" --config "$config_file" > "$log_file" 2>&1 &
    local pid=$!

    # Wait for startup or timeout
    local timeout=60
    local elapsed=0
    while ! $INFERCTL --api-key dev-key-123 --port 8080 list >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "  ERROR: Timeout waiting for server"
            cat "$log_file" | tail -30
            kill $pid 2>/dev/null || true
            break 2
        fi
    done

    # Give advisor time to run
    sleep 1

    # Show advisor output
    echo "  Advisor output:"
    grep -A 20 "Startup Recommendations" "$log_file" | while read line; do
        echo "    $line"
    done || echo "    (No advisor output)"

    # Count recommendations
    local count=$(grep -c "RECOMMEND" "$log_file" 2>/dev/null || echo "0")
    echo "  Total recommendations: $count"

    # Kill server
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 2

    echo "  ✓ Test complete"
    echo
}

# ============================================================================
# Test 1: Safetensors BF16 with universal provider (FA enabled)
# Expected: native_kernel recommendation for safetensors on universal
# ============================================================================
test_case \
    "safetensors_bf16_universal" \
    "models/qwen2.5-3b-instruct-safetensors" \
    "auto" \
    "cuda" \
    "true" "true" "8" "32"

# ============================================================================
# Test 2: Safetensors BF16 with FA disabled
# Expected: FA2 recommendation (GPU SM >= 8.0)
# ============================================================================
test_case \
    "safetensors_bf16_no_fa" \
    "models/qwen2.5-3b-instruct-safetensors" \
    "auto" \
    "cuda" \
    "false" "true" "8" "32"

# ============================================================================
# Test 3: GGUF FP16 with native backend
# Expected: Minimal recommendations (well-tuned config)
# ============================================================================
test_case \
    "gguf_fp16_native" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf" \
    "gguf" \
    "cuda_native" \
    "true" "true" "16" "64"

# ============================================================================
# Test 4: GGUF Q4 quantized (smaller model, should suggest larger batch)
# Expected: batch_size recommendation
# ============================================================================
test_case \
    "gguf_q4_quantized" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "gguf" \
    "cuda" \
    "true" "true" "8" "32"

# ============================================================================
# Test 5: TinyLlama Q4 on CPU with GPU available
# Expected: GPU unused recommendation
# ============================================================================
test_case \
    "tinyllama_cpu_with_gpu" \
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    "gguf" \
    "cpu" \
    "true" "true" "8" "32"

# ============================================================================
# Test 6: Phase overlap disabled with large batch
# Expected: phase_overlap recommendation
# ============================================================================
test_case \
    "phase_overlap_disabled" \
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "gguf" \
    "cuda" \
    "true" "false" "8" "32"

echo "=== All Tests Complete ==="
echo
echo "Summary:"
printf "%-35s %10s %15s\n" "Test Case" "Recommends" "Config File"
echo "--------------------------------------------------------------------------------"
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    count=$(grep -c "RECOMMEND" "$log" 2>/dev/null || echo "0")
    config_file="$LOG_DIR/${name}.yaml"
    printf "%-35s %10d %15s\n" "$name" "$count" "${config_file##*/}"
done
