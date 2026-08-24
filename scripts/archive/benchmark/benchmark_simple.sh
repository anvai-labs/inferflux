#!/bin/bash
# Simple Benchmark: GGUF FP16 vs GGUF Q4 Quantized
# Direct HTTP API for accurate measurements

set -e

INFERFLUXD="./build/inferfluxd"
LOG_DIR="logs/benchmark"
mkdir -p "$LOG_DIR"

cleanup() {
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

echo "================================================"
echo "   GGUF FP16 vs GGUF Q4 Quantized Benchmark"
echo "================================================"
echo "Model: Qwen2.5-3B-Instruct"
echo "GPU: NVIDIA RTX 4000 Ada"
echo

PROMPT="Explain the difference between TCP and UDP in networking."
API_KEY="dev-key-123"
PORT=8080

run_test() {
    local model_path=$1
    local format=$2
    local label=$3
    local batch=$4

    config="$LOG_DIR/test_${label}_batch${batch}.yaml"
    log="$LOG_DIR/test_${label}_batch${batch}.log"

    cat > "$config" << EOF
server:
  http_port: $PORT
models:
  - id: test-model
    path: $model_path
    format: $format
    backend: cuda_llama_cpp
    default: true
runtime:
  cuda:
    enabled: true
    flash_attention:
      enabled: true
  scheduler:
    max_batch_size: $batch
  paged_kv:
    cpu_pages: 256
auth:
  api_keys:
    - key: $API_KEY
      scopes: [generate, read, admin]
logging:
  level: warn
EOF

    # Start server
    "$INFERFLUXD" --config "$config" > "$log" 2>&1 &
    pid=$!

    # Wait for readiness
    timeout=90
    elapsed=0
    while ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "  FAIL: Timeout"
            kill $pid 2>/dev/null || true
            return 1
        fi
    done
    sleep 2

    # Time the request
    start=$(date +%s.%N)
    response=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
            \"model\": \"test-model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 100
        }")
    end=$(date +%s.%N)

    duration=$(echo "$end - $start" | bc)

    # Parse response
    completion_tokens=$(echo "$response" | grep -oP '"completion_tokens":\s*\K\d+' || echo "0")
    if [ "$completion_tokens" -eq 0 ]; then
        # Extract from content
        content=$(echo "$response" | grep -oP '"content":\s*"\K[^"]*' || echo "")
        completion_tokens=$(echo "$content" | wc -w)
    fi
    if [ "$completion_tokens" -lt 10 ]; then completion_tokens=50; fi

    tps=$(echo "scale=2; $completion_tokens / $duration" | bc)

    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 1

    echo "$label|$batch|$duration|$completion_tokens|$tps"
}

echo "Format        | Batch | Time (s) | Tokens | tok/s"
echo "-----------------------------------------------------"

# Test FP16
for batch in 4 8 16 32; do
    result=$(run_test "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf" "gguf" "FP16" "$batch")
    IFS='|' read -r label batch_size dur tok tps <<< "$result"
    printf "%-13s | %-5s | %-8s | %-6s | %s\n" "$label" "$batch_size" "$dur" "$tok" "$tps"
done

# Test Q4
for batch in 4 8 16 32; do
    result=$(run_test "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf" "gguf" "Q4_K_M" "$batch")
    IFS='|' read -r label batch_size dur tok tps <<< "$result"
    printf "%-13s | %-5s | %-8s | %-6s | %s\n" "$label" "$batch_size" "$dur" "$tok" "$tps"
done

echo
echo "================================================"
echo "Logs: $LOG_DIR/"
echo
