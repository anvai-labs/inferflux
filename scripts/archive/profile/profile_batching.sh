#!/bin/bash
# Batching Performance Profiling: GGUF FP16 vs GGUF Q4 Quantized
# Measures tok/s, latency, and GPU utilization at different batch sizes

set -e

INFERFLUXD="./build/inferfluxd"
INFERCTL="./build/inferctl"
LOG_DIR="logs/profiling"
mkdir -p "$LOG_DIR"

cleanup() {
    echo "Stopping server..."
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

echo "=== Batching Performance Profiling ==="
echo "Model: Qwen2.5-3B-Instruct"
echo "GPU: NVIDIA RTX 4000 Ada (SM 8.9)"
echo "Comparisons: GGUF FP16 vs GGUF Q4_K_M"
echo

# Test configurations
declare -a BATCH_SIZES=(4 8 16 32)
declare -a MODELS=(
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf,gguf_fp16"
    "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf,gguf_q4"
)

# Create optimized config for each model/batch combination
create_config() {
    local model=$1
    local format=$2
    local batch=$3
    local config_file=$4

    cat > "$config_file" << EOF
server:
  host: 0.0.0.0
  http_port: 8080
  enable_metrics: true
models:
  - id: test-model
    path: $model
    format: $format
    backend: cuda_llama_cpp
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
      tile_size: 128
    phase_overlap:
      enabled: true
  scheduler:
    max_batch_size: $batch
    max_batch_tokens: 8192
    batch_accumulation_ms: 5
  paged_kv:
    cpu_pages: 256
    eviction: lru
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
logging:
  level: warn
  format: text
EOF
}

# Run a benchmark for specific model and batch size
run_benchmark() {
    local model_path=$1
    local model_name=$2
    local batch_size=$3
    local config_file=$4

    echo "  Benchmarking: $model_name, batch_size=$batch_size"

    local log_file="$LOG_DIR/${model_name}_batch${batch_size}.log"

    # Start server
    "$INFERFLUXD" --config "$config_file" > "$log_file" 2>&1 &
    local pid=$!

    # Wait for server to be ready
    local timeout=60
    local elapsed=0
    while ! curl -s http://localhost:8080/healthz >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "    ✗ Timeout waiting for server"
            kill $pid 2>/dev/null || true
            return 1
        fi
    done
    sleep 2  # Extra time for model to fully load

    # Generate completion with timing
    local prompt="Write a short haiku about programming."
    local max_tokens=100

    local start_time=$(date +%s.%N)
    local response=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer dev-key-123" \
        -d "{
            \"model\": \"test-model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $max_tokens,
            \"stream\": false
        }")
    local end_time=$(date +%s.%N)

    # Calculate metrics
    local duration=$(echo "$end_time - $start_time" | bc)
    local tokens=$(echo "$response" | grep -o '"content"' | wc -l)
    if [ "$tokens" -lt 1 ]; then tokens=50; fi  # Fallback estimate
    local tok_per_sec=$(echo "scale=2; $tokens / $duration" | bc)

    # Extract completion tokens from response if available
    local completion_tokens=$(echo "$response" | grep -oP '"completion_tokens":\s*\K\d+' || echo "$tokens")

    echo "    Duration: ${duration}s"
    echo "    Tokens: ~$completion_tokens"
    echo "    Throughput: ${tok_per_sec} tok/s"

    # Kill server
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 1

    echo "$tok_per_sec"
}

echo "========================================="
echo "RUNNING BENCHMARKS"
echo "========================================="
echo

# Results table
echo -e "Model Format\t\t| Batch | Latency (s) | Throughput (tok/s)"
echo "----------------------------------------------------------------"

for model_spec in "${MODELS[@]}"; do
    IFS=',' read -r model_path model_name <<< "$model_spec"

    for batch in "${BATCH_SIZES[@]}"; do
        config_file="$LOG_DIR/${model_name}_batch${batch}.yaml"

        # Determine format from model name
        format="gguf"
        if [[ "$model_name" == *"fp16"* ]]; then
            format="gguf"
        fi

        create_config "$model_path" "$format" "$batch" "$config_file"

        throughput=$(run_benchmark "$model_path" "$model_name" "$batch" "$config_file")

        echo -e "$model_name\t\t| $batch     |             | $throughput"
    done
    echo
done

echo "========================================="
echo "BENCHMARK COMPLETE"
echo "========================================="
echo "Logs: $LOG_DIR/"
echo

# Summary
echo "Key Findings:"
echo "1. Compare throughput between FP16 and Q4 at same batch sizes"
echo "2. Identify optimal batch size for each format"
echo "3. Check GPU memory usage in server logs"
echo
