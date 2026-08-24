#!/usr/bin/env bash
#
# Kernel Timing Profiling - Uses INFERFLUX_NATIVE_TIMING_SAMPLE_RATE
#

MODEL="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
BUILD_DIR="./build"
API_KEY="dev-key-123"
CONCURRENCY=4
NUM_REQUESTS=4
MAX_TOKENS=32
OUTPUT_DIR="./kernel_timing_profile_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Kernel Timing Profiling"
echo "========================================"
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENCY"
echo "Requests: $NUM_REQUESTS"
echo ""

port=18500
config_file="$OUTPUT_DIR/config.yaml"

cat > "$config_file" <<EOF
server:
  host: "127.0.0.1"
  http_port: $port
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL")"
    format: gguf
    backend: cuda_native
    default: true

runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
    phase_overlap:
      enabled: true
  backend_exposure:
    prefer_native: true
    allow_llama_cpp_fallback: false
  scheduler:
    max_batch_size: 4
    max_batch_tokens: 16384
    min_batch_size: 1
    batch_accumulation_ms: 2
    policy: priority_age
  paged_kv:
    cpu_pages: 4096
    eviction: lru

auth:
  api_keys:
    - key: $API_KEY
      scopes: [generate, read, admin]

logging:
  level: warning
  format: text
EOF

echo "Starting server with timing instrumentation..."
INFERFLUX_PORT_OVERRIDE=$port \
INFERCTL_API_KEY=$API_KEY \
INFERFLUX_LOG_LEVEL=info \
INFERFLUX_NATIVE_TIMING_SAMPLE_RATE=1 \
INFERFLUX_NATIVE_PHASE_TIMING=1 \
INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE=1 \
INFERFLUX_NATIVE_CUDA_STRICT=1 \
"$BUILD_DIR/inferfluxd" --config "$config_file" \
    > "$OUTPUT_DIR/server.log" 2>&1 &
pid=$!

# Wait for readiness
waited=0
while [ $waited -lt 60 ]; do
    if ! kill -0 $pid 2>/dev/null; then
        echo "ERROR: Server exited early"
        tail -30 "$OUTPUT_DIR/server.log"
        exit 1
    fi
    if curl -sf http://127.0.0.1:$port/healthz >/dev/null 2>&1; then
        echo "Server ready (PID: $pid)"
        break
    fi
    sleep 1
    waited=$((waited + 1))
done

# Warmup
echo "Warming up..."
for i in {1..2}; do
    curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d '{"model":"default","prompt":"Warmup.","max_tokens":16}' \
        >/dev/null 2>&1 || true
done
sleep 2

# Clear metrics
curl -sf -X POST -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$port/metrics" >/dev/null 2>&1 || true

echo "Running concurrent workload (c=$CONCURRENCY, n=$NUM_REQUESTS)..."
start_time=$(date +%s%N)

pids=()
for i in $(seq 0 $((NUM_REQUESTS - 1))); do
    (
        prompt="Test request $i. Please respond with exactly 5 words."
        prompt_json=$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null)

        response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
            --max-time 120 2>/dev/null || echo '{"error": "failed"}')

        tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")

        echo "{\"request_id\": $i, \"tokens\": $tokens}" > "$OUTPUT_DIR/req_$i.json"
    ) &
    pids+=($!)

    if [ ${#pids[@]} -ge $CONCURRENCY ]; then
        wait "${pids[0]}" 2>/dev/null || true
        pids=("${pids[@]:1}")
    fi
done

for req_pid in "${pids[@]}"; do
    wait "$req_pid" 2>/dev/null || true
done

end_time=$(date +%s%N)
total_ms=$(( (end_time - start_time) / 1000000 ))

# Fetch metrics
echo "Fetching metrics..."
curl -sf -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$port/metrics" > "$OUTPUT_DIR/metrics.txt" 2>/dev/null || true

# Stop server
echo "Stopping server..."
kill $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
sleep 2

# Analyze results
echo ""
echo "========================================"
echo "Results"
echo "========================================"

total_tokens=0
for f in "$OUTPUT_DIR"/req_*.json; do
    [ -f "$f" ] || continue
    tokens=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo "0")
    total_tokens=$((total_tokens + tokens))
done

tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')" 2>/dev/null || echo "0")

# Extract batch distribution
batch_1=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="1"\} \K\d+' "$OUTPUT_DIR/metrics.txt" 2>/dev/null || echo "0")
batch_2=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="2"\} \K\d+' "$OUTPUT_DIR/metrics.txt" 2>/dev/null || echo "0")
batch_34=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="3_4"\} \K\d+' "$OUTPUT_DIR/metrics.txt" 2>/dev/null || echo "0")
batch_58=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="5_8"\} \K\d+' "$OUTPUT_DIR/metrics.txt" 2>/dev/null || echo "0")
total_decodes=$((batch_1 + batch_2 + batch_34 + batch_58))

echo "Throughput: $tok_per_sec tok/s"
echo "Total time: ${total_ms}ms"
echo "Total tokens: $total_tokens"
echo ""
echo "Batch distribution:"
echo "  B=1: $batch_1 passes"
echo "  B=2: $batch_2 passes"
echo "  B=3-4: $batch_34 passes"
echo "  B=5-8: $batch_58 passes"
echo "  Total: $total_decodes decode passes"
echo ""

# Calculate expected vs actual
expected_batches=$((NUM_REQUESTS * MAX_TOKENS))
efficiency=$(python3 -c "print(f'{($total_decodes / $expected_batches) * 100:.1f}')" 2>/dev/null || echo "N/A")
echo "Expected batches: ~$expected_batches"
echo "Actual batches: $total_decodes"
echo "Efficiency: $efficiency%"

echo ""
echo "Phase timing breakdown:"
grep -oP 'Native forward.*phase=\w+.*batch_size=\d+.*duration_ms=[\d.]+' "$OUTPUT_DIR/server.log" | \
    tail -20 || echo "No phase timing in logs (check log level)"

echo ""
echo "Full output in: $OUTPUT_DIR"
