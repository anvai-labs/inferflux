#!/usr/bin/env bash
#
# Test: Batch Accumulation Impact on Concurrent Throughput
#

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'
BOLD='\033[1m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

MODEL="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
BUILD_DIR="./build"
API_KEY="dev-key-123"
CONCURRENCY=4
NUM_REQUESTS=8
MAX_TOKENS=64
OUTPUT_DIR="./batch_accumulation_test_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

header "Batch Accumulation Test"
echo "  Model: $MODEL"
echo "  Concurrency: $CONCURRENCY"
echo "  Requests per test: $NUM_REQUESTS"
echo ""

# Test with different batch_accumulation_ms values
ACCUMULATION_VALUES=(0 1 2 5)

for accum_ms in "${ACCUMULATION_VALUES[@]}"; do
    port=$((18100 + accum_ms))
    config_file="$OUTPUT_DIR/config_accum${accum_ms}.yaml"

    header "Testing batch_accumulation_ms=$accum_ms"

    # Create config
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
    max_batch_size: 16
    max_batch_tokens: 16384
    min_batch_size: 1
    batch_accumulation_ms: $accum_ms
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

    # Start server
    log "Starting server on port $port..."
    INFERFLUX_PORT_OVERRIDE=$port \
    INFERCTL_API_KEY=$API_KEY \
    INFERFLUX_LOG_LEVEL=warning \
    INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE=1 \
    INFERFLUX_NATIVE_CUDA_STRICT=1 \
    "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$OUTPUT_DIR/server_accum${accum_ms}.log" 2>&1 &
    pid=$!

    # Wait for readiness
    waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $pid 2>/dev/null; then
            log "Server exited early"
            cat "$OUTPUT_DIR/server_accum${accum_ms}.log" | tail -20
            break 2
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Warmup
    log "Warmup..."
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

    # Run benchmark
    log "Running $NUM_REQUESTS requests (concurrency=$CONCURRENCY)..."
    start_time=$(date +%s%N)

    pids=()
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        (
            prompt="Test request $i. Please respond with exactly 10 words."
            prompt_json=$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null)

            req_start_ns=$(date +%s%N)
            response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
                --max-time 120 2>/dev/null || echo '{"error": "curl failed"}')
            req_end_ns=$(date +%s%N)
            req_latency_ms=$(( (req_end_ns - req_start_ns) / 1000000 ))

            tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")

            echo "{\"request_id\": $i, \"tokens\": $tokens, \"latency_ms\": $req_latency_ms}" > "$OUTPUT_DIR/req_accum${accum_ms}_$i.json"
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
    curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" > "$OUTPUT_DIR/metrics_accum${accum_ms}.txt" 2>/dev/null || true

    # Stop server
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 2

    # Aggregate results
    total_tokens=0
    total_latency=0
    success_count=0

    for f in "$OUTPUT_DIR"/req_accum${accum_ms}_*.json; do
        [ -f "$f" ] || continue
        req_tokens=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo "0")
        req_lat=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo "0")

        total_tokens=$((total_tokens + req_tokens))
        total_latency=$((total_latency + req_lat))
        success_count=$((success_count + 1))
    done

    tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')" 2>/dev/null || echo "0")
    avg_latency=0
    [ $success_count -gt 0 ] && avg_latency=$((total_latency / success_count))

    # Extract batch stats
    batch_1=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="1"\} \K\d+' "$OUTPUT_DIR/metrics_accum${accum_ms}.txt" 2>/dev/null || echo "0")
    batch_2=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="2"\} \K\d+' "$OUTPUT_DIR/metrics_accum${accum_ms}.txt" 2>/dev/null || echo "0")
    batch_34=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="3_4"\} \K\d+' "$OUTPUT_DIR/metrics_accum${accum_ms}.txt" 2>/dev/null || echo "0")
    batch_58=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="5_8"\} \K\d+' "$OUTPUT_DIR/metrics_accum${accum_ms}.txt" 2>/dev/null || echo "0")
    total_decodes=$((batch_1 + batch_2 + batch_34 + batch_58))

    pct_1=0 pct_2=0 pct_34=0 pct_58=0
    if [ $total_decodes -gt 0 ]; then
        pct_1=$(python3 -c "print(f'{$batch_1 / $total_decodes * 100:.1f}')" 2>/dev/null || echo "0")
        pct_2=$(python3 -c "print(f'{$batch_2 / $total_decodes * 100:.1f}')" 2>/dev/null || echo "0")
        pct_34=$(python3 -c "print(f'{$batch_34 / $total_decodes * 100:.1f}')" 2>/dev/null || echo "0")
        pct_58=$(python3 -c "print(f'{$batch_58 / $total_decodes * 100:.1f}')" 2>/dev/null || echo "0")
    fi

    log_ok "accum_ms=$accum_ms: $success_count/$NUM_REQUESTS OK, ${tok_per_sec} tok/s, avg ${avg_latency}ms"
    echo "  Batch distribution: B=1: ${pct_1}%, B=2: ${pct_2}%, B=3-4: ${pct_34}%, B=5-8: ${pct_58}%"
    echo "  Total decode passes: $total_decodes"
    echo ""

    # Save summary
    cat > "$OUTPUT_DIR/summary_accum${accum_ms}.json" <<EOF
{
  "accumulation_ms": $accum_ms,
  "tok_per_sec": $tok_per_sec,
  "avg_latency_ms": $avg_latency,
  "total_tokens": $total_tokens,
  "total_time_ms": $total_ms,
  "success_count": $success_count,
  "batch_distribution": {
    "1": {"count": $batch_1, "percent": $pct_1},
    "2": {"count": $batch_2, "percent": $pct_2},
    "3_4": {"count": $batch_34, "percent": $pct_34},
    "5_8": {"count": $batch_58, "percent": $pct_58}
  },
  "total_decode_passes": $total_decodes
}
EOF
done

# Final comparison
header "Results Summary"
echo ""
echo "accum_ms | Tok/s  | Avg Lat | B=1% | B=2% | B=3-4% | Decodes"
echo "---------|--------|---------|------|------|--------|--------"

for accum_ms in "${ACCUMULATION_VALUES[@]}"; do
    if [ -f "$OUTPUT_DIR/summary_accum${accum_ms}.json" ]; then
        data=$(cat "$OUTPUT_DIR/summary_accum${accum_ms}.json")
        tps=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['tok_per_sec'])" 2>/dev/null || echo "N/A")
        lat=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['avg_latency_ms'])" 2>/dev/null || echo "N/A")
        b1=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['batch_distribution']['1']['percent'])" 2>/dev/null || echo "N/A")
        b2=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['batch_distribution']['2']['percent'])" 2>/dev/null || echo "N/A")
        b34=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['batch_distribution']['3_4']['percent'])" 2>/dev/null || echo "N/A")
        decodes=$(echo "$data" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_decode_passes'])" 2>/dev/null || echo "N/A")

        printf "%-8d | %-6s | %-7s | %-4s | %-4s | %-6s | %s\n" \
            $accum_ms "$tps" "$lat" "$b1" "$b2" "$b34" "$decodes"
    fi
done

echo ""
log_ok "Test complete! Results saved to $OUTPUT_DIR"
