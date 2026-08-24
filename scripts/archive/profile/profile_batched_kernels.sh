#!/usr/bin/env bash
#
# Profile Batched GPU Kernels with Nsight Systems
#
# Goal: Understand why 140+ decode passes are needed for 8 concurrent requests
# Measure: B=1 vs B=2 vs B=4 kernel execution time and efficiency

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

MODEL="${INFERFLUX_MODEL:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-./build}"
API_KEY="dev-key-123"
CONCURRENCY=4
NUM_REQUESTS=8
MAX_TOKENS=32
OUTPUT_DIR="./nsight_kernel_profile_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

# Check for Nsight Systems
if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys (Nsight Systems) not found. Install with:"
    echo "  sudo apt-get install nvidia-nsight-systems"
    exit 1
fi

header "Batched GPU Kernel Profiling with Nsight Systems"
echo "  Model: $MODEL"
echo "  Concurrency: $CONCURRENCY"
echo "  Requests: $NUM_REQUESTS"
echo "  Max tokens: $MAX_TOKENS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check available NVTX markers
log "Checking for NVTX markers in native CUDA backend..."
if grep -q "nvtxMark\|NVTX" runtime/backends/cuda/native/transformer_forward.cu 2>/dev/null; then
    log_ok "NVTX markers found in transformer_forward.cu"
else
    log_warn "No NVTX markers found - consider adding for better trace visibility"
fi

# Test configurations
# Test 1: Default config (batch_accumulation_ms=2)
# Test 2: Force single-sequence batches (min_batch_size=1, max_batch_size=1, accum_ms=0)
# Test 3: Allow larger batches (max_batch_size=16, accum_ms=5)

configs=(
    "default:4:2"
    "single:1:0"
    "large:16:5"
)

for config_info in "${configs[@]}"; do
    IFS=':' read -r name max_batch accum_ms <<< "$config_info"

    port_hash=$(echo "$name" | md5sum | head -c 1 | tr -d '[:alpha:]')
    port_hash=${port_hash:-0}
    port=$((18200 + (port_hash % 100)))
    config_file="$OUTPUT_DIR/config_${name}.yaml"

    header "Profiling: $name (max_batch=$max_batch, accum_ms=$accum_ms)"

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
    max_batch_size: $max_batch
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

    # Start server under Nsight Systems
    log "Starting server under Nsight Systems profiling..."

    nsys_report="$OUTPUT_DIR/nsys_${name}"
    nsys_stats="$OUTPUT_DIR/nsys_${name}.sqlite"

    # Warmup run first (not profiled)
    log "Warmup run (not profiled)..."
    INFERFLUX_PORT_OVERRIDE=$port \
    INFERCTL_API_KEY=$API_KEY \
    INFERFLUX_LOG_LEVEL=warning \
    INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE=1 \
    INFERFLUX_NATIVE_CUDA_STRICT=1 \
    "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$OUTPUT_DIR/server_${name}_warmup.log" 2>&1 &
    warmup_pid=$!

    # Wait for server to be ready
    waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $warmup_pid 2>/dev/null; then
            log "Server exited during warmup"
            cat "$OUTPUT_DIR/server_${name}_warmup.log" | tail -20
            break 2
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Warmup requests
    for i in {1..3}; do
        curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -d '{"model":"default","prompt":"Warmup.","max_tokens":16}' \
            >/dev/null 2>&1 || true
    done

    # Stop warmup server
    kill $warmup_pid 2>/dev/null || true
    wait $warmup_pid 2>/dev/null || true
    sleep 2

    # Now profiled run
    log "Starting profiled server run..."
    nsys profile -o "$OUTPUT_DIR/nsys_${name}" \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --sample=none \
        "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$OUTPUT_DIR/server_${name}.log" 2>&1 &
    nsys_pid=$!

    # Wait for server to be ready
    waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $nsys_pid 2>/dev/null; then
            log "Server exited early during profiling"
            cat "$OUTPUT_DIR/server_${name}.log" | tail -30
            break 2
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_ok "Server ready, starting concurrent workload..."

    # Clear metrics
    curl -sf -X POST -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" >/dev/null 2>&1 || true

    # Run concurrent workload
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
                --max-time 120 2>/dev/null || echo '{"error": "curl failed"}')

            tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")

            echo "{\"request_id\": $i, \"tokens\": $tokens}" > "$OUTPUT_DIR/req_${name}_$i.json"
        ) &
        pids+=($!)

        if [ ${#pids[@]} -ge $CONCURRENCY ]; then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        fi
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    end_time=$(date +%s%N)
    total_ms=$(( (end_time - start_time) / 1000000 ))

    # Fetch metrics after workload
    curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" > "$OUTPUT_DIR/metrics_${name}.txt" 2>/dev/null || true

    # Stop server (nsys will capture the shutdown)
    log "Stopping server and finalizing profile..."
    kill -SIGTERM $nsys_pid 2>/dev/null || true
    wait $nsys_pid 2>/dev/null || true
    sleep 3

    # Check if nsys report was created
    if [ -f "${nsys_report}.nsys-rep" ] || [ -f "${nsys_report}.qdstrm" ] || [ -f "${nsys_report}.sqlite" ]; then
        log_ok "Profile saved: ${nsys_report}.nsys-rep"

        for report in cuda_gpu_kern_sum cuda_gpu_kern_gb_sum cuda_kern_exec_sum cuda_api_sum cuda_gpu_mem_time_sum; do
            nsys stats --force-export=true --report "$report" --format csv "${nsys_report}.nsys-rep" \
                > "$OUTPUT_DIR/${report}_${name}.csv" 2> "$OUTPUT_DIR/${report}_${name}.stderr" || true
        done

        # Extract basic stats from metrics
    batch_1=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="1"\} \K\d+' "$OUTPUT_DIR/metrics_${name}.txt" 2>/dev/null || echo "0")
    batch_2=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="2"\} \K\d+' "$OUTPUT_DIR/metrics_${name}.txt" 2>/dev/null || echo "0")
    batch_34=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="3_4"\} \K\d+' "$OUTPUT_DIR/metrics_${name}.txt" 2>/dev/null || echo "0")
    total_decodes=$((batch_1 + batch_2 + batch_34))

        # Calculate throughput
    total_tokens=0
        for f in "$OUTPUT_DIR"/req_${name}_*.json; do
            [ -f "$f" ] || continue
            req_tokens=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo "0")
            total_tokens=$((total_tokens + req_tokens))
        done

    tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')" 2>/dev/null || echo "0")

        echo ""
        echo "  Results: $tok_per_sec tok/s, $total_tokens tokens, ${total_ms}ms"
        echo "  Batch distribution: B=1: $batch_1, B=2: $batch_2, B=3-4: $batch_34 (total: $total_decodes)"
        echo ""
    else
        log_warn "Nsight report not found at $nsys_report"
        log "Check server log for errors: $OUTPUT_DIR/server_${name}.log"
    fi

    sleep 2
done

# Generate analysis summary
header "Profile Analysis Summary"

echo ""
echo "Configuration | Tok/s | Total Decodes | B=1 | B=2 | B=3-4 | Profile File"
echo "---------------|-------|---------------|-----|-----|-------|-------------"

for config_info in "${configs[@]}"; do
    IFS=':' read -r name max_batch accum_ms <<< "$config_info"

    metrics_file="$OUTPUT_DIR/metrics_${name}.txt"
    nsys_file="$OUTPUT_DIR/nsys_${name}.sqlite"

    if [ -f "$metrics_file" ]; then
    batch_1=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="1"\} \K\d+' "$metrics_file" 2>/dev/null || echo "N/A")
    batch_2=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="2"\} \K\d+' "$metrics_file" 2>/dev/null || echo "N/A")
    batch_34=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",bucket="3_4"\} \K\d+' "$metrics_file" 2>/dev/null || echo "N/A")
    total_decodes=$((batch_1 + batch_2 + batch_34))

        # Calculate throughput
    total_tokens=0
        for f in "$OUTPUT_DIR"/req_${name}_*.json; do
            [ -f "$f" ] || continue
            req_tokens=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo "0")
            total_tokens=$((total_tokens + req_tokens))
        done

    tok_per_sec="N/A"
        if [ -f "$OUTPUT_DIR/timing_${name}.txt" ]; then
    total_ms=$(cat "$OUTPUT_DIR/timing_${name}.txt" 2>/dev/null || echo "0")
            if [ "$total_ms" != "0" ]; then
                tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')" 2>/dev/null || echo "N/A")
            fi
        fi

    profile_status="✓"
        [ ! -f "$nsys_file" ] && profile_status="✗"

        printf "%-13s | %-5s | %-13s | %-3s | %-3s | %-5s | %s\n" \
            "$name" "$tok_per_sec" "$total_decodes" "$batch_1" "$batch_2" "$batch_34" "$profile_status"
    fi
done

echo ""
log_ok "Profiling complete!"
echo ""
echo "Next steps:"
echo "  1. Open profiles in Nsight Systems GUI:"
echo "     nsys-ui $OUTPUT_DIR/nsys_*.sqlite"
echo ""
echo "  2. Look for:"
echo "     - CUDA kernel execution time (GEMV, attention)"
echo "     - Batch size distribution in decode phases"
echo "     - GPU utilization percentage"
echo "     - Memory bandwidth saturation"
echo "     - Time spent in batched vs single-sequence kernels"
echo ""
echo "  3. Compare:"
echo "     - single (forced B=1) vs default vs large (B=16 allowed)"
echo "     - Kernel efficiency: Is B=4 actually 4× faster than B=1?"
echo ""
echo "  4. Export reports:"
echo "     nsys stats $OUTPUT_DIR/nsys_default.nsys-rep --report cuda_gpu_kern_gb_sum"
echo ""
