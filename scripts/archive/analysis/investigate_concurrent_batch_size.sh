#!/usr/bin/env bash
#
# Concurrent Throughput Investigation: Batch Size Profiling
#
# This script investigates the concurrent throughput gap by:
# 1. Measuring actual batch sizes used at different concurrency levels
# 2. Testing different max_batch_size configurations
# 3. Profiling scheduler behavior via metrics endpoint
#
# Hypothesis: Default max_batch_size=4 limits concurrent throughput
#

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
header()   { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# Configuration
MODEL="${INFERFLUX_MODEL:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-./build}"
API_KEY="dev-key-123"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8}"
NUM_REQUESTS=16
MAX_TOKENS=64
OUTPUT_DIR="./concurrent_batch_investigation_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

# Check dependencies
if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
    log_err "inferfluxd not found at $BUILD_DIR/inferfluxd"
    log "Build with: cmake -S . -B $BUILD_DIR -DENABLE_CUDA=ON && cmake --build $BUILD_DIR -j"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    log_err "Model not found: $MODEL"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    log_err "nvidia-smi not found"
    exit 1
fi

# GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

header "Concurrent Throughput Investigation"
echo "  Model:      $MODEL"
echo "  GPU:        $GPU_NAME"
echo "  GPU Memory: ${GPU_MEM_TOTAL} MB"
echo "  Build dir:  $BUILD_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Function to create config with specific max_batch_size
create_config() {
    local max_batch=$1
    local port=$2
    local config_file="$OUTPUT_DIR/config_batch${max_batch}.yaml"

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
    batch_accumulation_ms: 0
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
    echo "$config_file"
}

# Function to start server
start_server() {
    local max_batch=$1
    local port=$2
    local config_file=$(create_config $max_batch $port)
    local log_file="$OUTPUT_DIR/server_batch${max_batch}.log"
    local pid_file="$OUTPUT_DIR/server_batch${max_batch}.pid"

    log "Starting server with max_batch_size=$max_batch on port $port..."

    INFERFLUX_PORT_OVERRIDE=$port \
    INFERCTL_API_KEY=$API_KEY \
    INFERFLUX_LOG_LEVEL=warning \
    INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE=1 \
    INFERFLUX_NATIVE_CUDA_STRICT=1 \
    "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"

    # Wait for readiness
    local waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $pid 2>/dev/null; then
            log_err "Server exited early. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            log_ok "Server ready (PID $pid)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "Server did not start in 60s"
    tail -20 "$log_file"
    kill $pid 2>/dev/null || true
    return 1
}

# Function to stop server
stop_server() {
    local max_batch=$1
    local pid_file="$OUTPUT_DIR/server_batch${max_batch}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping server (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

# Function to fetch metrics
fetch_metrics() {
    local port=$1
    local output_file=$2

    curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" > "$output_file" 2>/dev/null || true
}

# Function to extract batch size stats from metrics
extract_batch_stats() {
    local metrics_file=$1

    # Extract scheduler batch limit
    local batch_limit=$(grep -oP 'inferflux_scheduler_batch_limit_size \K\d+' "$metrics_file" 2>/dev/null || echo "N/A")

    # Extract max batch size observed
    local batch_max=$(grep -oP 'inferflux_batch_size_max\{backend="native_cuda"\} \K\d+' "$metrics_file" 2>/dev/null || echo "N/A")

    # Extract decode batch size distribution
    local decode_batch_1=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",le="1"\} \K\d+' "$metrics_file" 2>/dev/null || echo "0")
    local decode_batch_2=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",le="2"\} \K\d+' "$metrics_file" 2>/dev/null || echo "0")
    local decode_batch_4=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",le="4"\} \K\d+' "$metrics_file" 2>/dev/null || echo "0")
    local decode_batch_8=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",le="8"\} \K\d+' "$metrics_file" 2>/dev/null || echo "0")
    local decode_batch_16=$(grep -oP 'inferflux_native_forward_batch_size_total\{phase="decode",le="16"\} \K\d+' "$metrics_file" 2>/dev/null || echo "0")

    # Calculate actual batch size counts (le_X is cumulative)
    local batch_1_count=$decode_batch_1
    local batch_2_count=$((decode_batch_2 - decode_batch_1))
    local batch_4_count=$((decode_batch_4 - decode_batch_2))
    local batch_8_count=$((decode_batch_8 - decode_batch_4))
    local batch_16_count=$((decode_batch_16 - decode_batch_8))

    # Extract total decode passes
    local total_decodes=$(grep -oP 'inferflux_native_forward_passes_total\{phase="decode"\} \K\d+' "$metrics_file" 2>/dev/null || echo "1")

    # Calculate percentages
    local pct_1=$(awk "BEGIN {printf \"%.1f\", ($batch_1_count / $total_decodes) * 100}")
    local pct_2=$(awk "BEGIN {printf \"%.1f\", ($batch_2_count / $total_decodes) * 100}")
    local pct_4=$(awk "BEGIN {printf \"%.1f\", ($batch_4_count / $total_decodes) * 100}")
    local pct_8=$(awk "BEGIN {printf \"%.1f\", ($batch_8_count / $total_decodes) * 100}")
    local pct_16=$(awk "BEGIN {printf \"%.1f\", ($batch_16_count / $total_decodes) * 100}")

    cat <<EOF
{
  "batch_limit": $batch_limit,
  "batch_max_observed": $batch_max,
  "total_decode_passes": $total_decodes,
  "batch_distribution": {
    "1": {"count": $batch_1_count, "percent": $pct_1},
    "2": {"count": $batch_2_count, "percent": $pct_2},
    "4": {"count": $batch_4_count, "percent": $pct_4},
    "8": {"count": $batch_8_count, "percent": $pct_8},
    "16": {"count": $batch_16_count, "percent": $pct_16}
  }
}
EOF
}

# Function to run concurrent benchmark
run_benchmark() {
    local max_batch=$1
    local concurrency=$2
    local port=$3

    local results_dir="$OUTPUT_DIR/responses_batch${max_batch}_c${concurrency}"
    mkdir -p "$results_dir"

    log "  Running $NUM_REQUESTS requests (concurrency=$concurrency)..."

    # Clear metrics before benchmark
    curl -sf -X POST -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" >/dev/null 2>&1 || true

    local start_time=$(date +%s%N)

    # Launch concurrent requests
    local pids=()
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        (
            local prompt="Test request $i. Please respond with exactly 10 words."
            local prompt_json=$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null)

            local start_ns=$(date +%s%N)
            local response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
                --max-time 120 2>/dev/null || echo '{"error": "curl failed"}')
            local end_ns=$(date +%s%N)
            local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

            local tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")

            echo "{\"request_id\": $i, \"tokens\": $tokens, \"latency_ms\": $latency_ms}" > "$results_dir/req_$i.json"
        ) &
        pids+=($!)

        # Enforce concurrency limit
        if [ ${#pids[@]} -ge $concurrency ]; then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for remaining requests
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    local end_time=$(date +%s%N)
    local total_ms=$(( (end_time - start_time) / 1000000 ))

    # Fetch metrics immediately after benchmark
    local metrics_file="$OUTPUT_DIR/metrics_batch${max_batch}_c${concurrency}.txt"
    fetch_metrics $port "$metrics_file"

    # Aggregate results
    local total_tokens=0
    local total_latency=0
    local success_count=0

    for f in "$results_dir"/req_*.json; do
        [ -f "$f" ] || continue
        local tokens=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo "0")
        local lat=$(cat "$f" | python3 -c "import json,sys; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo "0")

        total_tokens=$((total_tokens + tokens))
        total_latency=$((total_latency + lat))
        success_count=$((success_count + 1))
    done

    local tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')" 2>/dev/null || echo "0")
    local avg_latency=0
    [ $success_count -gt 0 ] && avg_latency=$((total_latency / success_count))

    log_ok "  max_batch=$max_batch @ c=$concurrency: $success_count/$NUM_REQUESTS OK, ${tok_per_sec} tok/s, avg ${avg_latency}ms"

    # Save stats
    cat > "$OUTPUT_DIR/stats_batch${max_batch}_c${concurrency}.json" <<EOF
{
  "max_batch_size": $max_batch,
  "concurrency": $concurrency,
  "tok_per_sec": $tok_per_sec,
  "avg_latency_ms": $avg_latency,
  "total_tokens": $total_tokens,
  "total_time_ms": $total_ms,
  "success_count": $success_count,
  "total_requests": $NUM_REQUESTS
}
EOF

    echo "$tok_per_sec"
}

# Main investigation
header "Batch Size Investigation"

# Test different max_batch_size values
BATCH_SIZES=(4 8 16 32)
BASE_PORT=18100

for batch_size in "${BATCH_SIZES[@]}"; do
    port=$((BASE_PORT + batch_size))

    header "Testing max_batch_size=$batch_size"

    # Start server
    if ! start_server $batch_size $port; then
        log_err "Failed to start server with max_batch_size=$batch_size"
        continue
    fi

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

    # Test each concurrency level
    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"
    for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
        tok_per_sec=$(run_benchmark $batch_size $concurrency $port)

        # Extract batch stats
        metrics_file="$OUTPUT_DIR/metrics_batch${batch_size}_c${concurrency}.txt"
        if [ -f "$metrics_file" ]; then
            batch_stats=$(extract_batch_stats "$metrics_file")
            echo "$batch_stats" > "$OUTPUT_DIR/batch_stats_batch${batch_size}_c${concurrency}.json"

            # Print summary
            echo ""
            echo "  Batch size distribution:"
            echo "$batch_stats" | python3 -c "
import json, sys
stats = json.load(sys.stdin)
print(f\"    Limit: {stats['batch_limit']}, Max observed: {stats['batch_max_observed']}\")
print(f\"    Total decode passes: {stats['total_decode_passes']}\")
print(f\"    Batch 1: {stats['batch_distribution']['1']['percent']}% ({stats['batch_distribution']['1']['count']}/{stats['total_decode_passes']})\")
print(f\"    Batch 2: {stats['batch_distribution']['2']['percent']}% ({stats['batch_distribution']['2']['count']}/{stats['total_decode_passes']})\")
print(f\"    Batch 4: {stats['batch_distribution']['4']['percent']}% ({stats['batch_distribution']['4']['count']}/{stats['total_decode_passes']})\")
print(f\"    Batch 8: {stats['batch_distribution']['8']['percent']}% ({stats['batch_distribution']['8']['count']}/{stats['total_decode_passes']})\")
print(f\"    Batch 16: {stats['batch_distribution']['16']['percent']}% ({stats['batch_distribution']['16']['count']}/{stats['total_decode_passes']})\")
" 2>/dev/null || true
        fi

        sleep 2
    done

    # Stop server
    stop_server $batch_size
    sleep 2
done

# Generate analysis report
header "Analysis Report"
python3 - "$OUTPUT_DIR" "${BATCH_SIZES[@]}" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json, os, sys

output_dir = sys.argv[1]
batch_sizes = [int(x) for x in sys.argv[2:-1]]
concurrency_levels = sys.argv[-1].split(',')

# Load all results
results = {}
for batch_size in batch_sizes:
    results[batch_size] = {}
    for c in concurrency_levels:
        stats_file = os.path.join(output_dir, f"stats_batch{batch_size}_c{c}.json")
        batch_stats_file = os.path.join(output_dir, f"batch_stats_batch{batch_size}_c{c}.json")
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                results[batch_size][int(c)] = json.load(f)
            if os.path.exists(batch_stats_file):
                with open(batch_stats_file) as f:
                    results[batch_size][int(c)]['batch_stats'] = json.load(f)

print()
print(f"{'Max Batch':<12} {'Conc':<6} {'Tok/s':>10} {'Avg Lat':>10} {'Actual Batch Sizes':<30}")
print("-" * 90)

for batch_size in sorted(results.keys()):
    for c in sorted(results[batch_size].keys()):
        r = results[batch_size][c]
        batch_dist = r.get('batch_stats', {}).get('batch_distribution', {})

        # Show most common batch sizes
        common_sizes = []
        for size in [1, 2, 4, 8, 16]:
            if batch_dist.get(str(size), {}).get('count', 0) > 0:
                pct = batch_dist[str(size)]['percent']
                common_sizes.append(f"{size}({pct:.0f}%)")

        batch_str = ", ".join(common_sizes[:3]) if common_sizes else "N/A"

        print(f"{batch_size:<12} {c:<6} {r['tok_per_sec']:>10.1f} {r['avg_latency_ms']:>10.0f} {batch_str:<30}")
    print()

# Calculate scaling efficiency
print()
print("Scaling Efficiency (Speedup from c=1):")
print("-" * 60)
for batch_size in sorted(results.keys()):
    if 1 in results[batch_size]:
        baseline = results[batch_size][1]['tok_per_sec']
        if baseline > 0:
            print(f"max_batch={batch_size:<4}: ", end='')
            for c in sorted([int(x) for x in concurrency_levels if int(x) > 1]):
                if c in results[batch_size]:
                    speedup = results[batch_size][c]['tok_per_sec'] / baseline
                    efficiency = (speedup / c) * 100
                    print(f"c={c}={speedup:.2f}x ({efficiency:.0f}%)  ", end='')
            print()

# Key finding: Does increasing max_batch_size help at c=4?
print()
print("Key Finding: Impact of max_batch_size at c=4")
print("-" * 60)
if 4 in concurrency_levels:
    baseline_c4 = None
    if 4 in results.get(4, {}):
        baseline_c4 = results[4][4]['tok_per_sec']

    if baseline_c4:
        for batch_size in sorted(results.keys()):
            if batch_size == 4:
                continue
            if 4 in results.get(batch_size, {}):
                tok_per_sec = results[batch_size][4]['tok_per_sec']
                improvement = ((tok_per_sec - baseline_c4) / baseline_c4) * 100
                print(f"max_batch={batch_size:<4} vs max_batch=4: {improvement:+.1f}% ({baseline_c4:.1f} -> {tok_per_sec:.1f} tok/s)")

PYEOF

log_ok "Investigation complete! Results saved to $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the analysis report above"
echo "  2. Check if increasing max_batch_size improves concurrent throughput"
echo "  3. If max_batch_size helps, update default config"
echo "  4. If not, investigate GPU resource contention or other bottlenecks"
