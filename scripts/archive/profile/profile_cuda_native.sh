#!/usr/bin/env bash
#
# cuda_native Nsight Systems Profiling Script
#
# Profiles cuda_native execution at different concurrency levels
# to identify bottlenecks and understand why it doesn't scale.
#
# Usage:
#   ./scripts/profile_cuda_native.sh [model.gguf]
#   CONCURRENCY="1,16" ./scripts/profile_cuda_native.sh model.gguf

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()  { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# Configuration
MODEL_PATH="${1:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-./build-cuda}"
CONCURRENCY="${CONCURRENCY:-1,16}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
OUTPUT_DIR="./cuda_native_profile_$(date +%Y%m%d_%H%M%S)"
API_KEY="dev-key-123"
PORT=18090

# Create output directory
mkdir -p "$OUTPUT_DIR"

header "cuda_native Nsight Systems Profiling"
echo "Model: $MODEL_PATH"
echo "Concurrency: $CONCURRENCY"
echo "Requests per level: $NUM_REQUESTS"
echo "Output: $OUTPUT_DIR"
echo ""

# Validate model
if [ ! -f "$MODEL_PATH" ]; then
    log_err "Model not found: $MODEL_PATH"
    exit 1
fi

# Validate binary
if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
    log_err "Binary not found: $BUILD_DIR/inferfluxd"
    log "Build with: cmake -S . -B $BUILD_DIR -DENABLE_CUDA=ON && cmake --build $BUILD_DIR -j"
    exit 1
fi

# Check nsys
if ! command -v nsys &> /dev/null; then
    log_err "nsys not found. Install Nsight Systems toolkit."
    exit 1
fi

log_ok "Prerequisites validated"

# Generate server config
cat > "$OUTPUT_DIR/config.yaml" <<EOF
server:
  host: "127.0.0.1"
  http_port: $PORT
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
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
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: 1
    batch_accumulation_ms: 2
  paged_kv:
    cpu_pages: 4096
    eviction: lru

auth:
  api_keys:
    - key: $API_KEY
      scopes: [generate, read, admin]
  rate_limit_per_minute: 600

guardrails:
  blocklist: []

logging:
  level: warning
  format: text
EOF

# Function to profile at specific concurrency
profile_concurrency() {
    local concurrency=$1
    local profile_file="$OUTPUT_DIR/profile_c${concurrency}.nsys-rep"

    header "Profiling at concurrency=$concurrency"

    # Start server in background with profiling
    log "Starting cuda_native server with profiling..."

    INFERFLUX_PORT_OVERRIDE=$PORT \
    INFERCTL_API_KEY=$API_KEY \
    INFERFLUX_LOG_LEVEL=warning \
    INFERFLUX_NATIVE_KV_MAX_BATCH=16 \
    INFERFLUX_NATIVE_KV_MAX_SEQ=2048 \
    nsys profile -o "$profile_file" \
        --trace=cuda,nvtx,osrt \
        --force-overwrite=true \
        --cuda-memory-usage=true \
        $BUILD_DIR/inferfluxd --config "$OUTPUT_DIR/config.yaml" \
        > "$OUTPUT_DIR/server_c${concurrency}.log" 2>&1 &

    local server_pid=$!
    log "Server PID: $server_pid"

    # Wait for server to be ready
    log "Waiting for server to be ready..."
    local waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $server_pid 2>/dev/null; then
            log_err "Server exited early"
            cat "$OUTPUT_DIR/server_c${concurrency}.log"
            return 1
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$PORT/livez" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if [ $waited -ge 60 ]; then
        log_err "Server did not become ready"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    log_ok "Server ready"

    # Warmup
    log "Running warmup..."
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d '{"model":"default","prompt":"Warmup","max_tokens":10,"temperature":0.0}' \
        >/dev/null 2>&1 || true

    sleep 2

    # Run workload
    log "Running workload: $NUM_REQUESTS requests @ concurrency=$concurrency..."

    cat > /tmp/workload.py <<'PYEOF'
import requests
import concurrent.futures
import time

API_KEY = "dev-key-123"
BASE_URL = f"http://127.0.0.1:$PORT"
PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France? Answer in one word.",
]

def make_request(i):
    prompt = PROMPTS[i % len(PROMPTS)]
    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "default", "prompt": prompt, "max_tokens": 64, "temperature": 0.0},
            timeout=120
        )
        end = time.time()
        if resp.status_code == 200:
            return True
        return False
    except Exception as e:
        return False

def run(concurrency, num_requests):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    elapsed = time.time() - start
    success = sum(results)
    return elapsed, success

if __name__ == "__main__":
    import sys
    concurrency = int(sys.argv[1])
    num_requests = int(sys.argv[2])
    elapsed, success = run(concurrency, num_requests)
    print(f"Elapsed: {elapsed:.1f}s, Success: {success}/{num_requests}")
PYEOF

    python3 /tmp/workload.py "$concurrency" "$NUM_REQUESTS"

    # Let profiler finish capturing
    log "Waiting for profiler to finish..."
    sleep 3

    # Stop server
    log "Stopping server..."
    kill -TERM $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true

    log_ok "Profile saved to: $profile_file"
    echo ""

    # Generate quick analysis
    log "Generating profile analysis..."
    nsys stats "$profile_file" > "$OUTPUT_DIR/stats_c${concurrency}.txt" 2>&1 || true

    # Extract key metrics
    if command -v nsys &> /dev/null; then
        echo "CUDA API calls:" > "$OUTPUT_DIR/summary_c${concurrency}.txt"
        nsys stats "$profile_file" 2>&1 | grep -E "CUDA|cuda" | head -20 >> "$OUTPUT_DIR/summary_c${concurrency}.txt" || true
    fi
}

# Run profiles for each concurrency level
IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY"

for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    profile_concurrency "$concurrency"
done

header "Profiling Complete"
log "Output directory: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"
echo ""
echo "To view profiles:"
echo "  nsys-ui $OUTPUT_DIR/profile_c1.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/profile_c16.nsys-rep"
echo ""
echo "To export to CSV:"
echo "  nsys export --type csv --output $OUTPUT_DIR/profile_c1.csv $OUTPUT_DIR/profile_c1.nsys-rep"
