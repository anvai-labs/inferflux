#!/bin/bash
# Quick GGUF backend performance benchmark
# Usage: ./quick_benchmark_gguf.sh [backend]
#   backend: "native" or "universal" (default: test both)

set -e

MODEL="${INFERFLUX_MODEL_PATH:-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
URL="${INFERFLUX_URL:-http://localhost:8080}"
CONCURRENT="${CONCURRENT:-8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-50}"
MAX_TOKENS="${MAX_TOKENS:-100}"
WARMUP_REQS="${WARMUP_REQS:-2}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# Check dependencies
check_dependencies() {
    local missing=0

    if ! command -v curl &> /dev/null; then
        log_error "curl not found. Install: apt-get install curl"
        missing=1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Install: apt-get install jq"
        missing=1
    fi

    return $missing
}

# Kill existing server
kill_server() {
    log_info "Stopping existing servers..."
    pkill -f "inferfluxd" || true
    sleep 2
}

# Start server with given backend
start_server() {
    local backend=$1

    kill_server

    local env_vars=""
    case "$backend" in
        "native")
            env_vars="INFERFLUX_BACKEND_PREFER_NATIVE=1"
            log_info "Starting server with cuda_native backend..."
            ;;
        "universal")
            env_vars="INFERFLUX_BACKEND_PREFER_NATIVE=0"
            log_info "Starting server with cuda_llama_cpp backend..."
            ;;
        *)
            log_error "Unknown backend: $backend"
            return 1
            ;;
    esac

    # Start server in background
    env $env_vars ./build/inferfluxd --config config/server.yaml > /tmp/inferflux_${backend}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local max_wait=30
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "$URL/healthz" > /dev/null 2>&1; then
            log_success "Server ready (PID: $SERVER_PID)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done

    log_error "Server failed to start within ${max_wait}s"
    return 1
}

# Stop server
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        log_info "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    kill_server
}

# Make a single request
make_request() {
    local prompt="The quick brown fox jumps over the lazy dog. "

    response=$(curl -s -w "\n%{http_code}" "$URL/v1/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $MAX_TOKENS,
            \"stream\": false
        }" 2>/dev/null)

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        tokens=$(echo "$body" | jq -r '.usage.completion_tokens // 0')
        echo "$tokens"
        return 0
    else
        echo "0"
        return 1
    fi
}

# Run benchmark
run_benchmark() {
    local backend=$1
    local concurrent=$2

    echo ""
    echo "========================================"
    echo "Backend: $backend"
    echo "Concurrent: $concurrent"
    echo "========================================"

    # Warmup
    log_info "Warmup ($WARMUP_REQS requests)..."
    for i in $(seq 1 $WARMUP_REQS); do
        make_request > /dev/null
    done

    # Benchmark
    log_info "Running benchmark..."
    local start_time=$(date +%s.%N)

    local pids=()
    local tokens_file="/tmp/tokens_${backend}_${concurrent}.txt"

    > "$tokens_file"

    for i in $(seq 1 $concurrent); do
        make_request >> "$tokens_file" &
        pids+=($!)
    done

    # Wait for all requests
    for pid in "${pids[@]}"; do
        wait $pid
    done

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    # Calculate results
    local total_tokens=$(awk '{s+=$1} END {print s}' "$tokens_file")
    local successful=$(wc -l < "$tokens_file")
    local tok_per_sec=$(echo "scale=2; $total_tokens / $duration" | bc)

    # Print results
    echo ""
    log_success "Results:"
    echo "  Successful: $successful / $concurrent"
    echo "  Total time: ${duration}s"
    echo "  Total tokens: $total_tokens"
    echo "  Throughput: ${tok_per_sec} tok/s"

    # Get metrics
    echo ""
    log_info "Key metrics:"
    curl -s "$URL/metrics" | grep -E "inferflux_native_forward_duration_ms.*{quantile=\"0.95\"}|cuda_lane_overlap_events_total" | head -5 || true

    # Return throughput for comparison
    echo "$tok_per_sec"
}

# Compare backends
compare_backends() {
    local native_tok=$1
    local universal_tok=$2

    if [ -z "$native_tok" ] || [ -z "$universal_tok" ]; then
        return
    fi

    echo ""
    echo "========================================"
    echo "COMPARISON"
    echo "========================================"
    echo "cuda_llama_cpp: ${universal_tok} tok/s"
    echo "cuda_native:    ${native_tok} tok/s"

    local improvement=$(echo "scale=1; (($native_tok - $universal_tok) / $universal_tok) * 100" | bc)
    local speedup=$(echo "scale=1; $native_tok / $universal_tok" | bc)

    if [ $(echo "$native_tok > $universal_tok" | bc) -eq 1 ]; then
        log_success "cuda_native is ${speedup}x faster (+${improvement}%)"
    else
        log_warn "cuda_llama_cpp is faster by ${improvement}%"
    fi
}

# Main
main() {
    local backend_to_test="${1:-both}"

    echo "========================================"
    echo "GGUF Backend Performance Benchmark"
    echo "========================================"
    echo "Model: $MODEL"
    echo "Concurrent: $CONCURRENT"
    echo "Max tokens: $MAX_TOKENS"
    echo "========================================"

    # Check dependencies
    if ! check_dependencies; then
        log_error "Missing dependencies. Please install required tools."
        exit 1
    fi

    # Check model exists
    if [ ! -f "$MODEL" ]; then
        log_error "Model not found: $MODEL"
        exit 1
    fi

    # Run benchmarks
    local native_tok=""
    local universal_tok=""

    if [ "$backend_to_test" = "both" ] || [ "$backend_to_test" = "native" ]; then
        if start_server "native"; then
            native_tok=$(run_benchmark "native" $CONCURRENT)
            stop_server
        else
            log_error "Failed to test cuda_native"
        fi
    fi

    if [ "$backend_to_test" = "both" ] || [ "$backend_to_test" = "universal" ]; then
        if start_server "universal"; then
            universal_tok=$(run_benchmark "universal" $CONCURRENT)
            stop_server
        else
            log_error "Failed to test cuda_llama_cpp"
        fi
    fi

    # Compare results
    if [ "$backend_to_test" = "both" ]; then
        compare_backends "$native_tok" "$universal_tok"
    fi

    echo ""
    log_success "Benchmark complete!"
    echo ""
    echo "Logs:"
    echo "  - Native: /tmp/inferflux_native.log"
    echo "  - Universal: /tmp/inferflux_universal.log"
}

# Cleanup on exit
trap stop_server EXIT

# Run main
main "$@"
