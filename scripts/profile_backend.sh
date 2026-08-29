#!/usr/bin/env bash
# Profile an InferFlux CUDA backend with Nsight Systems and export fine-grained reports.

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

BACKEND_INPUT="${1:-inferflux_cuda}"
MODEL_PATH="${2:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
OUTPUT_DIR="${3:-./nsys_backend_profile_${BACKEND_INPUT}_$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-./build-cuda}"
SERVER_BIN="$BUILD_DIR/inferfluxd"
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
GPU_PROFILE="${GPU_PROFILE:-ada_rtx_4000}"
WORKLOAD_GPU_PROFILE="${WORKLOAD_GPU_PROFILE:-none}"
PORT=""
HOST="127.0.0.1"
CONCURRENCY="${CONCURRENCY:-4}"
REQUESTS="${REQUESTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
WORKLOAD_PROMPT="${WORKLOAD_PROMPT:-Explain why throughput guardrails matter.}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
BATCH_ACCUMULATION_MS="${BATCH_ACCUMULATION_MS:-2}"
DECODE_BURST_TOKENS="${DECODE_BURST_TOKENS:-0}"
ENABLE_BATCHED_DECODE="${ENABLE_BATCHED_DECODE:-1}"
BENCH_LOG_LEVEL="${BENCH_LOG_LEVEL:-warning}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-}"
PROFILE_STEADY_STATE_ONLY="${PROFILE_STEADY_STATE_ONLY:-1}"
PROFILE_LOCK_FILE="${INFERFLUX_PROFILE_LOCK_FILE:-/tmp/inferflux_profile_backend.lock}"
PROFILE_LOCK_WAIT_SEC="${INFERFLUX_PROFILE_LOCK_WAIT_SEC:-300}"
PROFILE_PID=""
NSYS_SESSION_NAME=""

normalize_backend() {
  case "$BACKEND_INPUT" in
    native|inferflux|inferflux_cuda|cuda_native|native_cuda)
      BACKEND_ID="inferflux_cuda"
      BACKEND_PORT_DEFAULT=18088
      PREFER_INFERFLUX="true"
      ALLOW_LLAMA_FALLBACK="false"
      STRICT_NATIVE="1"
      DISABLE_PARITY_DELEGATE="1"
      ;;
    llamacpp|llama_cpp|llama_cpp_cuda|cuda_llama_cpp|cuda_llama)
      BACKEND_ID="llama_cpp_cuda"
      BACKEND_PORT_DEFAULT=18089
      PREFER_INFERFLUX="false"
      ALLOW_LLAMA_FALLBACK="true"
      STRICT_NATIVE="0"
      DISABLE_PARITY_DELEGATE="0"
      ;;
    *)
      log_err "Unknown backend '$BACKEND_INPUT' (use inferflux_cuda or llama_cpp_cuda)"
      exit 1
      ;;
  esac
  PORT="${INFERFLUX_PROFILE_PORT:-$BACKEND_PORT_DEFAULT}"
}

resolve_startup_timeout() {
  if [ -n "$STARTUP_TIMEOUT_SEC" ]; then
    echo "$STARTUP_TIMEOUT_SEC"
  elif [ "$BACKEND_ID" = "llama_cpp_cuda" ]; then
    echo 180
  else
    echo 60
  fi
}

require_tools() {
  command -v nsys >/dev/null 2>&1 || { log_err "nsys not found"; exit 1; }
  command -v curl >/dev/null 2>&1 || { log_err "curl not found"; exit 1; }
  [ -x "$SERVER_BIN" ] || { log_err "inferfluxd not found at $SERVER_BIN"; exit 1; }
  [ -f "$MODEL_PATH" ] || { log_err "model not found at $MODEL_PATH"; exit 1; }
}

write_config() {
  cat > "$OUTPUT_DIR/config.yaml" <<CFG
server:
  host: "$HOST"
  http_port: $PORT
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
    format: gguf
    backend: $BACKEND_ID
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
    prefer_inferflux: $PREFER_INFERFLUX
    allow_llama_cpp_fallback: $ALLOW_LLAMA_FALLBACK
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: $MIN_BATCH_SIZE
    batch_accumulation_ms: $BATCH_ACCUMULATION_MS
    decode_burst_tokens: $DECODE_BURST_TOKENS
  disaggregated:
    prefill_pool_size: 1
    decode_pool_size: 1
    kv_channel_capacity: 64
    kv_enqueue_max_retries: 3
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
  level: $BENCH_LOG_LEVEL
  format: text
CFG
}

tail_log() {
  local file=$1
  [ -f "$file" ] && tail -20 "$file"
}

wait_ready() {
  local startup_timeout
  startup_timeout=$(resolve_startup_timeout)
  local waited=0
  while [ "$waited" -lt "$startup_timeout" ]; do
    if [ -n "$PROFILE_PID" ] && ! kill -0 "$PROFILE_PID" 2>/dev/null; then
      return 1
    fi
    if curl -sf -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/livez" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

reset_cuda_device() {
  python3 - <<'PY' >/dev/null 2>&1
import ctypes, ctypes.util, sys
libname = ctypes.util.find_library('cudart')
if not libname:
    sys.exit(1)
try:
    cudart = ctypes.CDLL(libname)
except OSError:
    sys.exit(1)
if cudart.cudaDeviceReset() != 0:
    sys.exit(2)
PY
}

stop_profiled_server() {
  if [ -n "$NSYS_SESSION_NAME" ]; then
    nsys stop --session "$NSYS_SESSION_NAME" >/dev/null 2>&1 || true
    NSYS_SESSION_NAME=""
  fi
  if [ -n "$PROFILE_PID" ] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    kill -TERM "$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
  fi
  PROFILE_PID=""
}

cleanup() {
  stop_profiled_server
}
trap cleanup EXIT

acquire_profile_lock() {
  mkdir -p "$(dirname "$PROFILE_LOCK_FILE")"
  exec 9>"$PROFILE_LOCK_FILE"
  if ! flock -w "$PROFILE_LOCK_WAIT_SEC" 9; then
    log_err "Failed to acquire profile lock at $PROFILE_LOCK_FILE"
    exit 1
  fi
}

start_profiled_server() {
  local server_log="$OUTPUT_DIR/server.log"
  if [ "$PROFILE_STEADY_STATE_ONLY" = "1" ]; then
    NSYS_SESSION_NAME="inferflux_steady_${BACKEND_ID}_$$"
    env \
      INFERFLUX_PORT_OVERRIDE="$PORT" \
      INFERCTL_API_KEY="$API_KEY" \
      INFERFLUX_LOG_LEVEL="$BENCH_LOG_LEVEL" \
      INFERFLUX_ENABLE_BATCHED_DECODE="$ENABLE_BATCHED_DECODE" \
      INFERFLUX_CUDA_STRICT="$STRICT_NATIVE" \
      INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE="$DISABLE_PARITY_DELEGATE" \
      nsys launch \
        --session-new "$NSYS_SESSION_NAME" \
        --trace cuda,nvtx,osrt \
        --cuda-graph-trace node \
        --cuda-memory-usage true \
        "$SERVER_BIN" --config "$OUTPUT_DIR/config.yaml" \
        > "$server_log" 2>&1 &
  else
    local profile_prefix="$OUTPUT_DIR/profile"
    env \
      INFERFLUX_PORT_OVERRIDE="$PORT" \
      INFERCTL_API_KEY="$API_KEY" \
      INFERFLUX_LOG_LEVEL="$BENCH_LOG_LEVEL" \
      INFERFLUX_ENABLE_BATCHED_DECODE="$ENABLE_BATCHED_DECODE" \
      INFERFLUX_CUDA_STRICT="$STRICT_NATIVE" \
      INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE="$DISABLE_PARITY_DELEGATE" \
      nsys profile \
        --output "$profile_prefix" \
        --force-overwrite=true \
        --sample=none \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --cuda-memory-usage=true \
        "$SERVER_BIN" --config "$OUTPUT_DIR/config.yaml" \
        > "$server_log" 2>&1 &
  fi
  PROFILE_PID=$!

  if ! wait_ready; then
    log_err "Server failed to become ready. Last log lines:"
    tail_log "$server_log"
    return 1
  fi
  if [ "$PROFILE_STEADY_STATE_ONLY" = "1" ]; then
    log_ok "$BACKEND_ID ready under nsys session $NSYS_SESSION_NAME (PID $PROFILE_PID)"
  else
    log_ok "$BACKEND_ID ready under nsys (PID $PROFILE_PID)"
  fi
}

start_profile_collection() {
  if [ "$PROFILE_STEADY_STATE_ONLY" != "1" ]; then
    return 0
  fi
  local control_log="$OUTPUT_DIR/nsys_control.log"
  if ! nsys start \
      --session "$NSYS_SESSION_NAME" \
      --output "$OUTPUT_DIR/profile" \
      --force-overwrite true \
      --sample none \
      >"$control_log" 2>&1; then
    log_err "Failed to start steady-state nsys collection. Last control log lines:"
    tail_log "$control_log"
    return 1
  fi
}

stop_profile_collection() {
  if [ "$PROFILE_STEADY_STATE_ONLY" != "1" ] || [ -z "$NSYS_SESSION_NAME" ]; then
    return 0
  fi
  local control_log="$OUTPUT_DIR/nsys_control.log"
  if ! nsys stop --session "$NSYS_SESSION_NAME" >>"$control_log" 2>&1; then
    log_err "Failed to stop steady-state nsys collection. Last control log lines:"
    tail_log "$control_log"
    return 1
  fi
}

run_workload() {
  local metrics_file="$OUTPUT_DIR/metrics_prometheus.txt"
  local workload_log="$OUTPUT_DIR/workload.log"
  local results_dir="$OUTPUT_DIR/responses"
  mkdir -p "$results_dir"
  : > "$workload_log"

  send_profile_request() {
    local request_id=$1
    local output_file=$2
    local prompt_json
    prompt_json=$(printf '%s [req-%s]' "$WORKLOAD_PROMPT" "$request_id" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    local start_ns end_ns latency_ms
    start_ns=$(date +%s%N)
    local response
    response=$(curl -sf -X POST "http://$HOST:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_KEY" \
      -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
      --max-time "$REQUEST_TIMEOUT_SEC" 2>>"$workload_log") || {
        echo "ERROR" > "$output_file"
        return 1
      }
    end_ns=$(date +%s%N)
    latency_ms=$(( (end_ns - start_ns) / 1000000 ))
    python3 - <<'PY' "$response" "$latency_ms" "$output_file"
import json
import sys

payload = json.loads(sys.argv[1])
latency_ms = int(sys.argv[2])
out_path = sys.argv[3]
choice = payload.get("choices", [{}])[0]
text = choice.get("message", {}).get("content", "") or choice.get("text", "")
tokens = payload.get("usage", {}).get("completion_tokens", 0)
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"text": text.strip(), "tokens": tokens, "latency_ms": latency_ms}, handle)
PY
  }

  log "Warmup..."
  for i in $(seq 0 $((WARMUP_REQUESTS - 1))); do
    send_profile_request "warmup-$i" "/dev/null" || true
  done

  start_profile_collection
  log "Running profiled workload..."
  local start_ns
  start_ns=$(date +%s%N)
  local pids=()
  for i in $(seq 0 $((REQUESTS - 1))); do
    send_profile_request "$i" "$results_dir/req_$i.json" &
    pids+=($!)
    if [ ${#pids[@]} -ge "$CONCURRENCY" ]; then
      wait "${pids[0]}" 2>/dev/null || true
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  local end_ns elapsed_ms
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  stop_profile_collection

  python3 - <<'PY' "$results_dir" "$elapsed_ms" | tee -a "$workload_log"
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
elapsed_ms = int(sys.argv[2])
total_tokens = 0
success = 0
latencies = []
for path in sorted(results_dir.glob("req_*.json")):
    raw = path.read_text(encoding="utf-8").strip()
    if raw == "ERROR":
        continue
    payload = json.loads(raw)
    success += 1
    total_tokens += int(payload.get("tokens", 0))
    latencies.append(int(payload.get("latency_ms", 0)))

tok_per_sec = 0.0
if elapsed_ms > 0:
    tok_per_sec = total_tokens / (elapsed_ms / 1000.0)
avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
print(json.dumps({
    "success": success,
    "requests": len(list(results_dir.glob("req_*.json"))),
    "tokens": total_tokens,
    "elapsed_ms": elapsed_ms,
    "tok_per_sec": round(tok_per_sec, 2),
    "avg_latency_ms": round(avg_latency, 2),
}, indent=2))
PY

  curl -sf -H "Authorization: Bearer $API_KEY" \
    "http://$HOST:$PORT/metrics" > "$metrics_file" 2>/dev/null || true
}

export_reports() {
  local profile_file="$OUTPUT_DIR/profile.nsys-rep"
  local reports=(
    cuda_gpu_kern_sum
    cuda_gpu_kern_gb_sum
    cuda_kern_exec_sum
    cuda_api_sum
    cuda_gpu_mem_time_sum
  )

  if [ ! -f "$profile_file" ]; then
    log_warn "Profile not found at $profile_file"
    return 0
  fi

  for report in "${reports[@]}"; do
    local report_csv="$OUTPUT_DIR/${report}.csv"
    local report_stderr="$OUTPUT_DIR/${report}.stderr"
    if ! nsys stats --force-export=true --report "$report" --format csv "$profile_file" \
        > "$report_csv" 2> "$report_stderr"; then
      log_warn "Report $report failed; see $report_stderr"
    fi
  done
}

summarize_profile() {
  local summary_txt="$OUTPUT_DIR/summary.txt"
  local summary_json="$OUTPUT_DIR/summary.json"
  if python3 scripts/analyze_nsys_results.py "$OUTPUT_DIR" --json "$summary_json" > "$summary_txt"; then
    cat "$summary_txt"
  else
    log_warn "Profile analysis failed; inspect raw CSV reports in $OUTPUT_DIR"
  fi
}

main() {
  normalize_backend
  require_tools
  mkdir -p "$OUTPUT_DIR"

  header "Nsight Systems Backend Profile"
  echo "  Backend:     $BACKEND_ID"
  echo "  Build dir:   $BUILD_DIR"
  echo "  Model:       $MODEL_PATH"
  echo "  Output:      $OUTPUT_DIR"
  echo "  Host/port:   $HOST:$PORT"
  echo "  Workload:    requests=$REQUESTS concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS"
  echo "  Prompt:      $WORKLOAD_PROMPT"
  echo "  Scheduler:   min_batch=$MIN_BATCH_SIZE accum_ms=$BATCH_ACCUMULATION_MS decode_burst=$DECODE_BURST_TOKENS batched_decode=$ENABLE_BATCHED_DECODE"
  echo "  Startup:     $(resolve_startup_timeout)s timeout"
  echo "  Profile:     $([ "$PROFILE_STEADY_STATE_ONLY" = "1" ] && echo steady-state-only || echo includes-startup)"
  echo "  Lock file:   $PROFILE_LOCK_FILE"

  acquire_profile_lock
  write_config
  start_profiled_server
  run_workload
  stop_profiled_server
  export_reports
  summarize_profile

  if [ "$BACKEND_ID" = "inferflux_cuda" ]; then
    if reset_cuda_device; then
      log_ok "CUDA device reset"
    else
      log_warn "CUDA device reset unavailable"
    fi
  fi

  log_ok "Profile artifacts saved under $OUTPUT_DIR"
  echo "  Compare runs with: python3 scripts/analyze_nsys_results.py <dirA> <dirB>"
}

main "$@"
