#!/usr/bin/env bash
#
# GGUF Backend Comparison Benchmark
#
# Measures REAL throughput, memory, and response similarity between
# the `llama_cpp_cuda` and `inferflux_cuda` backends using an existing GGUF model.
#
# Usage:
#   ./scripts/run_gguf_comparison_benchmark.sh [model.gguf]
#   ./scripts/run_gguf_comparison_benchmark.sh models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#   ./scripts/run_gguf_comparison_benchmark.sh   # uses default model
#   NUM_REQUESTS=20 MAX_TOKENS=64 ./scripts/run_gguf_comparison_benchmark.sh models/foo.gguf
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROMPT_SUITE="$SCRIPT_DIR/../tests/data/benchmarks/prompt_suite_32.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# First positional argument overrides MODEL_PATH, env var is second priority
DEFAULT_MODEL="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH="${1:-${MODEL_PATH:-$DEFAULT_MODEL}}"
BUILD_DIR="${BUILD_DIR:-./build-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-./gguf_benchmark_results}"
NUM_REQUESTS="${NUM_REQUESTS:-16}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CONCURRENCY_LEVELS="${CONCURRENCY:-1,4,8}"
ENDPOINT_MODE="${INFERFLUX_BENCH_ENDPOINT_MODE:-chat}"
NATIVE_PHASE_TIMING="${INFERFLUX_BENCH_NATIVE_PHASE_TIMING:-0}"
NATIVE_TIMING_SAMPLE_RATE="${INFERFLUX_BENCH_NATIVE_TIMING_SAMPLE_RATE:-0}"
API_KEY="${API_KEY:-dev-key-123}"
QUANTIZE_TO="${QUANTIZE_TO:-}"  # Set to e.g. "q4_k_m" to convert safetensors → GGUF
PREFILL_POOL_SIZE="${INFERFLUX_BENCH_PREFILL_POOL_SIZE:-1}"
DECODE_POOL_SIZE="${INFERFLUX_BENCH_DECODE_POOL_SIZE:-1}"
KV_CHANNEL_CAPACITY="${INFERFLUX_BENCH_KV_CHANNEL_CAPACITY:-64}"
KV_ENQUEUE_MAX_RETRIES="${INFERFLUX_BENCH_KV_ENQUEUE_MAX_RETRIES:-3}"
BATCH_ACCUMULATION_MS="${INFERFLUX_BENCH_BATCH_ACCUMULATION_MS:-2}"
MIN_BATCH_SIZE="${INFERFLUX_BENCH_MIN_BATCH_SIZE:-1}"
ENABLE_BATCHED_DECODE="${INFERFLUX_BENCH_ENABLE_BATCHED_DECODE:-1}"
# Keep the benchmark on the stateless/native-safe decode policy by default.
# Higher values can improve throughput modestly, but they still regress native
# exact-match parity on the current Qwen 3B GGUF benchmark envelope.
DECODE_BURST_TOKENS="${INFERFLUX_BENCH_DECODE_BURST_TOKENS:-0}"
BENCH_LOG_LEVEL="${INFERFLUX_BENCH_LOG_LEVEL:-warning}"
DEBUG_SEQUENCE_SLOTS="${INFERFLUX_BENCH_DEBUG_SEQUENCE_SLOTS:-0}"
DEBUG_UNIFIED_ASSEMBLY="${INFERFLUX_BENCH_DEBUG_UNIFIED_ASSEMBLY:-0}"
DEBUG_UNIFIED_ASSEMBLY_LIMIT="${INFERFLUX_BENCH_DEBUG_UNIFIED_ASSEMBLY_LIMIT:-200}"
DEBUG_DECODE_MAPPING="${INFERFLUX_BENCH_NATIVE_DEBUG_DECODE_MAPPING:-0}"
DEBUG_DECODE_MAPPING_LIMIT="${INFERFLUX_BENCH_NATIVE_DEBUG_DECODE_MAPPING_LIMIT:-64}"
DEBUG_OPERATOR_SELECTION="${INFERFLUX_BENCH_NATIVE_DEBUG_OPERATOR_SELECTION:-0}"
DEBUG_OPERATOR_SELECTION_LIMIT="${INFERFLUX_BENCH_NATIVE_DEBUG_OPERATOR_SELECTION_LIMIT:-64}"
DEBUG_LOGITS="${INFERFLUX_BENCH_DEBUG_LOGITS:-0}"
DEBUG_LOGITS_LIMIT="${INFERFLUX_BENCH_DEBUG_LOGITS_LIMIT:-64}"
DEBUG_TOKEN_TRACE="${INFERFLUX_BENCH_DEBUG_TOKEN_TRACE:-0}"
DEBUG_TOKEN_TRACE_LIMIT="${INFERFLUX_BENCH_DEBUG_TOKEN_TRACE_LIMIT:-128}"
PROMPT_SUITE_PATH="${INFERFLUX_BENCH_PROMPT_SUITE:-$DEFAULT_PROMPT_SUITE}"
PORT_LLAMA=18090
PORT_NATIVE=18091

# llama.cpp tools (from submodule)
LLAMA_CONVERT="external/llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANTIZE="external/llama.cpp/build/bin/llama-quantize"

load_prompt_suite() {
    local suite_path=$1
    python3 - "$suite_path" <<'PYEOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    suite = json.load(f)
for entry in suite.get("prompts", []):
    prompt = entry.get("prompt", "")
    if prompt:
        print(prompt)
PYEOF
}

write_prompt_suite_artifacts() {
    local suite_copy="$OUTPUT_DIR/prompt_suite.json"
    local summary_file="$OUTPUT_DIR/prompt_suite_summary.json"
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        python3 - "$summary_file" <<'PYEOF'
import json
import sys

summary = {
    "suite_id": "single_prompt_override",
    "prompt_count": 1,
    "categories": {"override": 1},
    "output_modes": {"text": 1},
    "length_buckets": {"custom": 1},
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PYEOF
        return 0
    fi

    cp "$PROMPT_SUITE_PATH" "$suite_copy"
    python3 - "$PROMPT_SUITE_PATH" "$summary_file" <<'PYEOF'
import collections
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    suite = json.load(f)
prompts = suite.get("prompts", [])
summary = {
    "suite_id": suite.get("suite_id", "unknown"),
    "prompt_count": len(prompts),
    "categories": dict(collections.Counter(p.get("category", "unknown") for p in prompts)),
    "output_modes": dict(collections.Counter(p.get("output_mode", "unknown") for p in prompts)),
    "length_buckets": dict(collections.Counter(p.get("prompt_length_bucket", "unknown") for p in prompts)),
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PYEOF
}

write_prompt_rotation_plan() {
    local backend=$1 concurrency=$2
    local plan_file="$OUTPUT_DIR/prompt_plan_${backend}_c${concurrency}.json"
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        python3 - "$plan_file" "$NUM_REQUESTS" <<'PYEOF'
import json
import sys

count = int(sys.argv[2])
plan = [{
    "request_index": i,
    "prompt_id": "single_prompt_override",
    "category": "override",
    "prompt_length_bucket": "custom",
    "output_mode": "text",
} for i in range(count)]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)
PYEOF
        return 0
    fi

    python3 - "$PROMPT_SUITE_PATH" "$NUM_REQUESTS" "$plan_file" <<'PYEOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    prompts = json.load(f).get("prompts", [])
count = int(sys.argv[2])
plan = []
for i in range(count):
    entry = prompts[i % len(prompts)]
    plan.append({
        "request_index": i,
        "prompt_id": entry.get("id", f"prompt_{i}"),
        "category": entry.get("category", "unknown"),
        "prompt_length_bucket": entry.get("prompt_length_bucket", "unknown"),
        "output_mode": entry.get("output_mode", "unknown"),
    })
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)
PYEOF
}

# Prompts (deterministic, temperature=0)
if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
    PROMPTS=("$INFERFLUX_BENCH_SINGLE_PROMPT")
else
    mapfile -t PROMPTS < <(load_prompt_suite "$PROMPT_SUITE_PATH")
fi

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()   { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

if [ "$DECODE_BURST_TOKENS" -gt 1 ]; then
    log_warn "decode_burst_tokens=$DECODE_BURST_TOKENS is experimental for native accuracy; use 0 or 1 for parity-sensitive runs"
fi

capture_native_operator_metrics() {
    local port=$1 backend=$2 concurrency=$3
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_operator_summary_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ]; then
        return 0
    fi

    if ! curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" > "$metrics_file" 2>/dev/null; then
        log_warn "  Failed to scrape native metrics for operator summary"
        return 0
    fi

    python3 - "$metrics_file" "$summary_file" <<'PYEOF'
import json, re, sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
pattern = re.compile(
    r'^inferflux_cuda_down_proj_operator_total\{phase="([^"]+)",operator="([^"]+)"\}\s+(\d+)$'
)
summary = {}
with open(metrics_path) as f:
    for raw in f:
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        phase, op, count = m.group(1), m.group(2), int(m.group(3))
        summary.setdefault(phase, {})[op] = count

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

prefill = summary.get("prefill", {})
decode = summary.get("decode", {})
print("prefill=" + ",".join(f"{k}:{prefill.get(k, 0)}" for k in ("q8_1_gemv_v2", "q8_1_gemv", "q8_1_gemv_hot_fixed", "q8_1_gemv_row_pair_hot_fixed", "q8_1_gemv_row_pair_v2", "q8_1_gemv_row_pair", "q8_1_gemv_row_quad", "packed_gemv", "mmq", "fallback")))
print("decode=" + ",".join(f"{k}:{decode.get(k, 0)}" for k in ("q8_1_gemv_v2", "q8_1_gemv", "q8_1_gemv_hot_fixed", "q8_1_gemv_row_pair_hot_fixed", "q8_1_gemv_row_pair_v2", "q8_1_gemv_row_pair", "q8_1_gemv_row_quad", "packed_gemv", "mmq", "fallback")))
PYEOF
}

capture_native_ffn_proj_metrics() {
    local port=$1 backend=$2 concurrency=$3
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_ffn_proj_summary_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ]; then
        return 0
    fi

    python3 - "$metrics_file" "$summary_file" <<'PYEOF'
import json, re, sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
pattern = re.compile(
    r'^inferflux_cuda_ffn_proj_operator_total\{phase="([^"]+)",operator="([^"]+)"\}\s+(\d+)$'
)
summary = {}
with open(metrics_path) as f:
    for raw in f:
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        phase, op, count = m.group(1), m.group(2), int(m.group(3))
        summary.setdefault(phase, {})[op] = count

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

prefill = summary.get("prefill", {})
decode = summary.get("decode", {})
ops = ("q8_1_group_hot_q4k", "q8_1_group_row_pair_w4",
       "q8_1_group_row_quad_m4", "q8_1_group_mmq3", "q8_1_group_v2",
       "q8_1_group_generic", "q8_1_group_row_pair",
       "q8_1_group_row_quad", "packed_group", "fallback")
print("prefill=" + ",".join(f"{k}:{prefill.get(k, 0)}" for k in ops))
print("decode=" + ",".join(f"{k}:{decode.get(k, 0)}" for k in ops))
PYEOF
}

capture_native_batch_metrics() {
    local port=$1 backend=$2 concurrency=$3
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_batch_summary_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ]; then
        return 0
    fi

    python3 - "$metrics_file" "$summary_file" <<'PYEOF'
import json, re, sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
pattern = re.compile(
    r'^inferflux_cuda_forward_batch_size_total\{phase="([^"]+)",bucket="([^"]+)"\}\s+(\d+)$'
)
summary = {}
with open(metrics_path) as f:
    for raw in f:
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        phase, bucket, count = m.group(1), m.group(2), int(m.group(3))
        summary.setdefault(phase, {})[bucket] = count

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

orders = {
    "prefill": ("1", "2", "3_4", "5_8", "9_16", "17_32", "33_64", "65_128", "129_plus"),
    "decode": ("1", "2", "3_4", "5_8", "9_16", "17_plus"),
}
for phase in ("prefill", "decode"):
    phase_summary = summary.get(phase, {})
    order = orders[phase]
    print(phase + "=" + ",".join(f"{k}:{phase_summary.get(k, 0)}" for k in order))
PYEOF
}

capture_native_ffn_geometry_metrics() {
    local backend=$1 concurrency=$2
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_ffn_geometry_summary_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ]; then
        return 0
    fi

    python3 - "$metrics_file" "$summary_file" <<'PYEOF'
import json, re, sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
pattern = re.compile(
    r'^inferflux_cuda_ffn_proj_geometry_total\{phase="([^"]+)",operator="([^"]+)",quant="([^"]+)",m_bucket="([^"]+)",n="([^"]+)",n_bucket="([^"]+)",k="([^"]+)",k_bucket="([^"]+)",grouped_outputs="([^"]+)"\}\s+(\d+)$'
)
entries = []
with open(metrics_path) as f:
    for raw in f:
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        phase, op, quant, m_bucket, n, n_bucket, k, k_bucket, grouped, count = m.groups()
        count = int(count)
        entries.append({
            "phase": phase,
            "operator": op,
            "quant": quant,
            "m_bucket": m_bucket,
            "n": n,
            "n_bucket": n_bucket,
            "k": k,
            "k_bucket": k_bucket,
            "grouped_outputs": grouped,
            "count": count,
        })

entries.sort(key=lambda item: (-item["count"], item["phase"], item["operator"], item["quant"], item["m_bucket"]))
with open(out_path, "w") as f:
    json.dump(entries, f, indent=2, sort_keys=True)

for entry in entries[:8]:
    print(f'{entry["phase"]}:{entry["operator"]}:{entry["quant"]}:m={entry["m_bucket"]}:n={entry["n"]}:k={entry["k"]}:g={entry["grouped_outputs"]}:{entry["count"]}')
PYEOF
}

capture_native_downproj_geometry_metrics() {
    local backend=$1 concurrency=$2
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_downproj_geometry_summary_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ]; then
        return 0
    fi

    python3 - "$metrics_file" "$summary_file" <<'PYEOF'
import json, re, sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
pattern = re.compile(
    r'^inferflux_cuda_down_proj_geometry_total\{phase="([^"]+)",operator="([^"]+)",quant="([^"]+)",m_bucket="([^"]+)",n="([^"]+)",n_bucket="([^"]+)",k="([^"]+)",k_bucket="([^"]+)"\}\s+(\d+)$'
)
entries = []
with open(metrics_path) as f:
    for raw in f:
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        phase, op, quant, m_bucket, n, n_bucket, k, k_bucket, count = m.groups()
        count = int(count)
        entries.append({
            "phase": phase,
            "operator": op,
            "quant": quant,
            "m_bucket": m_bucket,
            "n": n,
            "n_bucket": n_bucket,
            "k": k,
            "k_bucket": k_bucket,
            "count": count,
        })

entries.sort(key=lambda item: (-item["count"], item["phase"], item["operator"], item["quant"], item["m_bucket"]))
with open(out_path, "w") as f:
    json.dump(entries, f, indent=2, sort_keys=True)

for entry in entries[:8]:
    print(f'{entry["phase"]}:{entry["operator"]}:{entry["quant"]}:m={entry["m_bucket"]}:n={entry["n"]}:k={entry["k"]}:{entry["count"]}')
PYEOF
}

capture_inferflux_cuda_bucket_winners() {
    local backend=$1 concurrency=$2
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_bucket_winners_${backend}_c${concurrency}.json"
    local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ] || [ ! -f "$stats_file" ]; then
        return 0
    fi

    python3 scripts/extract_native_dispatch_winners.py \
        "$metrics_file" "$summary_file" --stats "$stats_file"
}

has_fatal_runtime_signature() {
    local log_file=$1
    [ -f "$log_file" ] || return 1
    grep -Eqi 'double free|corruption \(|segmentation fault|addresssanitizer|terminate called after throwing|fatal glibc error|aborted \(core dumped\)|INFERFLUX CRASH DIAGNOSTIC' "$log_file"
}

# Extract and display crash diagnostics from server log if present.
report_crash_diagnostics() {
    local log_file=$1
    local backend=$2
    [ -f "$log_file" ] || return
    if grep -q "INFERFLUX CRASH DIAGNOSTIC" "$log_file"; then
        log_err "$backend crash diagnostic found in server log:"
        sed -n '/=== INFERFLUX CRASH DIAGNOSTIC ===/,/=== END CRASH DIAGNOSTIC ===/p' "$log_file" | while IFS= read -r line; do
            log_err "  $line"
        done
    fi
}

capture_inferflux_admin_cache_snapshot() {
    local port=$1 backend=$2 concurrency=$3
    local snapshot_file="$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json"
    local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"

    if ! curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/v1/admin/cache" > "$snapshot_file" 2>/dev/null; then
        log_warn "  Failed to capture admin cache snapshot for $backend"
        return 0
    fi

    python3 - "$snapshot_file" "$stats_file" <<'PYEOF'
import json, sys

snapshot_path, stats_path = sys.argv[1], sys.argv[2]
with open(snapshot_path) as f:
    snapshot = json.load(f)
with open(stats_path) as f:
    stats = json.load(f)

memory = snapshot.get("memory", {})
stats["cache_snapshot"] = {
    "size": snapshot.get("size", 0),
    "capacity": snapshot.get("capacity", 0),
    "hits": snapshot.get("hits", 0),
    "misses": snapshot.get("misses", 0),
    "hit_rate": snapshot.get("hit_rate", 0.0),
    "partial_hits": snapshot.get("partial_hits", 0),
    "matched_tokens": snapshot.get("matched_tokens", 0),
    "kv_reuse_count": snapshot.get("kv_reuse_count", 0),
    "kv_reuse_tokens": snapshot.get("kv_reuse_tokens", 0),
}
stats["memory_snapshot"] = memory

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

inferflux_cuda_model = memory.get("inferflux_cuda_model", {})
inferflux_cuda_kv = memory.get("inferflux_cuda_kv", {})
paged_kv = memory.get("paged_kv", {})

print("inferflux_cuda_model_reserved_bytes=" +
      str(inferflux_cuda_model.get("reserved_bytes", 0)))
print("inferflux_cuda_model_in_use_bytes=" +
      str(inferflux_cuda_model.get("in_use_bytes", 0)))
print("inferflux_cuda_kv_active_bytes=" +
      str(inferflux_cuda_kv.get("active_bytes", 0)))
print("inferflux_cuda_kv_prefix_retained_bytes=" +
      str(inferflux_cuda_kv.get("prefix_retained_bytes", 0)))
print("paged_kv_used_bytes=" + str(paged_kv.get("used_bytes", 0)))
print("paged_kv_prefix_retained_bytes=" +
      str(paged_kv.get("prefix_retained_bytes", 0)))
PYEOF
}

assert_backend_identity() {
    local backend=$1 port=$2 log_file=$3
    if [ "$backend" != "inferflux_cuda" ]; then
        return 0
    fi
    if [ "${SKIP_IDENTITY_CHECK:-0}" = "1" ]; then
        return 0
    fi

    # Scaffold model load (for weights/tokenizer) is expected — only forbid
    # parity delegate initialization, which indicates llama.cpp inference use.
    python3 scripts/check_backend_identity.py \
        --base-url "http://127.0.0.1:$port" \
        --model-id "bench-model" \
        --expected-provider "inferflux" \
        --expected-backend "inferflux_cuda" \
        --api-key "$API_KEY" \
        --log-file "$log_file" \
        --forbid-log-pattern 'Initialized native parity delegate backend'
}

reset_backend_artifacts() {
    local backend=$1 concurrency=${2:-}
    if [ -n "$concurrency" ]; then
        rm -rf "$OUTPUT_DIR/responses_${backend}_c${concurrency}"
        rm -f "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" \
              "$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt" \
              "$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json" \
              "$OUTPUT_DIR"/native_*_"${backend}"_c"${concurrency}".json \
              "$OUTPUT_DIR/similarity_c${concurrency}.json" \
              "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt"
    else
        # Clear all concurrency levels for this backend
        rm -rf "$OUTPUT_DIR/responses_${backend}"*
        rm -f "$OUTPUT_DIR/stats_${backend}"*.json \
              "$OUTPUT_DIR/metrics_${backend}"*.txt \
              "$OUTPUT_DIR/admin_cache_${backend}"*.json \
              "$OUTPUT_DIR"/native_*_"${backend}"*.json \
              "$OUTPUT_DIR"/similarity_c*.json \
              "$OUTPUT_DIR/mem_trace_${backend}"*.txt \
              "$OUTPUT_DIR/server_${backend}.log" \
              "$OUTPUT_DIR/config_${backend}.yaml" \
              "$OUTPUT_DIR/${backend}.pid"
    fi
}

# ============================================================================
# GPU memory measurement (real nvidia-smi)
# ============================================================================
gpu_mem_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_mem_total_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

# ============================================================================
# Server management
# ============================================================================
write_config() {
    local backend=$1 port=$2 config_file=$3
    cat > "$config_file" <<EOF
server:
  host: "127.0.0.1"
  http_port: $port
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
    format: gguf
    backend: $backend
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
    prefer_inferflux: $([ "$backend" = "inferflux_cuda" ] && echo "true" || echo "false")
    allow_llama_cpp_fallback: $([ "$backend" = "inferflux_cuda" ] && echo "false" || echo "true")
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: $MIN_BATCH_SIZE
    batch_accumulation_ms: $BATCH_ACCUMULATION_MS
    decode_burst_tokens: $DECODE_BURST_TOKENS
  disaggregated:
    prefill_pool_size: $PREFILL_POOL_SIZE
    decode_pool_size: $DECODE_POOL_SIZE
    kv_channel_capacity: $KV_CHANNEL_CAPACITY
    kv_enqueue_max_retries: $KV_ENQUEUE_MAX_RETRIES
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
EOF
}

start_server() {
    local backend=$1 port=$2
    local config_file="$OUTPUT_DIR/config_${backend}.yaml"
    local log_file="$OUTPUT_DIR/server_${backend}.log"

    write_config "$backend" "$port" "$config_file"

    log "Starting $backend on port $port..."

    # Right-size KV cache for native backend to reduce GPU memory overhead
    local kv_batch=${INFERFLUX_CUDA_KV_MAX_BATCH:-16}
    local kv_seq=${INFERFLUX_CUDA_KV_MAX_SEQ:-2048}

    INFERFLUX_PORT_OVERRIDE=$port INFERCTL_API_KEY=$API_KEY \
        INFERFLUX_LOG_LEVEL=$BENCH_LOG_LEVEL \
        INFERFLUX_ENABLE_BATCHED_DECODE=$ENABLE_BATCHED_DECODE \
        INFERFLUX_DEBUG_SEQUENCE_SLOTS=$DEBUG_SEQUENCE_SLOTS \
        INFERFLUX_DEBUG_UNIFIED_ASSEMBLY=$DEBUG_UNIFIED_ASSEMBLY \
        INFERFLUX_DEBUG_UNIFIED_ASSEMBLY_LIMIT=$DEBUG_UNIFIED_ASSEMBLY_LIMIT \
        INFERFLUX_DEBUG_LOGITS=$DEBUG_LOGITS \
        INFERFLUX_DEBUG_LOGITS_LIMIT=$DEBUG_LOGITS_LIMIT \
        INFERFLUX_DEBUG_TOKEN_TRACE=$DEBUG_TOKEN_TRACE \
        INFERFLUX_DEBUG_TOKEN_TRACE_LIMIT=$DEBUG_TOKEN_TRACE_LIMIT \
        INFERFLUX_CUDA_TIMING_SAMPLE_RATE=$([ "$backend" = "inferflux_cuda" ] && echo "$NATIVE_TIMING_SAMPLE_RATE" || echo "0") \
        INFERFLUX_CUDA_PHASE_TIMING=$([ "$backend" = "inferflux_cuda" ] && echo "$NATIVE_PHASE_TIMING" || echo "0") \
        INFERFLUX_CUDA_DEBUG_DECODE_MAPPING=$([ "$backend" = "inferflux_cuda" ] && echo "$DEBUG_DECODE_MAPPING" || echo "0") \
        INFERFLUX_CUDA_DEBUG_DECODE_MAPPING_LIMIT=$([ "$backend" = "inferflux_cuda" ] && echo "$DEBUG_DECODE_MAPPING_LIMIT" || echo "64") \
        INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION=$([ "$backend" = "inferflux_cuda" ] && echo "$DEBUG_OPERATOR_SELECTION" || echo "0") \
        INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION_LIMIT=$([ "$backend" = "inferflux_cuda" ] && echo "$DEBUG_OPERATOR_SELECTION_LIMIT" || echo "64") \
        INFERFLUX_CUDA_KV_MAX_BATCH=$kv_batch \
        INFERFLUX_CUDA_KV_MAX_SEQ=$kv_seq \
        INFERFLUX_CUDA_STRICT=$([ "$backend" = "inferflux_cuda" ] && echo "1" || echo "0") \
        INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE=$([ "$backend" = "inferflux_cuda" ] && echo "1" || echo "0") \
        INFERFLUX_DISABLE_CUDA_GRAPH=$([ "$backend" = "inferflux_cuda" ] && echo "${INFERFLUX_DISABLE_CUDA_GRAPH:-0}" || echo "0") \
        "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$OUTPUT_DIR/${backend}.pid"

    # Wait for readiness (max 120s — llama_cpp_cuda needs ~90s on WSL2)
    local waited=0
    while [ $waited -lt 120 ]; do
        if ! kill -0 $pid 2>/dev/null; then
            log_err "$backend server exited early. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/livez" >/dev/null 2>&1; then
            if ! assert_backend_identity "$backend" "$port" "$log_file"; then
                log_err "$backend failed backend identity contract. Last log lines:"
                tail -40 "$log_file"
                kill "$pid" 2>/dev/null || true
                return 1
            fi
            log_ok "$backend ready (PID $pid)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "$backend did not start in 120s. Last log lines:"
    tail -20 "$log_file"
    kill $pid 2>/dev/null || true
    return 1
}

stop_server() {
    local backend=$1
    local pidfile="$OUTPUT_DIR/${backend}.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $backend (PID $pid)..."
            # Allow in-flight requests to drain before sending SIGTERM.
            local grace=${INFERFLUX_BENCH_SHUTDOWN_GRACE:-5}
            sleep "$grace"
            kill "$pid" 2>/dev/null || true
            # Wait up to 30s for clean shutdown before force-killing.
            local waited=0
            while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
                sleep 1
                waited=$((waited + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                log "Force-killing $backend (PID $pid) after 30s..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi

    local log_file="$OUTPUT_DIR/server_${backend}.log"
    report_crash_diagnostics "$log_file" "$backend"
    if has_fatal_runtime_signature "$log_file"; then
        log_err "$backend log contains fatal runtime signature; treat this backend result as invalid"
        return 1
    fi

    return 0
}

reset_cuda_device() {
    log "  Resetting CUDA device..."
    if python3 - <<'PY' >/dev/null 2>&1
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
    then
        log_ok "CUDA device reset"
    else
        log_warn "CUDA device reset unavailable; GPU state may persist until OS cleanup"
    fi
}

# ============================================================================
# Request runner
# ============================================================================
send_request() {
    local port=$1 prompt=$2 max_tokens=$3 output_file=$4 request_tag=${5:-}

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)
    local prompt_json
    prompt_json=$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local response
    if [ "$ENDPOINT_MODE" = "chat" ]; then
        response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -H "x-inferflux-client-request-id: $request_tag" \
            -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":$prompt_json}],\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
            --max-time 120 2>/dev/null) || {
            echo "ERROR" > "$output_file"
            return 1
        }
    else
        response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -H "x-inferflux-client-request-id: $request_tag" \
            -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
            --max-time 120 2>/dev/null) || {
            echo "ERROR" > "$output_file"
            return 1
        }
    fi

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
choice = d.get('choices', [{}])[0]
text = choice.get('message', {}).get('content', '') or choice.get('text', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'client_request_id': '$request_tag', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

run_benchmark() {
    local backend=$1 port=$2 concurrency=${3:-4}
    local results_dir="$OUTPUT_DIR/responses_${backend}_c${concurrency}"
    mkdir -p "$results_dir"
    write_prompt_rotation_plan "$backend" "$concurrency"

    # Warmup (2 requests, sequential)
    log "  Warmup..."
    for i in 0 1; do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        send_request "$port" "$prompt" "$MAX_TOKENS" "/dev/null" "warmup-$i" || true
    done

    # Measure GPU memory after warmup (model loaded)
    sleep 1
    local mem_loaded=$(gpu_mem_mb)

    # Run benchmark
    log "  Running $NUM_REQUESTS requests (concurrency=$concurrency)..."
    local start_time=$(date +%s%N)

    # Track peak memory during benchmark
    local mem_peak=$mem_loaded
    (
        while true; do
            local m=$(gpu_mem_mb)
            echo "$m"
            sleep 0.2
        done
    ) > "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" &
    local monitor_pid=$!

    # Launch requests with concurrency limit
    local completed=0
    local pids=()
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        local outfile="$results_dir/req_${i}.json"
        local request_tag="bench-$i"

        send_request "$port" "$prompt" "$MAX_TOKENS" "$outfile" "$request_tag" &
        pids+=($!)

        # Enforce concurrency limit
        if [ ${#pids[@]} -ge $concurrency ]; then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    local end_time=$(date +%s%N)
    local total_ms=$(( (end_time - start_time) / 1000000 ))

    # Stop memory monitor
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true

    # Compute peak memory
    if [ -f "$OUTPUT_DIR/mem_trace_${backend}.txt" ]; then
        mem_peak=$(sort -rn "$OUTPUT_DIR/mem_trace_${backend}.txt" | head -1)
    fi

    # Aggregate results
    local total_tokens=0
    local total_latency=0
    local success_count=0
    local classified_failures=0
    local latencies=""

    for f in "$results_dir"/req_*.json; do
        [ -f "$f" ] || continue
        local content=$(cat "$f")
        if [ "$content" = "ERROR" ] || [ "$content" = "PARSE_ERROR" ]; then
            continue
        fi

        if ! python3 scripts/classify_benchmark_response.py "$f" >/dev/null 2>&1; then
            classified_failures=$((classified_failures + 1))
            continue
        fi

        local tokens=$(echo "$content" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo 0)
        local lat=$(echo "$content" | python3 -c "import json,sys; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo 0)

        total_tokens=$((total_tokens + tokens))
        total_latency=$((total_latency + lat))
        success_count=$((success_count + 1))
        latencies="$latencies $lat"
    done

    # Compute stats
    local tok_per_sec=0
    if [ $total_ms -gt 0 ]; then
        tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')")
    fi

    local avg_latency=0
    local p50=0 p95=0 p99=0
    if [ $success_count -gt 0 ]; then
        avg_latency=$((total_latency / success_count))
        # Compute percentiles
        local sorted_lats=$(echo $latencies | tr ' ' '\n' | sort -n)
        p50=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.50+0.5){print}")
        p95=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.95+0.5){print}")
        p99=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.99+0.5){print}")
    fi

    # Store results in file for later use
    cat > "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" <<EOF
{
    "backend": "$backend",
    "concurrency": $concurrency,
    "tok_per_sec": $tok_per_sec,
    "avg_latency_ms": $avg_latency,
    "p50_latency_ms": ${p50:-0},
    "p95_latency_ms": ${p95:-0},
    "p99_latency_ms": ${p99:-0},
    "total_tokens": $total_tokens,
    "total_time_ms": $total_ms,
    "success_count": $success_count,
    "classified_failures": $classified_failures,
    "total_requests": $NUM_REQUESTS,
    "gpu_mem_loaded_mb": $mem_loaded,
    "gpu_mem_peak_mb": ${mem_peak:-0}
}
EOF

    local memory_summary
    memory_summary=$(capture_inferflux_admin_cache_snapshot "$port" "$backend" "$concurrency" 2>/dev/null || true)

    local classified_note=""
    if [ $classified_failures -gt 0 ]; then
        classified_note=", rejected=$classified_failures"
    fi
    log_ok "  $backend: $success_count/$NUM_REQUESTS OK${classified_note}, ${tok_per_sec} tok/s, avg ${avg_latency}ms, mem ${mem_loaded}→${mem_peak} MB"
    if [ -n "${memory_summary:-}" ]; then
        log "  Memory snapshot:"
        printf '%s\n' "$memory_summary" | sed 's/^/    /'
    fi

    if [ "$backend" = "inferflux_cuda" ] && [ "$NATIVE_PHASE_TIMING" != "0" ]; then
        local phase_json="$OUTPUT_DIR/native_phase_timing_${backend}_c${concurrency}.json"
        if python3 scripts/parse_native_phase_timing.py "$OUTPUT_DIR/server_${backend}.log" --json > "$phase_json" 2>/dev/null; then
            log "  Native phase timing summary:"
            python3 scripts/parse_native_phase_timing.py "$OUTPUT_DIR/server_${backend}.log" 2>/dev/null | sed 's/^/    /'
        else
            log_warn "  Native phase timing enabled but no timing lines were parsed"
        fi
    fi

    if [ "$backend" = "inferflux_cuda" ]; then
        local op_summary
        op_summary=$(capture_native_operator_metrics "$port" "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${op_summary:-}" ]; then
            log "  Native down-proj operator summary:"
            printf '%s\n' "$op_summary" | sed 's/^/    /'
        fi
        local ffn_summary
        ffn_summary=$(capture_native_ffn_proj_metrics "$port" "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${ffn_summary:-}" ]; then
            log "  Native FFN projection summary:"
            printf '%s\n' "$ffn_summary" | sed 's/^/    /'
        fi
        local batch_summary
        batch_summary=$(capture_native_batch_metrics "$port" "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${batch_summary:-}" ]; then
            log "  Native forward batch-size summary:"
            printf '%s\n' "$batch_summary" | sed 's/^/    /'
        fi
        local ffn_geom_summary
        ffn_geom_summary=$(capture_native_ffn_geometry_metrics "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${ffn_geom_summary:-}" ]; then
            log "  Native FFN geometry summary:"
            printf '%s\n' "$ffn_geom_summary" | sed 's/^/    /'
        fi
        local down_geom_summary
        down_geom_summary=$(capture_native_downproj_geometry_metrics "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${down_geom_summary:-}" ]; then
            log "  Native down-proj geometry summary:"
            printf '%s\n' "$down_geom_summary" | sed 's/^/    /'
        fi
        local bucket_winner_summary
        bucket_winner_summary=$(capture_inferflux_cuda_bucket_winners "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${bucket_winner_summary:-}" ]; then
            log "  Native dispatch bucket winners:"
            printf '%s\n' "$bucket_winner_summary" | sed 's/^/    /'
        fi
    fi

    local pidfile="$OUTPUT_DIR/${backend}.pid"
    if [ $success_count -lt $NUM_REQUESTS ]; then
        if [ -f "$pidfile" ]; then
            local pid
            pid=$(cat "$pidfile")
            if ! kill -0 "$pid" 2>/dev/null; then
                log_err "  $backend exited during benchmark"
            fi
        fi
        return 1
    fi
}

# ============================================================================
# Response similarity comparison
# ============================================================================
compare_responses() {
    header "Response Similarity Analysis"

    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"
    for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
        echo ""
        echo -e "  ${BOLD}Concurrency = $concurrency${NC}"
        python3 - "$OUTPUT_DIR" "$concurrency" <<'PYEOF'
import collections
import json
import os
import re
import sys

results_dir, concurrency = sys.argv[1], sys.argv[2]
backends = ["llama_cpp_cuda", "inferflux_cuda"]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

def overlap(a, b):
    sa, sb = set(a), set(b)
    total = len(sa) + len(sb)
    return 2.0 * len(sa & sb) / total if total else 1.0

def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()

# Load responses per backend
responses = {}
for backend in backends:
    resp_dir = os.path.join(results_dir, f"responses_{backend}_c{concurrency}")
    if not os.path.isdir(resp_dir):
        print(f"  No responses for {backend}")
        continue
    responses[backend] = {}
    for f in sorted(os.listdir(resp_dir)):
        if not f.endswith(".json"):
            continue
        idx = int(f.split("_")[1].split(".")[0])
        try:
            with open(os.path.join(resp_dir, f)) as fh:
                data = json.load(fh)
            responses[backend][idx] = data.get("text", "")
        except:
            pass

if len(responses) < 2:
    print("  Cannot compare — need both backends")
    sys.exit(0)

# Compare matching request indices
a_resp = responses.get(backends[0], {})
b_resp = responses.get(backends[1], {})
common = sorted(set(a_resp.keys()) & set(b_resp.keys()))

if not common:
    print("  No matching requests to compare")
    sys.exit(0)

prompt_meta = {}
plan_path = os.path.join(results_dir, f"prompt_plan_inferflux_cuda_c{concurrency}.json")
if os.path.exists(plan_path):
    with open(plan_path, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            prompt_meta[int(entry.get("request_index", -1))] = entry

exact = 0
normalized_exact = 0
jaccards = []
overlaps = []
comparisons = []
details = []
category_summary = collections.defaultdict(lambda: {"count": 0, "exact": 0, "normalized_exact": 0})

for idx in common:
    ta, tb = a_resp[idx], b_resp[idx]
    is_exact = ta == tb
    is_normalized_exact = normalize(ta) == normalize(tb)
    if is_exact:
        exact += 1
    if is_normalized_exact:
        normalized_exact += 1
    toks_a, toks_b = tokenize(ta), tokenize(tb)
    j = jaccard(toks_a, toks_b)
    o = overlap(toks_a, toks_b)
    jaccards.append(j)
    overlaps.append(o)
    comparisons.append((idx, is_exact, j, o, ta[:60], tb[:60]))
    meta = prompt_meta.get(idx, {})
    category = meta.get("category", "unknown")
    category_summary[category]["count"] += 1
    if is_exact:
        category_summary[category]["exact"] += 1
    if is_normalized_exact:
        category_summary[category]["normalized_exact"] += 1
    details.append({
        "request_index": idx,
        "prompt_id": meta.get("prompt_id", f"prompt_{idx}"),
        "category": category,
        "prompt_length_bucket": meta.get("prompt_length_bucket", "unknown"),
        "output_mode": meta.get("output_mode", "unknown"),
        "exact": is_exact,
        "normalized_exact": is_normalized_exact,
        "jaccard": j,
        "overlap": o,
        "llama_cpp_cuda_text": ta,
        "inferflux_cuda_text": tb,
    })

n = len(common)
print(f"  Compared: {n} request pairs")
print(f"  Exact match rate: {exact}/{n} ({100*exact/n:.0f}%)")
print(f"  Normalized exact: {normalized_exact}/{n} ({100*normalized_exact/n:.0f}%)")
print(f"  Mean Jaccard:     {sum(jaccards)/n:.3f}")
print(f"  Mean overlap:     {sum(overlaps)/n:.3f}")
print()

fmt = "  {:>4}  {:>6}  {:>8}  {:>8}  {:<30}  {:<30}"
print(fmt.format("Req#", "Match", "Jaccard", "Overlap",
                  backends[0][:30], backends[1][:30]))
print("  " + "-" * 96)
for idx, match, j, o, ta, tb in comparisons:
    m = "\033[0;32mYES\033[0m" if match else "\033[0;31mNO\033[0m"
    print(f"  {idx:>4}  {m:>15}  {j:>8.3f}  {o:>8.3f}  {ta:<30}  {tb:<30}")

# Save comparison JSON
comp_data = {
    "concurrency": int(concurrency),
    "num_compared": n,
    "exact_match_rate": exact / n,
    "normalized_exact_match_rate": normalized_exact / n,
    "mean_jaccard": sum(jaccards) / n,
    "mean_overlap": sum(overlaps) / n,
    "category_summary": dict(category_summary),
}
with open(os.path.join(results_dir, f"similarity_c{concurrency}.json"), "w") as f:
    json.dump(comp_data, f, indent=2)
with open(os.path.join(results_dir, f"similarity_details_c{concurrency}.json"), "w") as f:
    json.dump(details, f, indent=2)

PYEOF
    done
}

# ============================================================================
# Final report
# ============================================================================
print_report() {
    header "Backend Comparison Report"

    echo ""
    echo "  Model:       $MODEL_PATH"
    echo "  GPU:         $(gpu_name)"
    echo "  GPU Memory:  $(gpu_mem_total_mb) MB total"
    echo "  Requests:    $NUM_REQUESTS"
    echo "  Concurrency: $CONCURRENCY_LEVELS"
    echo "  Max tokens:  $MAX_TOKENS"
    echo ""

    # Print report for each concurrency level
    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"
    for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
        echo ""
        echo -e "  ${BOLD}Concurrency = $concurrency${NC}"
        echo "  $(printf '=%.0s' $(seq 1 60))"

        printf "  ${BOLD}%-20s %8s %10s %10s %10s %10s %12s %12s${NC}\n" \
            "Backend" "Tok/s" "Avg(ms)" "P50(ms)" "P95(ms)" "P99(ms)" "GPU Load" "GPU Peak"
        printf "  %-20s %8s %10s %10s %10s %10s %12s %12s\n" \
            "--------------------" "--------" "----------" "----------" "----------" "----------" "------------" "------------"

        for backend in inferflux_cuda llama_cpp_cuda; do
            local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"
            if [ ! -f "$stats_file" ]; then
                printf "  %-20s %8s\n" "$backend" "SKIPPED"
                continue
            fi

            python3 -c "
import json
with open('$stats_file') as f:
    s = json.load(f)
print(f\"  {s['backend']:<20} {s['tok_per_sec']:>8} {s['avg_latency_ms']:>10} \"
      f\"{s['p50_latency_ms']:>10} {s['p95_latency_ms']:>10} {s['p99_latency_ms']:>10} \"
      f\"{s['gpu_mem_loaded_mb']:>10} MB {s['gpu_mem_peak_mb']:>10} MB\")
"
        done

        echo ""

        # Speedup ratio for this concurrency
        local llama_file="$OUTPUT_DIR/stats_llama_cpp_cuda_c${concurrency}.json"
        local native_file="$OUTPUT_DIR/stats_inferflux_cuda_c${concurrency}.json"
        if [ -f "$llama_file" ] && [ -f "$native_file" ]; then
            python3 -c "
import json
with open('$llama_file') as f:
    llama = json.load(f)
with open('$native_file') as f:
    native = json.load(f)
if llama['tok_per_sec'] > 0:
    ratio = native['tok_per_sec'] / llama['tok_per_sec']
    color = '\033[0;32m' if ratio >= 0.9 else '\033[1;33m' if ratio >= 0.5 else '\033[0;31m'
    print(f\"  Speedup (native / llama.cpp): {color}{ratio:.2f}x\033[0m (c=$concurrency)\")
    mem_saved = llama['gpu_mem_peak_mb'] - native['gpu_mem_peak_mb']
    if mem_saved > 0:
        print(f\"  GPU memory savings: \033[0;32m{mem_saved} MB\033[0m\")
    elif mem_saved < 0:
        print(f\"  GPU memory overhead: \033[1;33m{-mem_saved} MB\033[0m\")
"
        fi
    done

    echo ""
}

# ============================================================================
# Safetensors → GGUF conversion (uses llama.cpp tools from external/)
# ============================================================================
convert_safetensors_to_gguf() {
    local model_dir=$1
    local quant=${2:-q4_k_m}

    header "Converting safetensors → GGUF ($quant)"

    if [ ! -f "$LLAMA_CONVERT" ]; then
        log_err "convert_hf_to_gguf.py not found at $LLAMA_CONVERT"
        log "  Run: git submodule update --init --recursive"
        return 1
    fi

    local model_name=$(basename "$model_dir")
    local fp16_gguf="$OUTPUT_DIR/${model_name}-f16.gguf"
    local quant_gguf="$OUTPUT_DIR/${model_name}-${quant}.gguf"

    # Step 1: Convert to FP16 GGUF
    if [ -f "$fp16_gguf" ]; then
        log "  FP16 GGUF already exists: $fp16_gguf"
    else
        log "  Converting $model_dir → FP16 GGUF..."
        python3 "$LLAMA_CONVERT" "$model_dir" \
            --outfile "$fp16_gguf" --outtype f16 2>&1 | tail -5

        if [ ! -f "$fp16_gguf" ]; then
            log_err "FP16 conversion failed"
            return 1
        fi
        log_ok "  Created: $fp16_gguf ($(du -h "$fp16_gguf" | cut -f1))"
    fi

    # Step 2: Quantize
    if [ "$quant" = "f16" ]; then
        echo "$fp16_gguf"
        return 0
    fi

    if [ -f "$quant_gguf" ]; then
        log "  Quantized GGUF already exists: $quant_gguf"
    else
        if [ ! -f "$LLAMA_QUANTIZE" ]; then
            log_err "llama-quantize not found at $LLAMA_QUANTIZE"
            log "  Build llama.cpp: cd external/llama.cpp && cmake -B build && cmake --build build -j"
            return 1
        fi

        log "  Quantizing to $quant..."
        "$LLAMA_QUANTIZE" "$fp16_gguf" "$quant_gguf" "$quant" 2>&1 | tail -3

        if [ ! -f "$quant_gguf" ]; then
            log_err "Quantization failed"
            return 1
        fi
        log_ok "  Created: $quant_gguf ($(du -h "$quant_gguf" | cut -f1))"
    fi

    echo "$quant_gguf"
}

# ============================================================================
# Main
# ============================================================================
main() {
    header "GGUF Backend Comparison Benchmark"
    echo "  All measurements are REAL — no simulated or expected values."
    echo "  Scheduler tuning: min_batch_size=$MIN_BATCH_SIZE, batch_accumulation_ms=$BATCH_ACCUMULATION_MS, decode_burst_tokens=$DECODE_BURST_TOKENS, batched_decode=$ENABLE_BATCHED_DECODE"
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        echo "  Single prompt mode: ${INFERFLUX_BENCH_SINGLE_PROMPT}"
    fi
    if [ "$DEBUG_SEQUENCE_SLOTS" != "0" ] || [ "$DEBUG_UNIFIED_ASSEMBLY" != "0" ] || \
       [ "$DEBUG_DECODE_MAPPING" != "0" ] || [ "$DEBUG_OPERATOR_SELECTION" != "0" ] || \
       [ "$DEBUG_LOGITS" != "0" ] || [ "$DEBUG_TOKEN_TRACE" != "0" ]; then
        echo "  Debug knobs: sequence_slots=$DEBUG_SEQUENCE_SLOTS, unified_assembly=$DEBUG_UNIFIED_ASSEMBLY/$DEBUG_UNIFIED_ASSEMBLY_LIMIT, decode_mapping=$DEBUG_DECODE_MAPPING/$DEBUG_DECODE_MAPPING_LIMIT, operator_selection=$DEBUG_OPERATOR_SELECTION/$DEBUG_OPERATOR_SELECTION_LIMIT, logits=$DEBUG_LOGITS/$DEBUG_LOGITS_LIMIT, token_trace=$DEBUG_TOKEN_TRACE/$DEBUG_TOKEN_TRACE_LIMIT"
    fi
    echo ""

    mkdir -p "$OUTPUT_DIR"
    write_prompt_suite_artifacts

    # Handle safetensors → GGUF conversion
    if [ -d "$MODEL_PATH" ] || [[ "$MODEL_PATH" == *.safetensors ]] || [ -n "$QUANTIZE_TO" ]; then
        if [ -d "$MODEL_PATH" ]; then
            mkdir -p "$OUTPUT_DIR"
            local quant="${QUANTIZE_TO:-q4_k_m}"
            local converted
            converted=$(convert_safetensors_to_gguf "$MODEL_PATH" "$quant")
            if [ $? -ne 0 ] || [ -z "$converted" ]; then
                log_err "Conversion failed"
                exit 1
            fi
            MODEL_PATH="$converted"
            log "  Using converted model: $MODEL_PATH"
        else
            log_err "MODEL_PATH is not a directory for safetensors conversion: $MODEL_PATH"
            exit 1
        fi
    fi

    # Validate
    if [ ! -f "$MODEL_PATH" ]; then
        log_err "Model not found: $MODEL_PATH"
        echo "Set MODEL_PATH=path/to/model.gguf"
        echo "Or point to safetensors dir: MODEL_PATH=models/my-model-safetensors QUANTIZE_TO=q4_k_m"
        exit 1
    fi

    # Auto-build if binary is missing or stale (older than any source file).
    local needs_build=false
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        needs_build=true
    else
        # Check if any source file is newer than the binary.
        local newest_src
        newest_src=$(find runtime server scheduler model cli net io policy \
            -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' 2>/dev/null \
            | xargs stat --format='%Y' 2>/dev/null | sort -rn | head -1)
        local bin_mtime
        bin_mtime=$(stat --format='%Y' "$BUILD_DIR/inferfluxd" 2>/dev/null || echo 0)
        if [ -n "$newest_src" ] && [ "$newest_src" -gt "$bin_mtime" ]; then
            needs_build=true
        fi
    fi
    if $needs_build; then
        local build_log="$OUTPUT_DIR/build_$(date +%Y%m%d_%H%M%S).log"
        log "Building $BUILD_DIR/inferfluxd (CUDA enabled)..."
        cmake -S . -B "$BUILD_DIR" \
            -DENABLE_CUDA=ON \
            -DINFERFLUX_CUDA_NATIVE_ARCH=ON >/dev/null 2>&1 || {
            log_err "cmake configure failed"; exit 1; }
        if ! cmake --build "$BUILD_DIR" -j"$(nproc)" >"$build_log" 2>&1; then
            log_err "cmake build failed; last 50 lines:"
            tail -50 "$build_log"
            exit 1
        fi
        local warning_count=0
        warning_count=$(grep -Ec '(^|[^[:alnum:]_])(warning|error):' "$build_log" || true)
        if [ "$warning_count" -gt 0 ]; then
            log_warn "Build emitted ${warning_count} compiler diagnostics; see $build_log"
            grep -E 'warning:|error:' "$build_log" | tail -10 || true
        fi
        log "Build complete: $BUILD_DIR/inferfluxd"
    fi
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        log_err "inferfluxd not found at $BUILD_DIR/inferfluxd after build attempt"
        exit 1
    fi

    if ! nvidia-smi >/dev/null 2>&1; then
        log_err "nvidia-smi not found — CUDA GPU required"
        exit 1
    fi

    local mem_baseline=$(gpu_mem_mb)
    log "GPU baseline memory: ${mem_baseline} MB"
    log "GPU: $(gpu_name), $(gpu_mem_total_mb) MB"
    log "Concurrency levels: $CONCURRENCY_LEVELS"
    log "Requests: $NUM_REQUESTS across ${#PROMPTS[@]} unique prompts"
    log "Note: Each backend is benchmarked separately to ensure clean state"

    # Convert comma-separated concurrency levels to array
    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"

    # Benchmark each backend at each concurrency level
    local failed_backends=()
    for backend in inferflux_cuda llama_cpp_cuda; do
        local port=$PORT_LLAMA
        [ "$backend" = "inferflux_cuda" ] && port=$PORT_NATIVE

        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            echo ""
            header "Benchmarking: $backend @ concurrency=$concurrency"

            reset_backend_artifacts "$backend" "$concurrency"

            if ! start_server "$backend" "$port"; then
                log_err "Failed to start $backend — skipping"
                failed_backends+=("$backend")
                continue
            fi

            # Record memory before workload
            local mem_before=$(gpu_mem_mb)
            log "GPU memory before workload: ${mem_before} MB"

            if ! run_benchmark "$backend" "$port" "$concurrency"; then
                failed_backends+=("$backend")
            fi

            if ! stop_server "$backend"; then
                failed_backends+=("$backend")
            fi
            if [ "$backend" = "inferflux_cuda" ]; then
                reset_cuda_device
            fi

            # Verify memory was freed
            local mem_after=$(gpu_mem_mb)
            local mem_freed=$((mem_before - mem_after))
            log "GPU memory after workload: ${mem_after} MB (freed: ${mem_freed} MB)"

            if [ $mem_freed -gt 100 ]; then
                log_ok "Memory cleanup verified for $backend @ c=$concurrency"
            elif [ $mem_after -gt $((mem_baseline + 500)) ]; then
                log_warn "Memory not fully freed for $backend @ c=$concurrency (${mem_after} MB vs baseline ${mem_baseline} MB)"
            fi

            # Let GPU memory settle
            sleep 2
        done
    done

    # Compare
    echo ""
    compare_responses
    echo ""
    print_report

    # Save combined JSON (includes all concurrency levels)
    python3 -c "
import json, os, glob
combined = {'timestamp': '$(date -Iseconds)', 'model': '$MODEL_PATH',
            'gpu': '$(gpu_name)', 'num_requests': $NUM_REQUESTS,
            'max_tokens': $MAX_TOKENS, 'concurrency_levels': '$CONCURRENCY_LEVELS',
            'backends': {}, 'similarity': {}}
for backend in ['llama_cpp_cuda', 'inferflux_cuda']:
    combined['backends'][backend] = {}
    # Load stats for each concurrency level
    for concurrency in [int(x) for x in '$CONCURRENCY_LEVELS'.split(',')]:
        sf = '$OUTPUT_DIR/stats_' + backend + '_c' + str(concurrency) + '.json'
        if os.path.exists(sf):
            with open(sf) as f:
                combined['backends'][backend][concurrency] = json.load(f)
for concurrency in [int(x) for x in '$CONCURRENCY_LEVELS'.split(',')]:
    sf = '$OUTPUT_DIR/similarity_c' + str(concurrency) + '.json'
    if os.path.exists(sf):
        with open(sf) as f:
            combined['similarity'][concurrency] = json.load(f)
outf = '$OUTPUT_DIR/comparison_$(date +%Y%m%d_%H%M%S).json'
with open(outf, 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Results saved to: {outf}')
    "

    if [ ${#failed_backends[@]} -gt 0 ]; then
        log_err "Benchmark incomplete: failed backends: ${failed_backends[*]}"
        return 1
    fi

    log_ok "Benchmark complete!"
}

# Cleanup on exit
cleanup() {
    log "Running cleanup..."
    local mem_before=$(gpu_mem_mb 2>/dev/null || echo "0")
    stop_server llama_cpp_cuda 2>/dev/null || true
    stop_server inferflux_cuda 2>/dev/null || true
    sleep 2
    local mem_after=$(gpu_mem_mb 2>/dev/null || echo "0")
    local mem_freed=$((mem_before - mem_after))
    if [ $mem_before -gt 0 ] && [ $mem_after -ge 0 ]; then
        log "Cleanup: GPU memory ${mem_before} MB → ${mem_after} MB (freed: ${mem_freed} MB)"
    fi
}
trap cleanup EXIT

main "$@"
