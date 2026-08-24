#!/bin/bash
#
# Validate Concurrent Throughput Fix
#
# This script runs the multi-backend benchmark before and after the fix
# to measure the improvement in concurrent throughput.
#
# Usage:
#   ./scripts/validate_concurrent_fix.sh --model models/qwen2.5-3b-instruct-q4_k_m.gguf

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# Parse arguments
MODEL="${1:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,4,8}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"

resolve_build_dir() {
    if [ -n "$BUILD_DIR" ]; then
        echo "$BUILD_DIR"
        return 0
    fi

    if [ -f "./build/inferfluxd" ]; then
        echo "./build"
        return 0
    fi

    if [ -f "./build-cuda/inferfluxd" ]; then
        echo "./build-cuda"
        return 0
    fi

    echo "./build"
}

header "Validating Concurrent Throughput Fix"
BUILD_DIR="$(resolve_build_dir)"
echo "Model: $MODEL"
echo "Build dir: $BUILD_DIR"
echo "Concurrency levels: $CONCURRENCY_LEVELS"
echo "Requests per level: $NUM_REQUESTS"
echo ""

# Build if needed
if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
    log "Building inferfluxd..."
    cmake -S . -B "$BUILD_DIR" -DENABLE_CUDA=ON
    cmake --build "$BUILD_DIR" -j $(nproc)
fi

# Create output directory
OUTPUT_DIR="./concurrent_fix_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Test cuda_native backend
header "Benchmarking cuda_native Backend"
log "Testing at concurrency levels: $CONCURRENCY_LEVELS"

CONCURRENCY_LEVELS="$CONCURRENCY_LEVELS" \
    NUM_REQUESTS="$NUM_REQUESTS" \
    MAX_TOKENS="$MAX_TOKENS" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    SKIP_OLLAMA=true \
    ./scripts/benchmark_multi_backend_comparison.sh "$MODEL" || {
    log_err "Benchmark failed"
    exit 1
}

# Parse results
header "Analyzing Results"

python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json, os, sys

output_dir = sys.argv[1]
concurrency_levels = [int(c) for c in sys.argv[2].split(',')]

# Load results
results = {}
for c in concurrency_levels:
    stats_file = os.path.join(output_dir, f"stats_cuda_native_c{c}.json")
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            results[c] = json.load(f)

if not results:
    print("No results found!")
    sys.exit(1)

print()
print(f"{'Concurrency':<15} {'Tok/s':>12} {'Speedup':>12} {'GPU Mem (MB)':>15}")
print("-" * 60)

baseline = results.get(1, {})
baseline_tok_per_sec = baseline.get('tok_per_sec', 0)

for c in sorted(results.keys()):
    r = results[c]
    tok_per_sec = r['tok_per_sec']
    if baseline_tok_per_sec > 0:
        speedup = tok_per_sec / baseline_tok_per_sec
        print(f"{c:<15} {tok_per_sec:>12.1f} {speedup:>12.2f}x {r['gpu_mem_peak_mb']:>15.0f}")
    else:
        print(f"{c:<15} {tok_per_sec:>12.1f} {'N/A':>12} {r['gpu_mem_peak_mb']:>15.0f}")

print()

# Check scaling efficiency
if len(results) >= 2:
    c1_tps = results.get(1, {}).get('tok_per_sec', 0)
    c4_tps = results.get(4, {}).get('tok_per_sec', 0)
    c8_tps = results.get(8, {}).get('tok_per_sec', 0)

    if c1_tps > 0 and c4_tps > 0:
        eff_c4 = (c4_tps / c1_tps) / 4.0
        print(f"Scaling efficiency @ c=4: {eff_c4:.1%}")
        if eff_c4 >= 0.70:
            print(f"  {GREEN}✓ PASS: Good concurrent scaling{NC}")
        else:
            print(f"  {YELLOW}⚠ WARN: Suboptimal scaling{NC}")

    if c1_tps > 0 and c8_tps > 0:
        eff_c8 = (c8_tps / c1_tps) / 8.0
        print(f"Scaling efficiency @ c=8: {eff_c8:.1%}")
        if eff_c8 >= 0.50:
            print(f"  {GREEN}✓ PASS: Good concurrent scaling{NC}")
        else:
            print(f"  {YELLOW}⚠ WARN: Suboptimal scaling{NC}")

print()

# Compare with expected values
# Before fix: c=4 → ~0.50x scaling (87.6 / 72.8 / 4 = 0.30)
# After fix: c=4 → ~0.70x+ scaling
if c1_tps > 0 and c4_tps > 0:
    scaling_ratio = c4_tps / c1_tps
    print(f"Throughput ratio (c=4 / c=1): {scaling_ratio:.2f}x")
    if scaling_ratio >= 2.5:
        print(f"  {GREEN}✓ SUCCESS: Fix appears effective!{NC}")
        print(f"     Expected: 2.5x+ with max_batch_size=16")
    elif scaling_ratio >= 1.8:
        print(f"  {YELLOW}⚠ MODERATE: Some improvement, but not optimal{NC}")
    else:
        print(f"  {RED}✗ FAIL: No significant improvement detected{NC}")

PYEOF

echo ""
header "Validation Complete"
log "Results saved to: $OUTPUT_DIR"
