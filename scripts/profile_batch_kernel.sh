#!/bin/bash
#
# Profile batch GEMV kernel with Nsight Systems
#
# Profiles both baseline and batch kernels to compare:
# - Kernel execution time
# - Memory bandwidth utilization
# - GPU occupancy
#
# Usage: ./scripts/profile_batch_kernel.sh

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
header()  { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# Configuration
KERNEL_FILE="tests/unit/benchmark_batch_gemv.cu"
BINARY="tests/unit/benchmark_batch_gemv"
OUTPUT_DIR="./batch_kernel_profile_$(date +%Y%m%d_%H%M%S)"
KERNELS_DIR="runtime/backends/cuda/native/kernels"

# Check prerequisites
header "Prerequisites Check"

if [ ! -f "$KERNEL_FILE" ]; then
    log_err "Kernel file not found: $KERNEL_FILE"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    log_err "nvcc not found. Install CUDA Toolkit."
    exit 1
fi

if ! command -v nsys &> /dev/null; then
    log_err "nsys not found. Install Nsight Systems."
    exit 1
fi

log_ok "Prerequisites validated"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get CUDA architecture
CUDA_ARCH=${CUDA_ARCH:-"native"}
if [ "$CUDA_ARCH" = "native" ]; then
    CUDA_ARCH=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/\..*//')
    case $CUDA_ARCH in
        12) CUDA_ARCH="sm_89";;
        11) CUDA_ARCH="sm_86";;
        *)   CUDA_ARCH="sm_75";;
    esac
fi

# Build profiling binary
header "Building Profiling Binary"

log "CUDA architecture: $CUDA_ARCH"
log "Output directory: $OUTPUT_DIR"

nvcc -O3 -lineinfo \
    -arch=$CUDA_ARCH \
    --std=c++17 \
    -I"$KERNELS_DIR" \
    -o "$OUTPUT_DIR/benchmark_batch_gemv" \
    "$KERNEL_FILE"

log_ok "Build complete: $OUTPUT_DIR/benchmark_batch_gemv"

# Profile baseline kernel
header "Profiling Baseline Kernel"

log "Capturing baseline kernel execution..."

nsys profile -o "$OUTPUT_DIR/profile_baseline.nsys-rep" \
    --trace=cuda,nvtx,osrt,cudnn \
    --force-overwrite=true \
    --cuda-memory-usage=true \
    "$OUTPUT_DIR/benchmark_batch_gemv" 2>&1 | tee "$OUTPUT_DIR/baseline_output.txt"

# Profile batch kernels
header "Profiling Batch Kernels"

for batch_size in 1 2 4 8; do
    log "Capturing batch kernel (BatchSize=$batch_size)..."

    # Create a wrapper script that sets environment and runs specific batch size
    cat > "$OUTPUT_DIR/run_batch${batch_size}.sh" <<EOF
#!/bin/bash
cd "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0
echo "Running batch_size=$batch_size"
./benchmark_batch_gemv
EOF
    chmod +x "$OUTPUT_DIR/run_batch${batch_size}.sh"

    nsys profile -o "$OUTPUT_DIR/profile_batch${batch_size}.nsys-rep" \
        --trace=cuda,nvtx,osrt,cudnn \
        --force-overwrite=true \
        --cuda-memory-usage=true \
        "$OUTPUT_DIR/run_batch${batch_size}.sh" 2>&1 | tee "$OUTPUT_DIR/batch${batch_size}_output.txt"
done

# Generate analysis
header "Analysis Complete"

log "Generating comparative analysis..."

# Extract key metrics from profiles
cat > "$OUTPUT_DIR/ANALYSIS.md" <<EOF
# Batch GEMV Kernel Profiling Analysis

**Date**: $(date)
**Profile Directory**: $OUTPUT_DIR

## Files Generated

- profile_baseline.nsys-rep - Baseline kernel profile
- profile_batch1.nsys-rep - Batch kernel (BatchSize=1) profile
- profile_batch2.nsys-rep - Batch kernel (BatchSize=2) profile
- profile_batch4.nsys-rep - Batch kernel (BatchSize=4) profile
- profile_batch8.nsys-rep - Batch kernel (BatchSize=8) profile

## Analysis Commands

### View in Nsight Systems GUI
\`\`\`bash
nsys-ui $OUTPUT_DIR/profile_baseline.nsys-rep
nsys-ui $OUTPUT_DIR/profile_batch2.nsys-rep
nsys-ui $OUTPUT_DIR/profile_batch4.nsys-rep
nsys-ui $OUTPUT_DIR/profile_batch8.nsys-rep
\`\`\`

### Export to CSV for analysis
\`\`\`bash
nsys export --type csv --output $OUTPUT_DIR/baseline.csv $OUTPUT_DIR/profile_baseline.nsys-rep
nsys export --type csv --output $OUTPUT_DIR/batch2.csv $OUTPUT_DIR/profile_batch2.nsys-rep
\`\`\`

### Compare kernel execution times
\`\`\`bash
# View summary statistics
nsys stats $OUTPUT_DIR/profile_baseline.nsys-rep | grep -A 20 "CUDA Kernel"
nsys stats $OUTPUT_DIR/profile_batch2.nsys-rep | grep -A 20 "CUDA Kernel"
\`\`\`

## Key Metrics to Compare

1. **GPU Time**: Total GPU time for kernel execution
2. **Memory Bandwidth**: Bytes transferred per second
3. **Kernel Launch Overhead**: Time spent launching kernels
4. **Occupancy**: GPU SM utilization percentage
5. **Memory Efficiency**: Cache hit rates, bandwidth utilization

## Expected Findings

If batch processing is effective:
- **Fewer kernel launches**: Batch kernel should show reduced launch count
- **Better memory bandwidth**: Bytes/second should improve for batched workloads
- **Lower latency per sequence**: Amortized cost across batch

If batch processing is NOT effective:
- **No improvement**: Batch kernel matches or exceeds baseline time
- **Memory contention**: Increased memory bandwidth pressure
- **Kernel complexity**: Longer kernel execution times

## Next Steps

1. Review profiles in nsys-ui to identify bottlenecks
2. Compare metrics across batch sizes
3. Make go/no-go decision for full implementation

EOF

log_ok "Analysis saved to: $OUTPUT_DIR/ANALYSIS.md"

# Summary
header "Profiling Complete"

log "Output directory: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "  (No profile files found)"
echo ""
echo "To view profiles:"
echo "  nsys-ui $OUTPUT_DIR/profile_baseline.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/profile_batch2.nsys-rep"
echo ""
echo "Analysis document:"
echo "  cat $OUTPUT_DIR/ANALYSIS.md"

log "To re-run this profiling:"
echo "  ./scripts/profile_batch_kernel.sh"
