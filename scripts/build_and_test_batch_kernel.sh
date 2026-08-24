#!/bin/bash
#
# Build and run batch GEMV kernel benchmark
#
# Usage: ./scripts/build_and_test_batch_kernel.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }

# Configuration
KERNEL_FILE="tests/unit/benchmark_batch_gemv.cu"
BINARY="tests/unit/benchmark_batch_gemv"
KERNELS_DIR="runtime/backends/cuda/native/kernels"

log "Building batch GEMV kernel benchmark..."

# Check for CUDA compiler
if ! command -v nvcc &> /dev/null; then
    log_err "nvcc not found. Please ensure CUDA Toolkit is installed and in PATH."
    exit 1
fi

# Get CUDA architecture
CUDA_ARCH=${CUDA_ARCH:-"native"}
if [ "$CUDA_ARCH" = "native" ]; then
    CUDA_ARCH=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/\..*//')
    case $CUDA_ARCH in
        12) CUDA_ARCH="sm_89";;   # Ada (RTX 4000)
        11) CUDA_ARCH="sm_86";;   # Ampere (A100)
        10) CUDA_ARCH="sm_80";;   # Ampere (A100)
        9)  CUDA_ARCH="sm_75";;   # Turing
        *)   CUDA_ARCH="sm_75";;   # Default
    esac
fi

log "CUDA architecture: $CUDA_ARCH"

# Build
log "Compiling $KERNEL_FILE..."
nvcc -O3 \
    -arch=$CUDA_ARCH \
    --std=c++17 \
    -I"$KERNELS_DIR" \
    -o "$BINARY" \
    "$KERNEL_FILE" \
    2>&1 | tee build_log.txt

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_err "Compilation failed. Check build_log.txt for details."
    exit 1
fi

log_ok "Build successful: $BINARY"

# Run benchmark
log ""
log "Running benchmark..."
log "========================================"

if ./"$BINARY"; then
    log_ok "Benchmark completed successfully"
else
    log_err "Benchmark failed"
    exit 1
fi

log ""
log "Binary location: $BINARY"
log "To run manually: ./$BINARY"
