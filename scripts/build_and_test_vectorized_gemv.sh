#!/bin/bash
#
# Build and test vectorized GEMV kernel
#

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
header()  { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' ${#1})"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_DIR="runtime/backends/cuda/native/kernels"
TEST_FILE="tests/unit/test_vectorized_gemv.cu"
OUTPUT_BIN="test_vectorized_gemv"

header "Vectorized GEMV Kernel Test Build"

# Check prerequisites
if ! command -v nvcc &> /dev/null; then
    log_err "nvcc not found. Install CUDA Toolkit."
    exit 1
fi

# Auto-detect CUDA architecture
CUDA_ARCH=${CUDA_ARCH:-"native"}
if [ "$CUDA_ARCH" = "native" ]; then
    CUDA_ARCH=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/\..*//')
    case $CUDA_ARCH in
        12) CUDA_ARCH="sm_89";;
        11) CUDA_ARCH="sm_86";;
        *)   CUDA_ARCH="sm_75";;
    esac
fi

log "CUDA architecture: $CUDA_ARCH"
log "Test file: $TEST_FILE"
log "Include path: $KERNELS_DIR"

# Build
header "Building Test"

nvcc -O3 -lineinfo \
    -arch=$CUDA_ARCH \
    --std=c++17 \
    -I"$KERNELS_DIR" \
    -o "$OUTPUT_BIN" \
    "$TEST_FILE"

if [ $? -eq 0 ]; then
    log_ok "Build successful: $OUTPUT_BIN"
else
    log_err "Build failed"
    exit 1
fi

# Run test
header "Running Test"

./"$OUTPUT_BIN"
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    log_ok "Test PASSED"
    header "Next Steps"
    echo "1. Profile baseline vs vectorized with Nsight Compute"
    echo "2. Measure memory bandwidth improvement"
    echo "3. If successful, integrate into dispatch table"
else
    log_err "Test FAILED"
    exit 1
fi
