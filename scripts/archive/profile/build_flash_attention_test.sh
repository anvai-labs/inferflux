#!/bin/bash
# Build script for FlashAttention test program

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Building FlashAttention test program..."

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Check if CUDA runtime is available
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found in PATH"
    echo "CUDA toolkit may not be properly installed"
    echo ""
fi

# Compile the test program
# Note: This version only needs CUDA runtime, not llama.cpp internals
g++ -std=c++17 \
    -o "$BUILD_DIR/test_flash_attention" \
    "$PROJECT_ROOT/runtime/backends/cuda/kernels/test_flash_attention.cpp" \
    -lcudart

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "  Binary: $BUILD_DIR/test_flash_attention"
    echo ""
    echo "Run with:"
    echo "  $BUILD_DIR/test_flash_attention"
    echo ""
    echo "Or simply:"
    echo "  ./build/test_flash_attention"
    echo ""
else
    echo "✗ Build failed!"
    echo ""
    echo "If you get linker errors, make sure CUDA toolkit is installed:"
    echo "  nvcc --version"
    echo ""
    echo "And CUDA runtime libraries are available:"
    echo "  ldconfig -p | grep libcuda"
    exit 1
fi
