#!/bin/bash
# Verify OOM Fix Deployment
# Tests code changes without requiring full CUDA build

set -e

echo "========================================"
echo "Deployment Verification"
echo "========================================"
echo ""

ERRORS=0

# Test 1: Check git commit
echo "Test 1: Verify git commit"
COMMIT=$(git log --oneline -1 | grep "Implement FP16 OOM handling")
if [ -n "$COMMIT" ]; then
    echo "  ✅ Commit exists in local repository"
else
    echo "  ❌ Commit not found"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: Check remote repository
echo ""
echo "Test 2: Verify remote repository"
REMOTE_COMMIT=$(git ls-remote origin main | grep "707138b")
if [ -n "$REMOTE_COMMIT" ]; then
    echo "  ✅ Commit pushed to remote repository"
else
    echo "  ❌ Commit not found in remote"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: Check code changes present
echo ""
echo "Test 3: Verify code changes present"

# Check slot manager methods
if grep -q "bool CanAcceptRequest() const" runtime/scheduler/sequence_slot_manager.h; then
    echo "  ✅ CanAcceptRequest() method signature present"
else
    echo "  ❌ CanAcceptRequest() method signature missing"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "bool SequenceSlotManager::CanAcceptRequest() const" runtime/scheduler/sequence_slot_manager.cpp; then
    echo "  ✅ CanAcceptRequest() implementation present"
else
    echo "  ❌ CanAcceptRequest() implementation missing"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "int GetMemoryPressure() const" runtime/scheduler/sequence_slot_manager.h; then
    echo "  ✅ GetMemoryPressure() method signature present"
else
    echo "  ❌ GetMemoryPressure() method signature missing"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "bool PerformGracefulDegradation()" runtime/scheduler/sequence_slot_manager.h; then
    echo "  ✅ PerformGracefulDegradation() method signature present"
else
    echo "  ❌ PerformGracefulDegradation() method signature missing"
    ERRORS=$((ERRORS + 1))
fi

# Check main.cpp changes
if grep -q "am.quantization = inferflux::DetectQuantization" server/main.cpp; then
    echo "  ✅ Quantization detection fix present"
else
    echo "  ❌ Quantization detection fix missing"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "INFERFLUX_MODEL_PATH is set, overriding config file model path" server/main.cpp; then
    echo "  ✅ Config override fix present"
else
    echo "  ❌ Config override fix missing"
    ERRORS=$((ERRORS + 1))
fi

# Check startup_advisor.cpp changes
if grep -q "activation_multiplier = 1.5" server/startup_advisor.cpp; then
    echo "  ✅ FP16 activation multiplier (1.5x) present"
else
    echo "  ❌ FP16 activation multiplier missing"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Syntax validation
echo ""
echo "Test 4: Verify code compiles"

if g++ -std=c++17 -c -I. -Iruntime -Iserver runtime/scheduler/sequence_slot_manager.cpp -o /tmp/test1.o 2>/dev/null; then
    echo "  ✅ sequence_slot_manager.cpp compiles"
else
    echo "  ❌ sequence_slot_manager.cpp has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

if g++ -std=c++17 -c -I. -Iruntime -Iserver -Iexternal/nlohmann/include server/startup_advisor.cpp -o /tmp/test2.o 2>/dev/null; then
    echo "  ✅ startup_advisor.cpp compiles"
else
    echo "  ❌ startup_advisor.cpp has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

# Test 5: Documentation
echo ""
echo "Test 5: Verify documentation"

DOCS=(
    "docs/OOM_ROOT_CAUSE_ANALYSIS.md"
    "docs/FP16_OOM_FIX_VALIDATION.md"
    "docs/FP16_BENCHMARK_RESULTS_FINAL.md"
    "docs/FP16_STATUS.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✅ $doc exists"
    else
        echo "  ❌ $doc missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 6: Scripts
echo ""
echo "Test 6: Verify benchmark scripts"

if [ -x "scripts/test_fp16_concurrent_oom_fix.sh" ]; then
    echo "  ✅ OOM test script exists and executable"
else
    echo "  ❌ OOM test script missing or not executable"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"

if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Deployment verified successfully:"
    echo "  - Code changes present"
    echo "  - Syntax valid"
    echo "  - Documentation complete"
    echo "  - Pushed to remote repository"
    echo ""
    echo "Commit: 707138b"
    echo "Branch: main"
    exit 0
else
    echo "❌ $ERRORS check(s) failed"
    echo ""
    echo "Please review the errors above."
    exit 1
fi
