#!/usr/bin/env python3

from pathlib import Path


def test_benchmark_harness_uses_batched_decode_and_accuracy_safe_defaults() -> None:
    script = Path("scripts/run_gguf_comparison_benchmark.sh").read_text()
    assert (
        'ENABLE_BATCHED_DECODE="${INFERFLUX_BENCH_ENABLE_BATCHED_DECODE:-1}"'
        in script
    )
    assert (
        'DECODE_BURST_TOKENS="${INFERFLUX_BENCH_DECODE_BURST_TOKENS:-0}"'
        in script
    )
    assert (
        "experimental for native accuracy; use 0 or 1 for parity-sensitive runs"
        in script
    )


if __name__ == "__main__":
    test_benchmark_harness_uses_batched_decode_and_accuracy_safe_defaults()
    print("benchmark harness defaults OK")
