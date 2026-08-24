#!/usr/bin/env python3
"""Summarize InferFlux CUDA phase timing lines from a server log."""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^\[phase_timing\]\s+#(?P<idx>\d+)\s+L=(?P<layers>\d+)\s+tokens=(?P<tokens>\d+)\s+"
    r"embed=(?P<embed>[0-9.]+)\s+qkv=(?P<qkv>[0-9.]+)\s+rope=(?P<rope>[0-9.]+)\s+"
    r"kv=(?P<kv>[0-9.]+)\s+attn=(?P<attn>[0-9.]+)\s+o_proj=(?P<o_proj>[0-9.]+)\s+"
    r"(?:(?:ffn_proj=(?P<ffn_proj>[0-9.]+)\s+ffn_silu=(?P<ffn_silu>[0-9.]+)\s+"
    r"ffn_down=(?P<ffn_down>[0-9.]+)\s+)?)"
    r"ffn=(?P<ffn>[0-9.]+)\s+lm_head=(?P<lm_head>[0-9.]+)\s+total=(?P<total>[0-9.]+)\s+ms$"
)

PHASES = [
    "embed",
    "qkv",
    "rope",
    "kv",
    "attn",
    "o_proj",
    "ffn_proj",
    "ffn_silu",
    "ffn_down",
    "ffn",
    "lm_head",
    "total",
]


def parse_lines(text: str):
    rows = []
    for raw in text.splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        row = {
            "index": int(match.group("idx")),
            "layers": int(match.group("layers")),
            "tokens": int(match.group("tokens")),
        }
        for phase in PHASES:
            value = match.group(phase)
            row[phase] = float(value) if value is not None else 0.0
        row["kind"] = "decode" if row["tokens"] == 1 else "prefill"
        rows.append(row)
    return rows


def summarize(rows):
    def phase_summary(subset):
        if not subset:
            return {"count": 0}
        out = {"count": len(subset)}
        for phase in PHASES:
            values = [row[phase] for row in subset]
            out[f"{phase}_mean_ms"] = round(statistics.mean(values), 3)
            out[f"{phase}_max_ms"] = round(max(values), 3)
        return out

    return {
        "all": phase_summary(rows),
        "decode": phase_summary([r for r in rows if r["kind"] == "decode"]),
        "prefill": phase_summary([r for r in rows if r["kind"] == "prefill"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="Path to native server log file")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
    rows = parse_lines(text)
    if not rows:
        print("No phase timing lines found", file=sys.stderr)
        return 1

    summary = summarize(rows)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print("Native phase timing summary")
    for label in ["decode", "prefill", "all"]:
        block = summary[label]
        if block["count"] == 0:
            print(f"  {label}: no samples")
            continue
        print(
            f"  {label}: count={block['count']} total_mean={block['total_mean_ms']:.3f} ms "
            f"attn_mean={block['attn_mean_ms']:.3f} qkv_mean={block['qkv_mean_ms']:.3f} "
            f"ffn_mean={block['ffn_mean_ms']:.3f} "
            f"(proj={block['ffn_proj_mean_ms']:.3f} silu={block['ffn_silu_mean_ms']:.3f} "
            f"down={block['ffn_down_mean_ms']:.3f}) "
            f"lm_head_mean={block['lm_head_mean_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
