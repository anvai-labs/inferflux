#!/usr/bin/env python3
"""Summarize INFERFLUX_CUDA_DISPATCH_TRACE output from an inferfluxd log.

Reads dispatch_trace lines (one per projection dispatch) and reports:
  - divergence table: (layer, selected, actual, reason) -> count
  - tier distribution per layer and M bucket
  - selected-operator histogram per layer

Exit code 1 when any divergence is present, so it can gate benchmarks.

Usage:
  parse_dispatch_trace.py <server.log> [--json]
"""
import argparse
import json
import re
import sys
from collections import Counter

TRACE_RE = re.compile(r"\[dispatch_trace\] \[(\w+)\]: (.*)")


def m_bucket(m: int) -> str:
    if m <= 1:
        return "1"
    if m == 2:
        return "2"
    if m <= 4:
        return "3_4"
    if m <= 8:
        return "5_8"
    if m <= 16:
        return "9_16"
    if m <= 32:
        return "17_32"
    return "33_plus"


def parse_line(kv_text: str) -> dict:
    out = {}
    for token in kv_text.split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="server log path, or - for stdin")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")
    args = parser.parse_args()

    stream = sys.stdin if args.log == "-" else open(args.log, "r",
                                                    errors="replace")

    divergences = Counter()
    tier_dist = Counter()
    selected_hist = Counter()
    total = 0

    for line in stream:
        m = TRACE_RE.search(line)
        if not m:
            continue
        layer = m.group(1)
        rec = parse_line(m.group(2))
        total += 1

        selected = rec.get("selected", "unknown")
        actual = rec.get("actual", "unknown")
        tier = rec.get("tier", "unknown")
        reason = rec.get("reason", "")

        if reason:
            divergences[(layer, selected, actual, reason)] += 1
        try:
            bucket = m_bucket(int(rec.get("M", "0")))
        except ValueError:
            bucket = "unknown"
        tier_dist[(layer, bucket, tier)] += 1
        selected_hist[(layer, selected)] += 1

    if args.json:
        print(json.dumps({
            "total": total,
            "divergences": [
                {"layer": k[0], "selected": k[1], "actual": k[2],
                 "reason": k[3], "count": v}
                for k, v in sorted(divergences.items())
            ],
            "tier_distribution": [
                {"layer": k[0], "m_bucket": k[1], "tier": k[2], "count": v}
                for k, v in sorted(tier_dist.items())
            ],
            "selected_histogram": [
                {"layer": k[0], "selected": k[1], "count": v}
                for k, v in sorted(selected_hist.items())
            ],
        }, indent=2))
    else:
        print(f"dispatch records: {total}")
        print()
        if divergences:
            print("DIVERGENCES (selected != executed):")
            for (layer, sel, act, reason), count in sorted(
                    divergences.items()):
                print(f"  {layer}: {sel} -> {act} ({reason}) x{count}")
        else:
            print("divergences: none")
        print()
        print("tier distribution:")
        for (layer, bucket, tier), count in sorted(tier_dist.items()):
            print(f"  {layer:10s} m={bucket:8s} {tier:8s} {count}")
        print()
        print("selected operators:")
        for (layer, sel), count in sorted(selected_hist.items()):
            print(f"  {layer:10s} {sel:32s} {count}")

    return 1 if divergences else 0


if __name__ == "__main__":
    sys.exit(main())
