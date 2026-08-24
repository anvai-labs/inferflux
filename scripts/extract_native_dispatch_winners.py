#!/usr/bin/env python3
"""Extract native FFN/down-proj dispatch winners by workload bucket."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

FFN_RE = re.compile(
    r'^inferflux_cuda_ffn_proj_geometry_total\{phase="([^"]+)",operator="([^"]+)",'
    r'quant="([^"]+)",m_bucket="([^"]+)",n="([^"]+)",n_bucket="([^"]+)",'
    r'k="([^"]+)",k_bucket="([^"]+)",grouped_outputs="([^"]+)"\}\s+(\d+)$'
)
DOWN_RE = re.compile(
    r'^inferflux_cuda_down_proj_geometry_total\{phase="([^"]+)",operator="([^"]+)",'
    r'quant="([^"]+)",m_bucket="([^"]+)",n="([^"]+)",n_bucket="([^"]+)",'
    r'k="([^"]+)",k_bucket="([^"]+)"\}\s+(\d+)$'
)
BUCKET_ORDER = {
    "1": 0,
    "2": 1,
    "3_4": 2,
    "5_8": 3,
    "9_16": 4,
    "17_32": 5,
    "33_64": 6,
    "65_128": 7,
    "129_plus": 8,
    "17_plus": 9,
}
PHASE_ORDER = {"decode": 0, "prefill": 1, "unknown": 2}


def aggregate(metrics_path: Path):
    winners = {
        "ffn": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        "down_proj": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
    }

    for raw in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = FFN_RE.match(line)
        if match:
            phase, operator, _quant, m_bucket, *_rest, count = match.groups()
            winners["ffn"][phase][m_bucket][operator] += int(count)
            continue

        match = DOWN_RE.match(line)
        if match:
            phase, operator, _quant, m_bucket, *_rest, count = match.groups()
            winners["down_proj"][phase][m_bucket][operator] += int(count)

    return winners


def normalize(aggregated):
    output = {"ffn": {}, "down_proj": {}}
    for kind, phases in aggregated.items():
        for phase, buckets in sorted(
            phases.items(), key=lambda item: PHASE_ORDER.get(item[0], 99)
        ):
            phase_out = {}
            for bucket, operators in sorted(
                buckets.items(), key=lambda item: BUCKET_ORDER.get(item[0], 99)
            ):
                sorted_ops = sorted(operators.items(), key=lambda item: (-item[1], item[0]))
                total = sum(count for _, count in sorted_ops)
                if total <= 0:
                    continue
                top_count = sorted_ops[0][1]
                op_entries = [
                    {
                        "operator": op,
                        "count": count,
                        "share": round(count / total, 4),
                    }
                    for op, count in sorted_ops
                ]
                tied_winners = [entry for entry in op_entries if entry["count"] == top_count]
                phase_out[bucket] = {
                    "total": total,
                    "winner": op_entries[0],
                    "is_tie": len(tied_winners) > 1,
                    "tied_winners": tied_winners,
                    "operators": op_entries,
                }
            if phase_out:
                output[kind][phase] = phase_out
    return output


def merge_stats(stats_path: Path, summary):
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["inferflux_cuda_bucket_winners"] = summary
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def print_summary(summary):
    for kind in ("ffn", "down_proj"):
        for phase, buckets in summary.get(kind, {}).items():
            for bucket, bucket_summary in buckets.items():
                if bucket_summary.get("is_tie"):
                    tied = ",".join(
                        entry["operator"] for entry in bucket_summary.get("tied_winners", [])
                    )
                    count = bucket_summary["tied_winners"][0]["count"]
                    share = bucket_summary["tied_winners"][0]["share"]
                    print(
                        f"{kind}:{phase}:m={bucket}:tie={tied}:"
                        f"{count}/{bucket_summary['total']}:"
                        f"{share:.3f}"
                    )
                else:
                    winner = bucket_summary["winner"]
                    print(
                        f"{kind}:{phase}:m={bucket}:{winner['operator']}:"
                        f"{winner['count']}/{bucket_summary['total']}:"
                        f"{winner['share']:.3f}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", help="Prometheus metrics snapshot")
    parser.add_argument("output_json", help="Output JSON path")
    parser.add_argument(
        "--stats",
        help="Optional stats JSON file to augment with inferflux_cuda_bucket_winners",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    summary = normalize(aggregate(metrics_path))
    Path(args.output_json).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.stats:
        merge_stats(Path(args.stats), summary)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
