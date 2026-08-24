#!/usr/bin/env python3
"""Summarize and compare Nsight Systems backend profile exports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CSV_REPORTS = {
    "cuda_gpu_kern_sum": ["Name", "Instances", "Total Time", "Avg"],
    "cuda_gpu_kern_gb_sum": ["Name", "GridXYZ", "BlockXYZ", "Instances", "Total Time", "Avg"],
    "cuda_kern_exec_sum": ["Kernel Name", "API Name", "Count", "QCount", "AAvg", "QAvg", "KAvg", "TAvg"],
    "cuda_api_sum": ["Name", "Num Calls", "Total Time", "Avg"],
    "cuda_gpu_mem_time_sum": ["Operation", "Count", "Total Time", "Avg"],
}


def as_number(value: str) -> Optional[float]:
    raw = value.strip().replace(",", "")
    if not raw or raw in {"nan", "N/A", "None"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_key(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)", "", value)
    value = value.replace("%", "")
    return " ".join(value.strip().split())


def detect_header(lines: List[str], expected: Iterable[str]) -> int:
    expected_set = {normalize_key(item) for item in expected}
    for idx, line in enumerate(lines):
        if not line or line.startswith("NOTICE:") or line.startswith("Processing "):
            continue
        columns = [normalize_key(part.strip()) for part in next(csv.reader([line]))]
        if expected_set.issubset(columns):
            return idx
    return -1


def load_csv_report(path: Path, expected_columns: List[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SKIPPED:" in text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    header_idx = detect_header(lines, expected_columns)
    if header_idx < 0:
        return []

    rows: List[Dict[str, Any]] = []
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        if not row:
            continue
        if all((value is None or not str(value).strip()) for value in row.values()):
            continue
        cleaned: Dict[str, Any] = {}
        for key, value in row.items():
            if key is None:
                continue
            key = normalize_key(key)
            value = (value or "").strip()
            numeric = as_number(value)
            cleaned[key] = numeric if numeric is not None else value
        rows.append(cleaned)
    return rows


def sort_desc(rows: List[Dict[str, Any]], key: str, limit: int = 10) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0) or 0), reverse=True)[:limit]


def weighted_average(rows: List[Dict[str, Any]], value_key: str, weight_key: str) -> float:
    total_weight = 0.0
    total_value = 0.0
    for row in rows:
        value = float(row.get(value_key, 0) or 0)
        weight = float(row.get(weight_key, 0) or 0)
        total_value += value * weight
        total_weight += weight
    return total_value / total_weight if total_weight else 0.0


class ProfileSummary:
    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = profile_dir
        self.reports = {
            report: load_csv_report(profile_dir / f"{report}.csv", columns)
            for report, columns in CSV_REPORTS.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        kernels = self.reports["cuda_gpu_kern_sum"]
        kernel_geom = self.reports["cuda_gpu_kern_gb_sum"]
        exec_rows = self.reports["cuda_kern_exec_sum"]
        api_rows = self.reports["cuda_api_sum"]
        mem_rows = self.reports["cuda_gpu_mem_time_sum"]

        launch_rows = [
            row for row in api_rows
            if str(row.get("Name", "")).startswith("cudaLaunch")
            or str(row.get("Name", "")).startswith("cudaGraphLaunch")
        ]

        return {
            "profile_dir": str(self.profile_dir),
            "kernel_records": len(kernels),
            "kernel_total_time_ns": sum(float(row.get("Total Time", 0) or 0) for row in kernels),
            "kernel_total_instances": sum(float(row.get("Instances", 0) or 0) for row in kernels),
            "top_kernels": sort_desc(kernels, "Total Time"),
            "top_kernel_shapes": sort_desc(kernel_geom, "Total Time"),
            "top_launch_exec": sort_desc(exec_rows, "TAvg"),
            "top_queue": sort_desc(exec_rows, "QAvg"),
            "launch_api_total_calls": sum(float(row.get("Num Calls", 0) or 0) for row in launch_rows),
            "launch_api_total_time_ns": sum(float(row.get("Total Time", 0) or 0) for row in launch_rows),
            "cuda_launch_avg_ns_weighted": weighted_average(launch_rows, "Avg", "Num Calls"),
            "kernel_exec_avg_ns_weighted": weighted_average(exec_rows, "KAvg", "Count"),
            "queue_avg_ns_weighted": weighted_average(exec_rows, "QAvg", "QCount"),
            "top_cuda_apis": sort_desc(api_rows, "Total Time"),
            "top_mem_ops": sort_desc(mem_rows, "Total Time"),
        }


def format_ns(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f} us"
    return f"{value:.0f} ns"


def print_profile(name: str, summary: Dict[str, Any]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"kernel records:        {summary['kernel_records']}")
    print(f"kernel instances:      {int(summary['kernel_total_instances'])}")
    print(f"kernel total time:     {format_ns(summary['kernel_total_time_ns'])}")
    print(f"launch API calls:      {int(summary['launch_api_total_calls'])}")
    print(f"launch API total time: {format_ns(summary['launch_api_total_time_ns'])}")
    print(f"avg launch time:       {format_ns(summary['cuda_launch_avg_ns_weighted'])}")
    print(f"avg queue time:        {format_ns(summary['queue_avg_ns_weighted'])}")
    print(f"avg kernel time:       {format_ns(summary['kernel_exec_avg_ns_weighted'])}")
    if summary["kernel_records"] == 0 and summary["top_cuda_apis"]:
        print("kernel data:           unavailable in this nsys capture; use API deltas plus an ncu follow-up for kernel-level detail")

    print("\ntop kernels by total time:")
    for row in summary["top_kernels"][:8]:
        print(
            f"  {row.get('Name', '<unknown>')} | calls={int(row.get('Instances', 0) or 0)} "
            f"| total={format_ns(float(row.get('Total Time', 0) or 0))} "
            f"| avg={format_ns(float(row.get('Avg', 0) or 0))}"
        )

    print("\ntop kernel shapes:")
    for row in summary["top_kernel_shapes"][:8]:
        print(
            f"  {row.get('Name', '<unknown>')} | grid={row.get('GridXYZ', '?')} "
            f"| block={row.get('BlockXYZ', '?')} "
            f"| calls={int(row.get('Instances', 0) or 0)} "
            f"| total={format_ns(float(row.get('Total Time', 0) or 0))}"
        )

    print("\ntop launch/exec rows:")
    for row in summary["top_launch_exec"][:8]:
        print(
            f"  {row.get('Kernel Name', '<unknown>')} via {row.get('API Name', '<api>')} "
            f"| count={int(row.get('Count', 0) or 0)} qcount={int(row.get('QCount', 0) or 0)} "
            f"| api_avg={format_ns(float(row.get('AAvg', 0) or 0))} "
            f"| queue_avg={format_ns(float(row.get('QAvg', 0) or 0))} "
            f"| kernel_avg={format_ns(float(row.get('KAvg', 0) or 0))}"
        )

    print("\ntop CUDA APIs:")
    for row in summary["top_cuda_apis"][:8]:
        print(
            f"  {row.get('Name', '<unknown>')} | calls={int(row.get('Num Calls', 0) or 0)} "
            f"| total={format_ns(float(row.get('Total Time', 0) or 0))} "
            f"| avg={format_ns(float(row.get('Avg', 0) or 0))}"
        )


def print_comparison(lhs_name: str, lhs: Dict[str, Any], rhs_name: str, rhs: Dict[str, Any]) -> None:
    print("\ncomparison")
    print("----------")
    print(
        f"kernel instances: {lhs_name}={int(lhs['kernel_total_instances'])} "
        f"vs {rhs_name}={int(rhs['kernel_total_instances'])}"
    )
    print(
        f"kernel total time: {lhs_name}={format_ns(lhs['kernel_total_time_ns'])} "
        f"vs {rhs_name}={format_ns(rhs['kernel_total_time_ns'])}"
    )
    print(
        f"launch avg: {lhs_name}={format_ns(lhs['cuda_launch_avg_ns_weighted'])} "
        f"vs {rhs_name}={format_ns(rhs['cuda_launch_avg_ns_weighted'])}"
    )
    print(
        f"queue avg: {lhs_name}={format_ns(lhs['queue_avg_ns_weighted'])} "
        f"vs {rhs_name}={format_ns(rhs['queue_avg_ns_weighted'])}"
    )
    print(
        f"kernel avg: {lhs_name}={format_ns(lhs['kernel_exec_avg_ns_weighted'])} "
        f"vs {rhs_name}={format_ns(rhs['kernel_exec_avg_ns_weighted'])}"
    )

    interesting = [
        "cudaMemcpyAsync",
        "cudaLaunchKernel",
        "cudaGraphLaunch_v10000",
        "cudaStreamSynchronize",
        "cudaMalloc",
        "cudaFree",
    ]
    lhs_api = {str(row.get("Name", "")): row for row in lhs["top_cuda_apis"]}
    rhs_api = {str(row.get("Name", "")): row for row in rhs["top_cuda_apis"]}
    print("\ninteresting CUDA API deltas:")
    for name in interesting:
        lhs_row = lhs_api.get(name, {})
        rhs_row = rhs_api.get(name, {})
        lhs_calls = int(lhs_row.get("Num Calls", 0) or 0)
        rhs_calls = int(rhs_row.get("Num Calls", 0) or 0)
        lhs_total = float(lhs_row.get("Total Time", 0) or 0)
        rhs_total = float(rhs_row.get("Total Time", 0) or 0)
        lhs_avg = float(lhs_row.get("Avg", 0) or 0)
        rhs_avg = float(rhs_row.get("Avg", 0) or 0)
        print(
            f"  {name}: calls {lhs_name}={lhs_calls} vs {rhs_name}={rhs_calls}; "
            f"total {lhs_name}={format_ns(lhs_total)} vs {rhs_name}={format_ns(rhs_total)}; "
            f"avg {lhs_name}={format_ns(lhs_avg)} vs {rhs_name}={format_ns(rhs_avg)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze or compare Nsight Systems backend profiles")
    parser.add_argument("profile_dirs", nargs="+", help="One or two directories containing exported nsys CSV reports")
    parser.add_argument("--json", dest="json_path", help="Optional path for machine-readable summary JSON")
    args = parser.parse_args()

    if len(args.profile_dirs) not in {1, 2}:
        parser.error("provide one or two profile directories")

    summaries: Dict[str, Dict[str, Any]] = {}
    for raw in args.profile_dirs:
        profile_dir = Path(raw)
        if not profile_dir.exists():
            raise SystemExit(f"profile directory not found: {profile_dir}")
        summaries[profile_dir.name] = ProfileSummary(profile_dir).to_dict()

    names = list(summaries.keys())
    for name in names:
        print_profile(name, summaries[name])
    if len(names) == 2:
        print_comparison(names[0], summaries[names[0]], names[1], summaries[names[1]])

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
