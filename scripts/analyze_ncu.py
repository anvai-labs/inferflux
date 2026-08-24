"""Extract key ncu metrics from the details CSV."""
import csv
import sys
from collections import defaultdict

USAGE = "usage: analyze_ncu.py <ncu_details.csv>"

def main():
    metrics_of_interest = {
        "Memory Throughput": "mem_throughput_pct",
        "SM [%]": "sm_throughput_pct",
        "Compute (SM) Throughput": "compute_throughput_pct",
        "Duration": "duration",
        "Achieved Occupancy": "occupancy",
        "Registers Per Thread": "regs_per_thread",
        "Block Limit Registers": "block_limit_regs",
        "Theoretical Occupancy": "theoretical_occ",
        "Achieved Active Warps Per SM": "active_warps",
    }

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ncu_details.csv"
    if len(sys.argv) > 2:
        print(USAGE, file=sys.stderr)
        sys.exit(2)

    # Collect per-kernel-instance metrics
    kernels = defaultdict(lambda: {"block": "", "grid": ""})

    try:
        f = open(csv_path, 'r', encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"error: {csv_path} not found\n{USAGE}", file=sys.stderr)
        sys.exit(1)

    with f:
        reader = csv.DictReader(f)
        for row in reader:
            kid = row["ID"]
            kname = row["Kernel Name"]
            metric_name = row["Metric Name"]
            metric_value = row["Metric Value"]
            metric_unit = row.get("Metric Unit", "")

            if kid not in kernels:
                kernels[kid] = {"name": kname, "block": row["Block Size"], "grid": row["Grid Size"]}
            else:
                kernels[kid]["name"] = kname
                if row["Block Size"]:
                    kernels[kid]["block"] = row["Block Size"]
                if row["Grid Size"]:
                    kernels[kid]["grid"] = row["Grid Size"]

            for interest_key, short_name in metrics_of_interest.items():
                if interest_key == metric_name:
                    kernels[kid][short_name] = metric_value
                    if metric_unit:
                        kernels[kid][short_name + "_unit"] = metric_unit

    # Group by kernel name for summary
    kernel_groups = defaultdict(list)
    for kid, data in kernels.items():
        short_name = data.get("name", "unknown")
        # Shorten kernel name
        for prefix in ["void inferflux::runtime::cuda::native::", "void inferflux::", "inferflux::runtime::cuda::native::", "inferflux::", "void cuda_kernel::", "void "]:
            if short_name.startswith(prefix):
                short_name = short_name[len(prefix):]
                break
        # Truncate template params
        paren = short_name.find("(")
        if paren > 0:
            short_name = short_name[:paren]
        kernel_groups[short_name].append(data)

    print(f"{'Kernel':<55} {'Count':>5} {'Mem%':>6} {'SM%':>6} {'Occ%':>6} {'Regs':>5} {'Block':>15} {'Grid':>15} {'Duration':>12}")
    print("=" * 140)

    for kname, instances in sorted(kernel_groups.items(), key=lambda x: -len(x[1])):
        count = len(instances)
        mem_vals = [float(i.get("mem_throughput_pct", "0").replace(",", "")) for i in instances if i.get("mem_throughput_pct")]
        sm_vals = [float(i.get("compute_throughput_pct", i.get("sm_throughput_pct", "0")).replace(",", "")) for i in instances if i.get("compute_throughput_pct") or i.get("sm_throughput_pct")]
        occ_vals = [float(i.get("occupancy", "0").replace(",", "").replace("%", "")) for i in instances if i.get("occupancy")]
        reg_vals = [float(i.get("regs_per_thread", "0").replace(",", "")) for i in instances if i.get("regs_per_thread")]
        dur_vals = [i.get("duration", "") for i in instances]

        avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else 0
        avg_sm = sum(sm_vals) / len(sm_vals) if sm_vals else 0
        avg_occ = sum(occ_vals) / len(occ_vals) if occ_vals else 0
        avg_regs = sum(reg_vals) / len(reg_vals) if reg_vals else 0

        block = instances[0].get("block", "")
        grid = instances[0].get("grid", "")
        dur = dur_vals[0] if dur_vals else ""

        print(f"{kname:<55} {count:>5} {avg_mem:>5.1f}% {avg_sm:>5.1f}% {avg_occ:>5.1f}% {avg_regs:>5.0f} {block:>15} {grid:>15} {dur:>12}")

if __name__ == "__main__":
    main()
