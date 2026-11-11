#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect single-scan Nsight Compute diagnostics without changing GPU clocks."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

METRICS = {
    "gpu__time_duration.sum": "duration_ns",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__registers_per_thread_allocated": "allocated_registers_per_thread",
    "launch__shared_mem_per_block_dynamic": "dynamic_shared_bytes",
    "launch__occupancy_limit_shared_mem": "resident_blocks_limit_shared_memory",
    "launch__occupancy_limit_registers": "resident_blocks_limit_registers",
    "launch__occupancy_limit_barriers": "resident_blocks_limit_barriers",
    "sass__size": "kernel_sass_bytes",
    "smsp__inst_executed.sum": "executed_warp_instructions",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "active_warps_percent",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 1048576, 16777216]
    )
    parser.add_argument(
        "--type", choices=["int32", "float32", "float64"], default="float32"
    )
    parser.add_argument(
        "--operation", choices=["inclusive", "exclusive"], default="exclusive"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    binaries = {}
    commands = []
    for spec in args.binary:
        label, binary = spec.split("=", 1)
        binary = str(Path(binary).resolve())
        binaries[label] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        for count in args.sizes:
            raw = args.raw_dir / f"{label}-{args.type}-{count}.csv"
            command = [
                "ncu",
                "--clock-control",
                "none",
                "--cache-control",
                "none",
                "--replay-mode",
                "kernel",
                "--kernel-name",
                "regex:DeviceScanKernel",
                "--launch-count",
                "1",
                "--section",
                "LaunchStats",
                "--section",
                "InstructionStats",
                "--section",
                "Occupancy",
                "--section",
                "SpeedOfLight",
                "--csv",
                "--page",
                "raw",
                "--log-file",
                str(raw),
                binary,
                "--profile-case",
                args.type,
                args.operation,
                str(count),
            ]
            commands.append(command)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            raw.with_suffix(".stdout").write_text(result.stdout + result.stderr)
            contents = raw.read_text()
            data = csv.DictReader(io.StringIO(contents[contents.index('"ID",') :]))
            record = next(row for row in data if row["ID"])
            rows.append(
                {
                    "variant": label,
                    "type": args.type,
                    "operation": args.operation,
                    "elements": count,
                    **{label: record[metric] for metric, label in METRICS.items()},
                }
            )
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "ncu_version": subprocess.run(
            ["ncu", "--version"], check=True, capture_output=True, text=True
        ).stdout,
        "binary_sha256": binaries,
        "commands": commands,
        "notes": "One validated scan per process; clock and cache control disabled. Profiling duration is not the event benchmark. Total instruction counts vary with lookahead polling and CTA scheduling.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
