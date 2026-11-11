#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare original CUB, private group adapters, and public group adapters."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIVATE_REFERENCE_REF = "24ee635e35caa47b1602e129e49cea60db8b4f05"
VARIANTS = {
    "baseline": [],
    "groups": ["-D_CUB_WARPSPEED_USE_CUDAX_GROUPS"],
    "groups_scan": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN",
    ],
    "static_groups": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
    ],
    "static_groups_scan": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
        "-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN",
    ],
    "before_static_groups": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
    ],
    "before_static_groups_scan": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
        "-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN",
    ],
    "prior_static_groups": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
    ],
    "prior_static_groups_scan": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
        "-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN",
    ],
    "private_groups": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
    ],
    "private_groups_scan": [
        "-D_CUB_WARPSPEED_USE_CUDAX_GROUPS",
        "-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS",
        "-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN",
    ],
}
DEFAULT_VARIANTS = [
    "baseline",
    "private_groups",
    "private_groups_scan",
    "static_groups",
    "static_groups_scan",
]
HEADER_ROOTS = {
    "libcudacxx/include": "",
    "cudax/include": "",
    "cub/cub": "cub",
    "thrust/thrust": "thrust",
}


def source_hashes():
    sources = [
        HERE / "scan.cu",
        HERE / "private-before-public.patch",
        Path(__file__).resolve(),
    ]
    for directory in HEADER_ROOTS:
        sources.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    return {
        str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted(sources)
    }


def command(arguments, **kwargs):
    try:
        return subprocess.run(arguments, cwd=ROOT, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr)
        raise


def materialize_fixture(fixture, directory):
    previous_manifest = directory / "snapshot-sha256.json"
    if previous_manifest.exists():
        for header in json.loads(previous_manifest.read_text()):
            (directory / header).unlink(missing_ok=True)
    headers = [
        Path(line.removeprefix("--- "))
        for line in fixture.read_text().splitlines()
        if line.startswith("--- ")
    ]
    for header in headers:
        target = directory / header
        target.parent.mkdir(parents=True, exist_ok=True)
        if header.parts[0] == "cub":
            source = ROOT / "cub" / header
        elif header.parts[0] == "cuda":
            source = ROOT / "cudax/include" / header
        else:
            raise RuntimeError(f"Unexpected reference header: {header}")
        shutil.copy2(source, target)
    command(
        [
            "patch",
            "--batch",
            "--forward",
            "-p0",
            "--directory",
            str(directory),
            "--input",
            str(fixture),
        ],
        capture_output=True,
    )
    (directory / "snapshot-sha256.json").write_text(
        json.dumps(
            {
                str(header): hashlib.sha256(
                    (directory / header).read_bytes()
                ).hexdigest()
                for header in headers
            },
            indent=2,
        )
        + "\n"
    )


def validated_snapshot(overlay):
    contents = (overlay / "snapshot-sha256.json").read_bytes()
    manifest = json.loads(contents)
    observed = {
        str(header.relative_to(overlay))
        for header in overlay.rglob("*")
        if header.is_file() and header.name != "snapshot-sha256.json"
    }
    if observed != manifest.keys():
        raise RuntimeError(f"Reference snapshot file set changed: {overlay}")
    for name, digest in manifest.items():
        if hashlib.sha256((overlay / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Reference snapshot changed: {overlay / name}")
    return {
        "path": str(overlay.resolve()),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def materialize_private_reference(args, directory):
    """Recover all private adapter dependencies without using current group headers."""
    fixture = HERE / "private-before-public.patch"
    with tempfile.TemporaryDirectory(
        prefix="private-checkout-", dir=directory
    ) as temporary:
        checkout = Path(temporary)
        archive = subprocess.check_output(
            ["git", "archive", PRIVATE_REFERENCE_REF, *HEADER_ROOTS], cwd=ROOT
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as headers:
            headers.extractall(checkout, filter="data")
        command(
            [
                "patch",
                "--batch",
                "--forward",
                "-p1",
                "--directory",
                str(checkout),
                "--input",
                str(fixture),
            ],
            capture_output=True,
        )
        overlay = args.private_headers
        previous_manifest = overlay / "snapshot-sha256.json"
        if previous_manifest.exists():
            for name in json.loads(previous_manifest.read_text()):
                (overlay / name).unlink(missing_ok=True)
        overlay.mkdir(parents=True, exist_ok=True)
        for source, target in HEADER_ROOTS.items():
            shutil.copytree(checkout / source, overlay / target, dirs_exist_ok=True)
    manifest = {
        str(header.relative_to(overlay)): hashlib.sha256(
            header.read_bytes()
        ).hexdigest()
        for header in sorted(overlay.rglob("*"))
        if header.is_file() and header.name != "snapshot-sha256.json"
    }
    (overlay / "snapshot-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build(args, directory):
    overlays = {}
    for prefix, overlay in (
        ("before_", args.before_headers),
        ("prior_", args.prior_headers),
        ("private_", args.private_headers),
    ):
        if not any(variant.startswith(prefix) for variant in args.variants):
            continue
        overlays[prefix] = validated_snapshot(overlay)
    snapshot = source_hashes()
    (directory / "source-hashes.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    baseline = directory / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    # Always recover the exact committed baseline, independently of working-tree edits.
    archive = subprocess.check_output(
        ["git", "archive", args.baseline_ref, "cub/cub"], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as headers:
        for member in headers.getmembers():
            member.name = str(Path(member.name).relative_to("cub"))
            headers.extract(member, baseline, filter="data")
    for variant in args.variants:
        definitions = VARIANTS[variant]
        include = baseline if variant == "baseline" else ROOT / "cub"
        if variant.startswith("before_"):
            include = args.before_headers.resolve()
        elif variant.startswith("prior_"):
            include = args.prior_headers.resolve()
        elif variant.startswith("private_"):
            include = args.private_headers.resolve()
        if (
            variant.startswith(("before_", "prior_", "private_"))
            and not (include / "cub/detail/warpspeed/squad/squad.cuh").exists()
        ):
            raise RuntimeError(f"Missing reference header snapshot: {include}")
        temporary_root = directory / "nvcc-tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{variant}-", dir=temporary_root
        ) as temporary:
            environment = dict(os.environ, TMPDIR=temporary)
            result = command(
                [
                    args.nvcc,
                    f"-ccbin={args.cxx}",
                    "-std=c++17",
                    "-O3",
                    "-arch=sm_120",
                    "-lineinfo",
                    "-Xptxas=-v",
                    f"-I{include}",
                    *(
                        [f"-I{args.private_headers.resolve()}"]
                        if args.private_headers is not None and variant == "baseline"
                        else []
                    ),
                    f"-I{ROOT / 'cub'}",
                    f"-I{ROOT / 'thrust'}",
                    f"-I{ROOT / 'libcudacxx/include'}",
                    f"-I{ROOT / 'cudax/include'}",
                    "-D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
                    f'-DLOOKAHEAD_GROUPS_VARIANT="{variant}"',
                    *definitions,
                    str(HERE / "scan.cu"),
                    "-o",
                    str(directory / f"{variant}-scan"),
                ],
                capture_output=True,
                env=environment,
            )
        (directory / f"{variant}-build.log").write_text(result.stdout + result.stderr)
    if snapshot != source_hashes():
        raise RuntimeError("Source changed while compiling; rebuild before timing")
    for reference in overlays.values():
        if reference["sha256"] != validated_snapshot(Path(reference["path"]))["sha256"]:
            raise RuntimeError("Reference snapshot manifest changed while compiling")
    provenance = {
        "source_sha256": snapshot,
        "reference_overlays": overlays,
        "binary_sha256": {
            variant: hashlib.sha256(
                (directory / f"{variant}-scan").read_bytes()
            ).hexdigest()
            for variant in args.variants
        },
    }
    (directory / "build-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def write_codegen(directory, results, variants):
    rows = []
    for variant in variants:
        name = None
        stack = None
        for line in (directory / f"{variant}-build.log").read_text().splitlines():
            entry = re.search(r"Compiling entry function '([^']+)'", line)
            if entry:
                name = command(
                    ["c++filt", entry[1]], capture_output=True
                ).stdout.strip()
                stack = None
            frame = re.search(
                r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                line,
            )
            if frame:
                stack = frame.groups()
            resources = re.search(r"Used (\d+) registers, used (\d+) barriers", line)
            if not resources or not name or "::DeviceScanKernel<" not in name:
                continue
            if stack is None:
                raise RuntimeError(f"Missing stack resource record for {name}")
            value_type = "affine"
            for spelling, label in (
                ("double const*", "float64"),
                ("float const*", "float32"),
                ("int const*", "int32"),
            ):
                if spelling in name:
                    value_type = label
                    break
            rows.append(
                {
                    "variant": variant,
                    "type": value_type,
                    "operation": "exclusive"
                    if "::detail::InputValue<" in name
                    else "inclusive",
                    "registers": resources[1],
                    "barriers": resources[2],
                    "stack_bytes": stack[0],
                    "spill_store_bytes": stack[1],
                    "spill_load_bytes": stack[2],
                }
            )
    if not rows:
        raise RuntimeError("No scan kernel resources found in compiler logs")
    with (results / "codegen.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sass(args, directory, results):
    dispatch = []
    kernels = []
    opcodes = []
    dumps = directory / "profile"
    dumps.mkdir(exist_ok=True)
    instruction = re.compile(
        r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+[0-9]+\s+)?([A-Z][A-Z0-9_.]*)(?=[\s;])"
    )
    for variant in args.variants:
        dump = command(
            [
                str(Path(args.nvcc).with_name("cuobjdump")),
                "--dump-sass",
                str(directory / f"{variant}-scan"),
            ],
            capture_output=True,
        ).stdout
        (dumps / f"final-{variant}.sass").write_text(dump)
        counts = Counter(instruction.findall(dump))
        dispatch.append(
            {
                "variant": variant,
                "instruction_rows": sum(counts.values()),
                "brx": counts["BRX"],
                "brxu": counts["BRXU"],
                "bssy": sum(
                    value for name, value in counts.items() if name.startswith("BSSY")
                ),
                "bsync": sum(
                    value for name, value in counts.items() if name.startswith("BSYNC")
                ),
                "uniform_bra": counts["BRA.U"],
            }
        )
        for section in re.split(r"\s*Function : ", dump)[1:]:
            name = section.splitlines()[0]
            if "DeviceScanKernel" not in name:
                continue
            demangled = command(["c++filt", name], capture_output=True).stdout
            value_type = "affine"
            for spelling, label in (
                ("double const*", "float64"),
                ("float const*", "float32"),
                ("int const*", "int32"),
            ):
                if spelling in demangled:
                    value_type = label
                    break
            operation = (
                "exclusive" if "::detail::InputValue<" in demangled else "inclusive"
            )
            counts = Counter(instruction.findall(section))
            kernels.append(
                {
                    "variant": variant,
                    "type": value_type,
                    "operation": operation,
                    "instruction_rows": sum(counts.values()),
                }
            )
            opcodes.extend(
                {
                    "variant": variant,
                    "type": value_type,
                    "operation": operation,
                    "opcode": opcode,
                    "instruction_rows": count,
                }
                for opcode, count in sorted(counts.items())
            )
    for name, rows in (
        ("dispatch-codegen.csv", dispatch),
        ("sass-kernels.csv", kernels),
        ("sass-opcodes.csv", opcodes),
    ):
        with (results / name).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def write_comparisons(rows, results):
    grouped = {}
    for row in rows:
        key = tuple(
            row[name] for name in ("variant", "type", "operation", "elements", "shift")
        )
        grouped.setdefault(key, {})[int(row["round"])] = float(row["median_us"])
    comparisons = []
    for key, candidate in sorted(grouped.items()):
        variant, value_type, operation, elements, shift = key
        if variant == "baseline":
            continue
        references = ["baseline"]
        if variant in ("static_groups", "static_groups_scan"):
            controls = [f"prior_{variant}", variant.replace("static_", "private_", 1)]
            references.extend(
                control for control in controls if (control, *key[1:]) in grouped
            )
        for reference in references:
            control = grouped[(reference, *key[1:])]
            rounds = sorted(candidate.keys() & control.keys())
            ratios = []
            generator = random.Random(42)
            for _ in range(2000):
                selected = generator.choices(rounds, k=len(rounds))
                ratios.append(
                    statistics.median(candidate[index] for index in selected)
                    / statistics.median(control[index] for index in selected)
                )
            ratios.sort()
            ratio = statistics.median(candidate.values()) / statistics.median(
                control.values()
            )
            comparisons.append(
                {
                    "variant": variant,
                    "reference": reference,
                    "type": value_type,
                    "operation": operation,
                    "elements": elements,
                    "shift": shift,
                    "rounds": len(rounds),
                    "overhead_percent": f"{100 * (ratio - 1):.6f}",
                    "bootstrap_95_low_percent": f"{100 * (ratios[49] - 1):.6f}",
                    "bootstrap_95_high_percent": f"{100 * (ratios[1949] - 1):.6f}",
                }
            )
    with (results / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparisons[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)


def validated_build(args, directory):
    provenance = json.loads((directory / "build-provenance.json").read_text())
    if provenance["source_sha256"] != source_hashes():
        raise RuntimeError("Source changed after compilation; rebuild before timing")
    for prefix, overlay in (
        ("before_", args.before_headers),
        ("prior_", args.prior_headers),
        ("private_", args.private_headers),
    ):
        if any(variant.startswith(prefix) for variant in args.variants):
            recorded = provenance["reference_overlays"][prefix]
            if recorded["sha256"] != validated_snapshot(overlay)["sha256"]:
                raise RuntimeError(
                    f"Reference overlay changed after compilation: {overlay}"
                )
    for variant in args.variants:
        if (
            provenance["binary_sha256"][variant]
            != hashlib.sha256((directory / f"{variant}-scan").read_bytes()).hexdigest()
        ):
            raise RuntimeError(f"Binary changed after compilation: {variant}")
    return provenance


def run(args, directory):
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    provenance = validated_build(args, directory)
    rows = []
    logs = []
    orders = []
    properties = None
    # Five rounds give each of the five variants every process position once.
    for round_index in range(args.rounds):
        variants = list(args.variants)
        offset = round_index % len(variants)
        variants = variants[offset:] + variants[:offset]
        if (round_index // len(variants)) % 2:
            variants.reverse()
        orders.append(variants)
        for variant in variants:
            for selected_case in args.case or [None]:
                invocation = [str(directory / f"{variant}-scan")]
                if selected_case is not None:
                    invocation += ["--case", *selected_case]
                result = command(invocation, capture_output=True)
                logs.append(
                    f"round {round_index}, {variant}, {selected_case}\n{result.stderr}"
                )
                match = re.search(
                    r"device_properties: l2_cache_bytes=(\d+) multiprocessor_count=(\d+) "
                    r"shared_memory_per_multiprocessor_bytes=(\d+)",
                    result.stderr,
                )
                if not match:
                    raise RuntimeError(f"Missing device properties from {variant}")
                observed = dict(
                    zip(
                        (
                            "l2_cache_bytes",
                            "multiprocessor_count",
                            "shared_memory_per_multiprocessor_bytes",
                        ),
                        map(int, match.groups()),
                    )
                )
                if properties is not None and observed != properties:
                    raise RuntimeError(
                        "Device properties changed between benchmark processes"
                    )
                properties = observed
                for row in csv.DictReader(io.StringIO(result.stdout)):
                    rows.append({"round": round_index, **row})
        print(
            f"Completed round {round_index + 1}/{args.rounds}: {', '.join(variants)}",
            flush=True,
        )
    if provenance != validated_build(args, directory):
        raise RuntimeError("Build provenance changed during timing")
    with (results / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (directory / "validation.log").write_text("\n".join(logs))
    grouped = {}
    for row in rows:
        key = tuple(
            row[name] for name in ("variant", "type", "operation", "elements", "shift")
        )
        grouped.setdefault(key, []).append(float(row["median_us"]))
    summaries = []
    for key, values in grouped.items():
        variant, value_type, operation, elements, shift = key
        baseline_values = grouped[("baseline", value_type, operation, elements, shift)]
        median = statistics.median(values)
        baseline_median = statistics.median(baseline_values)
        summaries.append(
            {
                "variant": variant,
                "type": value_type,
                "operation": operation,
                "elements": elements,
                "shift": shift,
                "median_us": f"{median:.6f}",
                "min_round_us": f"{min(values):.6f}",
                "max_round_us": f"{max(values):.6f}",
                "speedup": f"{baseline_median / median:.6f}",
            }
        )
    summaries.sort(
        key=lambda row: (
            row["variant"],
            row["type"],
            row["operation"],
            int(row["elements"]),
            row["shift"],
        )
    )
    with (results / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_revision": command(
            ["git", "rev-parse", args.baseline_ref], capture_output=True
        ).stdout.strip(),
        "rounds": args.rounds,
        "selected_cases": args.case,
        "process_order": orders,
        "definitions": {variant: VARIANTS[variant] for variant in args.variants},
        "variant_include_overlay": {
            variant: str(args.before_headers.resolve())
            if variant.startswith("before_")
            else str(args.prior_headers.resolve())
            if variant.startswith("prior_")
            else str(args.private_headers.resolve())
            if variant.startswith("private_")
            else str((directory / "baseline").resolve())
            if variant == "baseline"
            else None
            for variant in args.variants
        },
        "language_standard": "C++17",
        "architecture": "sm_120",
        "clock_control": "Application clocks and power limit left unchanged",
        "build_temporary_directory": "Isolated per nvcc invocation, removed after compilation",
        "source_sha256": hashlib.sha256((HERE / "scan.cu").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiled_source_sha256": json.loads(
            (directory / "source-hashes.json").read_text()
        ),
        "build_provenance": provenance,
        "before_header_snapshot": str(args.before_headers.resolve()),
        "before_header_snapshot_sha256": hashlib.sha256(
            (args.before_headers / "snapshot-sha256.json").read_bytes()
        ).hexdigest()
        if any(variant.startswith("before_") for variant in args.variants)
        else None,
        "before_header_fixture_sha256": hashlib.sha256(
            (HERE / "static-before-fix.patch").read_bytes()
        ).hexdigest()
        if any(variant.startswith("before_") for variant in args.variants)
        else None,
        "prior_header_snapshot": str(args.prior_headers.resolve()),
        "prior_header_snapshot_sha256": hashlib.sha256(
            (args.prior_headers / "snapshot-sha256.json").read_bytes()
        ).hexdigest()
        if any(variant.startswith("prior_") for variant in args.variants)
        else None,
        "prior_header_fixture_sha256": hashlib.sha256(
            (HERE / "static-before-refinement.patch").read_bytes()
        ).hexdigest()
        if any(variant.startswith("prior_") for variant in args.variants)
        and (HERE / "static-before-refinement.patch").exists()
        else None,
        "private_header_snapshot": str(args.private_headers.resolve())
        if args.private_headers is not None
        else None,
        "private_header_snapshot_sha256": hashlib.sha256(
            (args.private_headers / "snapshot-sha256.json").read_bytes()
        ).hexdigest()
        if any(variant.startswith("private_") for variant in args.variants)
        else None,
        "private_header_fixture_sha256": hashlib.sha256(
            (HERE / "private-before-public.patch").read_bytes()
        ).hexdigest()
        if any(variant.startswith("private_") for variant in args.variants)
        else None,
        "private_header_reference_revision": PRIVATE_REFERENCE_REF
        if any(variant.startswith("private_") for variant in args.variants)
        else None,
        "baseline_support_headers": str(args.private_headers.resolve())
        if args.private_headers is not None
        else None,
        "compiler": command([args.nvcc, "--version"], capture_output=True).stdout,
        "host_compiler": command([args.cxx, "--version"], capture_output=True).stdout,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory",
                "--format=csv",
            ],
            capture_output=True,
        ).stdout,
        "device_properties": properties,
        "binary_sha256": {
            variant: hashlib.sha256(
                (directory / f"{variant}-scan").read_bytes()
            ).hexdigest()
            for variant in args.variants
        },
        "timing": "CUDA graph replay, 64 scans/batch for <=2^20 elements else 8; at least 50ms graph execution warm-up per timed case; median of 9 event batches per round",
        "comparison_statistics": "Ratio of process medians; 95% percentile bootstrap interval from 2000 paired round resamples with seed42. Intervals describe measured run variation, not systematic clock or cache effects.",
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_codegen(directory, results, args.variants)
    write_sass(args, directory, results)
    write_comparisons(rows, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="/usr/local/cuda-13.5/bin/nvcc")
    parser.add_argument("--cxx", default="/usr/bin/g++-13")
    parser.add_argument(
        "--baseline-ref", default="24ee635e35caa47b1602e129e49cea60db8b4f05"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("TYPE", "OPERATION", "ELEMENTS"),
        help="repeat a selected aligned case instead of running the full matrix; may be repeated",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS
    )
    parser.add_argument(
        "--before-headers",
        type=Path,
        help="optional full before-fix snapshot; otherwise materialize the reference patch",
    )
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build/lookahead-groups"
    )
    parser.add_argument(
        "--prior-headers",
        type=Path,
        help="header overlay for the uniform-dispatch implementation before further refinement",
    )
    parser.add_argument(
        "--private-headers",
        type=Path,
        help="complete private-adapter dependency snapshot; otherwise recover the pinned reference patch",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.case and len({tuple(case) for case in args.case}) != len(args.case):
        parser.error("--case selections must be distinct")
    if not args.build_only and "baseline" not in args.variants:
        parser.error("timing comparisons require baseline in --variants")
    directory = args.build_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.private_headers is None and any(
        variant.startswith("private_") for variant in args.variants
    ):
        args.private_headers = directory / "portable-private-headers"
        materialize_private_reference(args, directory)
    if args.prior_headers is None:
        args.prior_headers = directory / "portable-prior-headers"
        if any(variant.startswith("prior_") for variant in args.variants):
            materialize_fixture(
                HERE / "static-before-refinement.patch", args.prior_headers
            )
    if any(variant.startswith("before_") for variant in args.variants):
        if args.before_headers is None:
            args.before_headers = directory / "portable-before-headers"
            materialize_fixture(HERE / "static-before-fix.patch", args.before_headers)
    elif args.before_headers is None:
        args.before_headers = directory / "portable-before-headers"
    if not args.run_only:
        build(args, directory)
    if not args.build_only:
        run(args, directory)


if __name__ == "__main__":
    main()
