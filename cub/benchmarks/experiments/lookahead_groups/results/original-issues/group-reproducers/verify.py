#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Reproduce original coop header compile failures against a frozen fixed control."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("build/original-group-issues")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = Path(__file__).resolve().parent
    revisions = {
        "original": "24ee635e35caa47b1602e129e49cea60db8b4f05",
        "fixed": subprocess.check_output(
            ["git", "rev-parse", "0662ef1152"], cwd=root, text=True
        ).strip(),
    }
    for label, revision in revisions.items():
        archive = subprocess.check_output(
            ["git", "archive", revision, "cudax/include/cuda/experimental/coop"],
            cwd=root,
        )
        prefix = Path("cudax/include")
        with tarfile.open(fileobj=io.BytesIO(archive)) as headers:
            for member in headers.getmembers():
                path = Path(member.name)
                if path == prefix or prefix not in path.parents:
                    continue
                member.name = str(path.relative_to(prefix))
                headers.extract(member, output / label, filter="data")

    nvcc_toolchains = {
        "latest": [
            "/usr/local/cuda-13.5/bin/nvcc",
            "-ccbin=/usr/bin/g++-13",
            "-arch=sm_120",
        ],
        "cuda12": [
            "/usr/local/cuda-12.0/bin/nvcc",
            "-ccbin=/usr/bin/g++-7",
            "-gencode=arch=compute_90,code=compute_90",
        ],
    }
    common = [
        "-std=c++17",
        "-D_CCCL_NO_SYSTEM_HEADER",
        "-D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
    ]
    jobs = []
    for label in revisions:
        includes = [
            "-I" + str(output / label),
            "-Icudax/include",
            "-Ilibcudacxx/include",
        ]
        for toolchain, compiler in nvcc_toolchains.items():
            for source in ["view_type", "grid_sync"]:
                name = f"{label}-{source}-{toolchain}"
                command = [
                    *compiler,
                    *common,
                    *includes,
                    "-c",
                    str(sources / (source + ".cu")),
                    "-o",
                    str(output / (name + ".o")),
                ]
                expected = 2 if label == "original" and source == "view_type" else 0
                jobs.append((name, command, expected))
        for side in ["host", "device"]:
            for source in ["include_only", "grid_sync"]:
                name = f"{label}-{source}-clang-{side}"
                command = [
                    "/usr/bin/clang++-23",
                    "-x",
                    "cuda",
                    "--cuda-" + side + "-only",
                    "--cuda-gpu-arch=sm_120",
                    "--cuda-path=/usr/local/cuda-13.5",
                    "-fsyntax-only",
                    "-Wno-unknown-cuda-version",
                    "-D_CCCL_CLANG_TIDY_INVOKED=1",
                    *common,
                    *includes,
                    str(sources / (source + ".cu")),
                ]
                jobs.append((name, command, 1 if label == "original" else 0))

    def run(job):
        name, command, expected = job
        with tempfile.TemporaryDirectory(prefix="nvcc-", dir=output) as temporary:
            result = subprocess.run(
                command,
                cwd=root,
                env=dict(os.environ, TMPDIR=temporary),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        (output / (name + ".log")).write_text(
            shlex.join(command) + "\n" + result.stdout
        )
        print(name, result.returncode, "expected", expected, flush=True)
        return {
            "case": name,
            "command": command,
            "exit_code": result.returncode,
            "expected_exit_code": expected,
            "matches_expected": result.returncode == expected,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, jobs))
    versions = {}
    for binary in [
        "/usr/local/cuda-13.5/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/bin/g++-13",
        "/usr/bin/g++-7",
        "/usr/bin/clang++-23",
    ]:
        versions[binary] = subprocess.check_output([binary, "--version"], text=True)
    hashes = {}
    for label in revisions:
        for path in (output / label).rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(output))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for name in ["include_only.cu", "view_type.cu", "grid_sync.cu", "verify.py"]:
        path = sources / name
        hashes["reproducers/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "revisions": revisions,
        "toolchains": versions,
        "source_sha256": hashes,
        "cases": results,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(result["matches_expected"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
