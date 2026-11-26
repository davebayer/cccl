# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measure one range benchmark against two libcudacxx/include directories on Linux."""

import argparse
import ctypes
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median


def nvcc_cpu(command):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(command, check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)


def nvrtc_cpu(lib, source, name, options):
    program = ctypes.c_void_p()
    status = lib.nvrtcCreateProgram(ctypes.byref(program), source, name, 0, None, None)
    if status:
        raise RuntimeError(f"nvrtcCreateProgram failed with status {status}")
    try:
        argv = (ctypes.c_char_p * len(options))(*(os.fsencode(opt) for opt in options))
        start = time.process_time()
        status = lib.nvrtcCompileProgram(program, len(options), argv)
        elapsed = time.process_time() - start
        if status:
            size = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(program, log)
            raise RuntimeError(
                f"nvrtcCompileProgram failed ({status}): {log.value.decode()}"
            )
        return elapsed
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(program))


def measure(args, toolkit, directory):
    src = args.source.resolve()
    source = src.read_bytes()
    name = os.fsencode(src.name)
    lib = (
        ctypes.CDLL(str(toolkit / "lib64/libnvrtc.so"))
        if args.compiler == "nvrtc"
        else None
    )
    host = {"gcc": "g++-15", "clang": "clang++-22"}.get(args.compiler)
    sides = {"before": args.before.resolve(), "after": args.after.resolve()}
    samples = {side: [] for side in sides}
    for include in sides.values():
        if not (include / "cuda/std/ranges").is_file():
            raise ValueError(f"{include} must be a libcudacxx/include directory")

    for trial in range(args.repeats + 1):
        for side in sides if trial % 2 == 0 else reversed(sides):
            options = [
                f"-std=c++{args.std}",
                "-arch=sm_120a",
                f"-I{sides[side]}",
                f"-I{src.parent}",
            ]
            if lib is not None:
                elapsed = nvrtc_cpu(
                    lib,
                    source,
                    name,
                    [*options, f"-I{toolkit / 'include'}", "-no-cache"],
                )
            else:
                elapsed = nvcc_cpu(
                    [
                        toolkit / "bin/nvcc",
                        f"-ccbin={host}",
                        *options,
                        "-O2",
                        "-c",
                        src,
                        "-o",
                        directory / "probe.o",
                    ]
                )
            if trial:
                samples[side].append(elapsed)

    return {
        "benchmark": src.stem,
        "compiler": args.compiler,
        "std": args.std,
        "cpu_seconds": samples,
        "median_seconds": {side: median(times) for side, times in samples.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Benchmark .cu translation unit")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--compiler", required=True, choices=["gcc", "clang", "nvrtc"])
    parser.add_argument("--std", required=True, choices=["17", "20"])
    parser.add_argument("--toolkit", type=Path, default=Path("/usr/local/cuda-13.5"))
    parser.add_argument(
        "--repeats", type=int, default=5, help="Measured pairs after one warm-up pair"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with tempfile.TemporaryDirectory() as directory:
        print(
            json.dumps(measure(args, args.toolkit.resolve(), Path(directory)), indent=2)
        )


if __name__ == "__main__":
    main()
