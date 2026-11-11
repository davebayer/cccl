<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

> Archived cohort measured before the private shuffle implementation include was replaced by the public algorithm umbrella. The active report is [README.md](../../README.md); exact historical headers and binaries are identified in [archive.json](archive.json).

# CUB lookahead scan using public cudax group APIs

Experiment date: 2026-11-11. Original CUB baseline:
`24ee635e35caa47b1602e129e49cea60db8b4f05`.

CUB now uses public group APIs throughout its group-enabled implementation.
Private mapping types, synchronizer-instance types and metadata getters are
removed. All **16 scan kernel bodies have exactly the same SASS instructions and
operands as the previous private-API implementation**: 13,216 instruction rows
for the main path and 13,536 with the optional group shuffle scan. In these kernels, public APIs
add no device instructions, registers, barriers, stack or spill traffic.

In the fresh 36-case sweep, aggregate time changes relative to original CUB are
**-0.318%** for public groups and
**-0.241%** with the optional scan. The private controls
measure -0.309% and -0.076%.
These are point estimates across an equally weighted matrix; per-case confidence
intervals below describe remaining differences from original CUB.

## Public group support

[The group view API](../../../../../../cudax/include/cuda/experimental/coop/__group/group_view.cuh)
provides these operations:

| API | Purpose |
| --- | --- |
| `group_view_t<Unit, Group>` | Name a view using public types, including in nvcc's host pass. |
| `view.with_static_unit_count<N>()` | Refine a known count in the view's units without callback dispatch. |
| `view.with_unit_rank(rank)` | Supply an equivalent cached rank while preserving its compiler properties. |
| `View::static_unit_count()` | Query the compile-time unit count, or `dynamic_extent`. |

Count and rank refinements preserve membership, group ordinal, hierarchy, masks
and synchronization state. They flatten view types rather than nesting them.
Invalid nonexhaustive views remain invalid. For valid views, the supplied count
or rank must equal the original query; debug assertions check those promises.
They provide cached equivalent queries and do not define a new partition.

Type-only mapping queries and explicit existing rank return types make public
view aliases work in strict nvcc host compilation, including conversions of
nonuniform grouped warps to thread views. These changes preserve runtime types
and values.

[WarpSpeed's static dispatcher](../../../../../cub/detail/warpspeed/squad/squad.cuh)
keeps its balanced physical-prefix selection before count specialization. It
uses the new public operations to retain the cached warp-uniform parent rank
and the selected descriptor's static count/local rank. Existing public
`group_as`, `make_group`, identity mapping and native synchronizers construct
the actual groups. Rank, count and synchronization queries use those views.

Dynamic squads are templated on their public view type. RLE's callback accepts
that type, removing the need for a concrete alias containing private mapping
and synchronizer types. Descriptor-array forwarding uses public `data()`.
No group-private `__` names or `array.__elems_` access remain in this CUB code.
The default CUB implementation remains available for the control comparison.

## SASS evidence and experiments

| Complete benchmark | Original CUB | Private groups | Public groups |
| --- | ---: | ---: | ---: |
| Main instructions | 13,216 | 13,216 | 13,216 |
| Optional group scan instructions | — | 13,536 | 13,536 |
| Sum registers/thread | 87–89 | 86–89 | 86–89 |
| Named barriers per scan kernel | 3 | 3 | 3 |
| Stack / spill stores / spill loads | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The literal private/public comparison covers all eight specializations in each
configuration, including noncommutative affine scans. Equal totals with original
CUB do not imply identical original code: float32 inclusive and int32 exclusive
retain eight extra instructions each, offset by affine savings. The earlier
23-instruction group/role dispatch remains reduced to the original five-instruction
shape for the representative int32 inclusive kernel.

Rejected experiments are retained only as evidence:

- A temporary cached thread view using `threadIdx - 32 * prefix` reduced code size
  and helped 1K, but regressed 1M and did not remove the 65K gap.
- Caching the original thread mask for aligned power-of-two groups produced no
  clear speedup sufficient to justify extra CUB logic.
- Explicitly folding singleton mapping ranks to zero produced identical scan
  SASS, showing that this consumer already obtains that optimization.

None of these experimental changes remain in production. Their separately
frozen headers, commands, results and hashes are under
`build/lookahead-groups/public-cub-{rank,mask,singleton}-*`.
Per-kernel counts, opcode histograms, resources and literal comparison evidence
are in [results/public-api](.).

## Measurements

Hardware: RTX PRO 6000 Blackwell Workstation Edition, SM120, 188 SMs and 128 MiB
L2. All GPU executions run outside the sandbox. Compilation uses latest installed
CUDA 13.5.37 and explicit GCC 13.3 (`/usr/bin/g++-13`), C++17,
`-O3 -arch=sm_120 -lineinfo`. GPU clocks and power settings are unchanged. Every
nvcc invocation has an isolated temporary directory.

The same [harness](../../scan.cu) builds original CUB, private groups, private groups
with optional scan, public groups and public groups with optional scan. A complete
2,907-header snapshot pins every private reference dependency. A portable
[reference patch](../../private-before-public.patch) reconstructs it from the pinned
baseline. Live header and runner/harness hashes are checked before and after
compilation/timing. Full, focused and profiled binaries have matching hashes.

The matrix covers int32/float32/float64 inclusive and exclusive sum, five aligned
sizes and a shifted-pointer tail case: 36 cases per variant. Guards, exact
floating-point references, in-place checks and forced three-warp noncommutative
lookahead validate outputs.

Each timed case receives at least 50 ms of GPU graph execution warm-up. Graphs
contain 64 initialization/scan pairs at sizes up to 2^20 and eight for larger
cases. Nine event batches give one process median. Ten rounds rotate and reverse
the five variants so every variant occupies each position twice. Full results
contain 1,800 records and 180 summaries.

Separate float32 exclusive trials at 1K, 65K and 1M have 40 rounds: 600 records
and 15 summaries. Confidence intervals use 2,000 paired-round bootstrap resamples
of ratios of process medians. They describe observed run variation and do not
cover systematic clock/cache effects. Allocation, queries, generation and checking
are outside timing. Repeated small buffers can reside in cache.

Speedup is original time divided by variant time; below one is slower. Size rows
geometrically average the three types and two operations; the aggregate gives
every case equal weight.

| Elements | Shift | Private | Private + scan | Public | Public + scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0 | 1.0016x | 0.9982x | 1.0003x | 0.9991x |
| 65,536 | 0 | 0.9993x | 1.0023x | 1.0009x | 1.0020x |
| 1,048,576 | 0 | 1.0015x | 1.0006x | 1.0009x | 1.0009x |
| 1,048,579 | 1 | 1.0080x | 0.9961x | 1.0136x | 1.0050x |
| 16,777,216 | 0 | 0.9999x | 0.9990x | 0.9996x | 0.9989x |
| 67,108,864 | 0 | 1.0084x | 1.0085x | 1.0039x | 1.0085x |
| **All 36 cases** | | **1.0031x** | **1.0008x** | **1.0032x** | **1.0024x** |

Forty-round float32 exclusive time changes versus original, with 95% bootstrap
intervals in brackets. Positive values are slower:

| Elements | Private | Private + scan | Public | Public + scan |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | +0.123% [-0.045, +0.473] | +0.135% [-0.034, +0.495] | +0.168% [+0.022, +0.529] | +0.146% [+0.000, +0.540] |
| 65,536 | +1.162% [+0.959, +1.425] | +0.532% [+0.427, +0.749] | +1.192% [+1.071, +1.395] | +0.547% [+0.435, +0.899] |
| 1,048,576 | -0.339% [-0.385, -0.135] | -0.058% [-0.175, +0.064] | -0.257% [-0.368, -0.111] | -0.058% [-0.105, +0.105] |

All six public-versus-private confidence intervals include zero, consistent with
the literal shader identity. Original CUB is still faster in the focused 65K case:
public groups retain 1.192% overhead, and the optional scan retains
0.547%. These gaps also occur with the private implementation and
were not eliminated by the rejected rank experiments. The result matches
aggregate baseline performance, rather than establishing parity for every case.

The [full comparison](comparison.csv) and
[focused comparison](focused-repeatability/comparison.csv)
also compare public groups directly with their private controls. Identical shader
bodies are stronger evidence of zero API instruction overhead than small timing
differences between these two variants.

## Profiling

Nsight Compute executed warp-instruction totals for float32 exclusive scan:

| Elements | Original | Private | Public |
| ---: | ---: | ---: | ---: |
| 1,024 | 3,979 | 3,958 | 3,956 |
| 65,536 | 37,481 | 37,742 | 38,219 |
| 1,048,576 | 612,499 | 607,798 | 624,921 |
| 67,108,864 | 85,719,570 | 85,823,489 | 85,520,263 |

Registers remain 89 used / 96 allocated, dynamic shared memory remains 98,656
bytes, and the shared-memory residency limit remains one CTA for this kernel.
Public APIs do not change occupancy. Counts, particularly for large inputs,
vary with CTA scheduling and lookahead polling. Profiler replay durations are
diagnostic; event timings above supply the performance comparison.
See [profiling.csv](profiling.csv) and its binary-hash metadata.

## Validation and scope

Both compiler matrices use C++17: CUDA13.5/GCC13 and CUDA12.0/GCC7. New public API
checks cover strict namespace aliases, unequal native warp partitions,
noncontiguous lane masks, invalid sentinels, repeated refinements, and native
grid/cluster synchronization. Existing static-count tests also pass both matrices.

CUB checks pass 26,586 assertions across 12 scan cases and 3,331 assertions across
41 RLE cases, plus shifted-output regressions, warp lengths 0–32, forced RLE
lookahead, in-place/noncommutative scans, and static/dynamic squad layouts through
1,024 threads with reversed descriptor IDs. Huge-input RLE stress cases are
excluded from the targeted run. CUDA12 cannot compile Blackwell/PTX8.6 lookahead;
its scan tests cover legacy/header compatibility, while actual group tests run
compute90 PTX through the RTX driver's JIT.

All edited product headers pass full host/device clang-tidy; CUDA tests and
applicable pre-commit hooks pass. The final measured optional binary passes
compute-sanitizer memcheck. RLE's full-header lint cleanup preserves its atomic,
mask and index behavior and is covered by the RLE cases above.
Detailed commands and limitations are in
[validation-cub.txt](validation-cub.txt) and
[validation-group-api.txt](validation-group-api.txt).

The adapter remains an opt-in experiment. Define these before CCCL headers and
add `-Icudax/include`:

```text
-D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX
-D_CUB_WARPSPEED_USE_CUDAX_GROUPS
-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS
```

Add `-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN` for the optional shuffle scan. Static squads
cover whole one-dimensional blocks with full warps and use native barrier IDs
1–15; zero is reserved for block synchronization. Launch policies, tile sizes,
pipeline stages and numerical algorithms are unchanged.

## Reproduce

From the repository root, with GPU execution outside the sandbox:

```bash
CUDA_VISIBLE_DEVICES=0 python3 cub/benchmarks/experiments/lookahead_groups/benchmark.py \
  --nvcc /usr/local/cuda-13.5/bin/nvcc --cxx /usr/bin/g++-13 --rounds 10 \
  --build-dir build/lookahead-groups/public-api-final \
  --results-dir cub/benchmarks/experiments/lookahead_groups/results/public-api \
  --variants baseline private_groups private_groups_scan static_groups static_groups_scan
```

Repeat with `--run-only --rounds 40`, the three `--case float32 exclusive N`
arguments for 1024, 65536 and 1048576, and the `focused-repeatability` results
subdirectory. [profile.py](../../profile.py) reproduces diagnostics.
The prior implementation, SASS investigation and datasets remain intact in
[PRIVATE_API_REPORT.md](../../PRIVATE_API_REPORT.md) and the older results directories.
