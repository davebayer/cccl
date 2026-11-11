<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# CUB lookahead scan with cudax groups: further SASS refinement

Experiment date: 2026-11-11. Original CUB baseline: `24ee635e35caa47b1602e129e49cea60db8b4f05`.

Selecting the static squad descriptor **before** specializing its group count
removes redundant mapping/dispatch work. The refined benchmark has **13,216 SASS
instructions**, the same total as original CUB, versus 13,472 for the previous
static adapter. Native cudax group views still perform rank queries and synchronization.
The generic `group_as` algorithm is unchanged.

Across 36 same-run cases, geometric-mean time changes relative to original CUB
are **-0.16%** for the refinement and
**-0.37%** with the optional group shuffle scan.
The corresponding previous implementations measure
+0.53% and +0.56%.
These are point estimates; they establish neither an overall speedup nor parity
for every input. The focused float32 measurements below quantify the remaining
small-case differences.

## Inefficient code and retained change

Earlier fixes preserved WarpSpeed's cached warp-uniform rank and balanced the
count-specialization tree. In the representative int32 inclusive kernel, this
removed the initial adapter's extra warp collectives and reduced reconvergence
pairs from 50 to the original eight.
The starting implementation here already includes those fixes; its archived
results use an older warm-up protocol.

The previous static path did three selections: `group_as` materialized a runtime
group rank and warp prefix; `with_static_unit_count` selected a count-specific
view; a second descriptor dispatch selected the scan role. The heavy scan callback
made the resulting combinations difficult for nvcc to fold.

For int32 inclusive scan, the prior mapping/dispatch prelude has 23 static
instructions. Reducer warps execute ten uniform instructions and scanner warps
execute fourteen, compared with five each in original CUB. The avoidable sequence
includes `UMOV` prefix constants, `USEL` selection and a dependent `UIADD3` rank
reconstruction. It precedes squad memory work.

[The static dispatcher](../../../cub/detail/warpspeed/squad/squad.cuh) now:

1. Keeps the cached warp-uniform parent rank and actual `group_as` partition.
2. Selects the descriptor using a balanced tree of physical warp-prefix thresholds.
3. Binds that leaf's count and local rank from its `index_sequence` constants.
4. Constructs the same native synchronized group and invokes the existing callback.

The count is already known at the selected leaf, so the outer
`with_static_unit_count` callback is unnecessary. Debug assertions compare the
selected descriptor and local rank against the original mapping. Descriptor IDs
remain separate from their array positions; reversed-ID tests cover this.
This avoids changing generic mapping performance to optimize this one consumer.

| Complete benchmark code | Original | Previous groups | Rank-only experiment | Refined |
| --- | ---: | ---: | ---: | ---: |
| Main SASS instructions | 13,216 | 13,472 | 13,448 | 13,216 |
| Optional shuffle-scan instructions | — | 13,824 | 13,808 | 13,536 |
| Indirect branches in either group variant | — | 0 | 0 | 0 |

Counts cover eight scan specializations plus shared helper kernels and include
alignment NOPs. Float32 inclusive and int32 exclusive each retain eight extra
instructions; savings in the two affine kernels offset those. Main reconvergence
pairs total 69 versus the baseline's 67. Equal totals do not imply identical
instructions or equal runtime.
Per-kernel counts, opcode histograms and annotated paths are in
[sass-kernels.csv](results/sass-kernels.csv),
[sass-opcodes.csv](results/sass-opcodes.csv) and
[sass-paths.txt](results/sass-paths.txt).

Some rank-conversion work remains: a warp-unit group forms thread rank with
`USHF` → uniform-to-vector `MOV` → lane-bit `LOP3`, where original CUB can mask
`threadIdx.x` directly for power-of-two squads. Additional rank caches increased
register use to 128 in an affine kernel or expanded callback code, so they were
rejected. This is a residual instruction dependency, not proof that it explains
every measured time difference.

A binary generic mapper also looked promising in one configuration but expanded
the optional scan to 35,464 instructions and twenty indirect branches. Explicit
role tags repaired that combination. The selected structural approach produces
better code without changing generic mapping or adding role indices; both
alternatives were rejected. An early prototype comparison failed to freeze all
dependent headers; its attribution was corrected. Final results use complete source hashes,
pinned reference snapshots and freshly verified binaries.

## Benchmark method

Hardware: RTX PRO 6000 Blackwell Workstation Edition, SM120, 188 SMs and 128 MiB
L2. GPU programs execute outside the sandbox. Compiler: latest installed CUDA
13.5.37 with explicit GCC 13.3 (`/usr/bin/g++-13`), C++17,
`-O3 -arch=sm_120 -lineinfo`. Clocks and power limits are unchanged. Each nvcc
invocation uses a private temporary directory.

The same [harness](scan.cu) builds five variants:

| Variant | Behavior |
| --- | --- |
| baseline | Original committed CUB |
| prior_static_groups | Static groups at the start of this refinement |
| prior_static_groups_scan | Previous groups plus optional group shuffle scan |
| static_groups | Refined physical-prefix dispatch and native groups |
| static_groups_scan | Refinement plus optional group shuffle scan |

All variants use the new common harness. Each case receives at least 50 ms of
GPU graph execution for warm-up, replacing the earlier three short graph replays.
Graphs contain 64 initialization/scan pairs at sizes up to 2^20 and eight for
larger cases. Nine event batches yield one process median. Ten process rounds
rotate and reverse the five variants, giving each every position twice. Reported
times are medians of process medians: 1,800 rows and 180 summaries.

The matrix covers inclusive/exclusive sum for int32, float32 and float64; five
aligned sizes and one shifted-pointer tail case. Exact floating-point references,
output guards, in-place checks and forced noncommutative three-warp lookahead
checks validate correctness. Allocation, generation, temporary-storage queries
and checking are outside timing.

The two small float32 exclusive cases also have a separate 40-round run
(400 rows). Confidence intervals are percentile bootstrap estimates from 2,000
paired-round resamples of the ratio of process medians. They describe observed
run variation and do not cover systematic clock/cache effects. Effective bandwidth
counts input and output payload only; repeated small buffers can reside in L2.

[metadata.json](results/metadata.json) records compiler/GPU details, process order,
source and binary hashes, and reference provenance. The portable
[prior fixture](static-before-refinement.patch) restores the exact starting headers;
[the older fixture](static-before-fix.patch) retains the earlier experiment.
The default CUB control still has the baseline's instruction count and ordering,
with only the eight previously recorded signedness comparisons differing; see
[control.txt](results/control.txt).

## Timings

Speedup is original time divided by variant time; values below one are slower.
Each size row averages the three types and both operations geometrically. The
36-case aggregate weights all cases equally, rather than modeling an application.

| Elements | Shift | Previous | Previous + scan | Refined | Refined + scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0 | 0.9833x | 0.9852x | 1.0013x | 0.9999x |
| 65,536 | 0 | 0.9878x | 0.9879x | 0.9964x | 0.9969x |
| 1,048,576 | 0 | 0.9899x | 0.9894x | 1.0013x | 0.9996x |
| 1,048,579 | 1 | 0.9990x | 0.9932x | 1.0012x | 1.0065x |
| 16,777,216 | 0 | 0.9979x | 0.9998x | 0.9986x | 0.9997x |
| 67,108,864 | 0 | 1.0104x | 1.0115x | 1.0108x | 1.0199x |
| **All 36 cases** | | **0.9947x** | **0.9945x** | **1.0016x** | **1.0037x** |

Representative exclusive times in microseconds:

| Type | Elements | Original | Previous | Previous + scan | Refined | Refined + scan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| int32 | 1,024 | 2.154 | 2.164 | 2.171 | 2.138 | 2.159 |
| float32 | 1,024 | 2.240 | 2.285 | 2.267 | 2.243 | 2.245 |
| float64 | 1,024 | 3.869 | 3.930 | 3.936 | 3.872 | 3.869 |
| float32 | 1,048,576 | 4.280 | 4.343 | 4.388 | 4.271 | 4.288 |
| float64 | 16,777,216 | 180.092 | 180.762 | 180.538 | 180.650 | 180.466 |
| int32 | 67,108,864 | 364.838 | 364.396 | 364.474 | 364.952 | 364.612 |

Forty-round float32 exclusive time differences relative to original, with 95%
bootstrap intervals in brackets; positive values are slower:

| Elements | Previous | Previous + scan | Refined | Refined + scan |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | +1.693% [+1.592, +2.590] | +1.402% [+1.133, +1.846] | -0.011% [-0.168, +0.428] | +0.000% [-0.292, +0.451] |
| 1,048,576 | +1.636% [+1.518, +1.818] | +2.715% [+2.579, +2.950] | -0.018% [-0.328, +0.023] | +0.311% [+0.135, +0.381] |

The main-path estimates are compatible with original performance at both focused
sizes. The optional scan retains a measured 0.311% slowdown at
1M elements, with its interval above zero. That configuration has a small
remaining cost even though its aggregate point estimate is favorable.

The per-case [comparison.csv](results/comparison.csv) and
[focused comparison](results/focused-repeatability/comparison.csv) also compare
refinement directly with the corresponding previous implementation. Full timings
are in [raw.csv](results/raw.csv) and [summary.csv](results/summary.csv).

## Profiling and resources

Nsight Compute diagnostics for float32 exclusive scan at 1,024 elements:

| Variant | Executed warp instructions | Registers/thread: used / allocated | Dynamic shared bytes | CTA limit from shared memory |
| --- | ---: | ---: | ---: | ---: |
| original | 3,977 | 89 / 96 | 98,656 | 1 |
| previous | 4,133 | 89 / 96 | 98,656 | 1 |
| refined | 3,958 | 89 / 96 | 98,656 | 1 |

The unchanged allocation and residency support a dispatch/dependency explanation,
rather than an occupancy improvement. Executed instruction totals, especially on
large inputs, vary with CTA scheduling and lookahead polling. Profiler replay
durations are diagnostic; the CUDA-event results above supply timing comparisons.
See [profiling.csv](results/profiling.csv) and [profiling.json](results/profiling.json).

All final sum kernels use 86–89 registers per thread, three named barriers, zero
stack and zero spill loads/stores; original sums use 87–89 registers. Resources
for all eight specializations are in [codegen.csv](results/codegen.csv).

## Scope and validation

Enable the adapter before including CCCL headers and add `-Icudax/include`:

```text
-D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX
-D_CUB_WARPSPEED_USE_CUDAX_GROUPS
-D_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS
```

`-D_CUB_LOOKAHEAD_USE_CUDAX_SCAN` additionally enables the experimental
order-preserving shuffle warp scan, including operators without a declared
identity and partial tiles. The scan partition remains
`index_sequence<R, R, 1, 1, 1>`. Launch policies, pipeline stages and numerical
algorithms were not retuned. RLE continues to use dynamic dispatch through the
shared generic helpers.

The experiment assumes full warps covering one-dimensional blocks and valid
native barrier IDs 1–15 (0 is reserved for block synchronization). Static group
views use internal cudax mapping types; that coupling needs review before upstream integration. Consistent definitions
across translation units are required. These internal switches select the
experimental implementation.

Final validation passed: 26,586 assertions across 12 existing CUB cases; shifted
outputs with both compiler matrices; warp lengths 0–32; noncommutative/in-place
checks; RLE smoke; and debug squads with single/uniform/uneven/1,024-thread layouts
and reversed IDs. Expanded `group_as` boundary tests also pass both toolchains.
The required matrices use C++17: CUDA13.5/GCC13 and CUDA12.0/GCC7. CUDA12 scan
checks cover the legacy path; separate real group tests run compute90 PTX via
JIT on the RTX because CUDA12 cannot compile Blackwell/PTX8.6 lookahead.

Full edited-header host/device clang-tidy, CUDA test lint and applicable pre-commit
checks pass. The final measured optional binary passes compute-sanitizer memcheck.
Commands and limitations are in
[validation-sass-refinement.txt](results/validation-sass-refinement.txt),
[validation-static-map.txt](results/validation-static-map.txt) and
[validation-benchmark.txt](results/validation-benchmark.txt).

The [rank-only reference](results/rank-rebind-reference/) preserves the measured
intermediate result; [uniform-before-refinement](results/uniform-before-refinement/)
preserves the previous protocol/results. Earlier archives remain under
[static-before-fix](results/static-before-fix/) and [dynamic-first](results/dynamic-first/).
Their absolute times should not be mixed with this run's longer warm-up protocol.

## Reproduce

From the repository root, with GPU execution outside the sandbox:

```bash
CUDA_VISIBLE_DEVICES=0 python3 cub/benchmarks/experiments/lookahead_groups/benchmark.py \
  --nvcc /usr/local/cuda-13.5/bin/nvcc --cxx /usr/bin/g++-13 \
  --baseline-ref 24ee635e35caa47b1602e129e49cea60db8b4f05 --rounds 10 \
  --build-dir build/lookahead-groups/structural-final \
  --variants baseline prior_static_groups prior_static_groups_scan \
             static_groups static_groups_scan
```

Repeat with `--run-only --rounds 40 --case float32 exclusive 1024
--case float32 exclusive 1048576` and set `--results-dir` to the
`results/focused-repeatability` subdirectory. `--build-only`, `--run-only`,
`--variants`, `--prior-headers` and `--results-dir` control scope and output.
[profile.py](profile.py) reproduces diagnostic Nsight runs; [squad.cu](squad.cu)
and [warp_scan.cu](warp_scan.cu) reproduce focused group/helper checks.
