<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Rank reduction owned by group_as

Experiment date: 2026-11-12. Original CUB baseline:
`24ee635e35caa47b1602e129e49cea60db8b4f05`. The previous public-group control
is the frozen 2026-11-11 implementation recorded in [README.md](README.md).

The group-enabled implementation now performs its rank reduction only in
`group_as`. It passes correctness and lint checks, but **does not match the
previous implementation's performance**: the 36-case geometric-mean time change
is +1.521% relative to original CUB, or +1.156% with the optional group scan.
Relative to the previous public-group controls, the changes are +1.376% and
+0.910%. These aggregate values are point estimates.

## Implementation

- Group-enabled `getSpecialRegisters()` computes `threadIdx.x / 32` directly.
  The legacy configuration retains its existing reduction.
- For valid warp-sized or larger units on SM90+, `group_as` applies
  `__reduce_min_sync(__activemask(), unit_rank)` before selecting a subgroup.
  This preserves the equal rank while making uniformity visible to the compiler.
  Thread-level mappings and older architectures retain their existing path.
- Static and dynamic squad dispatch select descriptors using the group ordinal.
  Dynamic dispatch preserves exclusion of prefix/suffix groups.
- CUB no longer calls `with_unit_rank(...)` for either parent or leaf views.
  The public helper itself remains defined; this change does not remove APIs.

The active mask supports construction through separately scheduled control-flow
paths. The reduction occurs after invalid-input propagation and does not add
collectives to ordinary `rank()` queries.

## SASS and resources

| Complete benchmark | Original CUB | Previous public groups | Mapping-owned rank |
| --- | ---: | ---: | ---: |
| Main instruction rows | 13,216 | 13,216 | 13,480 |
| Optional scan instruction rows | — | 13,536 | 13,808 |
| Main BSSY / BSYNC pairs | 67 | 69 | 69 |
| Optional BSSY / BSYNC pairs | — | 67 | 67 |
| Indirect branches | 0 | 0 | 0 |
| Stack / spill stores / spill loads | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Each of the eight scan specializations contains one `REDUX.MIN`, whose destination
is a uniform register. The original initial reduction has not been duplicated.
Sum kernels use 86–89 registers in both group implementations. Main affine
kernels use 93–96 registers, compared with 95–96 previously.

The remaining overhead includes active-mask collection and materialization of
the group ordinal and local prefix before the descriptor tree. Removing the
initial reduction alone, while continuing to dispatch on the raw physical warp
rank, caused substantial extra reconvergence in an isolated probe. Dispatching
on the mapping's ordinal prevents that regression. It still leaves more mapping
and dispatch instructions than the previous physical-prefix implementation.
Static instruction counts do not directly measure executed work or runtime.

## Timing

RTX PRO 6000 Blackwell, SM120, CUDA 13.5.37/GCC13, C++17, `-O3 -arch=sm_120`.
Ten balanced rounds cover 36 cases across five variants: original CUB, previous
public groups with/without optional scan, and current groups with/without it.
There are 1,800 timing rows. The harness, warm-up, graph replay, correctness
checks, and bootstrap procedure are unchanged from the previous report.

Selected float32 exclusive results, percent time change:

| Path / size | Versus original CUB, 95% interval | Versus previous public groups, 95% interval |
| --- | ---: | ---: |
| Main / 1K | +2.549% [1.780, 2.799] | +2.366% [1.626, 2.825] |
| Main / 65K | +1.127% [0.524, 2.881] | +0.097% [-0.768, 1.462] |
| Main / 1M | +1.519% [1.336, 1.816] | +1.603% [1.171, 1.746] |
| Optional / 1K | +2.035% [1.513, 2.663] | +1.694% [1.547, 2.533] |
| Optional / 65K | +1.857% [1.393, 2.361] | +1.316% [0.491, 2.229] |
| Optional / 1M | +1.408% [1.214, 1.834] | +1.397% [1.176, 1.585] |

All cases and confidence intervals are in
[comparison.csv](results/mapping-owned-rank/comparison.csv). Full source,
reference-overlay and binary provenance is in
[metadata.json](results/mapping-owned-rank/metadata.json). Raw timings, resource
counts and opcode histograms are in [results/mapping-owned-rank](results/mapping-owned-rank/).

## Validation

Both required matrices pass in C++17: CUDA13.5/GCC13 on SM120 and CUDA12.0/GCC7
using compute90 PTX JIT on the RTX PRO 6000. Tests cover static/runtime warp
partitions `[1, 2, 3]`, multidimensional hierarchy indexing, divergent construction
paths, composed mappings, and invalid nonexhaustive results. Existing debug squad
layouts cover single/uniform/uneven partitions, 1,024 threads, reversed IDs, and
subrange dispatch. Shifted-output scan checks and the forced dynamic RLE
lookahead smoke pass. The benchmark also checks outputs, in-place scans and
noncommutative affine scans.

CUDA12 scan checks use the legacy/header path: that toolkit cannot compile the
Blackwell/PTX8.6 lookahead body. Actual group/squad tests execute its compute90 PTX.
Full edited-header host/device clang-tidy, mapping-test device clang-tidy, and
applicable pre-commit hooks pass. Every NVCC invocation uses an isolated TMPDIR.
Detailed validation manifests and logs are under
`build/lookahead-groups/group-as-only-validation/`.

The relocation establishes uniformity inside the mapping. Eliminating the
remaining prefix/descriptor dispatch overhead is still needed for baseline
performance.

## Rerun in this workspace

The frozen previous-public control is under
`build/lookahead-groups/mapping-owned-rank/prior-public-headers/`; its manifest
is verified by the runner. It contains the prior CUB and cooperative-group
headers from `0662ef1152`. Other dependencies are covered by the complete source
hashes in the metadata. To preserve the recorded results, write a rerun to a
different results directory:

```bash
python3 cub/benchmarks/experiments/lookahead_groups/benchmark.py \
  --build-dir build/lookahead-groups/mapping-owned-rank \
  --results-dir build/lookahead-groups/mapping-owned-rank/rerun-results \
  --prior-headers build/lookahead-groups/mapping-owned-rank/prior-public-headers \
  --variants baseline prior_static_groups prior_static_groups_scan static_groups static_groups_scan \
  --rounds 10
```
