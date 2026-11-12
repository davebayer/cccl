<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Original implementation: defects and follow-up work

Audit date: 2026-11-12. Original revision:
`24ee635e35caa47b1602e129e49cea60db8b4f05`. The fixed control is
`0662ef1152`, which contains the experiment before relocating rank reduction
into `group_as`.

Two original compile defects were encountered and fixed. The remaining findings
are compatibility or contract hazards. **No incorrect result from a supported
original CUB scan or RLE configuration was demonstrated.** Performance regressions
introduced while adapting CUB to groups are not counted as original CUB bugs.
The RLE-specific audit below also records a bit-mask portability improvement and
an internal-helper compilation workaround that remains unfixed.

## Confirmed defects already fixed in the experiment

### B1. Public group-view type deduction fails outside device functions

**Trigger:** use existing public group APIs to name a warp view at namespace scope:

```cpp
namespace coop = cuda::experimental::coop;
using H = decltype(cuda::make_hierarchy(cuda::block_dims(128u)));
using B = coop::this_block<H>;
using V = decltype(coop::group_view{cuda::warp, cuda::std::declval<const B&>()});
```

**Observed:** the original headers fail with CUDA 13.5/GCC13 and CUDA 12.0/GCC7,
both in C++17. The diagnostic identifies a device-only `rank()` with a deduced
return type being referenced outside a device function. The expression is a
type query; it does not execute the rank query on the host.

**Cause:** deduction of mapping and view types instantiates device function
bodies to discover their `auto` return types. This makes otherwise valid
namespace-scope type composition depend on device execution context.

**Fix:** provide explicit existing rank return types and compute the view's
mapping-result type through type traits. This preserves the runtime rank types
and values. The fix spans `this_group.cuh`, `generic_group.cuh`,
`virtual_group.cuh`, `coalesced_group.cuh`, and
[group_view.cuh](../../../../cudax/include/cuda/experimental/coop/__group/group_view.cuh).
It can be separated from the experiment's additional public refinement APIs.

**Evidence:** [minimal reproducer](results/original-issues/group-reproducers/view_type.cu),
[original/latest diagnostic](results/original-issues/group-compile/original-view_type-latest.log),
[original/CUDA12 diagnostic](results/original-issues/group-compile/original-view_type-cuda12.log).
The frozen fixed control compiles successfully with both toolchains.

**Follow-up:** retain a namespace-scope CTAD regression case in ordinary group
tests, without relying on a new convenience alias to exercise the old API.

### B2. Grid synchronizer contains ill-formed list narrowing

**Trigger:** include the public group header with Clang's CUDA frontend. An
include-only translation unit is sufficient; grid synchronization need not run.

**Original code:** two sites in `level_synchronizer.cuh` initialize the unsigned
phase contribution with:

```cpp
cuda::std::uint32_t{cuda::std::numeric_limits<int>::min()}
```

The negative constant is not representable in `uint32_t`, so this braced
initialization is ill-formed. Clang 23 rejects the original header in both host
and device passes. **Both tested NVCC versions accept the original code**;
this finding is a portability and clang-tidy blocker, not evidence of a grid
barrier runtime failure.

**Fix:** the two sites now use an explicit `static_cast<uint32_t>`. It expresses
the intended unsigned conversion, retaining the phase-bit value. See
[level_synchronizer.cuh](../../../../cudax/include/cuda/experimental/coop/__group/synchronizer/level_synchronizer.cuh).

**Evidence:** [include-only reproducer](results/original-issues/group-reproducers/include_only.cu),
[original Clang host diagnostic](results/original-issues/group-compile/original-include_only-clang-host.log),
[original Clang device diagnostic](results/original-issues/group-compile/original-include_only-clang-device.log).
The frozen fixed control passes both Clang passes; explicit grid-sync
instantiations also compile with both NVCC toolchains.

**Follow-up:** submit the two casts as an independent portability fix. The added
parentheses around the polling comparison are only a readability change: the
original operator precedence already implements the same comparison.

## Original assumptions that need an explicit contract

### H1. Modulo-based ranks change ordering for unaligned squad starts

Original `Squad::warpRank()` computes `globalWarp % squadWarpCount`, and
`threadRank()` uses the analogous global-thread modulo. For the layout
`[1, 3, 2, 1]`, the second squad starts at physical warp 1:

| Physical warp | Original warp rank | Group-backed warp rank |
| --- | ---: | ---: |
| 1 | 1 | 0 |
| 2 | 2 | 1 |
| 3 | 0 | 2 |

Its original root is physical warp 3; the group-backed root is physical warp 1.
First-lane thread ranks likewise change from `[32, 64, 0]` to `[0, 32, 64]`.
Both numberings are bijections. The original helper does not explicitly promise
physical-order ranks, so this is a migration hazard, not a demonstrated
algorithm bug. Original scan uses aligned `[R, R, 1, 1, 1]` squads; RLE can
distribute its tasks using rotated logical ranks.

**Status:** group-backed squads use ascending ranks. The legacy helper remains
modulo-based. **Follow-up:** decide and document the ordering contract. If
ascending physical order is intended, pass the squad's start through dispatch
and subtract it. Cover unaligned starts, subrange dispatch, and root selection.

**Evidence:** [original output](results/original-issues/warpspeed/original-run.log),
[frozen group output](results/original-issues/warpspeed/groups-run.log).
Both toolchains reproduce the same difference on the RTX PRO 6000.

### H2. WarpSpeed helpers silently assume one-dimensional blocks

`getSpecialRegisters()` derives warp rank from `threadIdx.x / 32`, and dispatch
uses `blockDim.x` as its thread extent. Launching the same 224 threads as
`blockDim = (32, 7, 1)` makes every physical warp select descriptor 0 and report
warp rank 0 in the original helper. The frozen pre-relocation group adapter
also exhibits this behavior. The supported scan/RLE policies launch
one-dimensional blocks, so this does not demonstrate a failure of those policies.

CUDA's physical warp assignment uses linear thread IDs across all block
dimensions. See the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

**Status:** the experimental static dispatcher asserts a covering one-dimensional
layout. Legacy and dynamic entry points still need an explicit documented
restriction or consistent linearization throughout WarpSpeed.

**Follow-up:** either assert one-dimensional, full-warp launches at the shared
entry point, or support multidimensional indexing end to end. Changing only the
initial rank expression is insufficient while extent calculations remain x-only.

**Evidence:** the same original/group output files above include both launch
shapes. The reproducer deliberately avoids squad synchronization on the
unsupported shape.

### H3. Barrier IDs and participation have unchecked preconditions

Original `Squad::syncThreads()` uses `mSquadIdx + 1` as the named barrier ID and
reserves ID 0 for block synchronization. Hardware provides IDs 0 through 15;
explicit participant counts must be multiples of the warp size. See the
[PTX barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier).

Thus, squad IDs must be in `[0, 14]`; independently synchronizing active squads
must not share a barrier ID. The original full-mask reduction/election also
requires the corresponding warp lanes to participate. Current scan/RLE
descriptors and construction sites satisfy these requirements.

**Status:** these are existing preconditions, not failures observed in supported
kernels. Invalid barrier use was not run on the GPU. **Follow-up:** document
the requirements and add debug validation for IDs, simultaneous ID reuse,
coverage, and full-warp counts when generalizing the helper.

## RLE-specific changes

### R1. Bit-preserving arguments to __ffs

Three RLE sites now pass `cuda::std::bit_cast<int>(mask)` to `__ffs`, replacing
an implicit conversion from `unsigned`. A mask containing bit 31, such as
`0x80000000u`, is not representable as a positive `int`. The original conversion
is implementation-defined in C++17; it is not undefined behavior. See the
[C++17 integral conversion rules](https://timsong-cpp.github.io/cppwp/n4659/conv.integral).

**Fix:** make the intended 32-bit representation explicit before finding the
first set bit. This is a signedness/portability improvement in
[kernel_rle_encode_lookahead.cuh](../../../cub/device/dispatch/kernels/kernel_rle_encode_lookahead.cuh),
covering the decoded head word, first nonempty chunk, and first head position.
The original CUDA implementation already produces the intended results for these
masks; no incorrect original RLE output was demonstrated.

### R2. Other RLE edits preserve existing behavior

| Edit | Reason and correctness implication |
| --- | --- |
| Callback parameter `warpspeed::Squad` becomes `auto` | Accepts the group-backed squad type. This is integration work. |
| `lane_runs_before_word{}` | Defensive initialization for lint. `WarpScan::ExclusiveSum` writes the output in every lane, including zero in lane 0, before it is read. The original code does not have a demonstrated uninitialized read here. |
| `const atomic_ref` handles | The referenced state remains mutable. Device scope and relaxed memory ordering are unchanged. |
| Explicit integer casts | State conversion intent. The casts do not repair overflow; existing policy bounds keep the affected counts representable. |
| Merge nested wait/arrival conditions | Preserve the same predicates and synchronization calls. This does not fix a demonstrated race or barrier error. |
| Remove redundant `typename`, split declarations, move member initialization | Mechanical cleanup for full-header lint. |

The scan-output guarantee is directly implemented in
[WarpScan::ExclusiveSum](../../../cub/warp/warp_scan.cuh): it supplies a
zero-initialized initial value and writes the output through `ExclusiveScan`.
In particular, adding `{}` should not be described as fixing an original RLE
uninitialized-variable bug.

Existing RLE validation passed 3,331 assertions across 41 targeted cases;
huge-input stress cases were excluded. A separate forced-lookahead smoke covers
shifted buffers, output guards, tail lengths, and run counts. That separate
check matters because the default SM120 RLE policy selects lookback. These
checks establish the tested behavior; no RLE performance comparison was made.
See [validation-cub.txt](results/public-api/validation-cub.txt).

### R3. Internal RLE launch-bounds instantiation needs relaxed constexpr

The forced-lookahead smoke invokes internal RLE dispatch directly. With CUDA
13.5/GCC13 in C++17, both the original and current implementations fail to compile
this harness without `--expt-relaxed-constexpr`; adding that flag succeeds.

The diagnostic identifies `get_device_rle_encode_lookahead_launch_bounds()`, a
host/device constexpr helper, calling device-only `current_policy()`. A minimal
namespace-scope instantiation of the library's launch-bounds variable reproduces
the same issue with the built-in policy selector, without the smoke's custom
selector. This isolates an execution-space mismatch in the internal helper.

**Scope:** the tested public `DeviceRunLengthEncode::Encode` calls compile without
the flag, including a tuned-policy call. Default public calls also compile for
both SM100 and SM120. Therefore this is an internal-helper/direct-dispatch
compilation hazard, not a demonstrated failure of the supported public RLE API.

**Status:** the experiment used the flag as a harness workaround; it did not fix
the helper. This dependency should have been included in the original hazard
report. **Follow-up:** make the host/device launch-bounds calculation use
host/device constexpr policy evaluation rather than calling a device-only
function, and retain a strict compilation regression case.

An isolated candidate replaces `current_policy<PolicySelector>().lookahead` in
the helper with:

```cpp
select_policy<PolicySelector>(current_tuning_cc()).lookahead
```

Both functions in the replacement are host/device constexpr. This candidate
passes strict compilation for the internal smoke, namespace bounds query,
public default/tuned controls, and SM100 default lookahead. It is **not applied
to the working implementation** and has not received GPU or Clang validation.

**Evidence:** [minimal bounds query](results/original-issues/rle-compile/bounds.cu.txt),
[original strict diagnostic](results/original-issues/rle-compile/original-bounds-strict.log),
[internal/default compile matrix](results/original-issues/rle-compile/manifest.json),
[public tuning matrix](results/original-issues/rle-compile/custom-public-manifest.json),
[groups-enabled public control](results/original-issues/rle-compile/groups-manifest.json),
and [candidate compilation results](results/original-issues/rle-compile/candidate-manifest.json).
These are compile-only checks; no additional GPU execution was needed.

## Other maintenance change, not an observed defect

The original `cuda::std::array` dispatch overload accesses `squads.__elems_`.
The experiment replaces it with public `data()`, removing dependence on a
libcudacxx implementation detail. Neither tested original toolchain fails here.
Templating the shared-memory helpers enables group-backed squads; it does not
repair a demonstrated original algorithm bug.

## Reproduce and split the fixes

From the repository root, run:

```bash
python3 cub/benchmarks/experiments/lookahead_groups/results/original-issues/group-reproducers/verify.py \
  --root . --output build/original-group-issues
```

The script extracts the original and fixed cooperative-group headers from Git,
uses a separate temporary directory per NVCC process, and checks 16 expected
before/after results. Compiler paths are explicit: CUDA 13.5/GCC13,
CUDA 12.0/GCC7, and Clang 23. All C++ compilation uses C++17.

[verification.json](results/original-issues/group-compile/verification.json)
records revisions, compiler versions, commands, exit codes, and source hashes.
The [WarpSpeed evidence manifest](results/original-issues/warpspeed/manifest.json)
records the separate GPU reproducer commands and hashes; its
[source](results/original-issues/warpspeed/rank_repro.cu.txt) is preserved verbatim.
CUDA12 GPU checks use compute90 PTX JIT on the RTX PRO 6000.

The two compile fixes are independent candidates for small follow-up changes.
The three contract findings require an API decision before treating their
observed behavior as a correctness defect.
