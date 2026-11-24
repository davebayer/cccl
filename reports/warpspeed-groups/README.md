# WarpSpeed experimental groups on RTX PRO 6000

## Short scan latency

WarpSpeed initializes one shared-memory barrier per resource, stage and phase. All phases of a resource have the same stage count. Having each thread initialize every phase of its assigned stage removes repeated loop and reconvergence instructions at block startup. For u32 and u64 the number of SM120a instructions before the first block barrier fell from 158 to 117 and from 160 to 119, respectively. Both kernels still use 88 registers per thread with zero spills.

We compared the original implementation at `4db67c68d9`, the groups implementation after the register fix below, and the new initialization order using NVCC 13.5, GCC 14 and the RTX PRO 6000. The table gives the range of paired median latency reductions versus the original across aligned, shifted and in-place layouts. Each sample timed a replay of a CUDA Graph containing 192 consecutive scans. We reversed the executable order on alternate rounds. All twelve 95% bootstrap intervals favored the new code over the original across eight rounds.

| Elements | u32 faster than original | u64 faster than original |
| ---: | ---: | ---: |
| 65,536 | 2.69–3.17% | 1.64–1.76% |
| 1,048,576 | 2.58–2.80% | 2.02–2.04% |

Compared with the groups build immediately before this change, the twelve reductions ranged from 1.77% to 3.71%. At 33,554,432 elements we timed single scans. Aligned u32 was 1.10% faster than the original, and the other five paired median changes were 0.00%. We also sampled aligned scans at 16,384, 262,144 and 4,194,304 elements. All six paired medians improved, with three intervals including zero. [Per-layout times and confidence intervals](scan-short-timings.csv) · [Raw NVBench output and SASS](/tmp/scan-short-latency-bt7qcr5f)

## Scan register usage before the startup change

The original WarpSpeed kernel used 88 registers per thread for 32-bit and 64-bit integer exclusive sum on SM120a. The groups version used 91 before this fix. In the reduction role, `nvdisasm --print-life-ranges` puts the peak at the last shared-memory loads of the thread's tile fragment. At that point the groups version had 84 general-purpose registers simultaneously live for both types, up from 82. It kept the group-relative rank and shared-memory addresses live alongside the input values.

One change isolates the difference in the groups build. Switching only the per-warp aggregate store from its lane-zero check back to the original warp election restores 88 registers for both types. The store needs lane zero because that is where `WarpReduce::Reduce` guarantees the result. The fix keeps that check, calculates the partial-warp bound from the number of valid items in that warp, and stores each thread's aggregate as soon as its shared-memory phase is available. This shortens register lifetimes without changing the reduction order. [WarpReduce contract](../../cub/cub/warp/warp_reduce.cuh) · [PTX lane election](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)

| Exclusive sum type | Original registers/thread | Groups before fix | Groups after fix |
| --- | ---: | ---: | ---: |
| Signed 8-bit | 89 | 90 | 88 |
| Signed 16-bit | 151 | 152 | 150 |
| Signed and unsigned 32- and 64-bit | 88 | 91 | 88 |
| Float | 89 | 91 | 89 |
| Double | 88 | 91 | 88 |

All eight instantiations used zero stack and spill bytes. The fixed u32 and u64 kernels peak at 82 and 81 simultaneously live registers, respectively. Both still use three barrier slots and run one block per SM. A separate experiment delayed consuming the next-tile index until after processing the current tile. It reduced allocated registers further, but slowed the 1,048,576-element scan by about 4% for u32 and 6% for u64, so the fix keeps the existing pipeline order.

We timed the original, pre-fix and fixed kernels in eight rounds on the RTX PRO 6000, reversing their order on alternate rounds. Across aligned, shifted and in-place layouts, paired median times for the fixed kernels were within 0.01% of the original at 33,554,432 elements. In 192-call graphs they were 0.13–0.60% slower at 1,048,576 elements and 0.28–1.00% slower at 65,536 elements. The paired estimates and 95% bootstrap intervals include comparisons against both baselines. [Scan timing results](scan-register-timings.csv) · [Disassembly and raw measurements](/tmp/cccl-scan-registers-bench-iwtLb9bD)

## Full rewrite measurements before the scan register fix

At 33,554,432 elements, 11 of the 12 scan and RLE configurations had a paired median latency change that rounded to 0.00%. Dense RLE using the position-only policy was 0.55% faster. The initial rewrite slowed the position-only RLE path by 8.1% and 12.0% for run lengths 16 and 256. Making the warp role visibly uniform to the compiler recovered those losses. The graph measurements below cover the remaining differences on shorter inputs.

### Large inputs

Each measurement includes the initialization kernel and the main kernel. Positive deltas indicate longer GPU time. Absolute times are medians across runs for each executable. The paired change is the median of within-pair ratios. These statistics can differ when timings cluster at multiple values.

| Kernel | Layout or input | Baseline µs | Candidate µs | Paired change | 95% bootstrap interval |
| --- | --- | ---: | ---: | ---: | ---: |
| RLE position-only | run 1 | 298.61 | 296.96 | -0.55% | [-0.57%, -0.51%] |
| RLE position-only | run 16 | 126.98 | 126.98 | +0.00% | [-0.03%, +0.03%] |
| RLE position-only | run 256 | 118.75 | 118.78 | +0.00% | [+0.00%, +0.03%] |
| RLE key-staged | run 1 | 278.53 | 278.53 | +0.00% | [+0.00%, +0.00%] |
| RLE key-staged | run 16 | 110.56 | 110.56 | +0.00% | [+0.00%, +0.00%] |
| RLE key-staged | run 256 | 102.37 | 102.37 | +0.00% | [+0.00%, +0.00%] |
| Scan u32 | aligned | 178.14 | 178.14 | +0.00% | [+0.00%, +0.00%] |
| Scan u32 | shifted | 178.14 | 178.16 | +0.00% | [+0.00%, +0.02%] |
| Scan u32 | in-place | 178.14 | 178.14 | +0.00% | [-0.02%, +0.02%] |
| Scan u64 | aligned | 362.46 | 362.46 | +0.00% | [+0.00%, +0.00%] |
| Scan u64 | shifted | 362.50 | 362.46 | -0.00% | [-0.01%, +0.00%] |
| Scan u64 | in-place | 362.48 | 362.46 | +0.00% | [-0.01%, +0.00%] |

### Short inputs

NVBench single-operation timings cluster about 2 µs apart on this GPU. For shifted 32-bit scan at 1,048,576 elements the paired cold median was 0.00% with a 95% interval from −20.06% to +25.10%. To resolve smaller differences we captured 24 successive calls in a CUDA Graph and divided each sample by 24. This also warms the working set between calls and measures steady-state GPU execution with a single host launch. At 1,048,576 elements we observed these per-call results.

| Kernel | Layout or input | Baseline µs | Candidate µs | Paired change | 95% bootstrap interval |
| --- | --- | ---: | ---: | ---: | ---: |
| RLE position-only | run 1 | 7.17 | 7.64 | +2.92% | [-10.03%, +16.25%] |
| RLE position-only | run 16 | 4.78 | 4.69 | -1.77% | [-1.79%, -0.03%] |
| RLE position-only | run 256 | 4.61 | 4.61 | -0.01% | [-1.82%, +0.00%] |
| RLE key-staged | run 1 | 6.48 | 6.40 | -0.01% | [-1.30%, +0.02%] |
| RLE key-staged | run 16 | 5.25 | 5.20 | -0.03% | [-1.59%, -0.01%] |
| RLE key-staged | run 256 | 5.21 | 5.21 | -0.01% | [-1.61%, +0.00%] |
| Scan u32 | aligned | 4.56 | 4.52 | -1.85% | [-1.92%, +0.00%] |
| Scan u32 | shifted | 4.56 | 4.52 | -1.87% | [-1.92%, +0.00%] |
| Scan u32 | in-place | 4.48 | 4.52 | -0.01% | [-1.89%, +0.00%] |
| Scan u64 | aligned | 6.31 | 6.31 | +0.00% | [+0.00%, +0.00%] |
| Scan u64 | shifted | 6.31 | 6.31 | +0.00% | [-0.02%, +0.00%] |
| Scan u64 | in-place | 6.23 | 6.31 | +0.00% | [+0.00%, +1.35%] |

At 65,536 elements the three 64-bit scan layouts shifted by +2.48% to +2.51%, or approximately 0.08 µs per call. Eight-pair graph medians for the six key-staged RLE cases ranged from −1.45% to 0.00%. The 1,048,576-element position-only RLE case with all unique keys varied by much more than its +2.92% paired median, with a 95% interval of −10.03% to +16.25%. [Short-input graph summary](graph-summary.csv)

A follow-up captured 192 calls per graph and repeated ten crossed pairs for these four cases. The 64-bit scan medians were +1.03% aligned (95% interval −1.24% to +1.27%), +1.00% shifted (−1.86% to +3.59%) and +1.26% in place (−0.96% to +5.24%). Dense position-only RLE changed direction to −1.23% (−7.16% to +4.21%). Both experiments put the 64-bit scan shift on the order of hundredths of a microsecond. Its main kernel uses three more registers and eight more SASS instructions with the same occupancy and zero spills. The timings do not establish how much any individual instruction contributes. [192-call summary](graph192-summary.csv) · [192-call per-pair results](graph192-pairs.csv)

### Regression investigation

The first conversion fed a rank derived from `threadIdx.x / 32` directly to the role dispatch. The SM120a compiler emitted extra reconvergence and warp synchronization instructions throughout specialized role bodies. For example, the position-only RLE main kernel grew from 2,136 to 3,656 SASS instructions. For a converged block role partition, applying a full-warp minimum reduction to the identical warp index emits `REDUX.MIN`. This makes the dispatch value warp-uniform to the compiler. This intervention alone recovered both large RLE regressions in a separate eight-pair measurement. Specializing the full-warp shuffle operations removed a further 56 instructions from each RLE kernel. [Initial comparison](draft-summary.csv) · [Uniform partition comparison](uniform-summary.csv)

Eight pairs of graph measurements on that version exposed a further 1.6–4.0% slowdown on short and medium key-staged RLE runs. The baseline assigned tiles using the warp's physical block rank modulo the number of compute or store warps. A relative group rank counts from the start of each role, changing which physical warp handles each tile. Restoring the original assignment in the RLE pipeline preserves group-relative rank throughout the groups API. In a six-way crossed experiment, this change alone moved the six short key-staged results to −1.44% through +0.20% of baseline. Electing the block initializer and reordering the roles each left a dense-input slowdown. The SASS also contained unnecessary grid Y/Z reads. The final 1D grid-rank specialization removed the redundant Y/Z coordinate reads, and restored the four initialization kernels to their baseline size of 24 instructions. [Short RLE investigation](rle-variants.csv) · [Short scan investigation](scan-variants.csv) · [Compiler resources](compiler-resources.csv) · [SASS opcode counts](sass-opcodes.csv)

| Main kernel | Threads/block | Registers/thread, baseline → initial → final | SASS instructions, baseline → initial → final | Static + dynamic shared bytes | Barrier slots | Active blocks/SM |
| --- | ---: | --- | --- | --- | ---: | ---: |
| Scan u32 | 352 | 88 → 109 → 91 | 1,504 → 1,888 → 1,520 | 0 + 98,656 | 3 | 1 |
| Scan u64 | 352 | 88 → 116 → 91 | 1,304 → 1,840 → 1,312 | 0 + 98,712 | 3 | 1 |
| RLE key-staged | 608 | 62 → 64 → 56 | 1,640 → 2,504 → 1,656 | 6,208 + 53,408 | 1 | 1 |
| RLE position-only | 608 | 86 → 90 → 80 | 2,136 → 3,656 → 2,152 | 6,208 + 32,768 | 1 | 1 |

PTXAS reported zero spill loads, spill stores and stack bytes for all four main kernels in each build. Kernel attributes and the occupancy API on the actual graph nodes confirmed one block/SM for both revisions. With 48 warps/SM, the maximum resident warp fractions were 11/48 (22.9%) for scan and 19/48 (39.6%) for RLE. Scan issues immediate `BAR.SYNC` for IDs 0, 1 and 2 with 128 participating threads on the role barriers. RLE uses immediate ID 0. Both revisions used these immediate IDs.

In the compiler CSVs, `old` is the baseline, `new1` the initial rewrite, `uniform1` the warp-uniform dispatch, `final` that version with full-warp shuffles, `rotategraph` the restored RLE assignment and `shipping` the measured final candidate with the 1D grid-rank path.

### Workload and measurement conditions

- The baseline was CCCL commit `4db67c68d9f3177a06b403002af619cc420fb200`. The candidate used the experimental-group implementation in this checkout. Both used NVCC 13.5.37 with GCC 14.3.0, C++17, `-O3 -DNDEBUG -arch=sm_120a -lineinfo`, and NVBench `2514944a603a71f37cde6aeee7384fda94959542`.
- The device was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 188 SMs, ECC enabled, 128 MiB L2. We used the default clock policy and reserved the GPU during timed runs. NVBench discarded a trial if sample clocks fell below its throttle threshold. Median observed SM clock per workload and executable ranged from 2.848 to 2.888 GHz in the final cold suite.
- Public `cub::DeviceScan::ExclusiveSum` ran on `uint32_t` and `uint64_t`, with distinct aligned buffers, output shifted by one element and in-place operation. Initial sample checks used an input of ones. Repeated in-place timing used zeros to keep the input fixed.
- Production RLE tuning on SM120 does not select WarpSpeed. The benchmark explicitly selected the lookahead RLE dispatcher with `uint32_t` keys, 32-bit counts and 32-bit offsets. The SM100 policy with 32 items/thread exceeds the shared-memory allowance and executes the position-only path. An 8 items/thread policy with the same remaining parameters stages keys. Both use 8 compute warps. Inputs contained consecutive, exactly equal-length runs of 1, 16 or 256. Captured launch parameters distinguish the paths.
- We allocated temporary storage before timing and checked output samples against expected keys, counts and prefix values for every workload. Each run used 15 warmups, the entropy stopping criterion and at least 60 samples with a 2 s timeout for cold runs, 0.35 s for graphs. The final cold suite included eight chronological pairs and 2,238–3,672 samples per executable and workload across those pairs.
- Order alternated baseline/candidate then candidate/baseline. Intervals are the 2.5th and 97.5th percentiles of 5,000 resamples of the eight paired median ratios, using a fixed seed. The 192-call follow-up used ten pairs and a 0.45 s timeout. The intervals describe variability between these runs on one GPU, with coarse event quantization affecting small single-operation cases.
- The workload covers one architecture, these policies, types, layouts and input distributions. RLE timings describe a forced specialization on SM120a.

[Cold summary](final-summary.csv) · [Cold per-pair results](final-pairs.csv) · [Graph summary](graph-summary.csv) · [Graph per-pair results](graph-pairs.csv) · [Reproduction commands](reproduce.md) · [Benchmark harness](harness.tar.gz)

After moving the experimental group headers back to cudax, we rebuilt the timed programs with CUB reading those headers directly from `cudax/include`. The complete SM120a SASS dumps matched the measured binaries byte for byte. The manifest records the CUB, cudax and libcu++ headers used for the full rewrite measurements before the scan register fix. [Header SHA-256 manifest](verified-headers.sha256)

The local evidence directory holds all original NVBench Markdown, JSON and logs, disassembled kernels and ptxas output. [Raw NVBench files](/tmp/cccl-warpspeed-perf-6hQmMGH1/measurements) · [Baseline SASS](/tmp/cccl-warpspeed-perf-6hQmMGH1/old.sass) · [Candidate SASS](/tmp/cccl-warpspeed-perf-6hQmMGH1/shipping.sass)
