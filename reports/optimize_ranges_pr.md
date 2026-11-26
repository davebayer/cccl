# Reduce compile time for libcu++ range access and adaptors

## Summary

- In C++17, skip lower-priority ADL and iterator expressions once a member customization wins for `ranges::begin`, `end`, `data`, `empty`, `rbegin`, and `rend`. Check borrowing first for the operations that require it.
- Probe the `ranges::size` subtraction fallback only after member and ADL customizations fail. Check the actual subtraction of the two returned values inside the constraint. This also avoids eager trailing-return substitution with NVCC and Clang in C++20.
- For `views::take` and `views::drop`, check whether the range is one of the explicitly supported specializations before testing random access and sizing. Ordinary user ranges go directly to their generic adaptor.
- Skip irrelevant checks in range concepts, `reverse_view`, `common_view`, `views::all`, and range adaptor composition. Add sixteen standalone workloads and an include-only control.

## First nine range optimizations

The generic C++20 `views::take` workload used 13.8% less compiler CPU with both NVCC/GCC and NVRTC, and 26.1% less with NVCC/Clang. With NVCC/Clang, the C++20 member-`size()` workload used 43.4% less CPU in the main run and 44.1% less in a repeat. The repeat also shows which large changes in the main table were timing noise.

Each row uses the same benchmark source on both sides. The two header trees are the optimization commit and its immediate parent, starting from NVIDIA/cccl main at `ebeb7505ad`. The include-only control compares main with the first nine optimizations. Times are median compiler CPU seconds. Negative percentages indicate shorter compiles.

| Benchmark | C++ | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
|---|---:|---:|---:|---:|
| `ranges::begin` | 17 | 3.136 -> 2.747 (−12.4%) | 2.793 -> 2.900 (+3.8%) | 0.452 -> 0.443 (−1.9%) |
| `ranges::begin` | 20 | 2.096 -> 2.104 (+0.4%) | 2.188 -> 2.149 (−1.8%) | 0.441 -> 0.314 (−28.7%) |
| `ranges::end` | 17 | 3.696 -> 3.444 (−6.8%) | 4.082 -> 3.915 (−4.1%) | 0.735 -> 0.650 (−11.6%) |
| `ranges::end` | 20 | 2.398 -> 2.240 (−6.6%) | 4.111 -> 2.868 (−30.3%) | 0.471 -> 0.485 (+2.8%) |
| `ranges::data` | 17 | 2.476 -> 1.983 (−19.9%) | 2.461 -> 2.150 (−12.6%) | 0.283 -> 0.274 (−3.3%) |
| `ranges::data` | 20 | 1.855 -> 1.955 (+5.4%) | 2.821 -> 2.571 (−8.9%) | 0.375 -> 0.341 (−9.0%) |
| `ranges::size` | 17 | 4.267 -> 3.420 (−19.8%) | 3.341 -> 3.448 (+3.2%) | 0.447 -> 0.433 (−3.3%) |
| `ranges::size` | 20 | 2.002 -> 2.058 (+2.8%) | 3.442 -> 1.949 (−43.4%) | 0.176 -> 0.173 (−1.4%) |
| `ranges::empty` | 17 | 2.715 -> 2.140 (−21.2%) | 3.522 -> 3.307 (−6.1%) | 0.282 -> 0.280 (−0.7%) |
| `ranges::empty` | 20 | 3.079 -> 3.099 (+0.7%) | 2.014 -> 1.930 (−4.1%) | 0.168 -> 0.172 (+2.3%) |
| `ranges::rbegin` | 17 | 3.569 -> 3.371 (−5.5%) | 3.666 -> 4.232 (+15.5%) | 0.496 -> 0.506 (+2.0%) |
| `ranges::rbegin` | 20 | 2.317 -> 2.260 (−2.5%) | 2.241 -> 2.210 (−1.4%) | 0.261 -> 0.256 (−1.8%) |
| `ranges::rend` | 17 | 4.384 -> 3.883 (−11.4%) | 4.444 -> 4.262 (−4.1%) | 0.655 -> 0.642 (−2.0%) |
| `ranges::rend` | 20 | 2.293 -> 2.310 (+0.8%) | 3.013 -> 2.835 (−5.9%) | 0.366 -> 0.343 (−6.2%) |
| `views::take` | 17 | 14.220 -> 13.780 (−3.1%) | 15.705 -> 12.438 (−20.8%) | 3.064 -> 2.458 (−19.8%) |
| `views::take` | 20 | 5.764 -> 4.969 (−13.8%) | 7.853 -> 5.803 (−26.1%) | 1.023 -> 0.882 (−13.8%) |
| `views::drop` | 17 | 15.158 -> 13.645 (−10.0%) | 15.972 -> 14.429 (−9.7%) | 3.091 -> 2.962 (−4.2%) |
| `views::drop` | 20 | 5.731 -> 5.515 (−3.8%) | 7.693 -> 6.357 (−17.4%) | 1.063 -> 1.052 (−1.0%) |
| Include only | 17 | 1.955 -> 1.929 (−1.3%) | 1.952 -> 1.962 (+0.5%) | 0.246 -> 0.244 (−0.9%) |
| Include only | 20 | 1.750 -> 1.761 (+0.6%) | 1.803 -> 1.836 (+1.9%) | 0.159 -> 0.163 (+2.9%) |

Each workload instantiates the operation on 64 distinct range types. For `begin` and `end`, a member returning a pointer competes with an ADL overload returning a reverse iterator. For `data`, `size`, and `empty`, a valid member supersedes a reverse-iterator fallback. The `rbegin` benchmark supplies forward-access members returning reverse iterators. The `rend` benchmark supplies a competing ADL overload. The generic `take` and `drop` inputs use reverse iterators and check the resulting view type. In these workloads both adaptors compiled faster in all six configurations. The include-only control ranged from 1.3% faster to 2.9% slower.

Native C++20 constraints already short-circuit most range-access expressions. Standalone Clang 22 traces gave identical constraint-check and constant-evaluation counts for all seven C++20 access workloads before and after. NVCC with Clang 22 previously instantiated the subtraction fallback in the C++20 `size` regression test even after finding a valid size customization.

For the adaptors, Clang 22 counted the following constant-expression evaluations on the same inputs. Counts are from standalone host compilation.

| Benchmark | C++17 constant evaluations | C++20 constant evaluations |
|---|---:|---:|
| `views::take` | 300,808 -> 227,515 (−24.4%) | 80,032 -> 47,135 (−41.1%) |
| `views::drop` | 316,361 -> 302,474 (−4.4%) | 97,764 -> 80,741 (−17.4%) |

### Repeat measurements

Two large C++20 movements occurred on access paths where native constraints were unchanged. We repeated those cells, the largest C++17 access slowdown, and the C++20 `size` result with nine additional measured pairs per cell. The primary table retains the original five-pair measurements. The first three rows came back within 1% of baseline. Five more nine-pair repeats below check positive C++20 results from the original matrix. The C++20 `size` improvement reproduced.

| Benchmark | Configuration | Five-pair median (s) | Nine-pair median (s) |
|---|---|---:|---:|
| `ranges::begin` | NVRTC, C++20 | 0.441 -> 0.314 (−28.7%) | 0.255 -> 0.257 (+0.8%) |
| `ranges::end` | NVCC + Clang, C++20 | 4.111 -> 2.868 (−30.3%) | 2.218 -> 2.231 (+0.6%) |
| `ranges::rbegin` | NVCC + Clang, C++17 | 3.666 -> 4.232 (+15.5%) | 2.653 -> 2.666 (+0.5%) |
| `ranges::size` | NVCC + Clang, C++20 | 3.442 -> 1.949 (−43.4%) | 3.143 -> 1.758 (−44.1%) |
| `ranges::data` | NVCC + GCC, C++20 | 1.855 -> 1.955 (+5.4%) | 1.7693 -> 1.7611 (−0.5%) |
| `ranges::size` | NVCC + GCC, C++20 | 2.002 -> 2.058 (+2.8%) | 1.7093 -> 1.7087 (−0.03%) |
| `ranges::end` | NVRTC, C++20 | 0.471 -> 0.485 (+2.8%) | 0.2598 -> 0.2596 (−0.07%) |
| `ranges::empty` | NVRTC, C++20 | 0.168 -> 0.172 (+2.3%) | 0.1553 -> 0.1558 (+0.3%) |
| Include only | NVRTC, C++20 | 0.159 -> 0.163 (+2.9%) | 0.1438 -> 0.1427 (−0.8%) |

For the original C++20 `data` workload with NVCC/GCC, the +5.4% compares the two separate five-sample medians. The median of the five within-pair changes was +1.3%. In a separate nine-pair control, recompiling the same pre-change headers on both sides gave 1.7476 -> 1.7400 (−0.4%) seconds. CPU time also varies with clock frequency and cache contention on a pinned core.

For `begin`, `end`, `data`, `empty`, `rbegin`, and `rend`, NVCC/GCC produced identical preprocessed C++20 source lines for the host and both device passes after stripping line markers and blank lines. Those changes affect the C++17 constraint implementation. The full include-only comparison also parses the changed C++20 declarations for `size`, `take`, and `drop`.

## Seven additional range optimizations

Each row compares one change with its immediate parent, using one warm-up pair and nine measured pairs. The include-only row compares the original main revision with all sixteen optimizations. Times are median compiler CPU seconds. The C++17 `views::all` wrapper workload saved 17.5% with NVCC/GCC, 18.6% with NVCC/Clang and 24.7% with NVRTC. The `reverse_view` workload saved 5.2–11.2% across all six configurations.

| Benchmark | C++ | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
|---|---:|---:|---:|---:|
| `reverse_view` | 17 | 10.419 -> 9.655 (−7.3%) | 11.379 -> 10.482 (−7.9%) | 2.479 -> 2.202 (−11.2%) |
| `reverse_view` | 20 | 5.261 -> 4.989 (−5.2%) | 6.434 -> 5.908 (−8.2%) | 1.509 -> 1.359 (−9.9%) |
| `ranges::viewable_range` | 17 | 2.048 -> 2.006 (−2.0%) | 3.383 -> 3.316 (−2.0%) | 0.279 -> 0.269 (−3.5%) |
| `ranges::viewable_range` | 20 | 2.287 -> 2.045 (−10.6%) | 2.121 -> 2.005 (−5.5%) | 0.191 -> 0.193 (+1.3%) |
| `ranges::contiguous_range` | 17 | 9.692 -> 9.563 (−1.3%) | 10.607 -> 10.696 (+0.8%) | 2.258 -> 2.265 (+0.3%) |
| `ranges::contiguous_range` | 20 | 4.772 -> 4.736 (−0.8%) | 5.650 -> 5.537 (−2.0%) | 0.777 -> 0.768 (−1.2%) |
| `common_view::begin/end` | 17 | 8.078 -> 8.572 (+6.1%) | 8.932 -> 8.257 (−7.6%) | 1.636 -> 1.619 (−1.0%) |
| `common_view::begin/end` | 20 | 4.820 -> 4.751 (−1.4%) | 6.155 -> 6.056 (−1.6%) | 0.662 -> 0.637 (−3.7%) |
| Adaptor closure composition | 17 | 12.864 -> 12.869 (0.0%) | 13.306 -> 13.206 (−0.8%) | 2.350 -> 2.312 (−1.6%) |
| Adaptor closure composition | 20 | 11.080 -> 11.549 (+4.2%) | 12.147 -> 11.959 (−1.5%) | 2.054 -> 2.066 (+0.6%) |
| `ranges::view` | 17 | 3.691 -> 3.564 (−3.4%) | 3.763 -> 3.782 (+0.5%) | 0.585 -> 0.583 (−0.4%) |
| `ranges::view` | 20 | 3.207 -> 3.200 (−0.2%) | 3.388 -> 3.383 (−0.2%) | 0.453 -> 0.455 (+0.4%) |
| `views::all` | 17 | 12.518 -> 10.331 (−17.5%) | 13.416 -> 10.926 (−18.6%) | 3.079 -> 2.319 (−24.7%) |
| `views::all` | 20 | 6.356 -> 6.368 (+0.2%) | 6.915 -> 6.702 (−3.1%) | 1.213 -> 1.205 (−0.7%) |
| Include only | 17 | 2.306 -> 2.258 (−2.1%) | 2.289 -> 2.334 (+2.0%) | 0.280 -> 0.272 (−3.0%) |
| Include only | 20 | 2.053 -> 2.028 (−1.2%) | 2.136 -> 2.133 (−0.2%) | 0.171 -> 0.171 (−0.4%) |

- `reverse_view` uses the end iterator directly for a common range. The workload constructs reverse views on 64 common ranges and avoids checking every iterator for random access.
- In C++17, `viewable_range` stops as soon as its argument fails `range`. The workload checks 64 non-range types as both values and const lvalues.
- In C++17, `contiguous_range` checks whether the iterator is contiguous before forming the range's `data()` expression. The workload checks 64 ranges with reverse iterators and a deduced `data()` return type.
- `common_view::begin/end` queries the base size only for random-access iterators. The workload checks all four const and mutable overloads on 32 forward ranges. Clang deduced the return type of the 32 base `size()` members before the change and zero afterwards in both dialects.
- In C++17, `views::all` reuses the range and movability guarantees of the primary `ref_view` and `owning_view` wrappers and avoids testing lower-priority constructions. User specializations and derived types retain the full `view` check. C++20 retains its native constraints so a more constrained lvalue-view overload still wins. The workload passes a const reference wrapper and an rvalue owning wrapper for each of 64 base ranges.
- The C++17 `operator|` overloads check each operand before considering invocation or construction. The workload composes 32 distinct `transform | filter` pipelines.
- The C++17 `view` predicate checks `range`, then `movable`, then `enable_view`. The workload rejects 64 tuple types before querying their elements' movability.

Standalone GCC class dumps and Clang time traces count compiler work on the same translation units. GCC counts class records in `-fdump-lang-class`. Clang reports the evaluation count in the `Total EvaluateAsConstantExpr` entry from `-ftime-trace`.

| Benchmark | GCC 15 class records, C++17 | Clang 22 constant evaluations, C++17 | Clang 22 constant evaluations, C++20 |
|---|---:|---:|---:|
| `reverse_view` | 8,957 -> 8,893 (−0.7%) | 236,141 -> 214,234 (−9.3%) | 57,631 -> 52,255 (−9.3%) |
| `ranges::viewable_range` | 2,860 -> 1,195 (−58.2%) | 4,476 -> 3,964 (−11.4%) | 4,457 -> 4,457 (0.0%) |
| `ranges::contiguous_range` | 9,845 -> 8,182 (−16.9%) | 214,214 -> 214,022 (−0.1%) | 56,948 -> 56,948 (0.0%) |
| `common_view::begin/end` | 5,910 -> 5,908 (0.0%) | 134,298 -> 129,018 (−3.9%) | 61,912 -> 59,571 (−3.8%) |
| Adaptor closure composition | 7,304 -> 6,492 (−11.1%) | 160,864 -> 160,160 (−0.4%) | 152,332 -> 152,332 (0.0%) |
| `ranges::view` | 3,264 -> 1,980 (−39.3%) | 4,811 -> 4,683 (−2.7%) | 5,304 -> 5,304 (0.0%) |
| `views::all` | 11,451 -> 8,698 (−24.0%) | 220,769 -> 177,635 (−19.5%) | 45,151 -> 45,151 (0.0%) |

For `viewable_range`, `contiguous_range`, `operator|`, `view`, and `views::all`, NVCC/GCC produced identical preprocessed C++20 source for the host and both device passes before and after each change, after removal of line markers and blank lines. Treat C++20 timing differences on those five workloads as measurement variation.

The largest measured slowdown on a changed code path was C++17 `common_view` with NVCC/GCC. The repeat used nine fresh pairs and the same source and header revisions.

| Benchmark | Configuration | First run (s) | Repeat (s) |
|---|---|---:|---:|
| `common_view::begin/end` | NVCC + GCC, C++17 | 8.078 -> 8.572 (+6.1%) | 8.341 -> 8.219 (−1.5%) |

## Reproduce

Linux x86-64, Intel Core i9-14900K, CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each matrix cell used one warm-up compile per side, followed by five measured pairs for the first nine optimizations and nine pairs for the seven additional optimizations and the repeat measurements in alternating before/after order on the same pinned logical CPU. Independent pairs of header revisions were measured on multiple physical cores. NVCC CPU time includes the driver and compiler subprocesses with `-O2 -c`. NVRTC CPU time covers `nvrtcCompileProgram` with a new program on every call and its backend cache disabled. No benchmark launches a kernel. Individual sample times, in measured-pair order, are in `reports/optimize_ranges_samples.csv`.

Run from the version of the branch containing the benchmark suite. Substitute any operation and its commit from the list below to run all six configurations for that operation.

```bash
suite=libcudacxx/benchmarks/compile_time/ranges
op=size
rev=c2b2b01a67
scratch=$(mktemp -d)
mkdir -p "$scratch/before" "$scratch/after"
git archive "$rev^" libcudacxx/include | tar -x -C "$scratch/before"
git archive "$rev"  libcudacxx/include | tar -x -C "$scratch/after"

for compiler in gcc clang nvrtc; do
  for std in 17 20; do
    python3 "$suite/measure.py" "$suite/$op.cu" --compiler "$compiler" --std "$std" \
      --repeats 5 --before "$scratch/before/libcudacxx/include" --after "$scratch/after/libcudacxx/include"
  done
done
```

The first nine optimization commits, in measurement order, are `begin` (`94d82ac3dc`), `end` (`2c89252ff6`), `data` (`79067beee9`), `size` (`c2b2b01a67`), `empty` (`60b57f8e9c`), `rbegin` (`d3d5ebb2e3`), `rend` (`5d0f627f49`), `take` (`7704c20c64`), `drop` (`ad1197594f`).

For the seven additional workloads, use `--repeats 9`. Their benchmark filenames and optimization commits are `reverse.cu` (`df2bad63a7`), `viewable_range.cu` (`9088e5a063`), `contiguous_range.cu` (`f8444aa65e`), `common.cu` (`6924b7409b`), `pipe.cu` (`c0737d8109`), `view.cu` (`8b716764ca`), `all.cu` (`99db57a0b9`).

## Validation

- On an RTX PRO 6000 Blackwell Workstation Edition, all 38 targeted lit tests passed in each of the four NVCC host-compiler/dialect combinations (15 range-access, 14 take, and 9 drop). All five new regression sources also compiled with NVRTC 13.5 in both dialects.
- Ran the root clang-tidy configuration on the modified headers, benchmarks, and regression tests in CUDA host and device mode.
- Two runtime consumers compiled with NVCC 13.5 and GCC 15 produced identical `sm_120a` SASS before and after in both dialects. The seven-operation access kernel used 48 instructions and 18 registers. The generic/specialized take-and-drop kernel used 40 instructions and 14 registers. Each used zero stack bytes and zero spill loads or stores. GPU timings below are medians over 21 alternating batches of 32 launches, processing 4,194,304 elements per launch.

| GPU workload | C++17 runtime | C++20 runtime |
|---|---:|---:|
| Seven range-access operations | 8.172 µs -> 8.161 µs (−0.011 µs) | 8.197 µs -> 8.165 µs (−0.032 µs) |
| Generic and specialized take/drop | 8.142 µs -> 8.203 µs (+0.061 µs) | 8.113 µs -> 8.133 µs (+0.020 µs) |

### Additional range changes

- The 100 targeted lit tests cover range access, concepts, and the affected adaptors. For both GCC 15 and Clang 22 with NVCC 13.5, 96 tests passed in C++17 and 100 passed in C++20 on the RTX PRO 6000. Four tests require C++20. All seven new regression sources also compiled with NVRTC 13.5 and MSVC 19.52 in both dialects.
- Root-configured clang-tidy 22 with the CCCL plugin passed on the seven changed headers, seven benchmarks, and seven regression sources in CUDA host and device modes for both dialects.
- A consumer composing `all | reverse`, querying range concepts and indexing a counted `common_view` produced identical `sm_120a` SASS before and after the seven changes. In both dialects it used 40 instructions, 12 registers, zero stack bytes and zero spills. Every one of the 4,194,304 output elements matched. Timings use the same 21 alternating batches of 32 launches as above.

| GPU workload | C++17 runtime | C++20 runtime |
|---|---:|---:|
| Combined range consumer | 8.144 µs -> 8.177 µs (+0.033 µs) | 8.071 µs -> 8.095 µs (+0.024 µs) |
