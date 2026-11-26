# Reduce compile time for libcu++ range access and adaptors

## Summary

- In C++17, skip lower-priority ADL and iterator expressions once a member customization wins for `ranges::begin`, `end`, `data`, `empty`, `rbegin`, and `rend`. Check borrowing first for the operations that require it.
- Probe the `ranges::size` subtraction fallback only after member and ADL customizations fail. Check the actual subtraction of the two returned values inside the constraint. This also avoids eager trailing-return substitution with NVCC and Clang in C++20.
- For `views::take` and `views::drop`, check whether the range is one of the explicitly supported specializations before testing random access and sizing. Ordinary user ranges go directly to their generic adaptor.
- Add nine standalone translation units that each exercise one public operation through `<cuda/std/ranges>`, plus an include-only control.

## Compile-time results

The generic C++20 `views::take` workload used 13.8% less compiler CPU with both NVCC/GCC and NVRTC, and 26.1% less with NVCC/Clang. With NVCC/Clang, the C++20 member-`size()` workload used 43.4% less CPU in the main run and 44.1% less in a repeat. The repeat also shows which large changes in the main table were timing noise.

Each row uses the same benchmark source on both sides. The two header trees are the optimization commit and its immediate parent, starting from NVIDIA/cccl main at `ebeb7505ad`. The include-only control compares main with the complete patch. Times are median compiler CPU seconds. Negative percentages indicate shorter compiles.

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

Two large C++20 movements occurred on access paths where native constraints were unchanged. We repeated those cells, the largest C++17 access slowdown, and the C++20 `size` result with nine additional measured pairs per cell. The primary table retains the original five-pair measurements. The repeat put the first three changes within 1% of baseline. The C++20 `size` improvement reproduced.

| Benchmark | Configuration | Five-pair median (s) | Nine-pair median (s) |
|---|---|---:|---:|
| `ranges::begin` | NVRTC, C++20 | 0.441 -> 0.314 (−28.7%) | 0.255 -> 0.257 (+0.8%) |
| `ranges::end` | NVCC + Clang, C++20 | 4.111 -> 2.868 (−30.3%) | 2.218 -> 2.231 (+0.6%) |
| `ranges::rbegin` | NVCC + Clang, C++17 | 3.666 -> 4.232 (+15.5%) | 2.653 -> 2.666 (+0.5%) |
| `ranges::size` | NVCC + Clang, C++20 | 3.442 -> 1.949 (−43.4%) | 3.143 -> 1.758 (−44.1%) |

## Reproduce

Linux x86-64, Intel Core i9-14900K, CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each matrix cell used one warm-up compile per side, followed by five measured pairs in alternating before/after order on the same pinned logical CPU. Independent pairs of header revisions were measured on multiple physical cores. NVCC CPU time includes the driver and compiler subprocesses with `-O2 -c`. NVRTC CPU time covers `nvrtcCompileProgram` with a new program on every call and its backend cache disabled. No benchmark launches a kernel. Individual sample times, in measured-pair order, are in `reports/optimize_ranges_samples.csv`.

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
      --before "$scratch/before/libcudacxx/include" --after "$scratch/after/libcudacxx/include"
  done
done
```

The optimization commits, in measurement order, are `begin` (`94d82ac3dc`), `end` (`2c89252ff6`), `data` (`79067beee9`), `size` (`c2b2b01a67`), `empty` (`60b57f8e9c`), `rbegin` (`d3d5ebb2e3`), `rend` (`5d0f627f49`), `take` (`7704c20c64`), `drop` (`ad1197594f`).

## Validation

- On an RTX PRO 6000 Blackwell Workstation Edition, all 38 targeted lit tests passed in each of the four NVCC host-compiler/dialect combinations (15 range-access, 14 take, and 9 drop). All five new regression sources also compiled with NVRTC 13.5 in both dialects.
- Ran the root clang-tidy configuration on the modified headers, benchmarks, and regression tests in CUDA host and device mode.
- Two runtime consumers compiled with NVCC 13.5 and GCC 15 produced identical `sm_120a` SASS before and after in both dialects. The seven-operation access kernel used 48 instructions and 18 registers. The generic/specialized take-and-drop kernel used 40 instructions and 14 registers. Each used zero stack bytes and zero spill loads or stores. GPU timings below are medians over 21 alternating batches of 32 launches, processing 4,194,304 elements per launch.

| GPU workload | C++17 runtime | C++20 runtime |
|---|---:|---:|
| Seven range-access operations | 8.172 µs -> 8.161 µs (−0.011 µs) | 8.197 µs -> 8.165 µs (−0.032 µs) |
| Generic and specialized take/drop | 8.142 µs -> 8.203 µs (+0.061 µs) | 8.113 µs -> 8.133 µs (+0.020 µs) |
