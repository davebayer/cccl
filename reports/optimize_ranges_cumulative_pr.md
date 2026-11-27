# Reduce compile time for libcu++ range access, concepts, and view adaptors

## Summary

This PR reduces template instantiation in `<cuda/std/ranges>`. Range access and iterator customization points defer lower-priority probes until overload resolution needs them. Range concepts and aliases check their prerequisites before probing size, subtraction, or movability. Adaptors reuse known iterator and result types in operations such as `zip_view::size`.

Across the three C++17 compiler configurations, the `borrowed_subrange_t` workload used 60.5–70.2% less compiler CPU time, the ADL `iter_swap` workload used 51.4–62.2% less, and `zip_view::size` used 15.7–17.4% less. In a separate C++17 host-only check, GCC 15 emitted 9,212 class records for the baseline `borrowed_subrange_t` workload and 3,639 for the optimized headers (`-fdump-lang-class`).

The new `views::iota` and `views::drop_while` workloads ranged from 0.7% faster to 0.9% slower across the three compilers.

**C++17 `view_interface` regressed on all three compilers.** Independent repeats measured +4.2% with NVCC/GCC, +4.8% with NVCC/Clang, and +5.5% with NVRTC. This workload calls inherited observers and element access on a view with mutable and const `begin()` and `end()`. A separate NVRTC C++17 run on a P core localized part of the cost to the change in `ranges::begin` dispatch. Comparing that commit with its parent in both orders, the changed code took longer in all ten measured pairs. [Change to C++17 member and ADL lookup](https://github.com/davebayer/cccl/commit/94d82ac3dc745fde26820c24aadcda880b324bc4)

The summary uses the geometric mean of the per-workload median compiler CPU seconds. Each feature workload has equal weight within its group. The include-only controls are reported separately in the full table.

| Workloads | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: |
| 27 existing workloads | 6.440 -> 5.296 (−17.8%) | 6.533 -> 5.600 (−14.3%) | 1.193 -> 0.981 (−17.7%) |
| 53 additional workloads | 8.041 -> 7.934 (−1.3%) | 8.811 -> 8.712 (−1.1%) | 1.510 -> 1.484 (−1.7%) |

## Cumulative C++17 compile-time results

Every cell compares headers at upstream `ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae` with the requested branch HEAD, `372cf1d3294b7badac9eb7eb1f44930d2ae6631a`. Both sides compile identical benchmark source. Values are median compiler CPU seconds in the form **before -> after (percentage change)**. Negative percentages mean shorter compiles.

**P** marks the 27 existing workloads measured on performance cores. **E** marks 53 new workloads covering the other public facilities, measured on efficiency cores. Each workload measures the whole translation unit, and some exercise several members or overloads of one facility. The include-only control ran on both core classes and gives context for small changes. All reported times include header parsing. Compare revisions within each cell since absolute times also depend on the core class.

| Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | --- | ---: | ---: | ---: |
| `ranges::begin` | P | 3.737 -> 3.362 (−10.0%) | 3.505 -> 3.509 (+0.1%) | 0.612 -> 0.597 (−2.5%) |
| `ranges::end` | P | 4.560 -> 4.070 (−10.7%) | 4.448 -> 4.361 (−2.0%) | 0.806 -> 0.779 (−3.4%) |
| `ranges::cbegin` | E | 4.946 -> 4.920 (−0.5%) | 5.321 -> 5.314 (−0.1%) | 0.758 -> 0.746 (−1.7%) |
| `ranges::cend` | E | 5.787 -> 5.684 (−1.8%) | 6.299 -> 6.184 (−1.8%) | 0.973 -> 0.950 (−2.4%) |
| `ranges::rbegin` | P | 3.935 -> 3.379 (−14.1%) | 3.563 -> 3.493 (−2.0%) | 0.609 -> 0.579 (−5.0%) |
| `ranges::rend` | P | 4.563 -> 4.130 (−9.5%) | 4.403 -> 4.363 (−0.9%) | 0.820 -> 0.791 (−3.5%) |
| `ranges::crbegin` | E | 8.096 -> 7.923 (−2.1%) | 8.958 -> 8.756 (−2.3%) | 1.577 -> 1.519 (−3.7%) |
| `ranges::crend` | E | 8.310 -> 8.152 (−1.9%) | 9.162 -> 8.980 (−2.0%) | 1.655 -> 1.604 (−3.1%) |
| `ranges::data` | P | 3.059 -> 2.399 (−21.6%) | 2.532 -> 2.471 (−2.4%) | 0.359 -> 0.336 (−6.3%) |
| `ranges::cdata` | E | 7.763 -> 7.589 (−2.2%) | 8.503 -> 8.315 (−2.2%) | 1.491 -> 1.444 (−3.1%) |
| `ranges::empty` | P | 2.950 -> 2.315 (−21.5%) | 2.499 -> 2.403 (−3.8%) | 0.331 -> 0.323 (−2.4%) |
| `ranges::size` | P | 2.958 -> 2.385 (−19.4%) | 2.531 -> 2.483 (−1.9%) | 0.343 -> 0.315 (−8.2%) |
| `ranges::ssize` | E | 7.348 -> 7.177 (−2.3%) | 8.225 -> 8.207 (−0.2%) | 1.396 -> 1.378 (−1.3%) |
| `ranges::disable_sized_range` | E | 7.356 -> 7.145 (−2.9%) | 8.144 -> 7.899 (−3.0%) | 1.388 -> 1.346 (−3.1%) |
| `ranges::range` | E | 6.171 -> 6.052 (−1.9%) | 6.640 -> 6.537 (−1.5%) | 1.012 -> 0.988 (−2.4%) |
| `ranges::input_range` | E | 8.831 -> 8.628 (−2.3%) | 9.842 -> 9.608 (−2.4%) | 1.767 -> 1.703 (−3.6%) |
| `ranges::output_range` | E | 6.524 -> 6.503 (−0.3%) | 7.076 -> 7.012 (−0.9%) | 1.132 -> 1.136 (+0.3%) |
| `ranges::forward_range` | E | 10.240 -> 9.931 (−3.0%) | 11.424 -> 11.136 (−2.5%) | 2.112 -> 2.047 (−3.1%) |
| `ranges::bidirectional_range` | E | 17.027 -> 16.687 (−2.0%) | 18.990 -> 18.519 (−2.5%) | 4.058 -> 3.870 (−4.6%) |
| `ranges::random_access_range` | E | 15.888 -> 15.465 (−2.7%) | 17.930 -> 17.489 (−2.5%) | 3.690 -> 3.578 (−3.0%) |
| `ranges::contiguous_range` | P | 11.088 -> 10.719 (−3.3%) | 12.183 -> 11.638 (−4.5%) | 2.575 -> 2.473 (−4.0%) |
| `ranges::common_range` | E | 6.839 -> 6.728 (−1.6%) | 7.306 -> 7.183 (−1.7%) | 1.203 -> 1.169 (−2.8%) |
| `ranges::sized_range` | P | 4.060 -> 3.843 (−5.3%) | 4.180 -> 4.133 (−1.1%) | 0.717 -> 0.715 (−0.4%) |
| `ranges::borrowed_range` | E | 8.685 -> 8.639 (−0.5%) | 9.617 -> 9.536 (−0.8%) | 1.730 -> 1.705 (−1.4%) |
| `ranges::enable_borrowed_range` | E | 7.728 -> 7.572 (−2.0%) | 8.301 -> 8.333 (+0.4%) | 1.500 -> 1.490 (−0.7%) |
| `ranges::view` | P | 4.428 -> 4.297 (−3.0%) | 4.440 -> 4.435 (−0.1%) | 0.686 -> 0.683 (−0.4%) |
| `ranges::viewable_range` | P | 2.485 -> 2.387 (−4.0%) | 2.500 -> 2.461 (−1.6%) | 0.333 -> 0.331 (−0.5%) |
| `ranges::enable_view` | E | 4.419 -> 4.390 (−0.7%) | 4.772 -> 4.664 (−2.3%) | 0.615 -> 0.608 (−1.2%) |
| `ranges::view_base` | E | 6.562 -> 6.469 (−1.4%) | 7.172 -> 7.039 (−1.9%) | 1.203 -> 1.184 (−1.6%) |
| `ranges::iterator_t` | E | 6.316 -> 6.237 (−1.3%) | 6.787 -> 6.693 (−1.4%) | 1.112 -> 1.090 (−1.9%) |
| `ranges::sentinel_t` | E | 10.867 -> 10.653 (−2.0%) | 11.946 -> 11.650 (−2.5%) | 2.201 -> 2.174 (−1.2%) |
| `ranges::range_difference_t` | P | 6.258 -> 4.328 (−30.8%) | 6.678 -> 4.710 (−29.5%) | 1.287 -> 0.818 (−36.4%) |
| `ranges::range_size_t` | P | 5.034 -> 3.975 (−21.0%) | 5.294 -> 4.313 (−18.5%) | 0.906 -> 0.705 (−22.2%) |
| `ranges::range_value_t` | E | 10.676 -> 10.441 (−2.2%) | 12.224 -> 11.913 (−2.5%) | 2.265 -> 2.195 (−3.1%) |
| `ranges::range_reference_t` | E | 11.103 -> 10.789 (−2.8%) | 12.183 -> 11.897 (−2.3%) | 2.275 -> 2.210 (−2.9%) |
| `ranges::range_rvalue_reference_t` | E | 6.957 -> 6.698 (−3.7%) | 7.537 -> 7.270 (−3.5%) | 1.231 -> 1.197 (−2.8%) |
| `ranges::range_common_reference_t` | E | 8.915 -> 8.659 (−2.9%) | 10.086 -> 9.748 (−3.3%) | 1.813 -> 1.689 (−6.8%) |
| `ranges::dangling` | E | 6.723 -> 6.579 (−2.1%) | 7.265 -> 7.225 (−0.6%) | 1.212 -> 1.187 (−2.0%) |
| `ranges::borrowed_iterator_t` | E | 8.774 -> 8.684 (−1.0%) | 9.526 -> 9.418 (−1.1%) | 1.778 -> 1.758 (−1.1%) |
| `ranges::borrowed_subrange_t` | P | 11.231 -> 4.432 (−60.5%) | 12.275 -> 4.531 (−63.1%) | 2.754 -> 0.820 (−70.2%) |
| `ranges::advance` | E | 12.101 -> 11.806 (−2.4%) | 14.035 -> 13.775 (−1.9%) | 2.643 -> 2.581 (−2.3%) |
| `ranges::distance` | E | 10.905 -> 10.719 (−1.7%) | 12.208 -> 11.996 (−1.7%) | 2.404 -> 2.356 (−2.0%) |
| `ranges::next` | E | 12.169 -> 11.919 (−2.1%) | 14.168 -> 13.898 (−1.9%) | 2.688 -> 2.609 (−2.9%) |
| `ranges::prev` | E | 12.062 -> 11.846 (−1.8%) | 13.647 -> 13.317 (−2.4%) | 2.817 -> 2.754 (−2.2%) |
| `ranges::swap` | E | 3.945 -> 3.951 (+0.2%) | 4.184 -> 4.230 (+1.1%) | 0.509 -> 0.501 (−1.4%) |
| `ranges::iter_move` (ADL customization) | P | 5.577 -> 4.999 (−10.4%) | 5.252 -> 5.152 (−1.9%) | 0.924 -> 0.889 (−3.7%) |
| `ranges::iter_move` (raw pointer) | P | 4.467 -> 2.469 (−44.7%) | 4.637 -> 2.502 (−46.1%) | 0.748 -> 0.334 (−55.4%) |
| `ranges::iter_swap` (ADL customization) | P | 9.545 -> 4.621 (−51.6%) | 10.255 -> 4.988 (−51.4%) | 2.132 -> 0.806 (−62.2%) |
| `view_interface` | E | 7.429 -> 7.731 (+4.1%) | 8.225 -> 8.612 (+4.7%) | 1.384 -> 1.461 (+5.6%) |
| `ref_view` members | E | 5.601 -> 5.539 (−1.1%) | 6.040 -> 6.089 (+0.8%) | 0.902 -> 0.907 (+0.5%) |
| `owning_view` members | E | 6.844 -> 6.764 (−1.2%) | 7.481 -> 7.383 (−1.3%) | 1.228 -> 1.206 (−1.8%) |
| `subrange` construction and navigation | E | 9.617 -> 9.724 (+1.1%) | 10.603 -> 10.690 (+0.8%) | 1.777 -> 1.756 (−1.2%) |
| `subrange` tuple access | E | 7.768 -> 7.946 (+2.3%) | 8.441 -> 8.591 (+1.8%) | 1.340 -> 1.336 (−0.3%) |
| `views::all` | P | 14.234 -> 10.983 (−22.8%) | 14.472 -> 11.447 (−20.9%) | 2.881 -> 2.180 (−24.3%) |
| `views::all_t` | E | 8.588 -> 8.451 (−1.6%) | 9.296 -> 9.129 (−1.8%) | 1.642 -> 1.600 (−2.6%) |
| `range_adaptor_closure` application | E | 4.545 -> 4.467 (−1.7%) | 4.909 -> 4.863 (−0.9%) | 0.642 -> 0.630 (−2.0%) |
| `views::transform` and `views::filter` closure composition | P | 12.397 -> 12.111 (−2.3%) | 13.150 -> 12.913 (−1.8%) | 2.320 -> 2.315 (−0.3%) |
| `views::empty` | E | 4.710 -> 4.643 (−1.4%) | 4.995 -> 4.945 (−1.0%) | 0.672 -> 0.663 (−1.3%) |
| `views::single` | E | 7.601 -> 7.548 (−0.7%) | 8.226 -> 8.179 (−0.6%) | 1.412 -> 1.393 (−1.3%) |
| `views::iota` | E | 7.422 -> 7.392 (−0.4%) | 8.135 -> 8.075 (−0.7%) | 1.341 -> 1.333 (−0.6%) |
| `views::indices` | E | 4.585 -> 4.543 (−0.9%) | 5.069 -> 5.011 (−1.2%) | 0.652 -> 0.643 (−1.4%) |
| `views::repeat` | E | 6.355 -> 6.306 (−0.8%) | 6.948 -> 6.902 (−0.7%) | 1.085 -> 1.089 (+0.3%) |
| `views::counted` | E | 19.441 -> 19.048 (−2.0%) | 21.391 -> 21.029 (−1.7%) | 4.725 -> 4.640 (−1.8%) |
| `common_view::begin/end` | P | 9.101 -> 8.840 (−2.9%) | 10.032 -> 9.410 (−6.2%) | 2.056 -> 1.939 (−5.7%) |
| `views::common` | E | 9.930 -> 9.808 (−1.2%) | 10.805 -> 10.673 (−1.2%) | 1.974 -> 1.965 (−0.5%) |
| `reverse_view` construction | P | 12.704 -> 11.468 (−9.7%) | 13.721 -> 12.448 (−9.3%) | 3.053 -> 2.681 (−12.2%) |
| `views::reverse` | E | 12.440 -> 12.217 (−1.8%) | 13.807 -> 13.544 (−1.9%) | 2.676 -> 2.629 (−1.7%) |
| `views::take` | P | 12.869 -> 10.645 (−17.3%) | 13.944 -> 11.416 (−18.1%) | 2.960 -> 2.376 (−19.7%) |
| `take_view` traversal and observers | E | 7.716 -> 7.555 (−2.1%) | 8.118 -> 7.952 (−2.0%) | 1.518 -> 1.472 (−3.0%) |
| `views::take_while` | E | 7.576 -> 7.625 (+0.7%) | 8.191 -> 8.263 (+0.9%) | 1.470 -> 1.465 (−0.4%) |
| `views::drop` | P | 13.323 -> 12.988 (−2.5%) | 14.529 -> 13.902 (−4.3%) | 3.057 -> 2.918 (−4.6%) |
| `drop_view::begin` | P | 9.023 -> 8.148 (−9.7%) | 9.502 -> 8.474 (−10.8%) | 2.046 -> 1.786 (−12.7%) |
| `drop_view` traversal and observers | E | 7.947 -> 7.800 (−1.8%) | 8.487 -> 8.640 (+1.8%) | 1.512 -> 1.478 (−2.2%) |
| `views::drop_while` | E | 8.940 -> 8.947 (+0.1%) | 9.693 -> 9.691 (0.0%) | 1.798 -> 1.815 (+0.9%) |
| `views::filter` | E | 6.695 -> 6.743 (+0.7%) | 7.281 -> 7.352 (+1.0%) | 1.189 -> 1.192 (+0.3%) |
| `views::transform` | P | 8.073 -> 7.629 (−5.5%) | 8.622 -> 8.346 (−3.2%) | 1.798 -> 1.719 (−4.4%) |
| `transform_view` traversal and observers | E | 6.191 -> 6.214 (+0.4%) | 6.619 -> 6.687 (+1.0%) | 1.163 -> 1.175 (+1.1%) |
| `views::zip` | E | 14.958 -> 14.594 (−2.4%) | 16.368 -> 16.003 (−2.2%) | 3.484 -> 3.381 (−3.0%) |
| `zip_view::begin/end` | P | 8.149 -> 7.924 (−2.8%) | 8.831 -> 8.328 (−5.7%) | 1.857 -> 1.771 (−4.6%) |
| `zip_view::size` | P | 9.773 -> 8.241 (−15.7%) | 11.068 -> 9.176 (−17.1%) | 2.237 -> 1.847 (−17.4%) |
| `<cuda/std/ranges>` include only | P | 2.261 -> 2.237 (−1.1%) | 2.334 -> 2.317 (−0.7%) | 0.297 -> 0.288 (−2.9%) |
| `<cuda/std/ranges>` include only | E | 3.525 -> 3.503 (−0.6%) | 3.775 -> 3.727 (−1.3%) | 0.440 -> 0.431 (−2.0%) |

<details>
<summary>What the workloads exercise</summary>

The inventory covers the public range facilities shipped by this libcu++ header, including its iterator operations and the `views::indices` extension. [Pinned libcu++ header](https://github.com/davebayer/cccl/blob/372cf1d3294b7badac9eb7eb1f44930d2ae6631a/libcudacxx/include/cuda/std/ranges) · [Benchmark sources](https://github.com/davebayer/cccl/tree/092804fa5b1f879b2abc310b0d83527ef5a08b1c/libcudacxx/benchmarks/compile_time/ranges)

- **Access and size.** Member, ADL and iterator fallback selection, const access on a mutable range, signed sizing, and `disable_sized_range`.
- **Range requirements and associated types.** Iterator-category boundaries, common and distinct sentinels, borrowed and owning ranges, opting into views and borrowing, reference and value types, and dangling results.
- **Iterator operations.** Distance, bounded and unbounded navigation, element swap through a customization, custom iterator moves and swaps, and moving through a raw pointer.
- **Storage and view utilities.** The CRTP `view_interface` observers and element access (including `at`), `ref_view` and `owning_view` access and `base()`, sized and unsized `subrange_kind`, subrange construction and navigation, and the subrange tuple protocol (`get`, `tuple_size`, `tuple_element`).
- **Factories.** Empty and single views, bounded and unbounded iota and repeat views, the libcu++ `views::indices` extension, and all three dispatch paths for `views::counted`.
- **Adaptors.** Construction, the resulting view classes and their member operations for all, common, reverse, take, take-while, drop, drop-while, filter, transform and zip. Separate cases cover closure composition, piping a range into a user-defined `range_adaptor_closure`, and `views::all_t`. `cuda::std::views` aliases `cuda::std::ranges::views`.
- **Input counts.** Existing workloads use 24–64 distinct type indices. Most new probes use 24 or 32, and single and empty views use 64. The indices workload checks six standard integral types and rejection of a floating-point argument.

</details>

### C++17 repeat measurements

Every configuration whose first median was more than 2.5% slower received nine additional measured pairs at the same revisions. The primary table retains the first run.

| Workload | Compiler | Core | First run (s) | Repeat (s) |
| --- | --- | --- | ---: | ---: |
| `view_interface` | NVCC + GCC | E | 7.429 -> 7.731 (+4.1%) | 6.497 -> 6.772 (+4.2%) |
| `view_interface` | NVCC + Clang | E | 8.225 -> 8.612 (+4.7%) | 7.193 -> 7.541 (+4.8%) |
| `view_interface` | NVRTC | E | 1.384 -> 1.461 (+5.6%) | 1.253 -> 1.322 (+5.5%) |

## Measurement and reproduction

The benchmarks ran on an Intel Core i9-14900K with CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each configuration used one warm-up per revision, followed by nine measured pairs in alternating order on one pinned physical core. Independent configurations ran on distinct cores within the two groups. NVCC time includes the compiler driver and child processes for `-O2 -c`. NVRTC time covers `nvrtcCompileProgram` with a fresh program and its cache disabled. All workloads compile without launching a kernel.

In the raw CSV, select `standard=c++17` for the measurements shown here. Primary samples use `series=cumulative`, with `cumulative-e-control` for the second include-only control. Confirmation samples append `-repeat` to the series name. Each percentage uses the two unrounded medians. The per-optimization tables compare each change with its parent. [Raw CPU-time samples](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_samples.csv) · [Per-optimization tables](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_pr.md)

To reproduce one cell, run from a checkout containing the measured revisions and the benchmark source commit. The 53 new workloads were added at `092804fa5b1f879b2abc310b0d83527ef5a08b1c`, which has the same library header tree as the measured HEAD. Select another source filename or compiler (`gcc`, `clang`, or `nvrtc`) to reproduce the other cells.

```bash
base=ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae
head=372cf1d3294b7badac9eb7eb1f44930d2ae6631a
workloads=092804fa5b1f879b2abc310b0d83527ef5a08b1c
suite=libcudacxx/benchmarks/compile_time/ranges
scratch=$(mktemp -d)
cpu=0  # On the measured host, P cores used 0,2,...,14 and E cores used 16 through 31.
mkdir -p "$scratch/before" "$scratch/after"
git archive "$base" libcudacxx/include | tar -xf - -C "$scratch/before"
git archive "$head" libcudacxx/include | tar -xf - -C "$scratch/after"
git archive "$workloads" "$suite" | tar -xf - -C "$scratch/after"

taskset -c "$cpu" python3 "$scratch/after/$suite/measure.py" "$scratch/after/$suite/zip_size.cu" \
  --before "$scratch/before/libcudacxx/include" --after "$scratch/after/libcudacxx/include" \
  --compiler clang --std 17 --repeats 9
```

## Validation

All 53 additional benchmarks passed the repository's clang-tidy configuration in CUDA host and device mode for both C++17 and C++20.

At the same two revisions, a consumer dropped variable-length prefixes from two forward ranges, zipped the results, queried their size, and called `iter_move` on the zip iterator and a raw pointer. All 524,288 outputs per build matched the CPU reference on an RTX PRO 6000 Blackwell Workstation Edition. NVCC 13.5 with GCC 15 produced byte-identical `sm_120a` cubins across both revisions and dialects, with 36 instructions through `EXIT`, 17 registers per thread, zero stack bytes, and zero spills. Kernel times are medians of 15 alternating batches of 32 warm launches.

| Dialect | GPU time |
|---|---:|
| C++17 | 9.134 µs -> 9.127 µs (−0.08%) |
