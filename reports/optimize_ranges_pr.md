# Reduce compile time for libcu++ range access and adaptors

The [cumulative compile-time comparison](optimize_ranges_cumulative_pr.md) measures every workload against the same upstream baseline. The tables below measure each optimization against its parent commit.

## Summary

- In C++17, skip lower-priority ADL and iterator expressions once a member customization wins for `ranges::begin`, `end`, `data`, `empty`, `rbegin`, and `rend`. Check borrowing first for the operations that require it.
- Probe the `ranges::size` subtraction fallback only after member and ADL customizations fail. Check the actual subtraction of the two returned values inside the constraint. This also avoids eager trailing-return substitution with NVCC and Clang in C++20.
- For `views::take` and `views::drop`, check whether the range is one of the explicitly supported specializations before testing random access and sizing. Ordinary user ranges go directly to their generic adaptor.
- Honor one-way `iter_swap` customizations, short-circuit iterator difference-type and borrowed-subrange selection, and avoid redundant deduction in transform and zip views.
- Skip irrelevant checks in range concepts, `reverse_view`, `common_view`, `views::all`, and range adaptor composition.
- Check the `range_size_t` constraint before substitution, skip size checks for forward drop views, give `zip_view::size` an explicit result type, and short-circuit both stages of `iter_move` dispatch. The suite contains 27 standalone workloads and an include-only control.

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

Each row compares one change with its immediate parent, using one warm-up pair and nine measured pairs. The include-only row compares the original main revision with the sixteen optimizations through this update. Times are median compiler CPU seconds. The C++17 `views::all` wrapper workload saved 17.5% with NVCC/GCC, 18.6% with NVCC/Clang and 24.7% with NVRTC. The `reverse_view` workload saved 5.2–11.2% across all six configurations.

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

## Six further range and iterator optimizations

Times are median compiler CPU seconds from nine measured pairs per configuration. Each operation uses identical benchmark source against the optimization commit and its immediate parent. The include-only control compares upstream main (`ebeb7505ad`) with the 22 optimizations through that update. On the 64-type workloads, `iter_swap` used 31.8–62.9% less compiler CPU and `range_difference_t` used 17.4–35.9% less across the six configurations.

| Benchmark | C++ | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
|---|---:|---:|---:|---:|
| `ranges::sized_range` | 17 | 3.640 -> 3.467 (−4.7%) | 3.732 -> 3.734 (0.0%) | 0.563 -> 0.550 (−2.2%) |
| `ranges::sized_range` | 20 | 2.277 -> 2.234 (−1.9%) | 2.380 -> 2.427 (+2.0%) | 0.255 -> 0.259 (+1.3%) |
| `ranges::iter_swap` | 17 | 8.938 -> 4.306 (−51.8%) | 9.426 -> 4.550 (−51.7%) | 1.876 -> 0.696 (−62.9%) |
| `ranges::iter_swap` | 20 | 5.772 -> 3.935 (−31.8%) | 6.609 -> 4.122 (−37.6%) | 1.040 -> 0.533 (−48.8%) |
| `ranges::borrowed_subrange_t` | 17 | 11.333 -> 11.251 (−0.7%) | 12.553 -> 11.976 (−4.6%) | 2.374 -> 2.350 (−1.0%) |
| `ranges::borrowed_subrange_t` | 20 | 5.366 -> 5.552 (+3.5%) | 6.297 -> 5.972 (−5.2%) | 0.947 -> 0.909 (−4.1%) |
| `views::transform` | 17 | 7.141 -> 7.392 (+3.5%) | 7.493 -> 7.640 (+2.0%) | 1.694 -> 1.669 (−1.5%) |
| `views::transform` | 20 | 4.379 -> 4.281 (−2.2%) | 4.861 -> 4.655 (−4.2%) | 0.759 -> 0.705 (−7.1%) |
| `ranges::range_difference_t` | 17 | 5.992 -> 4.271 (−28.7%) | 6.435 -> 4.517 (−29.8%) | 1.206 -> 0.773 (−35.9%) |
| `ranges::range_difference_t` | 20 | 3.406 -> 2.814 (−17.4%) | 3.747 -> 3.057 (−18.4%) | 0.480 -> 0.341 (−28.9%) |
| `zip_view::begin/end` | 17 | 7.908 -> 7.760 (−1.9%) | 8.446 -> 8.159 (−3.4%) | 1.767 -> 1.709 (−3.2%) |
| `zip_view::begin/end` | 20 | 5.851 -> 5.650 (−3.4%) | 6.257 -> 6.204 (−0.8%) | 1.032 -> 0.997 (−3.3%) |
| Include only | 17 | 2.226 -> 2.203 (−1.0%) | 2.290 -> 2.263 (−1.2%) | 0.278 -> 0.280 (+0.7%) |
| Include only | 20 | 2.044 -> 2.003 (−2.0%) | 1.997 -> 2.009 (+0.6%) | 0.178 -> 0.180 (+1.0%) |

- `sized_range` rejects a non-range before probing its size in C++17. The benchmark queries 64 queue-like types that expose the size of their underlying storage. In C++20, NVCC/GCC produced identical preprocessed input for the host and both device passes after stripping line markers and blank lines.
- `iter_swap` checks for a customization in the caller's operand order before considering either fallback. The benchmark swaps a cursor and a pointer through a one-way ADL overload on 64 tuple types. The previous move-exchange overload could conflict with a valid customization or suppress the fallback for the opposite order. [C++ draft, iterator.cust.swap](https://eel.is/c++draft/iterator.cust.swap)
- `borrowed_subrange_t` constructs the iterator-based subrange type only for borrowed ranges. The benchmark requests the result for 64 temporary ranges with a reverse cursor. The regression test also checks that a temporary with a move-only input iterator yields `dangling`. [LWG 3551](https://cplusplus.github.io/LWG/issue3551)
- `views::transform` constructs the known result type directly and checks that construction during substitution. The benchmark adapts 32 reverse-iterator ranges. The new test checks function-reference decay and distinguishes an rvalue move-only callable from an lvalue that would need a copy.
- `zip_view::begin/end` constructs the already-known iterator or sentinel tuple with direct calls to each view, in tuple order. The benchmark obtains an iterator and a non-common sentinel for 24 pairs of views.
- `range_difference_t` uses a named iterator difference type before probing subtraction. The benchmark queries 64 strided ranges. Clang deduced 64 subtraction function bodies before the change and zero after it in both dialects. When the iterator has no named type, the existing fallback still derives a signed type from integral subtraction.

| Benchmark | GCC 15 class records, C++17 | Clang 22 constant evaluations, C++17 | Clang 22 constant evaluations, C++20 |
|---|---:|---:|---:|
| `ranges::sized_range` | 3,310 -> 2,925 (−11.6%) | 48,620 -> 48,429 (−0.4%) | 13,615 -> 13,615 (0.0%) |
| `ranges::iter_swap` | 7,553 -> 2,303 (−69.5%) | 100,257 -> 6,517 (−93.5%) | 36,145 -> 5,535 (−84.7%) |
| `ranges::borrowed_subrange_t` | 9,213 -> 9,151 (−0.7%) | 216,358 -> 203,558 (−5.9%) | 56,693 -> 53,237 (−6.1%) |
| `views::transform` | 5,820 -> 5,820 (0.0%) | 122,321 -> 121,297 (−0.8%) | 35,691 -> 30,155 (−15.5%) |
| `ranges::range_difference_t` | 5,432 -> 3,576 (−34.2%) | 95,898 -> 49,950 (−47.9%) | 27,829 -> 15,474 (−44.4%) |
| `zip_view::begin/end` | 4,342 -> 4,084 (−5.9%) | 91,730 -> 81,828 (−10.8%) | 41,902 -> 33,629 (−19.7%) |

The first C++20 NVCC/GCC run for `borrowed_subrange_t` was 3.5% slower, and a fresh nine-pair run was 3.9% faster. For the final typed zip implementation, NVCC/Clang C++20 used 0.8% less CPU in the main run and 6.2% less in the repeat. The two C++17 transform rows also have independent nine-pair repeats below.

| Benchmark | Configuration | First run (s) | Repeat (s) |
|---|---|---:|---:|
| `ranges::borrowed_subrange_t` | NVCC + GCC, C++20 | 5.366 -> 5.552 (+3.5%) | 5.289 -> 5.083 (−3.9%) |
| `views::transform` | NVCC + Clang, C++17 | 7.493 -> 7.640 (+2.0%) | 6.491 -> 6.337 (−2.4%) |
| `views::transform` | NVCC + GCC, C++17 | 7.141 -> 7.392 (+3.5%) | 5.973 -> 5.931 (−0.7%) |
| `zip_view::begin/end` | NVCC + Clang, C++20 | 6.257 -> 6.204 (−0.8%) | 5.081 -> 4.766 (−6.2%) |

## Five more range and iterator optimizations

Each row measures one optimization against its immediate parent with nine alternating measured pairs after warm-up. Times are median compiler CPU seconds on the isolated workloads. The include-only control compares the original upstream revision (`ebeb7505ad`) with all 27 optimizations.

| Benchmark | C++ | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: | ---: |
| `ranges::range_size_t` | 17 | 4.224 -> 3.418 (−19.1%) | 4.676 -> 3.857 (−17.5%) | 0.809 -> 0.633 (−21.8%) |
| `ranges::range_size_t` | 20 | 2.347 -> 2.326 (−0.9%) | 2.291 -> 2.289 (−0.1%) | 0.306 -> 0.304 (−0.6%) |
| `ranges::iter_move` | 17 | 5.442 -> 4.853 (−10.8%) | 5.018 -> 4.982 (−0.7%) | 0.849 -> 0.837 (−1.4%) |
| `ranges::iter_move` | 20 | 4.529 -> 4.457 (−1.6%) | 8.603 -> 4.532 (−47.3%) | 0.701 -> 0.701 (0.0%) |
| `drop_view::begin` | 17 | 8.339 -> 7.662 (−8.1%) | 8.736 -> 7.910 (−9.5%) | 1.925 -> 1.723 (−10.5%) |
| `drop_view::begin` | 20 | 4.302 -> 3.992 (−7.2%) | 5.353 -> 5.054 (−5.6%) | 0.676 -> 0.604 (−10.6%) |
| `zip_view::size` | 17 | 9.239 -> 7.788 (−15.7%) | 9.893 -> 8.440 (−14.7%) | 2.046 -> 1.705 (−16.7%) |
| `zip_view::size` | 20 | 6.226 -> 5.261 (−15.5%) | 6.784 -> 5.744 (−15.3%) | 1.210 -> 0.959 (−20.7%) |
| `ranges::iter_move` on pointers | 17 | 4.287 -> 2.342 (−45.4%) | 4.383 -> 2.479 (−43.4%) | 0.659 -> 0.293 (−55.5%) |
| `ranges::iter_move` on pointers | 20 | 2.120 -> 2.162 (+1.9%) | 2.223 -> 2.234 (+0.5%) | 0.221 -> 0.221 (0.0%) |
| Include only | 17 | 2.175 -> 2.163 (−0.6%) | 2.234 -> 2.230 (−0.2%) | 0.264 -> 0.264 (+0.2%) |
| Include only | 20 | 1.943 -> 1.962 (+1.0%) | 2.043 -> 2.050 (+0.3%) | 0.179 -> 0.177 (−1.2%) |

- The C++17 `range_size_t` alias checks `sized_range` before forming a size expression. The benchmark probes 64 queue types whose storage exposes a size but whose interface has no iterators. In C++20, the normalized preprocessor output was identical before and after for the NVCC/GCC host and both device passes.
- `ranges::iter_move` checks for an ADL customization before forming the dereference fallback. The fallback uses the value category of the forwarded operand, which matters when an iterator overloads `operator*() &` and `operator*() &&` differently. The benchmark queries 64 reverse zip cursors with a custom move. [C++ draft, iterator.cust.move](https://eel.is/c++draft/iterator.cust.move)
- `drop_view::begin` checks the iterator category before examining the base size. For 32 forward views, Clang 22 deduced the underlying `size()` return types 32 -> 0 (−100%) in both dialects. The regression test also covers a move-only input iterator. [C++ draft, range.drop.view](https://eel.is/c++draft/range.drop.view)
- `zip_view::size` derives its unsigned common result type and conditional `noexcept` directly from the underlying views. Checking its signature no longer instantiates the tuple access and minimum computation in its body. The benchmark checks both the result type and the exception specification for 48 zip views, each containing four underlying views. [C++ draft, range.zip.view](https://eel.is/c++draft/range.zip.view)
- For pointer operands, C++17 skips the lookup for a custom `iter_move` before it can instantiate pointee types. The benchmark moves through pointers to 64 distinct three-field tuple types. The existing incomplete-pointee regression test now runs in C++17 as well as C++20. The normalized C++20 preprocessor input matched for all three NVCC/GCC passes.

| Benchmark | GCC 15 class records, C++17 | Clang 22 constant evaluations, C++17 | Clang 22 constant evaluations, C++20 |
| --- | ---: | ---: | ---: |
| `ranges::range_size_t` | 3,319 -> 2,996 (−9.7%) | 56,926 -> 48,604 (−14.6%) | 13,658 -> 13,658 (0.0%) |
| `ranges::iter_move` | 6,924 -> 2,308 (−66.7%) | 11,835 -> 11,121 (−6.0%) | 10,093 -> 10,127 (+0.3%) |
| `drop_view::begin` | 5,638 -> 5,411 (−4.0%) | 136,811 -> 125,673 (−8.1%) | 72,264 -> 68,960 (−4.6%) |
| `zip_view::size` | 7,120 -> 6,326 (−11.2%) | 145,725 -> 101,538 (−30.3%) | 73,078 -> 30,073 (−58.8%) |
| `ranges::iter_move` on pointers | 2,115 -> 1,202 (−43.2%) | 4,222 -> 2,416 (−42.8%) | 3,349 -> 3,349 (0.0%) |

For the C++20 ADL-customized `iter_move` workload, NVCC with Clang spent about half as much time in `cudafe++` and each `cicc` pass after the change. The final host compilation took about the same time. The following diagnostic uses `nvcc --time` and reports median **wall seconds per phase** from three alternating pairs after warm-up, on a separate pinned core for each host compiler.

| Host compiler | cudafe++ | cicc, compute_120 | cicc, compute_120a | Final host compile |
|---|---:|---:|---:|---:|
| GCC 15 | 1.976 -> 2.028 (+2.6%) | 1.679 -> 1.697 (+1.1%) | 1.706 -> 1.704 (−0.1%) | 0.748 -> 0.754 (+0.8%) |
| Clang 22 | 4.145 -> 1.951 (−52.9%) | 3.623 -> 1.615 (−55.4%) | 3.545 -> 1.636 (−53.8%) | 1.239 -> 1.237 (−0.2%) |

The `drop_view::begin` workload used 5.6–10.6% less compiler CPU, and `zip_view::size` used 14.7–20.7% less across the six configurations. The include-only control varied from 1.2% faster to 1.0% slower. Nine fresh pairs repeated the C++20 NVCC/Clang results for customized `iter_move` and `zip_view::size`, as well as the largest apparent slowdown on an unchanged C++20 path.

| Benchmark | Configuration | First run (s) | Repeat (s) |
| --- | ---: | ---: | ---: |
| `ranges::iter_move` | NVCC + Clang, C++20 | 8.603 -> 4.532 (−47.3%) | 6.451 -> 3.450 (−46.5%) |
| `zip_view::size` | NVCC + Clang, C++20 | 6.784 -> 5.744 (−15.3%) | 5.217 -> 4.287 (−17.8%) |
| `ranges::iter_move` on pointers | NVCC + GCC, C++20 | 2.120 -> 2.162 (+1.9%) | 1.706 -> 1.708 (+0.1%) |

## Reproduce

Linux x86-64, Intel Core i9-14900K, CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each matrix cell used one warm-up compile per side, followed by five measured pairs for the first nine optimizations and nine pairs for the subsequent optimizations and repeat measurements in alternating before/after order on the same pinned logical CPU. Independent pairs of header revisions were measured on multiple physical cores. NVCC CPU time includes the driver and compiler subprocesses with `-O2 -c`. NVRTC CPU time covers `nvrtcCompileProgram` with a new program on every call and its backend cache disabled. No benchmark launches a kernel. Individual sample times, in measured-pair order, are in `reports/optimize_ranges_samples.csv`.

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

For the six further workloads use `--repeats 9`. The benchmark files and commits are `sized_range.cu` (`7181b31341`), `iter_swap.cu` (`5ae0992c44`), `borrowed_subrange.cu` (`c94857dc9c`), `transform.cu` (`3730061915`), `range_difference.cu` (`9ffc892405`), `zip_access.cu` (`a1fb9b4ae8`).

For this update use `--repeats 9`. The benchmark files and optimization commits are `range_size_type.cu` (`323d6bb434`), `iter_move.cu` (`a532938897`), `drop_begin.cu` (`2b6156c031`), `zip_size.cu` (`56e664527f`), `iter_move_pointer.cu` (`e1f34843cf`).

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

### Six further changes

The six affected lit directories contain 57 tests. With each of GCC 15 and Clang 22 under NVCC 13.5, 56 passed in C++17 and 57 in C++20 on the RTX PRO 6000. The remaining C++17 case tests C++20 concept subsumption. All six regression sources compiled with NVRTC 13.5 and MSVC 19.52 in both dialects. Root clang-tidy 22 with the CCCL plugin passed on all 18 changed C++ files in CUDA host and device modes under both dialects.

The combined consumer swaps through ADL, zips and transforms two distinct input buffers, obtains a borrowed subrange and indexes the result using its range difference type. All 4,194,304 output elements matched. The before and after `sm_120a` SASS was identical in both dialects, with 32 instructions, 12 registers, zero stack bytes and zero spills. Runtime uses 21 alternating measured batches of 32 launches.

| GPU workload | C++17 runtime | C++20 runtime |
|---|---:|---:|
| Combined range and iterator consumer | 8.192 µs -> 8.224 µs (+0.032 µs) | 8.212 µs -> 8.222 µs (+0.010 µs) |

### Five further changes

The targeted suites cover 58 distinct tests across range concepts, drop and zip views, and the iterators that delegate to `ranges::iter_move`. With each of GCC 15 and Clang 22 under NVCC 13.5 on the RTX PRO 6000, 57 passed in C++17 and 58 passed in C++20. The remaining test requires C++20 concept subsumption. All 27 zip tests also passed with NVRTC in each dialect. All five benchmark sources and five regression sources compiled with NVRTC 13.5 and MSVC 19.52 in both dialects. Root clang-tidy 22 with the CCCL plugin passed for all 14 changed C++ files in CUDA host and device modes under both standards.

The combined kernel drops variable prefixes from two forward ranges, zips them, obtains the common size, and calls `iter_move` through both iterator customization and pointer fallback. All 524,288 output rows matched an independent CPU calculation in both revisions and dialects. The four `sm_120a` cubins built with NVCC 13.5 and GCC 15 were byte-identical. Kernel timings are medians of 15 alternating batches of 32 warm launches on the RTX PRO 6000.

| C++ | SASS instructions through EXIT | Registers per thread | Stack bytes | Spill load/store bytes | GPU time |
|---|---:|---:|---:|---:|---:|
| 17 | 36 -> 36 (0) | 17 -> 17 (0) | 0 -> 0 (0) | 0 -> 0 (0) | 9.129 µs -> 9.153 µs (+0.26%) |
| 20 | 36 -> 36 (0) | 17 -> 17 (0) | 0 -> 0 (0) | 0 -> 0 (0) | 9.132 µs -> 9.106 µs (−0.28%) |

## Additional C++17 compile-time measurements

Each workload compares the listed commit with its immediate parent on the same core, using one warm-up per revision and nine alternating before/after pairs. Cells show median compiler CPU seconds and their percentage change. P and E denote performance and efficiency cores. The raw CSV records all three compilers in C++17 and C++20 under `update5`. Use the cumulative report's reproduction command with the indicated commit and its parent. The CSV's `benchmark` field gives the source filename without its `.cu` extension.

| Change | Commit | Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Reuse C++17 begin/end constraint results and iterator types | [93bd457e74](https://github.com/davebayer/cccl/commit/93bd457e7493fb3fe8b22d4621221c73e7a3aed1) | `view_interface` | P | 3.532 -> 3.160 (−10.5%) | 3.775 -> 3.337 (−11.6%) | 0.671 -> 0.555 (−17.3%) |
| Include the optional definition used by movable-box | [683d389115](https://github.com/davebayer/cccl/commit/683d389115a19dbb0bdf0bf880a869dab634b653) | `<cuda/std/ranges>` include only | E | 3.595 -> 3.572 (−0.6%) | 3.778 -> 3.763 (−0.4%) | 0.435 -> 0.428 (−1.5%) |
| Specialize incrementability and sentinel checks for raw pointers | [0d520586e5](https://github.com/davebayer/cccl/commit/0d520586e5dc5e3454cd1475b61e30b48ead049a) | `ranges::sized_range` | E | 7.045 -> 4.458 (−36.7%) | 7.730 -> 4.845 (−37.3%) | 1.299 -> 0.638 (−50.9%) |
| Construct the known common view type directly | [6093c884bb](https://github.com/davebayer/cccl/commit/6093c884bbadd6d68481bafc92638031d15d66f8) | `views::common` | E | 9.129 -> 9.146 (+0.2%) | 9.832 -> 10.138 (+3.1%) | 1.928 -> 1.894 (−1.7%) |
| Construct the known take-while view type directly | [5972c2ee38](https://github.com/davebayer/cccl/commit/5972c2ee38266b13be28199b65b5ad435d566f66) | `views::take_while` | E | 7.651 -> 7.413 (−3.1%) | 8.253 -> 7.931 (−3.9%) | 1.423 -> 1.377 (−3.3%) |
| Skip the redundant ADL swap probe in C++17 movable | [b959653566](https://github.com/davebayer/cccl/commit/b9596535668a82f27d7fba4830d05330486df7ef) | `subrange` tuple access | E | 6.920 -> 6.726 (−2.8%) | 7.605 -> 7.371 (−3.1%) | 1.342 -> 1.279 (−4.7%) |
| Recognize compatible references when checking iterator requirements | [bb68664d97](https://github.com/davebayer/cccl/commit/bb68664d972ef454b81db9f943570ae144f0c43a) | `ranges::advance` | E | 9.080 -> 8.591 (−5.4%) | 10.041 -> 9.467 (−5.7%) | 2.039 -> 1.912 (−6.2%) |
| Detect ADL swap with a direct C++17 expression | [b585c026fc](https://github.com/davebayer/cccl/commit/b585c026fc71112fb2965f4d78da313e3a739c57) | `ranges::swap` | E | 3.976 -> 3.956 (−0.5%) | 4.190 -> 4.177 (−0.3%) | 0.513 -> 0.513 (+0.1%) |
| Evaluate the bounded iota constructor ordering check only when assertions run | [e26bf74a57](https://github.com/davebayer/cccl/commit/e26bf74a572d2e5a8ea9d2ee30d01445a67c9504) | `iota_view` bounded pointer construction | E | 7.035 -> 6.731 (−4.3%) | 7.623 -> 7.202 (−5.5%) | 1.286 -> 1.188 (−7.7%) |

### C++17 confirmations

| Workload | Compiler | Core | First run (s) | Repeat (s) |
| --- | --- | --- | ---: | ---: |
| `views::common` | NVCC 13.5 + Clang 22 | E | 9.832 -> 10.138 (+3.1%) | 9.173 -> 9.319 (+1.6%) |

## C++17 compile-time changes for the 10% target

Each row compares the listed commit with its immediate parent. The compiler columns show median CPU seconds, with one warm-up per revision and nine alternating measured pairs. The core column gives the class used by GCC, Clang and NVRTC respectively when they differ.

| Change | Commit | Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Expose the ADL iterator move result without deducing the function body | [b902859ccb](https://github.com/davebayer/cccl/commit/b902859ccbcb9348673dd9a65f162607e506eea4) | `ranges::iter_move` (ADL customization) | P | 4.702 -> 3.992 (−15.1%) | 4.813 -> 3.986 (−17.2%) | 0.795 -> 0.624 (−21.5%) |
| Reuse the common assignment requirement for pointer copies | [ed6e9c51e9](https://github.com/davebayer/cccl/commit/ed6e9c51e9debb6cbb8b48cd3c3f04c5cb1447c8) | `iota_view` bounded pointer construction | P | 3.440 -> 3.198 (−7.0%) | 3.538 -> 3.242 (−8.4%) | 0.634 -> 0.556 (−12.4%) |
| Recognize built-in counting iterator arithmetic for pointers | [8bae5b1fba](https://github.com/davebayer/cccl/commit/8bae5b1fbaad731b5ede3f00d5f72c8c2db0a43f) | `views::iota` | P | 3.451 -> 3.223 (−6.6%) | 3.519 -> 3.402 (−3.3%) | 0.591 -> 0.564 (−4.6%) |
| Reuse the built-in properties of bounded arrays in range refinements | [08bb594de5](https://github.com/davebayer/cccl/commit/08bb594de5cb30d80b5fe2a53765671556c87808) | `ref_view` members | P | 2.700 -> 2.623 (−2.8%) | 2.813 -> 2.699 (−4.1%) | 0.417 -> 0.389 (−6.7%) |
| Use the array forward declaration in span | [a844d73030](https://github.com/davebayer/cccl/commit/a844d73030966f9cbbd9e22eeccfe03a8671d4e2) | `<cuda/std/ranges>` include only | P | 2.138 -> 2.086 (−2.4%) | 2.202 -> 2.130 (−3.3%) | 0.269 -> 0.258 (−4.1%) |
| Store a partial range adaptor's bound argument directly | [c61ea8ec38](https://github.com/davebayer/cccl/commit/c61ea8ec381a1925128ddc6c8e66c4a2bd0798b5) | `views::transform` and `views::filter` closure composition | P | 13.073 -> 9.022 (−31.0%) | 13.760 -> 9.279 (−32.6%) | 2.598 -> 1.739 (−33.1%) |
| Parse bounds exception types only for host compilation | [1aad72294a](https://github.com/davebayer/cccl/commit/1aad72294a2d1150aeacb4a9c67a397761269eb4) | `ranges::viewable_range` | P | 2.261 -> 2.156 (−4.6%) | 2.328 -> 2.174 (−6.6%) | 0.278 -> 0.277 (−0.4%) |
| Reuse direct construction checks in C++17 object concepts | [2e6a21f7a0](https://github.com/davebayer/cccl/commit/2e6a21f7a0f5d1d8dc66ab949a22f8a5d5c9e957) | `views::single` | P | 3.636 -> 3.299 (−9.3%) | 3.927 -> 3.654 (−6.9%) | 0.653 -> 0.579 (−11.3%) |
| Cache the C++17 iterator rvalue reference result | [b011e551f4](https://github.com/davebayer/cccl/commit/b011e551f437eb0a2c8bbfc4609a29bfcd683386) | `ranges::range_rvalue_reference_t` | P | 2.994 -> 2.929 (−2.2%) | 3.091 -> 3.033 (−1.9%) | 0.504 -> 0.492 (−2.5%) |
| Check ADL eligibility before substituting the result (Clang C++20 correctness) | [0e77f5e09c](https://github.com/davebayer/cccl/commit/0e77f5e09c4b20645d7cd090bfbc3090c5e562fc) | `ranges::iter_move` (ADL customization) | E | 7.364 -> 7.276 (−1.2%) | 7.895 -> 7.910 (+0.2%) | 1.257 -> 1.264 (+0.5%) |
| Limit swap ADL to class and enumeration operands | [56c8a438e5](https://github.com/davebayer/cccl/commit/56c8a438e5d04e11c079c7c8e1c077f7852bee6f) | `ranges::swap` (raw pointers) | E | 8.236 -> 8.041 (−2.4%) | 8.742 -> 8.544 (−2.3%) | 1.520 -> 1.461 (−3.9%) |
| Include the tuple facilities used by forwarding wrappers and repeat views | [ea1c244f01](https://github.com/davebayer/cccl/commit/ea1c244f01196c3f261219fff892da11e22fa9d5) | `views::empty` | E | 4.559 -> 4.448 (−2.4%) | 4.934 -> 4.744 (−3.9%) | 0.718 -> 0.698 (−2.8%) |
| Cache compiler-version checks repeated by each header prologue | [44c5a12d26](https://github.com/davebayer/cccl/commit/44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97) | `ranges::view` | E | 7.484 -> 7.437 (−0.6%) | 7.514 -> 7.565 (+0.7%) | 1.290 -> 1.252 (−3.0%) |

The first ADL iterator move change made the result type available in the declaration. The later eligibility check prevents substitution for operands outside the ADL branch, including pointers with an associated user-defined type. Both commits appear above so the cost of the correctness fix is visible. The cumulative ADL row uses the final implementation.

The sample series are `update6` and `update6-repeat`. The CSV contains both C++17 and C++20 measurements for each change. [Raw CPU-time samples](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_samples.csv)

### Direct measurements for the two workloads below 10%

These paired C++17 builds compare the previous headers (`e26bf74a572d2e5a8ea9d2ee30d01445a67c9504`) directly with the final headers (`44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97`). All six comparisons ran on performance cores. For `views::empty`, the cumulative upstream comparison ran on efficiency cores. Each cell gives median CPU seconds. The CSV series is `targeted3-direct`.

| Workload | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: |
| `ranges::view` | 4.145 -> 3.828 (−7.6%) | 4.195 -> 4.015 (−4.3%) | 0.645 -> 0.649 (+0.7%) |
| `views::empty` | 2.613 -> 2.346 (−10.2%) | 2.719 -> 2.387 (−12.2%) | 0.393 -> 0.360 (−8.4%) |

## Intermediate C++17 results for the 20 targeted workloads

The first twelve changes were measured at `ea1c244f01196c3f261219fff892da11e22fa9d5` against upstream `ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae` before the final header change. This partial sweep completed 134 of 252 cumulative compiler/workload comparisons. The table contains all three compilers for each of the 20 workloads previously under 10%. Cells give median CPU seconds and their percentage change. The last column compares mean savings at `e26bf74a57` and this intermediate commit. All completed checkpoint samples are in the `cumulative3-intermediate` and `cumulative3-intermediate-e-control` CSV series.

| Targeted workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 | Mean savings, previous -> intermediate |
| --- | --- | ---: | ---: | ---: | ---: |
| `views::transform` and `views::filter` closure composition | P | 13.889 -> 7.078 (−49.0%) | 14.394 -> 7.278 (−49.4%) | 2.814 -> 1.376 (−51.1%) | 0.8% -> 49.9% (+49.1 pp) |
| `ranges::enable_view` | E | 4.270 -> 3.747 (−12.2%) | 4.824 -> 4.259 (−11.7%) | 0.631 -> 0.583 (−7.6%) | 1.3% -> 10.6% (+9.3 pp) |
| `ranges::view` | P | 4.436 -> 4.090 (−7.8%) | 4.496 -> 4.314 (−4.1%) | 0.698 -> 0.684 (−2.0%) | 1.3% -> 4.6% (+3.3 pp) |
| `views::empty` | E | 4.742 -> 4.387 (−7.5%) | 5.201 -> 4.707 (−9.5%) | 0.732 -> 0.691 (−5.7%) | 1.7% -> 7.6% (+5.8 pp) |
| `ranges::swap` | E | 4.074 -> 3.582 (−12.1%) | 4.514 -> 3.833 (−15.1%) | 0.550 -> 0.498 (−9.4%) | 2.4% -> 12.2% (+9.8 pp) |
| `ranges::viewable_range` | P | 2.551 -> 2.141 (−16.1%) | 2.568 -> 2.222 (−13.5%) | 0.337 -> 0.306 (−9.2%) | 3.3% -> 13.0% (+9.7 pp) |
| `range_adaptor_closure` application | E | 4.935 -> 4.234 (−14.2%) | 5.403 -> 4.624 (−14.4%) | 0.699 -> 0.617 (−11.8%) | 3.6% -> 13.5% (+9.8 pp) |
| `views::indices` | E | 4.916 -> 4.096 (−16.7%) | 5.274 -> 4.408 (−16.4%) | 0.700 -> 0.582 (−16.8%) | 4.0% -> 16.6% (+12.6 pp) |
| `ranges::iter_move` (ADL customization) | P | 5.558 -> 4.092 (−26.4%) | 5.162 -> 4.270 (−17.3%) | 0.858 -> 0.675 (−21.4%) | 4.6% -> 21.8% (+17.2 pp) |
| `views::repeat` | E | 6.846 -> 5.725 (−16.4%) | 7.371 -> 6.077 (−17.5%) | 1.202 -> 0.986 (−18.0%) | 5.2% -> 17.3% (+12.1 pp) |
| `transform_view` traversal and observers | E | 7.120 -> 5.979 (−16.0%) | 7.639 -> 6.401 (−16.2%) | 1.304 -> 1.091 (−16.4%) | 6.0% -> 16.2% (+10.2 pp) |
| `views::drop_while` | E | 9.392 -> 7.242 (−22.9%) | 10.308 -> 7.793 (−24.4%) | 1.935 -> 1.442 (−25.5%) | 6.1% -> 24.3% (+18.1 pp) |
| `views::iota` | E | 8.031 -> 6.161 (−23.3%) | 8.620 -> 6.557 (−23.9%) | 1.445 -> 1.047 (−27.5%) | 6.3% -> 24.9% (+18.6 pp) |
| `ref_view` members | E | 5.916 -> 4.755 (−19.6%) | 6.356 -> 5.192 (−18.3%) | 0.928 -> 0.722 (−22.2%) | 6.6% -> 20.1% (+13.4 pp) |
| `views::filter` | E | 7.098 -> 5.828 (−17.9%) | 7.818 -> 6.387 (−18.3%) | 1.262 -> 1.015 (−19.5%) | 7.7% -> 18.6% (+10.9 pp) |
| `views::take_while` | E | 8.153 -> 6.627 (−18.7%) | 8.845 -> 7.097 (−19.8%) | 1.574 -> 1.226 (−22.1%) | 7.8% -> 20.2% (+12.4 pp) |
| `views::single` | E | 8.188 -> 6.509 (−20.5%) | 8.885 -> 6.939 (−21.9%) | 1.568 -> 1.200 (−23.5%) | 7.9% -> 22.0% (+14.0 pp) |
| `ranges::size` | P | 2.946 -> 2.092 (−29.0%) | 2.498 -> 2.157 (−13.7%) | 0.319 -> 0.282 (−11.6%) | 8.3% -> 18.5% (+10.2 pp) |
| `ranges::range_rvalue_reference_t` | E | 6.918 -> 5.413 (−21.7%) | 7.549 -> 5.899 (−21.9%) | 1.233 -> 0.930 (−24.6%) | 8.4% -> 22.7% (+14.3 pp) |
| `ranges::data` | P | 3.092 -> 2.154 (−30.3%) | 2.583 -> 2.221 (−14.0%) | 0.363 -> 0.325 (−10.5%) | 9.6% -> 18.7% (+9.2 pp) |
