# Reduce compile time for libcu++ range access, concepts, and adaptors

This PR reduces template substitution in range access, iterator requirements and view adaptors. In this update, a closure stores its single bound argument directly, C++17 copy and move concepts reuse the direct construction check, and iterator move constraints reuse the ADL result. Pointer arithmetic and pointer assignment use their built-in operations. Bounded arrays reuse their range properties, and the ranges header parses fewer dependencies. The shared header prologue evaluates three compiler-version checks once per translation unit and reuses their results on subsequent includes.

Compared with each commit's own parent, storing the bound argument directly reduced compiler time for transform/filter closure composition by 31.0–33.1%. Reusing direct construction checks reduced compiler time for `views::single` by 6.9–11.3%. The isolated header-prologue change had smaller effects on the tuple-based `ranges::view` workload (NVCC 13.5 + GCC 15 −0.6%, NVCC 13.5 + Clang 22 +0.7%, NVRTC 13.5 −3.0%).

**78 of the 80 original feature workloads now compile at least 10% faster** than upstream in C++17, up from 60. All 20 targeted workloads improved, and 18 reached the threshold.

**Still below 10%.** `ranges::view` (4.8%), `views::empty` (9.7%). The `view` workload rejects 64 three-element tuple types. The `views::empty` workload queries five constant observers for 64 element types. The tuple-based `ranges::view` workload used 9.0% less CPU time with NVCC/GCC and 5.2% less with NVCC/Clang. Its NVRTC medians were 0.7027 -> 0.7034 seconds (+0.1%). [Direct comparisons with the previous headers](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_pr.md#direct-measurements-for-the-two-workloads-below-10)

The group means below give each original feature workload equal weight. The two include-only controls and the two additional pointer workloads appear in the detailed matrix.

| C++17 workload group | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: |
| 27 performance-core workloads | 6.473 -> 3.940 (−39.1%) | 6.569 -> 4.159 (−36.7%) | 1.191 -> 0.700 (−41.2%) |
| 53 efficiency-core workloads | 8.187 -> 6.015 (−26.5%) | 8.937 -> 6.529 (−26.9%) | 1.536 -> 1.054 (−31.4%) |

Median compiler time fell in 251 of 252 primary C++17 comparisons. The only increase was the 0.1% difference for `ranges::view` with NVRTC.

## Workloads previously below 10%

The compiler columns show median CPU seconds as **before -> after (percentage change)**, comparing upstream `ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae` with `44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97`. Negative deltas mean shorter compiles. The previous header revision was `e26bf74a572d2e5a8ea9d2ee30d01445a67c9504`. P and E identify performance and efficiency cores. Workload savings are 100 × (1 − the geometric mean of the three after/before ratios). The last column compares savings against upstream in the previous and current sweeps. Parentheses give the change in percentage points. The 10% threshold uses unrounded values.

| Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 | Mean savings, previous -> current |
| --- | --- | ---: | ---: | ---: | ---: |
| `views::transform` and `views::filter` closure composition | P | 14.362 -> 7.334 (−48.9%) | 14.820 -> 7.459 (−49.7%) | 3.003 -> 1.449 (−51.8%) | 0.8% -> 50.1% (+49.4 pp) |
| `ranges::enable_view` | E | 4.660 -> 4.063 (−12.8%) | 4.910 -> 4.285 (−12.7%) | 0.600 -> 0.539 (−10.1%) | 1.3% -> 11.9% (+10.6 pp) |
| `ranges::view` | P | 4.562 -> 4.150 (−9.0%) | 4.555 -> 4.320 (−5.2%) | 0.703 -> 0.703 (+0.1%) | 1.3% -> 4.8% (+3.4 pp) |
| `views::empty` | E | 4.947 -> 4.423 (−10.6%) | 5.087 -> 4.591 (−9.8%) | 0.753 -> 0.687 (−8.8%) | 1.7% -> 9.7% (+8.0 pp) |
| `ranges::swap` | E | 4.131 -> 3.537 (−14.4%) | 4.582 -> 3.942 (−14.0%) | 0.565 -> 0.491 (−13.1%) | 2.4% -> 13.8% (+11.4 pp) |
| `ranges::viewable_range` | P | 2.575 -> 2.147 (−16.6%) | 2.571 -> 2.232 (−13.2%) | 0.333 -> 0.301 (−9.5%) | 3.3% -> 13.2% (+9.9 pp) |
| `range_adaptor_closure` application | E | 4.810 -> 4.088 (−15.0%) | 5.313 -> 4.518 (−15.0%) | 0.689 -> 0.580 (−15.8%) | 3.6% -> 15.3% (+11.6 pp) |
| `views::indices` | E | 4.855 -> 4.022 (−17.2%) | 5.229 -> 4.321 (−17.4%) | 0.678 -> 0.547 (−19.3%) | 4.0% -> 18.0% (+14.0 pp) |
| `ranges::iter_move` (ADL customization) | P | 5.470 -> 3.913 (−28.5%) | 5.086 -> 4.074 (−19.9%) | 0.905 -> 0.698 (−22.9%) | 4.6% -> 23.8% (+19.3 pp) |
| `views::repeat` | E | 6.822 -> 5.634 (−17.4%) | 7.364 -> 5.989 (−18.7%) | 1.166 -> 0.936 (−19.7%) | 5.2% -> 18.6% (+13.4 pp) |
| `transform_view` traversal and observers | E | 7.094 -> 5.940 (−16.3%) | 7.643 -> 6.383 (−16.5%) | 1.263 -> 1.038 (−17.8%) | 6.0% -> 16.9% (+10.9 pp) |
| `views::drop_while` | E | 9.528 -> 7.303 (−23.3%) | 10.285 -> 7.801 (−24.2%) | 1.911 -> 1.411 (−26.2%) | 6.1% -> 24.6% (+18.4 pp) |
| `views::iota` | E | 8.070 -> 6.163 (−23.6%) | 8.819 -> 6.627 (−24.9%) | 1.454 -> 1.051 (−27.7%) | 6.3% -> 25.4% (+19.1 pp) |
| `ref_view` members | E | 5.958 -> 4.747 (−20.3%) | 6.520 -> 5.224 (−19.9%) | 0.972 -> 0.744 (−23.4%) | 6.6% -> 21.2% (+14.6 pp) |
| `views::filter` | E | 7.320 -> 5.943 (−18.8%) | 7.799 -> 6.329 (−18.8%) | 1.321 -> 1.051 (−20.4%) | 7.7% -> 19.4% (+11.7 pp) |
| `views::take_while` | E | 8.091 -> 6.493 (−19.7%) | 8.810 -> 7.000 (−20.5%) | 1.554 -> 1.205 (−22.5%) | 7.8% -> 20.9% (+13.1 pp) |
| `views::single` | E | 8.138 -> 6.442 (−20.8%) | 8.894 -> 6.914 (−22.3%) | 1.517 -> 1.151 (−24.2%) | 7.9% -> 22.4% (+14.5 pp) |
| `ranges::size` | P | 2.957 -> 2.099 (−29.0%) | 2.514 -> 2.131 (−15.2%) | 0.342 -> 0.310 (−9.4%) | 8.3% -> 18.3% (+10.0 pp) |
| `ranges::range_rvalue_reference_t` | E | 6.868 -> 5.344 (−22.2%) | 7.529 -> 5.783 (−23.2%) | 1.215 -> 0.890 (−26.8%) | 8.4% -> 24.1% (+15.7 pp) |
| `ranges::data` | P | 3.026 -> 2.077 (−31.3%) | 2.471 -> 2.121 (−14.1%) | 0.334 -> 0.288 (−13.9%) | 9.6% -> 20.2% (+10.6 pp) |

<details>
<summary>Other features, include-only controls and additional workloads</summary>

| Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 | Mean savings, previous -> current |
| --- | --- | ---: | ---: | ---: | ---: |
| `ranges::begin` | P | 3.913 -> 2.360 (−39.7%) | 3.614 -> 2.456 (−32.0%) | 0.604 -> 0.353 (−41.5%) | 31.3% -> 37.9% (+6.6 pp) |
| `ranges::end` | P | 4.552 -> 2.369 (−48.0%) | 4.435 -> 2.467 (−44.4%) | 0.784 -> 0.376 (−52.1%) | 43.8% -> 48.2% (+4.4 pp) |
| `ranges::cbegin` | E | 5.390 -> 4.050 (−24.9%) | 5.781 -> 4.363 (−24.5%) | 0.827 -> 0.565 (−31.7%) | 19.8% -> 27.1% (+7.3 pp) |
| `ranges::cend` | E | 5.743 -> 3.827 (−33.4%) | 5.660 -> 3.736 (−34.0%) | 0.952 -> 0.536 (−43.8%) | 30.2% -> 37.2% (+7.0 pp) |
| `ranges::rbegin` | P | 3.906 -> 2.337 (−40.2%) | 3.603 -> 2.402 (−33.3%) | 0.599 -> 0.349 (−41.8%) | 31.1% -> 38.6% (+7.4 pp) |
| `ranges::rend` | P | 4.565 -> 2.460 (−46.1%) | 4.440 -> 2.499 (−43.7%) | 0.804 -> 0.373 (−53.6%) | 43.1% -> 48.0% (+4.9 pp) |
| `ranges::crbegin` | E | 8.463 -> 5.830 (−31.1%) | 9.425 -> 6.476 (−31.3%) | 1.648 -> 1.050 (−36.3%) | 25.7% -> 32.9% (+7.2 pp) |
| `ranges::crend` | E | 8.389 -> 5.835 (−30.4%) | 9.438 -> 6.522 (−30.9%) | 1.674 -> 1.083 (−35.3%) | 26.1% -> 32.2% (+6.2 pp) |
| `ranges::cdata` | E | 7.740 -> 5.342 (−31.0%) | 8.596 -> 5.916 (−31.2%) | 1.595 -> 0.998 (−37.4%) | 26.3% -> 33.3% (+7.0 pp) |
| `ranges::empty` | P | 2.847 -> 1.960 (−31.1%) | 2.391 -> 2.031 (−15.1%) | 0.299 -> 0.265 (−11.5%) | 10.3% -> 19.7% (+9.4 pp) |
| `ranges::ssize` | E | 7.465 -> 4.933 (−33.9%) | 8.342 -> 5.456 (−34.6%) | 1.422 -> 0.837 (−41.1%) | 29.1% -> 36.6% (+7.6 pp) |
| `ranges::disable_sized_range` | E | 7.283 -> 4.811 (−33.9%) | 7.857 -> 5.205 (−33.8%) | 1.368 -> 0.792 (−42.1%) | 30.0% -> 36.7% (+6.8 pp) |
| `ranges::range` | E | 5.754 -> 3.755 (−34.7%) | 6.186 -> 4.009 (−35.2%) | 0.962 -> 0.527 (−45.2%) | 29.8% -> 38.6% (+8.8 pp) |
| `ranges::input_range` | E | 8.790 -> 6.157 (−30.0%) | 9.772 -> 6.764 (−30.8%) | 1.774 -> 1.151 (−35.1%) | 20.3% -> 32.0% (+11.7 pp) |
| `ranges::output_range` | E | 6.117 -> 4.003 (−34.5%) | 6.541 -> 4.289 (−34.4%) | 1.067 -> 0.596 (−44.1%) | 27.2% -> 37.9% (+10.7 pp) |
| `ranges::forward_range` | E | 9.576 -> 6.170 (−35.6%) | 10.663 -> 6.798 (−36.2%) | 1.978 -> 1.136 (−42.5%) | 27.0% -> 38.2% (+11.2 pp) |
| `ranges::bidirectional_range` | E | 16.796 -> 12.524 (−25.4%) | 18.948 -> 13.892 (−26.7%) | 3.957 -> 2.825 (−28.6%) | 14.0% -> 26.9% (+13.0 pp) |
| `ranges::random_access_range` | E | 15.746 -> 11.188 (−28.9%) | 17.624 -> 12.471 (−29.2%) | 3.560 -> 2.368 (−33.5%) | 20.7% -> 30.6% (+9.9 pp) |
| `ranges::contiguous_range` | P | 11.083 -> 7.440 (−32.9%) | 12.017 -> 8.307 (−30.9%) | 2.503 -> 1.616 (−35.5%) | 25.9% -> 33.1% (+7.2 pp) |
| `ranges::common_range` | E | 6.166 -> 4.194 (−32.0%) | 6.783 -> 4.548 (−33.0%) | 1.091 -> 0.658 (−39.7%) | 28.5% -> 35.0% (+6.5 pp) |
| `ranges::sized_range` | P | 4.125 -> 2.257 (−45.3%) | 4.175 -> 2.405 (−42.4%) | 0.715 -> 0.336 (−53.0%) | 40.1% -> 47.1% (+7.0 pp) |
| `ranges::borrowed_range` | E | 8.495 -> 5.890 (−30.7%) | 9.080 -> 6.232 (−31.4%) | 1.750 -> 1.079 (−38.4%) | 24.0% -> 33.6% (+9.6 pp) |
| `ranges::enable_borrowed_range` | E | 7.346 -> 4.858 (−33.9%) | 7.972 -> 5.203 (−34.7%) | 1.405 -> 0.810 (−42.3%) | 26.9% -> 37.1% (+10.2 pp) |
| `ranges::view_base` | E | 6.003 -> 3.831 (−36.2%) | 6.257 -> 4.051 (−35.3%) | 1.077 -> 0.583 (−45.9%) | 31.3% -> 39.3% (+8.0 pp) |
| `ranges::iterator_t` | E | 5.851 -> 3.897 (−33.4%) | 6.300 -> 4.187 (−33.5%) | 0.994 -> 0.567 (−42.9%) | 29.2% -> 36.8% (+7.6 pp) |
| `ranges::sentinel_t` | E | 10.373 -> 7.528 (−27.4%) | 11.283 -> 8.046 (−28.7%) | 2.247 -> 1.524 (−32.2%) | 23.2% -> 29.5% (+6.3 pp) |
| `ranges::range_difference_t` | P | 5.984 -> 3.377 (−43.6%) | 6.528 -> 3.605 (−44.8%) | 1.219 -> 0.618 (−49.3%) | 37.0% -> 45.9% (+9.0 pp) |
| `ranges::range_size_t` | P | 5.059 -> 2.336 (−53.8%) | 5.221 -> 2.464 (−52.8%) | 0.938 -> 0.361 (−61.5%) | 50.9% -> 56.2% (+5.3 pp) |
| `ranges::range_value_t` | E | 10.228 -> 6.987 (−31.7%) | 11.450 -> 7.735 (−32.4%) | 2.142 -> 1.349 (−37.0%) | 24.0% -> 33.8% (+9.8 pp) |
| `ranges::range_reference_t` | E | 10.401 -> 7.115 (−31.6%) | 11.484 -> 7.828 (−31.8%) | 2.181 -> 1.356 (−37.8%) | 24.6% -> 33.8% (+9.2 pp) |
| `ranges::range_common_reference_t` | E | 8.970 -> 6.626 (−26.1%) | 10.019 -> 7.274 (−27.4%) | 1.766 -> 1.233 (−30.2%) | 15.2% -> 27.9% (+12.8 pp) |
| `ranges::dangling` | E | 7.207 -> 5.247 (−27.2%) | 7.680 -> 5.580 (−27.3%) | 1.222 -> 0.825 (−32.4%) | 20.8% -> 29.0% (+8.2 pp) |
| `ranges::borrowed_iterator_t` | E | 8.438 -> 5.908 (−30.0%) | 9.404 -> 6.402 (−31.9%) | 1.636 -> 1.034 (−36.8%) | 24.0% -> 33.0% (+8.9 pp) |
| `ranges::borrowed_subrange_t` | P | 10.343 -> 3.203 (−69.0%) | 11.259 -> 3.337 (−70.4%) | 2.579 -> 0.619 (−76.0%) | 67.2% -> 72.0% (+4.8 pp) |
| `ranges::advance` | E | 11.893 -> 8.450 (−28.9%) | 13.210 -> 9.459 (−28.4%) | 2.557 -> 1.714 (−33.0%) | 21.9% -> 30.1% (+8.2 pp) |
| `ranges::distance` | E | 10.340 -> 7.168 (−30.7%) | 11.486 -> 7.880 (−31.4%) | 2.108 -> 1.350 (−36.0%) | 24.1% -> 32.7% (+8.7 pp) |
| `ranges::next` | E | 12.207 -> 8.635 (−29.3%) | 13.426 -> 9.651 (−28.1%) | 2.701 -> 1.825 (−32.4%) | 21.4% -> 30.0% (+8.5 pp) |
| `ranges::prev` | E | 12.005 -> 8.486 (−29.3%) | 13.535 -> 9.486 (−29.9%) | 2.660 -> 1.775 (−33.3%) | 21.1% -> 30.8% (+9.8 pp) |
| `ranges::iter_move` (raw pointer) | P | 4.383 -> 2.132 (−51.4%) | 4.651 -> 2.285 (−50.9%) | 0.733 -> 0.299 (−59.2%) | 49.9% -> 54.0% (+4.1 pp) |
| `ranges::iter_swap` (ADL customization) | P | 8.683 -> 4.074 (−53.1%) | 9.336 -> 4.311 (−53.8%) | 1.997 -> 0.745 (−62.7%) | 54.9% -> 56.8% (+1.9 pp) |
| `view_interface` | E | 7.909 -> 6.108 (−22.8%) | 8.851 -> 6.775 (−23.5%) | 1.548 -> 1.139 (−26.4%) | 17.6% -> 24.2% (+6.6 pp) |
| `owning_view` members | E | 7.415 -> 5.875 (−20.8%) | 8.211 -> 6.460 (−21.3%) | 1.324 -> 0.995 (−24.8%) | 15.3% -> 22.3% (+7.0 pp) |
| `subrange` construction and navigation | E | 10.217 -> 7.485 (−26.7%) | 11.440 -> 8.379 (−26.8%) | 1.977 -> 1.440 (−27.1%) | 14.1% -> 26.9% (+12.7 pp) |
| `subrange` tuple access | E | 8.473 -> 6.080 (−28.2%) | 9.180 -> 6.592 (−28.2%) | 1.457 -> 1.022 (−29.9%) | 12.6% -> 28.8% (+16.2 pp) |
| `views::all` | P | 13.555 -> 7.304 (−46.1%) | 14.590 -> 7.946 (−45.5%) | 3.190 -> 1.555 (−51.3%) | 40.7% -> 47.7% (+7.0 pp) |
| `views::all_t` | E | 7.914 -> 5.470 (−30.9%) | 8.682 -> 5.977 (−31.2%) | 1.607 -> 1.005 (−37.5%) | 26.3% -> 33.3% (+6.9 pp) |
| `views::counted` | E | 20.958 -> 16.530 (−21.1%) | 22.922 -> 17.893 (−21.9%) | 4.853 -> 3.726 (−23.2%) | 11.8% -> 22.1% (+10.3 pp) |
| `common_view::begin/end` | P | 8.865 -> 6.880 (−22.4%) | 9.613 -> 7.314 (−23.9%) | 1.872 -> 1.464 (−21.8%) | 14.4% -> 22.7% (+8.3 pp) |
| `views::common` | E | 10.705 -> 7.849 (−26.7%) | 11.610 -> 8.454 (−27.2%) | 2.131 -> 1.472 (−30.9%) | 13.1% -> 28.3% (+15.2 pp) |
| `reverse_view` construction | P | 12.104 -> 7.500 (−38.0%) | 13.441 -> 8.331 (−38.0%) | 2.843 -> 1.671 (−41.2%) | 30.9% -> 39.1% (+8.2 pp) |
| `views::reverse` | E | 13.495 -> 10.054 (−25.5%) | 14.947 -> 11.085 (−25.8%) | 2.996 -> 2.117 (−29.3%) | 18.1% -> 26.9% (+8.8 pp) |
| `views::take` | P | 15.310 -> 8.638 (−43.6%) | 16.938 -> 9.292 (−45.1%) | 3.637 -> 1.916 (−47.3%) | 40.4% -> 45.4% (+5.0 pp) |
| `take_view` traversal and observers | E | 8.482 -> 6.250 (−26.3%) | 9.339 -> 6.875 (−26.4%) | 1.619 -> 1.122 (−30.7%) | 15.6% -> 27.8% (+12.2 pp) |
| `views::drop` | P | 15.963 -> 10.950 (−31.4%) | 17.513 -> 11.724 (−33.1%) | 3.746 -> 2.473 (−34.0%) | 25.1% -> 32.8% (+7.7 pp) |
| `drop_view::begin` | P | 8.749 -> 6.185 (−29.3%) | 9.504 -> 6.675 (−29.8%) | 1.897 -> 1.289 (−32.0%) | 20.8% -> 30.4% (+9.5 pp) |
| `drop_view` traversal and observers | E | 8.549 -> 6.329 (−26.0%) | 9.297 -> 6.897 (−25.8%) | 1.670 -> 1.163 (−30.3%) | 15.5% -> 27.4% (+12.0 pp) |
| `views::transform` | P | 7.911 -> 5.677 (−28.2%) | 8.737 -> 5.963 (−31.8%) | 1.757 -> 1.180 (−32.8%) | 23.4% -> 31.0% (+7.6 pp) |
| `views::zip` | E | 16.298 -> 13.296 (−18.4%) | 17.530 -> 14.151 (−19.3%) | 3.775 -> 3.055 (−19.1%) | 13.9% -> 18.9% (+5.0 pp) |
| `zip_view::begin/end` | P | 8.003 -> 6.055 (−24.3%) | 8.516 -> 6.534 (−23.3%) | 1.731 -> 1.302 (−24.8%) | 16.6% -> 24.1% (+7.5 pp) |
| `zip_view::size` | P | 9.899 -> 6.113 (−38.2%) | 10.641 -> 6.565 (−38.3%) | 2.170 -> 1.244 (−42.7%) | 34.3% -> 39.8% (+5.4 pp) |
| `<cuda/std/ranges>` include only | P | 2.364 -> 1.943 (−17.8%) | 2.456 -> 2.031 (−17.3%) | 0.282 -> 0.239 (−15.2%) | — |
| `<cuda/std/ranges>` include only | E | 4.155 -> 3.459 (−16.8%) | 4.452 -> 3.696 (−17.0%) | 0.538 -> 0.433 (−19.6%) | — |
| `iota_view` bounded pointer construction | E | 7.632 -> 5.468 (−28.4%) | 8.398 -> 5.922 (−29.5%) | 1.425 -> 0.948 (−33.4%) | 10.2% -> 30.5% (+20.3 pp) |
| `ranges::swap` (raw pointers) | E | 9.181 -> 8.076 (−12.0%) | 9.754 -> 8.495 (−12.9%) | 1.660 -> 1.452 (−12.5%) | 12.5% (new workload) |

The bounded iota workload constructs and traverses a range of pointers during constant evaluation. The pointer swap workload checks that exchanging pointers to 64 distinct tuple specializations is non-throwing. The original ADL swap workload measures a custom overload.

</details>

<details>
<summary>What the workloads exercise</summary>

The inventory covers the public range facilities shipped by this libcu++ header, including its iterator operations and the `views::indices` extension. [Pinned libcu++ header](https://github.com/davebayer/cccl/blob/44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97/libcudacxx/include/cuda/std/ranges) · [Benchmark sources](https://github.com/davebayer/cccl/tree/44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97/libcudacxx/benchmarks/compile_time/ranges)

- **Access and size.** Member, ADL and iterator fallback selection, const access on a mutable range, signed sizing, and `disable_sized_range`.
- **Range requirements and associated types.** Iterator-category boundaries, common and distinct sentinels, borrowed and owning ranges, opting into views and borrowing, reference and value types, and dangling results.
- **Iterator operations.** Distance, bounded and unbounded navigation, element swap through a customization and built-in pointer exchange, custom iterator moves and swaps, and moving through a raw pointer.
- **Storage and view utilities.** The CRTP `view_interface` observers and element access (including `at`), `ref_view` and `owning_view` access and `base()`, sized and unsized `subrange_kind`, subrange construction and navigation, and the subrange tuple protocol (`get`, `tuple_size`, `tuple_element`).
- **Factories.** Empty and single views, bounded and unbounded iota and repeat views, the libcu++ `views::indices` extension, and all three dispatch paths for `views::counted`.
- **Adaptors.** Construction, the resulting view classes and their member operations for all, common, reverse, take, take-while, drop, drop-while, filter, transform and zip. Separate cases cover closure composition, piping a range into a user-defined `range_adaptor_closure`, and `views::all_t`. `cuda::std::views` aliases `cuda::std::ranges::views`.
- **Input counts.** Existing workloads use 24–64 distinct type indices. Most new probes use 24 or 32, and single and empty views use 64. The indices workload checks six standard integral types and rejection of a floating-point argument.

</details>

## Measurement and reproduction

The benchmarks ran on an Intel Core i9-14900K with CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each configuration used one warm-up per revision, followed by nine measured pairs in alternating order on one pinned physical core. Independent configurations ran on distinct cores within the two groups. NVCC time includes the compiler driver and child processes for `-O2 -c`. NVRTC time covers `nvrtcCompileProgram` with a fresh program and its cache disabled. All workloads compile without launching a kernel.

Filter the raw CSV to `standard=c++17` and series `cumulative3` for the primary sweep. The E-core include-only control uses `cumulative3-e-control`. Confirmations append `-repeat`. For the preceding sweep use `cumulative2` and `cumulative2-e-control`. A single translation unit includes both header parsing and template instantiation. [Per-change tables](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_pr.md#c17-compile-time-changes-for-the-10-target) · [Raw CPU-time samples](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_samples.csv) · [Previous C++17 report](https://github.com/davebayer/cccl/blob/871b9c1bef3dab5ff672982643c8787dd48b1443/reports/optimize_ranges_cumulative_pr.md)

Both header trees compile the same benchmark source from the measured revision. Choose the filename from the CSV and a CPU in the core class shown in the table.

```bash
base=ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae
head=44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97
workloads=44c5a12d265d1cad9df7b0d886c4c1b92e3b2a97
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

On the final measured headers, the eight targeted integration tests passed with both NVCC host compilers in both dialects on the RTX PRO 6000 (32 passes). Before the header-prologue change, the integrated ranges and iterator suites passed all 232 applicable configurations on the same GPU with NVCC 13.5 and `sm_120a`. Each of GCC 15 and Clang 22 passed 56 tests in C++17 and 60 in C++20. Four tests require native C++20 concepts. Targeted suites also covered closure construction, ADL eligibility, span bounds and tuple forwarding.

The generator reproduced both generated headers and the generated lit test. Twelve preprocessing comparisons across C++17 and C++20 produced identical tokens and pragmas, including restoration of caller-defined `min` and `max` macros. The generated lit test passed all four GPU configurations. Root-configured clang-tidy passed for all changed C++ headers, regression sources and the new benchmark in CUDA host and device modes for both dialects.
