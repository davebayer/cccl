# Reduce compile time for libcu++ range access, concepts, and view adaptors

## Summary

This PR reduces template instantiation in `<cuda/std/ranges>`. Range access and iterator customization points defer lower-priority probes until overload resolution needs them. Range concepts and aliases check their prerequisites before probing size, subtraction, or movability. Adaptors reuse known iterator and result types in operations such as `zip_view::size`.

Across all six compiler and language configurations, the `borrowed_subrange_t` workload used 49.3–70.2% less compiler CPU time, the ADL `iter_swap` workload used 33.6–62.2% less, and `zip_view::size` used 15.7–21.4% less. In a separate C++17 host-only check, GCC 15 emitted 9,212 class records for the baseline `borrowed_subrange_t` workload and 3,639 for the optimized headers (`-fdump-lang-class`).

The new `views::iota` and `views::drop_while` workloads ranged from 1.2% faster to 0.9% slower across the six configurations. Of the six C++20 cases initially more than 2.5% slower, the largest increase on repeat was 0.7%.

**C++17 `view_interface` regressed on all three compilers.** Independent repeats measured +4.2% with NVCC/GCC, +4.8% with NVCC/Clang, and +5.5% with NVRTC. This workload calls inherited observers and element access on a view with mutable and const `begin()` and `end()`. A separate NVRTC C++17 run on a P core localized part of the cost to the change in `ranges::begin` dispatch. Comparing that commit with its parent in both orders, the changed code took longer in all ten measured pairs. [Change to C++17 member and ADL lookup](https://github.com/davebayer/cccl/commit/94d82ac3dc745fde26820c24aadcda880b324bc4)

The summary uses the geometric mean of the per-workload median compiler CPU seconds. Each feature workload has equal weight within its group. The include-only controls are reported separately in the full table.

| Workloads | C++ | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: | ---: |
| 27 existing workloads | 17 | 6.440 -> 5.296 (−17.8%) | 6.533 -> 5.600 (−14.3%) | 1.193 -> 0.981 (−17.7%) |
| 27 existing workloads | 20 | 3.983 -> 3.707 (−6.9%) | 4.651 -> 4.080 (−12.3%) | 0.573 -> 0.518 (−9.7%) |
| 53 additional workloads | 17 | 8.041 -> 7.934 (−1.3%) | 8.811 -> 8.712 (−1.1%) | 1.510 -> 1.484 (−1.7%) |
| 53 additional workloads | 20 | 5.335 -> 5.321 (−0.3%) | 6.267 -> 6.255 (−0.2%) | 0.698 -> 0.695 (−0.5%) |

## Cumulative compile-time results

Every cell compares headers at upstream `ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae` with the requested branch HEAD, `372cf1d3294b7badac9eb7eb1f44930d2ae6631a`. Both sides compile identical benchmark source. Values are median compiler CPU seconds in the form **before -> after (percentage change)**. Negative percentages mean shorter compiles.

**P** marks the 27 existing workloads measured on performance cores. **E** marks 53 new workloads covering the other public facilities, measured on efficiency cores. Each workload measures the whole translation unit, and some exercise several members or overloads of one facility. The include-only control ran on both core classes and gives context for small changes. All reported times include header parsing. Compare revisions within each cell since absolute times also depend on the core class.

| Workload | C++ | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | --- | ---: | ---: | ---: |
| `ranges::begin` | 17 | P | 3.737 -> 3.362 (−10.0%) | 3.505 -> 3.509 (+0.1%) | 0.612 -> 0.597 (−2.5%) |
| `ranges::begin` | 20 | P | 2.532 -> 2.530 (−0.1%) | 2.670 -> 2.756 (+3.2%) | 0.326 -> 0.328 (+0.5%) |
| `ranges::end` | 17 | P | 4.560 -> 4.070 (−10.7%) | 4.448 -> 4.361 (−2.0%) | 0.806 -> 0.779 (−3.4%) |
| `ranges::end` | 20 | P | 2.749 -> 2.692 (−2.1%) | 2.850 -> 2.841 (−0.3%) | 0.357 -> 0.351 (−1.6%) |
| `ranges::cbegin` | 17 | E | 4.946 -> 4.920 (−0.5%) | 5.321 -> 5.314 (−0.1%) | 0.758 -> 0.746 (−1.7%) |
| `ranges::cbegin` | 20 | E | 4.063 -> 4.074 (+0.3%) | 4.640 -> 4.634 (−0.1%) | 0.435 -> 0.437 (+0.4%) |
| `ranges::cend` | 17 | E | 5.787 -> 5.684 (−1.8%) | 6.299 -> 6.184 (−1.8%) | 0.973 -> 0.950 (−2.4%) |
| `ranges::cend` | 20 | E | 4.214 -> 4.189 (−0.6%) | 5.042 -> 5.022 (−0.4%) | 0.472 -> 0.468 (−0.9%) |
| `ranges::rbegin` | 17 | P | 3.935 -> 3.379 (−14.1%) | 3.563 -> 3.493 (−2.0%) | 0.609 -> 0.579 (−5.0%) |
| `ranges::rbegin` | 20 | P | 2.548 -> 2.538 (−0.4%) | 2.636 -> 2.612 (−0.9%) | 0.313 -> 0.308 (−1.5%) |
| `ranges::rend` | 17 | P | 4.563 -> 4.130 (−9.5%) | 4.403 -> 4.363 (−0.9%) | 0.820 -> 0.791 (−3.5%) |
| `ranges::rend` | 20 | P | 2.690 -> 2.671 (−0.7%) | 2.879 -> 2.881 (+0.1%) | 0.339 -> 0.349 (+2.8%) |
| `ranges::crbegin` | 17 | E | 8.096 -> 7.923 (−2.1%) | 8.958 -> 8.756 (−2.3%) | 1.577 -> 1.519 (−3.7%) |
| `ranges::crbegin` | 20 | E | 5.126 -> 5.106 (−0.4%) | 6.897 -> 6.816 (−1.2%) | 0.685 -> 0.681 (−0.6%) |
| `ranges::crend` | 17 | E | 8.310 -> 8.152 (−1.9%) | 9.162 -> 8.980 (−2.0%) | 1.655 -> 1.604 (−3.1%) |
| `ranges::crend` | 20 | E | 5.138 -> 5.074 (−1.3%) | 6.985 -> 6.983 (0.0%) | 0.697 -> 0.691 (−0.9%) |
| `ranges::data` | 17 | P | 3.059 -> 2.399 (−21.6%) | 2.532 -> 2.471 (−2.4%) | 0.359 -> 0.336 (−6.3%) |
| `ranges::data` | 20 | P | 2.220 -> 2.244 (+1.1%) | 2.396 -> 2.300 (−4.0%) | 0.229 -> 0.224 (−2.0%) |
| `ranges::cdata` | 17 | E | 7.763 -> 7.589 (−2.2%) | 8.503 -> 8.315 (−2.2%) | 1.491 -> 1.444 (−3.1%) |
| `ranges::cdata` | 20 | E | 4.792 -> 4.789 (−0.1%) | 6.189 -> 6.113 (−1.2%) | 0.620 -> 0.619 (−0.2%) |
| `ranges::empty` | 17 | P | 2.950 -> 2.315 (−21.5%) | 2.499 -> 2.403 (−3.8%) | 0.331 -> 0.323 (−2.4%) |
| `ranges::empty` | 20 | P | 2.186 -> 2.177 (−0.4%) | 2.238 -> 2.223 (−0.6%) | 0.207 -> 0.206 (−0.6%) |
| `ranges::size` | 17 | P | 2.958 -> 2.385 (−19.4%) | 2.531 -> 2.483 (−1.9%) | 0.343 -> 0.315 (−8.2%) |
| `ranges::size` | 20 | P | 2.176 -> 2.143 (−1.5%) | 4.150 -> 2.323 (−44.0%) | 0.206 -> 0.204 (−1.2%) |
| `ranges::ssize` | 17 | E | 7.348 -> 7.177 (−2.3%) | 8.225 -> 8.207 (−0.2%) | 1.396 -> 1.378 (−1.3%) |
| `ranges::ssize` | 20 | E | 4.793 -> 4.809 (+0.3%) | 5.357 -> 5.363 (+0.1%) | 0.611 -> 0.615 (+0.5%) |
| `ranges::disable_sized_range` | 17 | E | 7.356 -> 7.145 (−2.9%) | 8.144 -> 7.899 (−3.0%) | 1.388 -> 1.346 (−3.1%) |
| `ranges::disable_sized_range` | 20 | E | 4.753 -> 4.712 (−0.9%) | 5.382 -> 5.335 (−0.9%) | 0.616 -> 0.614 (−0.4%) |
| `ranges::range` | 17 | E | 6.171 -> 6.052 (−1.9%) | 6.640 -> 6.537 (−1.5%) | 1.012 -> 0.988 (−2.4%) |
| `ranges::range` | 20 | E | 4.396 -> 4.384 (−0.3%) | 4.839 -> 4.843 (+0.1%) | 0.473 -> 0.474 (+0.2%) |
| `ranges::input_range` | 17 | E | 8.831 -> 8.628 (−2.3%) | 9.842 -> 9.608 (−2.4%) | 1.767 -> 1.703 (−3.6%) |
| `ranges::input_range` | 20 | E | 5.827 -> 5.803 (−0.4%) | 6.726 -> 6.660 (−1.0%) | 0.762 -> 0.748 (−1.8%) |
| `ranges::output_range` | 17 | E | 6.524 -> 6.503 (−0.3%) | 7.076 -> 7.012 (−0.9%) | 1.132 -> 1.136 (+0.3%) |
| `ranges::output_range` | 20 | E | 4.713 -> 4.718 (+0.1%) | 5.201 -> 5.184 (−0.3%) | 0.579 -> 0.578 (−0.2%) |
| `ranges::forward_range` | 17 | E | 10.240 -> 9.931 (−3.0%) | 11.424 -> 11.136 (−2.5%) | 2.112 -> 2.047 (−3.1%) |
| `ranges::forward_range` | 20 | E | 5.894 -> 5.849 (−0.8%) | 7.029 -> 7.012 (−0.2%) | 0.849 -> 0.830 (−2.2%) |
| `ranges::bidirectional_range` | 17 | E | 17.027 -> 16.687 (−2.0%) | 18.990 -> 18.519 (−2.5%) | 4.058 -> 3.870 (−4.6%) |
| `ranges::bidirectional_range` | 20 | E | 7.893 -> 7.863 (−0.4%) | 10.696 -> 10.661 (−0.3%) | 1.302 -> 1.291 (−0.9%) |
| `ranges::random_access_range` | 17 | E | 15.888 -> 15.465 (−2.7%) | 17.930 -> 17.489 (−2.5%) | 3.690 -> 3.578 (−3.0%) |
| `ranges::random_access_range` | 20 | E | 8.162 -> 8.047 (−1.4%) | 10.026 -> 10.031 (0.0%) | 1.239 -> 1.225 (−1.1%) |
| `ranges::contiguous_range` | 17 | P | 11.088 -> 10.719 (−3.3%) | 12.183 -> 11.638 (−4.5%) | 2.575 -> 2.473 (−4.0%) |
| `ranges::contiguous_range` | 20 | P | 5.324 -> 5.213 (−2.1%) | 6.102 -> 6.044 (−1.0%) | 0.912 -> 0.949 (+4.1%) |
| `ranges::common_range` | 17 | E | 6.839 -> 6.728 (−1.6%) | 7.306 -> 7.183 (−1.7%) | 1.203 -> 1.169 (−2.8%) |
| `ranges::common_range` | 20 | E | 4.534 -> 4.492 (−0.9%) | 4.998 -> 4.977 (−0.4%) | 0.534 -> 0.530 (−0.9%) |
| `ranges::sized_range` | 17 | P | 4.060 -> 3.843 (−5.3%) | 4.180 -> 4.133 (−1.1%) | 0.717 -> 0.715 (−0.4%) |
| `ranges::sized_range` | 20 | P | 2.579 -> 2.588 (+0.4%) | 2.785 -> 2.773 (−0.5%) | 0.331 -> 0.336 (+1.8%) |
| `ranges::borrowed_range` | 17 | E | 8.685 -> 8.639 (−0.5%) | 9.617 -> 9.536 (−0.8%) | 1.730 -> 1.705 (−1.4%) |
| `ranges::borrowed_range` | 20 | E | 5.438 -> 5.436 (0.0%) | 5.849 -> 5.829 (−0.3%) | 0.705 -> 0.715 (+1.3%) |
| `ranges::enable_borrowed_range` | 17 | E | 7.728 -> 7.572 (−2.0%) | 8.301 -> 8.333 (+0.4%) | 1.500 -> 1.490 (−0.7%) |
| `ranges::enable_borrowed_range` | 20 | E | 5.013 -> 4.969 (−0.9%) | 5.506 -> 5.526 (+0.4%) | 0.655 -> 0.655 (0.0%) |
| `ranges::view` | 17 | P | 4.428 -> 4.297 (−3.0%) | 4.440 -> 4.435 (−0.1%) | 0.686 -> 0.683 (−0.4%) |
| `ranges::view` | 20 | P | 4.062 -> 3.917 (−3.6%) | 4.039 -> 4.222 (+4.5%) | 0.551 -> 0.545 (−1.0%) |
| `ranges::viewable_range` | 17 | P | 2.485 -> 2.387 (−4.0%) | 2.500 -> 2.461 (−1.6%) | 0.333 -> 0.331 (−0.5%) |
| `ranges::viewable_range` | 20 | P | 2.152 -> 2.154 (+0.1%) | 2.286 -> 2.284 (−0.1%) | 0.204 -> 0.205 (+0.6%) |
| `ranges::enable_view` | 17 | E | 4.419 -> 4.390 (−0.7%) | 4.772 -> 4.664 (−2.3%) | 0.615 -> 0.608 (−1.2%) |
| `ranges::enable_view` | 20 | E | 3.968 -> 3.961 (−0.2%) | 4.286 -> 4.266 (−0.5%) | 0.373 -> 0.368 (−1.1%) |
| `ranges::view_base` | 17 | E | 6.562 -> 6.469 (−1.4%) | 7.172 -> 7.039 (−1.9%) | 1.203 -> 1.184 (−1.6%) |
| `ranges::view_base` | 20 | E | 4.489 -> 4.501 (+0.3%) | 4.823 -> 4.796 (−0.6%) | 0.531 -> 0.531 (0.0%) |
| `ranges::iterator_t` | 17 | E | 6.316 -> 6.237 (−1.3%) | 6.787 -> 6.693 (−1.4%) | 1.112 -> 1.090 (−1.9%) |
| `ranges::iterator_t` | 20 | E | 4.820 -> 4.811 (−0.2%) | 5.092 -> 5.057 (−0.7%) | 0.581 -> 0.585 (+0.7%) |
| `ranges::sentinel_t` | 17 | E | 10.867 -> 10.653 (−2.0%) | 11.946 -> 11.650 (−2.5%) | 2.201 -> 2.174 (−1.2%) |
| `ranges::sentinel_t` | 20 | E | 7.424 -> 7.334 (−1.2%) | 9.365 -> 9.398 (+0.4%) | 0.904 -> 0.901 (−0.4%) |
| `ranges::range_difference_t` | 17 | P | 6.258 -> 4.328 (−30.8%) | 6.678 -> 4.710 (−29.5%) | 1.287 -> 0.818 (−36.4%) |
| `ranges::range_difference_t` | 20 | P | 3.556 -> 2.990 (−15.9%) | 3.919 -> 3.210 (−18.1%) | 0.515 -> 0.365 (−29.0%) |
| `ranges::range_size_t` | 17 | P | 5.034 -> 3.975 (−21.0%) | 5.294 -> 4.313 (−18.5%) | 0.906 -> 0.705 (−22.2%) |
| `ranges::range_size_t` | 20 | P | 2.731 -> 2.696 (−1.3%) | 2.903 -> 3.004 (+3.5%) | 0.342 -> 0.336 (−1.8%) |
| `ranges::range_value_t` | 17 | E | 10.676 -> 10.441 (−2.2%) | 12.224 -> 11.913 (−2.5%) | 2.265 -> 2.195 (−3.1%) |
| `ranges::range_value_t` | 20 | E | 6.057 -> 5.995 (−1.0%) | 6.800 -> 6.731 (−1.0%) | 0.914 -> 0.892 (−2.4%) |
| `ranges::range_reference_t` | 17 | E | 11.103 -> 10.789 (−2.8%) | 12.183 -> 11.897 (−2.3%) | 2.275 -> 2.210 (−2.9%) |
| `ranges::range_reference_t` | 20 | E | 6.114 -> 6.056 (−1.0%) | 6.896 -> 6.845 (−0.7%) | 0.915 -> 0.898 (−1.9%) |
| `ranges::range_rvalue_reference_t` | 17 | E | 6.957 -> 6.698 (−3.7%) | 7.537 -> 7.270 (−3.5%) | 1.231 -> 1.197 (−2.8%) |
| `ranges::range_rvalue_reference_t` | 20 | E | 4.793 -> 4.757 (−0.8%) | 5.394 -> 5.322 (−1.3%) | 0.566 -> 0.565 (0.0%) |
| `ranges::range_common_reference_t` | 17 | E | 8.915 -> 8.659 (−2.9%) | 10.086 -> 9.748 (−3.3%) | 1.813 -> 1.689 (−6.8%) |
| `ranges::range_common_reference_t` | 20 | E | 5.672 -> 5.625 (−0.8%) | 6.536 -> 6.497 (−0.6%) | 0.820 -> 0.804 (−1.9%) |
| `ranges::dangling` | 17 | E | 6.723 -> 6.579 (−2.1%) | 7.265 -> 7.225 (−0.6%) | 1.212 -> 1.187 (−2.0%) |
| `ranges::dangling` | 20 | E | 4.802 -> 4.794 (−0.2%) | 5.355 -> 5.468 (+2.1%) | 0.639 -> 0.637 (−0.4%) |
| `ranges::borrowed_iterator_t` | 17 | E | 8.774 -> 8.684 (−1.0%) | 9.526 -> 9.418 (−1.1%) | 1.778 -> 1.758 (−1.1%) |
| `ranges::borrowed_iterator_t` | 20 | E | 5.431 -> 5.436 (+0.1%) | 5.887 -> 5.908 (+0.4%) | 0.736 -> 0.736 (−0.1%) |
| `ranges::borrowed_subrange_t` | 17 | P | 11.231 -> 4.432 (−60.5%) | 12.275 -> 4.531 (−63.1%) | 2.754 -> 0.820 (−70.2%) |
| `ranges::borrowed_subrange_t` | 20 | P | 5.533 -> 2.807 (−49.3%) | 6.308 -> 3.101 (−50.8%) | 1.015 -> 0.366 (−63.9%) |
| `ranges::advance` | 17 | E | 12.101 -> 11.806 (−2.4%) | 14.035 -> 13.775 (−1.9%) | 2.643 -> 2.581 (−2.3%) |
| `ranges::advance` | 20 | E | 6.619 -> 6.611 (−0.1%) | 7.843 -> 7.776 (−0.9%) | 1.045 -> 1.033 (−1.1%) |
| `ranges::distance` | 17 | E | 10.905 -> 10.719 (−1.7%) | 12.208 -> 11.996 (−1.7%) | 2.404 -> 2.356 (−2.0%) |
| `ranges::distance` | 20 | E | 6.287 -> 6.208 (−1.3%) | 7.157 -> 7.067 (−1.3%) | 0.933 -> 0.913 (−2.2%) |
| `ranges::next` | 17 | E | 12.169 -> 11.919 (−2.1%) | 14.168 -> 13.898 (−1.9%) | 2.688 -> 2.609 (−2.9%) |
| `ranges::next` | 20 | E | 6.712 -> 6.658 (−0.8%) | 8.106 -> 7.961 (−1.8%) | 1.112 -> 1.087 (−2.2%) |
| `ranges::prev` | 17 | E | 12.062 -> 11.846 (−1.8%) | 13.647 -> 13.317 (−2.4%) | 2.817 -> 2.754 (−2.2%) |
| `ranges::prev` | 20 | E | 6.609 -> 6.573 (−0.5%) | 7.878 -> 7.801 (−1.0%) | 1.123 -> 1.118 (−0.4%) |
| `ranges::swap` | 17 | E | 3.945 -> 3.951 (+0.2%) | 4.184 -> 4.230 (+1.1%) | 0.509 -> 0.501 (−1.4%) |
| `ranges::swap` | 20 | E | 3.607 -> 3.586 (−0.6%) | 3.912 -> 3.998 (+2.2%) | 0.329 -> 0.327 (−0.7%) |
| `ranges::iter_move` (ADL customization) | 17 | P | 5.577 -> 4.999 (−10.4%) | 5.252 -> 5.152 (−1.9%) | 0.924 -> 0.889 (−3.7%) |
| `ranges::iter_move` (ADL customization) | 20 | P | 4.530 -> 4.689 (+3.5%) | 9.072 -> 4.967 (−45.2%) | 0.735 -> 0.743 (+1.2%) |
| `ranges::iter_move` (raw pointer) | 17 | P | 4.467 -> 2.469 (−44.7%) | 4.637 -> 2.502 (−46.1%) | 0.748 -> 0.334 (−55.4%) |
| `ranges::iter_move` (raw pointer) | 20 | P | 2.179 -> 2.177 (−0.1%) | 2.322 -> 2.347 (+1.1%) | 0.214 -> 0.207 (−2.9%) |
| `ranges::iter_swap` (ADL customization) | 17 | P | 9.545 -> 4.621 (−51.6%) | 10.255 -> 4.988 (−51.4%) | 2.132 -> 0.806 (−62.2%) |
| `ranges::iter_swap` (ADL customization) | 20 | P | 6.436 -> 4.271 (−33.6%) | 7.196 -> 4.537 (−36.9%) | 1.148 -> 0.585 (−49.1%) |
| `view_interface` | 17 | E | 7.429 -> 7.731 (+4.1%) | 8.225 -> 8.612 (+4.7%) | 1.384 -> 1.461 (+5.6%) |
| `view_interface` | 20 | E | 4.745 -> 4.758 (+0.3%) | 6.794 -> 6.830 (+0.5%) | 0.545 -> 0.544 (−0.2%) |
| `ref_view` members | 17 | E | 5.601 -> 5.539 (−1.1%) | 6.040 -> 6.089 (+0.8%) | 0.902 -> 0.907 (+0.5%) |
| `ref_view` members | 20 | E | 4.361 -> 4.374 (+0.3%) | 5.230 -> 5.188 (−0.8%) | 0.476 -> 0.480 (+0.9%) |
| `owning_view` members | 17 | E | 6.844 -> 6.764 (−1.2%) | 7.481 -> 7.383 (−1.3%) | 1.228 -> 1.206 (−1.8%) |
| `owning_view` members | 20 | E | 4.831 -> 4.840 (+0.2%) | 6.025 -> 6.063 (+0.6%) | 0.585 -> 0.586 (+0.2%) |
| `subrange` construction and navigation | 17 | E | 9.617 -> 9.724 (+1.1%) | 10.603 -> 10.690 (+0.8%) | 1.777 -> 1.756 (−1.2%) |
| `subrange` construction and navigation | 20 | E | 5.724 -> 5.715 (−0.2%) | 6.752 -> 6.744 (−0.1%) | 0.796 -> 0.784 (−1.6%) |
| `subrange` tuple access | 17 | E | 7.768 -> 7.946 (+2.3%) | 8.441 -> 8.591 (+1.8%) | 1.340 -> 1.336 (−0.3%) |
| `subrange` tuple access | 20 | E | 6.879 -> 6.863 (−0.2%) | 7.666 -> 7.662 (−0.1%) | 1.093 -> 1.099 (+0.6%) |
| `views::all` | 17 | P | 14.234 -> 10.983 (−22.8%) | 14.472 -> 11.447 (−20.9%) | 2.881 -> 2.180 (−24.3%) |
| `views::all` | 20 | P | 6.716 -> 6.412 (−4.5%) | 7.473 -> 7.405 (−0.9%) | 1.319 -> 1.323 (+0.3%) |
| `views::all_t` | 17 | E | 8.588 -> 8.451 (−1.6%) | 9.296 -> 9.129 (−1.8%) | 1.642 -> 1.600 (−2.6%) |
| `views::all_t` | 20 | E | 5.392 -> 5.399 (+0.1%) | 5.969 -> 6.005 (+0.6%) | 0.700 -> 0.697 (−0.4%) |
| `range_adaptor_closure` application | 17 | E | 4.545 -> 4.467 (−1.7%) | 4.909 -> 4.863 (−0.9%) | 0.642 -> 0.630 (−2.0%) |
| `range_adaptor_closure` application | 20 | E | 3.882 -> 3.879 (−0.1%) | 4.237 -> 4.240 (+0.1%) | 0.392 -> 0.395 (+0.8%) |
| `views::transform` and `views::filter` closure composition | 17 | P | 12.397 -> 12.111 (−2.3%) | 13.150 -> 12.913 (−1.8%) | 2.320 -> 2.315 (−0.3%) |
| `views::transform` and `views::filter` closure composition | 20 | P | 12.075 -> 11.937 (−1.1%) | 13.031 -> 12.589 (−3.4%) | 2.170 -> 2.165 (−0.2%) |
| `views::empty` | 17 | E | 4.710 -> 4.643 (−1.4%) | 4.995 -> 4.945 (−1.0%) | 0.672 -> 0.663 (−1.3%) |
| `views::empty` | 20 | E | 4.061 -> 4.046 (−0.4%) | 4.410 -> 4.396 (−0.3%) | 0.420 -> 0.417 (−0.8%) |
| `views::single` | 17 | E | 7.601 -> 7.548 (−0.7%) | 8.226 -> 8.179 (−0.6%) | 1.412 -> 1.393 (−1.3%) |
| `views::single` | 20 | E | 5.211 -> 5.171 (−0.8%) | 5.491 -> 5.514 (+0.4%) | 0.678 -> 0.670 (−1.1%) |
| `views::iota` | 17 | E | 7.422 -> 7.392 (−0.4%) | 8.135 -> 8.075 (−0.7%) | 1.341 -> 1.333 (−0.6%) |
| `views::iota` | 20 | E | 4.828 -> 4.834 (+0.1%) | 5.486 -> 5.506 (+0.3%) | 0.607 -> 0.600 (−1.2%) |
| `views::indices` | 17 | E | 4.585 -> 4.543 (−0.9%) | 5.069 -> 5.011 (−1.2%) | 0.652 -> 0.643 (−1.4%) |
| `views::indices` | 20 | E | 3.858 -> 3.863 (+0.1%) | 4.249 -> 4.273 (+0.6%) | 0.376 -> 0.372 (−1.0%) |
| `views::repeat` | 17 | E | 6.355 -> 6.306 (−0.8%) | 6.948 -> 6.902 (−0.7%) | 1.085 -> 1.089 (+0.3%) |
| `views::repeat` | 20 | E | 4.672 -> 4.678 (+0.1%) | 5.227 -> 5.232 (+0.1%) | 0.571 -> 0.573 (+0.4%) |
| `views::counted` | 17 | E | 19.441 -> 19.048 (−2.0%) | 21.391 -> 21.029 (−1.7%) | 4.725 -> 4.640 (−1.8%) |
| `views::counted` | 20 | E | 9.281 -> 9.329 (+0.5%) | 15.017 -> 14.878 (−0.9%) | 1.598 -> 1.596 (−0.2%) |
| `common_view::begin/end` | 17 | P | 9.101 -> 8.840 (−2.9%) | 10.032 -> 9.410 (−6.2%) | 2.056 -> 1.939 (−5.7%) |
| `common_view::begin/end` | 20 | P | 5.398 -> 5.183 (−4.0%) | 6.877 -> 6.529 (−5.1%) | 0.828 -> 0.779 (−5.9%) |
| `views::common` | 17 | E | 9.930 -> 9.808 (−1.2%) | 10.805 -> 10.673 (−1.2%) | 1.974 -> 1.965 (−0.5%) |
| `views::common` | 20 | E | 5.970 -> 6.008 (+0.6%) | 7.051 -> 7.032 (−0.3%) | 0.897 -> 0.887 (−1.1%) |
| `reverse_view` construction | 17 | P | 12.704 -> 11.468 (−9.7%) | 13.721 -> 12.448 (−9.3%) | 3.053 -> 2.681 (−12.2%) |
| `reverse_view` construction | 20 | P | 5.880 -> 5.614 (−4.5%) | 7.191 -> 6.667 (−7.3%) | 1.070 -> 1.001 (−6.5%) |
| `views::reverse` | 17 | E | 12.440 -> 12.217 (−1.8%) | 13.807 -> 13.544 (−1.9%) | 2.676 -> 2.629 (−1.7%) |
| `views::reverse` | 20 | E | 6.708 -> 6.676 (−0.5%) | 8.797 -> 8.632 (−1.9%) | 1.040 -> 1.038 (−0.2%) |
| `views::take` | 17 | P | 12.869 -> 10.645 (−17.3%) | 13.944 -> 11.416 (−18.1%) | 2.960 -> 2.376 (−19.7%) |
| `views::take` | 20 | P | 5.857 -> 5.085 (−13.2%) | 8.001 -> 5.944 (−25.7%) | 1.108 -> 0.937 (−15.4%) |
| `take_view` traversal and observers | 17 | E | 7.716 -> 7.555 (−2.1%) | 8.118 -> 7.952 (−2.0%) | 1.518 -> 1.472 (−3.0%) |
| `take_view` traversal and observers | 20 | E | 4.856 -> 4.906 (+1.0%) | 6.064 -> 6.041 (−0.4%) | 0.639 -> 0.643 (+0.5%) |
| `views::take_while` | 17 | E | 7.576 -> 7.625 (+0.7%) | 8.191 -> 8.263 (+0.9%) | 1.470 -> 1.465 (−0.4%) |
| `views::take_while` | 20 | E | 5.026 -> 5.011 (−0.3%) | 6.330 -> 6.404 (+1.2%) | 0.676 -> 0.681 (+0.7%) |
| `views::drop` | 17 | P | 13.323 -> 12.988 (−2.5%) | 14.529 -> 13.902 (−4.3%) | 3.057 -> 2.918 (−4.6%) |
| `views::drop` | 20 | P | 6.012 -> 5.813 (−3.3%) | 8.145 -> 6.801 (−16.5%) | 1.146 -> 1.120 (−2.3%) |
| `drop_view::begin` | 17 | P | 9.023 -> 8.148 (−9.7%) | 9.502 -> 8.474 (−10.8%) | 2.046 -> 1.786 (−12.7%) |
| `drop_view::begin` | 20 | P | 4.664 -> 4.348 (−6.8%) | 6.549 -> 5.840 (−10.8%) | 0.763 -> 0.697 (−8.7%) |
| `drop_view` traversal and observers | 17 | E | 7.947 -> 7.800 (−1.8%) | 8.487 -> 8.640 (+1.8%) | 1.512 -> 1.478 (−2.2%) |
| `drop_view` traversal and observers | 20 | E | 5.123 -> 5.161 (+0.7%) | 6.515 -> 6.655 (+2.1%) | 0.639 -> 0.646 (+1.2%) |
| `views::drop_while` | 17 | E | 8.940 -> 8.947 (+0.1%) | 9.693 -> 9.691 (0.0%) | 1.798 -> 1.815 (+0.9%) |
| `views::drop_while` | 20 | E | 5.808 -> 5.832 (+0.4%) | 7.019 -> 7.052 (+0.5%) | 0.890 -> 0.890 (0.0%) |
| `views::filter` | 17 | E | 6.695 -> 6.743 (+0.7%) | 7.281 -> 7.352 (+1.0%) | 1.189 -> 1.192 (+0.3%) |
| `views::filter` | 20 | E | 4.965 -> 4.981 (+0.3%) | 6.179 -> 6.205 (+0.4%) | 0.651 -> 0.656 (+0.7%) |
| `views::transform` | 17 | P | 8.073 -> 7.629 (−5.5%) | 8.622 -> 8.346 (−3.2%) | 1.798 -> 1.719 (−4.4%) |
| `views::transform` | 20 | P | 4.479 -> 4.414 (−1.4%) | 4.891 -> 4.676 (−4.4%) | 0.701 -> 0.691 (−1.5%) |
| `transform_view` traversal and observers | 17 | E | 6.191 -> 6.214 (+0.4%) | 6.619 -> 6.687 (+1.0%) | 1.163 -> 1.175 (+1.1%) |
| `transform_view` traversal and observers | 20 | E | 4.746 -> 4.774 (+0.6%) | 5.716 -> 5.697 (−0.3%) | 0.633 -> 0.639 (+1.0%) |
| `views::zip` | 17 | E | 14.958 -> 14.594 (−2.4%) | 16.368 -> 16.003 (−2.2%) | 3.484 -> 3.381 (−3.0%) |
| `views::zip` | 20 | E | 11.454 -> 11.331 (−1.1%) | 12.342 -> 12.197 (−1.2%) | 2.129 -> 2.096 (−1.6%) |
| `zip_view::begin/end` | 17 | P | 8.149 -> 7.924 (−2.8%) | 8.831 -> 8.328 (−5.7%) | 1.857 -> 1.771 (−4.6%) |
| `zip_view::begin/end` | 20 | P | 6.065 -> 5.876 (−3.1%) | 6.545 -> 6.436 (−1.7%) | 1.070 -> 1.041 (−2.7%) |
| `zip_view::size` | 17 | P | 9.773 -> 8.241 (−15.7%) | 11.068 -> 9.176 (−17.1%) | 2.237 -> 1.847 (−17.4%) |
| `zip_view::size` | 20 | P | 6.784 -> 5.718 (−15.7%) | 7.309 -> 5.980 (−18.2%) | 1.233 -> 0.969 (−21.4%) |
| `<cuda/std/ranges>` include only | 17 | P | 2.261 -> 2.237 (−1.1%) | 2.334 -> 2.317 (−0.7%) | 0.297 -> 0.288 (−2.9%) |
| `<cuda/std/ranges>` include only | 20 | P | 2.035 -> 2.063 (+1.4%) | 2.134 -> 2.120 (−0.7%) | 0.185 -> 0.186 (+1.0%) |
| `<cuda/std/ranges>` include only | 17 | E | 3.525 -> 3.503 (−0.6%) | 3.775 -> 3.727 (−1.3%) | 0.440 -> 0.431 (−2.0%) |
| `<cuda/std/ranges>` include only | 20 | E | 3.251 -> 3.247 (−0.1%) | 3.427 -> 3.437 (+0.3%) | 0.275 -> 0.276 (+0.1%) |

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

### Repeat measurements

Every configuration whose first median was more than 2.5% slower received nine additional measured pairs at the same revisions. The primary table retains the first run.

| Workload | Configuration | Core | First run (s) | Repeat (s) |
| --- | --- | --- | ---: | ---: |
| `ranges::begin` | NVCC + Clang, C++20 | P | 2.670 -> 2.756 (+3.2%) | 2.482 -> 2.500 (+0.7%) |
| `ranges::rend` | NVRTC, C++20 | P | 0.339 -> 0.349 (+2.8%) | 0.326 -> 0.321 (−1.7%) |
| `ranges::contiguous_range` | NVRTC, C++20 | P | 0.912 -> 0.949 (+4.1%) | 0.878 -> 0.866 (−1.5%) |
| `ranges::view` | NVCC + Clang, C++20 | P | 4.039 -> 4.222 (+4.5%) | 3.758 -> 3.700 (−1.5%) |
| `ranges::range_size_t` | NVCC + Clang, C++20 | P | 2.903 -> 3.004 (+3.5%) | 2.669 -> 2.623 (−1.7%) |
| `ranges::iter_move` (ADL customization) | NVCC + GCC, C++20 | P | 4.530 -> 4.689 (+3.5%) | 4.182 -> 4.202 (+0.5%) |
| `view_interface` | NVCC + GCC, C++17 | E | 7.429 -> 7.731 (+4.1%) | 6.497 -> 6.772 (+4.2%) |
| `view_interface` | NVCC + Clang, C++17 | E | 8.225 -> 8.612 (+4.7%) | 7.193 -> 7.541 (+4.8%) |
| `view_interface` | NVRTC, C++17 | E | 1.384 -> 1.461 (+5.6%) | 1.253 -> 1.322 (+5.5%) |

## Measurement and reproduction

The benchmarks ran on an Intel Core i9-14900K with CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each configuration used one warm-up per revision, followed by nine measured pairs in alternating order on one pinned physical core. Independent configurations ran on distinct cores within the two groups. NVCC time includes the compiler driver and child processes for `-O2 -c`. NVRTC time covers `nvrtcCompileProgram` with a fresh program and its cache disabled. All workloads compile without launching a kernel.

In the raw CSV, primary samples use `series=cumulative`, with `cumulative-e-control` for the second include-only control. Confirmation samples append `-repeat` to the series name. Each percentage uses the two unrounded medians. The per-optimization tables compare each change with its parent. [Raw CPU-time samples](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_samples.csv) · [Per-optimization tables](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_pr.md)

To reproduce one cell, run from a checkout containing the measured revisions and the benchmark source commit. The 53 new workloads were added at `092804fa5b1f879b2abc310b0d83527ef5a08b1c`, which has the same library header tree as the measured HEAD. Select another source filename, compiler (`gcc`, `clang`, or `nvrtc`), or standard (`17` or `20`) to reproduce the other cells.

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
  --compiler clang --std 20 --repeats 9
```

## Validation

All 53 additional benchmarks passed the repository's clang-tidy configuration in CUDA host and device mode for both C++17 and C++20.

At the same two revisions, a consumer dropped variable-length prefixes from two forward ranges, zipped the results, queried their size, and called `iter_move` on the zip iterator and a raw pointer. All 524,288 outputs per build matched the CPU reference on an RTX PRO 6000 Blackwell Workstation Edition. NVCC 13.5 with GCC 15 produced byte-identical `sm_120a` cubins across both revisions and dialects, with 36 instructions through `EXIT`, 17 registers per thread, zero stack bytes, and zero spills. Kernel times are medians of 15 alternating batches of 32 warm launches.

| Dialect | GPU time |
|---|---:|
| C++17 | 9.134 µs -> 9.127 µs (−0.08%) |
| C++20 | 9.132 µs -> 9.127 µs (−0.05%) |
