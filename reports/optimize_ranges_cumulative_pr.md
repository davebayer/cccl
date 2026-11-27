# Reduce compile time for libcu++ range access, concepts, and view adaptors

## Summary

Range access defers lower-priority ADL and iterator probes until overload resolution needs them. C++17 now reuses the results of `begin` and `end` constraints and the iterator type across repeated queries. The iterator concepts specialize raw-pointer requirements, and the common-reference, movable, and swap concepts avoid redundant substitution.

`views::common` and `views::take_while` construct their known result types directly. View storage includes only the optional definition it uses. The bounded `iota_view` constructor checks its ordering constraint only when assertions run.

In isolated C++17 comparisons with each commit's parent, specializing iterator requirements for raw pointers cut CPU time for `sized_range` by 36.7–50.9% across the three compilers. Reusing the range-access results cut `view_interface` time by 10.5–17.3%. For the direct construction change in `views::common`, GCC was 0.2% slower and NVRTC 1.7% faster. NVCC with Clang was 3.1% slower initially and 1.6% slower on repeat.

**71 of 80 feature workloads now use at least 5% less compiler CPU time than upstream in C++17**, using the geometric mean of their three ratios of compiler CPU time (current divided by upstream). The previous header revision met that threshold on 16 workloads. Of the 64 previously below 5%, **56 reached it** and 8 stayed below. One workload crossed in the other direction. The ADL `ranges::iter_move` workload went from 5.4% savings to 4.6%.

The largest mean savings over upstream were 67.2% for `ranges::borrowed_subrange_t`, 54.9% for the ADL `ranges::iter_swap` workload, and 50.9% for `ranges::range_size_t`. The median compiler time fell in 246 of the 249 primary C++17 comparisons and rose in three. No primary C++17 comparison exceeded the 2.5% slowdown threshold.

The table gives the geometric mean of per-workload median compiler CPU seconds for each core group. Each feature workload has equal weight. Include-only controls appear in the full matrix.

| C++17 workload group | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | ---: | ---: | ---: |
| 27 performance-core workloads | 6.242 -> 4.368 (−30.0%) | 6.332 -> 4.600 (−27.3%) | 1.128 -> 0.745 (−34.0%) |
| 53 efficiency-core workloads | 8.114 -> 6.839 (−15.7%) | 8.919 -> 7.441 (−16.6%) | 1.511 -> 1.179 (−22.0%) |

### Previously below the 5% threshold

Each percentage is the saving against the same upstream revision. The first uses measurements at `372cf1d3294b7badac9eb7eb1f44930d2ae6631a` and the second uses the current headers. The parenthesized change is in percentage points. These six examples span view members, concepts, iterator operations, adaptors and factories. The full matrix includes both percentages for every original workload. [Previous C++17 report](https://github.com/davebayer/cccl/blob/4d022121c0302fbb875c7904fd4ffc6a4e32d90f/reports/optimize_ranges_cumulative_pr.md)

| Previously below 5% | Mean savings, previous -> current |
| --- | ---: |
| `view_interface` | −4.8% -> 17.6% (+22.4 pp) |
| `ranges::range` | 1.9% -> 29.8% (+27.9 pp) |
| `ranges::advance` | 2.2% -> 21.9% (+19.7 pp) |
| `views::common` | 1.0% -> 13.1% (+12.1 pp) |
| `views::iota` | 0.6% -> 6.3% (+5.7 pp) |
| `views::drop_while` | −0.3% -> 6.1% (+6.5 pp) |

**Still below 5% (9 workloads).** `ranges::view`, `ranges::viewable_range`, `ranges::enable_view`, `ranges::swap`, `ranges::iter_move` (ADL customization), `range_adaptor_closure` application, `views::transform` and `views::filter` closure composition, `views::empty`, `views::indices`.

## Cumulative C++17 compile-time results

The comparison uses upstream `ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae` and the measured branch commit `e26bf74a572d2e5a8ea9d2ee30d01445a67c9504`. Both sides compile the same benchmark sources. Each compiler cell shows median CPU seconds as **before -> after (percentage change)**. A negative compiler delta means a faster compile. Positive values in the last column mean savings against upstream. The 5% classification uses unrounded values.

P marks the 27 workloads measured on performance cores. E marks the 53 measured on efficiency cores. Each workload measures an entire translation unit, including header parsing. Some exercise several members or overloads of the same facility. The include-only control ran on both core classes. Compare revisions within a cell because absolute times vary by core class.

| Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 | Mean savings, previous -> current |
| --- | --- | ---: | ---: | ---: | ---: |
| `ranges::begin` | P | 3.588 -> 2.417 (−32.6%) | 3.331 -> 2.512 (−24.6%) | 0.537 -> 0.343 (−36.2%) | 4.2% -> 31.3% (+27.1 pp) |
| `ranges::end` | P | 4.298 -> 2.495 (−42.0%) | 4.118 -> 2.565 (−37.7%) | 0.732 -> 0.360 (−50.9%) | 5.5% -> 43.8% (+38.3 pp) |
| `ranges::cbegin` | E | 5.011 -> 4.297 (−14.2%) | 5.340 -> 4.353 (−18.5%) | 0.741 -> 0.547 (−26.2%) | 0.8% -> 19.8% (+19.0 pp) |
| `ranges::cend` | E | 5.918 -> 4.453 (−24.8%) | 6.407 -> 4.739 (−26.0%) | 0.936 -> 0.571 (−39.0%) | 2.0% -> 30.2% (+28.2 pp) |
| `ranges::rbegin` | P | 3.638 -> 2.434 (−33.1%) | 3.318 -> 2.501 (−24.6%) | 0.541 -> 0.350 (−35.2%) | 7.2% -> 31.1% (+23.9 pp) |
| `ranges::rend` | P | 4.232 -> 2.496 (−41.0%) | 4.204 -> 2.602 (−38.1%) | 0.721 -> 0.365 (−49.5%) | 4.7% -> 43.1% (+38.4 pp) |
| `ranges::crbegin` | E | 8.406 -> 6.490 (−22.8%) | 9.372 -> 7.128 (−24.0%) | 1.588 -> 1.109 (−30.1%) | 2.7% -> 25.7% (+23.0 pp) |
| `ranges::crend` | E | 8.513 -> 6.486 (−23.8%) | 9.502 -> 7.224 (−24.0%) | 1.600 -> 1.116 (−30.2%) | 2.3% -> 26.1% (+23.7 pp) |
| `ranges::data` | P | 2.865 -> 2.269 (−20.8%) | 2.407 -> 2.345 (−2.5%) | 0.329 -> 0.315 (−4.2%) | 10.5% -> 9.6% (−0.9 pp) |
| `ranges::cdata` | E | 7.920 -> 6.094 (−23.1%) | 9.082 -> 6.883 (−24.2%) | 1.576 -> 1.082 (−31.3%) | 2.5% -> 26.3% (+23.8 pp) |
| `ranges::empty` | P | 2.727 -> 2.172 (−20.4%) | 2.285 -> 2.228 (−2.5%) | 0.299 -> 0.278 (−7.2%) | 9.7% -> 10.3% (+0.6 pp) |
| `ranges::size` | P | 2.722 -> 2.200 (−19.2%) | 2.319 -> 2.275 (−1.9%) | 0.284 -> 0.276 (−2.6%) | 10.1% -> 8.3% (−1.8 pp) |
| `ranges::ssize` | E | 7.806 -> 5.841 (−25.2%) | 8.725 -> 6.397 (−26.7%) | 1.422 -> 0.925 (−35.0%) | 1.3% -> 29.1% (+27.8 pp) |
| `ranges::disable_sized_range` | E | 7.732 -> 5.743 (−25.7%) | 8.750 -> 6.360 (−27.3%) | 1.404 -> 0.893 (−36.4%) | 3.0% -> 30.0% (+27.0 pp) |
| `ranges::range` | E | 6.096 -> 4.584 (−24.8%) | 6.686 -> 4.951 (−25.9%) | 1.045 -> 0.649 (−37.9%) | 1.9% -> 29.8% (+27.9 pp) |
| `ranges::input_range` | E | 8.808 -> 7.206 (−18.2%) | 9.740 -> 7.926 (−18.6%) | 1.758 -> 1.335 (−24.0%) | 2.8% -> 20.3% (+17.6 pp) |
| `ranges::output_range` | E | 6.497 -> 5.002 (−23.0%) | 7.094 -> 5.434 (−23.4%) | 1.132 -> 0.742 (−34.5%) | 0.3% -> 27.2% (+26.9 pp) |
| `ranges::forward_range` | E | 10.381 -> 7.872 (−24.2%) | 11.490 -> 8.602 (−25.1%) | 2.159 -> 1.481 (−31.4%) | 2.9% -> 27.0% (+24.1 pp) |
| `ranges::bidirectional_range` | E | 17.111 -> 14.935 (−12.7%) | 19.008 -> 16.388 (−13.8%) | 3.888 -> 3.292 (−15.3%) | 3.0% -> 14.0% (+10.9 pp) |
| `ranges::random_access_range` | E | 15.646 -> 12.628 (−19.3%) | 18.039 -> 14.352 (−20.4%) | 3.656 -> 2.841 (−22.3%) | 2.7% -> 20.7% (+18.0 pp) |
| `ranges::contiguous_range` | P | 9.938 -> 7.500 (−24.5%) | 11.184 -> 8.326 (−25.6%) | 2.257 -> 1.637 (−27.5%) | 3.9% -> 25.9% (+21.9 pp) |
| `ranges::common_range` | E | 6.674 -> 5.082 (−23.9%) | 7.346 -> 5.500 (−25.1%) | 1.193 -> 0.765 (−35.9%) | 2.1% -> 28.5% (+26.4 pp) |
| `ranges::sized_range` | P | 3.742 -> 2.354 (−37.1%) | 3.786 -> 2.438 (−35.6%) | 0.603 -> 0.320 (−46.9%) | 2.3% -> 40.1% (+37.8 pp) |
| `ranges::borrowed_range` | E | 8.626 -> 6.862 (−20.4%) | 9.645 -> 7.496 (−22.3%) | 1.669 -> 1.187 (−28.9%) | 0.9% -> 24.0% (+23.0 pp) |
| `ranges::enable_borrowed_range` | E | 7.685 -> 5.894 (−23.3%) | 8.351 -> 6.373 (−23.7%) | 1.488 -> 0.994 (−33.2%) | 0.8% -> 26.9% (+26.1 pp) |
| `ranges::view` | P | 3.866 -> 3.804 (−1.6%) | 3.977 -> 3.960 (−0.4%) | 0.623 -> 0.610 (−2.0%) | 1.2% -> 1.3% (+0.2 pp) |
| `ranges::viewable_range` | P | 2.297 -> 2.182 (−5.0%) | 2.249 -> 2.249 (0.0%) | 0.295 -> 0.281 (−4.8%) | 2.0% -> 3.3% (+1.3 pp) |
| `ranges::enable_view` | E | 4.517 -> 4.511 (−0.1%) | 4.839 -> 4.809 (−0.6%) | 0.608 -> 0.590 (−3.0%) | 1.4% -> 1.3% (−0.1 pp) |
| `ranges::view_base` | E | 6.695 -> 4.944 (−26.1%) | 7.440 -> 5.320 (−28.5%) | 1.198 -> 0.736 (−38.6%) | 1.6% -> 31.3% (+29.7 pp) |
| `ranges::iterator_t` | E | 6.416 -> 4.853 (−24.4%) | 6.850 -> 5.136 (−25.0%) | 1.081 -> 0.676 (−37.5%) | 1.5% -> 29.2% (+27.7 pp) |
| `ranges::sentinel_t` | E | 10.854 -> 8.596 (−20.8%) | 12.072 -> 9.458 (−21.7%) | 2.213 -> 1.616 (−27.0%) | 1.9% -> 23.2% (+21.3 pp) |
| `ranges::range_difference_t` | P | 5.403 -> 3.582 (−33.7%) | 5.843 -> 3.802 (−34.9%) | 1.138 -> 0.660 (−42.0%) | 32.3% -> 37.0% (+4.7 pp) |
| `ranges::range_size_t` | P | 4.475 -> 2.398 (−46.4%) | 4.712 -> 2.511 (−46.7%) | 0.812 -> 0.337 (−58.5%) | 20.6% -> 50.9% (+30.3 pp) |
| `ranges::range_value_t` | E | 10.635 -> 8.342 (−21.6%) | 12.175 -> 9.379 (−23.0%) | 2.192 -> 1.592 (−27.4%) | 2.6% -> 24.0% (+21.4 pp) |
| `ranges::range_reference_t` | E | 10.984 -> 8.525 (−22.4%) | 12.202 -> 9.366 (−23.2%) | 2.335 -> 1.678 (−28.1%) | 2.7% -> 24.6% (+22.0 pp) |
| `ranges::range_rvalue_reference_t` | E | 6.905 -> 6.453 (−6.5%) | 7.586 -> 7.019 (−7.5%) | 1.254 -> 1.115 (−11.1%) | 3.4% -> 8.4% (+5.1 pp) |
| `ranges::range_common_reference_t` | E | 9.041 -> 7.852 (−13.2%) | 9.949 -> 8.534 (−14.2%) | 1.833 -> 1.502 (−18.0%) | 4.4% -> 15.2% (+10.8 pp) |
| `ranges::dangling` | E | 6.800 -> 5.599 (−17.7%) | 7.342 -> 6.059 (−17.5%) | 1.226 -> 0.896 (−26.9%) | 1.6% -> 20.8% (+19.2 pp) |
| `ranges::borrowed_iterator_t` | E | 8.835 -> 6.964 (−21.2%) | 9.720 -> 7.603 (−21.8%) | 1.779 -> 1.264 (−28.9%) | 1.1% -> 24.0% (+23.0 pp) |
| `ranges::borrowed_subrange_t` | P | 10.787 -> 3.891 (−63.9%) | 11.617 -> 3.973 (−65.8%) | 2.449 -> 0.703 (−71.3%) | 64.9% -> 67.2% (+2.3 pp) |
| `ranges::advance` | E | 12.066 -> 9.577 (−20.6%) | 13.552 -> 10.808 (−20.2%) | 2.578 -> 1.937 (−24.9%) | 2.2% -> 21.9% (+19.7 pp) |
| `ranges::distance` | E | 10.811 -> 8.496 (−21.4%) | 12.135 -> 9.372 (−22.8%) | 2.292 -> 1.654 (−27.9%) | 1.8% -> 24.1% (+22.2 pp) |
| `ranges::next` | E | 12.182 -> 9.783 (−19.7%) | 13.740 -> 10.944 (−20.3%) | 2.599 -> 1.973 (−24.1%) | 2.3% -> 21.4% (+19.1 pp) |
| `ranges::prev` | E | 11.978 -> 9.645 (−19.5%) | 13.717 -> 10.989 (−19.9%) | 2.637 -> 2.009 (−23.8%) | 2.1% -> 21.1% (+18.9 pp) |
| `ranges::swap` | E | 4.390 -> 4.365 (−0.6%) | 4.657 -> 4.549 (−2.3%) | 0.566 -> 0.542 (−4.3%) | 0.1% -> 2.4% (+2.3 pp) |
| `ranges::iter_move` (ADL customization) | P | 5.377 -> 4.758 (−11.5%) | 5.010 -> 4.939 (−1.4%) | 0.836 -> 0.833 (−0.3%) | 5.4% -> 4.6% (−0.9 pp) |
| `ranges::iter_move` (raw pointer) | P | 4.233 -> 2.297 (−45.7%) | 4.395 -> 2.352 (−46.5%) | 0.671 -> 0.291 (−56.6%) | 48.9% -> 49.9% (+0.9 pp) |
| `ranges::iter_swap` (ADL customization) | P | 8.889 -> 4.426 (−50.2%) | 9.853 -> 4.775 (−51.5%) | 1.840 -> 0.700 (−62.0%) | 55.3% -> 54.9% (−0.5 pp) |
| `view_interface` | E | 7.381 -> 6.246 (−15.4%) | 8.115 -> 6.865 (−15.4%) | 1.386 -> 1.083 (−21.9%) | −4.8% -> 17.6% (+22.4 pp) |
| `ref_view` members | E | 5.768 -> 5.517 (−4.3%) | 6.129 -> 5.918 (−3.4%) | 0.908 -> 0.801 (−11.9%) | −0.1% -> 6.6% (+6.7 pp) |
| `owning_view` members | E | 7.001 -> 6.025 (−13.9%) | 7.633 -> 6.574 (−13.9%) | 1.303 -> 1.067 (−18.1%) | 1.4% -> 15.3% (+13.9 pp) |
| `subrange` construction and navigation | E | 9.512 -> 8.206 (−13.7%) | 10.513 -> 8.991 (−14.5%) | 1.789 -> 1.535 (−14.2%) | −0.3% -> 14.1% (+14.4 pp) |
| `subrange` tuple access | E | 7.799 -> 6.830 (−12.4%) | 8.473 -> 7.324 (−13.6%) | 1.339 -> 1.182 (−11.7%) | −1.3% -> 12.6% (+13.8 pp) |
| `views::all` | P | 13.596 -> 8.353 (−38.6%) | 14.752 -> 9.002 (−39.0%) | 2.872 -> 1.601 (−44.3%) | 22.7% -> 40.7% (+18.0 pp) |
| `views::all_t` | E | 8.454 -> 6.589 (−22.1%) | 9.336 -> 7.087 (−24.1%) | 1.665 -> 1.126 (−32.4%) | 2.0% -> 26.3% (+24.3 pp) |
| `range_adaptor_closure` application | E | 4.573 -> 4.444 (−2.8%) | 4.974 -> 4.821 (−3.1%) | 0.656 -> 0.623 (−5.0%) | 1.5% -> 3.6% (+2.1 pp) |
| `views::transform` and `views::filter` closure composition | P | 14.502 -> 14.182 (−2.2%) | 14.791 -> 14.721 (−0.5%) | 2.973 -> 2.985 (+0.4%) | 1.5% -> 0.8% (−0.7 pp) |
| `views::empty` | E | 4.662 -> 4.633 (−0.6%) | 4.970 -> 4.891 (−1.6%) | 0.688 -> 0.667 (−3.0%) | 1.2% -> 1.7% (+0.5 pp) |
| `views::single` | E | 7.698 -> 7.187 (−6.6%) | 8.236 -> 7.612 (−7.6%) | 1.430 -> 1.293 (−9.6%) | 0.9% -> 7.9% (+7.1 pp) |
| `views::iota` | E | 7.463 -> 7.060 (−5.4%) | 8.106 -> 7.611 (−6.1%) | 1.367 -> 1.266 (−7.4%) | 0.6% -> 6.3% (+5.7 pp) |
| `views::indices` | E | 4.595 -> 4.453 (−3.1%) | 5.019 -> 4.828 (−3.8%) | 0.653 -> 0.620 (−5.1%) | 1.2% -> 4.0% (+2.9 pp) |
| `views::repeat` | E | 6.332 -> 6.082 (−4.0%) | 6.890 -> 6.559 (−4.8%) | 1.077 -> 1.002 (−7.0%) | 0.4% -> 5.2% (+4.9 pp) |
| `views::counted` | E | 19.411 -> 17.299 (−10.9%) | 21.450 -> 18.994 (−11.5%) | 4.551 -> 3.962 (−12.9%) | 1.8% -> 11.8% (+9.9 pp) |
| `common_view::begin/end` | P | 9.327 -> 8.072 (−13.5%) | 9.977 -> 8.463 (−15.2%) | 1.999 -> 1.709 (−14.5%) | 4.9% -> 14.4% (+9.4 pp) |
| `views::common` | E | 9.937 -> 8.764 (−11.8%) | 11.008 -> 9.639 (−12.4%) | 1.965 -> 1.671 (−15.0%) | 1.0% -> 13.1% (+12.1 pp) |
| `reverse_view` construction | P | 12.539 -> 8.847 (−29.4%) | 13.490 -> 9.491 (−29.6%) | 2.978 -> 1.976 (−33.6%) | 10.4% -> 30.9% (+20.5 pp) |
| `views::reverse` | E | 12.452 -> 10.395 (−16.5%) | 13.833 -> 11.402 (−17.6%) | 2.718 -> 2.167 (−20.3%) | 1.8% -> 18.1% (+16.3 pp) |
| `views::take` | P | 15.691 -> 9.676 (−38.3%) | 17.121 -> 10.382 (−39.4%) | 3.614 -> 2.046 (−43.4%) | 18.4% -> 40.4% (+22.0 pp) |
| `take_view` traversal and observers | E | 7.951 -> 6.800 (−14.5%) | 8.663 -> 7.434 (−14.2%) | 1.505 -> 1.233 (−18.1%) | 2.4% -> 15.6% (+13.2 pp) |
| `views::take_while` | E | 7.581 -> 7.072 (−6.7%) | 8.248 -> 7.688 (−6.8%) | 1.429 -> 1.287 (−10.0%) | −0.4% -> 7.8% (+8.2 pp) |
| `views::drop` | P | 16.047 -> 12.148 (−24.3%) | 17.450 -> 13.188 (−24.4%) | 3.911 -> 2.874 (−26.5%) | 3.8% -> 25.1% (+21.3 pp) |
| `drop_view::begin` | P | 8.846 -> 6.985 (−21.0%) | 9.537 -> 7.846 (−17.7%) | 1.981 -> 1.513 (−23.7%) | 11.1% -> 20.8% (+9.8 pp) |
| `drop_view` traversal and observers | E | 8.091 -> 6.957 (−14.0%) | 8.687 -> 7.485 (−13.8%) | 1.515 -> 1.236 (−18.4%) | 0.8% -> 15.5% (+14.7 pp) |
| `views::drop_while` | E | 8.857 -> 8.376 (−5.4%) | 9.721 -> 9.175 (−5.6%) | 1.884 -> 1.746 (−7.3%) | −0.3% -> 6.1% (+6.5 pp) |
| `views::filter` | E | 6.756 -> 6.328 (−6.3%) | 7.255 -> 6.763 (−6.8%) | 1.209 -> 1.090 (−9.9%) | −0.7% -> 7.7% (+8.3 pp) |
| `views::transform` | P | 8.168 -> 6.545 (−19.9%) | 9.003 -> 7.001 (−22.2%) | 1.858 -> 1.342 (−27.8%) | 4.4% -> 23.4% (+19.0 pp) |
| `transform_view` traversal and observers | E | 6.577 -> 6.285 (−4.4%) | 7.016 -> 6.604 (−5.9%) | 1.159 -> 1.072 (−7.6%) | −0.8% -> 6.0% (+6.8 pp) |
| `views::zip` | E | 14.983 -> 12.964 (−13.5%) | 16.413 -> 14.158 (−13.7%) | 3.448 -> 2.950 (−14.4%) | 2.5% -> 13.9% (+11.4 pp) |
| `zip_view::begin/end` | P | 8.334 -> 6.946 (−16.6%) | 8.913 -> 7.457 (−16.3%) | 1.771 -> 1.471 (−16.9%) | 4.4% -> 16.6% (+12.3 pp) |
| `zip_view::size` | P | 9.667 -> 6.539 (−32.4%) | 10.177 -> 6.950 (−31.7%) | 2.206 -> 1.353 (−38.7%) | 16.7% -> 34.3% (+17.6 pp) |
| `<cuda/std/ranges>` include only | P | 2.316 -> 2.309 (−0.3%) | 2.349 -> 2.367 (+0.7%) | 0.297 -> 0.292 (−1.7%) | — |
| `<cuda/std/ranges>` include only | E | 3.883 -> 3.851 (−0.8%) | 4.032 -> 3.959 (−1.8%) | 0.468 -> 0.457 (−2.5%) | — |

### Additional bounded iota construction workload

The original `views::iota` workload checks factory and observer result types in unevaluated expressions. The additional workload executes the bounded constructor during constant evaluation and checks size and traversal for 32 distinct pointer element types. These builds use the default configuration with CCCL assertions disabled, so the constructor can skip substituting its ordering check. It uses the same upstream headers, E cores and three C++17 compiler configurations as the other comparisons. The threshold counts and cohort means use the 80 feature workloads common to both sweeps.

| Workload | Core | NVCC 13.5 + GCC 15 | NVCC 13.5 + Clang 22 | NVRTC 13.5 |
| --- | --- | ---: | ---: | ---: |
| `iota_view` bounded pointer construction | E | 6.569 -> 5.993 (−8.8%) | 7.120 -> 6.578 (−7.6%) | 1.263 -> 1.086 (−14.0%) |

<details>
<summary>What the workloads exercise</summary>

The inventory covers the public range facilities shipped by this libcu++ header, including its iterator operations and the `views::indices` extension. [Pinned libcu++ header](https://github.com/davebayer/cccl/blob/e26bf74a572d2e5a8ea9d2ee30d01445a67c9504/libcudacxx/include/cuda/std/ranges) · [Benchmark sources](https://github.com/davebayer/cccl/tree/e26bf74a572d2e5a8ea9d2ee30d01445a67c9504/libcudacxx/benchmarks/compile_time/ranges)

- **Access and size.** Member, ADL and iterator fallback selection, const access on a mutable range, signed sizing, and `disable_sized_range`.
- **Range requirements and associated types.** Iterator-category boundaries, common and distinct sentinels, borrowed and owning ranges, opting into views and borrowing, reference and value types, and dangling results.
- **Iterator operations.** Distance, bounded and unbounded navigation, element swap through a customization, custom iterator moves and swaps, and moving through a raw pointer.
- **Storage and view utilities.** The CRTP `view_interface` observers and element access (including `at`), `ref_view` and `owning_view` access and `base()`, sized and unsized `subrange_kind`, subrange construction and navigation, and the subrange tuple protocol (`get`, `tuple_size`, `tuple_element`).
- **Factories.** Empty and single views, bounded and unbounded iota and repeat views, the libcu++ `views::indices` extension, and all three dispatch paths for `views::counted`.
- **Adaptors.** Construction, the resulting view classes and their member operations for all, common, reverse, take, take-while, drop, drop-while, filter, transform and zip. Separate cases cover closure composition, piping a range into a user-defined `range_adaptor_closure`, and `views::all_t`. `cuda::std::views` aliases `cuda::std::ranges::views`.
- **Input counts.** Existing workloads use 24–64 distinct type indices. Most new probes use 24 or 32, and single and empty views use 64. The indices workload checks six standard integral types and rejection of a floating-point argument.

</details>

## Measurement and reproduction

The benchmarks ran on an Intel Core i9-14900K with CUDA Toolkit 13.5.37, GCC 15.2.0, Clang 22.1.8, and `sm_120a`. Each configuration used one warm-up per revision, followed by nine measured pairs in alternating order on one pinned physical core. Independent configurations ran on distinct cores within the two groups. NVCC time includes the compiler driver and child processes for `-O2 -c`. NVRTC time covers `nvrtcCompileProgram` with a fresh program and its cache disabled. All workloads compile without launching a kernel.

Filter the raw CSV to `standard=c++17`. The primary series is `cumulative2`, with `cumulative2-e-control` for the E-core include-only control. Confirmations append `-repeat`. The previous sweep uses `cumulative` and `cumulative-e-control`. For each workload, mean savings are 100 times one minus the geometric mean of the three ratios of after to before median CPU seconds. Both include-only controls are excluded from threshold counts. [Raw CPU-time samples](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_samples.csv) · [Per-change measurements](https://github.com/davebayer/cccl/blob/optimize_ranges/reports/optimize_ranges_pr.md)

To reproduce a cell, check out a commit containing the measured revisions and benchmark sources. The 81 original source files and their helpers match suite commit `092804fa5b1f879b2abc310b0d83527ef5a08b1c`. The final header commit also contains the new bounded-construction workload. Use the CSV's `benchmark` field as the source filename stem and select the compiler and core class shown in the table.

```bash
base=ebeb7505ad946445b2c2fd8d7b1b8f07a474fdae
head=e26bf74a572d2e5a8ea9d2ee30d01445a67c9504
workloads=e26bf74a572d2e5a8ea9d2ee30d01445a67c9504
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

- On the final headers, the full range-access, range-requirements, and `view_interface` lit directories passed 168 configurations on the RTX PRO 6000 with NVCC 13.5. Each host compiler (GCC 15 and Clang 22) passed 40 tests in C++17 and 44 in C++20. Four tests require native concepts and were skipped in C++17. An earlier isolated run of the access change passed six targeted tests per dialect with NVCC/GCC (12 passes).
- Seven swap and view-factory tests passed with both GCC 15 and Clang 22 under NVCC in each dialect (28 lit passes). The focused common-view, take-while, and movable suites passed 11 applicable cases per host in C++17 and 12 per host in C++20.
- Each NVCC host compiler passed 15 applicable iterator-concept tests in C++17 and 21 in C++20. Six require native C++20 concepts. Each host/dialect combination also passed five targeted common-reference, proxy-iterator, and movable tests.
- All 29 iota and indices tests passed with each NVCC host compiler in each dialect (116 lit passes). On the RTX PRO 6000, the host-only and device-only assertion modes each ran one ordering check on their enabled side. Debug mode ran one on each side, and disabling assertions ran none, in both dialects.
- Root-configured clang-tidy 22 with the CCCL plugin checked the changed C++ headers, benchmark, and regression sources in CUDA host and device modes for both dialects.
