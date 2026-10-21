[//]: <> (asdf)


# CCCL Cooperative

## Why?
- We need a good abstraction that will help us organize threads into groups that can cooperate on solving problems
  - CCCL Cooperative Groups

- We need APIs that will implement algorithms that solve the common problems
  - CCCL Cooperative Algorithms

## When?
Ambitious goal: GTC 03/27

[//]: <> (The contents should be a timeline of 2 columns, the first one for groups, the second one for algorithms, those lines with ? comment are plans, make them different from the others)
- 01/26 First ideas on groups
- 03/26 Initial implementation
- 05/26 Improved implementation
- 08/26 Documentation
- 11/26 Alpha release to gather feedback
- 01/27 Beta release [//]: <> (?)
- 03/27 Release [//]: <> (?)

- 04/26 First ideas on algorithms
- 06/26 Initial `reduce` implementation
- 07/26 Initial `any_of`, `all_of`, `shuffle` implementation
- 11/26 Finalizing algorithms design, reworking existing algorithms
- ??/?? Alpha release to gather feedback [//]: <> (?)
- ??/?? Beta release [//]: <> (?)
- 03/27 (Partial) release [//]: <> (?)

# CCCL Cooperative Groups

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns with the definition)
- Generic Group
  - Groups arbitrary threads.
- CTK's Cooperative Groups
  - Groups threads within an upper boundary.
- CCCL Cooperative Groups
  - Groups units within an upper boundary.

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns, all of the pictures should be a block of 128 threads)
- Generic Group
  [//]: <> (Picture of arbitrary GPU threads being highlighted)
- CTK's Cooperative Groups
  [//]: <> (Picture of arbitrary GPU threads being highlighted within the block being the upper bound)
- CCCL Cooperative Groups
  [//]: <> (Picture of arbitrary GPU threads being highlighted within the block being the upper bound)

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns, all of the pictures should be a block of 128 threads)
- Generic Group
  [//]: <> (Picture of GPU threads from first and third warp being highlighted)
- CTK's Cooperative Groups
  [//]: <> (Picture of GPU threads from first and third warp being highlighted within the block being the upper bound)
- CCCL Cooperative Groups
  [//]: <> (Picture of GPU threads from first and third warp being highlighted within the block being the upper bound, the warps should be highlighted, too)

## What makes them different?
### Hierarchy-awareness
[//]: <> (Here I want 2 columns, this example computes 1D thread rank within a block, non-of the dimensions are statically known)
- CTK's Cooperative Groups
  ```cpp
  // What cooperative_groups::this_thread_block().thread_rank() would compute.
  const auto thread_rank =
    (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```
- CCCL Cooperative Groups
  ```cpp
  // What CCCL Cooperative Groups equivalent would compute.
  const auto thread_rank =
    (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```

## What makes them different?
### Hierarchy-awareness
[//]: <> (Here I want 2 columns, this example computes 1D thread rank within a block, we know the block is 1D)
- CTK's Cooperative Groups
  ```cpp
  // What cooperative_groups::this_thread_block().thread_rank() would compute.
  const auto thread_rank =
    (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```
- CCCL Cooperative Groups
  ```cpp
  // What CCCL Cooperative Groups equivalent would compute.
  const auto thread_rank = threadIdx.x;
  ```

## What makes them different?
### Group queries
[//]: <> (Here I want 2 columns, again the picture of GPU threads within a block, Both should show the same thread rank)
- CTK's Cooperative Groups
- CCCL Cooperative Groups

## What makes them different?
### Group queries
[//]: <> (Here I want 2 columns, again the picture of GPU first and third warp within a block)
- CTK's Cooperative Groups
  [//]: <> (Only shows thread rank)
- CCCL Cooperative Groups
  [//]: <> (Shows both thread rank and warp rank)

## What makes them different?
### Separation of Partition and Synchronization
When creating a group, *how the units are grouped* are independent on and *how the units are synchronized*.
[//]: <> (Place here similar picture to docs/cudax/coop/group/images/group-design.svg)

## What makes them different?
### Optional participation
CCCL Cooperative Groups add support for non-exhaustive mapping.
[//]: <> (Place here a picture of threads within a warp being grouped by 5, leaving 2 threads without any group. The parent groups should be placed on the left, then an arrow to the mapping that groups by 5 and arrow of the split parent group)

## Requirements on groups
### How a group should look like
It should specify:
1. What are the units being grouped (threads, warps, ...)
2. Within what level (block, grid, ...)
3. In which hierarchy

It should provide:
1. APIs for queries
2. APIs for synchronization

## Requirements on groups
### The group concept
[//]: <> (2 columns, the concept is on the left)
```cpp
template <typename T>
concept group = requires (const T& t)
{
  // Member types.
  requires /*is-hierarchy-level*/<typename T::unit_type>;     // (1)
  requires /*is-hierarchy-level*/<typename T::level_type>;    // (2)
  requires /*is-hierarchy*/<typename T::hierarchy_type>;      // (3)

  // Getters.
  { t.hierarchy() } noexcept
    -> cuda::std::same_as<const typename T::hierarchy_type&>; // (4)

  // Synchronization.
  { t.sync() } noexcept -> cuda::std::same_as<void>;          // (5)
  { t.sync_aligned() } noexcept -> cuda::std::same_as<void>;  // (6)

  // Properties.
  { T::is_always_contiguous() } noexcept
    -> cuda::std::same_as<bool>;                              // (7), must be constexpr
};
```
1. what units are being grouped
2. within which level
3. in which hierarchy
4. hierarchy getter
5. synchronizes all units
6. synchronizes all units, requires context alignment
7. `true` if each group occupies a contiguous range of physical units

## Group queries
[//]: <> (2 columns, fit it in one slide)
- Sub-unit queries
  - Query information about levels that are below the group's unit type.
  ```cpp
  qunit.static_count(group); // (1), always constexpr
  qunit.count_as<T>(group);  // (2)
  qunit.rank_as<T>(group);   // (3)

  qunit.count(group);        // (4)
  qunit.rank(group);         // (5)
  qunit.is_root_rank(group); // (6)
  ```
  1. Statically known number of `qunit` within `group`
  2. Number of `qunit` within `group` returned as `T`
  3. Current `qunit`'s rank within `group` returned as `T`
  4. Same as `qunit.count_as</*default-T*/>(group)`
  5. Same as `qunit.rank_as</*default-T*/>(group)`
  6. Same as `qunit.rank(group) == 0`

- Super-level queries
  - Query information about levels that are above the group's upper boundary type.
  ```cpp
  group.static_count(/*parent-group-or-qlevel*/); // (1), always constexpr
  group.count_as<T>(/*parent-group-or-qlevel*/);  // (2)
  group.rank_as<T>(/*parent-group-or-qlevel*/);   // (3)

  group.count(/*parent-group-or-qlevel*/);        // (4)
  group.rank(/*parent-group-or-qlevel*/);         // (5)
  ```
  1. Statically known number of `group` within `/*parent-group-or-qlevel*/`
  2. Number of `group` within `/*parent-group-or-qlevel*/` returned as `T`
  3. Current `group`'s rank within `/*parent-group-or-qlevel*/` returned as `T`
  4. Same as `group.count_as</*default-T*/>(/*parent-group-or-qlevel*/)`
  5. Same as `group.rank_as</*default-T*/>(/*parent-group-or-qlevel*/)`

## Predefined Groups
[//]: <> (Make this slide somehow interesting, don't make it just a list)
CCCL Cooperative Groups come with several predefined group types:

- Implicit groups without a parent group. They are essential building blocks for other groups.
  - this_{level} groups
  - coalesced_group

- Groups that are created by splitting another group.
  - generic_group
  - virtual_group

- A non-owning group view over a group.
  - group_view
