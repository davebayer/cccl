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
  - `this_{level} groups`
  - `coalesced_group` (new)

- Groups that are created by splitting another group. We call them derived.
  - `generic_group`
  - `virtual_group` (new)

- A non-owning group view over a group.
  - `group_view` (new)

## Implicit Groups
### This groups
[//]: <> (Put code on the left, the text on the right)
```cpp
template <typename Level, typename Hierarchy>
class /*this-group*/ : public /*group-interface-for*/</*this-group*/>
{
public:
  using unit_type       = Level;
  using level_type      = Level;
  using hierarchy_type  = Hierarchy;

  __device__ /*this-group*/(const auto& hierarchy_like) noexcept;
};

template <typename Hierarchy>
class this_thread : public /*this-group*/<thread_level, Hierarchy> {};
template <typename Hierarchy>
class this_warp : public /*this-group*/<warp_level, Hierarchy> {};
template <typename Hierarchy>
class this_block : public /*this-group*/<block_level, Hierarchy> {};
template <typename Hierarchy>
class this_cluster : public /*this-group*/<cluster_level, Hierarchy> {};
template <typename Hierarchy>
class this_grid : public /*this-group*/<grid_level, Hierarchy> {};
```
- one class per level
- contain a single unit within the same level
- building blocks for other groups
- always contiguous
- use `level_synchronizer` for synchronization

## Implicit Groups
### Coalesced group
[//]: <> (Put code on the left, the text on the right)
```cpp
template <typename Hierarchy>
class coalesced_group : public /*group-interface-for*/<coalesced_group>
{
public:
  using unit_type       = thread_level;
  using level_type      = warp_level;
  using hierarchy_type  = Hierarchy;

  __device__ explicit coalesced_group(const auto& hierarchy_like) noexcept;
};

template </*hierarchy-like-type*/ HierarchyLike>
coalesced_group(const HierarchyLike&)
  -> coalesced_group</*hierarchy-type-of*/<HierarchyLike>>;
```
- groups all currently active threads within a warp based on `__activemask()`
- useful for opprotunistic programming
- it is not always contiguous
- uses `lane_synchronizer` for synchronization

## Derived Groups
### Generic group
[//]: <> (Put code on the left, the text on the right)
```cpp
template <typename Unit,
          typename ParentGroup,
          typename MappingResult,
          typename SynchronizerInstance>
class generic_group : public /*group-interface-for*/<generic_group>
{
public:
  using unit_type       = Unit;
  using level_type      = typename ParentGroup::level_type;
  using hierarchy_type  = typename ParentGroup::hierarchy_type;

  __device__ explicit generic_group(const Unit& unit,
                                    const ParentGroup& parent_group,
                                    auto&& mapping,
                                    auto&& synchronizer) noexcept;

};

template </*hierarchy-level-type*/ Unit,
          group ParentGroup,
          typename Mapping,
          typename Synchronizer>
  requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
generic_group(const Unit&, const ParentGroup&, Mapping&&, Synchronizer&&)
  -> generic_group<Unit,
                   ParentGroup,
                   /*result-of-mapping*/,
                   /*synchronizer-instance*/>;
```
- splits `parent_group` into independent subgroups of `unit` units as defined by `mapping`
- uses `synchronizer` for synchronization
- construction is a collective operation

## Derived Groups
### Generic group
[//]: <> (Put code on the left, the text on the right)
```cpp
template <typename Unit, typename ParentGroup, typename MappingResult>
class virtual_group : public /*group-interface-for*/<virtual_group>
{
public:
  using unit_type       = Unit;
  using level_type      = typename ParentGroup::level_type;
  using hierarchy_type  = typename ParentGroup::hierarchy_type;

  __device__ explicit virtual_group(const Unit& unit,
                                    const ParentGroup& parent_group,
                                    auto&& mapping) noexcept;
};

template </*hierarchy-level-type*/ Unit, group ParentGroup, typename Mapping>
  requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
virtual_group(const Unit&, const ParentGroup&, Mapping&&)
  -> virtual_group<Unit, ParentGroup, /*result-of-mapping*/>;
```
- creates a virtual layer over the `parent_group`
- splits `parent_group` into independent subgroups of `unit` units as defined by `mapping`
- uses `parent_group` for synchronization
- all subgroups must share the synchronization points
- construction is a collective operation

## Group View
[//]: <> (Put code on the left, the text on the right)
```cpp
template <typename Unit, typename Group>
class group_view : public /*group-interface-for*/<group_view>
{
public:
  using unit_type       = Unit;
  using level_type      = typename Group::level_type;
  using hierarchy_type  = typename Group::hierarchy_type;

  __device__ group_view(const Group& group) noexcept
    requires cuda::std::same_as<Unit, typename Group::unit_type>;

  __device__ explicit group_view(const Unit& unit, const Group& group) noexcept;
};

template <group Group>
group_view(const Group&) -> group_view<typename Group::unit_type, Group>;

template </*hierarchy-level-type*/ Unit, group Group>
group_view(const Unit&, const Group&) -> group_view<Unit, Group>;
```
- non-owning view over `group`
- allows to view `group` as if it had a different `unit`
- implements the same queries as `group`

## Group Mappings
- Determines how the units are split into subgroups
- Needs to implement the `map` method that transforms `/*group-mapping-result*/` types:
```cpp
template <typename T, typename Unit, typename ParentGroup, typename PrevMappingResult>
concept /*group-mapping*/ = requires (T& t,
                                      const Unit& unit,
                                      const ParentGroup& parent_group,
                                      const PrevMappingResult& prev_mapping_result)
{
  { t.map(unit, parent_group, prev_mapping_result) } -> /*group-mapping-result*/;
  { t.map(non_exhaustive, unit, parent_group, prev_mapping_result) } -> /*group-mapping-result*/;
};
```
- The overload that takes the `non_exhaustive` tag does the non-exhaustive mapping
- Mappings can be chained with `|` to transform the result from left to right
- Goal is to preserve as much compile-time information as possible

## Predefined Group Mappings
[//]: <> (This slide should generate as much slides as there are items. Put the list on the left and highlight 1 item per slide. On the right show the description and make a diagram example how this mapping is used on threads within a single warp)
- `identity_mapping` preserves the previous mapping result without further partitioning
- `group_by` splits each group into subgroups with the same specified number of units
- `group_as` splits each group into subgroups with individually specified sizes
- `take` selects the first specified number of units from each group
- `binary_partition` splits threads within a warp into two groups based on a predicate
- `composite_mapping` applies multiple mappings in sequence, composed with the `|` operator

## Group Synchronizers
- Determines how the subgroups are synchronized
- Needs to implement the `make_instance` method that returns the `/*group-synchronizer-instance*/` type:
```cpp
template <typename T, typename Unit, typename ParentGroup, typename MappingResult>
concept /*group-synchronizer*/ = requires (T& t,
                                           const Unit& unit,
                                           const ParentGroup& parent_group,
                                           const MappingResult& mapping_result)
{
  { t.make_instance(unit, parent_group, mapping_result) }
    -> /*group-synchronizer-instance*/<MappingResult, typename ParentGroup::hierarchy_type>;
};
```

## Predefined Group Synchronizers
[//]: <> (This slide should generate as much slides as there are items. Put the list on the left and highlight 1 item per slide. On the right show how the type can be constructed and what's the sync and sync_aligned() behaviour of the group is)
- `level_synchronizer` synchronizes all threads within the group's level.
- `lane_synchronizer` uses `__syncwarp(mask)` for synchronization.
- `interwarp_synchronizer` uses `__barrier_sync_count(barrier_id, nthreads)` for synchronization.
- `barrier_synchronizer` uses `cuda::barrier::arrive_and_wait()` for synchronization.
