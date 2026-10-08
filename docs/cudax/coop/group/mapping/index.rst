.. _cudax-coop-group-mapping:

Group Mappings
==============

Group mappings are types that implement the mapping interface. They define how a group is split into subgroups independent on the group's synchronization mechanism. To communicate this information, they are required to accept and return types that satisfy the *group mapping result* concept.

Group Mapping Result Concept
----------------------------

*Group mapping result* is an implementation-defined concept that defines the interface for data exchange between groups and mappings. It is defined as:

.. code:: c++

  template <typename T>
  concept /*group-mapping-result*/ = requires (const T& t)
  {
    requires(cuda::std::is_copy_constructible_v<T>);

    // Group in the parent group info.
    { T::static_group_count() } -> cuda::std::size_t; // must be constexpr
    { t.group_count() } -> cuda::std::uint32_t;
    { t.group_rank() } -> cuda::std::uint32_t;

    // Units in the group info.
    { T::static_unit_count() } -> cuda::std::size_t; // must be constexpr
    { t.unit_count() } -> cuda::std::uint32_t;
    { t.unit_rank() } -> cuda::std::uint32_t;

    // Properties of the mapping result.
    { T::is_always_exhaustive() } -> bool; // must be constexpr
    { T::is_always_contiguous() } -> bool; // must be constexpr

    // Other group related info.
    { t.lane_mask() } -> cuda::device::lane_mask;
  };

Where:

- ``static_group_count()`` is the compile-time known number of groups. When it is equal to ``cuda::std::dynamic_extents``, it means that the group count is runtime known only.
- ``group_count()`` is the runtime known number of groups. It is equal to ``static_cast<cuda::std::uint32_t>(static_group_count())`` when ``static_group_count() != cuda::std::dynamic_extent``.
- ``group_rank()`` is this unit's group rank.
- ``static_unit_count()`` is the compile-time known number of units within the every group. When equal to ``cuda::std::dynamic_extent``, it means that the number of units is runtime known only.
- ``unit_count()`` is the runtime known number of units within the unit's group. It is equal to ``static_cast<cuda::std::uint32_t>(static_unit_count())`` when ``static_unit_count() != cuda::std::dynamic_extent``
- ``unit_rank()`` is the unit's rank.
- ``is_always_exhaustive()`` is a compile-time known property that specifies whether all units are part of a group.
- ``is_always_contiguous()`` is a compile-time known property that specifies whether all units within the group are mapped to a contiguous block of hardware resources.
- ``lane_mask()`` is the mask of threads within this warp that are part of this group.

Group Mapping Process
---------------------

The group mapping process has 3 steps:

1. The initial mapping result is created from the parent group's mapping result. The static group count is set to ``1``, group to ``0``, unit count and rank copied (or recomputed if the unit level was changed), the always exhaustive property is set to ``true`` and the always contiguous property is copied as well as the lane mask.
2. The mapping's ``.map(...)`` member function is invoked with the initial mapping result.
3. The transformed mapping result is used to initialize the newly created group.

The key idea is that a mapping tries to provide as many information as compile-time constants as possible. This information can be used to better optimize queries and cooperative algorithms.

Group Mapping Interface
-----------------------

Every group mapping type must implement the group mapping interface that can be defined as:

.. .. code:: c++
..
..   template <typename T, typename Unit, typename ParentGroup>
..   concept /*group-mapping-interface*/ = requires (T& t, const Unit& unit, const ParentGroup& parent_group)
..   {
..     t.map(unit, parent_group, )
..   };

Predefined Mappings
-------------------

.. toctree::
    :maxdepth: 1

    identity_mapping
    group_by
    group_as
    take
    binary_partition
    composite_mapping
