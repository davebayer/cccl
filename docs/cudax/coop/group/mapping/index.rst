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
    { T::static_group_count() } -> cuda::std::same_as<cuda::std::size_t>; // must be constexpr
    { t.group_count() } -> cuda::std::same_as<cuda::std::uint32_t>;
    { t.group_rank() } -> cuda::std::same_as<cuda::std::uint32_t>;

    // Units in the group info.
    { T::static_unit_count() } -> cuda::std::same_as<cuda::std::size_t>; // must be constexpr
    { t.unit_count() } -> cuda::std::same_as<cuda::std::uint32_t>;
    { t.unit_rank() } -> cuda::std::same_as<cuda::std::uint32_t>;

    // Properties of the mapping result.
    { T::is_always_exhaustive() } -> cuda::std::same_as<bool>; // must be constexpr
    { T::is_always_contiguous() } -> cuda::std::same_as<bool>; // must be constexpr
    { t.is_valid() } -> cuda::std::same_as<bool>;

    // Other group related info.
    { t.lane_mask() } -> cuda::std::same_as<cuda::device::lane_mask>;
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
- ``is_valid()`` is a property that defines whether the mapping result is valid for the current unit. If not, it means that the unit won't be part of the newly created group.
- ``lane_mask()`` is the mask of threads within this warp that are part of this group.

.. note:: When a mapping result is not valid (``.is_valid() == false``), calling any other non-static methods is undefined behaviour.

Group Mapping Process
---------------------

The group mapping process has 3 steps:

1. The initial mapping result is created from the parent group's mapping result. The static group count is set to ``1``, group to ``0``, unit count and rank copied (or recomputed if the unit level was changed), the always exhaustive property is set to ``true`` and the always contiguous property is copied as well as the lane mask.
2. The mapping's ``.map(...)`` member function is invoked with the initial mapping result.
3. The transformed mapping result is used to initialize the newly created group.

The key idea is that a mapping tries to provide as many information as compile-time constants as possible. This information can be used to better optimize queries and cooperative algorithms.

Group Mapping Interface
-----------------------

Every group mapping type must implement the group mapping interface which can be defined as *group-mapping* concept:

.. code:: c++

  template <typename T, typename Unit, typename ParentGroup, typename PrevMappingResult>
  concept /*group-mapping*/ = requires (T& t,
                                        const Unit& unit,
                                        const ParentGroup& parent_group,
                                        const PrevMappingResult& prev_mapping_result)
  {
    { t.map(unit, parent_group, prev_mapping_result) } -> /*group-mapping-result*/;
  };

Where:

- ``Unit`` is the unit level type of the newly created group.
- ``ParentGroup`` is the parent group of the newly created group. It can be used to synchronize all units.
- ``PrevMappingResult`` is the input mapping result being transformed.

The design of

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
