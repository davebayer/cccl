.. _cudax-coop-group:

CCCL Cooperative Groups
=======================

.. toctree::
  :maxdepth: 1

  mapping/index
  synchronizer/index

CCCL Cooperative Groups are available through the ``<cuda/experimental/coop/group>`` header.

Concerning Groups
-----------------

Let's start with a quick introduction to groups regarding the CUDA programming model. A group could be simply defined as a collection of threads that can synchronize, thus can cooperate on solving an algorithmic problems. However, such a generic group isn't too suitable for the CUDA programming model, because the definition doesn't allow an efficient implementation of the synchronization in the hardware.

CTK's Cooperative Groups solve this issue by adding an upper boundary within which the threads are grouped to the definition. It means that you can't simply group arbitrary threads, but you can only group threads within a given CUDA hierarchy level. This information is used by the implementation to select the right synchronization mechanism for a given group.

Even though this design solves the synchronization efficiency issue, the design could be improved even further. When writing a collective algorithm, often you can do an operation more efficiently when you know how the threads are actually grouped within the upper boundary. For example, when you have a group of threads within a block and you know that the threads always form a full warp, you could use this information to implement the algorithm more efficiently. However, CTK's Cooperative Groups don't store this information inside the group.

CCCL Cooperative Groups address this problem by changing the group definition again. In this new design, a group is a collection of units within an upper boundary that can synchronize, where the unit can be any CUDA hierarchy level below or same as the upper boundary. This means that when writing a cooperative algorithm, you what is the biggest building block of the group and optimize the algorithm accordingly.



Optional Participation
----------------------



Requirements on Groups
----------------------

The requirements on groups are formally specified by the ``cudax::coop::group`` concept, which is defined as:

.. code:: cpp

  namespace cudax::coop
  {

  template <typename T>
  concept group = requires (const T& t)
  {
    // Member types.
    requires /*is-hierarchy-level*/<typename T::unit_type>;
    requires /*is-hierarchy-level*/<typename T::level_type>;
    requires /*is-hierarchy*/<typename T::hierarchy_type>;

    // Getters.
    { t.hierarchy() } noexcept -> cuda::std::same_as<typename T::hierarchy_type&>;

    // Synchronization.
    { t.sync() } noexcept -> cuda::std::same_as<void>;
    { t.sync_aligned() } noexcept -> cuda::std::same_as<void>;

    // Properties.
    { T::is_always_exhaustive() } noexcept -> cuda::std::same_as<bool>; // must be constexpr
    { T::is_always_contiguous() } noexcept -> cuda::std::same_as<bool>; // must be constexpr
  };

  } // namespace cudax::coop

Where:

- ``unit_type`` is the unit level type that defines what units are being grouped.
- ``level_type`` is the upper level bound type that defines within which level the units are grouped.
- ``hierarchy_type`` is the hierarchy type within which the group is operating.
- ``hierarchy()`` is a member function that returns ``const hierarchy_type&`` of the group.
- ``sync()`` is a member function that synchronizes all units that are part of the group.
- ``sync_aligned()`` is same as ``sync()``, but all of the units are contextually aligned (execute the same instruction).
- ``is_always_exhaustive()`` is a compile-time known property that specifies whether all units are always part of a group.
- ``is_always_contiguous()`` is a compile-time known property that specifies whether all units within the group are mapped to a contiguous block of hardware resources.

Group Queries
-------------

Group queries can be used to study group-related information. They are similar to hierarchy queries, however only a subset of the hierarchy queries is supported. There are 2 types of group queries - the sub-unit and the super-level queries.

Sub-Unit Queries
^^^^^^^^^^^^^^^^

Sub-unit queries can be used to query information about levels that are below the group's unit type. They look as:

.. code:: cpp

  qunit.query(group);

Where:

- ``qunit`` is the query unit which must be below or same as the group's unit type.
- ``query`` is one of the supported query kinds.
- ``group`` is the group the query is done for.

The supported query kinds are:

.. code:: cpp
  qunit.static_count(group); // (1), always constexpr
  qunit.count_as<T>(group);  // (2)
  qunit.rank_as<T>(group);   // (3)
  qunit.is_part_of(group);   // (4)

  qunit.count(group);        // (5)
  qunit.rank(group);         // (6)
  qunit.is_root_rank(group); // (7)

#. Queries the statically-known number of ``qunit`` units inside the ``group`` group. When equal to ``cuda::std::dynamic_extent``, it's only runtime-known value.
#. Queries the number of ``qunit`` units inside the ``group`` group returned as type ``T``.
#. Queries the current ``qunit``'s rank within the ``group`` group. Returns a value in range ``[0, qunit.count(group))`` as type ``T``.
#. Queries whether ``group`` is a valid object for the current ``qunit``. Always returns ``true`` when ``group.is_always_exahustive() == true``.
#. Equivalent to ``qunit.count_as</*default-T*/>(group)``.
#. Equivalent to ``qunit.rank_as</*default-T*/>(group)``.
#. Equivalent to ``qunit.rank(group) == 0``.

Super-Level Queries
^^^^^^^^^^^^^^^^^^^

Super-level queries can be used to query a group's information within the group's parent group or a hierarchy level when the group is one of the special group types. When it's a hierarchy level, it must be same as or above group's level type. See the concrete group's documentation for more information.

Super-level queries look as:

.. code:: cpp

  group.query(/*parent-group-or-qlevel*/);

Where:

- ``group`` is the group the ``query`` is done for.
- ``query`` is one of the supported query kinds.
- ``/*parent-group-or-qlevel*/`` is either ``group``'s parent group or a hierarchy level.

The supported query kinds are:

.. code:: cpp

  // Super-level queries.
  group.static_count(/*parent-group-or-qlevel*/); // (1), always constexpr
  group.count_as<T>(/*parent-group-or-qlevel*/);  // (2)
  group.rank_as<T>(/*parent-group-or-qlevel*/);   // (3)

  group.count(/*parent-group-or-qlevel*/);        // (4)
  group.rank(/*parent-group-or-qlevel*/);         // (5)

#. Queries the statically-known number of ``group`` groups made from ``group``'s parent group or contained within ``qlevel`` level. When equal to ``cuda::std::dynamic_extent``, it's only runtime-known value.
#. Queries the number of number of ``group`` groups made from ``group``'s parent group or contained within ``qlevel`` level returned as type ``T``.
#. Queries the current ``group``'s rank within the ``group``'s parent group or ``qlevel`` level. Returns a value in range ``[0, qunit.count(/*parent-group-or-qlevel*/))`` as type ``T``.
#. Equivalent to ``group.count_as</*default-T*/>(/*parent-group-or-qlevel*/)``.
#. Equivalent to ``group.rank_as</*default-T*/>(/*parent-group-or-qlevel*/)``.

.. warning:: All non-static queries except for ``is_part_of(...)`` query can be done only by units that belong to the group. Executing any non-static query by units for which ``qunit.is_part_of(group) == false`` is undefined behaviour.

Predefined Groups
-----------------

CCCL Cooperative Groups come with several predefined group types:

- :ref:`This Groups <cudax-group-this-group>`
- :ref:`Coalesced Group <cudax-group-coalesced-group>`
- :ref:`Generic Group <cudax-group-generic-group>`
- :ref:`Virtual Group <cudax-group-virtual-group>`
- :ref:`Group View <cudax-group-group-view>`
