.. _cudax-coop-group-coalesced-group:

``cudax::coop::coalesced_group``
================================

.. code:: cuda

   namespace cudax::coop {

   template <typename Hierarchy>
   class coalesced_group;

   template </*hierarchy-like-type*/ HierarchyLike>
   coalesced_group(const HierarchyLike&) -> coalesced_group</*hierarchy-type-of*/<HierarchyLike>>;

   } // namespace cudax::coop

Overview
--------

``cudax::coop::coalesced_group`` is a predefined group that groups all currently active threads within a warp. It is useful for opportunistic programming. The mapping result is determined based on the lane mask returned by the ``__activemask()`` intrinsic. The group can be explicitly constructed from a hierarchy-like object.

A *coalesced* group is always exhaustive, however it's not always contiguous.

A *coalesced* group uses the :ref:`cudax::coop::lane_synchronizer <cudax-coop-group-synchronizer-lane-synchronizer>` synchronizer for synchronization.

Queries
-------

Sub-unit queries work as with any other group types, however since the *coalesced* group doesn't have a parent group, a native hierarchy level must be used in the super-level queries as the upper bound. For example ``cudax::coop::coalesced_group{...}.count(cuda::block)`` queries the number of coalesced groups within a block. It is is equivalent to the number of warps within a block.

Examples
--------

.. TODO: Add usage examples.
