.. _cudax-coop-group-synchronizer:

Synchronizer Instance
=====================

Group synchronizers are types that implement the synchronizer interface. They define what synchronization mechanism shall be used to synchronize the group. The type needs to implement the ``.make_instance(...)`` method that returns a type that satisfies the *group synchronizer instance* concept.

Group Synchronizer Instance Concept
----------------------------

*Group synchronizer instance* is an implementation-defined concept that defines the interface for a group synchronizer instance. It is defined as:

.. code:: c++

  template <typename T, typename MappingResult, typename Hierarchy>
  concept /*group-synchronizer-instance*/ = requires (T& t,
                                                      const T& ct,
                                                      const MappingResult& mapping_result,
                                                      const Hierarchy& hierarchy)
  {
    // Lifetime-related methods.
    { t.deinit(mapping_result, hierarchy) } -> cuda::std::same_as<void>;
    { ct.view() } -> /*group-synchronizer-instance*/<MappingResult, Hierarchy>;

    // Synchronization interface.
    { ct.do_sync(mapping_result, hierarchy) } -> cuda::std::same_as<void>;
    { ct.do_sync_aligned(mapping_result, hierarchy) } -> cuda::std::same_as<void>;
  };

Where:

- ``deinit(...)`` is a method that deinitializes the synchronizer instance.
- ``view()`` is a method that creates a non-owning view over the synchronizer instance resources. It must also satisfy the *group-synchronizer-instance* concept.
- ``do_sync(...)`` is a method that triggers the synchronization. It doesn't require the units to be contextually-aligned.
- ``do_sync_aligned(...)`` is a method that triggers the synchronization. The caller guarantees that all of the units are contextually-aligned (execute the exact same statement in the code).

.. note:: When a mapping result is not valid (``.is_valid() == false``), any of the synchronization interfaces is undefined behaviour.

Group Synchronizer Instantiation Process
----------------------------------------

The group mapping process has 3 steps:

1. The initial mapping result is created from the parent group's mapping result. The static group count is set to ``1``, group to ``0``, unit count and rank copied (or recomputed if the unit level was changed), the always exhaustive property is set to ``true`` and the always contiguous property is copied as well as the lane mask.
2. The mapping's ``.map(...)`` member function is invoked with the initial mapping result.
3. The transformed mapping result is used to initialize the newly created group.

The key idea is that a mapping tries to provide as many information as compile-time constants as possible. This information can be used to better optimize queries and cooperative algorithms.

Group Synchronizer Interface
----------------------------

Every group mapping type must implement the group synchronizer interface which can be defined as *group-synchronizer* concept:

.. code:: c++

  template <typename T, typename Unit, typename ParentGroup, typename MappingResult>
  concept /*group-synchronizer*/ = requires (T& t,
                                             const Unit& unit,
                                             const ParentGroup& parent_group,
                                             const PrevMappingResult& prev_mapping_result)
  {
    { t.make_instance(unit, parent_group, mapping_result) } -> /*group-synchronizer-instance*/;
  };

Where:

- ``Unit`` is the unit level type of the newly created group.
- ``ParentGroup`` is the parent group of the newly created group. It can be used to synchronize all units.
- ``MappingResult`` is the final mapping result that describes the newly created group.

Predefined Synchronizers
------------------------

.. toctree::
  :maxdepth: 1

  level_synchronizer
  lane_synchronizer
  interwarp_synchronizer
  barrier_synchronizer
