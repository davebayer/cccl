.. _cudax-coop-group-generic-group:

``cudax::coop::generic_group``
==============================

.. code:: cuda

    namespace cudax::coop {

    template <typename Unit,
              typename ParentGroup,
              typename MappingResult,
              typename SynchronizerInstance>
    class generic_group : public /*group-interface-for*/<generic_group>
    {
    public:
      template <typename Mapping, typename Synchronizer>
        requires cuda::std::same_as</*mapping-result-of*/<Mapping>, MappingResult>
          && cuda::std::same_as</*synchronizer-instance-for*/<Unit, ParentGroup, MappingResult, Synchronizer>,
                                SynchronizerInstance>
      __device__ generic_group(const Unit& unit, const ParentGroup& parent_group, const Mapping& mapping) noexcept

      generic_group(const generic_group&) = delete;

      generic_group(generic_group&&) = delete;

      generic_group& operator=(const generic_group&) = delete;

      generic_group& operator=(generic_group&&) = delete;
    };

    template </*hierarchy-level-type*/ Unit,
              group ParentGroup,
              typename Mapping,
              typename Synchronizer,
              typename MappingResult = /*mapping-result-of*/<Unit, ParentGroup, Mapping>>
      requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
    generic_group(const Unit&, const ParentGroup&, const Mapping&, const Synchronizer&)
      -> generic_group<Unit,
                       ParentGroup,
                       MappingResult,
                       /*synchronizer-instance-for*/<Unit, ParentGroup, MappingResult, Synchronizer>>;

    } // namespace cudax::coop

Overview
--------

``cudax::coop::generic_group`` is a group type that splits the parent group into subgroups according to the ``mapping``. Every subgroup is instantiated with a synchronization mechanism provided by the ``synchronizer``, which makes the subgroups completely independent on each other. The group's constructor is the only statement that all of the units from the parent group are required to execute uniformly.

Queries
-------

Sub-unit queries work as with any other group types. The group may be queried for rank and (static) count within the parent group.

Examples
--------

.. TODO: Add usage examples.
