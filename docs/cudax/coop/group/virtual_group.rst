.. _cudax-coop-group-virtual-group:

``cudax::coop::virtual_group``
==============================

.. code:: cuda

   namespace cudax::coop {

   template <typename Unit, typename ParentGroup, typename MappingResult>
   class virtual_group : public /*group-interface-for*/<virtual_group>
   {
   public:
     template <typename Mapping>
       requires cuda::std::same_as</*mapping-result-of*/<Mapping>, MappingResult>
     __device__ virtual_group(const Unit& unit, const ParentGroup& parent_group, const Mapping& mapping) noexcept

    virtual_group(const virtual_group&) = delete;

    virtual_group(virtual_group&&) = delete;

    virtual_group& operator=(const virtual_group&) = delete;

    virtual_group& operator=(virtual_group&&) = delete;
   };

   template </*hierarchy-level-type*/ Unit, group ParentGroup, typename Mapping>
     requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
   virtual_group(const Unit&, const ParentGroup&, const Mapping&)
     -> virtual_group<Unit, ParentGroup, /*mapping-result-of*/<Mapping>>;

   } // namespace cudax::coop

Overview
--------

``cudax::coop::virtual_group`` is a group type that provides an abstraction layer over a group. It allows you to take an existing group, split it into subgroups while still using the parent's group synchronization mechanism. This means that the created subgroups are not truly independent on each other and are required to share synchronization points.

Because of this limitation, virtual group doesn't support optional participation in the group. If the input mapping outputs a unit without a valid mapping result, the behaviour is undefined.

The type can be constructed by specifying what is being grouped (``unit``) within what (``parent_group``) and how (``mapping``). The type is not copyable, movable, nor assignable. The user should always use deduction guides to deduce the template parameters.

Queries
-------

Sub-unit queries work as with any other group types. The group may be queried for rank and (static) count within the parent group.

Examples
--------

.. TODO: Add usage examples.
