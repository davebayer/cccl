.. _cudax-coop-group-mapping-group-by:

``cudax::coop::group_by``
=========================

.. code:: cuda

   namespace cudax::coop {

   template <cuda::std::size_t StaticUnitCount, bool IsAlwaysExhaustive>
   class group_by
   {
     cuda::std::uint32_t /*unit_count_*/; // exposition-only

   public:
     __device__ explicit group_by(cuda::std::uint32_t unit_count) noexcept
       requires IsAlwaysExhaustive
       : /*unit_count_*/{unit_count}
     {}

     __device__ explicit group_by(const cudax::coop::non_exhaustive_t&, cuda::std::uint32_t unit_count) noexcept
       requires (!IsAlwaysExhaustive)
       : /*unit_count_*/{unit_count}
     {}

     template <typename Unit, typename ParentGroup, typename PrevMappingResult>
     [[nodiscard]] __device__
     auto map(const Unit&, const ParentGroup&, const PrevMappingResult&) const noexcept;
   };

   } // namespace cudax::coop

Overview
--------

``cudax::coop::take`` is a mapping that splits every group from the previous mapping result into groups of ``unit_count`` units. The ``unit_count`` can be either static (when the type is ``/*integral_constant-like*/``) or dynamic otherwise.

By default, the mapping is always exhaustive, so if the previous mapping result's ``unit_count`` is not divisible by ``unit_count`` without a remainder, the behaviour is undefined. This behaviour can be altered by passing the ``cudax::coop::non_exhaustive`` tag as the first parameter to the mapping. Then, all of the ranks that would not form a full group, will be left without a group.

The returned mapping result is always contiguous if the previous mapping result was contiguous.

Examples
--------

.. TODO: Add usage examples.
