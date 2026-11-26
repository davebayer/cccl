.. _cudax-coop-group-synchronizer-interwarp-synchronizer:

``cuda::experimental::coop::interwarp_synchronizer``
====================================================

.. code:: cuda

  namespace cuda::experimental::coop {

  using /*interwarp-barrier-id-type*/ = cuda::std::uint32_t;

  template <typename RangeView>
  class interwarp_synchronizer
  {
    RangeView /*barrier_ids_*/; // exposition-only

  public:
    template <typename Range>
      requires cuda::std::same_as<RangeView, cuda::std::ranges::views::all_t<Range>>
    __device__ interwarp_synchronizer(Range&& barrier_ids)
      : /*barrier_ids_*/{cuda::std::ranges::views::all(cuda::std::forward<Range>(barrier_ids))}
    {}

    template <typename Unit, typename ParentGroup, typename MappingResult>
      requires (!cuda::std::ranges::range<const RangeView>)
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&);

    template <typename Unit, typename ParentGroup, typename MappingResult>
      requires cuda::std::ranges::range<const RangeView>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const;
  };

  template <typename Range>
    requires cuda::std::ranges::viewable_range<Range>
      && cuda::std::is_convertible_v<cuda::std::ranges::range_value_t<Range>, /*interwarp-barrier-id-type*/>
  interwarp_synchronizer(Range&&)
    -> interwarp_synchronizer<cuda::std::ranges::views::all_t<Range>>;

  template <typename Int, Int... Ids>
  interwarp_synchronizer(const cuda::std::integer_sequence<Int, Ids...>&)
    -> interwarp_synchronizer<cuda::std::index_sequence<cuda::narrow<cuda::std::size_t>(Ids)...>>;

  template <auto Range, typename T>
    requires /*compile-time-sized-forward-range-of-indices*/<decltype((Range))>
  interwarp_synchronizer(const cuda::std::constant_wrapper<Range, T>&)
    -> interwarp_synchronizer</*index-sequence-of*/<Range>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::interwarp_synchronizer`` is a synchronizer that uses one of the 16 built-in warp barriers to synchronize the group. The type is constructible from a range of barrier IDs, which are sequentially assigned to each of the created groups during the synchronizer instance creation. It can only be used with groups that group warps within the block level.

To supply barrier IDs at compile time, pass a ``cuda::std::integer_sequence`` or wrap a compile-time sized forward range in ``cuda::std::cw``.
The synchronizer instance retains the ID sequence in its type.
When dispatch specializes the group rank, the instance uses the ID at that index as a compile-time constant.
When dispatch specializes the unit count, it uses the corresponding number of threads as the barrier's participant count.
For example, groups with the same size can share a ``by_unit_count`` callback while synchronizing on distinct barrier IDs.

Users should always rely on template argument deduction and never set the template arguments themselves. All of the barrier IDs must be less than ``16`` and can't be repeated multiple times.

.. warning:: The barrier ID ``0`` is also used by the ``__syncthreads()`` operation. It is recommended to avoid this barrier ID and start from barrier ID ``1`` instead.

Examples
--------

.. code:: cuda

  #include <cuda/std/array>
  #include <cuda/std/ranges>
  #include <cuda/std/utility>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Interwarp synchronizer from a list of barrier IDs.
    cuda::std::array array{1, 3, 5, 7};
    const cudax::coop::interwarp_synchronizer s1{array};

    // Interwarp synchronizer from an infinite iota range.
    const cudax::coop::interwarp_synchronizer s2{cuda::std::ranges::views::iota(1)};

    // Ranks 0, 1, and 2 use barriers 7, 3, and 12, respectively.
    const cudax::coop::interwarp_synchronizer s3{cuda::std::integer_sequence<int, 7, 3, 12>{}};

  #if __cplusplus >= 202002L
    // A compile-time array supplies the same IDs.
    const cudax::coop::interwarp_synchronizer s4{cuda::std::cw<cuda::std::array{7, 3, 12}>};
  #endif
  }
