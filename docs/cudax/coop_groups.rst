.. _cudax-coop-groups:

CCCL Cooperative Groups
=======================

The groups API in ``cuda::experimental::coop`` describes collections of CUDA
execution units that cooperate and synchronize. It separates **which units
participate**, specified by a mapping, from **how they synchronize**, specified by
a synchronizer.

Include ``<cuda/experimental/coop/group>`` to use the API. The examples use
``cudax`` as an alias for ``cuda::experimental``:

.. code-block:: cpp

   #include <cuda/hierarchy>
   #include <cuda/experimental/coop/group>

   namespace cudax = cuda::experimental;

The API is experimental and subject to change. The examples below run in device
code; compile them with a CUDA compiler in C++20 mode and make the CCCL headers
available on the include path. This implementation also requires defining
``_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX`` before including any CUDA headers
to enable group queries on hierarchy levels. For example, from the repository root:

.. code-block:: bash

   nvcc -std=c++20 --expt-relaxed-constexpr \
     -D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX \
     -I cudax/include -I libcudacxx/include -c example.cu

.. contents:: On this page
   :local:
   :depth: 2

What is a group?
----------------

A group is a collection of **units** within an upper **level** of the CUDA
hierarchy. Both the unit and the level are part of the group's type:

* ``unit_type`` is the building block of the group, such as a thread or a warp.
  Every thread in a participating unit belongs to the group.
* ``level_type`` is the upper boundary containing all participating units, such
  as a warp or a block.
* ``hierarchy_type`` describes the hierarchy in which the group was created.
  ``hierarchy()`` returns that hierarchy by const reference.

For example, a group of warps within a block contains one or more complete warps
from that block. Synchronizing the group requires participation from every thread
in those warps, rather than just one representative per warp.

The unit lets an algorithm exploit known structure. A reduction over a group of
warps can first reduce within each warp and then combine the partial results.
The upper level constrains synchronization: a warp synchronization instruction
can synchronize threads within one warp, but cannot synchronize arbitrary threads
from different warps in a block.

The hierarchy must describe the execution configuration and contain at least the
group's upper level. It also provides the dimensions needed to translate queries
between levels. See :doc:`/libcudacxx/runtime/hierarchy` for the hierarchy API.

A mapping assigns participating units consecutive ranks starting at zero. It can
exclude units, in which case those units must check membership before using the
resulting group. A group also has a rank among the groups created from its parent.
These are separate quantities: a thread's rank in its group is not the group's
rank in its parent.

Creating groups
---------------

Groups for the current hierarchy level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``this_thread``, ``this_warp``, ``this_block``, ``this_cluster``, and ``this_grid``
represent the current instance of the corresponding hierarchy level. Each has
one unit, with ``unit_type`` equal to ``level_type``. For example, ``this_block``
contains one block, and therefore all threads in that block.

Construct these groups from a hierarchy, or from an object exposing a hierarchy
such as a launch configuration. The hierarchy must match the actual launch:

.. code-block:: cpp

   // Device code in a kernel launched with 128 threads per block.
   auto hierarchy = cuda::hierarchy(cuda::grid_dims(gridDim.x), cuda::block_dims<128>());

   cudax::coop::this_thread thread{hierarchy};
   cudax::coop::this_warp warp{hierarchy};
   cudax::coop::this_block block{hierarchy};

   // Select the group type generically from a hierarchy level.
   auto same_block = cudax::coop::make_this_group(cuda::block, hierarchy);

These groups are always exhaustive and contiguous. They have no parent group;
queries about their position instead name a hierarchy level at or above their
own level. Synchronization uses ``level_synchronizer``.

Generic groups
~~~~~~~~~~~~~~

Use ``generic_group`` to partition a parent group with a mapping and select the
synchronization mechanism:

.. code-block:: cpp

   cudax::coop::generic_group child{unit, parent, mapping, synchronizer};

``unit`` must be at or below the parent's unit level. The child inherits the
parent's upper level and hierarchy. When selecting a lower unit level, the mapping
operates on all such units in the parent. For example, selecting ``cuda::warp``
from ``this_block`` makes the block's warps available to the mapping.

Construction is a cooperative operation. All threads belonging to the parent
must construct the child uniformly, including threads that the mapping will
exclude. Only members of the parent may construct a child. A mapping or
synchronizer may synchronize the parent during construction.

After construction, a ``generic_group`` uses its own mapping and synchronizer
instance. It does not use the parent's synchronization resources unless the caller
explicitly supplies shared resources. Keep separately synchronized groups' resources
distinct while they are in use.

For example, divide each full warp into eight groups of four threads:

.. code-block:: cpp

   cudax::coop::this_warp parent{hierarchy};
   cudax::coop::generic_group tile{
     cuda::gpu_thread, parent, cudax::coop::group_by<4>{}, cudax::coop::lane_synchronizer{}};

   auto thread_rank = cuda::gpu_thread.rank(tile); // 0, 1, 2, or 3
   auto tile_rank   = tile.rank(parent);           // 0 through 7
   tile.sync();

The name ``cudax::coop::group`` denotes the group concept; it is not the concrete
class used for construction.

Lifetime and views
~~~~~~~~~~~~~~~~~~

``this_*`` groups and ``generic_group`` objects cannot be copied, moved, or
assigned. Pass them by reference, or use a copyable ``group_view``:

.. code-block:: cpp

   cudax::coop::group_view view{tile};
   cudax::coop::group_view threads{cuda::gpu_thread, parent};

A view retains the group's membership and synchronization resources. The second
form expresses the same membership using a lower unit level; it does not
partition the group. Keep the original group and its resources alive while a view
is in use.

A ``generic_group`` destructor calls its synchronizer instance's cleanup operation
for participating threads. For barrier synchronization, finish all uses of the
barrier before leaving the group's scope. Destruction is not an implicit final
``sync()``.

Group queries
-------------

Queries within a group
~~~~~~~~~~~~~~~~~~~~~~

Use a hierarchy unit at or below the group's ``unit_type`` to query its members:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Expression
     - Result
   * - ``unit.count(group)``
     - Number of query units in the group.
   * - ``unit.static_count(group)``
     - Compile-time count, or ``cuda::std::dynamic_extent`` if unknown.
   * - ``unit.rank(group)``
     - Calling unit's zero-based rank in the group.
   * - ``unit.is_part_of(group)``
     - Whether the calling unit belongs to the group.
   * - ``unit.is_root_rank(group)``
     - Whether the calling unit has rank zero.
   * - ``unit.count_as<T>(group)``
     - Count expressed as integer type ``T``.
   * - ``unit.rank_as<T>(group)``
     - Rank expressed as integer type ``T``.

For a group containing three warps, ``cuda::warp.count(group)`` is 3 and
``cuda::gpu_thread.count(group)`` is 96. A query below the group's unit level
combines the unit's rank in the group with the calling sub-unit's rank within
that unit.

Use ``is_part_of`` before runtime rank, count, root-rank, or synchronization
operations when a mapping can exclude the caller:

.. code-block:: cpp

   cudax::coop::generic_group first_ten{
     cuda::gpu_thread, parent, cudax::coop::take{10}, cudax::coop::lane_synchronizer{}};

   if (cuda::gpu_thread.is_part_of(first_ten))
   {
     auto rank = cuda::gpu_thread.rank(first_ten);
     first_ten.sync();
   }

Here ``parent`` is the full-warp group from the preceding example. All parent
threads construct ``first_ten``, but only the first ten participate afterward.

Queries relative to a parent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a generic group, pass the parent from which it was directly constructed:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Expression
     - Result
   * - ``child.count(parent)``
     - Number of groups produced in the parent.
   * - ``child.static_count(parent)``
     - Compile-time group count, or ``cuda::std::dynamic_extent``.
   * - ``child.rank(parent)``
     - This group's zero-based rank in the parent.
   * - ``child.count_as<T>(parent)``
     - Group count expressed as integer type ``T``.
   * - ``child.rank_as<T>(parent)``
     - Group rank expressed as integer type ``T``.

For ``this_*`` groups, replace the parent with a hierarchy level at or above the
group's level, provided the hierarchy supports the query:

.. code-block:: cpp

   auto threads_per_block = cuda::gpu_thread.count(block);
   auto blocks_in_grid    = block.count(cuda::grid);
   auto block_rank        = block.rank(cuda::grid);

Static properties
~~~~~~~~~~~~~~~~~

``Group::is_always_exhaustive()`` reports whether membership is guaranteed for
all input units. ``Group::is_always_contiguous()`` reports whether each group
occupies a contiguous range of physical units, numbered in ascending physical
order. Both properties are compile-time guarantees. A false result means that
the property is not guaranteed; a particular runtime instance may still satisfy it.

Mappings
--------

Mappings describe membership and ranks independently of the synchronization
mechanism. They retain compile-time counts and properties where possible so that
queries and cooperative algorithms can use that information.

Identity mapping
~~~~~~~~~~~~~~~~

``identity_mapping`` returns the previous mapping result unchanged. It is useful
in generic code that optionally applies a transformation.

.. code-block:: cpp

   cudax::coop::identity_mapping mapping{};

Group-by mapping
~~~~~~~~~~~~~~~~

``group_by`` partitions the input into groups of ``N`` consecutive units, where
``N`` must be positive. By default, the number of input units must be divisible
by ``N``. With ``non_exhaustive``, a trailing incomplete group is discarded and
its units do not belong to any result group.

.. code-block:: cpp

   // Compile-time group size.
   cudax::coop::group_by<4> static_mapping{};

   // Runtime group size.
   cudax::coop::group_by dynamic_mapping{4};

   // Discard any incomplete group at the end.
   cudax::coop::group_by partial_mapping{10, cudax::coop::non_exhaustive};

For example, applying the last mapping to 32 threads creates three groups of ten
threads and excludes threads with input ranks 30 and 31. The mapping preserves
contiguity when the input is contiguous.

Group-as mapping
~~~~~~~~~~~~~~~~

``group_as`` partitions consecutive units into groups with specified positive
sizes. The number of sizes is known at compile time. Supply a fixed-size range
of ``cuda::std::uint32_t`` values for runtime sizes, or a
``cuda::std::index_sequence`` for compile-time sizes:

.. code-block:: cpp

   #include <cuda/std/cstdint>
   #include <cuda/std/utility>

   const cuda::std::uint32_t sizes[]{4, 4, 1, 1};
   cudax::coop::group_as dynamic_mapping{sizes};
   cudax::coop::group_as static_mapping{cuda::std::index_sequence<4, 4, 1, 1>{}};
   cudax::coop::group_as partial_mapping{sizes, cudax::coop::non_exhaustive};

Without ``non_exhaustive``, the sum of sizes must equal the number of input
units. With the tag, the sum may be smaller, and units whose input ranks are at
least that sum are excluded. The requested sizes must not exceed the available
units. This mapping preserves contiguity when the input is contiguous.

Take mapping
~~~~~~~~~~~~

``take`` retains the first ``N`` units of each input group. There must be at least
``N`` available units. Units with ranks greater than or equal to ``N`` are
excluded. The operation preserves the number and ranks of groups and preserves
contiguity when the input is contiguous.

.. code-block:: cpp

   cudax::coop::take<16> static_mapping{16};
   cudax::coop::take dynamic_mapping{16};

The runtime form is not statically exhaustive. The compile-time form can retain
an exhaustive guarantee when its size equals the statically known input size.

Binary partition mapping
~~~~~~~~~~~~~~~~~~~~~~~~

``binary_partition`` splits threads according to a predicate. It currently
requires thread units and a parent whose upper level is ``cuda::warp_level``.
The predicate is a callable receiving the previous **mapping result**, so it can
inspect the ranks and counts produced by an earlier mapping.

.. code-block:: cpp

   auto mapping = cudax::coop::binary_partition{
     [](const auto& previous) { return previous.unit_rank() % 2 == 0; }};

   cudax::coop::generic_group partition{
     cuda::gpu_thread, parent, mapping, cudax::coop::lane_synchronizer{}};

To use a previously computed Boolean, capture it in a callable:

.. code-block:: cpp

   bool predicate = (cuda::gpu_thread.rank(parent) < 3);
   auto mapping = cudax::coop::binary_partition{
     [predicate](const auto&) { return predicate; }};

For a single input group, the false partition has group rank 0 and the true
partition has group rank 1. The group count is two even if one partition is
empty. All previously participating threads remain in a partition, so the mapping
preserves exhaustiveness. It does not guarantee contiguity.

Composite mappings
~~~~~~~~~~~~~~~~~~

Combine mappings with ``operator|``. The result of the left mapping becomes the
input to the next mapping. Only the final result is used to instantiate the
synchronizer; intermediate mappings do not construct independently synchronized
groups.

.. code-block:: cpp

   #include <cuda/std/utility>

   auto mapping = cudax::coop::group_by<4>{}
                | cudax::coop::group_as{cuda::std::index_sequence<3, 1>{}};

Applied to a full warp, this first forms eight groups of four threads, then
splits each into groups of three and one. The final result contains sixteen
groups. Querying the resulting generic group's rank or count uses its original
parent, not an intermediate mapping result.

Synchronization
---------------

Every group provides two synchronization operations:

* ``sync()`` permits participating threads to reach the corresponding operation
  from different locations in the code.
* ``sync_aligned()`` requires all participating threads to execute the operation
  uniformly at the same location in the code.

Both require every thread in every participating unit to take part in the
corresponding synchronization phase. An aligned call can permit a more efficient
hardware instruction. Neither operation changes group membership.

Level synchronizer
~~~~~~~~~~~~~~~~~~

``level_synchronizer`` supplies the synchronization used by ``this_*`` groups.
It synchronizes the whole hierarchy level, so it is not suitable for an arbitrary
subset of that level.

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Level
     - ``sync()``
     - ``sync_aligned()``
   * - Thread
     - No operation.
     - No operation.
   * - Warp
     - ``__syncwarp()``
     - ``__syncwarp()``
   * - Block
     - ``__barrier_sync(0)``
     - ``__syncthreads()``
   * - Cluster
     - Cluster barrier arrive and wait.
     - Aligned cluster barrier arrive and wait.
   * - Grid
     - Block barriers around an atomic inter-block barrier.
     - Aligned block barriers around an atomic inter-block barrier.

Cluster synchronization uses cluster instructions on SM 90 and later; the
implementation uses block synchronization on earlier architectures, where the
cluster is limited to a single block. Grid synchronization requires a cooperative
kernel launch and driver support. Constructing a ``this_grid`` object does not
itself make an ordinary launch cooperative.

Lane synchronizer
~~~~~~~~~~~~~~~~~

``lane_synchronizer`` synchronizes thread units with ``__syncwarp(mask)``. All
members of each resulting group must lie in the same physical warp. Having at
most 32 members is not sufficient if they span multiple warps. Both synchronization
operations use the group's lane mask.

Barrier synchronizer
~~~~~~~~~~~~~~~~~~~~

``barrier_synchronizer`` takes a contiguous range of ``cuda::barrier`` objects.
There must be at least one barrier for each result group; a group's rank selects
its barrier. During construction, one thread per group initializes its barrier
with the group's **thread count**, then the parent is synchronized to make the
barriers ready for use. Do not separately initialize these barriers.

The barrier scope must cover the group's upper level. Warp and block levels
require at least block scope; cluster and grid levels require at least device
scope. The caller must also place the barriers in memory accessible to every
participating thread. Block-scoped shared-memory barriers are suitable for groups
within a block.

Both synchronization operations call ``arrive_and_wait()``. The group cleans up
its barrier on destruction. Keep the supplied storage alive, and do not reuse or
destroy a barrier while any participant can still access it. A final group
synchronization after the work can establish that all members have finished.

Inter-warp synchronizer
~~~~~~~~~~~~~~~~~~~~~~~

``interwarp_synchronizer`` synchronizes groups of complete warps within a block
using native barrier identifiers. Supply a range containing at least one
identifier per resulting group. Each identifier must be in the range 0 through
15. Use distinct identifiers for concurrently used barriers, and avoid identifier
0 when it is also used for block synchronization.

Both synchronization operations use ``__barrier_sync_count`` with an arrival
count equal to the number of participating threads.

.. code-block:: cpp

   // A block with four warps, split into two groups of two warps.
   unsigned barrier_ids[]{1, 2};
   cudax::coop::generic_group pair{
     cuda::warp, block, cudax::coop::group_by<2>{},
     cudax::coop::interwarp_synchronizer{barrier_ids}};
   pair.sync();

Mapping and synchronizer protocols
----------------------------------

Custom mappings
~~~~~~~~~~~~~~~

A mapping implements ``map(unit, parent_group, previous_mapping_result)``. The
parent can be used for cooperative work during construction. When mappings are
composed, the parent remains the original parent while the previous mapping
result changes at each step.

The returned mapping result is copy constructible and provides the following
operations. Static counts and properties preserve information in the type:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Operation
     - Return type
     - Meaning
   * - ``T::static_group_count()``
     - ``cuda::std::size_t``
     - Compile-time number of groups, or ``cuda::std::dynamic_extent``.
   * - ``result.group_count()``
     - ``cuda::std::uint32_t``
     - Runtime number of groups.
   * - ``result.group_rank()``
     - ``cuda::std::uint32_t``
     - Current group's rank.
   * - ``T::static_unit_count()``
     - ``cuda::std::size_t``
     - Compile-time number of units, or ``cuda::std::dynamic_extent``.
   * - ``result.unit_count()``
     - ``cuda::std::uint32_t``
     - Number of units in this group.
   * - ``result.unit_rank()``
     - ``cuda::std::uint32_t``
     - Current unit's rank.
   * - ``T::is_always_exhaustive()``
     - ``bool``
     - Whether all input units are guaranteed to participate.
   * - ``T::is_always_contiguous()``
     - ``bool``
     - Whether physical membership and rank order are guaranteed contiguous.
   * - ``result.lane_mask()``
     - ``cuda::device::lane_mask``
     - Participating lanes in the calling thread's warp.
   * - ``result.is_valid()``
     - ``bool``
     - Whether the caller belongs to the result group.

Runtime counts must agree with known static counts. Valid ranks are less than
the corresponding counts. For thread units, the lane mask contains the calling
thread and the other participating lanes in its warp; for larger units, it is
``cuda::device::lane_mask::all()``. A contiguous result must number its physical
units in ascending order, including after mapping composition.

The implementation's mapping-result concept checks most of this interface;
``is_valid()`` is additionally required by group construction and queries.
These extension interfaces are experimental implementation contracts.

Custom synchronizers
~~~~~~~~~~~~~~~~~~~~

A synchronizer implements
``make_instance(unit, parent_group, mapping_result)``. It can use the parent for
synchronization during construction, including participation from units that the
new mapping excludes. It returns an instance storing the state needed by the
new group.

The group invokes the following instance operations:

.. code-block:: text

   instance.do_sync(mapping_result, hierarchy)
   instance.do_sync_aligned(mapping_result, hierarchy)
   instance.deinit(mapping_result, hierarchy)

The synchronization operations are callable on a const instance and return
``void``. Cleanup is invoked at group destruction for members of the group.
Instances supporting ``group_view`` also provide ``view()``, which returns a
copyable view of the synchronization state. The generic construction path for a
non-exhaustive parent additionally requires ``Instance::invalid()``.

Additional group types
----------------------

``coalesced_group{hierarchy}`` captures the currently active threads in the
calling warp. Its unit is a thread, its upper level is a warp, and it uses lane
synchronization. It does not guarantee contiguous membership. Construct it at the
point where the active threads should form the group.

``virtual_group{unit, parent, mapping}`` applies a mapping while reusing a view
of the parent's synchronizer instance. It does not create independent
synchronization resources. Keep the parent alive and use it only when the
inherited synchronization mechanism is valid for the mapped membership. For
example, a block-wide synchronizer still requires the whole block to participate,
even if a virtual group's mapping selects fewer threads.

Interoperability with CUDA Cooperative Groups
---------------------------------------------

When CUDA Toolkit Cooperative Groups headers are available, use
``make_cg_equivalent_group(cg_group, hierarchy)`` to create a CCCL equivalent.
The current conversions include:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - CUDA Cooperative Groups type
     - CCCL equivalent
   * - ``cooperative_groups::grid_group``
     - ``this_grid``
   * - ``cooperative_groups::cluster_group`` (when available)
     - ``this_cluster``
   * - ``cooperative_groups::thread_block``
     - ``this_block``
   * - ``cooperative_groups::coalesced_group``
     - ``coalesced_group``, for an unpartitioned coalesced group.
   * - ``thread_block_tile<1, void>``
     - ``this_thread``
   * - ``thread_block_tile<N, thread_block>`` with ``N < 32``
     - A thread group within a warp using ``group_by<N>`` and lane synchronization.
   * - ``thread_block_tile<32, thread_block>``
     - ``this_warp``

Other tile sizes and parent types are not currently supported. Converting a
coalesced group requires the active group at the conversion point to match the
source; dynamically partitioned coalesced groups are not supported.

.. code-block:: cpp

   #include <cooperative_groups.h>

   auto cg_block = cooperative_groups::this_thread_block();
   auto equivalent = cudax::coop::make_cg_equivalent_group(cg_block, hierarchy);

Warp-specialized programming example
------------------------------------

This kernel assigns eleven warps to five squads of sizes 4, 4, 1, 1, and 1. Each
squad has its own barrier, so it can synchronize independently after construction.
Launch with exactly 352 threads per block and allocate one output element per
launched thread.

.. code-block:: cpp

   #include <cuda/barrier>
   #include <cuda/hierarchy>
   #include <cuda/std/cstdint>
   #include <cuda/experimental/coop/group>

   namespace cudax = cuda::experimental;

   __global__ void warp_specialized(int* output)
   {
     enum : unsigned
     {
       reduce_squad,
       scan_store_squad,
       load_squad,
       sched_squad,
       lookback_squad,
       squad_count
     };

     auto hierarchy = cuda::hierarchy(cuda::grid_dims(gridDim.x), cuda::block_dims<352>());
     cudax::coop::this_block parent{hierarchy};
     const cuda::std::uint32_t sizes[]{4, 4, 1, 1, 1};
     __shared__ cuda::barrier<cuda::thread_scope_block> barriers[squad_count];

     cudax::coop::generic_group squad{
       cuda::warp, parent, cudax::coop::group_as{sizes},
       cudax::coop::barrier_synchronizer{barriers}};

     int value = 0;
     switch (squad.rank(parent))
     {
       case reduce_squad:
       case scan_store_squad:
       case load_squad:
         value = cuda::gpu_thread.is_root_rank(squad) ? 2 : 3;
         break;
       case sched_squad:
       case lookback_squad:
         value = 1;
         break;
     }

     output[blockIdx.x * blockDim.x + threadIdx.x] = value;
     squad.sync();
   }

All block threads construct ``squad`` before branching on the squad's rank.
``cuda::warp.count(squad)`` reports the squad's number of warps, while
``cuda::gpu_thread.count(squad)`` reports its number of threads. Only one thread
per squad satisfies ``cuda::gpu_thread.is_root_rank(squad)``. The final
synchronization completes before the squad's barrier is cleaned up.
