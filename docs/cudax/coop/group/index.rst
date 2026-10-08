.. _cudax-coop-group:

CCCL Cooperative Groups
=======================

Group families
--------------

.. toctree::
   :maxdepth: 1

   this_group
   generic_group
   virtual_group
   coalesced_group
   group_view

Mappings
--------

.. toctree::
   :maxdepth: 1

   mapping/identity_mapping
   mapping/group_by
   mapping/group_as
   mapping/take
   mapping/binary_partition
   mapping/composite_mapping

Synchronizers
-------------

.. toctree::
   :maxdepth: 1

   synchronizer/level_synchronizer
   synchronizer/lane_synchronizer
   synchronizer/interwarp_synchronizer
   synchronizer/barrier_synchronizer

Introduction to CCCL Cooperative Groups
---------------------------------------

CCCL Cooperative Groups is an implementation of a successor to CTK's Cooperative Groups (``<cooperative_groups.h>``). The goal is to provide a more capable API that provides better flexibility, performance and safety.

Concerning Groups
^^^^^^^^^^^^^^^^^

Let's start with a quick introduction to groups regarding the CUDA programming model. A group could be simply defined as a collection of threads that can synchronize, thus can cooperate on solving an algorithmic problems. However, such a generic group isn't too suitable for the CUDA programming model, because the definition doesn't allow an efficient implementation of the synchronization in the hardware.

CTK's Cooperative Groups solve this issue by adding an upper boundary within which the threads are grouped to the definition. It means that you can't simply group arbitrary threads, but you can only group threads within a given CUDA hierarchy level. This information is used by the implementation to select the right synchronization mechanism for a given group.

Even though this design solves the synchronization efficiency issue, the design could be improved even further. When writing a collective algorithm, often you can do an operation more efficiently when you know how the threads are actually grouped within the upper boundary. For example, when you have a group of threads within a block and you know that the threads always form a full warp, you could use this information to implement the algorithm more efficiently. However, CTK's Cooperative Groups don't store this information inside the group.

CCCL Cooperative Groups address this problem by changing the group definition again. In this new design, a group is a collection of units within an upper boundary that can synchronize, where the unit can be any CUDA hierarchy level below or same as the upper boundary. This means that when writing a cooperative algorithm, you what is the biggest building block of the group and optimize the algorithm accordingly.
