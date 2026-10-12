.. _cudax-coop:

CCCL Cooperative
================

CCCL Cooperative is a set of device-side APIs for organizing groups of collaborating threads and execution of collective algorithms. They are placed in ``cudax::coop`` namespace and available through ``<cuda/experimental/coop/_header_>`` include files.

These APIs are successors to both CTK Cooperative Groups (``<cooperative_groups.h>``) and CUB thread, warp and block-wide collective algorithms. The goal is to provide a more capable API that provides better flexibility, performance and safety.

.. toctree::
  :maxdepth: 1

  group/index
  algorithm/index
