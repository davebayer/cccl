[//]: <> (asdf)


# CCCL Cooperative

## Why?
- We need a good abstraction that will help us organize threads into groups that can cooperate on solving problems
  - CCCL Cooperative Groups

- We need APIs that will implement algorithms that solve the common problems
  - CCCL Cooperative Algorithms

## When?
Ambitious goal: GTC 03/27

[//]: <> (The contents should be a timeline of 2 columns, the first one for groups, the second one for algorithms, those lines with ? comment are plans, make them different from the others)
- 01/26 First ideas on groups
- 03/26 Initial implementation
- 05/26 Improved implementation
- 08/26 Documentation
- 11/26 Alpha release to gather feedback
- 01/27 Beta release [//]: <> (?)
- 03/27 Release [//]: <> (?)

- 04/26 First ideas on algorithms
- 06/26 Initial `reduce` implementation
- 07/26 Initial `any_of`, `all_of`, `shuffle` implementation
- 11/26 Finalizing algorithms design, reworking existing algorithms
- ??/?? Alpha release to gather feedback [//]: <> (?)
- ??/?? Beta release [//]: <> (?)
- 03/27 (Partial) release [//]: <> (?)

# CCCL Cooperative Groups

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns with the definition)
- Generic Group
  - Groups arbitrary threads.
- CTK's Cooperative Groups
  - Groups threads within an upper boundary.
- CCCL Cooperative Groups
  - Groups units within an upper boundary.

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns, all of the pictures should be a block of 128 threads)
- Generic Group
  [//]: <> (Picture of arbitrary GPU threads being highlighted)
- CTK's Cooperative Groups
  [//]: <> (Picture of arbitrary GPU threads being highlighted within the block being the upper bound)
- CCCL Cooperative Groups
  [//]: <> (Picture of arbitrary GPU threads being highlighted within the block being the upper bound)

## What makes them different?
### Group Definition
[//]: <> (Here I want 3 columns, all of the pictures should be a block of 128 threads)
- Generic Group
  [//]: <> (Picture of GPU threads from first and third warp being highlighted)
- CTK's Cooperative Groups
  [//]: <> (Picture of GPU threads from first and third warp being highlighted within the block being the upper bound)
- CCCL Cooperative Groups
  [//]: <> (Picture of GPU threads from first and third warp being highlighted within the block being the upper bound, the warps should be highlighted, too)

## What makes them different?
### Hierarchy-awareness
[//]: <> (Here I want 2 columns, this example computes 1D thread rank within a block, non-of the dimensions are statically known)
- CTK's Cooperative Groups
  ```cpp
  // What cooperative_groups::this_thread_block().thread_rank() would compute.
  const auto thread_rank = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```
- CCCL Cooperative Groups
  ```cpp
  // What CCCL Cooperative Groups equivalent would compute.
  const auto thread_rank = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```

## What makes them different?
### Hierarchy-awareness
[//]: <> (Here I want 2 columns, this example computes 1D thread rank within a block, we know the block is 1D)
- CTK's Cooperative Groups
  ```cpp
  // What cooperative_groups::this_thread_block().thread_rank() would compute.
  const auto thread_rank = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ```
- CCCL Cooperative Groups
  ```cpp
  // What CCCL Cooperative Groups equivalent would compute.
  const auto thread_rank = threadIdx.x;
  ```

## What makes them different?
### Group queries
[//]: <> (Here I want 2 columns, again the picture of GPU threads within a block, Both should show the same thread rank)
- CTK's Cooperative Groups
- CCCL Cooperative Groups

## What makes them different?
### Group queries
[//]: <> (Here I want 2 columns, again the picture of GPU first and third warp within a block)
- CTK's Cooperative Groups
  [//]: <> (Only shows thread rank)
- CCCL Cooperative Groups
  [//]: <> (Shows both thread rank and warp rank)

## What makes them different?
### Separation of Partition and Synchronization
When creating a group, *how the units are grouped* are independent on and *how the units are synchronized*.
[//]: <> (Place here similar picture to docs/cudax/coop/group/images/group-design.svg)

## What makes them different?
### Optional participation
CCCL Cooperative Groups add support for non-exhaustive mapping.
[//]: <> (Place here a picture of threads within a warp being grouped by 5, leaving 2 threads without any group. The parent groups should be placed on the left, then an arrow to the mapping that groups by 5 and arrow of the split parent group)
