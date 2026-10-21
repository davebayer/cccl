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

##
