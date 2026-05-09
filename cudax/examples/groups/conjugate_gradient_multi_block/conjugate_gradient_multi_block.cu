/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample implements a conjugate gradient solver on GPU using
 * Multi Block Cooperative Groups, also uses Unified Memory.
 *
 */

#include <cub/warp/warp_reduce.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include <cstdio>

// includes, system
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utilities and system includes
#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>

namespace cg    = cooperative_groups;
namespace cudax = cuda::experimental;

#define ENABLE_CPU_DEBUG_CODE 0
constexpr int THREADS_PER_BLOCK = 512;

template <class Hierarchy>
double sum(cuda::this_warp<Hierarchy>, double v)
{
  return cub::WarpReduce::Sum(v);
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int* I, int* J, float* val, int N, int nz)
{
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = static_cast<float>(rand()) / RAND_MAX + 10.0f;
  val[1] = static_cast<float>(rand()) / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++)
  {
    if (i > 1)
    {
      I[i] = I[i - 1] + 3;
    }
    else
    {
      I[1] = 2;
    }

    start        = (i - 1) * 3 + 2;
    J[start]     = i - 1;
    J[start + 1] = i;

    if (i < N - 1)
    {
      J[start + 2] = i + 1;
    }

    val[start]     = val[start - 1];
    val[start + 1] = static_cast<float>(rand()) / RAND_MAX + 10.0f;

    if (i < N - 1)
    {
      val[start + 2] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  I[N] = nz;
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int* I, int* J, float* val, int nnz, int num_rows, float alpha, float* inputVecX, float* outputVecY)
{
  for (int i = 0; i < num_rows; i++)
  {
    int num_elems_this_row = I[i + 1] - I[i];

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++)
    {
      output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
    }
    outputVecY[i] = output;
  }

  return;
}

double dotProduct(float* vecA, float* vecB, int size)
{
  double result = 0.0;

  for (int i = 0; i < size; i++)
  {
    result = result + (vecA[i] * vecB[i]);
  }

  return result;
}

void scaleVector(float* vec, float alpha, int size)
{
  for (int i = 0; i < size; i++)
  {
    vec[i] = alpha * vec[i];
  }
}

void saxpy(float* x, float* y, float a, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = a * x[i] + y[i];
  }
}

void cpuConjugateGrad(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, int nnz, int N, float tol)
{
  int max_iter = 10000;

  float alpha   = 1.0;
  float alpham1 = -1.0;
  float r0      = 0.0, b, a, na;

  cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
  saxpy(Ax, r, alpham1, N);

  float r1 = dotProduct(r, r, N);

  int k = 1;

  while (r1 > tol * tol && k <= max_iter)
  {
    if (k > 1)
    {
      b = r1 / r0;
      scaleVector(p, b, N);

      saxpy(r, p, alpha, N);
    }
    else
    {
      for (int i = 0; i < N; i++)
      {
        p[i] = r[i];
      }
    }

    cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

    float dot = dotProduct(p, Ax, N);
    a         = r1 / dot;

    saxpy(p, x, a, N);
    na = -a;
    saxpy(Ax, r, na, N);

    r0 = r1;
    r1 = dotProduct(r, r, N);

    printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }
}

template <class Hierarchy>
__device__ void gpuSpMV(
  int* I,
  int* J,
  float* val,
  int nnz,
  int num_rows,
  float alpha,
  float* inputVecX,
  float* outputVecY,
  const cudax::this_grid<Hierarchy>& grid)
{
  for (auto i = cuda::gpu_thread.rank_as<int>(grid); i < num_rows; i += cuda::gpu_thread.count_as<int>(grid))
  {
    int row_elem           = I[i];
    int next_row_elem      = I[i + 1];
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++)
    {
      // I or J or val arrays - can be put in shared memory
      // as the access is random and reused in next calls of gpuSpMV function.
      output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
    }

    outputVecY[i] = output;
  }
}

template <class Hierarchy>
__device__ void gpuSaxpy(float* x, float* y, float a, int size, const cudax::this_grid<Hierarchy>& grid)
{
  for (auto i = cuda::gpu_thread.rank_as<int>(grid); i < size; i += cuda::gpu_thread.count_as<int>(grid))
  {
    y[i] = a * x[i] + y[i];
  }
}

template <class Hierarchy>
__device__ void gpuDotProduct(
  float* vecA,
  float* vecB,
  double* result,
  int size,
  const cudax::this_block<Hierarchy>& cta,
  const cudax::this_grid<Hierarchy>& grid)
{
  extern __shared__ double tmp[];

  double temp_sum = 0.0;
  for (auto i = cuda::gpu_thread.rank_as<int>(grid); i < size; i += cuda::gpu_thread.count_as<int>(grid))
  {
    temp_sum += static_cast<double>(vecA[i] * vecB[i]);
  }

  cudax::this_warp tile32{cta.hierarchy()};
  temp_sum = sum(tile32, temp_sum); // todo: replace with

  if (cuda::gpu_thread.rank(tile32) == 0)
  {
    tmp[tile32.rank(cuda::block)] = temp_sum;
  }

  cta.sync();

  if (tile32.rank(cuda::block) == 0)
  {
    temp_sum = (cuda::gpu_thread.rank(tile32) < tile32.count(cuda::block)) ? tmp[cuda::gpu_thread.rank(tile32)] : 0.0;
    temp_sum = sum(tile32, temp_sum);

    if (tile32.thread_rank() == 0)
    {
      atomicAdd(result, temp_sum);
    }
  }
}

template <class Hierarchy>
__device__ void gpuCopyVector(float* srcA, float* destB, int size, const cudax::this_grid<Hierarchy>& grid)
{
  for (auto i = cuda::gpu_thread.rank_as<int>(grid); i < size; i += cuda::gpu_thread.count_as<int>(grid))
  {
    destB[i] = srcA[i];
  }
}

template <class Hierarchy>
__device__ void gpuScaleVectorAndSaxpy(
  const float* x, float* y, float a, float scale, int size, const cudax::this_grid<Hierarchy>& grid)
{
  for (auto i = cuda::gpu_thread.rank_as<int>(grid); i < size; i += cuda::gpu_thread.count_as<int>(grid))
  {
    y[i] = a * x[i] + scale * y[i];
  }
}

struct ConjugateGradientKernel
{
  template <class Config>
  __device__ void operator()(
    const Config& config,
    int* I,
    int* J,
    float* val,
    float* x,
    float* Ax,
    float* p,
    float* r,
    double* dot_result,
    int nnz,
    int N,
    float tol) const noexcept
  {
    cudax::this_block cta{config};
    cudax::this_grid grid{config};

    int max_iter = 10000;

    float alpha   = 1.0;
    float alpham1 = -1.0;
    float r0      = 0.0, r1, b, a, na;

    gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, cta, grid);

    grid.sync();

    gpuSaxpy(Ax, r, alpham1, N, grid);

    grid.sync();

    gpuDotProduct(r, r, dot_result, N, cta, grid);

    grid.sync();

    r1 = *dot_result;

    int k = 1;
    while (r1 > tol * tol && k <= max_iter)
    {
      if (k > 1)
      {
        b = r1 / r0;
        gpuScaleVectorAndSaxpy(r, p, alpha, b, N, grid);
      }
      else
      {
        gpuCopyVector(r, p, N, grid);
      }

      grid.sync();

      gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, cta, grid);

      if (threadIdx.x == 0 && blockIdx.x == 0)
      {
        *dot_result = 0.0;
      }

      grid.sync();

      gpuDotProduct(p, Ax, dot_result, N, cta, grid);

      grid.sync();

      a = r1 / *dot_result;

      gpuSaxpy(p, x, a, N, grid);
      na = -a;
      gpuSaxpy(Ax, r, na, N, grid);

      r0 = r1;

      grid.sync();
      if (threadIdx.x == 0 && blockIdx.x == 0)
      {
        *dot_result = 0.0;
      }

      grid.sync();

      gpuDotProduct(r, r, dot_result, N, cta, grid);

      grid.sync();

      r1 = *dot_result;
      k++;
    }
  }
};

bool areAlmostEqual(float a, float b, float maxRelDiff)
{
  float diff    = cuda::std::fabs(a - b);
  float abs_a   = cuda::std::fabs(a);
  float abs_b   = cuda::std::fabs(b);
  float largest = abs_a > abs_b ? abs_a : abs_b;

  if (diff <= largest * maxRelDiff)
  {
    return true;
  }
  else
  {
    std::printf("maxRelDiff = %.8e\n", maxRelDiff);
    std::printf("diff %.8e > largest * maxRelDiff %.8e therefore %.8e and %.8e are not "
                "same\n",
                diff,
                largest * maxRelDiff,
                a,
                b);
    return false;
  }
}

int main(int argc, char** argv)
{
  if (cuda::devices.size() == 0)
  {
    std::fputs("error: no CUDA device found", stderr);
    return EXIT_FAILURE;
  }

  const auto device = cuda::devices[0];

  // This sample requires being run on a device that supports Unified Memory.
  if (!cuda::device_attributes::managed_memory(device))
  {
    std::fputs("error: unified Memory not supported on this device", stderr);
    return EXIT_FAILURE;
  }

  // This sample requires being run on a device that supports Cooperative Kernel Launch.
  if (!cuda::device_attributes::cooperative_launch(device))
  {
    std::fputs("error: cooperative kernel launch is not supported on this device", stderr);
    return EXIT_FAILURE;
  }

  const auto multiprocessor_count = cuda::device_attributes::multiprocessor_count(device);

  // Statistics about the GPU device.
  std::printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
              multiprocessor_count,
              cuda::device_attributes::compute_capability_major(device),
              cuda::device_attributes::compute_capability_minor(device));

  cuda::stream stream{device};

  // Generate a random tridiagonal symmetric matrix in CSR format.
  constexpr int N     = 1048576;
  constexpr int nz    = (N - 2) * 3 + 4;
  constexpr float tol = 1e-5f;

  auto I   = cuda::make_managed_buffer<int>(stream, N + 1, cuda::no_init);
  auto J   = cuda::make_managed_buffer<int>(stream, nz, cuda::no_init);
  auto val = cuda::make_managed_buffer<float>(stream, nz, cuda::no_init);
  stream.sync();

  genTridiag(I.data(), J.data(), val.data(), N, nz);

  auto x          = cuda::make_managed_buffer<float>(stream, N, 0.f);
  auto rhs        = cuda::make_managed_buffer<float>(stream, N, 1.f);
  auto dot_result = cuda::make_managed_buffer<double>(stream, 1, 0.0);

  // Temp memory for CG.
  auto r  = cuda::make_managed_buffer<float>(stream, N, 1.f);
  auto p  = cuda::make_managed_buffer<float>(stream, N, cuda::no_init);
  auto Ax = cuda::make_managed_buffer<float>(stream, N, cuda::no_init);
  stream.sync();

  cuda::timed_event start{device};
  cuda::timed_event stop{device};

  const auto config = cuda::make_config(
    cuda::grid_dims(multiprocessor_count), cuda::block_dims<THREADS_PER_BLOCK>(), cuda::cooperative_launch{});

  start.record(stream);
  cuda::launch(
    stream,
    config,
    ConjugateGradientKernel{},
    I.data(),
    J.data(),
    val.data(),
    x.data(),
    Ax.data(),
    p.data(),
    r.data(),
    dot_result.data(),
    nz,
    N,
    tol);

#if ENABLE_CPU_DEBUG_CODE
  float* Ax_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * N));
  float* r_cpu  = reinterpret_cast<float*>(malloc(sizeof(float) * N));
  float* p_cpu  = reinterpret_cast<float*>(malloc(sizeof(float) * N));
  float* x_cpu  = reinterpret_cast<float*>(malloc(sizeof(float) * N));

  for (int i = 0; i < N; i++)
  {
    r_cpu[i]  = 1.0;
    Ax_cpu[i] = x_cpu[i] = 0.0;
  }
#endif

  float* x;
  float* rhs;
  float r1;
  float *r, *p, *Ax;
  cudaEvent_t start, stop;

  void* kernelArgs[] = {
    (void*) &I,
    (void*) &J,
    (void*) &val,
    (void*) &x,
    (void*) &Ax,
    (void*) &p,
    (void*) &r,
    (void*) &dot_result,
    (void*) &nz,
    (void*) &N,
    (void*) &tol,
  };

  int sMemSize       = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
  int numBlocksPerSm = 0;
  int numThreads     = THREADS_PER_BLOCK;

  checkCudaErrors(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, gpuConjugateGradient, numThreads, sMemSize));

  dim3 dimGrid(multiprocessor_count * numBlocksPerSm, 1, 1), dimBlock(THREADS_PER_BLOCK, 1, 1);
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(
    cudaLaunchCooperativeKernel((void*) gpuConjugateGradient, dimGrid, dimBlock, kernelArgs, sMemSize, NULL));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

  r1 = *dot_result;

  printf("GPU Final, residual = %e, kernel execution time = %f ms\n", sqrt(r1), time);

#if ENABLE_CPU_DEBUG_CODE
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++)
  {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++)
    {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err)
    {
      err = diff;
    }
  }

  checkCudaErrors(cudaFree(I));
  checkCudaErrors(cudaFree(J));
  checkCudaErrors(cudaFree(val));
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(rhs));
  checkCudaErrors(cudaFree(r));
  checkCudaErrors(cudaFree(p));
  checkCudaErrors(cudaFree(Ax));
  checkCudaErrors(cudaFree(dot_result));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

#if ENABLE_CPU_DEBUG_CODE
  free(Ax_cpu);
  free(r_cpu);
  free(p_cpu);
  free(x_cpu);
#endif

  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiBlockCG %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");
  exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}
