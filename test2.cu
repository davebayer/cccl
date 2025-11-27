__device__ void fn(void*);

__global__ void kernel()
{
  __shared__ int shm[10];
  fn(shm);
}
