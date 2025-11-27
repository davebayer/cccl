__device__ void fn(void*);

template <class T>
struct static_shared_memory
{
  __device__ __forceinline__ static_shared_memory()
  {
    __shared__ T value;
    ptr_ = &value;
  }

  void* ptr_;
};

__global__ void kernel()
{
  static_shared_memory<int[10]> shm;
  fn(shm.ptr_);
}
