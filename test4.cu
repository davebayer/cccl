__device__ void fn(void*);

template <class T>
struct static_shared_memory
{
  void* ptr_;
};

template <class T>
__device__ auto make_static_shared_object()
{
  __shared__ T value;
  return static_shared_memory<T>{&value};
}

__global__ void kernel()
{
  auto shm = make_static_shared_object<int[10]>();
  fn(shm.ptr_);
}
