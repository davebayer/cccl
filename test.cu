#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda.h>

namespace cuda
{

stream_ref __thread_private_stream{reinterpret_cast<cudaStream_t>(0x2ull)};

class mempool_ref
{
  ::CUmemoryPool __pool_;

public:
  void* allocate(stream_ref __stream, size_t __nbytes, size_t __alignment)
  {
    // check alignment
    CUdeviceptr __ret{};
    cuMemAllocFromPoolAsync(&__ret, __nbytes, __pool_, __stream.get());
    return reinterpret_cast<void*>(__ret);
  }

  void* allocate_sync(size_t __nbytes, size_t __alignment)
  {
    void* __ret = allocate(__thread_private_stream, __nbytes, __alignment);
    __thread_private_stream.sync();
    return __ret;
  }

  void* deallocate(stream_ref __stream, void* __ptr, size_t __nbytes)
  {
    cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(__ptr), __stream.get());
  }

  void* deallocate_sync(void* __ptr, size_t __nbytes)
  {
    deallocate(__thread_private_stream, __ptr, __nbytes);
    __thread_private_stream.sync();
  }
};

template <class T>
class buffer
{
  stream_ref __stream_{};
  mempool_ref __mempool_{};
  void* __data_{};
  size_t __size_{};

public:
  buffer() = default;

  buffer(mempool_ref __pool, size_t __n)
      : __stream_{CU_STREAM_PER_THREAD}
      , __mempool_{__pool}
      , __data_{__pool.allocate_sync(__n * sizeof(T), alignof(T))}
      , __size_{__n * sizeof(T)}
  {}

  buffer(stream_ref __stream, mempool_ref __pool, size_t __n)
      : __mempool_{__pool}
      , __data_{__pool.allocate(__stream, __n * sizeof(T), alignof(T))}
      , __size_{__n * sizeof(T)}
  {}

  explicit buffer(const buffer& __other)
      : __mempool_{__other.__pool_}
      , __data_{__mempool_.allocate(__thread_private_stream, __other.__size_, alignof(T))}
      , __size_{__other.__size_}
  {
    // if T is trivially copyable
    cuMemcpyAsync(__data_, __other.__data_, __size_, __thread_private_stream);
    __thread_private_stream.sync();
  }

  buffer(stream_ref __stream, const buffer& __other)
      : __mempool_{__other.__pool_}
      , __data_{__mempool_.allocate(__stream, __other.__size_, alignof(T))}
      , __size_{__other.__size_}
  {
    // if T is trivially copyable
    cuMemcpyAsync(__data_, __other.__data_, __size_, __stream.get());
  }

  buffer(buffer&& __other)
      : __mempool_{::cuda::std::exchange(__other.__pool_, nullptr)}
      , __data_{::cuda::std::exchange(__other.__data_, nullptr)}
      , __size_{::cuda::std::exchange(__other.__size_, 0)}
  {}

  ~buffer()
  {
    __mempool_.deallocate_sync(__data_, __size_);
  }

  buffer& operator=(const buffer&) = delete;

  buffer& operator=(buffer&& __other)
  {
    if (this != ::cuda::std::addressof(__other))
    {
      destroy(__thread_private_stream);
      __thread_private_stream.sync();

      __mempool_ = __other.__mempool_;
      __data_    = __other.__data_;
      __size_    = __other.__size_;
    }
    return *this;
  }

  T* data() const
  {
    return reinterpret_cast<T*>(__data_);
  }

  size_t size() const
  {
    return __size_;
  }

  void destroy(stream_ref __stream)
  {
    __mempool_.deallocate(__stream);
    __mempool_ = nullptr;
    __data_    = nullptr;
    __size_    = 0;
  }
};

} // namespace cuda

void do_some_stuff(...);

int main()
{
  cuda::mempool_ref mempool{};
  cuda::stream_ref stream{};

  cuda::buffer<float> buffer{mempool, 256};
  cuda::buffer<double> buffer2{stream, mempool, 200};

  cuda::buffer<float> buffer3{buffer}; // synchronous copy
  cuda::buffer<float> buffer4{stream, buffer3};

  cuda::buffer<float> buffer5{cuda::std::move(buffer)};

  cuda::buffer<float> buffer6;
  buffer6 = cuda::buffer<float>{buffer};
  buffer6 = cuda::std::move(buffer);

  {
    cuda::buffer<float> async_buffer{stream, mempool, 235};

    do_some_stuff(stream, async_buffer.data());

    async_buffer.destroy(stream);
  }
}
