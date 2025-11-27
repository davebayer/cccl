#include <cuda/std/__charconv_>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/utility>

__device__ void fn(void*);

template <class _Tp, ::cuda::std::size_t _Alignment = alignof(_Tp)>
class [[nodiscard]] static_shared_memory
{
  unsigned __ptr_{};

  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto __make_str(::cuda::std::size_t __v) noexcept
  {
    ::cuda::std::array<char, 1 /*::cuda::std::numeric_limits<::cuda::std::size_t>::digits10*/> __buff{};
    auto __fmt_result = ::cuda::std::to_chars(__buff.begin(), __buff.end(), __v);
    _CCCL_VERIFY(__fmt_result.ec == cuda::std::errc{}, "Failed to format");
    return __buff;
  }

  template <const auto& _AlignmentStr,
            const auto& _SizeStr,
            ::cuda::std::size_t... _AlignIdcs,
            ::cuda::std::size_t... _SizeIdcs>
  _CCCL_DEVICE_API static _CCCL_FORCEINLINE unsigned
    __define_smem_ptx_variable(::cuda::std::index_sequence<_AlignIdcs...>, ::cuda::std::index_sequence<_SizeIdcs...>)
  {
    static constexpr char __alignment_array[]{_AlignmentStr[_AlignIdcs]..., '\0'};
    static constexpr char __size_array[]{_SizeStr[_SizeIdcs]..., '\0'};

    unsigned __ret{};
    asm volatile(
      "{\n\t"
      ".shared .align %1 .b8 shm_var[%2];\n\t"
      "mov.u32 	%0, shm_var;\n\t"
      "}"
      : "=r"(__ret)
      : "C"(__alignment_array), "C"(__size_array));
    return __ret;
  }

public:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static_shared_memory() noexcept
  {
    static constexpr auto __alignment_str = __make_str(sizeof(_Tp));
    static constexpr auto __size_str      = __make_str(_Alignment);
    __ptr_                                = __define_smem_ptx_variable<__alignment_str, __size_str>(
      cuda::std::make_index_sequence<__alignment_str.size()>{}, cuda::std::make_index_sequence<__size_str.size()>{});
  }

  static_shared_memory(const static_shared_memory&)            = delete;
  static_shared_memory(static_shared_memory&&)                 = delete;
  static_shared_memory& operator=(const static_shared_memory&) = delete;
  static_shared_memory& operator=(static_shared_memory&&)      = delete;

  [[nodiscard]] _CCCL_DEVICE_API _Tp* get() const noexcept
  {
    void* __ptr = (void*) static_cast<unsigned long long>(__ptr_);
    __builtin_assume(__isShared(__ptr));
    return (_Tp*) __ptr;
  }
};

__global__ void kernel()
{
  static_shared_memory<int> shm;
  // static_shared_memory<float[10]> shm2;
  fn(shm.get());
  // fn(shm2.get());
}
