#include <cuda/functional>

#define IS_D_LAMBDA(X)    cuda::is_device_lambda_v<X>
#define IS_DPRT_LAMBDA(X) __nv_is_extended_device_lambda_with_preserved_return_type(X)
#define IS_HD_LAMBDA(X)   cuda::is_host_device_lambda_v<X>

auto lam0 = [] __host__ __device__ {};

void foo(void)
{
  auto lam1 = [] {};
  auto lam2 = [] __device__ {};
  auto lam3 = [] __host__ __device__ {};
  auto lam4 = [] __device__() -> double {
    return 3.14;
  };
  auto lam5 = [] __device__(int x) -> decltype(&x) {
    return 0;
  };

  // lam0 is not an extended lambda (since defined outside function scope)
  static_assert(!IS_D_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam0)), "");

  // lam1 is not an extended lambda (since no execution space annotations)
  static_assert(!IS_D_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam1)), "");

  // lam2 is an extended __device__ lambda
  static_assert(IS_D_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam2)), "");

  // lam3 is an extended __host__ __device__ lambda
  static_assert(!IS_D_LAMBDA(decltype(lam3)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam3)), "");
  static_assert(IS_HD_LAMBDA(decltype(lam3)), "");

  // lam4 is an extended __device__ lambda with preserved return type
  static_assert(IS_D_LAMBDA(decltype(lam4)), "");
  static_assert(IS_DPRT_LAMBDA(decltype(lam4)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam4)), "");

  // lam5 is not an extended __device__ lambda with preserved return type
  // because it references the operator()'s parameter types in the trailing return type.
  static_assert(IS_D_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam5)), "");
}
