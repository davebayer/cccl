.. _libcudacxx-extended-api-math-clip:

``cuda::clip``
==============

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   T clip(T v, T lo, T hi) noexcept;

   } // namespace cuda

``cuda::clip`` clamps a numeric value to the inclusive range ``[lo, hi]``.
It takes its arguments and returns its result by value, so the result does not
refer to any of the arguments.

**Parameters**

- ``v``: The input value.
- ``lo``: The lower bound.
- ``hi``: The upper bound.

**Return value**

For integer types, returns ``lo`` if ``v < lo``, ``hi`` if ``hi < v``, otherwise
``v``. Integer values retain their type and precision.

For floating-point types, computes ``cuda::std::fmin(hi, cuda::std::fmax(lo, v))``.
A NaN input value returns ``lo``. The sign of a zero result follows the underlying
``fmin`` and ``fmax`` operations; an input zero's sign is not guaranteed to be preserved.

**Constraints**

- ``T`` must be an integer type or a supported floating-point type, including
  ``float``, ``double``, ``long double``, ``__half``, ``__nv_bfloat16``, and
  ``__float128``, when available. Other CUDA floating-point types are not supported.
- All three arguments must have the same type when deducing ``T``.

**Preconditions**

- ``lo <= hi`` must hold. Floating-point bounds must not be NaN.

Example
-------

.. code:: cuda

   #include <cuda/cmath>
   #include <cuda/std/cassert>

   static_assert(cuda::clip(5, 0, 3) == 3);

   void example()
   {
       assert(cuda::clip(-1.0f, 0.0f, 1.0f) == 0.0f);
       assert(cuda::clip(0.5f, 0.0f, 1.0f) == 0.5f);
   }
