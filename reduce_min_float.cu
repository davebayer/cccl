#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

inline constexpr auto full_warp_mask = ~0u;

__device__ float reduce_min_float(float v)
{
  const auto is_nan = cuda::std::isnan(v);
  const auto is_neg = cuda::std::signbit(v);

  const auto is_nan_mask = __ballot_sync(full_warp_mask, is_nan);
  const auto is_neg_mask = __ballot_sync(full_warp_mask, is_neg && !is_nan);

  cuda::std::uint32_t word;
  if (!is_nan && is_neg && is_neg_mask != 0)
  {
    word = cuda::std::bit_cast<cuda::std::uint32_t>(v);
  }
  else
  {
    word = cuda::std::bit_cast<cuda::std::uint32_t>(cuda::std::numeric_limits<float>::infinity());
  }

  auto word = cuda::std::bit_cast<cuda::std::uint32_t>(v);
  if (cuda::std::isnan(v))
  {
    word = cuda::std::bit_cast<cuda::std::uint32_t>(cuda::std::numeric_limits<float>::infinity());
  }

  auto result = __reduce_min_sync(full_warp_mask, static_cast<int>(word));
  if (is_nan_mask == full_warp_mask)
  {
    result = cuda::std::numeric_limits<float>::quiet_NaN();
  }
  return result;
}
