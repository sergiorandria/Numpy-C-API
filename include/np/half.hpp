/**
 * @file half.hpp
 * @brief FP16 / BF16 half-precision for powerful GPU tensor cores.
 *
 * Provides np::half (float16) and np::bfloat16 wrappers with conversion to/from float.
 * Uses _Float16 on GCC/Clang (AVX512-FP16, ARMv8.2) or std::float16_t if C++23,
 * otherwise emulates via float. Header-only, for Hopper/Blackwell FP16 tensor cores.
 */
#ifndef NP_HALF_HPP
#define NP_HALF_HPP

#include "api_macros.hpp"
#include <cstdint>
#include <cstring>

namespace np
{

#if defined(__FLT16_MAX__) || defined(__HAVE_FLOAT16)
  using float16 = _Float16;
#define NP_HAS_FLOAT16 1
#elif __has_include(<stdfloat>)
#include <stdfloat>
#if defined(__STDCPP_FLOAT16_T__)
  using float16 = std::float16_t;
#define NP_HAS_FLOAT16 1
#endif
#endif

#ifndef NP_HAS_FLOAT16
  // Emulated half via float (fallback for CI without FP16 HW)
  struct float16
  {
    uint16_t bits = 0;
    float16() = default;
    explicit float16(float f)
    {
      uint32_t u;
      std::memcpy(&u, &f, sizeof(float));
      // Simple truncation: high 16 bits
      bits = static_cast<uint16_t>(u >> 16);
    }
    operator float() const noexcept
    {
      uint32_t u = static_cast<uint32_t>(bits) << 16;
      float f;
      std::memcpy(&f, &u, sizeof(float));
      return f;
    }
  };
#define NP_HAS_FLOAT16 1
#endif

  struct bfloat16
  {
    uint16_t bits = 0;
    bfloat16() = default;
    explicit bfloat16(float f)
    {
      uint32_t u;
      std::memcpy(&u, &f, sizeof(float));
      bits = static_cast<uint16_t>(u >> 16);
    }
    operator float() const noexcept
    {
      uint32_t u = static_cast<uint32_t>(bits) << 16;
      float f;
      std::memcpy(&f, &u, sizeof(float));
      return f;
    }
  };

  // Traits
  template <typename T>
  struct is_half : std::false_type
  {
  };
  template <>
  struct is_half<float16> : std::true_type
  {
  };
  template <>
  struct is_half<bfloat16> : std::true_type
  {
  };
  template <typename T>
  constexpr bool is_half_v = is_half<T>::value;

} // namespace np

#endif // NP_HALF_HPP
