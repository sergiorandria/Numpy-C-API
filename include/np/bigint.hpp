/**
 * @file bigint.hpp
 * @brief Arbitrary-precision integer support (np::bigint) with GMP mpz backend.
 *
 * Provides `np::bigint` as a header-only wrapper over
 * `boost::multiprecision::cpp_int` (default) and `boost::multiprecision::mpz_int`
 * (when GMP is available). Designed for `np::ndarray<bigint>`:
 *   - creation / manipulation / math / linalg exact integer paths
 *   - `dtype::bigint` enumeration + `dtype_of<bigint>` == bigint
 *   - `to_mpz` / `from_mpz` helpers for `mpz_t`
 *   - constexpr auto-promotion via `auto_bigint_t<T>` / `as_bigint()`
 *
 * Usage:
 *   #include <np/np.hpp>          // includes bigint.hpp automatically
 *   #include <np/bigint.hpp>
 *   np::ndarray<np::bigint> a = {np::bigint("12345678901234567890"), 42};
 *   auto b = np::as_bigint(a);    // ndarray<int> -> ndarray<bigint>
 *   auto c = np::linalg::det(a);  // exact bigint determinant
 *   mpz_t z; mpz_init_set_str(z, "999...", 10);
 *   np::bigint bi = np::from_mpz(z);
 *   np::to_mpz(bi, z);
 *
 * Reference:
 * https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/primetest.html
 *            https://gmplib.org/manual/Integer-Functions
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_BIGINT_HPP
#define NP_BIGINT_HPP

#include <string>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"
#include "dtype.hpp"

// ── Boost.Multiprecision backend ────────────────────────────────────────
#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
#define NP_HAS_CPP_INT 1
#else
#define NP_HAS_CPP_INT 0
#endif

#if defined(NP_HAS_GMP) && __has_include(<boost/multiprecision/gmp.hpp>)
#include <boost/multiprecision/gmp.hpp>
#define NP_HAS_GMP_MPZ_INT 1
#else
#define NP_HAS_GMP_MPZ_INT 0
#endif

#if defined(NP_HAS_GMP) && __has_include(<gmp.h>)
#include <gmp.h>
// gmp.h defines NZERO macro that conflicts with np::constants::NZERO
#ifdef NZERO
#undef NZERO
#endif
#define NP_HAS_GMP_H 1
#else
#define NP_HAS_GMP_H 0
#endif

namespace np
{

#if NP_HAS_CPP_INT
  using bigint = boost::multiprecision::cpp_int;
#if NP_HAS_GMP_MPZ_INT
  using mpz_bigint = boost::multiprecision::mpz_int;
#else
  using mpz_bigint = bigint;
#endif
#else
  // Minimal fallback when Boost not available
  struct bigint
  {
    std::string value = "0";
    bigint() = default;
    bigint(long long v) : value(std::to_string(v))
    {
    }
    bigint(const std::string& s) : value(s)
    {
    }
    bigint(const char* s) : value(s)
    {
    }
    bool operator==(const bigint& o) const
    {
      return value == o.value;
    }
    bool operator<(const bigint& o) const
    {
      return value < o.value;
    }
    bool operator>(const bigint& o) const
    {
      return o < *this;
    }
  };
  using mpz_bigint = bigint;
#endif

  namespace detail
  {
    template <typename T>
    struct is_bigint : std::false_type
    {
    };
    template <>
    struct is_bigint<bigint> : std::true_type
    {
    };
#if NP_HAS_GMP_MPZ_INT
    template <>
    struct is_bigint<mpz_bigint> : std::true_type
    {
    };
#endif
    template <typename T>
    inline constexpr bool is_bigint_v = is_bigint<std::remove_cv_t<T>>::value;

    template <typename A, typename B>
    struct common_bigint
    {
      using type = std::conditional_t<
          is_bigint_v<A> || is_bigint_v<B>,
          bigint,
          std::common_type_t<A, B>>;
    };
    template <typename A, typename B>
    using common_bigint_t = typename common_bigint<A, B>::type;

  } // namespace detail

  // ——— bigint arithmetic (ADL-visible in np) ———
  inline bigint operator+(const bigint& a, const bigint& b)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r += b;
    return r;
#else
    return bigint(std::to_string(std::stoll(a.value) + std::stoll(b.value)));
#endif
  }
  inline bigint operator-(const bigint& a, const bigint& b)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r -= b;
    return r;
#else
    return bigint(std::to_string(std::stoll(a.value) - std::stoll(b.value)));
#endif
  }
  inline bigint operator*(const bigint& a, const bigint& b)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r *= b;
    return r;
#else
    return bigint(std::to_string(std::stoll(a.value) * std::stoll(b.value)));
#endif
  }
  inline bigint operator/(const bigint& a, const bigint& b)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r /= b;
    return r;
#else
    return bigint(std::to_string(std::stoll(a.value) / std::stoll(b.value)));
#endif
  }
  inline bigint operator%(const bigint& a, const bigint& b)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r %= b;
    return r;
#else
    return bigint(std::to_string(std::stoll(a.value) % std::stoll(b.value)));
#endif
  }
  inline bigint operator-(const bigint& a)
  {
#if NP_HAS_CPP_INT
    bigint r = a;
    r = -r;
    return r;
#else
    return bigint(std::to_string(-std::stoll(a.value)));
#endif
  }
  inline bigint operator+(const bigint& a)
  {
    return a;
  }

  /**
   * @brief Constexpr auto-promotion to bigint.
   *
   * Any integral type (except bool) maps to `np::bigint`; floating / complex / bigint
   * stay. `np::auto_bigint_t<int>` == `np::bigint`.
   */
  template <typename T>
  using auto_bigint_t = std::conditional_t<
      detail::is_bigint_v<T>,
      T,
      std::conditional_t<
          std::is_integral_v<std::remove_cv_t<T>>
              && !std::is_same_v<std::remove_cv_t<T>, bool>,
          bigint,
          T>>;

  template <typename T>
  inline constexpr bool auto_promotes_to_bigint_v =
      std::is_same_v<auto_bigint_t<T>, bigint> && !detail::is_bigint_v<T>;

  // Converters
  NP_NODISCARD inline bigint to_bigint(long long v)
  {
    return bigint(v);
  }
  NP_NODISCARD inline bigint to_bigint(const std::string& s)
  {
    return bigint(s);
  }
  NP_NODISCARD inline bigint to_bigint(const char* s)
  {
    return bigint(s);
  }
  NP_NODISCARD inline bigint to_bigint(const bigint& v)
  {
    return v;
  }
  template <typename T>
    requires(std::is_arithmetic_v<T> && !detail::is_bigint_v<T>)
  NP_NODISCARD inline bigint to_bigint(T v)
  {
    return bigint(v);
  }

#if NP_HAS_GMP_H
  NP_NODISCARD inline bigint from_mpz(const mpz_t z)
  {
#if NP_HAS_GMP_MPZ_INT
    mpz_bigint tmp(z);
    return bigint(tmp);
#else
#if NP_HAS_CPP_INT
    char* s = mpz_get_str(nullptr, 10, z);
    bigint out(s);
    ::free(s);
    return out;
#else
    char* s = mpz_get_str(nullptr, 10, z);
    bigint out(s);
    ::free(s);
    return out;
#endif
#endif
  }

  inline void to_mpz(const bigint& bi, mpz_t rop)
  {
#if NP_HAS_GMP_MPZ_INT
    mpz_bigint tmp(bi);
    mpz_set(rop, tmp.backend().data());
#else
#if NP_HAS_CPP_INT
    std::string s = bi.convert_to<std::string>();
    mpz_set_str(rop, s.c_str(), 10);
#else
    mpz_set_str(rop, bi.value.c_str(), 10);
#endif
#endif
  }

  NP_NODISCARD inline mpz_bigint to_mpz_bigint(const bigint& bi)
  {
    return mpz_bigint(bi);
  }
#endif // NP_HAS_GMP_H

  inline namespace literals
  {
    NP_NODISCARD inline bigint operator""_bigint(const char* str, std::size_t len)
    {
      return bigint(std::string(str, len));
    }
    NP_NODISCARD inline bigint operator""_mpz(const char* str, std::size_t len)
    {
      return bigint(std::string(str, len));
    }
  } // namespace literals

  //  Ergonomic helpers

  /**
   * @brief Create `bigint` from any string/arithmetic literal (constexpr where possible).
   * `auto b = np::make_bigint("12345678901234567890");`
   */
  template <typename T>
  NP_NODISCARD inline auto make_bigint(T&& v) -> bigint
  {
    if constexpr (detail::is_bigint_v<std::remove_cv_t<T>>)
      return bigint(std::forward<T>(v));
    else if constexpr (
        std::is_same_v<std::remove_cv_t<T>, const char*>
        || std::is_same_v<std::remove_cv_t<T>, char*>)
      return bigint(std::string(v));
    else
      return bigint(std::forward<T>(v));
  }

  /**
   * @brief `constexpr` promote any integral `ndarray` to `ndarray<bigint>`.
   * `auto b = np::promote_to_bigint(a); // ndarray<int> -> ndarray<bigint> if needed`
   * If already `bigint`, returns copy.
   */
  template <typename T>
  NP_NODISCARD inline auto promote_to_bigint(const ndarray<T>& a)
  {
    if constexpr (detail::is_bigint_v<T>)
      return a.copy();
    else
      return as_bigint(a);
  }

  /**
   * @brief Create `ndarray<bigint>` from initializer list of ints/strings.
   * `auto a = np::make_bigint_array({1, 2, 3});`
   * `auto b = np::make_bigint_array({"123", "456"});`
   */
  template <typename T>
  NP_NODISCARD inline auto make_bigint_array(std::initializer_list<T> list)
      -> ndarray<bigint>
  {
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto& v : list)
      data.emplace_back(make_bigint(v));
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (size_t i = 0; i < data.size(); ++i)
      out[i] = data[i];
    return out;
  }

} // namespace np

// dtype integration: map bigint -> dtype::bigint
namespace np::detail
{
  template <>
  struct cxx_to_np_type_impl<np::bigint>
  {
    static constexpr np::dtype value = np::dtype::bigint;
  };
#if NP_HAS_GMP_MPZ_INT
  template <>
  struct cxx_to_np_type_impl<np::mpz_bigint>
  {
    static constexpr np::dtype value = np::dtype::bigint;
  };
#endif
} // namespace np::detail

// std::common_type
namespace std
{
#if NP_HAS_CPP_INT
  template <>
  struct common_type<np::bigint, np::bigint>
  {
    using type = np::bigint;
  };
  template <typename T>
  struct common_type<T, np::bigint>
  {
    using type = np::bigint;
  };
  template <typename T>
  struct common_type<np::bigint, T>
  {
    using type = np::bigint;
  };
#endif
} // namespace std

//  ndarray converters (need ndarray definition)
#include "ndarray.hpp"
namespace np
{
  /**
   * @brief Convert `ndarray<T>` → `ndarray<bigint>` (exact).
   * `ndarray<int>{1,2} -> ndarray<bigint>{1,2}`
   */
  template <typename T>
    requires(!detail::is_bigint_v<T>)
  NP_NODISCARD inline auto as_bigint(const ndarray<T>& a) -> ndarray<bigint>
  {
    ndarray<bigint> out(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      out.data()[i] = bigint(a.data()[a._flat_logical(i)]);
    }
    return out;
  }

  // Identity overload
  NP_NODISCARD inline auto as_bigint(const ndarray<bigint>& a) -> ndarray<bigint>
  {
    return a.copy();
  }

  /**
   * @brief Convert `ndarray<bigint>` → `ndarray<T>` with truncation.
   * For `T` integral, uses `convert_to<T>()` which throws on overflow for cpp_int.
   */
  template <typename T>
  NP_NODISCARD inline auto from_bigint(const ndarray<bigint>& a) -> ndarray<T>
  {
    ndarray<T> out(a.shape);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      const bigint& bi = a.data()[a._flat_logical(i)];
#if NP_HAS_CPP_INT
      if constexpr (std::is_integral_v<T>)
        out.data()[i] = bi.convert_to<T>();
      else
        out.data()[i] = static_cast<T>(bi.convert_to<long double>());
#else
      out.data()[i] = static_cast<T>(std::stoll(bi.value));
#endif
    }
    return out;
  }

  /**
   * @brief Create ndarray<bigint> from string literals (convenience).
   * `np::bigints({"123456...", "999..."})`
   */
  NP_NODISCARD inline auto bigints(std::initializer_list<std::string> list)
      -> ndarray<bigint>
  {
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto& s : list)
      data.emplace_back(s.c_str());
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
      out[i] = data[i];
    return out;
  }
  NP_NODISCARD inline auto bigints(std::initializer_list<const char*> list)
      -> ndarray<bigint>
  {
    std::vector<bigint> data;
    data.reserve(list.size());
    for (auto s : list)
      data.emplace_back(s);
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
      out[i] = data[i];
    return out;
  }
  // Overload for already-bigint list
  NP_NODISCARD inline auto bigints(std::initializer_list<bigint> list) -> ndarray<bigint>
  {
    std::vector<bigint> data(list.begin(), list.end());
    ndarray<bigint> out(std::vector<int>{static_cast<int>(data.size())});
    for (std::size_t i = 0; i < data.size(); ++i)
      out[i] = data[i];
    return out;
  }

} // namespace np

#endif // NP_BIGINT_HPP
