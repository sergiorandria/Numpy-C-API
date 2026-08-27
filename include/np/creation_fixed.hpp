/**
 * @file creation_fixed.hpp
 * @brief Compile-time-first array creation for the fixed-shape path.
 *
 * The NumPy shape argument becomes the template parameter list, so the
 * extent checks are static:
 *   np::zeros<2, 3>()          -> ndarrayf<double, 2, 3>
 *   np::ones<4, int>()         -> ndarrayf<int, 4>
 *   np::full<2, 2>(7)          -> ndarrayf<int, 2, 2>
 *   np::eye<3>()               -> ndarrayf<double, 3, 3>
 *   np::eye<3, 4, 1>()         -> ndarrayf<double, 3, 4> with k = 1
 *   np::identity<3, int>()     -> ndarrayf<int, 3, 3>
 *   np::arange<6>(1, 7, 2)     -> {1, 3, 5, 7, 9, 11}
 *   np::linspace<5>(0.0, 1.0)  -> {0.0, 0.25, 0.5, 0.75, 1.0}
 *
 * All functions return C-contiguous arrays. The fixed path uses
 * value semantics (no shared storage, no views).
 *
 * Signature ground truth: numpy-reference/reference/generated/
 *   numpy.zeros.html, numpy.ones.html, numpy.full.html, numpy.eye.html,
 *   numpy.identity.html, numpy.arange.html, numpy.linspace.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CREATION_FIXED_HPP
#define NP_CREATION_FIXED_HPP

#include <cstddef>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray_fixed.hpp"

namespace np
{

  /* @brief Array of zeros with the given compile-time shape. Two spellings:
   *        np::zeros<2, 3>() (double) or np::zeros<int, 2, 3>().
   *
   * @tparam E  Compile-time extents (must all be > 0).
   * @return     ndarrayf<double, E...> filled with 0.0.
   *
   * Reference: numpy-reference/reference/generated/numpy.zeros.html
   */
  NP_API template <int... E>
  NP_NODISCARD constexpr ndarrayf<double, E...> zeros()
  {
    return ndarrayf<double, E...>{};
  }

  /* @brief Array of zeros with compile-time shape and element type.
   *
   * @tparam T  Element type.
   * @tparam E  Compile-time extents (must all be > 0).
   * @return     ndarrayf<T, E...> filled with T{0}.
   */
  NP_API template <typename T, int... E>
  NP_NODISCARD constexpr ndarrayf<T, E...> zeros()
  {
    return ndarrayf<T, E...>{};
  }

  /* @brief Array of ones with the given compile-time shape.
   *
   * @tparam E  Compile-time extents (must all be > 0).
   * @return     ndarrayf<double, E...> filled with 1.0.
   *
   * Reference: numpy-reference/reference/generated/numpy.ones.html
   */
  NP_API template <int... E>
  NP_NODISCARD constexpr ndarrayf<double, E...> ones()
  {
    ndarrayf<double, E...> out{};
    out.fill(1.0);
    return out;
  }

  /* @brief Array of ones with compile-time shape and element type.
   *
   * @tparam T  Element type.
   * @tparam E  Compile-time extents (must all be > 0).
   * @return     ndarrayf<T, E...> filled with T{1}.
   */
  NP_API template <typename T, int... E>
  NP_NODISCARD constexpr ndarrayf<T, E...> ones()
  {
    ndarrayf<T, E...> out{};
    out.fill(T{1});
    return out;
  }

  /* @brief Array filled with a constant value.
   *
   * @tparam E  Compile-time extents (must all be > 0).
   * @tparam T  Element type (deduced from fill_value).
   * @param  fill_value  Value to fill every element with.
   * @return             ndarrayf<T, E...> filled with fill_value.
   *
   * Reference: numpy-reference/reference/generated/numpy.full.html
   */
  NP_API template <int... E, typename T>
  NP_NODISCARD constexpr ndarrayf<T, E...> full(const T& fill_value)
  {
    ndarrayf<T, E...> out{};
    out.fill(fill_value);
    return out;
  }

  /* @brief Array filled with a constant value (element type first).
   *
   * @tparam T  Element type.
   * @tparam E  Compile-time extents (must all be > 0).
   * @param  fill_value  Value to fill every element with.
   * @return             ndarrayf<T, E...> filled with fill_value.
   */
  NP_API template <typename T, int... E>
  NP_NODISCARD constexpr ndarrayf<T, E...> full(const T& fill_value)
  {
    ndarrayf<T, E...> out{};
    out.fill(fill_value);
    return out;
  }

  /* @brief Identity-like matrix with ones on the k-th diagonal.
   *
   * @tparam N   Number of rows (must be > 0).
   * @tparam M   Number of columns (default: N, making it square).
   * @tparam k   Diagonal offset (default: 0; positive = above main).
   * @tparam T   Element type (default: double).
   * @return     ndarrayf<T, N, M> with ones on the k-th diagonal.
   *
   * Reference: numpy-reference/reference/generated/numpy.eye.html
   */
  NP_API template <std::size_t N, std::size_t M = N, int k = 0, typename T = double>
  NP_NODISCARD constexpr auto eye() -> ndarrayf<T, N, M>
  {
    ndarrayf<T, N, M> out{};
    const std::ptrdiff_t kk = k;
    for (std::size_t i = 0; i < N; ++i)
    {
      const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + kk;
      if (j >= 0 && j < static_cast<std::ptrdiff_t>(M))
      {
        out.m_data[i * M + static_cast<std::size_t>(j)] = T{1};
      }
    }
    return out;
  }

  /* @brief Square identity matrix of size N x N.
   *
   * @tparam N  Size of the square matrix (must be > 0).
   * @tparam T  Element type (default: double).
   * @return    ndarrayf<T, N, N> with ones on the main diagonal.
   *
   * Reference: numpy-reference/reference/generated/numpy.identity.html
   */
  NP_API template <std::size_t N, typename T = double>
  NP_NODISCARD constexpr ndarrayf<T, N, N> identity()
  {
    return eye<N, N, 0, T>();
  }

  /* @brief Values 0..N-1 (numpy arange with a compile-time element count).
   *
   * @tparam N  Number of elements (must be > 0).
   * @tparam T  Element type (default: int).
   * @return    ndarrayf<T, N> containing {0, 1, 2, ..., N-1}.
   *
   * Reference: numpy-reference/reference/generated/numpy.arange.html
   */
  NP_API template <std::size_t N, typename T = int>
  NP_NODISCARD constexpr ndarrayf<T, N> arange()
  {
    ndarrayf<T, N> out{};
    for (std::size_t i = 0; i < N; ++i)
    {
      out[i] = static_cast<T>(i);
    }
    return out;
  }

  /* @brief N values from start (inclusive), step 1.
   *
   * @tparam N  Number of elements (must be > 0).
   * @tparam T  Element type.
   * @param start  Start value (inclusive).
   * @param stop   Stop value (exclusive; not used in the generated sequence,
   *               only determines the element count N).
   * @return       ndarrayf<T, N> containing {start, start+1, ..., start+N-1}.
   */
  NP_API template <std::size_t N, typename T>
  NP_NODISCARD constexpr auto arange(T start, T stop) -> ndarrayf<T, N>
  {
    (void)stop;
    ndarrayf<T, N> out{};
    for (std::size_t i = 0; i < N; ++i)
    {
      out[i] = start + static_cast<T>(i);
    }
    return out;
  }

  /* @brief N values from start (inclusive) with the given step.
   *
   * @tparam N     Number of elements (must be > 0).
   * @tparam T     Element type.
   * @param start  Start value (inclusive).
   * @param stop   Stop value (exclusive; not used in the generated sequence,
   *               only determines the element count N).
   * @param step   Step size.
   * @return       ndarrayf<T, N> containing {start, start+step, ...,
   * start+(N-1)*step}.
   */
  NP_API template <std::size_t N, typename T>
  NP_NODISCARD constexpr auto arange(T start, T stop, T step) -> ndarrayf<T, N>
  {
    (void)stop;
    ndarrayf<T, N> out{};
    for (std::size_t i = 0; i < N; ++i)
    {
      out[i] = start + step * static_cast<T>(i);
    }
    return out;
  }

  /* @brief N evenly spaced values from start to stop (inclusive).
   *        Integer inputs are promoted to double, as in numpy.
   *
   * @tparam N        Number of elements (must be > 0).
   * @tparam endpoint Whether to include stop in the sequence (default: true).
   * @tparam T        Element type (deduced from arguments).
   * @param start     Start value (inclusive).
   * @param stop      Stop value (inclusive when endpoint is true).
   * @return          ndarrayf<R, N> where R is double if T is integral,
   *                   otherwise T.
   *
   * Reference: numpy-reference/reference/generated/numpy.linspace.html
   */
  NP_API template <std::size_t N, bool endpoint = true, typename T>
  NP_NODISCARD constexpr auto linspace(T start, T stop)
  {
    using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
    ndarrayf<R, N> out{};
    if constexpr (N == 1)
    {
      out[0] = static_cast<R>(start);
    }
    else
    {
      const R delta = endpoint
          ? (static_cast<R>(stop) - static_cast<R>(start)) / static_cast<R>(N - 1)
          : (static_cast<R>(stop) - static_cast<R>(start)) / static_cast<R>(N);
      for (std::size_t i = 0; i < N; ++i)
      {
        out[i] = static_cast<R>(start) + delta * static_cast<R>(i);
      }
    }
    return out;
  }

} // namespace np

#endif // NP_CREATION_FIXED_HPP
