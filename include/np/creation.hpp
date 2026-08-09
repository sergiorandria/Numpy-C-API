/**
 * @file creation.hpp
 * @brief Array creation routines (np::zeros, np::ones, np::arange, ...).
 *
 * Mirrors numpy's creation API:
 *   zeros, ones, full, empty, empty_like/zeros_like/ones_like,
 *   arange, linspace, logspace, eye, identity, asarray.
 *
 * All functions return C-contiguous arrays with row-major strides.
 * The dynamic path throws std::invalid_argument on shape mismatches;
 * the fixed-shape path (creation_fixed.hpp) encodes shape in the
 * type and rejects mismatches at compile time.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CREATION_HPP
#define NP_CREATION_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np {

/** @brief Array of zeros with the given shape.
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector; must have at least one element.
 * @return       ndarray<T> of the given shape, filled with T{0}.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.zeros.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto zeros(const std::vector<int> &shape) -> ndarray<T> {
  return ndarray<T>(shape, dtype_of<T>, T{0});
}

/** @brief Array of ones with the given shape.
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector; must have at least one element.
 * @return       ndarray<T> of the given shape, filled with T{1}.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.ones.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto ones(const std::vector<int> &shape) -> ndarray<T> {
  return ndarray<T>(shape, dtype_of<T>, T{1});
}

/** @brief Array filled with a constant value.
 *
 * @tparam T  Element type (deduced from fill_value).
 * @param shape      Shape vector.
 * @param fill_value Value to fill every element with.
 * @return           ndarray<T> of the given shape, filled with fill_value.
 * @throws           std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.full.html
 */
NP_API template <typename T>
NP_NODISCARD auto full(const std::vector<int> &shape, const T &fill_value) -> ndarray<T> {
  return ndarray<T>(shape, dtype_of<T>, fill_value);
}

/** @brief Uninitialized array (values are default-constructed in C++).
 *
 * The memory is allocated but not zeroed; elements hold their
 * default-constructed values. For scalar types this means
 * indeterminate values for built-in types (same as `new T[n]`).
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector.
 * @return       ndarray<T> of the given shape with default-constructed
 * elements.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.empty.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto empty(const std::vector<int> &shape) -> ndarray<T> {
  return ndarray<T>(shape, dtype_of<T>, T{});
}

/** @brief New array with the same shape as `a` (uninitialized).
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, default-constructed
 * elements.
 */
NP_API template <typename T>
NP_NODISCARD auto empty_like(const ndarray<T> &a) -> ndarray<T> {
  return ndarray<T>(a.shape, a.type);
}

/** @brief Zeros with the same shape as `a`.
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, filled with T{0}.
 */
NP_API template <typename T>
NP_NODISCARD auto zeros_like(const ndarray<T> &a) -> ndarray<T> {
  return ndarray<T>(a.shape, a.type, T{0});
}

/** @brief Ones with the same shape as `a`.
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, filled with T{1}.
 */
NP_API template <typename T>
NP_NODISCARD auto ones_like(const ndarray<T> &a) -> ndarray<T> {
  return ndarray<T>(a.shape, a.type, T{1});
}

/** @brief Filled with `fill_value` using the shape and dtype of `a`.
 *
 * @tparam T       Element type of `a` and `fill_value`.
 * @param a        Source array whose shape and dtype are copied.
 * @param fill_value Value to fill every element with.
 * @return         ndarray<T> with the shape and dtype of `a`, filled with
 * fill_value.
 */
NP_API template <typename T>
NP_NODISCARD auto full_like(const ndarray<T> &a, const T &fill_value) -> ndarray<T> {
  return ndarray<T>(a.shape, a.type, fill_value);
}

/** @brief Values evenly spaced from start (inclusive) to stop (exclusive).
 *
 * Computes the number of elements as ceil((stop - start) / step),
 * then generates start + step * i for i in [0, n). If step > 0 and
 * stop <= start, returns a 1-D array of size 0. If step < 0 and
 * stop >= start, returns a 1-D array of size 0.
 *
 * @tparam T  Element type (deduced from arguments).
 * @param start  Start value (inclusive).
 * @param stop   Stop value (exclusive).
 * @param step   Step size (default: T{1}); must not be zero.
 * @return       1-D ndarray<T> of the computed length.
 * @throws       std::invalid_argument if step is zero.
 *
 * Reference: numpy-reference/reference/generated/numpy.arange.html
 */
NP_API template <typename T>
NP_NODISCARD auto arange(T start, T stop, T step = T{1}) -> ndarray<T> {
  if (step == T{0}) {
    throw std::invalid_argument("arange step cannot be zero");
  }
  std::vector<T> out;
  if (step > T{0}) {
    if (stop <= start) {
      return ndarray<T>({0}, dtype_of<T>, T{});
    }
    const std::size_t n = static_cast<std::size_t>(
        std::ceil((static_cast<double>(stop) - static_cast<double>(start)) /
                  static_cast<double>(step)));
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(start + step * static_cast<T>(i));
    }
  } else {
    if (stop >= start) {
      return ndarray<T>({0}, dtype_of<T>, T{});
    }
    const std::size_t n = static_cast<std::size_t>(
        std::ceil((static_cast<double>(stop) - static_cast<double>(start)) /
                  static_cast<double>(step)));
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(start + step * static_cast<T>(i));
    }
  }
  const int n_elems = static_cast<int>(out.size());
  return ndarray<T>::from_data(std::vector<int>{n_elems}, std::move(out));
}

/** @brief Values from 0 to stop (exclusive).
 *
 * Equivalent to arange(T{0}, stop, T{1}).
 *
 * @tparam T  Element type (deduced from stop).
 * @param stop  Exclusive upper bound.
 * @return      1-D ndarray<T> of length max(0, ceil(stop)).
 */
NP_API template <typename T>
NP_NODISCARD auto arange(T stop) -> ndarray<T> {
  return arange(T{0}, stop, T{1});
}

/** @brief num evenly spaced values from start to stop (inclusive).
 *
 * When endpoint is true (default), the sequence includes stop.
 * When endpoint is false, stop is excluded and the step is
 * (stop - start) / num. Integer inputs are promoted to double.
 *
 * @tparam T  Element type (deduced from start/stop).
 * @param start    Start value (inclusive).
 * @param stop     Stop value (inclusive when endpoint is true).
 * @param num      Number of samples (default: 50); must be > 0.
 * @param endpoint Whether to include stop in the sequence.
 * @return         1-D ndarray<R> where R is double if T is integral,
 *                 otherwise T.
 *
 * Reference: numpy-reference/reference/generated/numpy.linspace.html
 */
NP_API template <typename T>
NP_NODISCARD auto linspace(T start, T stop, std::size_t num = 50, bool endpoint = true)
    -> ndarray<std::conditional_t<std::is_floating_point_v<T>, T, double>> {
  using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  if (num == 0) {
    return ndarray<R>(std::vector<int>{0});
  }
  std::vector<R> out;
  out.reserve(num);
  if (num == 1) {
    out.push_back(static_cast<R>(start));
    return ndarray<R>::from_data(std::vector<int>{1}, std::move(out));
  }
  const R delta = endpoint ? (static_cast<R>(stop) - static_cast<R>(start)) /
                                 static_cast<R>(num - 1)
                           : (static_cast<R>(stop) - static_cast<R>(start)) /
                                 static_cast<R>(num);
  for (std::size_t i = 0; i < num; ++i) {
    out.push_back(static_cast<R>(start) + delta * static_cast<R>(i));
  }
  return ndarray<R>::from_data(std::vector<int>{static_cast<int>(num)},
                               std::move(out));
}

/** @brief Logarithmically spaced values from base^start to base^stop.
 *
 * Uses linspace internally to generate the exponent values, then
 * applies std::pow(base, exponent) element-wise.
 *
 * @tparam T  Element type (deduced from start/stop).
 * @param start  Start exponent (inclusive).
 * @param stop   Stop exponent (inclusive).
 * @param num    Number of samples (default: 50).
 * @param base   The base of the logarithm (default: T{10}).
 * @return       1-D ndarray<double> of num elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.logspace.html
 */
NP_API template <typename T>
NP_NODISCARD auto logspace(T start, T stop, std::size_t num = 50, T base = T{10})
    -> ndarray<double> {
  auto powers = linspace(start, stop, num);
  ndarray<double> out(std::vector<int>{static_cast<int>(num)});
  for (std::size_t i = 0; i < num; ++i) {
    out.data()[i] = std::pow(static_cast<double>(base),
                             static_cast<double>(powers.data()[i]));
  }
  return out;
}

/** @brief Identity matrix of size n x n with optional offset k.
 *
 * The diagonal at offset k is set to T{1}. k > 0 places the
 * diagonal above the main diagonal; k < 0 below it.
 *
 * @tparam T  Element type (default: double).
 * @param n  Number of rows.
 * @param m  Number of columns (default: n, making a square matrix).
 * @param k  Diagonal offset (default: 0).
 * @return   ndarray<T> of shape (n, m) with ones on the k-th diagonal.
 *
 * Reference: numpy-reference/reference/generated/numpy.eye.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto eye(std::size_t n, std::size_t m = 0, int k = 0) -> ndarray<T> {
  if (m == 0) {
    m = n;
  }
  std::vector<int> shape = {static_cast<int>(n), static_cast<int>(m)};
  ndarray<T> out(shape, dtype_of<T>, T{0});
  const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
  const std::ptrdiff_t cols = static_cast<std::ptrdiff_t>(m);
  for (std::ptrdiff_t i = 0; i < rows; ++i) {
    const std::ptrdiff_t j = i + k;
    if (j >= 0 && j < cols) {
      out.set(std::array<std::size_t, 2>{static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(j)},
              T{1});
    }
  }
  return out;
}

/** @brief Identity matrix of size n x n.
 *
 * Equivalent to eye<T>(n, n, 0).
 *
 * @tparam T  Element type (default: double).
 * @param n  Size of the square matrix.
 * @return   ndarray<T> of shape (n, n) with ones on the main diagonal.
 *
 * Reference: numpy-reference/reference/generated/numpy.identity.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto identity(std::size_t n) -> ndarray<T> {
  return eye<T>(n, n, 0);
}

/** @brief 1D array from a std::vector (copies).
 *
 * @tparam T  Element type.
 * @param values  Source vector (copied).
 * @return        1-D ndarray<T> with the same elements as values.
 */
NP_API template <typename T>
NP_NODISCARD auto asarray(const std::vector<T> &values) -> ndarray<T> {
  return ndarray<T>::from_data(
      std::vector<int>{static_cast<int>(values.size())},
      std::vector<T>(values));
}

/** @brief 1D array from a std::array (copies).
 *
 * @tparam T  Element type.
 * @tparam N  Size of the source array.
 * @param values  Source std::array (copied).
 * @return        1-D ndarray<T> with the same elements.
 */
NP_API template <typename T, std::size_t N>
NP_NODISCARD auto asarray(const std::array<T, N> &values) -> ndarray<T> {
  return ndarray<T>::from_data(std::vector<int>{static_cast<int>(N)},
                               std::vector<T>(values.begin(), values.end()));
}

/** @brief Array of the given shape from a contiguous std::vector.
 *
 * The total number of elements in values must equal the product
 * of the shape dimensions. The data is copied.
 *
 * @tparam T  Element type.
 * @param values  Source vector (copied).
 * @param shape   Target shape; must have at least one element.
 * @return        ndarray<T> of the given shape.
 * @throws        std::invalid_argument if sizes do not match.
 *
 * Reference: numpy-reference/reference/generated/numpy.asarray.html
 */
NP_API template <typename T>
NP_NODISCARD auto asarray(const std::vector<T> &values, const std::vector<int> &shape)
    -> ndarray<T> {
  return ndarray<T>::from_data(shape, std::vector<T>(values));
}

} // namespace np

#endif // NP_CREATION_HPP
