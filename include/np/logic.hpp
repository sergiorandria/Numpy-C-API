/**
 * @file logic.hpp
 * @brief Logic functions (truth value testing, type checks, comparisons).
 *
 * Provides NumPy-compatible logical operations:
 *   Type checks: isfinite, isinf, isnan, isreal, iscomplex
 *   Logical ops: logical_and, logical_or, logical_not, logical_xor
 *   Comparisons: allclose, isclose, array_equal, array_equiv
 *   Element-wise comparisons: greater, less, equal, not_equal, etc.
 *
 * All functions return C-contiguous arrays with row-major strides.
 * Binary operations broadcast shapes according to NumPy rules.
 *
 * Reference: numpy-reference/reference/routines.logic.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_LOGIC_HPP
#define NP_LOGIC_HPP

#include <cmath>
#include <limits>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np {

// =================================================================
// Type checks
// Reference: numpy-reference/reference/generated/numpy.isfinite.html (etc.)
// =================================================================

/** @brief Test element-wise for finiteness (not infinity and not NaN).
 *
 * @tparam T  Element type (floating-point or complex).
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is finite.
 */
NP_API template <typename T> NP_NODISCARD auto isfinite(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    result.set(idx, std::isfinite(x.get(idx)));
    // Increment index
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Test element-wise for positive or negative infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is inf.
 */
NP_API template <typename T> NP_NODISCARD auto isinf(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    result.set(idx, std::isinf(x.get(idx)));
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Test element-wise for NaN.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is NaN.
 */
NP_API template <typename T> NP_NODISCARD auto isnan(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    result.set(idx, std::isnan(x.get(idx)));
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Test element-wise for negative infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is -inf.
 */
NP_API template <typename T> NP_NODISCARD auto isneginf(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const T val = x.get(idx);
    result.set(idx, std::isinf(val) && (val < T{0}));
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Test element-wise for positive infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is +inf.
 */
NP_API template <typename T> NP_NODISCARD auto isposinf(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const T val = x.get(idx);
    result.set(idx, std::isinf(val) && (val > T{0}));
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Returns True if input is complex.
 *
 * The result is uniform across all elements (the dtype
 * of the array determines whether elements are complex).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true for all elements if T
 *            is a complex type, false otherwise.
 */
NP_API template <typename T> NP_NODISCARD auto iscomplex(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  result.fill(detail::is_complex_v<T>);
  return result;
}

/** @brief Returns True if input is real (not complex).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true for all elements if T
 *            is not a complex type, false otherwise.
 */
NP_API template <typename T> NP_NODISCARD auto isreal(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  result.fill(!detail::is_complex_v<T>);
  return result;
}

/** @brief Returns True if input is a scalar type.
 *
 * @tparam T  Type to check.
 * @param x   Value (unused; only the type matters).
 * @return    True if T is arithmetic or a complex instantiation.
 */
NP_API template <typename T> constexpr bool isscalar([[maybe_unused]] const T &x) {
  return std::is_arithmetic_v<T> || detail::is_complex_v<T>;
}

// =================================================================
// Logical operations
// Reference: numpy-reference/reference/generated/numpy.logical_and.html (etc.)
// =================================================================

/** @brief Compute truth value of x1 AND x2 element-wise.
 *
 * Broadcasts x1 and x2 to a common shape.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array (converted to bool).
 * @param x2  Second input array (converted to bool).
 * @return    ndarray<bool> with x1[i] && x2[i].
 * @throws    std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_and(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
  ndarray<bool> result(out_shape, dtype::bool_);

  const auto ndim_out = out_shape.size();
  std::vector<std::size_t> idx(ndim_out, 0);

  for (std::size_t i = 0; i < result.size(); ++i) {
    std::vector<std::size_t> idx1(x1.ndim(), 0);
    std::vector<std::size_t> idx2(x2.ndim(), 0);

    for (std::size_t d = 0; d < ndim_out; ++d) {
      if (d >= ndim_out - x1.ndim()) {
        const auto d1 = d - (ndim_out - x1.ndim());
        idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
      }
      if (d >= ndim_out - x2.ndim()) {
        const auto d2 = d - (ndim_out - x2.ndim());
        idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
      }
    }

    const bool val1 = static_cast<bool>(x1.get(idx1));
    const bool val2 = static_cast<bool>(x2.get(idx2));
    result.set(idx, val1 && val2);

    for (std::size_t d = ndim_out; d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }

  return result;
}

/** @brief Compute truth value of x1 OR x2 element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] || x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_or(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
  ndarray<bool> result(out_shape, dtype::bool_);

  const auto ndim_out = out_shape.size();
  std::vector<std::size_t> idx(ndim_out, 0);

  for (std::size_t i = 0; i < result.size(); ++i) {
    std::vector<std::size_t> idx1(x1.ndim(), 0);
    std::vector<std::size_t> idx2(x2.ndim(), 0);

    for (std::size_t d = 0; d < ndim_out; ++d) {
      if (d >= ndim_out - x1.ndim()) {
        const auto d1 = d - (ndim_out - x1.ndim());
        idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
      }
      if (d >= ndim_out - x2.ndim()) {
        const auto d2 = d - (ndim_out - x2.ndim());
        idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
      }
    }

    const bool val1 = static_cast<bool>(x1.get(idx1));
    const bool val2 = static_cast<bool>(x2.get(idx2));
    result.set(idx, val1 || val2);

    for (std::size_t d = ndim_out; d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }

  return result;
}

/** @brief Compute truth value of NOT x element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with !x[i].
 */
NP_API template <typename T> NP_NODISCARD auto logical_not(const ndarray<T> &x) -> ndarray<bool> {
  ndarray<bool> result(x.shape, dtype::bool_);
  std::vector<std::size_t> idx(x.ndim(), 0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    result.set(idx, !static_cast<bool>(x.get(idx)));
    for (std::size_t d = x.ndim(); d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(x.shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return result;
}

/** @brief Compute truth value of x1 XOR x2 element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with (x1[i] != x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_xor(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
  ndarray<bool> result(out_shape, dtype::bool_);

  const auto ndim_out = out_shape.size();
  std::vector<std::size_t> idx(ndim_out, 0);

  for (std::size_t i = 0; i < result.size(); ++i) {
    std::vector<std::size_t> idx1(x1.ndim(), 0);
    std::vector<std::size_t> idx2(x2.ndim(), 0);

    for (std::size_t d = 0; d < ndim_out; ++d) {
      if (d >= ndim_out - x1.ndim()) {
        const auto d1 = d - (ndim_out - x1.ndim());
        idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
      }
      if (d >= ndim_out - x2.ndim()) {
        const auto d2 = d - (ndim_out - x2.ndim());
        idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
      }
    }

    const bool val1 = static_cast<bool>(x1.get(idx1));
    const bool val2 = static_cast<bool>(x2.get(idx2));
    result.set(idx, val1 != val2);

    for (std::size_t d = ndim_out; d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }

  return result;
}

// =================================================================
// Comparison functions
// Reference: numpy-reference/reference/generated/numpy.greater.html (etc.)
// =================================================================

/** @brief Return (x1 > x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] > x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto greater(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  return x1 > x2;
}

/** @brief Return (x1 >= x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] >= x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto greater_equal(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<bool> {
  return x1 >= x2;
}

/** @brief Return (x1 < x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] < x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto less(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  return x1 < x2;
}

/** @brief Return (x1 <= x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] <= x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto less_equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  return x1 <= x2;
}

/** @brief Return (x1 == x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] == x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  return x1 == x2;
}

/** @brief Return (x1 != x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] != x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto not_equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool> {
  return x1 != x2;
}

// =================================================================
// Array comparison
// Reference: numpy-reference/reference/generated/numpy.array_equal.html (etc.)
// =================================================================

/** @brief True if two arrays have the same shape and elements.
 *
 * Time complexity: O(N) where N is the total element count.
 *
 * @tparam T  Element type of a1.
 * @tparam U  Element type of a2.
 * @param a1  First array.
 * @param a2  Second array.
 * @return    True if shapes match and all elements are equal.
 */
NP_API template <typename T, typename U>
bool array_equal(const ndarray<T> &a1, const ndarray<U> &a2) {
  if (a1.shape != a2.shape) {
    return false;
  }
  auto it1 = a1.begin();
  auto it2 = a2.begin();
  for (; it1 != a1.end(); ++it1, ++it2) {
    if (*it1 != static_cast<T>(*it2)) {
      return false;
    }
  }
  return true;
}

/** @brief True if two arrays are element-wise equal within a tolerance.
 *
 * Uses the standard absolute + relative tolerance formula:
 *   |a - b| <= atol + rtol * |b|
 *
 * @tparam T  Element type of a.
 * @tparam U  Element type of b.
 * @param a   First array.
 * @param b   Second array.
 * @param rtol Relative tolerance (default: 1e-5).
 * @param atol Absolute tolerance (default: 1e-8).
 * @return     ndarray<bool> with true where elements are close.
 * @throws     std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto isclose(const ndarray<T> &a, const ndarray<U> &b, double rtol = 1e-5,
             double atol = 1e-8) -> ndarray<bool> {
  const auto out_shape = detail::broadcast_shapes(a.shape, b.shape);
  ndarray<bool> result(out_shape, dtype::bool_);

  const auto ndim_out = out_shape.size();
  std::vector<std::size_t> idx(ndim_out, 0);

  for (std::size_t i = 0; i < result.size(); ++i) {
    std::vector<std::size_t> idx_a(a.ndim(), 0);
    std::vector<std::size_t> idx_b(b.ndim(), 0);

    for (std::size_t d = 0; d < ndim_out; ++d) {
      if (d >= ndim_out - a.ndim()) {
        const auto da = d - (ndim_out - a.ndim());
        idx_a[da] = (a.shape[da] == 1) ? 0 : idx[d];
      }
      if (d >= ndim_out - b.ndim()) {
        const auto db = d - (ndim_out - b.ndim());
        idx_b[db] = (b.shape[db] == 1) ? 0 : idx[d];
      }
    }

    const double val_a = static_cast<double>(a.get(idx_a));
    const double val_b = static_cast<double>(b.get(idx_b));
    const double diff = std::abs(val_a - val_b);
    const double threshold = atol + rtol * std::abs(val_b);

    result.set(idx, diff <= threshold);

    for (std::size_t d = ndim_out; d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }

  return result;
}

/** @brief True if two arrays are element-wise equal within a tolerance.
 *
 * Reduces isclose() to a single boolean via .all().
 *
 * @tparam T  Element type of a.
 * @tparam U  Element type of b.
 * @param a   First array.
 * @param b   Second array.
 * @param rtol Relative tolerance (default: 1e-5).
 * @param atol Absolute tolerance (default: 1e-8).
 * @return     True if all elements are close.
 */
NP_API template <typename T, typename U>
bool allclose(const ndarray<T> &a, const ndarray<U> &b, double rtol = 1e-5,
              double atol = 1e-8) {
  return isclose(a, b, rtol, atol).all();
}

/** @brief True if two arrays are broadcastable and element-wise equal.
 *
 * Unlike array_equal(), this function broadcasts shapes before
 * comparing. Returns false if broadcasting fails.
 *
 * @tparam T  Element type of a1.
 * @tparam U  Element type of a2.
 * @param a1  First array.
 * @param a2  Second array.
 * @return    True if arrays are broadcast-equal.
 */
NP_API template <typename T, typename U>
bool array_equiv(const ndarray<T> &a1, const ndarray<U> &a2) {
  try {
    const auto out_shape = detail::broadcast_shapes(a1.shape, a2.shape);
    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    // Compute total elements
    std::size_t total_elems = 1;
    for (int d : out_shape) {
      total_elems *= static_cast<std::size_t>(d);
    }

    for (std::size_t i = 0; i < total_elems; ++i) {
      std::vector<std::size_t> idx1(a1.ndim(), 0);
      std::vector<std::size_t> idx2(a2.ndim(), 0);

      for (std::size_t d = 0; d < ndim_out; ++d) {
        if (d >= ndim_out - a1.ndim()) {
          const auto d1 = d - (ndim_out - a1.ndim());
          idx1[d1] = (a1.shape[d1] == 1) ? 0 : idx[d];
        }
        if (d >= ndim_out - a2.ndim()) {
          const auto d2 = d - (ndim_out - a2.ndim());
          idx2[d2] = (a2.shape[d2] == 1) ? 0 : idx[d];
        }
      }

      if (a1.get(idx1) != static_cast<T>(a2.get(idx2))) {
        return false;
      }

      for (std::size_t d = ndim_out; d-- > 0;) {
        if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
          break;
        }
        idx[d] = 0;
      }
    }

    return true;
  } catch (...) {
    return false;
  }
}

} // namespace np

#endif // NP_LOGIC_HPP
