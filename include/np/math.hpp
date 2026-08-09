/**
 * @file math.hpp
 * @brief Element-wise mathematical functions (NumPy ufuncs).
 *
 * Provides broadcasting-aware wrappers around standard math functions:
 *   Trigonometric: sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot
 *   Hyperbolic: sinh, cosh, tanh, arcsinh, arccosh, arctanh
 *   Exponential/Logarithmic: exp, log, log10, log2, sqrt, power
 *   Rounding: floor, ceil, trunc, rint
 *   Arithmetic: absolute, sign, maximum, minimum, fmod, fmax, fmin
 *   Misc: degrees, radians, square, cbrt, reciprocal
 *
 * All functions return C-contiguous arrays with row-major strides.
 * Binary operations broadcast shapes according to NumPy rules
 * (see numpy-reference/user/basics.broadcasting.html).
 *
 * Reference: numpy-reference/reference/routines.math.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MATH_HPP
#define NP_MATH_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np {

namespace detail {

/** @brief Apply unary function element-wise with broadcasting.
 *
 * Time complexity: O(N) where N is the number of elements
 * in the broadcast output shape. Space complexity: O(N).
 *
 * @tparam T  Element type of the input array.
 * @tparam Fn  Unary callable of the form `R(T)`.
 * @param arr  Input array.
 * @param fn   Unary function applied to each element.
 * @return     ndarray<T> with the same shape as `arr`.
 */
NP_API template <typename T, typename Fn>
NP_NODISCARD auto ufunc_unary(const ndarray<T> &arr, Fn &&fn) -> ndarray<T> {
  ndarray<T> result(arr.shape, arr.type);
  auto it_in = arr.begin();
  auto it_out = result.begin();
  for (; it_in != arr.end(); ++it_in, ++it_out) {
    *it_out = fn(*it_in);
  }
  return result;
}

/** @brief Apply binary function element-wise with broadcasting.
 *
 * Time complexity: O(N) where N is the number of elements
 * in the broadcast output shape. Space complexity: O(N).
 *
 * @tparam T  Element type of the left-hand side array.
 * @tparam U  Element type of the right-hand side array.
 * @tparam Fn Binary callable of the form `R(T, U)`.
 * @param lhs Left-hand side array.
 * @param rhs Right-hand side array.
 * @param fn  Binary function applied to each pair of elements.
 * @return    ndarray<std::common_type_t<T, U>> with the
 *            broadcast shape.
 * @throws    std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U, typename Fn>
NP_NODISCARD auto ufunc_binary(const ndarray<T> &lhs, const ndarray<U> &rhs, Fn &&fn)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;

  // Broadcast shapes
  const auto out_shape = detail::broadcast_shapes(lhs.shape, rhs.shape);
  ndarray<R> result(out_shape, dtype_of<R>);

  // Iterate and apply
  const auto ndim_out = out_shape.size();
  std::vector<std::size_t> idx(ndim_out, 0);

  for (std::size_t i = 0; i < result.size(); ++i) {
    // Map output index to input indices
    std::vector<std::size_t> idx_lhs(lhs.ndim(), 0);
    std::vector<std::size_t> idx_rhs(rhs.ndim(), 0);

    for (std::size_t d = 0; d < ndim_out; ++d) {
      if (d >= ndim_out - lhs.ndim()) {
        const auto d_lhs = d - (ndim_out - lhs.ndim());
        idx_lhs[d_lhs] = (lhs.shape[d_lhs] == 1) ? 0 : idx[d];
      }
      if (d >= ndim_out - rhs.ndim()) {
        const auto d_rhs = d - (ndim_out - rhs.ndim());
        idx_rhs[d_rhs] = (rhs.shape[d_rhs] == 1) ? 0 : idx[d];
      }
    }

    const T val_lhs = lhs.get(idx_lhs);
    const U val_rhs = rhs.get(idx_rhs);
    result.set(idx, fn(val_lhs, val_rhs));

    // Increment index
    for (std::size_t d = ndim_out; d-- > 0;) {
      if (++idx[d] < static_cast<std::size_t>(out_shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }

  return result;
}
} // namespace detail

// =================================================================
// Trigonometric functions
// Reference: numpy-reference/reference/generated/numpy.sin.html (etc.)
// =================================================================

/** @brief Trigonometric sine, element-wise.
 *
 * @tparam T  Element type (must be floating-point or complex).
 * @param x   Input array.
 * @return    ndarray<T> with sin(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto sin(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sin(v); });
}

/** @brief Trigonometric cosine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with cos(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto cos(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cos(v); });
}

/** @brief Trigonometric tangent, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with tan(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto tan(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::tan(v); });
}

/** @brief Inverse sine, element-wise.
 *
 * Returns values in [-pi/2, pi/2]. For real inputs outside
 * [-1, 1], the result is complex (promoted to complex output).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with asin(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto arcsin(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::asin(v); });
}

/** @brief Inverse cosine, element-wise.
 *
 * Returns values in [0, pi]. For real inputs outside [-1, 1],
 * the result is complex (promoted to complex output).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with acos(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto arccos(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::acos(v); });
}

/** @brief Inverse tangent, element-wise.
 *
 * Returns values in [-pi/2, pi/2].
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with atan(x[i]) for each element.
 */
NP_API template <typename T> NP_NODISCARD auto arctan(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::atan(v); });
}

/** @brief Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
 *
 * The four quadrants are distinguished by the signs of both
 * arguments. For real inputs, the result is in [-pi, pi].
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  y-coordinate array.
 * @param x2  x-coordinate array.
 * @return    ndarray<std::common_type_t<T, U>> with atan2(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto arctan2(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &y, const U &x) {
    return static_cast<R>(std::atan2(y, x));
  });
}

/** @brief Given sides of a right triangle, return its hypotenuse.
 *
 * Computes sqrt(x1^2 + x2^2) in a numerically stable way.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First side array.
 * @param x2  Second side array.
 * @return    ndarray<std::common_type_t<T, U>> with hypot(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto hypot(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::hypot(a, b));
  });
}

/** @brief Convert angles from radians to degrees.
 *
 * @tparam T  Element type.
 * @param x   Input array in radians.
 * @return    ndarray<T> with degrees(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto degrees(const ndarray<T> &x) -> ndarray<T> {
  constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
  return detail::ufunc_unary(
      x, [](const T &v) { return static_cast<T>(v * rad_to_deg); });
}

/** @brief Convert angles from degrees to radians.
 *
 * @tparam T  Element type.
 * @param x   Input array in degrees.
 * @return    ndarray<T> with radians(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto radians(const ndarray<T> &x) -> ndarray<T> {
  constexpr double deg_to_rad = 3.14159265358979323846 / 180.0;
  return detail::ufunc_unary(
      x, [](const T &v) { return static_cast<T>(v * deg_to_rad); });
}

/** @brief Alias for degrees().
 *
 * @tparam T  Element type.
 * @param x   Input array in radians.
 * @return    ndarray<T> with degrees(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto rad2deg(const ndarray<T> &x) -> ndarray<T> {
  return degrees(x);
}

/** @brief Alias for radians().
 *
 * @tparam T  Element type.
 * @param x   Input array in degrees.
 * @return    ndarray<T> with radians(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto deg2rad(const ndarray<T> &x) -> ndarray<T> {
  return radians(x);
}

// =================================================================
// Hyperbolic functions
// Reference: numpy-reference/reference/generated/numpy.sinh.html (etc.)
// =================================================================

/** @brief Hyperbolic sine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with sinh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto sinh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sinh(v); });
}

/** @brief Hyperbolic cosine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with cosh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto cosh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cosh(v); });
}

/** @brief Hyperbolic tangent, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with tanh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto tanh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::tanh(v); });
}

/** @brief Inverse hyperbolic sine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with asinh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto arcsinh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::asinh(v); });
}

/** @brief Inverse hyperbolic cosine, element-wise.
 *
 * Domain: x >= 1. For real inputs < 1, the result is NaN.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with acosh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto arccosh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::acosh(v); });
}

/** @brief Inverse hyperbolic tangent, element-wise.
 *
 * Domain: |x| < 1. For |x| >= 1, the result is NaN or inf.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with atanh(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto arctanh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::atanh(v); });
}

// =================================================================
// Exponential and logarithmic functions
// Reference: numpy-reference/reference/generated/numpy.exp.html (etc.)
// =================================================================

/** @brief Calculate the exponential of all elements.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with exp(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto exp(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::exp(v); });
}

/** @brief Calculate exp(x) - 1 for all elements.
 *
 * More accurate than exp(x) - 1 for small x.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with expm1(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto expm1(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::expm1(v); });
}

/** @brief Calculate 2**x for all elements.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with 2^x[i].
 */
NP_API template <typename T> NP_NODISCARD auto exp2(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::exp2(v); });
}

/** @brief Natural logarithm, element-wise.
 *
 * For x <= 0, the result is NaN (or -inf for x == 0).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto log(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log(v); });
}

/** @brief Base-10 logarithm, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log10(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto log10(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log10(v); });
}

/** @brief Base-2 logarithm, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log2(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto log2(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log2(v); });
}

/** @brief Calculate log(1 + x) for all elements.
 *
 * More accurate than log(1 + x) for small x.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log1p(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto log1p(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log1p(v); });
}

/** @brief Non-negative square root, element-wise.
 *
 * For negative real inputs, the result is NaN. For complex
 * inputs, the principal square root is returned.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with sqrt(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto sqrt(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sqrt(v); });
}

/** @brief Cube root, element-wise.
 *
 * For real inputs, the real cube root is returned (including
 * for negative inputs).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with cbrt(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto cbrt(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cbrt(v); });
}

/** @brief Element-wise square.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with x[i]^2.
 */
NP_API template <typename T> NP_NODISCARD auto square(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return v * v; });
}

/** @brief First array elements raised to powers from second array, element-wise.
 *
 * For integer exponents, the result is exact (no floating-point
 * rounding in the exponentiation itself). For non-integer
 * exponents or negative bases, std::pow is used.
 *
 * @tparam T  Element type of the base array.
 * @tparam U  Element type of the exponent array.
 * @param x1  Base array.
 * @param x2  Exponent array.
 * @return    ndarray<std::common_type_t<T, U>> with x1[i]^x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto power(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &base, const U &exp) {
    return static_cast<R>(std::pow(base, exp));
  });
}

// =================================================================
// Rounding functions
// Reference: numpy-reference/reference/generated/numpy.floor.html (etc.)
// =================================================================

/** @brief Return the floor of the input, element-wise.
 *
 * The floor is the largest integer <= x.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with floor(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto floor(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::floor(v); });
}

/** @brief Return the ceiling of the input, element-wise.
 *
 * The ceiling is the smallest integer >= x.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with ceil(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto ceil(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::ceil(v); });
}

/** @brief Return the truncated value of the input, element-wise.
 *
 * Truncation rounds toward zero.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with trunc(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto trunc(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::trunc(v); });
}

/** @brief Round to nearest integer, element-wise.
 *
 * Uses half-to-even rounding (banker's rounding), matching
 * numpy.round semantics.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with rint(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto rint(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::rint(v); });
}

// =================================================================
// Arithmetic functions
// Reference: numpy-reference/reference/generated/numpy.absolute.html (etc.)
// =================================================================

/** @brief Calculate the absolute value element-wise.
 *
 * For complex types, returns the magnitude. For real types,
 * returns the absolute value.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with abs(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto absolute(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::abs(v); });
}

/** @brief Alias for absolute().
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with abs(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto abs(const ndarray<T> &x) -> ndarray<T> {
  return absolute(x);
}

/** @brief Alias for absolute().
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with abs(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto fabs(const ndarray<T> &x) -> ndarray<T> {
  return absolute(x);
}

/** @brief Returns element-wise indication of the sign.
 *
 * sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with sign(x[i]).
 */
NP_API template <typename T> NP_NODISCARD auto sign(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) {
    if (v > T{0})
      return T{1};
    if (v < T{0})
      return T{-1};
    return T{0};
  });
}

/** @brief Element-wise maximum of array elements.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>> with max(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto maximum(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::max(a, b));
  });
}

/** @brief Element-wise minimum of array elements.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>> with min(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto minimum(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::min(a, b));
  });
}

/** @brief Element-wise maximum, propagating NaNs.
 *
 * Unlike maximum(), if either element is NaN, the result is NaN.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>>.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto fmax(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmax(a, b));
  });
}

/** @brief Element-wise minimum, propagating NaNs.
 *
 * Unlike minimum(), if either element is NaN, the result is NaN.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>>.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto fmin(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmin(a, b));
  });
}

/** @brief Return the element-wise remainder of division.
 *
 * Uses std::fmod for floating-point and the C % operator for
 * integral types. The result has the same sign as the divisor.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Dividend array.
 * @param x2  Divisor array.
 * @return    ndarray<std::common_type_t<T, U>> with x1[i] % x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto fmod(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmod(a, b));
  });
}

/** @brief Return element-wise remainder of division.
 *
 * Uses std::remainder (IEEE 754 remainder), which rounds to
 * the nearest integer rather than truncating toward zero.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Dividend array.
 * @param x2  Divisor array.
 * @return    ndarray<std::common_type_t<T, U>> with remainder(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto remainder(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::ufunc_binary(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::remainder(a, b));
  });
}

/** @brief Alias for remainder().
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Dividend array.
 * @param x2  Divisor array.
 * @return    ndarray<std::common_type_t<T, U>> with remainder(x1[i], x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto mod(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  return remainder(x1, x2);
}

/** @brief Return the reciprocal of the argument, element-wise.
 *
 * Computes 1/x[i] for each element. Division by zero
 * produces inf or -inf (for non-zero numerator) or NaN
 * (for 0/0).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with 1/x[i].
 */
NP_API template <typename T> NP_NODISCARD auto reciprocal(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return T{1} / v; });
}

/** @brief Numerical positive, element-wise.
 *
 * Returns a copy of the input array.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    Copy of x.
 */
NP_API template <typename T> NP_NODISCARD auto positive(const ndarray<T> &x) -> ndarray<T> {
  return x.copy();
}

/** @brief Numerical negative, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with -x[i].
 */
NP_API template <typename T> NP_NODISCARD auto negative(const ndarray<T> &x) -> ndarray<T> {
  return -x;
}

// =================================================================
// Miscellaneous
// Reference: numpy-reference/reference/generated/numpy.clip.html (etc.)
// =================================================================

/** @brief Clip values to [a_min, a_max].
 *
 * Values below a_min are set to a_min; values above a_max
 * are set to a_max. Elements within the range are unchanged.
 *
 * @tparam T  Element type.
 * @param x       Input array.
 * @param a_min   Lower bound.
 * @param a_max   Upper bound.
 * @return        ndarray<T> with clipped values.
 */
NP_API template <typename T>
NP_NODISCARD auto clip(const ndarray<T> &x, const T &a_min, const T &a_max) -> ndarray<T> {
  return x.clip(a_min, a_max);
}

/** @brief Replace NaN with zero and infinity with large finite numbers.
 *
 * @tparam T       Element type.
 * @param x        Input array.
 * @param nan_val  Replacement for NaN (default: T{0}).
 * @param posinf_val Replacement for +inf (default: max finite value).
 * @param neginf_val Replacement for -inf (default: lowest finite value).
 * @return         ndarray<T> with replaced values.
 */
NP_API template <typename T>
NP_NODISCARD auto nan_to_num(const ndarray<T> &x, const T &nan_val = T{0},
                const T &posinf_val = std::numeric_limits<T>::max(),
                const T &neginf_val = std::numeric_limits<T>::lowest())
    -> ndarray<T> {
  return detail::ufunc_unary(x, [=](const T &v) {
    if (std::isnan(v))
      return nan_val;
    if (std::isinf(v)) {
      return v > T{0} ? posinf_val : neginf_val;
    }
    return v;
  });
}

/** @brief Return (x1 * x2 + x3) element-wise.
 *
 * Fused multiply-add with broadcasting. All three arrays
 * must be broadcast-compatible.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @tparam V  Element type of x3.
 * @param x1  First multiplicand.
 * @param x2  Second multiplicand.
 * @param x3  Addend.
 * @return    ndarray<std::common_type_t<T, U, V>>.
 */
NP_API template <typename T, typename U, typename V>
NP_NODISCARD auto fma(const ndarray<T> &x1, const ndarray<U> &x2, const ndarray<V> &x3)
    -> ndarray<std::common_type_t<T, U, V>> {
  using R = std::common_type_t<T, U, V>;
  // Broadcast all three arrays
  auto temp = detail::ufunc_binary(
      x1, x2, [](const T &a, const U &b) { return static_cast<R>(a * b); });
  return detail::ufunc_binary(
      temp, x3, [](const R &a, const V &b) { return a + static_cast<R>(b); });
}

} // namespace np

#endif // NP_MATH_HPP
