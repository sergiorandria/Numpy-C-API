/**
 * @file math.hpp
 * @brief Element-wise mathematical functions (NumPy ufuncs).
 *
 * Provides broadcasting-aware wrappers around standard math functions:
 *   Trigonometric: sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot
 *   Angular conversion: degrees, radians, rad2deg, deg2rad
 *   Hyperbolic: sinh, cosh, tanh, arcsinh, arccosh, arctanh
 *   Exponential/Logarithmic: exp, expm1, exp2, log, log10, log2, log1p,
 *                            sqrt, cbrt, power, logaddexp, logaddexp2
 *   Rounding: floor, ceil, trunc, rint, round, around
 *   Arithmetic: absolute, sign, maximum, minimum, fmod, fmax, fmin,
 *               copysign, divide, true_divide, floor_divide, reciprocal,
 *               positive, negative
 *   Misc: square, nan_to_num, clip, fma
 *
 * All functions return C-contiguous arrays with row-major strides.
 * Binary operations broadcast shapes according to NumPy rules
 * (see numpy-reference/user/basics.broadcasting.html). Every unary,
 * binary and ternary ufunc also provides an `out` overload that writes
 * into a caller-provided destination array instead of allocating.
 *
 * Reference: numpy-reference/reference/routines.math.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MATH_HPP
#define NP_MATH_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"
#include "simd.hpp"

namespace np {

namespace detail {

// =================================================================
// Type constraints
// =================================================================

/** @brief Accepts arithmetic types and std::complex specializations. */
template <typename T>
concept Numeric = std::is_arithmetic_v<T> || is_complex_v<T>;

/** @brief Accepts floating-point types. */
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/** @brief Accepts integral and floating-point types (no complex). */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// =================================================================
// Element-wise helpers
// =================================================================

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

/** @brief Apply unary function writing into a pre-allocated output.
 *
 * The output array must have exactly the same shape as `arr`.
 *
 * @tparam T  Element type of the input array.
 * @tparam Fn  Unary callable of the form `R(T)`.
 * @param arr  Input array.
 * @param out  Destination array (same shape as `arr`).
 * @param fn   Unary function applied to each element.
 * @return     Reference to `out`, now filled with the results.
 * @throws     std::invalid_argument if `out.shape` differs from `arr.shape`.
 */
NP_API template <typename T, typename Fn>
auto ufunc_unary_into(const ndarray<T> &arr, ndarray<T> &out, Fn &&fn)
    -> ndarray<T> & {
  if (out.shape != arr.shape) {
    throw std::invalid_argument("out: shape does not match input");
  }
  auto it_in = arr.begin();
  auto it_out = out.begin();
  for (; it_in != arr.end(); ++it_in, ++it_out) {
    *it_out = fn(*it_in);
  }
  return out;
}

/** @brief Apply binary function element-wise with broadcasting,
 *         writing into a pre-allocated output.
 *
 * Mirrors np::detail::elementwise but writes into `out` instead of
 * allocating. The output shape must equal the broadcast shape.
 *
 * @tparam T  Element type of the left-hand side array.
 * @tparam U  Element type of the right-hand side array.
 * @tparam R  Element type of the output array.
 * @tparam Fn Binary callable of the form `R(T, U)`.
 * @param a   Left-hand side array.
 * @param b   Right-hand side array.
 * @param out Destination array (must match the broadcast shape).
 * @param fn  Binary function applied to each pair of elements.
 * @return    Reference to `out`.
 * @throws    std::invalid_argument if shapes cannot be broadcast
 *            or if `out.shape` differs from the broadcast shape.
 */
NP_API template <typename T, typename U, typename R, typename Fn>
auto elementwise_into(const ndarray<T> &a, const ndarray<U> &b, ndarray<R> &out,
                      Fn &&fn) -> ndarray<R> & {
  const std::vector<int> out_shape = broadcast_shapes(a.shape, b.shape);
  if (out.shape != out_shape) {
    throw std::invalid_argument("out: shape does not match broadcast result");
  }

  const int nr = static_cast<int>(out_shape.size());
  const int shift_a = nr - static_cast<int>(a.shape.size());
  const int shift_b = nr - static_cast<int>(b.shape.size());

  std::vector<std::size_t> adj_a(nr), adj_b(nr);
  for (int d = 0; d < nr; ++d) {
    const int ka = d - shift_a;
    const int kb = d - shift_b;
    adj_a[d] = (ka < 0 || a.shape[ka] == 1) ? 0 : a.strides[ka];
    adj_b[d] = (kb < 0 || b.shape[kb] == 1) ? 0 : b.strides[kb];
  }

  Odometer od(out_shape);
  while (!od.done()) {
    const auto &idx = od.idx();
    std::size_t fa = a.offset, fb = b.offset, fo = out.offset;
    for (int d = 0; d < nr; ++d) {
      fa += idx[d] * adj_a[d];
      fb += idx[d] * adj_b[d];
      fo += idx[d] * out.strides[d];
    }
    out.data()[fo] = fn(a.data()[fa], b.data()[fb]);
    od.advance();
  }
  return out;
}

/** @brief Apply ternary function element-wise with broadcasting.
 *
 * Computes the broadcast shape of all three inputs, then applies
 * `fn(a[i], b[i], c[i])` to every logical element.
 *
 * @tparam T  Element type of the first array.
 * @tparam U  Element type of the second array.
 * @tparam V  Element type of the third array.
 * @tparam Fn Ternary callable of the form `R(T, U, V)`.
 * @param a   First array.
 * @param b   Second array.
 * @param c   Third array.
 * @param fn  Ternary function applied to each triple of elements.
 * @return    ndarray<R> with the broadcast shape.
 * @throws    std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U, typename V, typename Fn>
NP_NODISCARD auto ufunc_ternary(const ndarray<T> &a, const ndarray<U> &b,
                                const ndarray<V> &c, Fn &&fn) {
  using OutT = std::invoke_result_t<Fn, T, U, V>;
  const std::vector<int> ab_shape = broadcast_shapes(a.shape, b.shape);
  const std::vector<int> out_shape = broadcast_shapes(ab_shape, c.shape);
  ndarray<OutT> out(out_shape);

  const int nr = static_cast<int>(out_shape.size());
  const int shift_a = nr - static_cast<int>(a.shape.size());
  const int shift_b = nr - static_cast<int>(b.shape.size());
  const int shift_c = nr - static_cast<int>(c.shape.size());

  std::vector<std::size_t> adj_a(nr), adj_b(nr), adj_c(nr);
  for (int d = 0; d < nr; ++d) {
    const int ka = d - shift_a;
    const int kb = d - shift_b;
    const int kc = d - shift_c;
    adj_a[d] = (ka < 0 || a.shape[ka] == 1) ? 0 : a.strides[ka];
    adj_b[d] = (kb < 0 || b.shape[kb] == 1) ? 0 : b.strides[kb];
    adj_c[d] = (kc < 0 || c.shape[kc] == 1) ? 0 : c.strides[kc];
  }

  Odometer od(out_shape);
  while (!od.done()) {
    const auto &idx = od.idx();
    std::size_t fa = a.offset, fb = b.offset, fc = c.offset, fo = 0;
    for (int d = 0; d < nr; ++d) {
      fa += idx[d] * adj_a[d];
      fb += idx[d] * adj_b[d];
      fc += idx[d] * adj_c[d];
      fo += idx[d] * out.strides[d];
    }
    out.data()[fo] = fn(a.data()[fa], b.data()[fb], c.data()[fc]);
    od.advance();
  }
  return out;
}

/** @brief Apply ternary function element-wise, writing into a
 *         pre-allocated output.
 *
 * @tparam T  Element type of the first array.
 * @tparam U  Element type of the second array.
 * @tparam V  Element type of the third array.
 * @tparam R  Element type of the output array.
 * @tparam Fn Ternary callable of the form `R(T, U, V)`.
 * @param a   First array.
 * @param b   Second array.
 * @param c   Third array.
 * @param out Destination array (must match the broadcast shape).
 * @param fn  Ternary function applied to each triple of elements.
 * @return    Reference to `out`.
 * @throws    std::invalid_argument if shapes cannot be broadcast
 *            or if `out.shape` differs from the broadcast shape.
 */
NP_API template <typename T, typename U, typename V, typename R, typename Fn>
auto ufunc_ternary_into(const ndarray<T> &a, const ndarray<U> &b,
                        const ndarray<V> &c, ndarray<R> &out, Fn &&fn)
    -> ndarray<R> & {
  const std::vector<int> ab_shape = broadcast_shapes(a.shape, b.shape);
  const std::vector<int> out_shape = broadcast_shapes(ab_shape, c.shape);
  if (out.shape != out_shape) {
    throw std::invalid_argument("out: shape does not match broadcast result");
  }

  const int nr = static_cast<int>(out_shape.size());
  const int shift_a = nr - static_cast<int>(a.shape.size());
  const int shift_b = nr - static_cast<int>(b.shape.size());
  const int shift_c = nr - static_cast<int>(c.shape.size());

  std::vector<std::size_t> adj_a(nr), adj_b(nr), adj_c(nr);
  for (int d = 0; d < nr; ++d) {
    const int ka = d - shift_a;
    const int kb = d - shift_b;
    const int kc = d - shift_c;
    adj_a[d] = (ka < 0 || a.shape[ka] == 1) ? 0 : a.strides[ka];
    adj_b[d] = (kb < 0 || b.shape[kb] == 1) ? 0 : b.strides[kb];
    adj_c[d] = (kc < 0 || c.shape[kc] == 1) ? 0 : c.strides[kc];
  }

  Odometer od(out_shape);
  while (!od.done()) {
    const auto &idx = od.idx();
    std::size_t fa = a.offset, fb = b.offset, fc = c.offset, fo = out.offset;
    for (int d = 0; d < nr; ++d) {
      fa += idx[d] * adj_a[d];
      fb += idx[d] * adj_b[d];
      fc += idx[d] * adj_c[d];
      fo += idx[d] * out.strides[d];
    }
    out.data()[fo] = fn(a.data()[fa], b.data()[fb], c.data()[fc]);
    od.advance();
  }
  return out;
}

/** @brief Round `v` to `decimals` digits using half-to-even rounding.
 *
 * @tparam T  Floating-point element type.
 * @param v         Input value.
 * @param decimals  Number of decimal places (may be negative).
 * @return          Rounded value.
 */
template <typename T> inline T roundto_elem(T v, int decimals) {
  const T scale = std::pow(T(10), static_cast<T>(decimals));
  return std::rint(v * scale) / scale;
}

} // namespace detail

// =================================================================
// Trigonometric functions
// Reference: numpy-reference/reference/generated/numpy.sin.html (etc.)
// =================================================================

/** @brief Trigonometric sine, element-wise.
 *
 * @tparam T  Element type (floating-point, integral or complex).
 * @param x   Input array.
 * @return    ndarray<T> with sin(x[i]) for each element.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto sin(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sin(v); });
}

/** @brief sin() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto sin(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::sin(v); });
}

/** @brief Trigonometric cosine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with cos(x[i]) for each element.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto cos(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cos(v); });
}

/** @brief cos() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto cos(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::cos(v); });
}

/** @brief Trigonometric tangent, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with tan(x[i]) for each element.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto tan(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::tan(v); });
}

/** @brief tan() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto tan(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::tan(v); });
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto arcsin(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::asin(v); });
}

/** @brief arcsin() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arcsin(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::asin(v); });
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto arccos(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::acos(v); });
}

/** @brief arccos() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arccos(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::acos(v); });
}

/** @brief Inverse tangent, element-wise.
 *
 * Returns values in [-pi/2, pi/2].
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with atan(x[i]) for each element.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto arctan(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::atan(v); });
}

/** @brief arctan() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arctan(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::atan(v); });
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto arctan2(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &y, const U &x) {
    return static_cast<R>(std::atan2(y, x));
  });
}

/** @brief arctan2() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto arctan2(const ndarray<T> &x1, const ndarray<U> &x2,
             ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &y, const U &x) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto hypot(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::hypot(a, b));
  });
}

/** @brief hypot() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto hypot(const ndarray<T> &x1, const ndarray<U> &x2,
           ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
    return static_cast<R>(std::hypot(a, b));
  });
}

/** @brief Convert angles from radians to degrees.
 *
 * Uses std::numbers::pi_v<T> for the conversion constant.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array in radians.
 * @return    ndarray<T> with degrees(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto degrees(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) {
    return static_cast<T>(static_cast<T>(180) / std::numbers::pi_v<T>) * v;
  });
}

/** @brief degrees() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto degrees(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) {
    return static_cast<T>(static_cast<T>(180) / std::numbers::pi_v<T>) * v;
  });
}

/** @brief Convert angles from degrees to radians.
 *
 * Uses std::numbers::pi_v<T> for the conversion constant.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array in degrees.
 * @return    ndarray<T> with radians(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto radians(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) {
    return static_cast<T>(std::numbers::pi_v<T> / static_cast<T>(180)) * v;
  });
}

/** @brief radians() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto radians(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) {
    return static_cast<T>(std::numbers::pi_v<T> / static_cast<T>(180)) * v;
  });
}

/** @brief Alias for degrees(). */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto rad2deg(const ndarray<T> &x) -> ndarray<T> {
  return degrees(x);
}

/** @brief rad2deg() writing into `out`. */
NP_API template <detail::FloatingPoint T>
auto rad2deg(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return degrees(x, out);
}

/** @brief Alias for radians(). */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto deg2rad(const ndarray<T> &x) -> ndarray<T> {
  return radians(x);
}

/** @brief deg2rad() writing into `out`. */
NP_API template <detail::FloatingPoint T>
auto deg2rad(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return radians(x, out);
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto sinh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sinh(v); });
}

/** @brief sinh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto sinh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::sinh(v); });
}

/** @brief Hyperbolic cosine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with cosh(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto cosh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cosh(v); });
}

/** @brief cosh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto cosh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::cosh(v); });
}

/** @brief Hyperbolic tangent, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with tanh(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto tanh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::tanh(v); });
}

/** @brief tanh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto tanh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::tanh(v); });
}

/** @brief Inverse hyperbolic sine, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with asinh(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto arcsinh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::asinh(v); });
}

/** @brief arcsinh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arcsinh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::asinh(v); });
}

/** @brief Inverse hyperbolic cosine, element-wise.
 *
 * Domain: x >= 1. For real inputs < 1, the result is NaN.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with acosh(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto arccosh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::acosh(v); });
}

/** @brief arccosh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arccosh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::acosh(v); });
}

/** @brief Inverse hyperbolic tangent, element-wise.
 *
 * Domain: |x| < 1. For |x| >= 1, the result is NaN or inf.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with atanh(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto arctanh(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::atanh(v); });
}

/** @brief arctanh() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto arctanh(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::atanh(v); });
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto exp(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::exp(v); });
}

/** @brief exp() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto exp(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::exp(v); });
}

/** @brief Calculate exp(x) - 1 for all elements.
 *
 * More accurate than exp(x) - 1 for small x.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with expm1(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto expm1(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::expm1(v); });
}

/** @brief expm1() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto expm1(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::expm1(v); });
}

/** @brief Calculate 2**x for all elements.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with 2^x[i].
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto exp2(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::exp2(v); });
}

/** @brief exp2() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto exp2(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::exp2(v); });
}

/** @brief Natural logarithm, element-wise.
 *
 * For x <= 0, the result is NaN (or -inf for x == 0).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto log(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log(v); });
}

/** @brief log() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto log(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::log(v); });
}

/** @brief Base-10 logarithm, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with log10(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto log10(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log10(v); });
}

/** @brief log10() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto log10(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::log10(v); });
}

/** @brief Base-2 logarithm, element-wise.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with log2(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto log2(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log2(v); });
}

/** @brief log2() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto log2(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::log2(v); });
}

/** @brief Calculate log(1 + x) for all elements.
 *
 * More accurate than log(1 + x) for small x.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with log1p(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto log1p(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::log1p(v); });
}

/** @brief log1p() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto log1p(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::log1p(v); });
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto sqrt(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::sqrt(v); });
}

/** @brief sqrt() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto sqrt(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::sqrt(v); });
}

/** @brief Cube root, element-wise.
 *
 * For real inputs, the real cube root is returned (including
 * for negative inputs).
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with cbrt(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto cbrt(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::cbrt(v); });
}

/** @brief cbrt() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto cbrt(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::cbrt(v); });
}

/** @brief Element-wise square.
 *
 * For contiguous float/double inputs a vectorized SIMD kernel is
 * used (np::simd::mul_vectorized). Generic element types fall back
 * to a scalar loop.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with x[i]^2.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto square(const ndarray<T> &x) -> ndarray<T> {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    // TODO: transcendental ufuncs (sin, exp, log, ...) are NOT
    // vectorized here; they would require a vector math library
    // (SLEEF/SVML). Only multiplication/division have kernels in
    // np::simd, so square/divide are the SIMD fast paths.
    if (x.is_contiguous()) {
      ndarray<T> result(x.shape, x.type);
      if (result.size() > 0) {
        np::simd::mul_vectorized(x.data().data(), x.data().data(),
                                 result.data().data(), result.size());
      }
      return result;
    }
  }
  return detail::ufunc_unary(x, [](const T &v) { return v * v; });
}

/** @brief square() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto square(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (x.is_contiguous() && out.is_contiguous() && out.shape == x.shape) {
      if (out.size() > 0) {
        np::simd::mul_vectorized(x.data().data(), x.data().data(),
                                 out.data().data(), out.size());
      }
      return out;
    }
  }
  return detail::ufunc_unary_into(x, out, [](const T &v) { return v * v; });
}

/** @brief First array elements raised to powers from second array, element-wise.
 *
 * For integral operands with a non-negative exponent the result is
 * exact (computes base^exp exactly, matching operator**); negative
 * integral exponents and non-integral operands fall back to
 * std::pow promoted back to the common type. Complex bases use
 * std::pow in the complex domain.
 *
 * @tparam T  Element type of the base array.
 * @tparam U  Element type of the exponent array.
 * @param x1  Base array.
 * @param x2  Exponent array.
 * @return    ndarray<std::common_type_t<T, U>> with x1[i]^x2[i].
 */
NP_API template <detail::Numeric T, detail::Numeric U>
NP_NODISCARD auto power(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &base, const U &exp) -> R {
    if constexpr (detail::is_complex_v<R>) {
      return std::pow(static_cast<R>(base), static_cast<R>(exp));
    } else {
      return static_cast<R>(detail::power_elem(base, exp));
    }
  });
}

/** @brief power() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Numeric T, detail::Numeric U>
auto power(const ndarray<T> &x1, const ndarray<U> &x2,
           ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out,
                                  [](const T &base, const U &exp) -> R {
    if constexpr (detail::is_complex_v<R>) {
      return std::pow(static_cast<R>(base), static_cast<R>(exp));
    } else {
      return static_cast<R>(detail::power_elem(base, exp));
    }
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
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto floor(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::floor(v); });
}

/** @brief floor() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto floor(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::floor(v); });
}

/** @brief Return the ceiling of the input, element-wise.
 *
 * The ceiling is the smallest integer >= x.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with ceil(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto ceil(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::ceil(v); });
}

/** @brief ceil() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto ceil(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::ceil(v); });
}

/** @brief Return the truncated value of the input, element-wise.
 *
 * Truncation rounds toward zero.
 *
 * @tparam T  Element type (floating-point).
 * @param x   Input array.
 * @return    ndarray<T> with trunc(x[i]).
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto trunc(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::trunc(v); });
}

/** @brief trunc() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto trunc(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::trunc(v); });
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
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto rint(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::rint(v); });
}

/** @brief rint() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::FloatingPoint T>
auto rint(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::rint(v); });
}

/** @brief Round to the given number of decimals, element-wise.
 *
 * Rounds to `decimals` digits with half-to-even (banker's) rounding
 * for floating-point inputs (numpy.round semantics). Integral inputs
 * are returned unchanged; complex inputs round both parts. `decimals`
 * may be negative to round to powers of ten.
 *
 * @tparam T  Element type.
 * @param x         Input array.
 * @param decimals  Number of decimal places (default 0).
 * @return          ndarray<T> with rounded elements.
 */
NP_API template <detail::Numeric T>
auto round(const ndarray<T> &x, int decimals, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [decimals](const T &v) -> T {
    if constexpr (detail::is_complex_v<T>) {
      return T{detail::roundto_elem(v.real(), decimals),
               detail::roundto_elem(v.imag(), decimals)};
    } else if constexpr (std::is_floating_point_v<T>) {
      return detail::roundto_elem(v, decimals);
    } else {
      return v;
    }
  });
}

/** @brief Round to the given number of decimals, element-wise.
 *
 * @param x  Input array.
 * @param decimals  Number of decimal places (default 0).
 * @return   ndarray<T> with rounded elements.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto round(const ndarray<T> &x, int decimals = 0) -> ndarray<T> {
  ndarray<T> out(x.shape, x.type);
  round(x, decimals, out);
  return out;
}

/** @brief round() writing into `out` with decimals = 0. */
NP_API template <detail::Numeric T>
auto round(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return round(x, 0, out);
}

/** @brief Alias for round(), element-wise.
 *
 * @param x         Input array.
 * @param decimals  Number of decimal places (default 0).
 * @return          ndarray<T> with rounded elements.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto around(const ndarray<T> &x, int decimals = 0) -> ndarray<T> {
  return round(x, decimals);
}

/** @brief around() writing into `out` with decimals = 0. */
NP_API template <detail::Numeric T>
auto around(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return round(x, 0, out);
}

/** @brief around() writing into `out`. */
NP_API template <detail::Numeric T>
auto around(const ndarray<T> &x, int decimals, ndarray<T> &out) -> ndarray<T> & {
  return round(x, decimals, out);
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto absolute(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return std::abs(v); });
}

/** @brief absolute() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto absolute(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return std::abs(v); });
}

/** @brief Alias for absolute(). */
NP_API template <detail::Numeric T>
NP_NODISCARD auto abs(const ndarray<T> &x) -> ndarray<T> {
  return absolute(x);
}

/** @brief abs() writing into `out`. */
NP_API template <detail::Numeric T>
auto abs(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return absolute(x, out);
}

/** @brief Alias for absolute(). */
NP_API template <detail::Numeric T>
NP_NODISCARD auto fabs(const ndarray<T> &x) -> ndarray<T> {
  return absolute(x);
}

/** @brief fabs() writing into `out`. */
NP_API template <detail::Numeric T>
auto fabs(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return absolute(x, out);
}

/** @brief Returns element-wise indication of the sign.
 *
 * sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0. NaN propagates
 * (sign(NaN) == NaN) and complex inputs return the phase unit
 * vector x/|x|.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with sign(x[i]).
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto sign(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) -> T {
    if constexpr (detail::is_complex_v<T>) {
      if (v == T{0}) {
        return T{0};
      }
      return v / std::abs(v);
    } else if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) {
        return v;
      }
      if (v > T{0}) {
        return T{1};
      }
      if (v < T{0}) {
        return T{-1};
      }
      return T{0};
    } else {
      if (v > T{0}) {
        return T{1};
      }
      if (v < T{0}) {
        return T{-1};
      }
      return T{0};
    }
  });
}

/** @brief sign() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto sign(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) -> T {
    if constexpr (detail::is_complex_v<T>) {
      if (v == T{0}) {
        return T{0};
      }
      return v / std::abs(v);
    } else if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) {
        return v;
      }
      if (v > T{0}) {
        return T{1};
      }
      if (v < T{0}) {
        return T{-1};
      }
      return T{0};
    } else {
      if (v > T{0}) {
        return T{1};
      }
      if (v < T{0}) {
        return T{-1};
      }
      return T{0};
    }
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto maximum(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::max(a, b));
  });
}

/** @brief maximum() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto maximum(const ndarray<T> &x1, const ndarray<U> &x2,
             ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto minimum(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::min(a, b));
  });
}

/** @brief minimum() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto minimum(const ndarray<T> &x1, const ndarray<U> &x2,
             ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto fmax(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmax(a, b));
  });
}

/** @brief fmax() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto fmax(const ndarray<T> &x1, const ndarray<U> &x2,
          ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto fmin(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmin(a, b));
  });
}

/** @brief fmin() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto fmin(const ndarray<T> &x1, const ndarray<U> &x2,
          ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto fmod(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::fmod(a, b));
  });
}

/** @brief fmod() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto fmod(const ndarray<T> &x1, const ndarray<U> &x2,
          ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
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
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto remainder(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(std::remainder(a, b));
  });
}

/** @brief remainder() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto remainder(const ndarray<T> &x1, const ndarray<U> &x2,
               ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
    return static_cast<R>(std::remainder(a, b));
  });
}

/** @brief Alias for remainder(). */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto mod(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  return remainder(x1, x2);
}

/** @brief mod() writing into `out`. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto mod(const ndarray<T> &x1, const ndarray<U> &x2,
         ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  return remainder(x1, x2, out);
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
NP_API template <detail::Numeric T>
NP_NODISCARD auto reciprocal(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return T{1} / v; });
}

/** @brief reciprocal() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto reciprocal(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out,
                                  [](const T &v) { return T{1} / v; });
}

/** @brief Numerical positive, element-wise.
 *
 * Returns a copy of the input array.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    Copy of x.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto positive(const ndarray<T> &x) -> ndarray<T> {
  return x.copy();
}

/** @brief positive() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto positive(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) { return v; });
}

/** @brief Numerical negative, element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<T> with -x[i].
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto negative(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) { return -v; });
}

/** @brief negative() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Numeric T>
auto negative(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) { return -v; });
}

/** @brief Element-wise change of sign of x1 to that of x2 (copysign).
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Array whose magnitude is kept.
 * @param x2  Array whose sign is used.
 * @return    ndarray<std::common_type_t<T, U>> with copysign(x1, x2).
 */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto copysign(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(
        std::copysign(static_cast<double>(a), static_cast<double>(b)));
  });
}

/** @brief copysign() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto copysign(const ndarray<T> &x1, const ndarray<U> &x2,
              ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
    return static_cast<R>(
        std::copysign(static_cast<double>(a), static_cast<double>(b)));
  });
}

/** @brief Element-wise logaddexp: log(exp(x1) + exp(x2)).
 *
 * Computed as max(x,y) + log1p(exp(-|x-y|)) for stability. Both
 * +inf and both -inf inputs are preserved exactly.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>>.
 */
NP_API template <detail::FloatingPoint T, detail::FloatingPoint U>
NP_NODISCARD auto logaddexp(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) -> R {
    const R x = static_cast<R>(a);
    const R y = static_cast<R>(b);
    if (std::isinf(x) && x == y) {
      return x;
    }
    const R m = std::max(x, y);
    return m + std::log1p(std::exp(-std::fabs(x - y)));
  });
}

/** @brief logaddexp() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::FloatingPoint T, detail::FloatingPoint U>
auto logaddexp(const ndarray<T> &x1, const ndarray<U> &x2,
               ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) -> R {
    const R x = static_cast<R>(a);
    const R y = static_cast<R>(b);
    if (std::isinf(x) && x == y) {
      return x;
    }
    const R m = std::max(x, y);
    return m + std::log1p(std::exp(-std::fabs(x - y)));
  });
}

/** @brief Element-wise log2(exp2(x1) + exp2(x2)).
 *
 * Computed as max(x,y) + log1p(exp2(-|x-y|))/ln(2) for stability.
 * Both +inf and both -inf inputs are preserved exactly.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<std::common_type_t<T, U>>.
 */
NP_API template <detail::FloatingPoint T, detail::FloatingPoint U>
NP_NODISCARD auto logaddexp2(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  return detail::elementwise(x1, x2, [](const T &a, const U &b) -> R {
    const R x = static_cast<R>(a);
    const R y = static_cast<R>(b);
    if (std::isinf(x) && x == y) {
      return x;
    }
    const R m = std::max(x, y);
    return m + std::log1p(std::exp2(-std::fabs(x - y))) /
                   std::numbers::ln2_v<R>;
  });
}

/** @brief logaddexp2() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::FloatingPoint T, detail::FloatingPoint U>
auto logaddexp2(const ndarray<T> &x1, const ndarray<U> &x2,
                ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) -> R {
    const R x = static_cast<R>(a);
    const R y = static_cast<R>(b);
    if (std::isinf(x) && x == y) {
      return x;
    }
    const R m = std::max(x, y);
    return m + std::log1p(std::exp2(-std::fabs(x - y))) /
                   std::numbers::ln2_v<R>;
  });
}

/** @brief Element-wise true division: x1 / x2.
 *
 * Mirrors the library's operator/ semantics (element-wise a/b
 * promoted to the common type). For same-typed contiguous float/
 * double arrays a vectorized SIMD kernel is used.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Dividend array.
 * @param x2  Divisor array.
 * @return    ndarray<std::common_type_t<T, U>> with x1[i] / x2[i].
 */
NP_API template <detail::Numeric T, detail::Numeric U>
NP_NODISCARD auto divide(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  using R = std::common_type_t<T, U>;
  if constexpr (std::is_same_v<T, U> &&
                (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
    if (x1.is_contiguous() && x2.is_contiguous() && x1.shape == x2.shape) {
      ndarray<R> result(x1.shape, dtype_of<R>);
      if (result.size() > 0) {
        np::simd::div_vectorized(x1.data().data(), x2.data().data(),
                                 result.data().data(), result.size());
      }
      return result;
    }
  }
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return static_cast<R>(a / b);
  });
}

/** @brief divide() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Numeric T, detail::Numeric U>
auto divide(const ndarray<T> &x1, const ndarray<U> &x2,
            ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  using R = std::common_type_t<T, U>;
  if constexpr (std::is_same_v<T, U> &&
                (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
    if (x1.is_contiguous() && x2.is_contiguous() && out.is_contiguous() &&
        out.shape == x1.shape && x1.shape == x2.shape) {
      if (out.size() > 0) {
        np::simd::div_vectorized(x1.data().data(), x2.data().data(),
                                 out.data().data(), out.size());
      }
      return out;
    }
  }
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
    return static_cast<R>(a / b);
  });
}

/** @brief Alias for divide() (element-wise x1 / x2). */
NP_API template <detail::Numeric T, detail::Numeric U>
NP_NODISCARD auto true_divide(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  return divide(x1, x2);
}

/** @brief true_divide() writing into `out`. */
NP_API template <detail::Numeric T, detail::Numeric U>
auto true_divide(const ndarray<T> &x1, const ndarray<U> &x2,
                 ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  return divide(x1, x2, out);
}

/** @brief Element-wise floor division: floor(x1 / x2).
 *
 * Matches numpy.floor_divide semantics (and the library's
 * floordiv()): the floor of the quotient, with the C-flavored
 * behavior adjusted for mismatched signs.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  Dividend array.
 * @param x2  Divisor array.
 * @return    ndarray<std::common_type_t<T, U>> with floor(x1 / x2).
 */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
NP_NODISCARD auto floor_divide(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>> {
  return detail::elementwise(x1, x2, [](const T &a, const U &b) {
    return detail::floored_div(a, b);
  });
}

/** @brief floor_divide() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Arithmetic T, detail::Arithmetic U>
auto floor_divide(const ndarray<T> &x1, const ndarray<U> &x2,
                  ndarray<std::common_type_t<T, U>> &out)
    -> ndarray<std::common_type_t<T, U>> & {
  return detail::elementwise_into(x1, x2, out, [](const T &a, const U &b) {
    return detail::floored_div(a, b);
  });
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
NP_API template <detail::Arithmetic T>
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

/** @brief nan_to_num() writing into `out`. Same shape as `x`.
 * @throws std::invalid_argument if `out.shape` differs from `x.shape`. */
NP_API template <detail::Arithmetic T>
auto nan_to_num(const ndarray<T> &x, ndarray<T> &out, const T &nan_val,
                const T &posinf_val, const T &neginf_val) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [=](const T &v) {
    if (std::isnan(v))
      return nan_val;
    if (std::isinf(v)) {
      return v > T{0} ? posinf_val : neginf_val;
    }
    return v;
  });
}

/** @brief Return (x1 * x2 + x3) element-wise, with a single
 *         fused multiply-add per element when supported.
 *
 * For floating-point common types the CPU fused multiply-add
 * instruction is used (std::fma), so each element is computed in
 * a single rounding step (matching numpy.fma). Integral and
 * complex types fall back to x1*x2 + x3. All three arrays must be
 * broadcast-compatible.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @tparam V  Element type of x3.
 * @param x1  First multiplicand.
 * @param x2  Second multiplicand.
 * @param x3  Addend.
 * @return    ndarray<std::common_type_t<T, U, V>>.
 */
NP_API template <detail::Numeric T, detail::Numeric U, detail::Numeric V>
NP_NODISCARD auto fma(const ndarray<T> &x1, const ndarray<U> &x2,
                      const ndarray<V> &x3)
    -> ndarray<std::common_type_t<T, U, V>> {
  using R = std::common_type_t<T, U, V>;
  return detail::ufunc_ternary(x1, x2, x3, [](const T &a, const U &b,
                                              const V &c) -> R {
    if constexpr (std::is_floating_point_v<R>) {
      return static_cast<R>(
          std::fma(static_cast<R>(a), static_cast<R>(b), static_cast<R>(c)));
    } else {
      return static_cast<R>(a) * static_cast<R>(b) + static_cast<R>(c);
    }
  });
}

/** @brief fma() writing into `out`. Must match broadcast shape.
 * @throws std::invalid_argument if shapes cannot be broadcast
 *         or if `out.shape` differs from the broadcast shape. */
NP_API template <detail::Numeric T, detail::Numeric U, detail::Numeric V>
auto fma(const ndarray<T> &x1, const ndarray<U> &x2, const ndarray<V> &x3,
         ndarray<std::common_type_t<T, U, V>> &out)
    -> ndarray<std::common_type_t<T, U, V>> & {
  using R = std::common_type_t<T, U, V>;
  return detail::ufunc_ternary_into(x1, x2, x3, out, [](const T &a,
                                                        const U &b,
                                                        const V &c) -> R {
    if constexpr (std::is_floating_point_v<R>) {
      return static_cast<R>(
          std::fma(static_cast<R>(a), static_cast<R>(b), static_cast<R>(c)));
    } else {
      return static_cast<R>(a) * static_cast<R>(b) + static_cast<R>(c);
    }
  });
}

// =================================================================
// Additional math utilities (sinc, unwrap, angle, fix, ediff1d)
// Reference: numpy-reference/reference/routines.math.html
// =================================================================

/** @brief Sinc function: sin(pi*x)/(pi*x), with sinc(0)=1.
 *
 * @tparam T Floating-point element type.
 * @param x Input array.
 * @return ndarray<T> with sinc values.
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto sinc(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) -> T {
    if (v == T{0}) return T{1};
    T pi_v = std::numbers::pi_v<T>;
    T arg = pi_v * v;
    return std::sin(arg) / arg;
  });
}

/** @brief sinc() writing into `out`. Same shape as `x`. */
NP_API template <detail::FloatingPoint T>
auto sinc(const ndarray<T> &x, ndarray<T> &out) -> ndarray<T> & {
  return detail::ufunc_unary_into(x, out, [](const T &v) -> T {
    if (v == T{0}) return T{1};
    T pi_v = std::numbers::pi_v<T>;
    T arg = pi_v * v;
    return std::sin(arg) / arg;
  });
}

/** @brief Unwrap phase (1-D).
 *
 * Unwraps `p` by adding multiples of 2*pi when the jump exceeds `discont`
 * (default pi). Mirrors `np.unwrap`.
 *
 * @tparam T Floating-point type.
 * @param p Phase array (1-D).
 * @param discont Maximum discontinuity between values.
 * @param axis Axis along which to unwrap (only -1/0 supported for 1-D).
 * @return ndarray<T> unwrapped.
 */
NP_API template <detail::FloatingPoint T>
NP_NODISCARD auto unwrap(const ndarray<T> &p, T discont = std::numbers::pi_v<T>,
                         int axis = -1) -> ndarray<T> {
  if (p.ndim() != 1) {
    throw std::invalid_argument("unwrap: only 1-D supported in this implementation");
  }
  (void)axis;
  ndarray<T> out(p.shape);
  if (p.size() == 0) return out;
  out.data()[0] = p.data()[p._flat_logical(0)];
  T two_pi = T{2} * std::numbers::pi_v<T>;
  for (std::size_t i = 1; i < p.size(); ++i) {
    T d = p.data()[p._flat_logical(i)] - p.data()[p._flat_logical(i - 1)];
    // numpy: d = d - round(d/2pi)*2pi  when |d|>discont else d
    if (std::abs(d) > discont) {
      d -= std::round(d / two_pi) * two_pi;
    }
    out.data()[i] = out.data()[i - 1] + d;
  }
  return out;
}

/** @brief Element-wise angle (phase) of complex array.
 *
 * For real input returns 0 for >=0 and pi for <0.
 *
 * @tparam T Element type (arithmetic or complex).
 * @param x Input array.
 * @return ndarray<double> phases in radians.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto angle(const ndarray<T> &x) -> ndarray<double> {
  ndarray<double> out(x.shape);
  for (std::size_t i = 0; i < x.size(); ++i) {
    T v = x.data()[x._flat_logical(i)];
    double ph;
    if constexpr (detail::is_complex_v<T>) {
      ph = std::arg(v);
    } else {
      ph = v < T{0} ? std::numbers::pi : 0.0;
      if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(static_cast<double>(v))) ph = std::numeric_limits<double>::quiet_NaN();
      }
    }
    out.data()[i] = ph;
  }
  return out;
}

/** @brief Fix: round to nearest integer towards zero.
 *
 * Mirrors `np.fix`. Equivalent to trunc for floating types, identity
 * for integers.
 *
 * @tparam T Numeric type.
 * @param x Input array.
 * @return ndarray<T> with fixed values.
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto fix(const ndarray<T> &x) -> ndarray<T> {
  return detail::ufunc_unary(x, [](const T &v) -> T {
    if constexpr (std::is_floating_point_v<T>) {
      return std::trunc(v);
    } else if constexpr (detail::is_complex_v<T>) {
      return T{std::trunc(v.real()), std::trunc(v.imag())};
    } else {
      return v;
    }
  });
}

/** @brief Differences between consecutive elements (1-D).
 *
 * Mirrors `np.ediff1d` for 1-D arrays (without to_begin/to_end).
 *
 * @tparam T Element type.
 * @param ary Input 1-D array.
 * @return ndarray<T> of size N-1 with ary[1:]-ary[:-1].
 */
NP_API template <detail::Numeric T>
NP_NODISCARD auto ediff1d(const ndarray<T> &ary) -> ndarray<T> {
  if (ary.ndim() != 1) {
    throw std::invalid_argument("ediff1d: input must be 1-D");
  }
  if (ary.size() <= 1) {
    return ndarray<T>(std::vector<int>{0});
  }
  ndarray<T> out(std::vector<int>{static_cast<int>(ary.size() - 1)});
  for (std::size_t i = 0; i + 1 < ary.size(); ++i) {
    out.data()[i] = ary.data()[ary._flat_logical(i + 1)] - ary.data()[ary._flat_logical(i)];
  }
  return out;
}

} // namespace np

#endif // NP_MATH_HPP