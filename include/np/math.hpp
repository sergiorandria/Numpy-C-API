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
 * Reference: numpy-reference/reference/routines.math.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MATH_HPP
#define NP_MATH_HPP

#include <cmath>
#include <algorithm>
#include <type_traits>

#include "ndarray.hpp"

namespace np {

    // =================================================================
    // Internal helper for unary element-wise operations
    // =================================================================

    namespace detail {
        /**
         * @brief Apply unary function element-wise with broadcasting.
         */
        template <typename T, typename Fn>
        auto ufunc_unary(const Ndarray<T>& arr, Fn&& fn) -> Ndarray<T> {
            Ndarray<T> result(arr.shape, arr.type);
            auto it_in = arr.begin();
            auto it_out = result.begin();
            for (; it_in != arr.end(); ++it_in, ++it_out) {
                *it_out = fn(*it_in);
            }
            return result;
        }

        /**
         * @brief Apply binary function element-wise with broadcasting.
         */
        template <typename T, typename U, typename Fn>
        auto ufunc_binary(const Ndarray<T>& lhs, const Ndarray<U>& rhs, Fn&& fn)
            -> Ndarray<std::common_type_t<T, U>> {
            using R = std::common_type_t<T, U>;
            
            // Broadcast shapes
            const auto out_shape = detail::broadcast_shapes(lhs.shape, rhs.shape);
            Ndarray<R> result(out_shape, dtype_of<R>);
            
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

    /** @brief Trigonometric sine, element-wise. */
    template <typename T>
    auto sin(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::sin(v); });
    }

    /** @brief Trigonometric cosine, element-wise. */
    template <typename T>
    auto cos(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::cos(v); });
    }

    /** @brief Trigonometric tangent, element-wise. */
    template <typename T>
    auto tan(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::tan(v); });
    }

    /** @brief Inverse sine, element-wise. */
    template <typename T>
    auto arcsin(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::asin(v); });
    }

    /** @brief Inverse cosine, element-wise. */
    template <typename T>
    auto arccos(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::acos(v); });
    }

    /** @brief Inverse tangent, element-wise. */
    template <typename T>
    auto arctan(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::atan(v); });
    }

    /** @brief Element-wise arc tangent of x1/x2 choosing the quadrant correctly. */
    template <typename T, typename U>
    auto arctan2(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& y, const U& x) {
            return static_cast<R>(std::atan2(y, x));
        });
    }

    /** @brief Given sides of a right triangle, return its hypotenuse. */
    template <typename T, typename U>
    auto hypot(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::hypot(a, b));
        });
    }

    /** @brief Convert angles from radians to degrees. */
    template <typename T>
    auto degrees(const Ndarray<T>& x) -> Ndarray<T> {
        constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
        return detail::ufunc_unary(x, [](const T& v) {
            return static_cast<T>(v * rad_to_deg);
        });
    }

    /** @brief Convert angles from degrees to radians. */
    template <typename T>
    auto radians(const Ndarray<T>& x) -> Ndarray<T> {
        constexpr double deg_to_rad = 3.14159265358979323846 / 180.0;
        return detail::ufunc_unary(x, [](const T& v) {
            return static_cast<T>(v * deg_to_rad);
        });
    }

    /** @brief Alias for degrees(). */
    template <typename T>
    auto rad2deg(const Ndarray<T>& x) -> Ndarray<T> {
        return degrees(x);
    }

    /** @brief Alias for radians(). */
    template <typename T>
    auto deg2rad(const Ndarray<T>& x) -> Ndarray<T> {
        return radians(x);
    }

    // =================================================================
    // Hyperbolic functions
    // Reference: numpy-reference/reference/generated/numpy.sinh.html (etc.)
    // =================================================================

    /** @brief Hyperbolic sine, element-wise. */
    template <typename T>
    auto sinh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::sinh(v); });
    }

    /** @brief Hyperbolic cosine, element-wise. */
    template <typename T>
    auto cosh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::cosh(v); });
    }

    /** @brief Hyperbolic tangent, element-wise. */
    template <typename T>
    auto tanh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::tanh(v); });
    }

    /** @brief Inverse hyperbolic sine, element-wise. */
    template <typename T>
    auto arcsinh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::asinh(v); });
    }

    /** @brief Inverse hyperbolic cosine, element-wise. */
    template <typename T>
    auto arccosh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::acosh(v); });
    }

    /** @brief Inverse hyperbolic tangent, element-wise. */
    template <typename T>
    auto arctanh(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::atanh(v); });
    }

    // =================================================================
    // Exponential and logarithmic functions
    // Reference: numpy-reference/reference/generated/numpy.exp.html (etc.)
    // =================================================================

    /** @brief Calculate the exponential of all elements. */
    template <typename T>
    auto exp(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::exp(v); });
    }

    /** @brief Calculate exp(x) - 1 for all elements. */
    template <typename T>
    auto expm1(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::expm1(v); });
    }

    /** @brief Calculate 2**x for all elements. */
    template <typename T>
    auto exp2(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::exp2(v); });
    }

    /** @brief Natural logarithm, element-wise. */
    template <typename T>
    auto log(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::log(v); });
    }

    /** @brief Base-10 logarithm, element-wise. */
    template <typename T>
    auto log10(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::log10(v); });
    }

    /** @brief Base-2 logarithm, element-wise. */
    template <typename T>
    auto log2(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::log2(v); });
    }

    /** @brief Calculate log(1 + x) for all elements. */
    template <typename T>
    auto log1p(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::log1p(v); });
    }

    /** @brief Non-negative square root, element-wise. */
    template <typename T>
    auto sqrt(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::sqrt(v); });
    }

    /** @brief Cube root, element-wise. */
    template <typename T>
    auto cbrt(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::cbrt(v); });
    }

    /** @brief Element-wise square. */
    template <typename T>
    auto square(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return v * v; });
    }

    /** @brief First array elements raised to powers from second array, element-wise. */
    template <typename T, typename U>
    auto power(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& base, const U& exp) {
            return static_cast<R>(std::pow(base, exp));
        });
    }

    // =================================================================
    // Rounding functions
    // Reference: numpy-reference/reference/generated/numpy.floor.html (etc.)
    // =================================================================

    /** @brief Return the floor of the input, element-wise. */
    template <typename T>
    auto floor(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::floor(v); });
    }

    /** @brief Return the ceiling of the input, element-wise. */
    template <typename T>
    auto ceil(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::ceil(v); });
    }

    /** @brief Return the truncated value of the input, element-wise. */
    template <typename T>
    auto trunc(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::trunc(v); });
    }

    /** @brief Round to nearest integer, element-wise. */
    template <typename T>
    auto rint(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::rint(v); });
    }

    // =================================================================
    // Arithmetic functions
    // Reference: numpy-reference/reference/generated/numpy.absolute.html (etc.)
    // =================================================================

    /** @brief Calculate the absolute value element-wise. */
    template <typename T>
    auto absolute(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return std::abs(v); });
    }

    /** @brief Alias for absolute(). */
    template <typename T>
    auto abs(const Ndarray<T>& x) -> Ndarray<T> {
        return absolute(x);
    }

    /** @brief Alias for absolute(). */
    template <typename T>
    auto fabs(const Ndarray<T>& x) -> Ndarray<T> {
        return absolute(x);
    }

    /** @brief Returns element-wise indication of the sign. */
    template <typename T>
    auto sign(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) {
            if (v > T{0}) return T{1};
            if (v < T{0}) return T{-1};
            return T{0};
        });
    }

    /** @brief Element-wise maximum of array elements. */
    template <typename T, typename U>
    auto maximum(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::max(a, b));
        });
    }

    /** @brief Element-wise minimum of array elements. */
    template <typename T, typename U>
    auto minimum(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::min(a, b));
        });
    }

    /** @brief Element-wise maximum, propagating NaNs. */
    template <typename T, typename U>
    auto fmax(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::fmax(a, b));
        });
    }

    /** @brief Element-wise minimum, propagating NaNs. */
    template <typename T, typename U>
    auto fmin(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::fmin(a, b));
        });
    }

    /** @brief Return the element-wise remainder of division. */
    template <typename T, typename U>
    auto fmod(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::fmod(a, b));
        });
    }

    /** @brief Return element-wise remainder of division. */
    template <typename T, typename U>
    auto remainder(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        return detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(std::remainder(a, b));
        });
    }

    /** @brief Alias for remainder(). */
    template <typename T, typename U>
    auto mod(const Ndarray<T>& x1, const Ndarray<U>& x2)
        -> Ndarray<std::common_type_t<T, U>> {
        return remainder(x1, x2);
    }

    /** @brief Return the reciprocal of the argument, element-wise. */
    template <typename T>
    auto reciprocal(const Ndarray<T>& x) -> Ndarray<T> {
        return detail::ufunc_unary(x, [](const T& v) { return T{1} / v; });
    }

    /** @brief Numerical positive, element-wise. */
    template <typename T>
    auto positive(const Ndarray<T>& x) -> Ndarray<T> {
        return x.copy();
    }

    /** @brief Numerical negative, element-wise. */
    template <typename T>
    auto negative(const Ndarray<T>& x) -> Ndarray<T> {
        return -x;
    }

    // =================================================================
    // Miscellaneous
    // Reference: numpy-reference/reference/generated/numpy.clip.html (etc.)
    // =================================================================

    /** @brief Clip values to [a_min, a_max]. */
    template <typename T>
    auto clip(const Ndarray<T>& x, const T& a_min, const T& a_max)
        -> Ndarray<T> {
        return x.clip(a_min, a_max);
    }

    /** @brief Replace NaN with zero and infinity with large finite numbers. */
    template <typename T>
    auto nan_to_num(const Ndarray<T>& x,
                    const T& nan_val = T{0},
                    const T& posinf_val = std::numeric_limits<T>::max(),
                    const T& neginf_val = std::numeric_limits<T>::lowest())
        -> Ndarray<T> {
        return detail::ufunc_unary(x, [=](const T& v) {
            if (std::isnan(v)) return nan_val;
            if (std::isinf(v)) {
                return v > T{0} ? posinf_val : neginf_val;
            }
            return v;
        });
    }

    /** @brief Return (x1 * x2 + x3) element-wise. */
    template <typename T, typename U, typename V>
    auto fma(const Ndarray<T>& x1, const Ndarray<U>& x2, const Ndarray<V>& x3)
        -> Ndarray<std::common_type_t<T, U, V>> {
        using R = std::common_type_t<T, U, V>;
        // Broadcast all three arrays
        auto temp = detail::ufunc_binary(x1, x2, [](const T& a, const U& b) {
            return static_cast<R>(a * b);
        });
        return detail::ufunc_binary(temp, x3, [](const R& a, const V& b) {
            return a + static_cast<R>(b);
        });
    }

} // namespace np

#endif // NP_MATH_HPP
