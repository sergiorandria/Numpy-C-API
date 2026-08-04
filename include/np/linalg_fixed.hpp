/**
 * @file linalg_fixed.hpp
 * @brief Compile-time-checked linear algebra for the fixed-shape path.
 *
 * Unlike the dynamic np::linalg (which throws std::invalid_argument at
 * runtime), the fixed versions encode the contraction dimension in the
 * types: np::linalg::dot / matmul only accept arrays whose inner dimension
 * matches, so every mismatched call is a compile-time error.
 *
 * Supported combinations (NumPy dot semantics, ndim <= 2):
 *   dot(1-D, 1-D)  -> scalar (R)
 *   dot(1-D, 2-D)  -> ndarray<R, M>    (a . b, contracting b's rows)
 *   dot(2-D, 1-D)  -> ndarray<R, N>
 *   dot(2-D, 2-D)  -> ndarray<R, N, M>
 *   matmul(2-D, 2-D) -> ndarray<R, N, M>
 *
 * Signature ground truth: numpy-reference/reference/generated/
 *   numpy.dot.html, numpy.matmul.html
 */
#ifndef NP_LINALG_FIXED_HPP
#define NP_LINALG_FIXED_HPP

#include <type_traits>

#include "ndarray_fixed.hpp"

namespace np::linalg {

    /** @brief Dot product of two 1-D arrays -> scalar (numpy dot, 1D . 1D). */
    template <typename T, int N, typename U>
    constexpr auto dot(const ndarray<T, N>& a, const ndarray<U, N>& b) {
        using R = std::common_type_t<T, U>;
        R acc{};
        for (int i = 0; i < N; ++i) {
            acc += static_cast<R>(a[i]) * static_cast<R>(b[i]);
        }
        return acc;
    }

    /** @brief Dot product (2-D . 1-D) -> 1-D. */
    template <typename T, int N, int K, typename U>
    constexpr auto dot(const ndarray<T, N, K>& a, const ndarray<U, K>& b) {
        using R = std::common_type_t<T, U>;
        ndarray<R, N> out{};
        for (int i = 0; i < N; ++i) {
            R acc{};
            for (int p = 0; p < K; ++p) {
                acc += static_cast<R>(a(i, p)) * static_cast<R>(b[p]);
            }
            out[i] = acc;
        }
        return out;
    }

    /** @brief Dot product (1-D . 2-D) -> 1-D. */
    template <typename T, int K, typename U, int M>
    constexpr auto dot(const ndarray<T, K>& a, const ndarray<U, K, M>& b) {
        using R = std::common_type_t<T, U>;
        ndarray<R, M> out{};
        for (int j = 0; j < M; ++j) {
            R acc{};
            for (int p = 0; p < K; ++p) {
                acc += static_cast<R>(a[p]) * static_cast<R>(b(p, j));
            }
            out[j] = acc;
        }
        return out;
    }

    /** @brief Dot product (2-D . 2-D) -> 2-D. */
    template <typename T, int N, int K, typename U, int M>
    constexpr auto dot(const ndarray<T, N, K>& a, const ndarray<U, K, M>& b) {
        using R = std::common_type_t<T, U>;
        ndarray<R, N, M> out{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                R acc{};
                for (int p = 0; p < K; ++p) {
                    acc += static_cast<R>(a(i, p)) * static_cast<R>(b(p, j));
                }
                out(i, j) = acc;
            }
        }
        return out;
    }

    /**
     * @brief Matrix multiplication (numpy.matmul, 2-D . 2-D).
     *        The contraction dimension K is shared by both operand types,
     *        so incompatible shapes cannot be expressed.
     * Reference: numpy-reference/reference/generated/numpy.matmul.html
     */
    template <typename T, int N, int K, typename U, int M>
    constexpr auto matmul(const ndarray<T, N, K>& a, const ndarray<U, K, M>& b) {
        return dot(a, b);
    }

} // namespace np::linalg

#endif // NP_LINALG_FIXED_HPP
