/**
 * @file creation_fixed.hpp
 * @brief Compile-time-first array creation for the fixed-shape path.
 *
 * The NumPy shape argument becomes the template parameter list, so the
 * extent checks are static:
 *   np::zeros<2, 3>()          -> ndarray<double, 2, 3>
 *   np::ones<4, int>()         -> ndarray<int, 4>
 *   np::full<2, 2>(7)          -> ndarray<int, 2, 2>
 *   np::eye<3>()               -> ndarray<double, 3, 3>
 *   np::eye<3, 4, 1>()         -> ndarray<double, 3, 4> with k = 1
 *   np::identity<3, int>()     -> ndarray<int, 3, 3>
 *   np::arange<6>(1, 7, 2)     -> {1, 3, 5, 7, 9, 11}
 *   np::linspace<5>(0.0, 1.0)  -> {0.0, 0.25, 0.5, 0.75, 1.0}
 *
 * Signature ground truth: numpy-reference/reference/generated/
 *   numpy.zeros.html, numpy.ones.html, numpy.full.html, numpy.eye.html,
 *   numpy.identity.html, numpy.arange.html, numpy.linspace.html
 */
#ifndef NP_CREATION_FIXED_HPP
#define NP_CREATION_FIXED_HPP

#include <cstddef>
#include <type_traits>

#include "ndarray_fixed.hpp"

namespace np {

    /**
     * @brief Array of zeros with the given compile-time shape. Two spellings:
     *        np::zeros<2, 3>() (double) or np::zeros<int, 2, 3>().
     * Reference: numpy-reference/reference/generated/numpy.zeros.html
     */
    template <int... E>
    constexpr ndarray<double, E...> zeros() {
        return ndarray<double, E...>{};
    }

    template <typename T, int... E>
    constexpr ndarray<T, E...> zeros() {
        return ndarray<T, E...>{};
    }

    /**
     * @brief Array of ones with the given compile-time shape.
     * Reference: numpy-reference/reference/generated/numpy.ones.html
     */
    template <int... E>
    constexpr ndarray<double, E...> ones() {
        ndarray<double, E...> out{};
        out.fill(1.0);
        return out;
    }

    template <typename T, int... E>
    constexpr ndarray<T, E...> ones() {
        ndarray<T, E...> out{};
        out.fill(T{1});
        return out;
    }

    /**
     * @brief Array filled with a constant value.
     * Reference: numpy-reference/reference/generated/numpy.full.html
     */
    template <int... E, typename T>
    constexpr ndarray<T, E...> full(const T& fill_value) {
        ndarray<T, E...> out{};
        out.fill(fill_value);
        return out;
    }

    template <typename T, int... E>
    constexpr ndarray<T, E...> full(const T& fill_value) {
        ndarray<T, E...> out{};
        out.fill(fill_value);
        return out;
    }

    /**
     * @brief Identity-like matrix with ones on the k-th diagonal.
     * Reference: numpy-reference/reference/generated/numpy.eye.html
     */
    template <std::size_t N, std::size_t M = N, int k = 0, typename T = double>
    constexpr ndarray<T, N, M> eye() {
        ndarray<T, N, M> out{};
        const std::ptrdiff_t kk = k;
        for (std::size_t i = 0; i < N; ++i) {
            const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + kk;
            if (j >= 0 && j < static_cast<std::ptrdiff_t>(M)) {
                out.m_data[i * M + static_cast<std::size_t>(j)] = T{1};
            }
        }
        return out;
    }

    /**
     * @brief Square identity matrix of size N x N.
     * Reference: numpy-reference/reference/generated/numpy.identity.html
     */
    template <std::size_t N, typename T = double>
    constexpr ndarray<T, N, N> identity() {
        return eye<N, N, 0, T>();
    }

    /**
     * @brief Values 0..N-1 (numpy arange with a compile-time element count).
     * Reference: numpy-reference/reference/generated/numpy.arange.html
     */
    template <std::size_t N, typename T = int>
    constexpr ndarray<T, N> arange() {
        ndarray<T, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = static_cast<T>(i);
        }
        return out;
    }

    /** @brief N values from start (inclusive), step 1. */
    template <std::size_t N, typename T>
    constexpr ndarray<T, N> arange(T start, T stop) {
        (void)stop;
        ndarray<T, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = start + static_cast<T>(i);
        }
        return out;
    }

    /** @brief N values from start (inclusive) with the given step. */
    template <std::size_t N, typename T>
    constexpr ndarray<T, N> arange(T start, T stop, T step) {
        (void)stop;
        ndarray<T, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = start + step * static_cast<T>(i);
        }
        return out;
    }

    /**
     * @brief N evenly spaced values from start to stop (inclusive).
     *        Integer inputs are promoted to double, as in numpy.
     * Reference: numpy-reference/reference/generated/numpy.linspace.html
     */
    template <std::size_t N, bool endpoint = true, typename T>
    constexpr auto linspace(T start, T stop) {
        using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        ndarray<R, N> out{};
        if constexpr (N == 1) {
            out[0] = static_cast<R>(start);
        } else {
            const R delta = endpoint
                                ? (static_cast<R>(stop) - static_cast<R>(start)) /
                                      static_cast<R>(N - 1)
                                : (static_cast<R>(stop) - static_cast<R>(start)) /
                                      static_cast<R>(N);
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = static_cast<R>(start) + delta * static_cast<R>(i);
            }
        }
        return out;
    }

} // namespace np

#endif // NP_CREATION_FIXED_HPP
