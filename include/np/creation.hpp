/**
 * @file creation.hpp
 * @brief Array creation routines (np::zeros, np::ones, np::arange, ...).
 *
 * Mirrors numpy's creation API:
 *   zeros, ones, full, empty, empty_like/zeros_like/ones_like,
 *   arange, linspace, logspace, eye, identity, asarray.
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

#include "ndarray.hpp"

namespace np {

    /** @brief Array of zeros with the given shape. */
    template <typename T = double>
    auto zeros(const std::vector<int>& shape) -> Ndarray<T> {
        return Ndarray<T>(shape, dtype_of<T>, T{0});
    }

    /** @brief Array of ones with the given shape. */
    template <typename T = double>
    auto ones(const std::vector<int>& shape) -> Ndarray<T> {
        return Ndarray<T>(shape, dtype_of<T>, T{1});
    }

    /** @brief Array filled with a constant value. */
    template <typename T>
    auto full(const std::vector<int>& shape, const T& fill_value)
        -> Ndarray<T> {
        return Ndarray<T>(shape, dtype_of<T>, fill_value);
    }

    /**
     * @brief Uninitialized array (values are default-constructed in C++).
     */
    template <typename T = double>
    auto empty(const std::vector<int>& shape) -> Ndarray<T> {
        return Ndarray<T>(shape, dtype_of<T>, T{});
    }

    /** @brief New array with the same shape as `a`. */
    template <typename T>
    auto empty_like(const Ndarray<T>& a) -> Ndarray<T> {
        return Ndarray<T>(a.shape, a.type);
    }

    /** @brief Zeros with the same shape as `a`. */
    template <typename T>
    auto zeros_like(const Ndarray<T>& a) -> Ndarray<T> {
        return Ndarray<T>(a.shape, a.type, T{0});
    }

    /** @brief Ones with the same shape as `a`. */
    template <typename T>
    auto ones_like(const Ndarray<T>& a) -> Ndarray<T> {
        return Ndarray<T>(a.shape, a.type, T{1});
    }

    /** @brief Filled with `fill_value` using the shape and dtype of `a`. */
    template <typename T>
    auto full_like(const Ndarray<T>& a, const T& fill_value) -> Ndarray<T> {
        return Ndarray<T>(a.shape, a.type, fill_value);
    }

    /** @brief Values evenly spaced from start (inclusive) to stop (exclusive). */
    template <typename T>
    auto arange(T start, T stop, T step = T{1}) -> Ndarray<T> {
        if (step == T{0}) {
            throw std::invalid_argument("arange step cannot be zero");
        }
        std::vector<T> out;
        if (step > T{0}) {
            if (stop <= start) {
                return Ndarray<T>({0}, dtype_of<T>, T{});
            }
            const std::size_t n = static_cast<std::size_t>(
                std::ceil((static_cast<double>(stop) -
                           static_cast<double>(start)) /
                          static_cast<double>(step)));
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                out.push_back(start + step * static_cast<T>(i));
            }
        } else {
            if (stop >= start) {
                return Ndarray<T>({0}, dtype_of<T>, T{});
            }
            const std::size_t n = static_cast<std::size_t>(
                std::ceil((static_cast<double>(stop) -
                           static_cast<double>(start)) /
                          static_cast<double>(step)));
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                out.push_back(start + step * static_cast<T>(i));
            }
        }
        const int n_elems = static_cast<int>(out.size());
        return Ndarray<T>::from_data(std::vector<int>{n_elems},
                                     std::move(out));
    }

    /** @brief Values from 0 to stop (exclusive). */
    template <typename T>
    auto arange(T stop) -> Ndarray<T> {
        return arange(T{0}, stop, T{1});
    }

    /**
     * @brief `num` evenly spaced values from start to stop (inclusive).
     */
    template <typename T>
    auto linspace(T start, T stop, std::size_t num = 50, bool endpoint = true)
        -> Ndarray<std::conditional_t<std::is_floating_point_v<T>, T, double>> {
        using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        if (num == 0) {
            return Ndarray<R>(std::vector<int>{0});
        }
        std::vector<R> out;
        out.reserve(num);
        if (num == 1) {
            out.push_back(static_cast<R>(start));
            return Ndarray<R>::from_data(std::vector<int>{1}, std::move(out));
        }
        const R delta = endpoint
                            ? (static_cast<R>(stop) - static_cast<R>(start)) /
                                  static_cast<R>(num - 1)
                            : (static_cast<R>(stop) - static_cast<R>(start)) /
                                  static_cast<R>(num);
        for (std::size_t i = 0; i < num; ++i) {
            out.push_back(static_cast<R>(start) + delta * static_cast<R>(i));
        }
        return Ndarray<R>::from_data(std::vector<int>{static_cast<int>(num)},
                                     std::move(out));
    }

    /**
     * @brief Logarithmically spaced values from base^start to base^stop.
     */
    template <typename T>
    auto logspace(T start, T stop, std::size_t num = 50, T base = T{10})
        -> Ndarray<double> {
        auto powers = linspace(start, stop, num);
        Ndarray<double> out(std::vector<int>{static_cast<int>(num)});
        for (std::size_t i = 0; i < num; ++i) {
            out.data()[i] = std::pow(static_cast<double>(base),
                                     static_cast<double>(powers.data()[i]));
        }
        return out;
    }

    /**
     * @brief Identity matrix of size n x n.
     */
    template <typename T = double>
    auto eye(std::size_t n, std::size_t m = 0, int k = 0) -> Ndarray<T> {
        if (m == 0) {
            m = n;
        }
        std::vector<int> shape = {static_cast<int>(n), static_cast<int>(m)};
        Ndarray<T> out(shape, dtype_of<T>, T{0});
        const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
        const std::ptrdiff_t cols = static_cast<std::ptrdiff_t>(m);
        for (std::ptrdiff_t i = 0; i < rows; ++i) {
            const std::ptrdiff_t j = i + k;
            if (j >= 0 && j < cols) {
                out.set(std::array<std::size_t, 2>{
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j)},
                        T{1});
            }
        }
        return out;
    }

    /** @brief Identity matrix of size n x n. */
    template <typename T = double>
    auto identity(std::size_t n) -> Ndarray<T> {
        return eye<T>(n, n, 0);
    }

    /**
     * @brief 1D array from a std::vector (copies).
     */
    template <typename T>
    auto asarray(const std::vector<T>& values) -> Ndarray<T> {
        return Ndarray<T>::from_data(std::vector<int>{static_cast<int>(values.size())},
                                     std::vector<T>(values));
    }

    /**
     * @brief 1D array from a std::array (copies).
     */
    template <typename T, std::size_t N>
    auto asarray(const std::array<T, N>& values) -> Ndarray<T> {
        return Ndarray<T>::from_data(std::vector<int>{static_cast<int>(N)},
                                     std::vector<T>(values.begin(), values.end()));
    }

    /**
     * @brief Array of the given shape from a contiguous std::vector.
     * @throws std::invalid_argument if the sizes do not match.
     */
    template <typename T>
    auto asarray(const std::vector<T>& values, const std::vector<int>& shape)
        -> Ndarray<T> {
        return Ndarray<T>::from_data(shape, std::vector<T>(values));
    }

} // namespace np

#endif // NP_CREATION_HPP
