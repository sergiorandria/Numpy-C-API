/**
 * @file linalg.hpp
 * @brief Linear algebra functions (np::linalg::dot, matmul, inner, outer).
 *
 * Mirrors the numpy.linalg subset most commonly used:
 *   dot, matmul, inner, outer, trace, transpose.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_LINALG_HPP
#define NP_LINALG_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ndarray.hpp"

namespace np::linalg {

    /**
     * @brief Dot product supporting 1D/2D combinations:
     *  1D . 1D -> scalar, 2D . 2D -> 2D, 2D . 1D and 1D . 2D -> 1D.
     *
     * @throws std::invalid_argument on incompatible shapes or ndim > 2.
     */
    template <typename T, typename U>
    auto dot(const Ndarray<T>& a, const Ndarray<U>& b)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        const std::size_t na = a.ndim();
        const std::size_t nb = b.ndim();
        if (na > 2 || nb > 2) {
            throw std::invalid_argument(
                "dot only supports arrays with ndim <= 2");
        }
        if (na == 0 || nb == 0) {
            throw std::invalid_argument("dot operands must be non-scalar");
        }

        const auto& ashape = a.shape;
        const auto& bshape = b.shape;

        // 1D . 1D -> scalar (0-d result)
        if (na == 1 && nb == 1) {
            if (ashape[0] != bshape[0]) {
                throw std::invalid_argument(
                    "dot: incompatible 1D sizes");
            }
            R acc{};
            for (std::size_t i = 0; i < static_cast<std::size_t>(ashape[0]); ++i) {
                acc += static_cast<R>(a.at(i)) * static_cast<R>(b.at(i));
            }
            return Ndarray<R>::from_data(std::vector<int>{},
                                         std::vector<R>{acc});
        }

        // 2D . 1D -> 1D
        if (na == 2 && nb == 1) {
            if (ashape[1] != bshape[0]) {
                throw std::invalid_argument("dot: incompatible shapes");
            }
            const std::size_t rows = static_cast<std::size_t>(ashape[0]);
            const std::size_t k = static_cast<std::size_t>(ashape[1]);
            Ndarray<R> out(std::vector<int>{static_cast<int>(rows)});
            for (std::size_t i = 0; i < rows; ++i) {
                R acc{};
                for (std::size_t j = 0; j < k; ++j) {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, j})) *
                           static_cast<R>(b.at(j));
                }
                out.data()[i] = acc;
            }
            return out;
        }

        // 1D . 2D -> 1D
        if (na == 1 && nb == 2) {
            if (ashape[0] != bshape[0]) {
                throw std::invalid_argument("dot: incompatible shapes");
            }
            const std::size_t k = static_cast<std::size_t>(ashape[0]);
            const std::size_t cols = static_cast<std::size_t>(bshape[1]);
            Ndarray<R> out(std::vector<int>{static_cast<int>(cols)});
            for (std::size_t j = 0; j < cols; ++j) {
                R acc{};
                for (std::size_t i = 0; i < k; ++i) {
                    acc += static_cast<R>(a.at(i)) *
                           static_cast<R>(b.get(std::array<std::size_t, 2>{i, j}));
                }
                out.data()[j] = acc;
            }
            return out;
        }

        // 2D . 2D -> 2D
        if (ashape[1] != bshape[0]) {
            throw std::invalid_argument("dot: incompatible shapes");
        }
        const std::size_t rows = static_cast<std::size_t>(ashape[0]);
        const std::size_t k = static_cast<std::size_t>(ashape[1]);
        const std::size_t cols = static_cast<std::size_t>(bshape[1]);
        Ndarray<R> out(
            std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)});
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                R acc{};
                for (std::size_t p = 0; p < k; ++p) {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, p})) *
                           static_cast<R>(b.get(std::array<std::size_t, 2>{p, j}));
                }
                out.data()[i * cols + j] = acc;
            }
        }
        return out;
    }

    /**
     * @brief Matrix multiplication (same semantics as dot for ndim <= 2).
     */
    template <typename T, typename U>
    auto matmul(const Ndarray<T>& a, const Ndarray<U>& b)
        -> Ndarray<std::common_type_t<T, U>> {
        return dot(a, b);
    }

    /**
     * @brief Inner product: contracts the last axes; 1D . 1D gives a scalar.
     */
    template <typename T, typename U>
    auto inner(const Ndarray<T>& a, const Ndarray<U>& b)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        const std::size_t na = a.ndim();
        const std::size_t nb = b.ndim();
        if (na == 0 || nb == 0) {
            throw std::invalid_argument("inner operands must be non-scalar");
        }
        const std::size_t la = static_cast<std::size_t>(a.shape[na - 1]);
        const std::size_t lb = static_cast<std::size_t>(b.shape[nb - 1]);
        if (la != lb) {
            throw std::invalid_argument("inner: last dimensions must match");
        }
        if (na == 1 && nb == 1) {
            return dot(a, b);
        }
        // Output shape: a.shape[0..na-2] + b.shape[0..nb-2]
        std::vector<int> out_shape;
        for (std::size_t d = 0; d + 1 < na; ++d) {
            out_shape.push_back(a.shape[d]);
        }
        for (std::size_t d = 0; d + 1 < nb; ++d) {
            out_shape.push_back(b.shape[d]);
        }
        Ndarray<R> out(out_shape);
        detail::Odometer oda(std::vector<int>(a.shape.begin(), a.shape.end() - 1));
        while (!oda.done()) {
            const auto& ia = oda.idx();
            detail::Odometer odb(
                std::vector<int>(b.shape.begin(), b.shape.end() - 1));
            while (!odb.done()) {
                const auto& ib = odb.idx();
                R acc{};
                for (std::size_t p = 0; p < la; ++p) {
                    std::vector<std::size_t> ai = ia;
                    ai.push_back(p);
                    std::vector<std::size_t> bi = ib;
                    bi.push_back(p);
                    acc += static_cast<R>(a.data()[a._flat(ai)]) *
                           static_cast<R>(b.data()[b._flat(bi)]);
                }
                std::vector<std::size_t> oi(ia.begin(), ia.end());
                oi.insert(oi.end(), ib.begin(), ib.end());
                out.data()[detail::flat_index(oi, out.strides, 0)] = acc;
                odb.advance();
            }
            oda.advance();
        }
        return out;
    }

    /**
     * @brief Outer product of two 1D arrays (i, j) -> a[i] * b[j].
     */
    template <typename T, typename U>
    auto outer(const Ndarray<T>& a, const Ndarray<U>& b)
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        if (a.ndim() != 1 || b.ndim() != 1) {
            throw std::invalid_argument("outer requires two 1D arrays");
        }
        const std::size_t m = static_cast<std::size_t>(a.shape[0]);
        const std::size_t n = static_cast<std::size_t>(b.shape[0]);
        Ndarray<R> out(
            std::vector<int>{static_cast<int>(m), static_cast<int>(n)});
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                out.data()[i * n + j] =
                    static_cast<R>(a.at(i)) * static_cast<R>(b.at(j));
            }
        }
        return out;
    }

    /** @brief Transpose of a 2D array (convenience). */
    template <typename T>
    auto transpose(const Ndarray<T>& a) -> Ndarray<T> {
        return a.transpose();
    }

    /** @brief Trace of a 2D array. */
    template <typename T>
    auto trace(const Ndarray<T>& a) -> T {
        return a.trace();
    }

} // namespace np::linalg

#endif // NP_LINALG_HPP
