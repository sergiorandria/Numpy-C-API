/**
 * @file matrix.hpp
 * @brief 2D matrix type with linear algebra helpers.
 *
 * np::Matrix<T> is a 2D Ndarray<T> with (i, j) access and factories
 * (zeros, ones, eye, identity). Free functions det/inverse/solve use
 * Gaussian elimination with partial pivoting and promote to double.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MATRIX_HPP
#define NP_MATRIX_HPP

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ndarray.hpp"

namespace np {

    /**
     * @brief 2D matrix: a Ndarray guaranteed to have ndim == 2.
     */
    template <typename T>
    class Matrix : public Ndarray<T> {
    public:
        using Base = Ndarray<T>;

        /** @brief r x c matrix filled with `fill`. */
        Matrix(std::size_t rows, std::size_t cols, T fill = T{})
            : Base(std::vector<int>{static_cast<int>(rows),
                                    static_cast<int>(cols)},
                   dtype_of<T>, fill) {}

        /** @brief Build from nested row initializer lists. */
        Matrix(std::initializer_list<std::initializer_list<T>> rows)
            : Base() {
            const std::size_t nrows = rows.size();
            std::size_t ncols = 0;
            for (const auto& row : rows) {
                if (row.size() > ncols) {
                    ncols = row.size();
                }
            }
            *this = Matrix(nrows, ncols);
            std::size_t i = 0;
            for (const auto& row : rows) {
                std::size_t j = 0;
                for (const T& v : row) {
                    (*this)(i, j) = v;
                    ++j;
                }
                ++i;
            }
        }

        /** @brief Element access without bounds checks. */
        auto operator()(std::size_t i, std::size_t j) -> T& {
            return this->data()[i * this->cols() + j];
        }

        /** @brief Element access without bounds checks (const). */
        auto operator()(std::size_t i, std::size_t j) const -> const T& {
            return this->data()[i * this->cols() + j];
        }

        /** @brief Number of rows. */
        auto rows() const -> std::size_t {
            return static_cast<std::size_t>(this->shape[0]);
        }

        /** @brief Number of columns. */
        auto cols() const -> std::size_t {
            return static_cast<std::size_t>(this->shape[1]);
        }

        /** @brief True if rows == cols. */
        auto is_square() const -> bool {
            return this->rows() == this->cols();
        }

        /** @brief Matrix transposition (returns a new matrix). */
        auto transpose() const -> Matrix<T> {
            Matrix<T> out(cols(), rows());
            for (std::size_t i = 0; i < rows(); ++i) {
                for (std::size_t j = 0; j < cols(); ++j) {
                    out(j, i) = (*this)(i, j);
                }
            }
            return out;
        }

        /** @brief Matrix-matrix product. */
        template <typename U>
        auto operator*(const Matrix<U>& rhs) const
            -> Matrix<std::common_type_t<T, U>> {
            using R = std::common_type_t<T, U>;
            if (cols() != rhs.rows()) {
                throw std::invalid_argument(
                    "matrix product: inner dimensions must match");
            }
            Matrix<R> out(rows(), rhs.cols());
            for (std::size_t i = 0; i < rows(); ++i) {
                for (std::size_t j = 0; j < rhs.cols(); ++j) {
                    R acc{};
                    for (std::size_t k = 0; k < cols(); ++k) {
                        acc += static_cast<R>((*this)(i, k)) *
                               static_cast<R>(rhs(k, j));
                    }
                    out(i, j) = acc;
                }
            }
            return out;
        }

        /** @brief Scalar multiplication. */
        template <typename U>
        auto operator*(U scalar) const -> Matrix<std::common_type_t<T, U>> {
            using R = std::common_type_t<T, U>;
            Matrix<R> out(rows(), cols());
            for (std::size_t i = 0; i < rows() * cols(); ++i) {
                out.data()[i] = static_cast<R>(this->data()[i]) *
                                static_cast<R>(scalar);
            }
            return out;
        }

        // Factories ------------------------------------------------------

        /** @brief r x c matrix of zeros. */
        static auto zeros(std::size_t rows, std::size_t cols) -> Matrix<T> {
            return Matrix<T>(rows, cols, T{0});
        }

        /** @brief r x c matrix of ones. */
        static auto ones(std::size_t rows, std::size_t cols) -> Matrix<T> {
            return Matrix<T>(rows, cols, T{1});
        }

        /** @brief n x n identity matrix. */
        static auto identity(std::size_t n) -> Matrix<T> {
            return eye(n);
        }

        /** @brief n x n identity matrix. */
        static auto eye(std::size_t n) -> Matrix<T> {
            Matrix<T> out(n, n, T{0});
            for (std::size_t i = 0; i < n; ++i) {
                out(i, i) = T{1};
            }
            return out;
        }

        /** @brief n x m matrix with ones on the k-th diagonal. */
        static auto eye(std::size_t n, std::size_t m, int k = 0) -> Matrix<T> {
            Matrix<T> out(n, m, T{0});
            for (std::size_t i = 0; i < n; ++i) {
                const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + k;
                if (j >= 0 && static_cast<std::size_t>(j) < m) {
                    out(i, static_cast<std::size_t>(j)) = T{1};
                }
            }
            return out;
        }
    };

    /** @brief Scalar * Matrix. */
    template <typename T, typename U>
    auto operator*(U scalar, const Matrix<T>& m)
        -> Matrix<std::common_type_t<T, U>> {
        return m * scalar;
    }

    /**
     * @brief Determinant via Gaussian elimination with partial pivoting.
     *
     * @throws std::invalid_argument if the matrix is not square.
     */
    template <typename T>
    auto det(const Matrix<T>& m) -> double {
        if (!m.is_square()) {
            throw std::invalid_argument("det requires a square matrix");
        }
        const std::size_t n = m.rows();
        std::vector<std::vector<double>> a(n, std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                a[i][j] = static_cast<double>(m(i, j));
            }
        }
        double detv = 1.0;
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            for (std::size_t r = col + 1; r < n; ++r) {
                if (std::abs(a[r][col]) > std::abs(a[piv][col])) {
                    piv = r;
                }
            }
            if (a[piv][col] == 0.0) {
                return 0.0;
            }
            if (piv != col) {
                std::swap(a[piv], a[col]);
                detv = -detv;
            }
            detv *= a[col][col];
            for (std::size_t r = col + 1; r < n; ++r) {
                const double f = a[r][col] / a[col][col];
                for (std::size_t c = col + 1; c < n; ++c) {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        return detv;
    }

    /**
     * @brief Inverse via Gauss-Jordan elimination.
     *
     * @throws std::invalid_argument if the matrix is not square or singular.
     */
    template <typename T>
    auto inverse(const Matrix<T>& m) -> Matrix<double> {
        if (!m.is_square()) {
            throw std::invalid_argument("inverse requires a square matrix");
        }
        const std::size_t n = m.rows();
        std::vector<std::vector<double>> a(n, std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                a[i][j] = static_cast<double>(m(i, j));
            }
        }
        std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
        for (std::size_t i = 0; i < n; ++i) {
            inv[i][i] = 1.0;
        }
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            for (std::size_t r = col + 1; r < n; ++r) {
                if (std::abs(a[r][col]) > std::abs(a[piv][col])) {
                    piv = r;
                }
            }
            if (a[piv][col] == 0.0) {
                throw std::invalid_argument("matrix is singular");
            }
            if (piv != col) {
                std::swap(a[piv], a[col]);
                std::swap(inv[piv], inv[col]);
            }
            const double diag = a[col][col];
            for (std::size_t c = 0; c < n; ++c) {
                a[col][c] /= diag;
                inv[col][c] /= diag;
            }
            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const double f = a[r][col];
                for (std::size_t c = 0; c < n; ++c) {
                    a[r][c] -= f * a[col][c];
                    inv[r][c] -= f * inv[col][c];
                }
            }
        }
        Matrix<double> out(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                out(i, j) = inv[i][j];
            }
        }
        return out;
    }

    /**
     * @brief Solve A x = b via Gaussian elimination with partial pivoting.
     *
     * @throws std::invalid_argument on shape mismatch or singularity.
     */
    template <typename T, typename U>
    auto solve(const Matrix<T>& a, const Ndarray<U>& b) -> Ndarray<double> {
        if (b.ndim() != 1) {
            throw std::invalid_argument("solve: b must be a 1D array");
        }
        if (!a.is_square() ||
            a.rows() != static_cast<std::size_t>(b.shape[0])) {
            throw std::invalid_argument("solve: incompatible shapes");
        }
        const std::size_t n = a.rows();
        std::vector<std::vector<double>> m(n, std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                m[i][j] = static_cast<double>(a(i, j));
            }
        }
        std::vector<double> rhs(n);
        for (std::size_t i = 0; i < n; ++i) {
            rhs[i] = static_cast<double>(b.at(i));
        }
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            for (std::size_t r = col + 1; r < n; ++r) {
                if (std::abs(m[r][col]) > std::abs(m[piv][col])) {
                    piv = r;
                }
            }
            if (m[piv][col] == 0.0) {
                throw std::invalid_argument("matrix is singular");
            }
            if (piv != col) {
                std::swap(m[piv], m[col]);
                std::swap(rhs[piv], rhs[col]);
            }
            const double diag = m[col][col];
            for (std::size_t c = col; c < n; ++c) {
                m[col][c] /= diag;
            }
            rhs[col] /= diag;
            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const double f = m[r][col];
                for (std::size_t c = col; c < n; ++c) {
                    m[r][c] -= f * m[col][c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
        return Ndarray<double>::from_data(
            std::vector<int>{static_cast<int>(n)}, std::move(rhs));
    }

} // namespace np

#endif // NP_MATRIX_HPP
