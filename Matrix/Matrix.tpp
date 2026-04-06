/**
 * @file Matrix.tpp
 * @brief Implementation of Matrix class template methods
 *
 * This file contains the implementation of all Matrix methods that were
 * declared in Matrix.hpp. It should be included at the end of Matrix.hpp.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 * @version 1.0
 * @date 2026
 */

#ifndef MATRIX_TPP
#define MATRIX_TPP

#include "Matrix.h"
#include "../numpy/exceptions/__matrix_dim_error.h"
#include <cmath>
#include <numeric>

namespace np
{

// ----------------------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------------------

    template<typename T>
    void Matrix<T>::enforce_2d() const
    {
        if (this->shape_.size() != 2)
        {
            throw MatrixDimError("Matrix must be 2‑dimensional");
        }
    }

    template<typename T>
    void Matrix<T>::validate_shape() const
    {
        if (rows() == 0 || cols() == 0)
        {
            throw MatrixDimError("Matrix dimensions cannot be zero");
        }
    }

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

    template<typename T>
    constexpr Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>>
                                rows_list)
        : MatrixInternal<T>(rows_list)
    {
        enforce_2d();
        validate_shape();
    }

    template<typename T>
    constexpr Matrix<T>::Matrix(std::initializer_list<T> col_list)
        : MatrixInternal<T>(col_list)
    {
        // Reshape to column vector if needed
        if (this->shape_.size() == 1)
        {
            std::vector<int> new_shape = {static_cast<int>(col_list.size()), 1};
            // Reconstruct as column
            Ndarray<T> reshaped(new_shape, this->type_, this->buffer_,
                                this->offset_, std::nullopt, this->order_);
            static_cast<MatrixInternal<T>&>(*this) = MatrixInternal<T>(reshaped);
        }
        enforce_2d();
        validate_shape();
    }

    template<typename T>
    template<typename U>
    requires ArrayLikeOrString<U>
    constexpr Matrix<T>::Matrix(const U &container, np::dtype type,
                                std::optional<bool> copy)
        : MatrixInternal<T>(container, type, copy)
    {
        enforce_2d();
        validate_shape();
    }

    template<typename T>
    constexpr Matrix<T>::Matrix(size_t rows, size_t cols, const T& init_value)
        : MatrixInternal<T>({static_cast<int>(rows), static_cast<int>(cols)},
    np::void_, std::nullopt, std::nullopt, std::nullopt,
    matrix::Order::C)
    {
        // FIX: check both rows and cols (original only checked cols)
        if (rows == 0 || cols == 0)
        {
            throw MatrixDimError("Matrix dimensions must be positive");
        }
        // FIX: added missing outer loop over rows (original had no loop variable i)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                this->get({i, j}) = init_value;
            }
        }
        enforce_2d();
        validate_shape();
    }

    template<typename T>
    Matrix<T>::Matrix(const Ndarray<T> &arr)
        : MatrixInternal<T>(arr)
    {
        // FIX: removed stray '&' that was a syntax error
        enforce_2d();
        validate_shape();
    }

    template<typename T>
    Matrix<T>::Matrix(Ndarray<T>&& arr)
        : MatrixInternal<T>(std::move(arr))
    {
        enforce_2d();
        validate_shape();
    }

// ----------------------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------------------

    template<typename T>
    auto Matrix<T>::rows() const noexcept -> size_t
    {
        return this->shape_.empty() ? 0 : static_cast<size_t>(this->shape_[0]);
    }


    template<typename T>
    auto Matrix<T>::cols() const noexcept -> size_t
    {
        return this->shape_.size() < 2 ? 0 : static_cast<size_t>(this->shape_[1]);
    }


    template<typename T>
    auto Matrix<T>::shape() const noexcept -> std::pair<size_t, size_t>
    {
        return {rows(), cols()};
    }


// ----------------------------------------------------------------------------
// Factory methods
// ----------------------------------------------------------------------------

    template<typename T>
    auto Matrix<T>::eye(size_t n) -> Matrix
    {
        if (n == 0)
        {
            throw MatrixDimError("Identity matrix size must be positive");
        }
        Matrix I(n, n, T(0));
        // FIX: added missing loop over i (original used undefined i)
        for (size_t i = 0; i < n; ++i)
        {
            I.get({i, i}) = T(1);
        }
        return I;
    }

    template<typename T>
    auto Matrix<T>::zeros(size_t rows, size_t cols) -> Matrix
    {
        return Matrix(rows, cols, T(0));
    }


    template<typename T>
    auto Matrix<T>::ones(size_t rows, size_t cols) -> Matrix
    {
        return Matrix(rows, cols, T(1));
    }


// ----------------------------------------------------------------------------
// Arithmetic operations
// ----------------------------------------------------------------------------

    template<typename T>
    auto Matrix<T>::operator+(const Matrix& other) const -> Matrix
    {
        if (rows() != other.rows() || cols() != other.cols())
        {
            throw MatrixDimError(
                "Matrix addition: dimension mismatch (" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " vs " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        Matrix result(rows(), cols());
        // FIX: added missing outer loop over i (original only looped over j)
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                result.get({i, j}) = this->get({i, j}) + other.get({i, j});
            }
        }
        return result;
    }

    template<typename T>
    auto Matrix<T>::operator-(const Matrix& other) const -> Matrix
    {
        if (rows() != other.rows() || cols() != other.cols())
        {
            throw MatrixDimError(
                "Matrix subtraction: dimension mismatch (" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " vs " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        Matrix result(rows(), cols());
        // FIX: added missing outer loop over i (original only looped over j)
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                result.get({i, j}) = this->get({i, j}) - other.get({i, j});
            }
        }
        return result;
    }

    template<typename T>
    auto Matrix<T>::operator*(const Matrix& other) const -> Matrix
    {
        if (cols() != other.rows())
        {
            throw MatrixDimError(
                "Matrix multiplication: incompatible dimensions(" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " and " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        size_t m = rows(), n = cols(), p = other.cols();
        // FIX: replaced deleted default constructor call with proper sized result matrix
        Matrix<T> result(m, p, T(0));
        // Optimized multiplication with loop ordering for cache efficiency (i-k-j)
        // FIX: added missing outer loop over i (original had no i loop)
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t k = 0; k < n; ++k)
            {
                T aik = this->get({i, k});
                if (aik != T(0))
                {
                    for (size_t j = 0; j < p; ++j)
                    {
                        // FIX: corrected typo 'ge' → 'get' and removed stray empty block
                        result.get({i, j}) += aik * other.get({k, j});
                    }
                }
            }
        }
        return result;
    }

    template<typename T>
    auto Matrix<T>::operator*(const T& scalar) const -> Matrix
    {
        Matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                result.get({i, j}) = this->get({i, j}) * scalar;
            }
        }
        return result;
    }

    template<typename T>
    auto operator*(const T& scalar, const Matrix<T> &mat) -> Matrix<T>
    {
        return mat * scalar;
    }


    template<typename T>
    auto Matrix<T>::operator+=(const Matrix& other) -> Matrix &
    {
        if (rows() != other.rows() || cols() != other.cols())
        {
            throw MatrixDimError(
                "Matrix addition: dimension mismatch (" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " vs " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        for (size_t i = 0; i < rows(); ++i)
        {
            // FIX: added missing inner loop over j (original used undefined j)
            for (size_t j = 0; j < cols(); ++j)
            {
                this->get({i, j}) += other.get({i, j});
            }
        }
        return *this;
    }

    template<typename T>
    auto Matrix<T>::operator-=(const Matrix& other) -> Matrix &
    {
        if (rows() != other.rows() || cols() != other.cols())
        {
            throw MatrixDimError(
                "Matrix subtraction: dimension mismatch (" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " vs " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        for (size_t i = 0; i < rows(); ++i)
        {
            // FIX: added missing inner loop over j (original used undefined j)
            for (size_t j = 0; j < cols(); ++j)
            {
                this->get({i, j}) -= other.get({i, j});
            }
        }
        return *this;
    }

    template<typename T>
    auto Matrix<T>::operator*=(const Matrix& other) -> Matrix &
    {
        if (!is_square())
        {
            throw MatrixDimError(
                "In-place multiplication only allowed for square matrices(current size : " +
                std::to_string(rows()) + "x" + std::to_string(cols()) + ")"
            );
        }
        if (rows() != other.rows() || cols() != other.cols())
        {
            // FIX: was a bare string expression (no throw), now properly thrown
            throw MatrixDimError(
                "In-place multiplication requires matrices of equal size(" +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " vs " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()) + ")"
            );
        }
        *this = (*this) * other;
        return *this;  // FIX: added missing return statement
    }

    template<typename T>
    auto Matrix<T>::operator*=(const T& scalar) -> Matrix &
    {
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                this->get({i, j}) *= scalar;
            }
        }
        return *this;
    }

// ----------------------------------------------------------------------------
// Matrix specific functions
// ----------------------------------------------------------------------------

    template<typename T>
    auto Matrix<T>::transpose() const -> Matrix
    {
        Matrix result(cols(), rows());
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                result.get({j, i}) = this->get({i, j});
            }
        }
        return result;
    }

    template<typename T>
    auto Matrix<T>::is_square() const noexcept -> bool
    {
        return rows() == cols();
    }


    template<typename T>
    auto Matrix<T>::determinant() const -> T
    {
        if (!is_square())
        {
            throw MatrixDimError(
                "Determinant only defined for square matrices(current size : " +
                std::to_string(rows()) + "x" + std::to_string(cols()) + ")"
            );
        }
        size_t n = rows();
        if (n == 1) return this->get({0, 0});
        if (n == 2)
        {
            return this->get({0, 0}) * this->get({1, 1}) -
                   this->get({0, 1}) * this->get({1, 0});
        }
        // LU decomposition with partial pivoting
        // FIX: added missing working copy 'A' (original used undefined 'A')
        Matrix A(*this);
        T det = T(1);
        // FIX: restructured entirely — original had misplaced braces that placed
        //      the pivot-swap block and elimination loop outside the outer for loop,
        //      added missing 'factor' computation and diagonal accumulation into det
        for (size_t i = 0; i < n; ++i)
        {
            size_t pivot = i;
            T max_val = std::abs(A.get({i, i}));
            for (size_t k = i + 1; k < n; ++k)
            {
                T abs_val = std::abs(A.get({k, i}));
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                    pivot = k;
                }
            }
            if (pivot != i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    std::swap(A.get({i, j}), A.get({pivot, j}));
                }
                det = -det;
            }
            T diag = A.get({i, i});
            if (diag == T(0)) return T(0);
            det *= diag;
            for (size_t k = i + 1; k < n; ++k)
            {
                T factor = A.get({k, i}) / diag;
                for (size_t j = i + 1; j < n; ++j)
                {
                    A.get({k, j}) -= factor * A.get({i, j});
                }
            }
        }
        return det;
    }

    template<typename T>
    auto Matrix<T>::inverse() const -> Matrix
    {
        if (!is_square())
        {
            // FIX: original had an illegal newline inside a string literal
            throw MatrixDimError(
                "Inverse only defined for square matrices(current size : " +
                std::to_string(rows()) + "x" + std::to_string(cols()) + ")"
            );
        }
        size_t n = rows();
        // FIX: 'det' was used before being declared; compute it here
        T det = this->determinant();
        if (det == T(0))
        {
            throw MatrixDimError("Matrix is singular and cannot be inverted");
        }
        // FIX: 'aug' was never declared; build augmented matrix [A | I]
        Matrix aug(n, 2 * n, T(0));
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                aug.get({i, j}) = this->get({i, j});
            }
            aug.get({i, n + i}) = T(1);
        }
        // FIX: Gaussian elimination was missing the outer loop over i,
        //      the row-swap was missing its column loop and was split with
        //      a newline inside the identifier ("std::sw\nap"), 'factor'
        //      was used but never declared, and the elimination lacked its
        //      outer loop over k
        for (size_t i = 0; i < n; ++i)
        {
            // Find pivot
            size_t pivot = i;
            T max_val = std::abs(aug.get({i, i}));
            for (size_t k = i + 1; k < n; ++k)
            {
                T abs_val = std::abs(aug.get({k, i}));
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                    pivot = k;
                }
            }
            if (pivot != i)
            {
                for (size_t j = 0; j < 2 * n; ++j)
                {
                    std::swap(aug.get({i, j}), aug.get({pivot, j}));
                }
            }
            T factor = aug.get({i, i});
            for (size_t j = 0; j < 2 * n; ++j)
            {
                aug.get({i, j}) /= factor;
            }
            // Eliminate other rows
            for (size_t k = 0; k < n; ++k)
            {
                if (k != i)
                {
                    T mult = aug.get({k, i});
                    for (size_t j = 0; j < 2 * n; ++j)
                    {
                        aug.get({k, j}) -= mult * aug.get({i, j});
                    }
                }
            }
        }
        // FIX: 'inv' was never declared
        Matrix inv(n, n);
        // Extract inverse from right half
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                inv.get({i, j}) = aug.get({i, n + j});
            }
        }
        return inv;
    }

// ----------------------------------------------------------------------------
// Printing & comparison
// ----------------------------------------------------------------------------

    template<typename T>
    void Matrix<T>::print(std::ostream& os) const
    {
        os << "Matrix(" << rows() << "x" << cols() << ")\n";
        for (size_t i = 0; i < rows(); ++i)
        {
            os << "[";
            for (size_t j = 0; j < cols(); ++j)
            {
                if (j != 0) os << ", ";
                os << this->get({i, j});
            }
            os << "]\n";
        }
    }

    template<typename T>
    auto Matrix<T>::operator==(const Matrix& other) const -> bool
    {
        if (rows() != other.rows() || cols() != other.cols())
        {
            return false;
        }
        for (size_t i = 0; i < rows(); ++i)
        {
            // FIX: added missing inner loop over j (original used undefined j)
            for (size_t j = 0; j < cols(); ++j)
            {
                if (this->get({i, j}) != other.get({i, j}))
                {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename T>
    auto Matrix<T>::operator!=(const Matrix& other) const -> bool
    {
        return !(*this == other);
    }


// ----------------------------------------------------------------------------
// Stream output operator
// ----------------------------------------------------------------------------

    template<typename T>
    auto operator<<(std::ostream& os, const Matrix<T> &mat) -> std::ostream &
    {
        mat.print(os);
        return os;  // FIX: removed stray '&' after the return statement
    }

} // namespace np

#endif // MATRIX_TPP
