/**
 * @file Matrix.hpp
 * @brief Defines the Matrix class for 2-dimensional linear algebra operations
 *
 * This file provides a specialized matrix class derived from Ndarray that enforces
 * 2-dimensional constraints and adds matrix-specific operations including
 * multiplication, transpose, determinant, and inverse.
 *
 * @author NP Library Team
 * @version 1.0
 * @date 2024
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "../ndarray.hpp"
#include "../dtype.hpp"
#include <cstddef>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <concepts>
#include <initializer_list>
#include "_Matrix_internal.h"

/**
 * @namespace np
 * @brief Main namespace for the NumPy-like library
 *
 * Contains all public interfaces including Matrix, Ndarray, and related
 * classes for numerical computing.
 */
namespace np
{
    /**
     * @brief Concept for types that behave like array-like containers or strings
     *
     * A type satisfies ArrayLikeOrString if it is either:
     * - A range (has begin/end iterators), OR
     * - Convertible to std::string
     *
     * @tparam ArrayLikeOrStringParam The type to check
     *
     * @example
     * @code
     * static_assert(ArrayLikeOrString<std::vector<int>>);     // true
     * static_assert(ArrayLikeOrString<std::string>);          // true
     * static_assert(ArrayLikeOrString<const char*>);          // true
     * static_assert(ArrayLikeOrString<int>);                  // false
     * @endcode
     */
    template <typename ArrayLikeOrStringParam>
    concept ArrayLikeOrString = std::ranges::range<ArrayLikeOrStringParam> ||
                                std::is_convertible_v<ArrayLikeOrStringParam, std::string>;

    /**
     * @brief Concept for array-like containers with standard interface
     *
     * A type satisfies ArrayLike if it provides:
     * - begin() and end() member functions
     * - size() member function
     * - std::begin() and std::end() free functions
     *
     * @tparam ArrayLikeParam The type to check
     *
     * @example
     * @code
     * static_assert(ArrayLike<std::vector<int>>);    // true
     * static_assert(ArrayLike<std::array<int,5>>);   // true
     * static_assert(ArrayLike<std::string>);         // false (no size()? actually has)
     * @endcode
     */
    template <typename ArrayLikeParam>
    concept ArrayLike =
        std::ranges::range<ArrayLikeParam> && requires(ArrayLikeParam &arr)
    {
        arr.begin();
        arr.end();
        arr.size();
        std::begin(arr);
        std::end(arr);
    };

    /**
     * @brief Concept for string-like types convertible to std::string
     *
     * @tparam StringLikeParam The type to check
     *
     * @example
     * @code
     * static_assert(StringLike<std::string>);        // true
     * static_assert(StringLike<const char*>);        // true
     * static_assert(StringLike<char*>);              // true
     * static_assert(StringLike<int>);                // false
     * @endcode
     */
    template <typename StringLikeParam>
    concept StringLike = std::is_convertible_v < StringLikeParam, std::string >;

    /**
     * @brief A 2‑dimensional matrix class derived from Ndarray
     *
     * Matrix<T> is a specialisation of Ndarray<T> that is always 2‑dimensional.
     * It inherits all the functionality of Ndarray, including the subscript
     * operator[] and the get/set methods, while adding matrix‑specific
     * operations such as transpose, determinant, and inverse.
     *
     * @tparam T Element type (should be numeric, e.g., int, float, double, complex)
     *
     * @note This class inherits from MatrixInternal which provides common
     *       implementation details for matrix operations.
     *
     * @example
     * @code
     * // Create a 2x3 matrix
     * np::Matrix<int> mat = {{1, 2, 3}, {4, 5, 6}};
     *
     * // Access elements using subscript notation
     * int elem = mat[0][1];  // Returns 2
     *
     * // Matrix multiplication
     * np::Matrix<int> result = mat * mat.transpose();
     *
     * // Compute determinant (square matrices only)
     * np::Matrix<double> square = {{1, 2}, {3, 4}};
     * double det = square.determinant();
     * @endcode
     */
    template <typename T>
    class Matrix : public MatrixInternal<T>
    {
      private:
        /**
         * @brief Enforces that the base Ndarray has exactly 2 dimensions
         *
         * @throws np::MatrixDimError if this->shape.size() != 2
         *
         * @note This is called automatically by all constructors to maintain
         *       the invariant that a Matrix is always 2-dimensional.
         */
        void enforce_2d() const;

        /**
         * @brief Validates that dimensions are non‑zero
         *
         * @throws np::MatrixDimError if rows == 0 or cols == 0
         *
         * @note A matrix must have at least one row and one column to be valid.
         */
        void validate_shape() const;

      public:
        /**
         * @brief Default constructor is deleted – a matrix must have dimensions
         *
         * Matrix objects cannot be default-constructed because they need
         * explicit dimensions. Use the parameterized constructors instead.
         */
        Matrix() = delete;

        /**
         * @brief Construct a matrix from a nested initializer list (2D)
         *
         * Creates a matrix where each inner initializer list represents a row.
         * All rows must have the same length to form a valid rectangular matrix.
         *
         * @param rows_list A list of rows, each being a list of column values
         *
         * @throws np::MatrixDimError if the list is empty or rows have inconsistent lengths
         *
         * @note This is the most intuitive way to create small matrices with
         *       known values at compile time.
         *
         * @code
         * np::Matrix<int> mat = {{1, 2, 3}, {4, 5, 6}};   // 2x3 matrix
         * np::Matrix<double> identity = {{1, 0}, {0, 1}};  // 2x2 identity
         * @endcode
         */
        constexpr Matrix(std::initializer_list<std::initializer_list<T>> rows_list);

        /**
         * @brief Construct a matrix from a 1D initializer list (column vector)
         *
         * Creates a column vector matrix with dimensions (n × 1) where n is
         * the size of the input list.
         *
         * @param col_list A list of elements forming a column vector (n×1)
         *
         * @throws np::MatrixDimError if the list is empty
         *
         * @code
         * np::Matrix<double> vec = {1.0, 2.0, 3.0};   // 3x1 column vector
         * np::Matrix<int> col = {42};                  // 1x1 matrix (scalar)
         * @endcode
         */
        constexpr Matrix(std::initializer_list<T> col_list);

        /**
         * @brief Construct a matrix from any array-like or string container
         *
         * This generic constructor accepts any type that satisfies ArrayLikeOrString,
         * allowing construction from std::vector, std::array, std::string, etc.
         *
         * @tparam U The container type (deduced)
         * @param container The source container with matrix data
         * @param type The NumPy data type (default: np::void_ for auto-detection)
         * @param copy Whether to copy data (default: false)
         *
         * @throws np::MatrixDimError If the container cannot form a valid 2D matrix
         *
         * @note For 1D containers, creates a column vector (n × 1)
         * @note For 2D containers (vector of vectors), creates a matrix with matching dimensions
         * @note For strings, creates a row vector (1 × n) of characters
         *
         * @code
         * std::vector<std::vector<int>> data = {{1,2}, {3,4}};
         * np::Matrix<int> mat(data);  // 2x2 matrix
         *
         * std::vector<double> vec = {1.0, 2.0, 3.0};
         * np::Matrix<double> col(vec);  // 3x1 column vector
         *
         * np::Matrix<char> str("hello");  // 1x5 row vector
         * @endcode
         */
        template <typename U>
        requires ArrayLikeOrString<U>
        constexpr Matrix(const U &container, np::dtype type = np::void_,
                         std::optional<bool> copy = false);

        /**
         * @brief Construct a matrix with given dimensions, optionally initialising all elements
         *
         * Creates a matrix of size rows × cols, filled with the specified initial value.
         *
         * @param rows Number of rows (must be > 0)
         * @param cols Number of columns (must be > 0)
         * @param init_value Value to fill the matrix with (default constructed if omitted)
         *
         * @throws np::MatrixDimError if rows == 0 or cols == 0
         *
         * @code
         * np::Matrix<int> zeros(3, 4);              // 3x4 matrix of default (0)
         * np::Matrix<double> ones(2, 2, 1.0);       // 2x2 matrix of ones
         * np::Matrix<float> custom(5, 5, 3.14f);    // 5x5 matrix filled with 3.14
         * @endcode
         */
        constexpr Matrix(size_t rows, size_t cols, const T& init_value = T());

        /**
         * @brief Construct a matrix from an existing 2D Ndarray (copy)
         *
         * @param arr A 2‑dimensional Ndarray
         *
         * @throws np::MatrixDimError if arr.ndim != 2
         *
         * @note This performs a deep copy of the data
         *
         * @code
         * np::Ndarray<int> arr({2, 2});
         * arr.get({0,0}) = 1;
         * arr.get({0,1}) = 2;
         * arr.get({1,0}) = 3;
         * arr.get({1,1}) = 4;
         * np::Matrix<int> mat(arr);  // Convert to matrix
         * @endcode
         */
        explicit Matrix(const Ndarray<T> &arr);

        /**
         * @brief Construct a matrix by moving an existing 2D Ndarray
         *
         * @param arr A 2‑dimensional Ndarray
         *
         * @throws np::MatrixDimError if arr.ndim != 2
         *
         * @note This transfers ownership of the data without copying
         *
         * @code
         * np::Ndarray<int> arr({2, 2});
         * // ... fill arr ...
         * np::Matrix<int> mat(std::move(arr));  // Efficient move construction
         * @endcode
         */
        explicit Matrix(Ndarray<T>&& arr);

        /**
         * @brief Copy constructor (default)
         *
         * Creates a deep copy of the matrix.
         */
        constexpr Matrix(const Matrix &) = default;

        /**
         * @brief Move constructor (default)
         *
         * Transfers ownership of data from another matrix.
         */
        constexpr Matrix(Matrix &&) noexcept = default;

        /**
         * @brief Copy assignment operator (default)
         *
         * @return Reference to this matrix
         */
        auto operator=(const Matrix &) -> Matrix & = default;

        /**
         * @brief Move assignment operator (default)
         *
         * @return Reference to this matrix
         */
        auto operator=(Matrix &&) noexcept -> Matrix & = default;

        /**
         * @brief Returns a square identity matrix of size n×n
         *
         * Creates an identity matrix with ones on the diagonal and zeros elsewhere.
         *
         * @param n Size of the identity matrix (n × n)
         * @return Matrix<T> Identity matrix
         *
         * @throws np::MatrixDimError if n == 0
         *
         * @code
         * auto I3 = np::Matrix<double>::eye(3);
         * // I3 = [[1, 0, 0],
         * //       [0, 1, 0],
         * //       [0, 0, 1]]
         * @endcode
         */
        static auto eye(size_t n) -> Matrix;

        /**
         * @brief Returns a matrix of given dimensions filled with zeros
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return Matrix<T> Zero matrix
         *
         * @code
         * auto zeros = np::Matrix<int>::zeros(3, 4);  // 3x4 matrix of zeros
         * @endcode
         */
        static auto zeros(size_t rows, size_t cols) -> Matrix;

        /**
         * @brief Returns a matrix of given dimensions filled with ones
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return Matrix<T> Matrix of ones
         *
         * @code
         * auto ones = np::Matrix<double>::ones(2, 2);  // 2x2 matrix of ones
         * @endcode
         */
        static auto ones(size_t rows, size_t cols) -> Matrix;

        /**
         * @brief Returns the number of rows
         *
         * @return size_t Number of rows (0 if matrix is uninitialized)
         */
        [[nodiscard]] auto rows() const noexcept -> size_t;

        /**
         * @brief Returns the number of columns
         *
         * @return size_t Number of columns (0 if matrix is uninitialized)
         */
        [[nodiscard]] auto cols() const noexcept -> size_t;

        /**
         * @brief Returns the shape as a pair (rows, cols)
         *
         * @return std::pair<size_t, size_t> Pair containing (rows, columns)
         *
         * @code
         * np::Matrix<int> mat(3, 4);
         * auto [r, c] = mat.shape();  // r=3, c=4
         * @endcode
         */
        [[nodiscard]] auto shape() const noexcept -> std::pair<size_t, size_t>;

        /**
         * @brief Matrix addition
         *
         * Performs element-wise addition of two matrices of the same dimensions.
         *
         * @param other Right‑hand side matrix
         * @return Matrix<T> New matrix containing element‑wise sum
         *
         * @throws np::MatrixDimError if dimensions do not match
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * np::Matrix<int> B = {{5, 6}, {7, 8}};
         * auto C = A + B;  // C = {{6, 8}, {10, 12}}
         * @endcode
         */
        auto operator+(const Matrix& other) const -> Matrix;

        /**
         * @brief Matrix subtraction
         *
         * Performs element-wise subtraction of two matrices of the same dimensions.
         *
         * @param other Right‑hand side matrix
         * @return Matrix<T> New matrix containing element‑wise difference
         *
         * @throws np::MatrixDimError if dimensions do not match
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{5, 6}, {7, 8}};
         * np::Matrix<int> B = {{1, 2}, {3, 4}};
         * auto C = A - B;  // C = {{4, 4}, {4, 4}}
         * @endcode
         */
        auto operator-(const Matrix& other) const -> Matrix;

        /**
         * @brief Matrix multiplication (matrix product)
         *
         * Performs matrix multiplication: (*this) × other.
         * The number of columns in the first matrix must equal the number
         * of rows in the second matrix.
         *
         * @param other Right‑hand side matrix
         * @return Matrix<T> Result of matrix multiplication
         *
         * @throws np::MatrixDimError if cols() != other.rows()
         *
         * @complexity O(rows × cols × other.cols()) using standard algorithm
         *
         * @code
         * np::Matrix<int> A(2, 3);  // 2x3
         * np::Matrix<int> B(3, 4);  // 3x4
         * auto C = A * B;            // 2x4 result
         * @endcode
         */
        auto operator*(const Matrix& other) const -> Matrix;

        /**
         * @brief Scalar multiplication (matrix × scalar)
         *
         * Multiplies every element of the matrix by a scalar value.
         *
         * @param scalar Scalar value to multiply by
         * @return Matrix<T> New matrix with each element multiplied by scalar
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * auto B = A * 2;  // B = {{2, 4}, {6, 8}}
         * @endcode
         */
        auto operator*(const T& scalar) const -> Matrix;

        /**
         * @brief Scalar multiplication (scalar × matrix) – non‑member friend
         *
         * @param scalar Scalar value
         * @param mat Matrix to multiply
         * @return Matrix<T> Result of scalar multiplication
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * auto B = 2 * A;  // B = {{2, 4}, {6, 8}}
         * @endcode
         */
        friend auto operator*(const T& scalar, const Matrix& mat) -> Matrix;

        /**
         * @brief Matrix addition in‑place
         *
         * Adds another matrix to this matrix element-wise.
         *
         * @param other Right‑hand side matrix
         * @return Matrix& Reference to this matrix
         *
         * @throws np::MatrixDimError if dimensions do not match
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * np::Matrix<int> B = {{5, 6}, {7, 8}};
         * A += B;  // A now = {{6, 8}, {10, 12}}
         * @endcode
         */
        auto operator+=(const Matrix& other) -> Matrix &;

        /**
         * @brief Matrix subtraction in‑place
         *
         * Subtracts another matrix from this matrix element-wise.
         *
         * @param other Right‑hand side matrix
         * @return Matrix& Reference to this matrix
         *
         * @throws np::MatrixDimError if dimensions do not match
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{5, 6}, {7, 8}};
         * np::Matrix<int> B = {{1, 2}, {3, 4}};
         * A -= B;  // A now = {{4, 4}, {4, 4}}
         * @endcode
         */
        auto operator-=(const Matrix& other) -> Matrix &;

        /**
         * @brief Matrix multiplication in‑place (only for square matrices)
         *
         * Replaces this matrix with its product with another matrix.
         * Both matrices must be square and of the same size.
         *
         * @param other Right‑hand side matrix
         * @return Matrix& Reference to this matrix
         *
         * @throws np::MatrixDimError if dimensions are incompatible or not square
         *
         * @complexity O(n³) where n is the matrix size
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * np::Matrix<int> B = {{5, 6}, {7, 8}};
         * A *= B;  // A now = A × B = {{19, 22}, {43, 50}}
         * @endcode
         */
        auto operator*=(const Matrix& other) -> Matrix &;

        /**
         * @brief Scalar multiplication in‑place
         *
         * Multiplies every element of the matrix by a scalar value.
         *
         * @param scalar Scalar value to multiply by
         * @return Matrix& Reference to this matrix
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * A *= 2;  // A now = {{2, 4}, {6, 8}}
         * @endcode
         */
        auto operator*=(const T& scalar) -> Matrix &;

        /**
         * @brief Transpose of the matrix
         *
         * Returns a new matrix that is the transpose of this matrix.
         * For a matrix A of size m×n, the transpose Aᵀ has size n×m.
         *
         * @return Matrix<T> Transposed matrix
         *
         * @complexity O(rows × cols)
         *
         * @code
         * np::Matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
         * auto AT = A.transpose();  // AT = {{1, 4}, {2, 5}, {3, 6}}
         * @endcode
         */
        auto transpose() const -> Matrix;

        /**
         * @brief Checks whether the matrix is square
         *
         * @return true if rows() == cols(), false otherwise
         *
         * @note Square matrices are required for determinant, inverse, and
         *       eigenvalue operations.
         *
         * @code
         * np::Matrix<int> A(3, 3);
         * bool sq = A.is_square();  // true
         * np::Matrix<int> B(2, 3);
         * sq = B.is_square();       // false
         * @endcode
         */
        [[nodiscard]] auto is_square() const noexcept -> bool;

        /**
         * @brief Computes the determinant of a square matrix
         *
         * Calculates the determinant using Gaussian elimination (LU decomposition)
         * with partial pivoting for numerical stability.
         *
         * @return T The determinant value
         *
         * @throws np::MatrixDimError if matrix is not square
         *
         * @complexity O(n³) where n is the matrix size
         *
         * @note For 1×1 matrices, returns the single element
         * @note For 2×2 matrices, uses direct formula
         * @note For larger matrices, uses LU decomposition
         *
         * @code
         * np::Matrix<double> A = {{1, 2}, {3, 4}};
         * double det = A.determinant();  // det = -2
         *
         * np::Matrix<double> I = np::Matrix<double>::eye(3);
         * double idet = I.determinant();  // idet = 1
         * @endcode
         */
        auto determinant() const -> T;

        /**
         * @brief Computes the inverse of a square matrix
         *
         * Calculates the matrix inverse A⁻¹ such that A × A⁻¹ = I.
         * Uses Gaussian elimination with partial pivoting.
         *
         * @return Matrix<T> The inverse matrix
         *
         * @throws np::MatrixDimError if matrix is not square
         * @throws np::MatrixDimError if matrix is singular (determinant = 0)
         *
         * @complexity O(n³) where n is the matrix size
         *
         * @warning For ill-conditioned matrices, the inverse may be numerically unstable
         *
         * @code
         * np::Matrix<double> A = {{1, 2}, {3, 4}};
         * auto Ainv = A.inverse();
         * auto I = A * Ainv;  // Should be identity (within numerical tolerance)
         * @endcode
         */
        auto inverse() const -> Matrix;

        /**
         * @brief Prints the matrix to an output stream
         *
         * Outputs the matrix in a human-readable format showing rows and columns.
         *
         * @param os Output stream (default: std::cout)
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * A.print();  // Outputs:
         *             // Matrix(2x2)
         *             // [1, 2]
         *             // [3, 4]
         * @endcode
         */
        void print(std::ostream& os = std::cout) const;

        /**
         * @brief Equality comparison operator
         *
         * Compares two matrices for element-wise equality.
         *
         * @param other Right-hand side matrix
         * @return true if all elements are equal, false otherwise
         *
         * @note Matrices with different dimensions are never equal
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * np::Matrix<int> B = {{1, 2}, {3, 4}};
         * bool eq = (A == B);  // true
         * @endcode
         */
        auto operator==(const Matrix& other) const -> bool;

        /**
         * @brief Inequality comparison operator
         *
         * @param other Right-hand side matrix
         * @return true if any element differs or dimensions differ
         *
         * @code
         * np::Matrix<int> A = {{1, 2}, {3, 4}};
         * np::Matrix<int> B = {{1, 2}, {3, 5}};
         * bool ne = (A != B);  // true
         * @endcode
         */
        auto operator!=(const Matrix& other) const -> bool;
    };

    /**
     * @brief Deduction guide for nested initializer lists
     *
     * Automatically deduces template parameter T from nested initializer lists.
     *
     * @code
     * np::Matrix mat = {{1, 2}, {3, 4}};  // Automatically deduces Matrix<int>
     * @endcode
     */
    template <typename T>
    Matrix(std::initializer_list<std::initializer_list<T>>) -> Matrix<T>;

    /**
     * @brief Deduction guide for flat initializer lists
     *
     * Automatically deduces template parameter T from flat initializer lists.
     *
     * @code
     * np::Matrix vec = {1, 2, 3};  // Automatically deduces Matrix<int>
     * @endcode
     */
    template <typename T>
    Matrix(std::initializer_list<T>) -> Matrix<T>;

    /**
     * @brief Stream output operator for Matrix
     *
     * Enables printing matrices using standard output streams.
     *
     * @param os Output stream
     * @param mat Matrix to print
     * @return std::ostream& Reference to the output stream for chaining
     *
     * @code
     * np::Matrix<int> A = {{1, 2}, {3, 4}};
     * std::cout << A << std::endl;  // Prints the matrix
     * @endcode
     */
    template <typename T>
    auto operator<<(std::ostream& os, const Matrix<T> &mat) -> std::ostream &;

} // namespace np

#include "Matrix.tpp"
#endif // MATRIX_HPP