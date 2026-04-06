/**
 * @file matrix_dim_error.hpp
 * @brief Exception class for matrix dimension-related errors
 *
 * This file defines a specialized exception class for reporting dimension
 * mismatches and shape-related errors in matrix operations. It automatically
 * captures source location information (file, line, function) for better
 * debugging and error tracking.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 * @version 1.0
 * @date 2026
 */

#ifndef MATRIX_DIM_ERROR
#define MATRIX_DIM_ERROR

#include <exception>
#include <string>
#include <format>
#include <source_location>

/**
 * @namespace np::exceptions
 * @brief Contains exception classes for the np library
 *
 * This namespace holds all exception types used throughout the NumPy-like
 * library, providing detailed error information with source location tracking.
 */
namespace np::exceptions
{
    /**
     * @class _Numpy_matrix_dim_error
     * @brief Exception thrown for matrix dimension and shape errors
     *
     * This exception is raised when matrix operations encounter dimension
     * mismatches, such as:
     * - Multiplying matrices with incompatible dimensions
     * - Accessing indices out of bounds
     * - Reshaping operations with invalid dimensions
     * - Broadcasting operations with incompatible shapes
     *
     * The exception automatically captures the source location (file name,
     * line number, function name) where the error occurred, providing
     * valuable debugging information without manual instrumentation.
     *
     * @note The leading underscore in the class name indicates this is an
     *       internal implementation detail. Users should use the alias
     *       `np::MatrixDimError` instead.
     *
     * @warning This exception uses C++20 features (`std::format` and
     *          `std::source_location`). Ensure your compiler supports
     *          these features.
     *
     * @example
     * @code
     * void multiply_matrices(const Matrix& A, const Matrix& B) {
     *     if (A.cols() != B.rows()) {
     *         throw np::MatrixDimError(
     *             "Cannot multiply " + std::to_string(A.rows()) + "x" +
     *             std::to_string(A.cols()) + " matrix with " +
     *             std::to_string(B.rows()) + "x" + std::to_string(B.cols())
     *         );
     *     }
     *     // Perform multiplication...
     * }
     *
     * try {
     *     auto result = multiply_matrices(A, B);
     * } catch (const np::MatrixDimError& e) {
     *     std::cerr << "Dimension error: " << e.what() << std::endl;
     *     // Output: "matrix.cpp:42: multiply_matrices: Cannot multiply 2x3 matrix with 4x5"
     * }
     * @endcode
     */
    class _Numpy_matrix_dim_error : public std::exception
    {
        std::string what_msg;  ///< Formatted error message with source location

      public:
        /**
         * @brief Constructs a matrix dimension error with automatic source location
         *
         * Creates an exception object that captures the error message along with
         * the source location (file, line, function) where the exception was thrown.
         * The final message is formatted as:
         * "filename:line: function_name: error_message"
         *
         * @param msg The detailed error message describing the dimension issue
         * @param location The source location where the error occurred (automatically
         *                 captured by default using `std::source_location::current()`)
         *
         * @throws std::bad_alloc If memory allocation for the formatted message fails
         *
         * @note The source location is automatically captured at the call site.
         *       You don't need to pass it explicitly unless you want to override it.
         *
         * @example
         * @code
         * // Automatic source location capture
         * throw np::MatrixDimError("Index 5 out of bounds for dimension 0");
         *
         * // Manual source location (rarely needed)
         * throw np::MatrixDimError("Error", std::source_location::current());
         * @endcode
         */
        explicit _Numpy_matrix_dim_error(
            const std::string& msg,
            const std::source_location& location = std::source_location::current()
        )
        {
            what_msg = std::format("{}:{}: {}: {}",
                                   location.file_name(),
                                   location.line(),
                                   location.function_name(),
                                   msg
                                  );
        }

        /**
         * @brief Returns the formatted error message
         *
         * Provides a null-terminated string containing the complete error
         * message with source location information. The returned pointer
         * remains valid throughout the lifetime of the exception object.
         *
         * @return const char* Pointer to the formatted error message string
         *
         * @note This function is marked `noexcept` and `[[nodiscard]]` to
         *       indicate that it doesn't throw and the return value should
         *       not be ignored.
         *
         * @warning The returned pointer becomes invalid when the exception
         *          object is destroyed. Do not store it beyond the exception's
         *          lifetime.
         *
         * @example
         * @code
         * try {
         *     // Some matrix operation...
         * } catch (const np::MatrixDimError& e) {
         *     std::cout << e.what() << std::endl;  // Safe
         *     const char* msg = e.what();          // Also safe
         *     // msg becomes invalid after catch block
         * }
         * @endcode
         */
        [[nodiscard]] auto what() const noexcept -> const char * override
        {
            return what_msg.c_str();
        }
    };
}

/**
 * @namespace np
 * @brief Main namespace for the NumPy-like library
 *
 * This namespace contains all public interfaces of the library, including
 * the Matrix, Ndarray, and related classes.
 */
namespace np
{
    /**
     * @brief User-friendly alias for matrix dimension error exception
     *
     * This alias provides a cleaner, more intuitive name for the dimension
     * error exception class, hiding the internal underscore naming convention.
     *
     * @note Always use this alias instead of the internal class name.
     *
     * @example
     * @code
     * // Preferred usage
     * throw np::MatrixDimError("Invalid dimensions");
     *
     * // Avoid using internal name
     * // throw np::exceptions::_Numpy_matrix_dim_error("Invalid dimensions"); // Not recommended
     * @endcode
     *
     * @see np::exceptions::_Numpy_matrix_dim_error
     */
    using MatrixDimError = exceptions::_Numpy_matrix_dim_error;
}

/**
 * @example
 * @code
 * // Complete usage example
 * #include "matrix_dim_error.hpp"
 * #include <iostream>
 *
 * class Matrix {
 *     size_t rows_, cols_;
 * public:
 *     Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}
 *
 *     Matrix operator*(const Matrix& other) const {
 *         if (cols_ != other.rows_) {
 *             throw np::MatrixDimError(
 *                 "Cannot multiply " + std::to_string(rows_) + "x" +
 *                 std::to_string(cols_) + " matrix with " +
 *                 std::to_string(other.rows_) + "x" + std::to_string(other.cols_)
 *             );
 *         }
 *         // Perform multiplication...
 *         return Matrix(rows_, other.cols_);
 *     }
 * };
 *
 * int main() {
 *     try {
 *         Matrix A(2, 3);
 *         Matrix B(4, 5);
 *         auto C = A * B;  // Throws exception
 *     } catch (const np::MatrixDimError& e) {
 *         std::cerr << "Error: " << e.what() << std::endl;
 *         // Output similar to: "matrix.cpp:15: Matrix::operator*: Cannot multiply 2x3 matrix with 4x5"
 *         return 1;
 *     }
 *     return 0;
 * }
 * @endcode
 */

#endif // MATRIX_DIM_ERROR