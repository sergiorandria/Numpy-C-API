/**
 * @file axis_error.hpp
 * @brief Exception classes for axis-related errors and rank warnings
 *
 * This file defines exception classes for handling invalid axis parameters
 * and rank-related warnings in NumPy-like operations. These exceptions are
 * raised when an axis parameter is outside the valid range for the array's
 * dimensionality.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 * @version 1.0
 * @date 2026
 */

#ifndef AXISERROR
#define AXISERROR

#include <exception>
#include <format>
#include <source_location>

/**
 * @namespace np::exceptions
 * @brief Internal exception classes for the np library
 *
 * This namespace contains implementation-specific exception classes.
 * Users should use the public aliases provided in the np namespace.
 */
namespace np::exceptions
{
    /**
     * @class _Numpy_rank_warning
     * @brief Warning emitted when an axis parameter is invalid or out of bounds
     *
     * This warning is raised whenever an axis parameter is specified that
     * is larger than or equal to the number of array dimensions, or when
     * the axis is otherwise invalid for the requested operation.
     *
     * Unlike exceptions, warnings are typically non-fatal and may be
     * filtered or ignored depending on the warning configuration.
     *
     * @note The leading underscore indicates this is an internal class.
     *       Users should use the public alias `np::RankWarning` instead.
     *
     * @warning In strict mode, this may be treated as an exception.
     *          By default, it serves as a warning that can be caught
     *          and handled gracefully.
     *
     * @example
     * @code
     * try {
     *     np::Ndarray<int> arr({3, 4});
     *     auto result = arr.sum(axis=5);  // Invalid axis for 2D array
     * } catch (const np::RankWarning& e) {
     *     std::cerr << "Warning: " << e.what() << std::endl;
     *     // Handle gracefully, perhaps default to axis=0
     * }
     * @endcode
     *
     * @see np::AxisError for the fatal exception version
     * @see np::RankWarning Public alias for this class
     */
    class _Numpy_rank_warning: public std::exception
    {
        std::string what_msg;  ///< Formatted warning message with source location

      public:
        /**
         * @brief Constructs a rank warning with automatic source location
         *
         * Creates a warning object that captures the error message along with
         * the source location (file, line, function) where the warning was emitted.
         * The final message is formatted as:
         * "filename:line: function_name: error_message"
         *
         * @param msg The detailed warning message describing the axis/rank issue
         * @param location The source location where the warning occurred (automatically
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
         * throw np::RankWarning("Axis 5 is out of bounds for 2D array");
         *
         * // Manual source location (rarely needed)
         * throw np::RankWarning("Error", std::source_location::current());
         * @endcode
         */
        explicit _Numpy_rank_warning(const std::string& msg,
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
         * @brief Returns the formatted warning message
         *
         * Provides a null-terminated string containing the complete warning
         * message with source location information. The returned pointer
         * remains valid throughout the lifetime of the warning object.
         *
         * @return const char* Pointer to the formatted warning message string
         *
         * @note This function is marked `noexcept` and `[[nodiscard]]` to
         *       indicate that it doesn't throw and the return value should
         *       not be ignored.
         *
         * @warning The returned pointer becomes invalid when the warning
         *          object is destroyed. Do not store it beyond the object's
         *          lifetime.
         *
         * @example
         * @code
         * try {
         *     // Some operation with invalid axis...
         * } catch (const np::RankWarning& w) {
         *     std::cout << w.what() << std::endl;  // Safe
         *     const char* msg = w.what();          // Also safe
         *     // msg becomes invalid after catch block
         * }
         * @endcode
         */
        [[nodiscard]] auto what() const noexcept -> const char * override;

    };
} // namespace np::exceptions

/**
 * @namespace np
 * @brief Main namespace for the NumPy-like library
 *
 * This namespace contains all public interfaces of the library, including
 * exception aliases for user-friendly error handling.
 */
namespace np
{
    /**
     * @brief User-friendly alias for rank/axis warning
     *
     * This alias provides a cleaner, more intuitive name for the rank warning
     * class, hiding the internal underscore naming convention.
     *
     * @note Always use this alias instead of the internal class name.
     *
     * @warning This is a warning, not a fatal exception. It can be caught
     *          and handled, or configured to be ignored in production code.
     *
     * @example
     * @code
     * // Preferred usage
     * throw np::RankWarning("Invalid axis parameter");
     *
     * // Catch and handle warnings
     * try {
     *     process_array(arr, axis=10);
     * } catch (const np::RankWarning& w) {
     *     std::cerr << "Warning: " << w.what() << std::endl;
     *     // Fall back to default behavior
     *     process_array(arr, axis=0);
     * }
     *
     * // Avoid using internal name
     * // throw np::exceptions::_Numpy_rank_warning("..."); // Not recommended
     * @endcode
     *
     * @see np::exceptions::_Numpy_rank_warning
     */
    using RankWarning = exceptions::_Numpy_rank_warning;

    // Note: AxisError would be a fatal exception (not a warning) for
    // truly invalid axis parameters. Consider adding:
    // using AxisError = exceptions::_Numpy_axis_error;
}

/**
 * @example
 * @code
 * // Complete usage example with axis validation
 * #include "axis_error.hpp"
 * #include <iostream>
 *
 * template<typename T>
 * class Ndarray {
 *     std::vector<int> shape_;
 *
 * public:
 *     // Validate axis parameter and throw appropriate warning/exception
 *     void validate_axis(int axis, bool fatal = false) const {
 *         int ndim = static_cast<int>(shape_.size());
 *
 *         if (axis < -ndim || axis >= ndim) {
 *             std::string msg = "Axis " + std::to_string(axis) +
 *                              " is out of bounds for array of dimension " +
 *                              std::to_string(ndim);
 *
 *             if (fatal) {
 *                 throw std::out_of_range(msg);  // Or AxisError
 *             } else {
 *                 throw np::RankWarning(msg);    // Non-fatal warning
 *             }
 *         }
 *     }
 *
 *     // Normalize axis (handle negative indices)
 *     int normalize_axis(int axis) const {
 *         int ndim = static_cast<int>(shape_.size());
 *         if (axis < 0) {
 *             axis += ndim;
 *         }
 *         return axis;
 *     }
 * };
 *
 * int main() {
 *     try {
 *         Ndarray<int> arr;
 *         // arr shape is (3, 4) - 2 dimensions
 *
 *         arr.validate_axis(5);  // Throws RankWarning
 *
 *     } catch (const np::RankWarning& e) {
 *         std::cerr << "Warning caught: " << e.what() << std::endl;
 *         // Continue execution - warning is non-fatal
 *         std::cout << "Continuing with default behavior..." << std::endl;
 *     }
 *
 *     return 0;
 * }
 * @endcode
 */

/**
 * @example
 * @code
 * // Common axis validation patterns
 *
 * // Pattern 1: Throw warning for invalid axis (default)
 * if (axis < -ndim || axis >= ndim) {
 *     throw np::RankWarning("Axis " + std::to_string(axis) +
 *                          " out of bounds for " + std::to_string(ndim) + "D array");
 * }
 *
 * // Pattern 2: Clamp axis to valid range with warning
 * if (axis >= ndim) {
 *     np::RankWarning("Axis " + std::to_string(axis) +
 *                    " clamped to " + std::to_string(ndim - 1));
 *     axis = ndim - 1;
 * }
 *
 * // Pattern 3: Handle negative axes (NumPy convention)
 * if (axis < 0) {
 *     axis += ndim;
 *     if (axis < 0) {
 *         throw np::RankWarning("Axis " + std::to_string(axis - ndim) +
 *                              " out of bounds for " + std::to_string(ndim) + "D array");
 *     }
 * }
 *
 * // Pattern 4: Allow None/optional axis (all dimensions)
 * if (axis.has_value()) {
 *     validate_axis(axis.value());
 * } else {
 *     // Operate on flattened array
 * }
 * @endcode
 */

/**
 * @see std::exception Base class for all standard exceptions
 * @see std::format Used for formatting error messages (C++20)
 * @see std::source_location Used for capturing source location (C++20)
 *
 * @note This exception class requires C++20 features. For C++17 compatibility,
 *       you would need to remove std::format and std::source_location.
 *
 * @todo Consider adding a severity level to distinguish between warnings
 *       and fatal errors.
 * @todo Add ability to configure whether RankWarning throws or just logs.
 * @todo Integrate with logging system for production environments.
 */

#endif // AXISERROR