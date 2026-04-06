/**
 * @file dtype.h
 * @brief NumPy-like data type enumeration and type management
 * @author Sergio Randriamihoatra
 * @version 1.0
 * @date 2026
 *
 * This file defines the dtype enumeration for NumPy-style data types and
 * provides internal classes for dtype management. It includes integer,
 * floating-point, complex, boolean, string, datetime, and special types.
 *
 * @note Send improvements and updates to: sergiorandriamihoatra@gmail.com
 */

#ifndef _NUMPY_DTYPE_FUND
#define _NUMPY_DTYPE_FUND

// Because C++ doesn't have a built-in dictionary type,
// we will use unordered_map for a dictionary implementation.
#include <unordered_map>
#include <type_traits>
#include <optional>

namespace np
{

    /**
     * @brief Enumeration of NumPy-compatible data types
     *
     * Tried to include every defined library numerical types here.
     *
     * @see float32, float64, int32, int64, bool_ for common type aliases
     */
#ifndef DTYPE_CLASS
    enum class dtype
    {
        // Integer types
        int8,   ///< 8-bit signed integer
        int16,  ///< 16-bit signed integer
        int32,  ///< 32-bit signed integer
        int64,  ///< 64-bit signed integer
        uint8,  ///< 8-bit unsigned integer
        uint16, ///< 16-bit unsigned integer
        uint32, ///< 32-bit unsigned integer
        uint64, ///< 64-bit unsigned integer

        // Floating-Point types
        float16, ///< 16-bit half-precision floating-point
        float32, ///< 32-bit single-precision floating-point
#ifndef FLOAT32_OTHER_EXTENSION
#define single float32
#endif
        float64, ///< 64-bit double-precision floating-point
#ifndef FLOAT64_OTHER_EXTENSION
#define double_ float64
#endif
        longdouble, ///< Extended-precision floating-point

        // Complex Number types
        complex64,  ///< 64-bit complex number (2x32-bit)
        complex128, ///< 128-bit complex number (2x64-bit)
#ifndef COMPLEX128_OTHER_EXTENSION
#define cdouble_ complex128
#endif
        clongdouble, ///< Extended-precision complex number

        // Boolean Type
        bool_, ///< Boolean type (true/false)

        // String and Unicode Types
        string_,  ///< Fixed-size string type
        unicode_, ///< Unicode string type

        // Datetime and Timedelta Types
        datetime64,  ///< 64-bit datetime type
        timedelta64, ///< 64-bit timedelta type

        // Special Types
        void_,   ///< Void/opaque type
        object_  ///< Generic Python object type
    };
#define DTYPE_CLASS
#endif

    /**
     * @brief Internal class for NumPy data type management
     *
     * Numpy numerical types are instances of np::Numpy_dtype_internal.
     * Once an array is created you can specify dtype using scalar types:
     * numpy.bool_, numpy.float32, etc.
     *
     * @tparam T The underlying type for dtype storage
     * @tparam TAlloc Allocator type for memory management
     */
    template<class T, class TAlloc>
    class Numpy_dtype_internal
    {
      public:
        /**
         * @brief Constructs a dtype object from a dtype enumeration
         * @param type The dtype enumeration value
         */
        explicit Numpy_dtype_internal(dtype type);

        /**
         * @brief Constructs a dtype object with alignment specification
         * @param type The dtype enumeration value
         * @param align Optional alignment value
         */
        Numpy_dtype_internal(dtype type, std::optional<T> align);

        /**
         * @brief Constructs a dtype object with copy and alignment
         * @param type The dtype enumeration value
         * @param copy Optional copy flag
         * @param align Boolean constant for alignment (always true)
         */
        Numpy_dtype_internal(dtype type, std::optional<T> copy,
                             std::bool_constant<true> align);

        /**
         * @brief Constructs a dtype object with full metadata
         * @param type The dtype enumeration value
         * @param copy Optional copy flag
         * @param align Boolean constant for alignment (always true)
         * @param metadata Additional metadata as key-value pairs
         */
        Numpy_dtype_internal(dtype type, std::optional<T> copy,
                             std::bool_constant<true> align,
                             std::unordered_map<T, std::optional<T>> metadata);

        /**
         * @brief Destructor for Numpy_dtype_internal
         */
        virtual ~Numpy_dtype_internal();

        /**
         * @brief Returns the stored dtype value
         * @return The dtype as type T
         */
        virtual auto getDtype() const -> decltype(T());

        /**
         * @brief Returns the alignment setting
         * @return Optional alignment value
         */
        [[nodiscard]] virtual auto getAlignement() const -> decltype(
            std::optional<T>());

        /**
         * @brief Returns the copy flag
         * @return Boolean constant indicating copy behavior
         */
        [[nodiscard]] virtual auto getCopy() const -> decltype(
            std::integral_constant<bool, true>());

        /**
         * @brief Returns the metadata dictionary
         * @return Unordered map of metadata key-value pairs
         */
        virtual std::unordered_map<T, std::optional<T>> getMetadata() const;

      private:
        T dtype_;                              ///< Stored dtype value
        std::optional<T> align;               ///< Alignment specification
        std::bool_constant<true> copy;        ///< Copy flag (always true)
        std::unordered_map<T, std::optional<T>> metadata; ///< Additional metadata
    };

    constexpr dtype float32 = dtype::float32;
    constexpr dtype float64 = dtype::float64;
    constexpr dtype int32 = dtype::int32;
    constexpr dtype int64 = dtype::int64;
    constexpr dtype bool_ = dtype::bool_;

#define int8  dtype::int8
#define int16 dtype::int16
#define uint8 dtype::uint8
#define uint16 dtype::uint16
#define uint32 dtype::uint32
#define uint64 dtype::uint64
#define float16 dtype::float16
#define longdouble dtype::longdouble
#define complex64 dtype::complex64
#define complex128 dtype::complex128
#define clongdouble dtype::clongdouble
#define string_ dtype::string_
#define unicode_ dtype::unicode_
#define datetime64 dtype::datetime64
#define timedelta64 dtype::timedelta64
#define void_ dtype::void_
#define object_ dtype::object_

} // namespace np

#include "dtype.tpp"

#endif // _NUMPY_DTYPE_FUND