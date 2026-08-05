/**
 * @file dtype.hpp
 * @brief NumPy-compatible data type system.
 *
 * Defines the `np::dtype` enumeration and compile-time bridges between
 * `np::dtype` values and native C++ types.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DTYPE_HPP
#define NP_DTYPE_HPP

#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace np {

    /**
     * @brief Enumeration of NumPy-compatible data types.
     *
     * The values mirror numpy.dtype names. `string_` and `unicode_`
     * are present for API parity but have no C++ storage representation
     * (size() returns 0). `datetime64` and `timedelta64` store as
     * int64_t units with a separate `np::datetime64` unit code.
     */
    enum class dtype {
        // Integer types
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,

        // Floating-point types
        float16, float32, float64, longdouble,

        // Complex types
        complex64, complex128, clongdouble,

        // Boolean
        bool_,

        // String / unicode
        string_, unicode_,

        // Datetime
        datetime64, timedelta64,

        // Special
        void_, object_
    };

    // Convenience constants (same spelling as NumPy scalars).
    inline constexpr dtype int8        = dtype::int8;
    inline constexpr dtype int16       = dtype::int16;
    inline constexpr dtype int32       = dtype::int32;
    inline constexpr dtype int64       = dtype::int64;
    inline constexpr dtype uint8       = dtype::uint8;
    inline constexpr dtype uint16      = dtype::uint16;
    inline constexpr dtype uint32      = dtype::uint32;
    inline constexpr dtype uint64      = dtype::uint64;
    inline constexpr dtype float16     = dtype::float16;
    inline constexpr dtype float32     = dtype::float32;
    inline constexpr dtype float64     = dtype::float64;
    inline constexpr dtype longdouble  = dtype::longdouble;
    inline constexpr dtype complex64   = dtype::complex64;
    inline constexpr dtype complex128  = dtype::complex128;
    inline constexpr dtype clongdouble = dtype::clongdouble;
    inline constexpr dtype bool_       = dtype::bool_;
    inline constexpr dtype string_     = dtype::string_;
    inline constexpr dtype unicode_    = dtype::unicode_;
    inline constexpr dtype datetime64  = dtype::datetime64;
    inline constexpr dtype timedelta64 = dtype::timedelta64;
    inline constexpr dtype void_       = dtype::void_;
    inline constexpr dtype object_     = dtype::object_;

    namespace detail {

        /**
         * @brief Maps a np::dtype value to its native C++ type.
         *
         * @tparam D  A np::dtype enumeration value.
         */
        template <dtype D>
        struct np_type_to_cxx;

        template <> struct np_type_to_cxx<dtype::int8>      { using type = std::int8_t; };
        template <> struct np_type_to_cxx<dtype::int16>     { using type = std::int16_t; };
        template <> struct np_type_to_cxx<dtype::int32>     { using type = std::int32_t; };
        template <> struct np_type_to_cxx<dtype::int64>     { using type = std::int64_t; };
        template <> struct np_type_to_cxx<dtype::uint8>     { using type = std::uint8_t; };
        template <> struct np_type_to_cxx<dtype::uint16>    { using type = std::uint16_t; };
        template <> struct np_type_to_cxx<dtype::uint32>    { using type = std::uint32_t; };
        template <> struct np_type_to_cxx<dtype::uint64>    { using type = std::uint64_t; };
        template <> struct np_type_to_cxx<dtype::float16>   { using type = std::uint16_t; };
        template <> struct np_type_to_cxx<dtype::float32>   { using type = float; };
        template <> struct np_type_to_cxx<dtype::float64>   { using type = double; };
        template <> struct np_type_to_cxx<dtype::longdouble>{ using type = long double; };
        template <> struct np_type_to_cxx<dtype::complex64> { using type = std::complex<float>; };
        template <> struct np_type_to_cxx<dtype::complex128>{ using type = std::complex<double>; };
        template <> struct np_type_to_cxx<dtype::clongdouble>{ using type = std::complex<long double>; };
        template <> struct np_type_to_cxx<dtype::bool_>     { using type = bool; };

        /**
         * @brief Maps a native C++ type to its np::dtype value.
         *
         * Uses a type-keyed constant expression so that aliased types
         * (e.g. std::int32_t == int on many ABIs) cannot collide.
         *
         * @tparam T  A native C++ type.
         */
        template <typename T>
        struct cxx_to_np_type_impl {
            static constexpr dtype value =
                std::is_same_v<T, std::int8_t> ? dtype::int8 :
                std::is_same_v<T, std::int16_t> ? dtype::int16 :
                std::is_same_v<T, std::int32_t> ? dtype::int32 :
                std::is_same_v<T, std::int64_t> ? dtype::int64 :
                std::is_same_v<T, std::uint8_t> ? dtype::uint8 :
                std::is_same_v<T, std::uint16_t> ? dtype::uint16 :
                std::is_same_v<T, std::uint32_t> ? dtype::uint32 :
                std::is_same_v<T, std::uint64_t> ? dtype::uint64 :
                std::is_same_v<T, float> ? dtype::float32 :
                std::is_same_v<T, double> ? dtype::float64 :
                std::is_same_v<T, long double> ? dtype::longdouble :
                std::is_same_v<T, std::complex<float>> ? dtype::complex64 :
                std::is_same_v<T, std::complex<double>> ? dtype::complex128 :
                std::is_same_v<T, std::complex<long double>> ? dtype::clongdouble :
                std::is_same_v<T, bool> ? dtype::bool_ :
                std::is_same_v<T, char> ? dtype::int8 :
                std::is_same_v<T, signed char> ? dtype::int8 :
                std::is_same_v<T, unsigned char> ? dtype::uint8 :
                std::is_same_v<T, wchar_t> ? dtype::int32 :
                std::is_same_v<T, std::size_t> ?
                    (sizeof(std::size_t) == 8 ? dtype::uint64 : dtype::uint32) :
                std::is_same_v<T, std::ptrdiff_t> ?
                    (sizeof(std::ptrdiff_t) == 8 ? dtype::int64 : dtype::int32) :
                dtype::void_;
        };

        template <typename T>
        struct cxx_to_np_type : cxx_to_np_type_impl<std::remove_cv_t<T>> {};

        /**
         * @brief True when T is a std::complex instantiation.
         *
         * @tparam T  A type to inspect.
         */
        template <typename T> struct is_complex : std::false_type {};
        template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
        template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

    } // namespace detail

    /**
     * @brief Native C++ type corresponding to a np::dtype value.
     *
     * @tparam D  A np::dtype enumeration value.
     */
    template <dtype D>
    using dtype_t = typename detail::np_type_to_cxx<D>::type;

    /**
     * @brief np::dtype value corresponding to a native C++ type.
     *
     * @tparam T  A native C++ type.
     */
    template <typename T>
    inline constexpr dtype dtype_of = detail::cxx_to_np_type<std::remove_cv_t<T>>::value;

    /** @brief True when T is a std::complex instantiation. */
    template <typename T>
    inline constexpr bool is_complex_v = detail::is_complex<T>::value;

    // ---------------------------------------------------------------------
    // Runtime helpers
    // ---------------------------------------------------------------------

    /**
     * @brief Human-readable name of a dtype.
     *
     * @param t  The dtype value.
     * @return   A string_view naming the dtype (e.g. "int8", "float64").
     */
    [[nodiscard]] constexpr std::string_view dtype_name(dtype t) {
        switch (t) {
            case dtype::int8:        return "int8";
            case dtype::int16:       return "int16";
            case dtype::int32:       return "int32";
            case dtype::int64:       return "int64";
            case dtype::uint8:       return "uint8";
            case dtype::uint16:      return "uint16";
            case dtype::uint32:      return "uint32";
            case dtype::uint64:      return "uint64";
            case dtype::float16:     return "float16";
            case dtype::float32:     return "float32";
            case dtype::float64:     return "float64";
            case dtype::longdouble:  return "longdouble";
            case dtype::complex64:   return "complex64";
            case dtype::complex128:  return "complex128";
            case dtype::clongdouble: return "clongdouble";
            case dtype::bool_:       return "bool";
            case dtype::string_:     return "str";
            case dtype::unicode_:    return "unicode";
            case dtype::datetime64:  return "datetime64";
            case dtype::timedelta64: return "timedelta64";
            case dtype::void_:       return "void";
            case dtype::object_:     return "object";
        }
        return "unknown";
    }

    /**
     * @brief Size in bytes of a dtype.
     *
     * Returns 0 for the special/variable dtypes (string_, unicode_,
     * void_, object_) and for longdouble/clongdouble (which are
     * platform-dependent).
     *
     * @param t  The dtype value.
     * @return   Number of bytes, or 0 for variable-length types.
     */
    [[nodiscard]] constexpr std::size_t dtype_size(dtype t) {
        switch (t) {
            case dtype::int8:  case dtype::uint8:  case dtype::bool_: return 1;
            case dtype::int16: case dtype::uint16: case dtype::float16: return 2;
            case dtype::int32: case dtype::uint32: case dtype::float32: return 4;
            case dtype::int64: case dtype::uint64: case dtype::float64:
            case dtype::complex64: case dtype::datetime64: case dtype::timedelta64: return 8;
            case dtype::complex128: return 16;
            case dtype::longdouble: return sizeof(long double);
            case dtype::clongdouble: return sizeof(std::complex<long double>);
            case dtype::string_: case dtype::unicode_: case dtype::void_:
            case dtype::object_: return 0;
        }
        return 0;
    }

    /**
     * @brief True for the complex dtypes.
     *
     * @param t  The dtype value.
     * @return   True if t is complex64, complex128, or clongdouble.
     */
    [[nodiscard]] constexpr bool dtype_is_complex(dtype t) {
        return t == dtype::complex64 || t == dtype::complex128 ||
               t == dtype::clongdouble;
    }

    /**
     * @brief True for floating-point (non-complex) dtypes.
     *
     * @param t  The dtype value.
     * @return   True if t is float16, float32, float64, or longdouble.
     */
    [[nodiscard]] constexpr bool dtype_is_floating(dtype t) {
        return t == dtype::float16 || t == dtype::float32 ||
               t == dtype::float64 || t == dtype::longdouble;
    }

    /**
     * @brief True for the integer dtypes (signed or unsigned).
     *
     * @param t  The dtype value.
     * @return   True if t is int8 through uint64.
     */
    [[nodiscard]] constexpr bool dtype_is_integer(dtype t) {
        return t >= dtype::int8 && t <= dtype::uint64;
    }

    /**
     * @brief True for signed integer dtypes.
     *
     * @param t  The dtype value.
     * @return   True if t is int8 through int64.
     */
    [[nodiscard]] constexpr bool dtype_is_signed(dtype t) {
        return t >= dtype::int8 && t <= dtype::int64;
    }

    /**
     * @brief True for unsigned integer dtypes.
     *
     * @param t  The dtype value.
     * @return   True if t is uint8 through uint64.
     */
    [[nodiscard]] constexpr bool dtype_is_unsigned(dtype t) {
        return t >= dtype::uint8 && t <= dtype::uint64;
    }

    /**
     * @brief True for boolean dtype.
     *
     * @param t  The dtype value.
     * @return   True if t is bool_.
     */
    [[nodiscard]] constexpr bool dtype_is_bool(dtype t) {
        return t == dtype::bool_;
    }

} // namespace np

#endif // NP_DTYPE_HPP
