#include <cstdint>

#include "__dtype_internal.h"
#include "dtype.hpp"

namespace
{

    /**
     * @brief Type traits to convert np::dtype to C++ type
     * @tparam DType The np::dtype enum value
     */
    template <np::dtype DType>
    struct np_type_to_cxx;

    /**
     * @brief Convert C++ type to np::dtype
     * @tparam T The C++ type
     */
    template <typename T>
    struct cxx_to_np_type;

    template <>
    struct np_type_to_cxx<np::int16>
    {
        using type = int16_t;
        static constexpr const char *name = "int16";
    };

    template <>
    struct np_type_to_cxx<np::int32>
    {
        using type = int32_t;
        static constexpr const char *name = "int32";
    };

    template <>
    struct np_type_to_cxx<np::int64>
    {
        using type = int64_t;
        static constexpr const char *name = "int64";
    };

    template <>
    struct np_type_to_cxx<np::uint8>
    {
        using type = uint8_t;
        static constexpr const char *name = "uint8";
    };

    template <>
    struct np_type_to_cxx<np::uint16>
    {
        using type = uint16_t;
        static constexpr const char *name = "uint16";
    };

    template <>
    struct np_type_to_cxx<np::uint32>
    {
        using type = uint32_t;
        static constexpr const char *name = "uint32";
    };

    template <>
    struct np_type_to_cxx<np::uint64>
    {
        using type = uint64_t;
        static constexpr const char *name = "uint64";
    };

// Floating-point types
    template <>
    struct np_type_to_cxx<np::float32>
    {
        using type = float;
        static constexpr const char *name = "float32";
    };

    template <>
    struct np_type_to_cxx<np::float64>
    {
        using type = double;
        static constexpr const char *name = "float64";
    };

    template <>
    struct np_type_to_cxx<np::longdouble>
    {
        using type = long double;
        static constexpr const char *name = "longdouble";
    };

// Complex types
    template <>
    struct np_type_to_cxx<np::complex64>
    {
        using type = std::complex<float>;
        static constexpr const char *name = "complex64";
    };

    template <>
    struct np_type_to_cxx<np::complex128>
    {
        using type = std::complex<double>;
        static constexpr const char *name = "complex128";
    };

    template <>
    struct np_type_to_cxx<np::clongdouble>
    {
        using type = std::complex<long double>;
        static constexpr const char *name = "clongdouble";
    };

// Reverse mapping: C++ type to dtype
    template <>
    struct cxx_to_np_type<float>
    {
        static constexpr np::dtype value = np::float32;
    };

    template <>
    struct cxx_to_np_type<double>
    {
        static constexpr np::dtype value = np::float64;
    };

    template <>
    struct cxx_to_np_type<long double>
    {
        static constexpr np::dtype value = np::longdouble;
    };

    template <>
    struct cxx_to_np_type<std::complex<float>>
    {
        static constexpr np::dtype value = np::complex64;
    };

    template <>
    struct cxx_to_np_type<std::complex<double>>
    {
        static constexpr np::dtype value = np::complex128;
    };

    template <>
    struct cxx_to_np_type<std::complex<long double>>
    {
        static constexpr np::dtype value = np::clongdouble;
    };

    template <>
    struct cxx_to_np_type<int16_t>
    {
        static constexpr np::dtype value = np::int16;
    };

    template <>
    struct cxx_to_np_type<int32_t>
    {
        static constexpr np::dtype value = np::int32;
    };

    template <>
    struct cxx_to_np_type<int64_t>
    {
        static constexpr np::dtype value = np::int64;
    };

    template <>
    struct cxx_to_np_type<uint8_t>
    {
        static constexpr np::dtype value = np::uint8;
    };

    template <>
    struct cxx_to_np_type<uint16_t>
    {
        static constexpr np::dtype value = np::uint16;
    };

    template <>
    struct cxx_to_np_type<uint32_t>
    {
        static constexpr np::dtype value = np::uint32;
    };

    template <>
    struct cxx_to_np_type<uint64_t>
    {
        static constexpr np::dtype value = np::uint64;
    };
    template <>
    struct cxx_to_np_type<const int16_t>
    {
        static constexpr np::dtype value = np::int16;
    };

    template <>
    struct cxx_to_np_type<const int32_t>
    {
        static constexpr np::dtype value = np::int32;
    };

    template <>
    struct cxx_to_np_type<const int64_t>
    {
        static constexpr np::dtype value = np::int64;
    };

    template <>
    struct cxx_to_np_type<const uint8_t>
    {
        static constexpr np::dtype value = np::uint8;
    };

    template <>
    struct cxx_to_np_type<const uint16_t>
    {
        static constexpr np::dtype value = np::uint16;
    };

    template <>
    struct cxx_to_np_type<const uint32_t>
    {
        static constexpr np::dtype value = np::uint32;
    };

    template <>
    struct cxx_to_np_type<const uint64_t>
    {
        static constexpr np::dtype value = np::uint64;
    };


    /**
     * @brief Helper to get the C++ type name from dtype
     * @tparam DType The np::dtype enum value
     */
    template <np::dtype DType>
    constexpr auto get_dtype_name() -> decltype(static_cast<const char *>(nullptr))
    {
        return np_type_to_cxx<DType>::name;
    }

    /**
     * @brief Helper to get the size of a dtype in bytes
     * @tparam DType The np::dtype enum value
     */
    template <np::dtype DType>
    constexpr auto get_dtype_size() -> decltype(std::size_t {})
    {
        using CppType = typename np_type_to_cxx<DType>::type;
        return sizeof(CppType);
    }
} // namespace

namespace np
{
    const np::dtype auto_detect =
    {



    };
} // namespace np