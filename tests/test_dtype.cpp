/**
 * @file test_dtype.cpp
 * @brief Tests for the dtype system (np/dtype.hpp).
 */
#include <cstdint>

#include "np/np.hpp"
#include "test_util.hpp"

using namespace np;

int main() {
    // Enum values
    test::check(np::int32 == dtype::int32, "int32 enum");
    test::check(np::float64 == dtype::float64, "float64 enum");
    test::check(np::bool_ == dtype::bool_, "bool_ enum");

    // dtype_t mapping
    static_assert(std::is_same_v<dtype_t<dtype::int8>, std::int8_t>);
    static_assert(std::is_same_v<dtype_t<dtype::uint8>, std::uint8_t>);
    static_assert(std::is_same_v<dtype_t<dtype::int32>, std::int32_t>);
    static_assert(std::is_same_v<dtype_t<dtype::float32>, float>);
    static_assert(std::is_same_v<dtype_t<dtype::float64>, double>);
    static_assert(std::is_same_v<dtype_t<dtype::complex128>,
                                 std::complex<double>>);
    static_assert(std::is_same_v<dtype_t<dtype::bool_>, bool>);

    // dtype_of mapping
    static_assert(dtype_of<int> == dtype::int32);
    static_assert(dtype_of<long long> == dtype::int64);
    static_assert(dtype_of<unsigned char> == dtype::uint8);
    static_assert(dtype_of<double> == dtype::float64);
    static_assert(dtype_of<float> == dtype::float32);
    static_assert(dtype_of<bool> == dtype::bool_);
    static_assert(dtype_of<std::complex<double>> == dtype::complex128);
    static_assert(dtype_of<std::complex<float>> == dtype::complex64);

    // Traits
    static_assert(is_complex_v<std::complex<double>>);
    static_assert(!is_complex_v<double>);
    static_assert(dtype_is_floating(dtype::float64));
    static_assert(!dtype_is_floating(dtype::int32));
    static_assert(dtype_is_integer(dtype::int64));
    static_assert(dtype_is_signed(dtype::int16));
    static_assert(dtype_is_unsigned(dtype::uint32));
    static_assert(dtype_is_bool(dtype::bool_));

    // Names and sizes
    test::check(std::string(dtype_name(dtype::float64)) == "float64",
                "dtype_name float64");
    test::check(dtype_size(dtype::int32) == 4, "dtype_size int32");
    test::check(dtype_size(dtype::complex128) == 16, "dtype_size c128");

    return test::failures() ? 1 : 0;
}
