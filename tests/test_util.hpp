/**
 * @file test_util.hpp
 * @brief Minimal assertion helpers for the test suite (no external deps).
 */
#ifndef NP_TEST_UTIL_HPP
#define NP_TEST_UTIL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>

namespace test {

    inline int& failures() {
        static int f = 0;
        return f;
    }

    inline void check(bool cond, const char* what) {
        if (!cond) {
            std::printf("FAIL: %s\n", what);
            ++failures();
        }
    }

    inline void check(bool cond, const char* what, const char* detail) {
        if (!cond) {
            std::printf("FAIL: %s (%s)\n", what, detail);
            ++failures();
        }
    }

    inline bool approx(double a, double b, double eps = 1e-9) {
        const double scale = std::max({1.0, std::abs(a), std::abs(b)});
        return std::abs(a - b) <= eps * scale;
    }

    inline bool approx_c(const std::complex<double>& a,
                         const std::complex<double>& b,
                         double eps = 1e-9) {
        return approx(a.real(), b.real(), eps) &&
               approx(a.imag(), b.imag(), eps);
    }

} // namespace test

#endif // NP_TEST_UTIL_HPP
