/**
 * @file test_fft.cpp
 * @brief Tests for np::fft (radix-2 and Bluestein paths).
 */
#include <cmath>
#include <complex>
#include <vector>

#include "np/np.hpp"
#include "test_util.hpp"

using Cplx = std::complex<double>;

namespace {
    bool approx_vec(const std::vector<Cplx>& a, const std::vector<Cplx>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!test::approx_c(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }
} // namespace

int main() {
    // Impulse -> all ones (radix-2, n = 4)
    {
        std::vector<Cplx> x{Cplx{1, 0}, {0, 0}, {0, 0}, {0, 0}};
        auto y = np::fft::fft(x);
        test::check(approx_vec(y, {Cplx{1, 0}, {1, 0}, {1, 0}, {1, 0}}),
                    "fft impulse");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x), "ifft roundtrip n=4");
    }

    // Known 4-point DFT
    {
        std::vector<Cplx> x{Cplx{1, 0}, {2, 0}, {3, 0}, {4, 0}};
        auto y = np::fft::fft(x);
        test::check(test::approx_c(y[0], Cplx{10, 0}), "DFT DC");
        test::check(test::approx_c(y[2], Cplx{-2, 0}), "DFT Nyquist");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x), "roundtrip values");
    }

    // Bluestein path (n = 5, 7 - not powers of two)
    {
        std::vector<Cplx> x5{Cplx{0, 1}, {1, 0}, {2, -1}, {-3, 2}, {0.5, 0.5}};
        auto y = np::fft::fft(x5);
        test::check(y.size() == 5, "bluestein size");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x5), "bluestein roundtrip n=5");

        std::vector<Cplx> x7{Cplx{1, 0}, {0, 0}, {0, 0}, {0, 0},
                             {0, 0}, {0, 0}, {0, 0}};
        auto y7 = np::fft::fft(x7);
        test::check(y7.size() == 7, "bluestein size 7");
        for (const auto& v : y7) {
            test::check(test::approx_c(v, Cplx{1, 0}), "bluestein impulse");
        }
    }

    // Parseval: sum |x|^2 == (1/n) sum |X|^2
    {
        std::vector<Cplx> x;
        for (int i = 0; i < 8; ++i) {
            x.emplace_back(std::sin(0.1 * i), 0.3 * std::cos(0.05 * i));
        }
        auto y = np::fft::fft(x);
        double l2x = 0.0, l2y = 0.0;
        for (const auto& v : x) {
            l2x += std::norm(v);
        }
        for (const auto& v : y) {
            l2y += std::norm(v);
        }
        test::check(test::approx(l2x, l2y / 8.0, 1e-9), "parseval");
    }

    // Ndarray<float> input with 2D last-axis transform
    {
        np::Ndarray<float> a(std::vector<int>{2, 4});
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                a(i, j) = static_cast<float>(i * 4 + j);
            }
        }
        auto y = np::fft::fft(a);
        test::check(y.ndim() == 2 && y.shape[1] == 4, "2D fft shape");
        // row 0 is {0,1,2,3}: DC = 6
        test::check(test::approx_c(y(0, 0), Cplx{6, 0}), "2D fft row0 DC");
        auto back = np::fft::ifft(y);
        test::check(test::approx_c(back(1, 2), Cplx{6, 0}), "2D ifft value");

        // magnitude of a real sine
        auto mag = np::fft::abs(y);
        test::check(mag.shape[0] == 2, "fft abs shape");
    }

    return test::failures() ? 1 : 0;
}
