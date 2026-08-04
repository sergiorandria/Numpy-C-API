/**
 * @file test_constexpr.cpp
 * @brief Compile-time evaluation of the fixed-shape path. Every check below
 *        is a static_assert: the whole program only compiles if the
 *        library actually folds these expressions at compile time.
 *        (std::sqrt/std::exp are not constexpr before C++26, so the math
 *        kernels are provided by np::detail::math.)
 */
#include "np/np.hpp"
#include "np/detail/math_constexpr.hpp"

namespace mce = np::detail::math;

constexpr np::ndarray<int, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
constexpr np::ndarray<double, 2> S{4.0, 9.0};

// Construction / access
static_assert(A.rank == 2);
static_assert(A.size_v == 6);
static_assert(A.size() == 6);
static_assert(A(1, 2) == 6);
static_assert(A[4] == 5);
static_assert(A.static_shape[0] == 2 && A.static_shape[1] == 3);

// Reductions
static_assert(A.sum() == 21);
static_assert(A.prod() == 720);
static_assert(A.sum<1>()[1] == 15);
static_assert(A.sum<0>()[0] == 5);
static_assert(A.min() == 1);
static_assert(A.max() == 6);
static_assert(A.argmin() == 0);
static_assert(A.argmax() == 5);
static_assert(A.mean() == 3.5);
static_assert(A.mean<1>()[0] == 2.0);

// Manipulation
static_assert(A.transpose()(1, 0) == 2);
static_assert(A.reshape<3, 2>()(2, 1) == 6);
static_assert(A.reshape<1, 6>()(0, 5) == 6);
static_assert(A.flatten()[5] == 6);
static_assert(A.expand_dims<0>()(0, 1, 2) == 6);
static_assert(np::ndarray<int, 1, 3, 1>{{1, 2, 3}}.squeeze().size_v == 3);

// Elementwise expressions
static_assert((A + A)(0, 2) == 6);
static_assert((A * 2).eval()(1, 0) == 8);
static_assert((A + A * 2)(1, 2) == 18);
static_assert((A + np::ndarray<int, 3>{10, 20, 30})(1, 2) == 36);
static_assert((-A)(0, 0) == -1);
static_assert((A > 3)(1, 0));
static_assert((A & np::ndarray<int, 2, 3>{{1, 1, 1}, {0, 0, 0}})(1, 0) == 0);

// Math functions
static_assert(np::square(A).eval()(1, 2) == 36);
static_assert(np::abs(np::ndarray<int, 2>{-3, 4}).eval()[0] == 3);
static_assert(np::power(np::ndarray<int, 2>{2, 3}, 3).eval()[1] == 27);
static_assert(np::sqrt(S).eval()[1] == 3.0);
static_assert(np::exp(np::ndarray<double, 1>{0.0}).eval()[0] == 1.0);
static_assert(np::log(np::ndarray<double, 1>{1.0}).eval()[0] == 0.0);
static_assert(np::sin(np::ndarray<double, 1>{0.0}).eval()[0] == 0.0);
static_assert(np::floor(np::ndarray<double, 1>{2.7}).eval()[0] == 2.0);

// Joins
static_assert(np::concatenate(np::ndarray<int, 3>{0, 1, 2},
                              np::ndarray<int, 2>{0, 1})
                  .size_v == 5);
static_assert(np::stack<0>(A, A)(1, 0, 1) == 2);
static_assert(np::stack<2>(A, A)(0, 1, 1) == 2);

// Creators
static_assert(np::zeros<int, 2, 2>()(1, 1) == 0);
static_assert(np::ones<int, 3>()[2] == 1);
static_assert(np::full<2, 2>(7)(1, 0) == 7);
static_assert(np::eye<3, 3, 1, int>()(0, 1) == 1 && np::eye<3, 3, 1, int>()(1, 0) == 0);
static_assert(np::identity<2, int>()(1, 1) == 1);
static_assert(np::arange<4, int>()[3] == 3);
static_assert(np::arange<4>(1, 7, 2)[3] == 7);
static_assert(np::linspace<5>(0.0, 1.0)[2] == 0.5);
static_assert(np::linspace<5>(0.0, 1.0)[4] == 1.0);

// Linalg
constexpr np::ndarray<int, 3> U{1, 2, 3};
constexpr np::ndarray<int, 3> V{4, 5, 6};
constexpr np::ndarray<int, 2, 3> M{{1, 2, 3}, {4, 5, 6}};
constexpr np::ndarray<int, 3, 2> N{{1, 2}, {3, 4}, {5, 6}};
static_assert(np::linalg::dot(U, V) == 32);
static_assert(np::linalg::dot(M, V)[1] == 77);
static_assert(np::linalg::dot(V, N)[0] == 49 && np::linalg::dot(V, N)[1] == 64);
static_assert(np::linalg::dot(M, N)(1, 1) == 64);
static_assert(np::linalg::matmul(M, N)(1, 0) == 49);

// constexpr math kernels (std::sqrt etc. are not constexpr until C++26)
static_assert(mce::abs(-3) == 3);
static_assert(mce::floor(2.7) == 2.0 && mce::ceil(2.1) == 3.0);
static_assert(mce::round(2.5) == 2.0 && mce::round(-2.5) == -2.0);
static_assert(mce::round(3.5) == 4.0 && mce::round(2.4) == 2.0);
static_assert(mce::square(5) == 25);
static_assert(mce::pow(2, 10) == 1024);
static_assert(mce::pow(2.0, -2.0) == 0.25);
static_assert(mce::pow(0.0, 0.0) == 1.0);
static_assert(mce::sqrt(16.0) == 4.0);
static_assert(mce::exp(0.0) == 1.0);
static_assert(mce::log(1.0) == 0.0);
static_assert(mce::sin(0.0) == 0.0);
static_assert(mce::cos(0.0) == 1.0);
static_assert(mce::tan(0.0) == 0.0);
static_assert(mce::fmod(7.0, 3.0) == 1.0);
static_assert(mce::pi_v > 3.1415926 && mce::pi_v < 3.1415927);

int main() { return 0; }
