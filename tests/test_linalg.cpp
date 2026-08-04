/**
 * @file test_linalg.cpp
 * @brief Tests for np::linalg functions.
 */
#include <cmath>

#include "np/np.hpp"
#include "test_util.hpp"

using namespace np;

int main() {
    // dot 1D . 1D -> scalar
    {
        Ndarray<int> a{1, 2, 3};
        Ndarray<int> b{4, 5, 6};
        auto d = linalg::dot(a, b);
        test::check(d.ndim() == 0, "dot scalar ndim");
        test::check(d.item() == 32, "dot 1D value");
    }

    // dot 2D . 2D
    {
        Ndarray<int> a{{1, 2}, {3, 4}};
        Ndarray<int> b{{5, 6}, {7, 8}};
        auto d = linalg::dot(a, b);
        test::check(d.shape[0] == 2 && d.shape[1] == 2, "dot 2D shape");
        test::check(d(0, 0) == 19 && d(0, 1) == 22, "dot 2D values");
        test::check(d(1, 0) == 43 && d(1, 1) == 50, "dot 2D values 2");
    }

    // dot mixed 1D/2D
    {
        Ndarray<int> m{{1, 2}, {3, 4}};
        Ndarray<int> v{1, 1};
        auto mv = linalg::dot(m, v);
        test::check(mv.ndim() == 1 && mv.size() == 2, "dot 2D.1D shape");
        test::check(mv(0) == 3 && mv(1) == 7, "dot 2D.1D values");
        auto vm = linalg::dot(v, m);
        test::check(vm.size() == 2 && vm(0) == 4 && vm(1) == 6, "dot 1D.2D values");
    }

    // matmul
    {
        Ndarray<int> a{{1, 0}, {0, 1}};
        Ndarray<int> b{{2, 3}, {4, 5}};
        auto mm = linalg::matmul(a, b);
        test::check(mm(1, 1) == 5, "matmul identity");
    }

    // inner
    {
        Ndarray<int> a{1, 2, 3};
        Ndarray<int> b{4, 5, 6};
        auto in = linalg::inner(a, b);
        test::check(in.ndim() == 0 && in.item() == 32, "inner 1D");
    }

    // outer
    {
        Ndarray<int> a{1, 2};
        Ndarray<int> b{3, 4};
        auto o = linalg::outer(a, b);
        test::check(o.shape[0] == 2 && o.shape[1] == 2, "outer shape");
        test::check(o(1, 0) == 6 && o(0, 1) == 4, "outer values");
    }

    // mixed types promote to common type
    {
        Ndarray<int> a{1, 2};
        Ndarray<double> b{0.5, 0.5};
        auto d = linalg::dot(a, b);
        test::check(test::approx(d.item(), 1.5), "dot mixed type");
    }

    return test::failures() ? 1 : 0;
}
