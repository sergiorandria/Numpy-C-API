/**
 * @file test_fixed.cpp
 * @brief Runtime tests for the fixed-shape np::ndarray<T, Extents...> path
 *        (construction, reductions, manipulation, elementwise operators,
 *        broadcasting, joins). Compile-time guarantees are exercised in
 *        test_compile_time.cpp / test_constexpr.cpp.
 */
#include <stdexcept>

#include "np/np.hpp"
#include "test_util.hpp"

int main() {
    // Construction and access
    {
        np::ndarray<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.rank == 2, "rank");
        test::check(a.size_v == 6 && a.size() == 6, "size");
        test::check(a.static_shape[0] == 2 && a.static_shape[1] == 3,
                    "static_shape");
        test::check(a(0, 1) == 2, "operator()");
        test::check(a[4] == 5, "flat operator[]");
        const auto& ca = a;
        test::check(ca(1, 2) == 6 && ca[0] == 1, "const access");
    }

    // Construction errors (ragged / wrong row count)
    {
        bool threw = false;
        try {
            np::ndarray<int, 2, 3> bad{{1, 2}, {3, 4, 5}};
            (void)bad;
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        test::check(threw, "ragged rows throw");

        threw = false;
        try {
            np::ndarray<int, 2, 3> bad{{1, 2, 3}};
            (void)bad;
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        test::check(threw, "wrong row count throws");

        np::ndarray<int, 2, 3> ok{{1, 2, 3}, {4, 5, 6}};
        test::check(ok(1, 0) == 4, "valid nested init");
        np::ndarray<int, 4> flat{1, 2, 3, 4};
        test::check(flat[3] == 4, "rank-1 init list");
        np::ndarray<double> scalar{2.5};
        test::check(scalar[0] == 2.5, "rank-0 init list");
    }

    // std::array construction
    {
        np::ndarray<int, 3> b{std::array<int, 3>{7, 8, 9}};
        test::check(b[1] == 8, "std::array ctor");
    }

    // fill
    {
        np::ndarray<int, 2, 2> f;
        f.fill(5);
        test::check(f(1, 1) == 5 && f(0, 0) == 5, "fill");
    }

    // Reductions
    {
        np::ndarray<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.sum() == 21, "sum all");
        test::check(a.prod() == 720, "prod all");
        test::check(a.min() == 1 && a.max() == 6, "min/max");
        test::check(a.argmin() == 0 && a.argmax() == 5, "argmin/argmax");
        test::check(a.mean() == 3.5, "mean all (double)");
        test::check(a.std() > 1.7 && a.std() < 1.8, "std all");

        auto rowsum = a.sum<1>();
        test::check(rowsum.rank == 1 && rowsum[0] == 6 && rowsum[1] == 15,
                    "sum axis=1");
        auto colsum = a.sum<0>();
        test::check(colsum.rank == 1 && colsum[0] == 5 && colsum[2] == 9,
                    "sum axis=0");
        auto rowmean = a.mean<1>();
        test::check(rowmean[0] == 2.0 && rowmean[1] == 5.0, "mean axis=1");
        auto colmin = a.min<0>();
        test::check(colmin[1] == 2 && colmin[0] == 1, "min axis=0");
        auto colmax = a.max<1>();
        test::check(colmax[0] == 3 && colmax[1] == 6, "max axis=1");

        np::ndarray<bool, 2, 2> tb{{true, true}, {true, false}};
        test::check(tb.all() == false && tb.any() == true, "bool all/any");
        test::check(tb.all<1>()[0] == true && tb.all<1>()[1] == false,
                    "bool all axis");
        test::check(tb.sum() == true, "bool sum");
    }

    // Manipulation
    {
        np::ndarray<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto t = a.transpose();
        test::check(t.rank == 2 && t.static_shape[0] == 3 &&
                        t.static_shape[1] == 2,
                    "transpose shape");
        test::check(t(0, 1) == 4 && t(1, 0) == 2, "transpose values");
        test::check(a(0, 1) == 2, "transpose does not mutate source");

        auto r = a.reshape<3, 2>();
        test::check(r.static_shape[0] == 3 && r(2, 1) == 6,
                    "reshape values");
        auto r1 = a.reshape<1, 6>();
        test::check(r1(0, 5) == 6, "reshape to rank 2");

        auto f = a.flatten();
        test::check(f.rank == 1 && f.size_v == 6 && f[5] == 6, "flatten");

        np::ndarray<int, 1, 3, 1> sq_src{{1, 2, 3}};
        auto sq = sq_src.squeeze();
        test::check(sq.rank == 1 && sq.size_v == 3 && sq[2] == 3,
                    "squeeze all");
        auto sq1 = sq_src.squeeze<0>();
        test::check(sq1.rank == 2 && sq1.static_shape[0] == 3,
                    "squeeze axis 0");
        auto sq2 = sq_src.squeeze<2>();
        test::check(sq2.rank == 2 && sq2.static_shape[1] == 3,
                    "squeeze axis 2");

        auto ex = a.expand_dims<0>();
        test::check(ex.rank == 3 && ex.static_shape[0] == 1 &&
                        ex(0, 1, 2) == 6,
                    "expand_dims front");
        auto ex2 = a.expand_dims<2>();
        test::check(ex2.rank == 3 && ex2.static_shape[2] == 1 &&
                        ex2(1, 0, 0) == 4,
                    "expand_dims back");
    }

    // Elementwise expressions
    {
        np::ndarray<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto e = a + a * 2;
        test::check(e(0, 0) == 3 && e(1, 2) == 18, "lazy chain");
        np::ndarray<int, 2, 3> conv = a + a;
        test::check(conv(0, 2) == 6, "implicit eval conversion");
        auto mat = e.eval();
        test::check(mat.rank == 2 && mat(1, 1) == 15, "explicit eval");

        np::ndarray<int, 2, 3> dm = a - 1;
        test::check(dm(1, 0) == 3, "scalar subtraction");
        np::ndarray<int, 2, 3> m2 = a * a;
        test::check(m2(1, 2) == 36, "elementwise multiply");
        np::ndarray<int, 2, 3> d = a / np::ndarray<int, 2, 3>{{1, 1, 1},
                                                               {2, 2, 2}};
        test::check(d(1, 0) == 2, "elementwise divide");
        np::ndarray<int, 2, 3> mm = a % 2;
        test::check(mm(0, 0) == 1 && mm(1, 0) == 0, "modulo");

        np::ndarray<bool, 2, 3> gt = a > 3;
        test::check(gt(0, 0) == false && gt(1, 0) == true, "comparison");
        np::ndarray<bool, 2, 3> le = a <= 3;
        test::check(le(1, 2) == false && le(0, 2) == true, "le comparison");

        auto neg = -a;
        test::check(neg(0, 0) == -1 && neg(1, 2) == -6, "unary minus");

        np::ndarray<int, 2, 3> ands = a & np::ndarray<int, 2, 3>{{1, 1, 1},
                                                                 {0, 0, 0}};
        test::check(ands(1, 0) == 0 && ands(0, 0) == 1, "bitwise and");
        np::ndarray<int, 2, 3> shifts = a << 1;
        test::check(shifts(0, 0) == 2 && shifts(1, 2) == 12, "left shift");
    }

    // Broadcasting
    {
        np::ndarray<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        np::ndarray<int, 3> col{10, 20, 30};
        np::ndarray<int, 2, 1> row{{100}, {200}};
        auto bc = a + col + row;
        test::check(bc.rank == 2 && bc.static_shape[0] == 2 &&
                        bc.static_shape[1] == 3,
                    "broadcast shape");
        test::check(bc(0, 0) == 111 && bc(1, 0) == 214 && bc(1, 2) == 236,
                    "broadcast values");
        auto scalar = a + 1;
        test::check(scalar(0, 2) == 4, "scalar broadcast");
    }

    // Elementwise math functions
    {
        np::ndarray<double, 2> s{4.0, 9.0};
        auto sr = np::sqrt(s).eval();
        test::check(sr[0] == 2.0 && sr[1] == 3.0, "np::sqrt");
        np::ndarray<int, 2> ai{-3, 4};
        test::check(np::abs(ai).eval()[0] == 3, "np::abs");
        test::check(np::square(np::ndarray<int, 2>{2, 3}).eval()[1] == 9,
                    "np::square");
        test::check(np::power(np::ndarray<int, 2>{2, 3}, 3).eval()[1] == 27,
                    "np::power");
        auto se = np::exp(np::ndarray<double, 1>{0.0}).eval();
        test::check(se[0] == 1.0, "np::exp");
        auto sl = np::log(np::ndarray<double, 1>{1.0}).eval();
        test::check(sl[0] == 0.0, "np::log");
        auto ss = np::sin(np::ndarray<double, 1>{0.0}).eval();
        test::check(ss[0] == 0.0, "np::sin");
        auto sf = np::floor(np::ndarray<double, 1>{2.7}).eval();
        test::check(sf[0] == 2.0, "np::floor");
    }

    // concatenate / stack
    {
        np::ndarray<int, 3> c1{{0, 1, 2}};
        np::ndarray<int, 2> c2{{0, 1}};
        auto cat = np::concatenate(c1, c2);
        test::check(cat.rank == 1 && cat.size_v == 5 && cat[3] == 0 &&
                        cat[4] == 1,
                    "concatenate");
        np::ndarray<int, 2, 2> m1{{1, 2}, {3, 4}};
        auto st = np::stack<0>(m1, m1);
        test::check(st.rank == 3 && st.static_shape[0] == 2 &&
                        st(1, 0, 1) == 2,
                    "stack axis 0");
        auto st1 = np::stack<2>(m1, m1);
        test::check(st1.rank == 3 && st1.static_shape[2] == 2 &&
                        st1(0, 1, 1) == 2,
                    "stack axis 2");
    }

    // Copy semantics
    {
        np::ndarray<int, 2, 2> a{{1, 2}, {3, 4}};
        auto b = a;
        b(0, 0) = 99;
        test::check(a(0, 0) == 1, "copy is deep");
        b = a;
        test::check(b(0, 0) == 1, "assignment");
    }

    // Fixed-shape creators
    {
        auto z = np::zeros<2, 3>();
        test::check(z(1, 2) == 0 && z.size_v == 6, "zeros");
        auto o = np::ones<int, 3>();
        test::check(o[0] == 1 && o[2] == 1, "ones");
        auto f = np::full<2, 2>(7);
        test::check(f(1, 1) == 7, "full");
        auto e = np::eye<3>();
        test::check(e(0, 0) == 1.0 && e(1, 2) == 0.0 && e(2, 2) == 1.0,
                    "eye");
        auto e1 = np::eye<3, 4, 1, int>();
        test::check(e1(0, 1) == 1 && e1(0, 0) == 0 && e1(2, 3) == 1,
                    "eye k=1");
        auto e2 = np::eye<3, 3, -1, int>();
        test::check(e2(1, 0) == 1 && e2(0, 0) == 0, "eye k=-1");
        auto i3 = np::identity<3, int>();
        test::check(i3(2, 2) == 1 && i3(1, 0) == 0, "identity");
        auto rng = np::arange<6, int>();
        test::check(rng[0] == 0 && rng[5] == 5, "arange");
        auto rng2 = np::arange<6>(1, 7, 2);
        test::check(rng2[0] == 1 && rng2[3] == 7 && rng2[5] == 11, "arange s/s");
        auto ls = np::linspace<5>(0.0, 1.0);
        test::check(ls[0] == 0.0 && ls[2] == 0.5 && ls[4] == 1.0, "linspace");
        auto ls1 = np::linspace<1>(5.0, 9.0);
        test::check(ls1[0] == 5.0, "linspace num=1");
    }

    // Fixed-shape linalg
    {
        np::ndarray<int, 3> u{1, 2, 3};
        np::ndarray<int, 3> v{4, 5, 6};
        test::check(np::linalg::dot(u, v) == 32, "dot 1D . 1D");

        np::ndarray<int, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
        auto mv = np::linalg::dot(m, v);
        test::check(mv.rank == 1 && mv[0] == 32 && mv[1] == 77,
                    "dot 2D . 1D");
        np::ndarray<int, 3, 2> n{{1, 2}, {3, 4}, {5, 6}};
        auto vm = np::linalg::dot(v, n);
        test::check(vm.rank == 1 && vm[0] == 49 && vm[1] == 64,
                    "dot 1D . 2D");

        auto mm = np::linalg::dot(m, n);
        test::check(mm.rank == 2 && mm(0, 0) == 22 && mm(1, 1) == 64,
                    "dot 2D . 2D");
        auto mat = np::linalg::matmul(m, n);
        test::check(mat(0, 1) == 28 && mat(1, 0) == 49, "matmul");
        auto roundtrip = np::linalg::matmul(m, np::eye<3, 3, 0, int>());
        test::check(roundtrip.rank == 2 && roundtrip(1, 1) == 5, "matmul eye");
    }

    return test::failures() ? 1 : 0;
}
