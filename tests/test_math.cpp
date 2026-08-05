/**
 * @file test_math.cpp
 * @brief Tests for element-wise mathematical functions (math.hpp).
 *
 * Verifies trigonometric, hyperbolic, exponential, rounding, and arithmetic
 * functions against known values.
 */
#include <np/np.hpp>
#include "test_util.hpp"

#include <cmath>
#include <numbers>

int main() {
    using namespace np;
    
    // --- Trigonometric functions ---
    {
        auto x = arange(-std::numbers::pi, std::numbers::pi, 0.5);
        auto s = sin(x);
        auto c = cos(x);
        auto t = tan(x);
        
        test::check(s.shape == x.shape, "sin: shape preserved");
        test::check(c.shape == x.shape, "cos: shape preserved");
        test::check(test::approx(s.at(0), std::sin(-std::numbers::pi)), "sin value");
        test::check(test::approx(c.at(0), std::cos(-std::numbers::pi)), "cos value");
    }
    
    // --- Inverse trig ---
    {
        auto x = linspace(-0.9, 0.9, 10);
        auto as = arcsin(x);
        auto ac = arccos(x);
        auto at = arctan(x);
        
        test::check(test::approx(as.at(5), std::asin(x.at(5))), "arcsin");
        test::check(test::approx(ac.at(5), std::acos(x.at(5))), "arccos");
        test::check(test::approx(at.at(5), std::atan(x.at(5))), "arctan");
    }
    
    // --- arctan2 and hypot ---
    {
        auto y = ones<double>({3});
        auto x = ones<double>({3});
        x.fill(1.0);
        y.fill(1.0);
        
        auto atan2_result = arctan2(y, x);
        test::check(test::approx(atan2_result.at(0), std::numbers::pi / 4.0),
                    "arctan2(1, 1) = pi/4");
        
        auto hyp = hypot(y, x);
        test::check(test::approx(hyp.at(0), std::sqrt(2.0)), "hypot(1, 1) = sqrt(2)");
    }
    
    // --- Degree/radian conversion ---
    {
        auto rad = linspace(0.0, std::numbers::pi, 5);
        auto deg = degrees(rad);
        test::check(test::approx(deg.at(4), 180.0), "pi radians = 180 degrees");
        
        auto back = radians(deg);
        test::check(test::approx(back.at(4), std::numbers::pi), "180 deg = pi rad");
    }
    
    // --- Hyperbolic functions ---
    {
        auto x = linspace(-1.0, 1.0, 5);
        auto sh = sinh(x);
        auto ch = cosh(x);
        auto th = tanh(x);
        
        test::check(test::approx(sh.at(2), std::sinh(0.0)), "sinh(0) = 0");
        test::check(test::approx(ch.at(2), std::cosh(0.0)), "cosh(0) = 1");
        test::check(test::approx(th.at(2), std::tanh(0.0)), "tanh(0) = 0");
    }
    
    // --- Exponential and logarithmic ---
    {
        auto x = linspace(0.1, 2.0, 10);
        auto e = exp(x);
        auto l = log(e);
        
        test::check(test::approx(l.at(0), x.at(0), 1e-6), "log(exp(x)) = x");
        
        auto l10 = log10(x);
        auto l2 = log2(x);
        test::check(test::approx(l10.at(5), std::log10(x.at(5))), "log10");
        test::check(test::approx(l2.at(5), std::log2(x.at(5))), "log2");
        
        auto sq = sqrt(x);
        test::check(test::approx(sq.at(5), std::sqrt(x.at(5))), "sqrt");
        
        auto cb = cbrt(x);
        test::check(test::approx(cb.at(5), std::cbrt(x.at(5))), "cbrt");
    }
    
    // --- Power and square ---
    {
        auto base = linspace(1.0, 3.0, 5);
        auto exponent = full({5}, 2.0);
        auto p = power(base, exponent);
        
        test::check(test::approx(p.at(0), 1.0), "1^2 = 1");
        test::check(test::approx(p.at(4), 9.0), "3^2 = 9");
        
        auto sq = square(base);
        test::check(test::approx(sq.at(0), 1.0), "square(1) = 1");
        test::check(test::approx(sq.at(4), 9.0), "square(3) = 9");
    }
    
    // --- Rounding ---
    {
        auto x = asarray(std::vector<double>{-1.7, -0.5, 0.0, 0.5, 1.7, 2.3});
        
        auto f = floor(x);
        test::check(f.at(0) == -2.0, "floor(-1.7) = -2");
        test::check(f.at(4) == 1.0, "floor(1.7) = 1");
        
        auto c = ceil(x);
        test::check(c.at(0) == -1.0, "ceil(-1.7) = -1");
        test::check(c.at(4) == 2.0, "ceil(1.7) = 2");
        
        auto tr = trunc(x);
        test::check(tr.at(0) == -1.0, "trunc(-1.7) = -1");
        test::check(tr.at(4) == 1.0, "trunc(1.7) = 1");
    }
    
    // --- Arithmetic ---
    {
        auto x = asarray(std::vector<double>{-3.5, -1.0, 0.0, 1.0, 3.5});
        
        auto ab = absolute(x);
        test::check(ab.at(0) == 3.5, "abs(-3.5) = 3.5");
        test::check(ab.at(1) == 1.0, "abs(-1.0) = 1.0");
        
        auto sg = sign(x);
        test::check(sg.at(0) == -1.0, "sign(-3.5) = -1");
        test::check(sg.at(2) == 0.0, "sign(0) = 0");
        test::check(sg.at(4) == 1.0, "sign(3.5) = 1");
        
        auto a = asarray(std::vector<double>{1.0, 5.0, 3.0});
        auto b = asarray(std::vector<double>{4.0, 2.0, 6.0});
        
        auto mx = maximum(a, b);
        test::check(mx.at(0) == 4.0, "max(1, 4) = 4");
        test::check(mx.at(1) == 5.0, "max(5, 2) = 5");
        
        auto mn = minimum(a, b);
        test::check(mn.at(0) == 1.0, "min(1, 4) = 1");
        test::check(mn.at(1) == 2.0, "min(5, 2) = 2");
    }
    
    // --- Reciprocal ---
    {
        auto x = asarray(std::vector<double>{1.0, 2.0, 4.0});
        auto r = reciprocal(x);
        test::check(test::approx(r.at(0), 1.0), "1/1 = 1");
        test::check(test::approx(r.at(1), 0.5), "1/2 = 0.5");
        test::check(test::approx(r.at(2), 0.25), "1/4 = 0.25");
    }
    
    return test::failures() ? 1 : 0;
}
