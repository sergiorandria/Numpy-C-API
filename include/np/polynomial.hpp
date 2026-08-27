/**
 * @file polynomial.hpp
 * @brief Polynomial helpers (np.poly, polyval, polyfit, roots, polyadd/mul/div).
 *
 * Minimal numpy.polynomial compatible subset – 1-D double coefficients
 * ordered from highest to lowest power (numpy convention).
 *
 * Reference: numpy-reference/reference/routines.polynomials.html
 */
#ifndef NP_POLYNOMIAL_HPP
#define NP_POLYNOMIAL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np
{

  // Normal comment: poly – coefficients from roots
  NP_API inline auto poly(const ndarray<double>& roots) -> ndarray<double>
  {
    ndarray<double> coeff(std::vector<int>{1});
    coeff.data()[0] = 1.0;
    for (std::size_t r = 0; r < roots.size(); ++r)
    {
      double root = roots.data()[roots._flat_logical(r)];
      ndarray<double> next(std::vector<int>{static_cast<int>(coeff.size() + 1)});
      std::fill(next.data().begin(), next.data().end(), 0.0);
      for (std::size_t i = 0; i < coeff.size(); ++i)
      {
        next.data()[i] += coeff.data()[coeff._flat_logical(i)];
        next.data()[i + 1] -= coeff.data()[coeff._flat_logical(i)] * root;
      }
      coeff = std::move(next);
    }
    return coeff;
  }

  // Normal comment: polyval – Horner evaluation
  NP_API inline auto polyval(const ndarray<double>& p, const ndarray<double>& x)
      -> ndarray<double>
  {
    if (p.size() == 0)
      throw std::invalid_argument("polyval: empty p");
    ndarray<double> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      double xv = x.data()[x._flat_logical(i)];
      double res = p.data()[p._flat_logical(0)];
      for (std::size_t k = 1; k < p.size(); ++k)
        res = res * xv + p.data()[p._flat_logical(k)];
      out.data()[out._flat_logical(i)] = res;
    }
    return out;
  }

  NP_API inline double polyval(const ndarray<double>& p, double x)
  {
    if (p.size() == 0)
      throw std::invalid_argument("polyval: empty p");
    double res = p.data()[p._flat_logical(0)];
    for (std::size_t k = 1; k < p.size(); ++k)
      res = res * x + p.data()[p._flat_logical(k)];
    return res;
  }

  // Normal comment: polyadd / polysub / polymul
  NP_API inline auto polyadd(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    std::size_t n = std::max(a.size(), b.size());
    ndarray<double> out(std::vector<int>{static_cast<int>(n)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    std::size_t off_a = n - a.size();
    std::size_t off_b = n - b.size();
    for (std::size_t i = 0; i < a.size(); ++i)
      out.data()[off_a + i] += a.data()[a._flat_logical(i)];
    for (std::size_t i = 0; i < b.size(); ++i)
      out.data()[off_b + i] += b.data()[b._flat_logical(i)];
    return out;
  }

  NP_API inline auto polysub(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    std::size_t n = std::max(a.size(), b.size());
    ndarray<double> out(std::vector<int>{static_cast<int>(n)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    std::size_t off_a = n - a.size();
    std::size_t off_b = n - b.size();
    for (std::size_t i = 0; i < a.size(); ++i)
      out.data()[off_a + i] += a.data()[a._flat_logical(i)];
    for (std::size_t i = 0; i < b.size(); ++i)
      out.data()[off_b + i] -= b.data()[b._flat_logical(i)];
    return out;
  }

  NP_API inline auto polymul(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    if (a.size() == 0 || b.size() == 0)
      return ndarray<double>(std::vector<int>{0});
    ndarray<double> out(std::vector<int>{static_cast<int>(a.size() + b.size() - 1)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = 0; j < b.size(); ++j)
        out.data()[i + j] += a.data()[a._flat_logical(i)] * b.data()[b._flat_logical(j)];
    return out;
  }

  // Normal comment: polydiv – long division, returns pair {quotient, remainder}
  NP_API inline auto polydiv(const ndarray<double>& u, const ndarray<double>& v)
      -> std::pair<ndarray<double>, ndarray<double>>
  {
    if (v.size() == 0)
      throw std::invalid_argument("polydiv: divisor empty");
    // trim leading zeros
    std::size_t a_start = 0;
    while (a_start < u.size() && u.data()[u._flat_logical(a_start)] == 0)
      ++a_start;
    std::size_t b_start = 0;
    while (b_start < v.size() && v.data()[v._flat_logical(b_start)] == 0)
      ++b_start;
    if (b_start == v.size())
      throw std::invalid_argument("polydiv: divisor zero polynomial");
    std::vector<double> a(u.size() - a_start);
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = u.data()[u._flat_logical(a_start + i)];
    std::vector<double> b(v.size() - b_start);
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = v.data()[v._flat_logical(b_start + i)];
    if (a.size() < b.size())
    {
      ndarray<double> q(std::vector<int>{1});
      q.data()[0] = 0.0;
      ndarray<double> r(std::vector<int>{static_cast<int>(a.size())});
      for (std::size_t i = 0; i < a.size(); ++i)
        r.data()[i] = a[i];
      return {q, r};
    }
    std::vector<double> q(a.size() - b.size() + 1, 0.0);
    std::vector<double> r = a;
    for (std::size_t k = 0; k < q.size(); ++k)
    {
      double coeff = r[k] / b[0];
      q[k] = coeff;
      for (std::size_t j = 0; j < b.size(); ++j)
        r[k + j] -= coeff * b[j];
    }
    // remainder is last b.size()-1 elements of r
    std::size_t rem_start = q.size();
    std::vector<double> rem;
    for (std::size_t i = rem_start; i < r.size(); ++i)
      if (std::abs(r[i]) > 1e-12)
        rem.push_back(r[i]);
    if (rem.empty())
    {
      rem.push_back(0.0);
    }
    ndarray<double> q_arr(std::vector<int>{static_cast<int>(q.size())});
    for (std::size_t i = 0; i < q.size(); ++i)
      q_arr.data()[i] = q[i];
    ndarray<double> r_arr(std::vector<int>{static_cast<int>(rem.size())});
    for (std::size_t i = 0; i < rem.size(); ++i)
      r_arr.data()[i] = rem[i];
    return {q_arr, r_arr};
  }

  // Normal comment: roots – quadratic and linear only for simplicity
  NP_API inline auto roots(const ndarray<double>& p) -> ndarray<std::complex<double>>
  {
    // trim leading zeros
    std::size_t start = 0;
    while (start < p.size() && p.data()[p._flat_logical(start)] == 0)
      ++start;
    std::size_t n = p.size() - start;
    if (n == 0)
      throw std::invalid_argument("roots: zero polynomial");
    if (n == 1)
      return ndarray<std::complex<double>>(std::vector<int>{0});
    if (n == 2)
    {
      double a = p.data()[p._flat_logical(start)];
      double b = p.data()[p._flat_logical(start + 1)];
      std::complex<double> r(-b / a, 0);
      ndarray<std::complex<double>> out(std::vector<int>{1});
      out.data()[0] = r;
      return out;
    }
    if (n == 3)
    {
      double a = p.data()[p._flat_logical(start)];
      double b = p.data()[p._flat_logical(start + 1)];
      double c = p.data()[p._flat_logical(start + 2)];
      std::complex<double> disc = std::sqrt(std::complex<double>(b * b - 4 * a * c, 0));
      ndarray<std::complex<double>> out(std::vector<int>{2});
      out.data()[0] = (-b + disc) / (2 * a);
      out.data()[1] = (-b - disc) / (2 * a);
      return out;
    }
    throw std::invalid_argument(
        "roots: only degree <=2 supported in this minimal implementation");
  }

  // Normal comment: polyfit – least squares via Vandermonde + lstsq
  NP_API inline auto polyfit(const ndarray<double>& x, const ndarray<double>& y, int deg)
      -> ndarray<double>
  {
    if (x.size() != y.size())
      throw std::invalid_argument("polyfit: x and y size mismatch");
    if (deg < 0)
      throw std::invalid_argument("polyfit: deg must be >=0");
    if (x.size() == 0)
      throw std::invalid_argument("polyfit: empty x");
    // Build Vandermonde with increasing=false (numpy default)
    int n = static_cast<int>(x.size());
    ndarray<double> V(std::vector<int>{n, deg + 1});
    for (int i = 0; i < n; ++i)
    {
      double xi = x.data()[x._flat_logical(i)];
      for (int j = 0; j <= deg; ++j)
      {
        int power = deg - j;
        double v = 1.0;
        for (int p = 0; p < power; ++p)
          v *= xi;
        V.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = v;
      }
    }
    auto res = linalg::lstsq(V, y);
    return res.x;
  }

} // namespace np

#endif // NP_POLYNOMIAL_HPP
